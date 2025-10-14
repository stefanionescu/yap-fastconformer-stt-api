from __future__ import annotations
import asyncio
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---- helpers for nested cache tensors ----
def _to_index_tensor(idx_list, device):
    return torch.tensor(idx_list, dtype=torch.long, device=device)

def _rec_zero_slot(x, slot):
    if x is None:
        return
    if torch.is_tensor(x):
        x[slot].zero_()
        return
    if isinstance(x, (list, tuple)):
        for e in x:
            _rec_zero_slot(e, slot)

class StreamState:
    __slots__ = ("sid", "sr", "buf", "closed")
    def __init__(self, sid: str, sr: int):
        self.sid = sid
        self.sr = sr
        self.buf: List[np.ndarray] = []
        self.closed: bool = False

    def push(self, pcm16_bytes: bytes):
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buf.append(arr)

    def pop_chunk(self, samples: int) -> Tuple[np.ndarray, int]:
        if not self.buf:
            return np.zeros((0,), dtype=np.float32), 0
        cat = np.concatenate(self.buf, axis=0)
        if cat.shape[0] < samples:
            return np.zeros((0,), dtype=np.float32), 0
        chunk, rest = cat[:samples], cat[samples:]
        self.buf = [rest] if rest.size else []
        return chunk, int(chunk.shape[0])


class GlobalBatcher:
    """
    High-throughput cache-aware streaming batcher for NeMo FastConformer-Hybrid.

    - Fixed slot assignment per stream → stable cache tensors
    - 20 ms ticks (configurable)
    - Gathers/scatters cache only for active slots
    - Emits interims via an asyncio.Queue as (sid, text, ts_ms)
    """

    def __init__(
        self,
        model,
        step_ms: int = 20,
        sample_rate: int = 16000,
        max_slots: int = 64,
        device: torch.device = torch.device("cuda:0"),
        verbose: bool = False,
    ):
        self.model = model
        self.step_ms = step_ms
        self.sample_rate = sample_rate
        self.samples_per_step = int(sample_rate * step_ms / 1000)
        self.max_slots = max_slots
        self.device = device
        self.verbose = verbose

        self._streams: Dict[str, StreamState] = {}
        self._sid2slot: Dict[str, int] = {}
        self._free_slots: List[int] = list(range(max_slots))
        decoding_module = getattr(self.model, "decoding", None)
        self._tokenizer = getattr(self.model, "tokenizer", None)
        self._blank_id = getattr(decoding_module, "blank_id", None)
        self._debug_limit = 256
        self._debug_count = 0

        (self._cache_ch, self._cache_t, self._cache_ch_len) = \
            self.model.encoder.get_initial_cache_state(batch_size=self.max_slots)
        self._log_debug(
            "initial cache shapes ch=%s t=%s ch_len=%s" % (
                self._describe_cache(self._cache_ch),
                self._describe_cache(self._cache_t),
                self._describe_cache(self._cache_ch_len),
            )
        )

        # Track per-slot streaming steps to control cohorting
        self._slot_step: List[int] = [0 for _ in range(self.max_slots)]

        # Maintain RNNT decoder streaming state across ticks, per slot
        # These are fed back into conformer_stream_step via previous_hypotheses
        self._prev_hypotheses: List[Optional[object]] = [None for _ in range(self.max_slots)]
        # Maintain RNNT predictor (prediction network) state across ticks, per slot
        self._prev_pred_out: List[Optional[object]] = [None for _ in range(self.max_slots)]

        self.results: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue(maxsize=8192)

        self._lock = asyncio.Lock()
        self._running = False
        self._tick_task: Optional[asyncio.Task] = None
        self._ema_step_ms = None

        # Track last emitted transcript per slot to avoid redundant queue traffic
        self._last_text: List[str] = ["" for _ in range(self.max_slots)]
        # Maintain a running transcript per slot by accumulating per-tick deltas
        self._running_text: List[str] = ["" for _ in range(self.max_slots)]

        # --- Pre-encode feature cache (features) per slot ---
        try:
            scfg = self.model.encoder.streaming_cfg
            pre_cache_pair = getattr(scfg, "pre_encode_cache", None)
            if pre_cache_pair is None:
                pre_cache_pair = getattr(scfg, "pre_encode_cache_size", [0, 0])
            if isinstance(pre_cache_pair, (list, tuple)):
                pre_cache_frames = int(pre_cache_pair[1]) if len(pre_cache_pair) > 1 else int(pre_cache_pair[0])
            else:
                pre_cache_frames = int(pre_cache_pair)
        except Exception:
            pre_cache_frames = 0
        try:
            n_mels = int(getattr(self.model.cfg.preprocessor, "features", 80))
        except Exception:
            n_mels = 80
        self._pre_cache_frames = pre_cache_frames
        self._n_mels = n_mels
        self._pre_cache = torch.zeros(
            (self.max_slots, n_mels, pre_cache_frames), dtype=torch.float32, device=self.device
        )

        # One-time init log with step sizing
        self._log_debug(
            f"init: step_ms={self.step_ms} samples_per_step={self.samples_per_step}",
        )

    async def add_stream(self, sid: str, sr: int = 16000):
        async with self._lock:
            if sid in self._sid2slot:
                return
            if not self._free_slots:
                raise RuntimeError("No free batcher slots; scale out or increase max_slots")
            slot = self._free_slots.pop(0)
            self._sid2slot[sid] = slot
            self._streams[sid] = StreamState(sid, sr)
            # Reset streaming decoder state for this slot
            self._prev_hypotheses[slot] = None
            self._prev_pred_out[slot] = None
            self._last_text[slot] = ""
            self._running_text[slot] = ""
            # Reset caches and slot step
            _rec_zero_slot(self._cache_ch, slot)
            _rec_zero_slot(self._cache_t, slot)
            _rec_zero_slot(self._cache_ch_len, slot)
            # Reset pre-encode feature cache for this slot
            try:
                self._pre_cache[slot].zero_()
            except Exception:
                pass
            self._slot_step[slot] = 0

    async def remove_stream(self, sid: str):
        async with self._lock:
            slot = self._sid2slot.pop(sid, None)
            self._streams.pop(sid, None)
            if slot is not None:
                # Clear any lingering streaming decoder state
                self._prev_hypotheses[slot] = None
                self._prev_pred_out[slot] = None
                # Reset caches and slot step for reuse safety
                self._slot_step[slot] = 0
                _rec_zero_slot(self._cache_ch, slot)
                _rec_zero_slot(self._cache_t, slot)
                _rec_zero_slot(self._cache_ch_len, slot)
                try:
                    self._pre_cache[slot].zero_()
                except Exception:
                    pass
                self._free_slots.append(slot)
                self._free_slots.sort()
            if slot is not None:
                self._last_text[slot] = ""
                self._running_text[slot] = ""

    async def push_audio(self, sid: str, pcm16: bytes):
        st = self._streams.get(sid)
        if st:
            st.push(pcm16)

    async def start(self):
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self):
        self._running = False
        if self._tick_task:
            await self._tick_task

    async def _tick_loop(self):
        period = self.step_ms / 1000.0
        next_t = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            start = time.perf_counter()
            try:
                await self._step_once()
            except Exception as e:
                if self.verbose:
                    print(f"[batcher] step error: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
            end = time.perf_counter()
            dt_ms = (end - start) * 1000.0
            self._ema_step_ms = dt_ms if self._ema_step_ms is None else (0.9 * self._ema_step_ms + 0.1 * dt_ms)
            next_t += period

    async def _step_once(self):
        async with self._lock:
            if not self._streams:
                return

            sids = list(self._streams.keys())
            slots = [self._sid2slot[sid] for sid in sids]

            # Collect audio for all streams (pad to step size; lens 0 if no audio)
            chunks: List[np.ndarray] = []
            lens: List[int] = []
            for sid in sids:
                st = self._streams[sid]
                chunk, n = st.pop_chunk(self.samples_per_step)
                if n == 0:
                    chunks.append(np.zeros((self.samples_per_step,), dtype=np.float32))
                    lens.append(0)
                else:
                    if chunk.shape[0] < self.samples_per_step:
                        pad = np.zeros((self.samples_per_step - chunk.shape[0],), dtype=np.float32)
                        chunk = np.concatenate([chunk, pad], axis=0)
                    else:
                        chunk = chunk[: self.samples_per_step]
                    chunks.append(chunk)
                    lens.append(self.samples_per_step)

            B = len(sids)
            audio_full = torch.from_numpy(np.stack(chunks, 0)).to(self.device, dtype=torch.float32)
            lengths_full = torch.tensor(lens, dtype=torch.int64, device=self.device)

            # Split into cohorts by per-slot step (0 → newcomers)
            new_idxs: List[int] = []
            act_idxs: List[int] = []
            for i, slot in enumerate(slots):
                if self._slot_step[slot] == 0:
                    new_idxs.append(i)
                else:
                    act_idxs.append(i)

            scfg = self.model.encoder.streaming_cfg
            drop_act = int(getattr(scfg, "drop_extra_pre_encoded", 0))

            expected_feat_dim = self._pre_cache.size(1)

            def run_cohort(idxs: List[int], drop: int):
                if not idxs:
                    return
                slot_rows: List[int] = [slots[i] for i in idxs]
                idx_t = _to_index_tensor(slot_rows, self.device)
                sel_t = _to_index_tensor(idxs, self.device)

                audio = audio_full.index_select(0, sel_t)
                lengths = lengths_full.index_select(0, sel_t)

                # Waveform -> features
                with torch.no_grad():
                    feats, feat_len = self.model.preprocessor(input_signal=audio, length=lengths)

                # Normalize to (B, n_mels, T)
                if feats.ndim == 3:
                    if feats.size(1) != expected_feat_dim and feats.size(2) == expected_feat_dim:
                        feats = feats.transpose(1, 2).contiguous()
                    elif feats.size(1) != expected_feat_dim and feats.size(-1) != expected_feat_dim:
                        self._log_debug(
                            f"unexpected feature dims {tuple(feats.shape)}, expected {expected_feat_dim}",
                            force=True,
                        )

                if feat_len.ndim > 1:
                    feat_len = feat_len.view(-1)

                # Pre-encode cache concat on time dim
                pre_cache_sel = self._pre_cache.index_select(0, idx_t)
                shape_msg = (
                    "stream step shapes audio=%s feats=%s pre_cache=%s feat_len=%s" % (
                        tuple(audio.shape),
                        tuple(feats.shape),
                        tuple(pre_cache_sel.shape),
                        (feat_len.detach().cpu().tolist() if hasattr(feat_len, "detach") else []),
                    )
                )
                self._log_debug(shape_msg)
                if feats.shape[1] != pre_cache_sel.shape[1]:
                    raise RuntimeError(
                        f"Feature dim mismatch PRE={tuple(pre_cache_sel.shape)} FEAT={tuple(feats.shape)}"
                    )
                try:
                    feats_cat = torch.cat([pre_cache_sel, feats], dim=-1)
                except Exception as exc:
                    self._log_debug(f"torch.cat failure: {exc} | {shape_msg}", force=True)
                    raise
                proc_len_frames = feat_len + pre_cache_sel.size(-1)

                # Gather caches with dimension-aware helpers for this cohort
                cache_ch = self._select_cache_rows(self._cache_ch, slot_rows)
                cache_t = self._select_cache_rows(self._cache_t, slot_rows)
                cache_ch_len = self._select_cache_rows(self._cache_ch_len, slot_rows)
                self._log_debug(
                    "cache sel shapes ch=%s t=%s ch_len=%s" % (
                        self._describe_cache(cache_ch),
                        self._describe_cache(cache_t),
                        self._describe_cache(cache_ch_len),
                    )
                )

                prev_h = [self._prev_hypotheses[slots[i]] for i in idxs]
                prev_p = [self._prev_pred_out[slots[i]] for i in idxs]

                with torch.inference_mode():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_ch_new,
                        cache_t_new,
                        cache_ch_len_new,
                        new_prev_hypotheses,
                    ) = self.model.conformer_stream_step(
                        processed_signal=feats_cat,
                        processed_signal_length=proc_len_frames,
                        cache_last_channel=cache_ch,
                        cache_last_time=cache_t,
                        cache_last_channel_len=cache_ch_len,
                        keep_all_outputs=False,
                        previous_hypotheses=prev_h,
                        previous_pred_out=prev_p,
                        drop_extra_pre_encoded=int(drop),
                        return_transcription=True,
                    )

                self._log_debug(
                    "stream step outputs="
                    f"{self._short_repr(transcribed_texts)}"
                )

                # Scatter caches back to full buffers (dimension-aware)
                self._scatter_cache_rows(self._cache_ch, cache_ch_new, slot_rows)
                self._scatter_cache_rows(self._cache_t, cache_t_new, slot_rows)
                self._scatter_cache_rows(self._cache_ch_len, cache_ch_len_new, slot_rows)

                # Update per-slot pre-encode cache: keep last K feature frames
                K = int(self._pre_cache_frames)
                if K > 0:
                    total_frames = feats_cat.size(-1)
                    if total_frames >= K:
                        tail = feats_cat[:, :, -K:]
                    else:
                        pad = torch.zeros(
                            (feats_cat.size(0), feats_cat.size(1), K - total_frames),
                            dtype=feats_cat.dtype,
                            device=feats_cat.device,
                        )
                        tail = torch.cat([pad, feats_cat], dim=-1)
                    self._pre_cache.index_copy_(0, idx_t, tail)

                # Persist RNNT predictor state
                try:
                    if isinstance(pred_out_stream, (list, tuple)):
                        for j, i in enumerate(idxs):
                            self._prev_pred_out[slots[i]] = pred_out_stream[j] if j < len(pred_out_stream) else None
                    else:
                        for i in idxs:
                            self._prev_pred_out[slots[i]] = pred_out_stream
                except Exception:
                    for i in idxs:
                        self._prev_pred_out[slots[i]] = None

                # Persist RNNT previous_hypotheses for label-loop greedy
                try:
                    if isinstance(new_prev_hypotheses, (list, tuple)):
                        for j, i in enumerate(idxs):
                            self._prev_hypotheses[slots[i]] = new_prev_hypotheses[j] if j < len(new_prev_hypotheses) else None
                    else:
                        for i in idxs:
                            self._prev_hypotheses[slots[i]] = None
                except Exception:
                    for i in idxs:
                        self._prev_hypotheses[slots[i]] = None

                # Bump slot steps
                for i in idxs:
                    self._slot_step[slots[i]] += 1

                # Fanout accumulated transcript text
                now_ms = int(time.time() * 1000)
                texts = self._decode_batch_texts(transcribed_texts, len(idxs))
                for k, i in enumerate(idxs):
                    sid = sids[i]
                    slot = slots[i]
                    txt = texts[k] if k < len(texts) else ""
                    if slot is None:
                        continue
                    if not txt:
                        continue

                    current = self._running_text[slot]
                    if len(txt) >= len(current) and txt.startswith(current):
                        merged = txt
                    else:
                        merged = (current + (" " if current and not current.endswith(" ") else "") + txt).strip()

                    if merged != current:
                        self._running_text[slot] = merged
                        self._last_text[slot] = merged
                        try:
                            self.results.put_nowait((sid, merged, now_ms))
                        except asyncio.QueueFull:
                            pass

            # 1) initialize newcomers with drop=0
            run_cohort(new_idxs, drop=0)
            # 2) run actives with configured drop
            run_cohort(act_idxs, drop=drop_act)

            if self.verbose and self._ema_step_ms is not None and (int(time.time() * 10) % 10 == 0):
                print(f"[batcher] active={len(act_idxs)}/{self.max_slots} tick≈{self._ema_step_ms:.2f} ms")


    async def flush_stream(self, sid: str) -> str:
        """Run a final step for SID with keep_all_outputs=True to release tail frames."""
        async with self._lock:
            if sid not in self._sid2slot:
                return ""
            slot = self._sid2slot[sid]

            # Use just the pre-encode cache to flush tail
            pre_cache = self._pre_cache[slot:slot+1]
            proc_len = torch.tensor([pre_cache.shape[-1]], dtype=torch.int64, device=self.device)
            cache_ch = self._select_cache_rows(self._cache_ch, [slot])
            cache_t = self._select_cache_rows(self._cache_t, [slot])
            cache_ch_len = self._select_cache_rows(self._cache_ch_len, [slot])
            self._log_debug(
                "flush shapes pre_cache=%s cache_ch=%s cache_t=%s cache_len=%s" % (
                    tuple(pre_cache.shape),
                    self._describe_cache(cache_ch),
                    self._describe_cache(cache_t),
                    self._describe_cache(cache_ch_len),
                )
            )

            prev_h = [self._prev_hypotheses[slot]]
            prev_p = [self._prev_pred_out[slot]]

            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_ch_new,
                    cache_t_new,
                    cache_ch_len_new,
                    new_prev_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=pre_cache,
                    processed_signal_length=proc_len,
                    cache_last_channel=cache_ch,
                    cache_last_time=cache_t,
                    cache_last_channel_len=cache_ch_len,
                    keep_all_outputs=True,
                    previous_hypotheses=prev_h,
                    previous_pred_out=prev_p,
                    drop_extra_pre_encoded=int(self.model.encoder.streaming_cfg.drop_extra_pre_encoded),
                    return_transcription=True,
                )

            self._log_debug(
                "flush outputs="
                f"{self._short_repr(transcribed_texts)}"
            )

            # Write back updated caches (harmless on final)
            self._scatter_cache_rows(self._cache_ch, cache_ch_new, [slot])
            self._scatter_cache_rows(self._cache_t, cache_t_new, [slot])
            self._scatter_cache_rows(self._cache_ch_len, cache_ch_len_new, [slot])

            # Persist RNNT predictor + hypotheses for completeness
            try:
                if isinstance(pred_out_stream, (list, tuple)):
                    self._prev_pred_out[slot] = pred_out_stream[0] if pred_out_stream else None
                else:
                    self._prev_pred_out[slot] = pred_out_stream
            except Exception:
                self._prev_pred_out[slot] = None
            try:
                if isinstance(new_prev_hypotheses, (list, tuple)):
                    self._prev_hypotheses[slot] = new_prev_hypotheses[0] if new_prev_hypotheses else None
                else:
                    self._prev_hypotheses[slot] = None
            except Exception:
                self._prev_hypotheses[slot] = None

            # Extract text and enqueue as an interim
            h0 = transcribed_texts[0] if isinstance(transcribed_texts, (list, tuple)) else transcribed_texts
            text = self._decode_single_text(h0)
            slot = self._sid2slot.get(sid)
            merged = text or ""
            if slot is not None:
                current = self._running_text[slot]
                if merged:
                    if len(merged) >= len(current) and merged.startswith(current):
                        final_text = merged
                    else:
                        final_text = (current + (" " if current and not current.endswith(" ") else "") + merged).strip()
                else:
                    final_text = current or self._last_text[slot]
                self._running_text[slot] = final_text
                self._last_text[slot] = final_text
            else:
                final_text = merged

            try:
                self.results.put_nowait((sid, final_text, int(time.time() * 1000)))
            except asyncio.QueueFull:
                pass
            return final_text

    def _decode_single_text(self, hyp) -> str:
        """Best-effort conversion of a NeMo streaming hypothesis into plain text."""
        if hyp is None:
            return ""

        # Direct access for dict-style payloads
        if isinstance(hyp, dict):
            for key in ("text", "transcript", "transcription"):
                if hyp.get(key):
                    return str(hyp[key])
            hyp = hyp.get("hypothesis") or hyp.get("hypotheses") or hyp
            if isinstance(hyp, (list, tuple)):
                if not hyp:
                    return ""
                hyp = hyp[0]

        # Unwrap list/tuple containers (e.g. beam hypotheses)
        if isinstance(hyp, (list, tuple)):
            for item in hyp:
                txt = self._decode_single_text(item)
                if txt:
                    return txt
            return ""

        # If the hypothesis already exposes text, trust it.
        txt = getattr(hyp, "text", None)
        if txt:
            return str(txt)

        # Attempt to leverage model decode helpers if available
        for decode_owner in (self.model, getattr(self.model, "decoding", None)):
            decode_fn = getattr(decode_owner, "decode_hypothesis", None)
            if callable(decode_fn):
                try:
                    decoded = decode_fn([hyp])
                    if decoded:
                        candidate = decoded[0]
                        if candidate is hyp:
                            continue
                        if isinstance(candidate, (list, tuple)):
                            # Some helpers return nested structures
                            candidate = next((c for c in candidate if c), "")
                        if isinstance(candidate, (dict, list, tuple)):
                            txt_candidate = self._decode_single_text(candidate)
                            if txt_candidate:
                                return txt_candidate
                        elif candidate:
                            return str(candidate)
                except Exception:
                    pass

        # Convert token IDs → text via tokenizer as a fallback
        tokens = getattr(hyp, "tokens", None)
        if tokens is None:
            tokens = getattr(hyp, "y_sequence", None)
        if tokens is not None:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.detach().cpu().tolist()
            elif hasattr(tokens, "cpu"):
                tokens = tokens.cpu().tolist()
            elif isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if isinstance(tokens, list):
                if tokens and isinstance(tokens[0], (list, tuple)):
                    tokens = list(tokens[0])
                filtered: List[int] = []
                for t in tokens:
                    if isinstance(t, (np.integer, int)):
                        tid = int(t)
                        if self._blank_id is not None and tid == self._blank_id:
                            continue
                        filtered.append(tid)
                tokens = filtered
                try:
                    if self._tokenizer and hasattr(self._tokenizer, "ids_to_text"):
                        return str(self._tokenizer.ids_to_text(tokens))
                    if self._tokenizer and hasattr(self._tokenizer, "ids2text"):
                        return str(self._tokenizer.ids2text(tokens))
                except Exception:
                    pass

        self._log_debug(
            "decode miss for type="
            f"{type(hyp).__name__} tokens={self._short_repr(tokens)}"
        )

        return ""

    def _decode_batch_texts(self, outputs, expected: int) -> List[str]:
        if expected <= 0:
            return []
        if outputs is None:
            return ["" for _ in range(expected)]

        items: List[object]
        if isinstance(outputs, dict):
            for key in ("text", "texts", "transcripts", "transcriptions", "hypotheses"):
                if key in outputs:
                    outputs = outputs[key]
                    break

        if isinstance(outputs, torch.Tensor):
            items = outputs.detach().cpu().tolist()
        elif isinstance(outputs, (list, tuple)):
            items = list(outputs)
        else:
            items = [outputs]

        # Flatten single-element container patterns common in NeMo beams
        if len(items) == 1 and isinstance(items[0], (list, tuple)) and len(items[0]) == expected:
            items = list(items[0])

        texts = [self._decode_single_text(items[i]) if i < len(items) else "" for i in range(expected)]
        if self.verbose and all(t == "" for t in texts):
            self._log_debug(
                "batch decode empty → raw="
                f"{self._short_repr(outputs)}"
            )
        return texts

    def _log_debug(self, message: str, *, force: bool = False) -> None:
        if not (self.verbose or force):
            return
        if self._debug_count < self._debug_limit:
            print(f"[batcher] {message}", flush=True)
        elif self._debug_count == self._debug_limit:
            print("[batcher] debug limit reached; suppressing further logs", flush=True)
        self._debug_count += 1

    @staticmethod
    def _short_repr(obj, limit: int = 256) -> str:
        try:
            text = repr(obj)
        except Exception:
            text = f"<repr_failed {type(obj).__name__}>"
        if text is None:
            return "None"
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _describe_cache(self, cache) -> str:
        if cache is None:
            return "None"
        if isinstance(cache, (list, tuple)):
            inner = ",".join(self._describe_cache(c) for c in cache)
            return f"[{inner}]"
        if torch.is_tensor(cache):
            return str(tuple(cache.shape))
        return type(cache).__name__

    def _cache_batch_dim(self, tensor: torch.Tensor) -> int:
        for dim in range(tensor.dim()):
            if tensor.size(dim) == self.max_slots:
                return dim
        # Default to first dimension if no explicit match
        return 0

    def _select_cache_rows(self, cache, idx: List[int]):
        if cache is None:
            return None
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._select_cache_rows(c, idx) for c in cache)
        if not torch.is_tensor(cache):
            return cache
        dim = self._cache_batch_dim(cache)
        index_tensor = torch.tensor(idx, dtype=torch.long, device=cache.device)
        return cache.index_select(dim, index_tensor)

    def _scatter_cache_rows(self, dst, src, idx: List[int]) -> None:
        if dst is None or src is None:
            return
        if isinstance(dst, (list, tuple)) and isinstance(src, (list, tuple)):
            for d_sub, s_sub in zip(dst, src):
                self._scatter_cache_rows(d_sub, s_sub, idx)
            return
        if not torch.is_tensor(dst) or not torch.is_tensor(src):
            return
        dim = self._cache_batch_dim(dst)
        index_tensor = torch.tensor(idx, dtype=torch.long, device=dst.device)
        dst.index_copy_(dim, index_tensor, src)

from __future__ import annotations
import asyncio
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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
        max_slots: int = 128,
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
        self._debug_limit = 64
        self._debug_count = 0

        (self._cache_ch, self._cache_t, self._cache_ch_len) = \
            self.model.encoder.get_initial_cache_state(batch_size=self.max_slots)

        # Track global streaming steps to control pre-encoded frame dropping
        self._global_step: int = 0

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

    async def remove_stream(self, sid: str):
        async with self._lock:
            slot = self._sid2slot.pop(sid, None)
            self._streams.pop(sid, None)
            if slot is not None:
                # Clear any lingering streaming decoder state
                self._prev_hypotheses[slot] = None
                self._prev_pred_out[slot] = None
                self._free_slots.append(slot)
                self._free_slots.sort()
            if slot is not None:
                self._last_text[slot] = ""

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

            chunks = []
            lens = []
            for sid in sids:
                st = self._streams[sid]
                chunk, n = st.pop_chunk(self.samples_per_step)
                if n == 0:
                    chunks.append(np.zeros((self.samples_per_step,), dtype=np.float32))
                    lens.append(0)
                else:
                    if chunk.shape[0] != self.samples_per_step:
                        if chunk.shape[0] < self.samples_per_step:
                            pad = np.zeros((self.samples_per_step - chunk.shape[0],), dtype=np.float32)
                            chunk = np.concatenate([chunk, pad], axis=0)
                        else:
                            chunk = chunk[: self.samples_per_step]
                    chunks.append(chunk)
                    lens.append(self.samples_per_step)

            audio = torch.from_numpy(np.stack(chunks, axis=0)).to(self.device, dtype=torch.float32)
            lengths_samples = torch.tensor(lens, dtype=torch.int64, device=self.device)

            # Select active subset (those that delivered nonzero audio this tick)
            active = [i for i, L in enumerate(lens) if L > 0]
            if not active:
                return

            act_slots = [slots[i] for i in active]
            active_idx = torch.tensor(active, dtype=torch.long, device=self.device)
            audio_act = audio.index_select(0, active_idx)
            len_act_samples = lengths_samples.index_select(0, active_idx)

            # 1) Waveform -> features (B_act, n_mels, T_frames)
            with torch.no_grad():
                feats_act, feat_len_act = self.model.preprocessor(
                    input_signal=audio_act, length=len_act_samples
                )

            # Normalise feature tensor layout to (B, n_mels, T)
            expected_feat_dim = self._pre_cache.size(1)
            if feats_act.ndim == 3:
                if feats_act.size(1) != expected_feat_dim and feats_act.size(2) == expected_feat_dim:
                    feats_act = feats_act.transpose(1, 2).contiguous()
                elif feats_act.size(1) != expected_feat_dim and feats_act.size(-1) != expected_feat_dim:
                    self._log_debug(
                        f"unexpected feature dims {tuple(feats_act.shape)}, expected {expected_feat_dim}",
                        force=True,
                    )

            if feat_len_act.ndim > 1:
                feat_len_act = feat_len_act.view(-1)
            try:
                feat_len_list = feat_len_act.detach().cpu().tolist()
            except Exception:
                feat_len_list = []

            # 2) Concat pre-encode cache on time dim
            act_slots_idx = torch.tensor(act_slots, dtype=torch.long, device=self.device)
            pre_cache_sel = self._pre_cache.index_select(0, act_slots_idx)
            shape_msg = (
                "stream step shapes audio=%s feats=%s pre_cache=%s feat_len=%s" % (
                    tuple(audio_act.shape),
                    tuple(feats_act.shape),
                    tuple(pre_cache_sel.shape),
                    feat_len_list,
                )
            )
            self._log_debug(shape_msg)
            if feats_act.shape[1] != pre_cache_sel.shape[1]:
                raise RuntimeError(
                    f"Feature dim mismatch PRE={tuple(pre_cache_sel.shape)} FEAT={tuple(feats_act.shape)}"
                )
            try:
                feats_cat = torch.cat([pre_cache_sel, feats_act], dim=-1)
            except Exception as exc:
                self._log_debug(f"torch.cat failure: {exc} | {shape_msg}", force=True)
                raise
            proc_len_frames = feat_len_act + pre_cache_sel.size(-1)

            def _gather_rows(t: Optional[torch.Tensor], idx: List[int]) -> Optional[torch.Tensor]:
                if t is None:
                    return None
                return t.index_select(0, torch.tensor(idx, dtype=torch.long, device=t.device))

            cache_ch = _gather_rows(self._cache_ch, act_slots)
            cache_t = _gather_rows(self._cache_t, act_slots)
            cache_ch_len = _gather_rows(self._cache_ch_len, act_slots)

            prev_hyp_batch: List[Optional[object]] = [self._prev_hypotheses[s] for s in act_slots]
            prev_pred_out_batch: List[Optional[object]] = [self._prev_pred_out[s] for s in act_slots]

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
                    previous_hypotheses=prev_hyp_batch,
                    previous_pred_out=prev_pred_out_batch,
                    drop_extra_pre_encoded=(
                        0 if self._global_step == 0
                        else int(self.model.encoder.streaming_cfg.drop_extra_pre_encoded)
                    ),
                    return_transcription=True,
                )
            self._global_step += 1

            self._log_debug(
                "stream step outputs="
                f"{self._short_repr(transcribed_texts)}"
            )

            def _scatter_rows(dst: Optional[torch.Tensor], src: Optional[torch.Tensor], idx: List[int]):
                if dst is None or src is None:
                    return
                dst.index_copy_(0, torch.tensor(idx, dtype=torch.long, device=dst.device), src)

            _scatter_rows(self._cache_ch, cache_ch_new, act_slots)
            _scatter_rows(self._cache_t, cache_t_new, act_slots)
            _scatter_rows(self._cache_ch_len, cache_ch_len_new, act_slots)

            # 4) Update per-slot pre-encode cache: keep last K feature frames
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
                self._pre_cache.index_copy_(0, act_slots_idx, tail)

            # Persist updated RNNT hypotheses back to their slots (only active)
            try:
                if isinstance(new_prev_hypotheses, (list, tuple)):
                    for i, slot in enumerate(act_slots):
                        if i < len(new_prev_hypotheses):
                            self._prev_hypotheses[slot] = new_prev_hypotheses[i]
                else:
                    for slot in act_slots:
                        self._prev_hypotheses[slot] = None
            except Exception:
                for slot in act_slots:
                    self._prev_hypotheses[slot] = None

            # Persist updated RNNT predictor state back to their slots (only active)
            try:
                if isinstance(pred_out_stream, (list, tuple)):
                    for i, slot in enumerate(act_slots):
                        if i < len(pred_out_stream):
                            self._prev_pred_out[slot] = pred_out_stream[i]
                else:
                    for slot in act_slots:
                        self._prev_pred_out[slot] = pred_out_stream
            except Exception:
                for slot in act_slots:
                    self._prev_pred_out[slot] = None

            now_ms = int(time.time() * 1000)
            texts = self._decode_batch_texts(transcribed_texts, len(active))

            for sid, txt in zip([sids[i] for i in active], texts):
                slot = self._sid2slot.get(sid)
                if slot is None:
                    continue
                if txt and txt != self._last_text[slot]:
                    self._last_text[slot] = txt
                elif not txt:
                    continue
                try:
                    self.results.put_nowait((sid, txt, now_ms))
                except asyncio.QueueFull:
                    pass

            if self.verbose and self._ema_step_ms is not None and (int(time.time() * 10) % 10 == 0):
                print(f"[batcher] active={len(active)}/{self.max_slots} tick≈{self._ema_step_ms:.2f} ms")


    async def flush_stream(self, sid: str) -> str:
        """Run a final step for SID with keep_all_outputs=True to release tail frames."""
        async with self._lock:
            if sid not in self._sid2slot:
                return ""
            slot = self._sid2slot[sid]

            # Use just the pre-encode cache to flush tail
            pre_cache = self._pre_cache[slot:slot+1]
            proc_len = torch.tensor([pre_cache.shape[-1]], dtype=torch.int64, device=self.device)
            self._log_debug(
                "flush shapes pre_cache=%s cache_ch=%s cache_t=%s cache_len=%s" % (
                    tuple(pre_cache.shape),
                    None if self._cache_ch is None else tuple(self._cache_ch[slot:slot+1].shape),
                    None if self._cache_t is None else tuple(self._cache_t[slot:slot+1].shape),
                    None if self._cache_ch_len is None else tuple(self._cache_ch_len[slot:slot+1].shape),
                )
            )

            def _slice_row(t: Optional[torch.Tensor], i: int) -> Optional[torch.Tensor]:
                if t is None:
                    return None
                return t[i:i+1]

            cache_ch = _slice_row(self._cache_ch, slot)
            cache_t = _slice_row(self._cache_t, slot)
            cache_ch_len = _slice_row(self._cache_ch_len, slot)

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
            if cache_ch_new is not None and self._cache_ch is not None:
                self._cache_ch[slot:slot+1].copy_(cache_ch_new)
            if cache_t_new is not None and self._cache_t is not None:
                self._cache_t[slot:slot+1].copy_(cache_t_new)
            if cache_ch_len_new is not None and self._cache_ch_len is not None:
                self._cache_ch_len[slot:slot+1].copy_(cache_ch_len_new)

            # Persist RNNT predictor + hypotheses for completeness
            try:
                if isinstance(pred_out_stream, (list, tuple)):
                    self._prev_pred_out[slot] = pred_out_stream[0] if pred_out_stream else None
                else:
                    self._prev_pred_out[slot] = pred_out_stream
            except Exception:
                self._prev_pred_out[slot] = None

            try:
                if isinstance(new_prev_hypotheses, (list, tuple)) and new_prev_hypotheses:
                    self._prev_hypotheses[slot] = new_prev_hypotheses[0]
            except Exception:
                pass

            # Extract text and enqueue as an interim
            h0 = transcribed_texts[0] if isinstance(transcribed_texts, (list, tuple)) else transcribed_texts
            text = ""
            text = self._decode_single_text(h0)
            slot = self._sid2slot.get(sid)
            if slot is not None:
                if text:
                    self._last_text[slot] = text
                else:
                    text = self._last_text[slot]

            try:
                self.results.put_nowait((sid, text, int(time.time() * 1000)))
            except asyncio.QueueFull:
                pass
            return text

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

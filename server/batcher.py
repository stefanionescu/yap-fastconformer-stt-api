from __future__ import annotations
import asyncio
import time
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

        (self._cache_ch, self._cache_t, self._cache_ch_len) = \
            self.model.encoder.get_initial_cache_state(batch_size=self.max_slots)
        self._prev_hypotheses = None
        self._prev_pred_out = None

        self.results: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue(maxsize=8192)

        self._lock = asyncio.Lock()
        self._running = False
        self._tick_task: Optional[asyncio.Task] = None
        self._ema_step_ms = None

    async def add_stream(self, sid: str, sr: int = 16000):
        async with self._lock:
            if sid in self._sid2slot:
                return
            if not self._free_slots:
                raise RuntimeError("No free batcher slots; scale out or increase max_slots")
            slot = self._free_slots.pop(0)
            self._sid2slot[sid] = slot
            self._streams[sid] = StreamState(sid, sr)

    async def remove_stream(self, sid: str):
        async with self._lock:
            slot = self._sid2slot.pop(sid, None)
            self._streams.pop(sid, None)
            if slot is not None:
                self._free_slots.append(slot)
                self._free_slots.sort()

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
                    print(f"[batcher] step error: {e}")
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

            B = len(sids)
            audio = torch.from_numpy(np.stack(chunks, axis=0)).to(self.device, dtype=torch.float32)
            lengths = torch.tensor(lens, dtype=torch.int64, device=self.device)

            def _gather_rows(t: Optional[torch.Tensor], idx: List[int]) -> Optional[torch.Tensor]:
                if t is None:
                    return None
                return t.index_select(0, torch.tensor(idx, dtype=torch.long, device=t.device))

            cache_ch = _gather_rows(self._cache_ch, slots)
            cache_t = _gather_rows(self._cache_t, slots)
            cache_ch_len = _gather_rows(self._cache_ch_len, slots)

            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_ch_new,
                    cache_t_new,
                    cache_ch_len_new,
                    prev_hypotheses_new,
                ) = self.model.conformer_stream_step(
                    processed_signal=audio,
                    processed_signal_length=lengths,
                    cache_last_channel=cache_ch,
                    cache_last_time=cache_t,
                    cache_last_channel_len=cache_ch_len,
                    keep_all_outputs=True,
                    previous_hypotheses=self._prev_hypotheses,
                    previous_pred_out=self._prev_pred_out,
                    drop_extra_pre_encoded=self.model.encoder.streaming_cfg.drop_extra_pre_encoded
                        if hasattr(self.model.encoder, "streaming_cfg") else 0,
                    return_transcription=True,
                )

            self._prev_hypotheses = prev_hypotheses_new
            self._prev_pred_out = pred_out_stream

            def _scatter_rows(dst: Optional[torch.Tensor], src: Optional[torch.Tensor], idx: List[int]):
                if dst is None or src is None:
                    return
                dst.index_copy_(0, torch.tensor(idx, dtype=torch.long, device=dst.device), src)

            _scatter_rows(self._cache_ch, cache_ch_new, slots)
            _scatter_rows(self._cache_t, cache_t_new, slots)
            _scatter_rows(self._cache_ch_len, cache_ch_len_new, slots)

            now_ms = int(time.time() * 1000)
            texts = [h.text if hasattr(h, "text") else (str(h) if h is not None else "") for h in transcribed_texts]
            for sid, txt in zip(sids, texts):
                try:
                    self.results.put_nowait((sid, txt, now_ms))
                except asyncio.QueueFull:
                    pass

            if self.verbose and self._ema_step_ms is not None and (int(time.time() * 10) % 10 == 0):
                print(f"[batcher] active={B}/{self.max_slots} tick≈{self._ema_step_ms:.2f} ms")



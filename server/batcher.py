from __future__ import annotations
import asyncio
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


class StreamState:
    __slots__ = ("sid", "sr", "buf", "closed")

    def __init__(self, sid: str, sr: int):
        self.sid = sid
        self.sr = sr
        self.buf: List[np.ndarray] = []
        self.closed = False

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
        return chunk, len(chunk)


class GlobalBatcher:
    """20ms stepping batcher feeding NeMo cache-aware streaming step."""

    def __init__(
        self,
        model,
        step_ms: int = 20,
        sample_rate: int = 16000,
        max_batch: int = 128,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.model = model
        self.step_ms = step_ms
        self.sample_rate = sample_rate
        self.samples_per_step = int(sample_rate * step_ms / 1000)
        self.device = device
        self.max_batch = max_batch

        self.streams: Dict[str, StreamState] = {}
        self.sid_to_slot: Dict[str, int] = {}
        self.slot_to_sid: List[Optional[str]] = [None] * self.max_batch
        self._lock = asyncio.Lock()
        self._running = False

        self.cache_last_channel = None
        self.cache_last_time = None
        self.cache_last_channel_len = None
        self.previous_hypotheses = None
        self.pred_out_stream = None

        self.streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )

        self.last_out: Optional[Tuple[List[str], List[str]]] = None

    async def add_stream(self, sid: str, sr: int = 16000):
        async with self._lock:
            if sid in self.streams:
                return
            # find free slot
            try:
                slot = self.slot_to_sid.index(None)
            except ValueError:
                # no capacity; drop the request silently
                return
            self.streams[sid] = StreamState(sid, sr)
            self.sid_to_slot[sid] = slot
            self.slot_to_sid[slot] = sid

    async def remove_stream(self, sid: str):
        async with self._lock:
            self.streams.pop(sid, None)
            slot = self.sid_to_slot.pop(sid, None)
            if slot is not None:
                self.slot_to_sid[slot] = None

    async def push_audio(self, sid: str, pcm16: bytes):
        st = self.streams.get(sid)
        if st:
            st.push(pcm16)

    async def start(self):
        if self._running:
            return
        self._running = True
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.model.encoder.get_initial_cache_state(
            batch_size=self.max_batch
        )
        asyncio.create_task(self._tick())

    async def _tick(self):
        period = self.step_ms / 1000.0
        while self._running:
            await asyncio.sleep(period)
            await self._step_once()

    async def _step_once(self):
        async with self._lock:
            active = any(self.slot_to_sid)
            if not active:
                return None

            chunks = []
            lengths = []
            active_sids: List[str] = []
            for slot in range(self.max_batch):
                sid = self.slot_to_sid[slot]
                if sid is None:
                    chunks.append(np.zeros((self.samples_per_step,), dtype=np.float32))
                    lengths.append(0)
                    continue
                st = self.streams.get(sid)
                if st is None:
                    chunks.append(np.zeros((self.samples_per_step,), dtype=np.float32))
                    lengths.append(0)
                    continue
                chunk, n = st.pop_chunk(self.samples_per_step)
                if n == 0:
                    chunks.append(np.zeros((self.samples_per_step,), dtype=np.float32))
                    lengths.append(0)
                else:
                    chunks.append(chunk)
                    lengths.append(n)
                active_sids.append(sid)

            audio = torch.from_numpy(np.stack(chunks, axis=0)).to(self.model.device, dtype=torch.float32)
            lens = torch.tensor(lengths, dtype=torch.int64, device=self.model.device)

            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    self.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=audio,
                    processed_signal_length=lens,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=True,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.pred_out_stream,
                    drop_extra_pre_encoded=self.model.encoder.streaming_cfg.drop_extra_pre_encoded
                    if hasattr(self.model.encoder, "streaming_cfg")
                    else 0,
                    return_transcription=True,
                )
                self.pred_out_stream = pred_out_stream

            texts = [h.text if hasattr(h, "text") else h for h in transcribed_texts]
            # Fan out only for active sids with non-None slots, preserving order by slot
            out_sids: List[str] = []
            out_texts: List[str] = []
            for slot in range(self.max_batch):
                sid = self.slot_to_sid[slot]
                if sid is None:
                    continue
                out_sids.append(sid)
                out_texts.append(texts[slot])

            self.last_out = (out_sids, out_texts)
            return self.last_out



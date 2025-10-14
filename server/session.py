from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from aiortc import RTCDataChannel

from .config import Config

_LOG = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    session: "Session"
    audio: np.ndarray
    is_final: bool


class Session:
    def __init__(
        self,
        sid: str,
        channel: RTCDataChannel,
        batcher: "BatchTranscriber",
        cfg: Config,
        on_finalized: Optional[callable] = None,
    ) -> None:
        self.sid = sid
        self._channel = channel
        self._batcher = batcher
        self._cfg = cfg
        self._on_finalized = on_finalized
        self._max_samples = int(cfg.max_buffer_seconds * cfg.sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = asyncio.Lock()
        self._last_text = ""
        self._dirty = False
        self._queued = False
        self._pending_final = False
        self._closed = False
        self._last_activity = time.monotonic()
        self._linger_task: Optional[asyncio.Task[None]] = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_text(self) -> str:
        return self._last_text

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    async def add_audio(self, data: bytes) -> None:
        if self._closed:
            return
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return
        audio = samples.astype(np.float32) / 32768.0
        async with self._lock:
            self._buffer = np.concatenate((self._buffer, audio))
            if self._max_samples > 0 and self._buffer.size > self._max_samples:
                excess = self._buffer.size - self._max_samples
                self._buffer = self._buffer[excess:]
            self._dirty = True
            self._last_activity = time.monotonic()
            queued = self._queued
        if not queued:
            await self._enqueue()

    async def request_final(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._pending_final = True
            queued = self._queued
        if not queued:
            await self._enqueue()

    async def abort(self, reason: str) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        payload = json.dumps({"op": "error", "sid": self.sid, "reason": reason})
        try:
            self._channel.send(payload)
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            await self._batcher.detach(self)
        finally:
            await self._schedule_close()
            if self._on_finalized is not None:
                try:
                    self._on_finalized()
                except Exception:  # pragma: no cover
                    _LOG.exception("on_finalized callback failed for %s", self.sid)

    async def extract_batch(self) -> Optional[BatchRequest]:
        async with self._lock:
            if self._closed:
                return None
            if not self._dirty and not self._pending_final:
                self._queued = False
                return None
            audio = np.copy(self._buffer)
            is_final = self._pending_final
            self._dirty = False
            self._pending_final = False
            self._queued = False
        return BatchRequest(self, audio, is_final)

    async def handle_transcript(self, text: str, is_final: bool, inference_ms: float) -> None:
        text = text.strip()
        should_emit = False
        op = "interim"
        async with self._lock:
            if self._closed and not is_final:
                return
            if text != self._last_text or is_final:
                self._last_text = text
                should_emit = True
                op = "final" if is_final else "interim"
            if is_final:
                self._closed = True
        if should_emit:
            payload = json.dumps(
                {
                    "op": op,
                    "sid": self.sid,
                    "text": text,
                    "final": bool(is_final),
                    "inference_ms": inference_ms,
                }
            )
            try:
                self._channel.send(payload)
            except Exception:  # pragma: no cover - best effort
                _LOG.warning("Failed to send payload to %s", self.sid, exc_info=True)
        if is_final:
            await self._schedule_close()
            if self._on_finalized is not None:
                try:
                    self._on_finalized()
                except Exception:  # pragma: no cover
                    _LOG.exception("on_finalized callback failed for %s", self.sid)

    async def handle_inference_error(self, message: str) -> None:
        await self.abort(message)

    async def _enqueue(self) -> None:
        async with self._lock:
            if self._queued or (not self._dirty and not self._pending_final) or self._closed:
                return
            self._queued = True
        await self._batcher.enqueue(self)

    async def _schedule_close(self) -> None:
        if self._linger_task is not None:
            return
        linger = max(0.0, self._cfg.linger_after_close_ms / 1000.0)
        if linger == 0:
            await self._channel_close()
            return

        async def _close() -> None:
            await asyncio.sleep(linger)
            self._channel_close()

        self._linger_task = asyncio.create_task(_close())

    def _channel_close(self) -> None:
        try:
            if self._channel.readyState != "closed":  # type: ignore[attr-defined]
                self._channel.close()
        except Exception:  # pragma: no cover - best effort
            pass


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .batching import BatchTranscriber

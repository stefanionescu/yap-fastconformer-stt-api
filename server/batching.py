from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, List, Optional, Set

import numpy as np

from .config import Config
from .moonshine_backend import MoonshineBackend

_LOG = logging.getLogger(__name__)


class BatchTranscriber:
    def __init__(self, backend: MoonshineBackend, cfg: Config) -> None:
        self._backend = backend
        self._cfg = cfg
        self._queue: asyncio.Queue["Session | None"] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._runner())
        self._active_sessions: Set["Session"] = set()

    async def enqueue(self, session: "Session") -> None:
        if session.closed:
            return
        await self._queue.put(session)

    async def detach(self, session: "Session") -> None:
        self._active_sessions.discard(session)

    async def close(self) -> None:
        self._stop_event.set()
        await self._queue.put(None)
        await asyncio.gather(self._task, return_exceptions=True)

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break
            if first is None:
                break
            if first.closed:
                continue
            batch = [first]
            start = time.perf_counter()
            while len(batch) < self._cfg.max_batch_size:
                remaining = (self._cfg.max_batch_wait_ms / 1000.0) - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    await self._queue.put(None)
                    break
                if nxt.closed:
                    continue
                batch.append(nxt)
            await self._process_batch(batch)

    async def _process_batch(self, batch: Iterable["Session"]) -> None:
        sessions = list(dict.fromkeys(batch))
        if not sessions:
            return
        self._active_sessions.update(sessions)
        requests: List[Optional["BatchRequest"]] = []
        for session in sessions:
            try:
                request = await session.extract_batch()
            except Exception:  # pragma: no cover - defensive
                _LOG.exception("Failed to extract audio for %s", session.sid)
                continue
            requests.append(request)
        payloads: List[np.ndarray] = []
        final_flags: List[bool] = []
        targets: List["Session"] = []
        zero_final: List["Session"] = []
        for req in requests:
            if req is None:
                continue
            if req.audio.size == 0:
                if req.is_final:
                    zero_final.append(req.session)
                continue
            payloads.append(req.audio)
            final_flags.append(req.is_final)
            targets.append(req.session)
        results: List[str] = []
        inference_ms = 0.0
        if payloads:
            t0 = time.perf_counter()
            try:
                results = self._backend.transcribe(payloads)
            except Exception as exc:  # pragma: no cover - runtime failure
                _LOG.exception("Moonshine inference failed: %s", exc)
                for session in targets:
                    await session.handle_inference_error("inference_failed")
                self._active_sessions.difference_update(sessions)
                return
            inference_ms = (time.perf_counter() - t0) * 1000.0
            for session, text, is_final in zip(targets, results, final_flags):
                await session.handle_transcript(text, is_final, inference_ms)
        for session in zero_final:
            await session.handle_transcript(session.last_text, True, inference_ms)
        self._active_sessions.difference_update(sessions)


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .session import BatchRequest, Session

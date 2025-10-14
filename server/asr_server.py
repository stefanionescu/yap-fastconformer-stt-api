from __future__ import annotations

import asyncio
import json
import logging
import signal
import uuid
from typing import Dict, Optional

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .batching import BatchTranscriber
from .config import Config, load_config
from .logging_utils import setup_logging
from .moonshine_backend import MoonshineBackend
from .session import Session

_LOG = logging.getLogger(__name__)


class AsrApplication:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backend = MoonshineBackend(cfg)
        self.batcher = BatchTranscriber(self.backend, cfg)
        self._pcs: set[RTCPeerConnection] = set()

    async def handle_offer(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.Response(status=400, text="invalid json")
        offer_sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not offer_sdp or not offer_type:
            return web.Response(status=400, text="missing sdp/type")
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        pc = RTCPeerConnection()
        self._pcs.add(pc)
        _LOG.info("New peer connection (%s)", pc)

        session_holder: Dict[str, Optional[Session]] = {"session": None}

        def finalize_pc() -> None:
            if session_holder["session"] is None:
                return
            asyncio.create_task(self._finalize_peer(pc))

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            _LOG.info("Data channel %s received", channel.label)

            @channel.on("open")
            def on_open() -> None:
                _LOG.info("Channel %s open", channel.label)

            async def process_json(message: str) -> None:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    channel.send(json.dumps({"op": "error", "reason": "bad_json"}))
                    return
                op = str(data.get("op") or "").lower()
                if op == "init":
                    if session_holder["session"] is not None:
                        channel.send(json.dumps({"op": "error", "reason": "already_initialized"}))
                        return
                    sr = int(data.get("sr") or self.cfg.sample_rate)
                    if sr != self.cfg.sample_rate:
                        channel.send(json.dumps({"op": "error", "reason": "unsupported_sample_rate"}))
                        return
                    sid = str(data.get("sid") or uuid.uuid4().hex)
                    session = Session(sid, channel, self.batcher, self.cfg, on_finalized=finalize_pc)
                    session_holder["session"] = session
                    channel.send(
                        json.dumps(
                            {
                                "op": "ready",
                                "sid": sid,
                                "max_batch": self.cfg.max_batch_size,
                                "model": self.cfg.model_name,
                            }
                        )
                    )
                elif op in {"close", "flush"}:
                    session = session_holder.get("session")
                    if session is not None:
                        await session.request_final()
                elif op == "ping":
                    channel.send(json.dumps({"op": "pong"}))
                else:
                    channel.send(json.dumps({"op": "error", "reason": "unknown_op"}))

            async def process_binary(message: bytes) -> None:
                session = session_holder.get("session")
                if session is None:
                    channel.send(json.dumps({"op": "error", "reason": "not_initialized"}))
                    return
                await session.add_audio(message)

            def on_message(message) -> None:
                if isinstance(message, bytes):
                    asyncio.create_task(process_binary(message))
                else:
                    asyncio.create_task(process_json(str(message)))

            @channel.on("message")
            def _on_message(message) -> None:
                on_message(message)

            @channel.on("close")
            def _on_close() -> None:
                _LOG.info("Channel %s closed", channel.label)
                session = session_holder.get("session")
                if session is not None and not session.closed:
                    asyncio.create_task(session.request_final())

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            _LOG.info("Peer %s state=%s", pc, pc.connectionState)
            if pc.connectionState in {"failed", "closed"}:
                await self._finalize_peer(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def _finalize_peer(self, pc: RTCPeerConnection) -> None:
        if pc not in self._pcs:
            return
        self._pcs.discard(pc)
        try:
            await pc.close()
        except Exception:  # pragma: no cover - best effort
            _LOG.exception("Failed to close peer %s", pc)

    async def shutdown(self) -> None:
        for pc in list(self._pcs):
            await self._finalize_peer(pc)
        await self.batcher.close()


async def _run_app(cfg: Config) -> None:
    app_state = AsrApplication(cfg)
    app = web.Application()
    app.router.add_post(cfg.http_offer_path, app_state.handle_offer)

    async def on_cleanup(_: web.Application) -> None:
        await app_state.shutdown()

    app.on_cleanup.append(on_cleanup)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.port)
    await site.start()
    _LOG.info("ASR server listening on %s:%d (path=%s)", cfg.host, cfg.port, cfg.http_offer_path)

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        _LOG.info("Received stop signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:  # pragma: no cover (Windows)
            signal.signal(sig, lambda *_: _handle_signal())

    await stop_event.wait()
    await runner.cleanup()


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.log_level)
    try:
        asyncio.run(_run_app(cfg))
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()

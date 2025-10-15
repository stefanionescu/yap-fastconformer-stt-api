"""Async WebSocket server that streams transcription results from Vosk."""

from __future__ import annotations

import asyncio
import json
import logging
from itertools import count
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol
from vosk import KaldiRecognizer, Model, SetLogLevel, GpuInit

from .settings import Settings

CONTROL_PREFIX = "__CTRL__:"
CTRL_EOS = "EOS"
CTRL_RESET = "RESET"

LOGGER = logging.getLogger("vosk_ws_server")


def _extract_control(payload: object) -> Optional[str]:
    """Return the control verb if the payload carries a control command."""
    if isinstance(payload, bytes):
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError:
            return None
    elif isinstance(payload, str):
        text = payload
    else:
        return None
    if not text.startswith(CONTROL_PREFIX):
        return None
    return text[len(CONTROL_PREFIX) :].strip()


def _recognizer_result(recognizer: KaldiRecognizer) -> dict[str, str]:
    try:
        return json.loads(recognizer.Result())
    except json.JSONDecodeError:
        return {}


def _recognizer_partial(recognizer: KaldiRecognizer) -> dict[str, str]:
    try:
        return json.loads(recognizer.PartialResult())
    except json.JSONDecodeError:
        return {}


def _recognizer_final(recognizer: KaldiRecognizer) -> dict[str, str]:
    try:
        return json.loads(recognizer.FinalResult())
    except json.JSONDecodeError:
        return {}


class VoskServer:
    """Manages shared recognizer state and serves WebSocket clients."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Initialize CUDA backend for Vosk before creating the Model
        try:
            GpuInit()
        except Exception:
            # If GPU init fails, continue; libvosk may still run in CPU mode
            LOGGER.warning("GpuInit() failed; falling back to CPU if available", exc_info=True)
        SetLogLevel(settings.vosk_log_level)
        LOGGER.info("Loading Vosk model from %s", settings.model_dir)
        self.model = Model(str(settings.model_dir))
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self._session_ids = count(start=1)

    async def start(self) -> None:
        LOGGER.info(
            "Starting WebSocket server on %s:%d (max_message_size=%d bytes, concurrency=%d)",
            self.settings.host,
            self.settings.port,
            self.settings.max_message_size,
            self.settings.concurrency,
        )
        async with websockets.serve(
            self._handle_client,
            host=self.settings.host,
            port=self.settings.port,
            max_size=self.settings.max_message_size,
        ):
            await asyncio.Future()

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        session_id = next(self._session_ids)
        recognizer = KaldiRecognizer(self.model, self.settings.sample_rate)
        if self.settings.enable_word_times:
            recognizer.SetWords(True)
        LOGGER.info("Client #%d connected", session_id)
        await self._semaphore.acquire()
        try:
            await self._serve_session(session_id, websocket, recognizer)
        except ConnectionClosed as exc:
            LOGGER.info("Client #%d disconnected (%s)", session_id, exc)
        except Exception:  # noqa: BLE001 - log unexpected errors and propagate
            LOGGER.exception("Unexpected error for client #%d", session_id)
            raise
        finally:
            self._semaphore.release()
            LOGGER.debug("Client #%d released semaphore", session_id)

    async def _serve_session(
        self,
        session_id: int,
        websocket: WebSocketServerProtocol,
        recognizer: KaldiRecognizer,
    ) -> None:
        while True:
            message = await websocket.recv()
            control = _extract_control(message)
            if control:
                if control == CTRL_EOS:
                    await self._finalize(session_id, websocket, recognizer)
                    await websocket.close()
                    LOGGER.info("Client #%d session completed", session_id)
                    return
                if control == CTRL_RESET:
                    LOGGER.debug("Client #%d reset requested", session_id)
                    recognizer = KaldiRecognizer(self.model, self.settings.sample_rate)
                    if self.settings.enable_word_times:
                        recognizer.SetWords(True)
                    await websocket.send(json.dumps({"type": "info", "text": "reset"}))
                    continue
                LOGGER.debug("Client #%d unknown control: %s", session_id, control)
                continue

            if not isinstance(message, (bytes, bytearray)):
                LOGGER.debug("Client #%d ignored non-binary frame", session_id)
                continue

            if recognizer.AcceptWaveform(message):
                result = _recognizer_result(recognizer)
                transcript = str(result.get("text", "")).strip()
                if transcript:
                    await websocket.send(json.dumps({"type": "partial", "text": transcript}))
            else:
                partial = _recognizer_partial(recognizer)
                transcript = str(partial.get("partial", "")).strip()
                if transcript:
                    await websocket.send(json.dumps({"type": "partial", "text": transcript}))

    async def _finalize(
        self,
        session_id: int,
        websocket: WebSocketServerProtocol,
        recognizer: KaldiRecognizer,
    ) -> None:
        final_payload = _recognizer_final(recognizer)
        transcript = str(final_payload.get("text", "")).strip()
        await websocket.send(json.dumps({"type": "final", "text": transcript}))
        LOGGER.debug("Client #%d final transcript length=%d", session_id, len(transcript))


async def main() -> None:
    try:
        import uvloop  # type: ignore

        uvloop.install()
    except ModuleNotFoundError:
        pass
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = VoskServer(settings)
    await server.start()


def run() -> None:
    """Convenience synchronous entrypoint used by Docker."""
    asyncio.run(main())


if __name__ == "__main__":
    run()

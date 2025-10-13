import asyncio
import json
from typing import Dict

import torch
import websockets

from nemo_loader import load_fastconformer
from batcher import GlobalBatcher
from config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_ATT_CONTEXT,
    DEFAULT_DECODER,
    DEFAULT_DEVICE,
    DEFAULT_STEP_MS,
    DEFAULT_MAX_BATCH,
)
import os


HOST = os.environ.get("ASR_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("ASR_PORT", str(DEFAULT_PORT)))
MODEL_NAME = os.environ.get("ASR_MODEL", DEFAULT_MODEL_NAME)
ATT_CTX_RAW = os.environ.get("ASR_ATT_CTX", f"{DEFAULT_ATT_CONTEXT[0]},{DEFAULT_ATT_CONTEXT[1]}")
try:
    ATT_CONTEXT = tuple(int(x) for x in ATT_CTX_RAW.split(","))
    if len(ATT_CONTEXT) != 2:
        ATT_CONTEXT = DEFAULT_ATT_CONTEXT
except Exception:
    ATT_CONTEXT = DEFAULT_ATT_CONTEXT
DECODER = os.environ.get("ASR_DECODER", DEFAULT_DECODER)
DEVICE = os.environ.get("ASR_DEVICE", DEFAULT_DEVICE)
STEP_MS = int(os.environ.get("ASR_STEP_MS", str(DEFAULT_STEP_MS)))
MAX_BATCH = int(os.environ.get("ASR_MAX_BATCH", str(DEFAULT_MAX_BATCH)))

CLIENTS: Dict[str, websockets.WebSocketServerProtocol] = {}
BATCHER: GlobalBatcher


async def fanout_loop(batcher: GlobalBatcher):
    while True:
        sid, text, ts = await batcher.results.get()
        ws = CLIENTS.get(sid)
        if ws is None:
            continue
        msg = {"op": "interim", "sid": sid, "text": text, "final": False, "ts": ts}
        try:
            await ws.send(json.dumps(msg))
        except Exception:
            pass


async def handler(websocket):
    init = await websocket.recv()
    try:
        init_msg = json.loads(init)
        assert init_msg.get("op") == "init"
        sid = init_msg["sid"]
        sr = int(init_msg.get("sr", 16000))
    except Exception:
        await websocket.close(code=1002, reason="Bad init")
        return

    CLIENTS[sid] = websocket
    await BATCHER.add_stream(sid, sr)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                await BATCHER.push_audio(sid, message)
            else:
                try:
                    msg = json.loads(message)
                    if msg.get("op") == "close":
                        break
                except Exception:
                    pass
    finally:
        CLIENTS.pop(sid, None)
        await BATCHER.remove_stream(sid)
        try:
            await websocket.close()
        except Exception:
            pass


async def main():
    global BATCHER

    model = load_fastconformer(
        model_name=MODEL_NAME,
        att_context_size=ATT_CONTEXT,
        decoder_type=DECODER,
        device=DEVICE,
    )
    # Log effective inference configuration (note: NeMo may print train/val/test configs from the checkpoint; those do not affect inference)
    print(
        f"[server] Config: model={MODEL_NAME} device={DEVICE} step_ms={STEP_MS} max_slots={MAX_BATCH}"
    )
    BATCHER = GlobalBatcher(
        model=model,
        step_ms=STEP_MS,
        sample_rate=16000,
        max_slots=MAX_BATCH,
        device=torch.device(DEVICE),
    )
    await BATCHER.start()
    asyncio.create_task(fanout_loop(BATCHER))

    async with websockets.serve(
        handler, HOST, PORT, max_size=2**22, ping_timeout=30, ping_interval=20
    ):
        print(f"ASR WS server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())



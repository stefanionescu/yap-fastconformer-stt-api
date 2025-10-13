import asyncio
import json
import os
import time
from typing import Dict

import websockets

from nemo_loader import load_fastconformer
from batcher import GlobalBatcher


HOST = os.environ.get("ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PORT", "8080"))

CLIENTS: Dict[str, websockets.WebSocketServerProtocol] = {}
BATCHER: GlobalBatcher


async def fanout_loop(batcher: GlobalBatcher):
    while True:
        await asyncio.sleep(0.02)
        out = await batcher._step_once()
        if not out:
            continue
        sids, texts = out
        now = int(time.time() * 1000)
        for sid, txt in zip(sids, texts):
            ws = CLIENTS.get(sid)
            if ws is None:
                continue
            msg = {"op": "interim", "sid": sid, "text": txt, "final": False, "ts": now}
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

    model_name = os.environ.get(
        "ASR_MODEL", "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )
    att_ctx = os.environ.get("ASR_ATT_CTX", "70,1")
    att_tuple = tuple(int(x) for x in att_ctx.split(","))
    decoder = os.environ.get("ASR_DECODER", "rnnt")
    device = os.environ.get("ASR_DEVICE", "cuda:0")
    step_ms = int(os.environ.get("ASR_STEP_MS", "20"))
    max_batch = int(os.environ.get("ASR_MAX_BATCH", "128"))

    model = load_fastconformer(
        model_name=model_name,
        att_context_size=att_tuple,
        decoder_type=decoder,
        device=device,
    )
    BATCHER = GlobalBatcher(model=model, step_ms=step_ms, sample_rate=16000, max_batch=max_batch)
    await BATCHER.start()
    asyncio.create_task(fanout_loop(BATCHER))

    async with websockets.serve(
        handler, HOST, PORT, max_size=2**22, ping_timeout=30, ping_interval=20
    ):
        print(f"ASR WS server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())



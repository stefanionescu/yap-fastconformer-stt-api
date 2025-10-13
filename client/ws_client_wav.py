import asyncio
import json
import sys
import time
import wave

import websockets


async def main(wav_path: str, ws_url: str = "ws://localhost:8080", sid: str = "demo1"):
    async with websockets.connect(ws_url, max_size=2**22) as ws:
        await ws.send(json.dumps({"op": "init", "sid": sid, "sr": 16000}))

        with wave.open(wav_path, "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            frames_per_chunk = int(0.02 * 16000)
            while True:
                data = w.readframes(frames_per_chunk)
                if not data:
                    break
                await ws.send(data)
                await asyncio.sleep(0.02)

        await ws.send(json.dumps({"op": "close", "sid": sid}))

        end = time.time() + 1.0
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                print(msg)
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ws_client_wav.py <wav_path> [ws_url] [sid]")
        sys.exit(1)
    wav_path = sys.argv[1]
    ws_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8080"
    sid = sys.argv[3] if len(sys.argv) > 3 else "demo1"
    asyncio.run(main(wav_path, ws_url, sid))



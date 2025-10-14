from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CONTROL_PREFIX = b"__CTRL__:"  # matches server control frames
CTRL_EOS = b"EOS"
SAMPLES_DIR = Path("samples")


def resolve_sample_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.parts and path.parts[0] == "samples":
        path = Path(*path.parts[1:])
    return SAMPLES_DIR / path


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype("int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}")
    return np.asarray(audio, dtype=np.int16)


def chunk_audio(audio: np.ndarray, frame_samples: int):
    total = len(audio)
    for start in range(0, total, frame_samples):
        yield audio[start : start + frame_samples]


async def stream_session(
    url: str,
    audio: np.ndarray,
    *,
    frame_ms: int = 20,
    rtf: float = 1.0,
    print_partials: bool = False,
) -> Dict[str, Any]:
    frame_samples = max(1, int(SAMPLE_RATE * (frame_ms / 1000.0)))
    start_ts = time.perf_counter()
    first_audio_ts: float | None = None
    first_partial_latency: float | None = None
    final_ts: float | None = None
    partial_count = 0
    final_text = ""
    done_event = asyncio.Event()

    async with websockets.connect(url, max_size=2**23) as ws:
        async def sender() -> None:
            nonlocal first_audio_ts
            sent_samples = 0
            send_start = time.perf_counter()
            for frame in chunk_audio(audio, frame_samples):
                if frame.size == 0:
                    continue
                await ws.send(frame.tobytes())
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter()
                sent_samples += int(frame.size)
                target = send_start + (sent_samples / SAMPLE_RATE) / max(rtf, 1e-6)
                sleep_for = target - time.perf_counter()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)
            await ws.send(CONTROL_PREFIX + CTRL_EOS)
            try:
                await asyncio.wait_for(done_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

        async def receiver() -> None:
            nonlocal final_text, final_ts, partial_count, first_partial_latency
            async for message in ws:
                now = time.perf_counter()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type") or "")
                text = str(payload.get("text") or "")
                if msg_type == "partial":
                    partial_count += 1
                    if print_partials and text:
                        print(f"[partial] {text}")
                    if first_audio_ts is not None and first_partial_latency is None:
                        first_partial_latency = now - first_audio_ts
                elif msg_type == "final":
                    final_text = text
                    final_ts = now
                    if print_partials and text:
                        print(f"[final] {text}")
                    done_event.set()
                    return
            done_event.set()

        await asyncio.gather(sender(), receiver())
        if not ws.closed:
            await ws.close()

    end_ts = final_ts or time.perf_counter()
    audio_s = len(audio) / SAMPLE_RATE
    wall_s = end_ts - start_ts
    return {
        "text": final_text,
        "audio_s": audio_s,
        "wall_s": wall_s,
        "rtf": (wall_s / audio_s) if audio_s > 0 else float("inf"),
        "xrt": (audio_s / wall_s) if wall_s > 0 else 0.0,
        "partials": partial_count,
        "ttfw_s": first_partial_latency,
    }


__all__ = [
    "resolve_sample_path",
    "load_audio",
    "stream_session",
]

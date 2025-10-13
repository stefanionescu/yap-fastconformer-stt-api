from __future__ import annotations
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, Tuple

import numpy as np
import soundfile as sf
import websockets


# ---------- URL helpers ----------

def build_ws_url(server: str, secure: bool = False) -> str:
    server = (server or "").strip()
    if server.startswith(("ws://", "wss://")):
        return server
    scheme = "wss" if secure else "ws"
    host = server.rstrip("/")
    return f"{scheme}://{host}"


# ---------- Audio helpers (PCM16 mono @ 16 kHz) ----------

SAMPLES_DIR = Path("samples")


def _ffmpeg_decode_to_pcm16_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pcm = np.frombuffer(p.stdout, dtype=np.int16)
    return pcm, 16000


def file_to_pcm16_mono_16k(path: str) -> bytes:
    try:
        x, sr = sf.read(path, dtype="int16", always_2d=False)
        if x.ndim > 1:
            x = x[:, 0]
        if sr != 16000:
            pcm, _ = _ffmpeg_decode_to_pcm16_mono_16k(path)
            return pcm.tobytes()
        return x.tobytes()
    except Exception:
        pcm, _ = _ffmpeg_decode_to_pcm16_mono_16k(path)
        return pcm.tobytes()


def file_duration_seconds(path: str) -> float:
    try:
        f = sf.SoundFile(path)
        return float(len(f) / f.samplerate)
    except Exception:
        pcm, sr = _ffmpeg_decode_to_pcm16_mono_16k(path)
        return float(len(pcm) / sr)


def resolve_sample_path(filename: str) -> Path:
    p = Path(filename)
    if p.is_absolute():
        return p
    return SAMPLES_DIR / filename


# ---------- Streaming helpers ----------

def iter_pcm_chunks(pcm_bytes: bytes, samples_per_chunk: int) -> AsyncIterator[bytes]:
    async def _gen() -> AsyncIterator[bytes]:
        hop_bytes = samples_per_chunk * 2  # int16
        view = memoryview(pcm_bytes)
        for i in range(0, len(pcm_bytes), hop_bytes):
            chunk = view[i : i + hop_bytes]
            if not chunk:
                break
            yield bytes(chunk)
    return _gen()


class InterimCollector:
    def __init__(self) -> None:
        self.partial_ts: list[float] = []
        self.last_partial_ts: float = 0.0
        self.final_text: str = ""
        self.first_partial_since_audio: float | None = None
        self._last_text: str = ""
        self.final_recv_ts: float = 0.0

    def handle_interim(self, msg: Dict[str, object], now: float, first_audio_sent_ts: float | None, t0: float) -> None:
        text = str(msg.get("text") or "").strip()
        if not text:
            return
        if first_audio_sent_ts is not None and self.first_partial_since_audio is None:
            self.first_partial_since_audio = now - first_audio_sent_ts
        if text != self._last_text:
            self.partial_ts.append(now - t0)
            self.last_partial_ts = now
            self._last_text = text
        self.final_text = text


async def run_streaming_session(
    server: str,
    audio_bytes: bytes,
    *,
    rtf: float = 1.0,
    sr: int = 16000,
    chunk_ms: int = 20,
    tail_linger_ms: int = 300,
    secure: bool = False,
    print_partials: bool = False,
) -> Dict[str, object]:
    # Clamp RTF to [1.0, 10.0]
    try:
        rtf = float(rtf)
    except Exception:
        rtf = 1.0
    if rtf < 1.0:
        rtf = 1.0
    elif rtf > 10.0:
        rtf = 10.0
    url = build_ws_url(server, secure=secure)
    sid = uuid.uuid4().hex

    samples_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
    audio_samples = len(audio_bytes) // 2
    audio_duration_s = audio_samples / float(sr)

    t0 = time.perf_counter()
    first_audio_sent_ts: float | None = None
    last_chunk_sent_ts: float | None = None
    collector = InterimCollector()

    async with websockets.connect(url, max_size=2**22, ping_timeout=30, ping_interval=20) as ws:
        init_msg = json.dumps({"op": "init", "sid": sid, "sr": sr})
        await ws.send(init_msg)

        async def recv_loop() -> None:
            nonlocal first_audio_sent_ts
            try:
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        # Server should not send binary frames; ignore.
                        continue
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    now = time.perf_counter()
                    if data.get("op") == "interim":
                        collector.handle_interim(data, now, first_audio_sent_ts, t0)
                        if print_partials and collector.final_text and collector.final_text != "":
                            # Print only when text changes (collector enforces timestamps on change)
                            if collector.partial_ts and abs(now - collector.last_partial_ts) < 0.050:
                                # recently updated
                                print(f"PART: {collector.final_text}")
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if collector.final_recv_ts == 0.0:
                    collector.final_recv_ts = time.perf_counter()

        recv_task = asyncio.create_task(recv_loop())

        # Stream audio with RTF pacing
        stream_start = time.perf_counter()
        samples_sent = 0
        async for chunk in iter_pcm_chunks(audio_bytes, samples_per_chunk):
            await ws.send(chunk)
            if first_audio_sent_ts is None:
                first_audio_sent_ts = time.perf_counter()
            samples_sent += len(chunk) // 2
            last_chunk_sent_ts = time.perf_counter()
            target = stream_start + (samples_sent / sr) / max(rtf, 1e-6)
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                await asyncio.sleep(0)  # yield

        # Linger briefly to allow last interims to arrive
        if tail_linger_ms > 0:
            await asyncio.sleep(tail_linger_ms / 1000.0)

        # Close sequence
        try:
            await ws.send(json.dumps({"op": "close"}))
        except Exception:
            pass

        try:
            await asyncio.wait_for(recv_task, timeout=2.0)
        except asyncio.TimeoutError:
            recv_task.cancel()

    final_ts = collector.final_recv_ts or time.perf_counter()
    wall_s = final_ts - t0

    metrics: Dict[str, object] = {
        "text": collector.final_text,
        "audio_s": audio_duration_s,
        "wall_s": wall_s,
        "rtf": (wall_s / audio_duration_s) if audio_duration_s > 0 else float("inf"),
        "xrt": (audio_duration_s / wall_s) if wall_s > 0 else 0.0,
        "partials": len(collector.partial_ts),
        "avg_partial_gap_ms": (
            float((sum(b - a for a, b in zip(collector.partial_ts[:-1], collector.partial_ts[1:])) / (len(collector.partial_ts) - 1)) * 1000.0)
            if len(collector.partial_ts) >= 2 else 0.0
        ),
        "ttfw_s": float(collector.first_partial_since_audio) if collector.first_partial_since_audio is not None else None,
        "send_duration_s": (last_chunk_sent_ts - t0) if last_chunk_sent_ts else 0.0,
        "post_send_final_s": (final_ts - last_chunk_sent_ts) if last_chunk_sent_ts else 0.0,
        "finalize_ms": ((final_ts - (last_chunk_sent_ts or t0)) * 1000.0),
        "delta_to_audio_ms": ((wall_s - audio_duration_s) * 1000.0),
    }
    return metrics


def summarize_results(title: str, results: list[Dict[str, object]]) -> None:
    if not results:
        print(f"{title}: no results")
        return
    def _vals(key: str) -> list[float]:
        out: list[float] = []
        for r in results:
            v = r.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out
    import statistics as stats
    wall = _vals("wall_s")
    audio = _vals("audio_s")
    rtf = _vals("rtf")
    xrt = _vals("xrt")
    ttfw = [v for v in _vals("ttfw_s") if v is not None]
    def pct(v: list[float], q: float) -> float:
        if not v:
            return 0.0
        k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
        return sorted(v)[k]
    print(f"\n== {title} ==")
    print(f"n={len(results)}")
    print(f"Wall s      | avg={stats.mean(wall):.4f}  p50={stats.median(wall):.4f}  p95={pct(wall,0.95):.4f}")
    print(f"Audio s     | avg={stats.mean(audio):.4f}")
    print(f"RTF         | avg={stats.mean(rtf):.4f}  p50={stats.median(rtf):.4f}  p95={pct(rtf,0.95):.4f}")
    print(f"xRT         | avg={stats.mean(xrt):.4f}")
    if ttfw:
        print(f"TTFW        | avg={stats.mean(ttfw):.4f}  p50={stats.median(ttfw):.4f}  p95={pct(ttfw,0.95):.4f}")



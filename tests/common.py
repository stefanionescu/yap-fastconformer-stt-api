from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Tuple

import aiohttp
import numpy as np
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription

# ---------- URL helpers ----------

def build_offer_url(server: str, *, secure: bool = False, path: str = "/webrtc") -> str:
    server = (server or "").strip()
    if server.startswith(("http://", "https://")):
        base = server.rstrip("/")
    else:
        host = server or "127.0.0.1:8000"
        scheme = "https" if secure else "http"
        base = f"{scheme}://{host.strip('/')}"
    path = path if path.startswith("/") else "/" + path
    return f"{base}{path}"


# ---------- Audio helpers (PCM16 mono @ 16 kHz) ----------

SAMPLES_DIR = Path("samples")


def _ffmpeg_decode_to_pcm16_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm, 16000


def file_to_pcm16_mono_16k(path: str) -> bytes:
    try:
        audio, sr = sf.read(path, dtype="int16", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != 16000:
            pcm, _ = _ffmpeg_decode_to_pcm16_mono_16k(path)
            return pcm.tobytes()
        return audio.tobytes()
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
    if p.exists():
        return p
    parts = p.parts
    if parts and parts[0] == "samples":
        p = Path(*parts[1:]) if len(parts) > 1 else Path("")
    return SAMPLES_DIR / p


# ---------- Streaming helpers ----------


def iter_pcm_chunks(pcm_bytes: bytes, samples_per_chunk: int) -> AsyncIterator[bytes]:
    async def _gen() -> AsyncIterator[bytes]:
        hop = samples_per_chunk * 2
        view = memoryview(pcm_bytes)
        for start in range(0, len(pcm_bytes), hop):
            chunk = view[start : start + hop]
            if not chunk:
                break
            yield bytes(chunk)
    return _gen()


@dataclass
class OfferClient:
    pc: RTCPeerConnection
    channel_open: asyncio.Event
    ready_event: asyncio.Event
    final_event: asyncio.Event
    error: asyncio.Future[BaseException]


class InterimCollector:
    def __init__(self) -> None:
        self.partial_ts: list[float] = []
        self.last_partial_ts: float = 0.0
        self.final_text: str = ""
        self.first_partial_since_audio: float | None = None
        self._last_text: str = ""
        self.final_recv_ts: float = 0.0

    def handle_interim(
        self,
        msg: Dict[str, object],
        now: float,
        first_audio_sent_ts: float | None,
        t0: float,
    ) -> None:
        text = str(msg.get("text") or "").strip()
        if not text and not msg.get("final"):
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
    offer_path: str = "/webrtc",
    headers: dict[str, str] | None = None,
) -> Dict[str, object]:
    if sr != 16000:
        raise ValueError("Moonshine server expects 16 kHz audio")
    url = build_offer_url(server, secure=secure, path=offer_path)
    sid = uuid.uuid4().hex

    pc = RTCPeerConnection()
    channel = pc.createDataChannel("audio")

    channel_open = asyncio.Event()
    ready_event = asyncio.Event()
    final_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    error_future: asyncio.Future[BaseException] = loop.create_future()
    collector = InterimCollector()

    t0 = time.perf_counter()
    first_audio_sent_ts: float | None = None
    last_chunk_sent_ts: float | None = None

    def set_error(exc: BaseException) -> None:
        if not error_future.done():
            error_future.set_result(exc)
        final_event.set()

    @channel.on("open")
    def _on_open() -> None:
        channel_open.set()

    @channel.on("close")
    def _on_close() -> None:
        if not final_event.is_set():
            final_event.set()

    @channel.on("message")
    def _on_message(message) -> None:
        nonlocal first_audio_sent_ts
        now = time.perf_counter()
        if isinstance(message, (bytes, bytearray)):
            return
        try:
            data = json.loads(str(message))
        except json.JSONDecodeError:
            return
        op = str(data.get("op") or "")
        if op == "ready":
            ready_event.set()
            return
        if op == "error":
            set_error(RuntimeError(str(data.get("reason"))))
            return
        if op in {"interim", "final"}:
            collector.handle_interim(data, now, first_audio_sent_ts, t0)
            if print_partials and collector.final_text:
                print(f"PART: {collector.final_text}")
            if data.get("final") or op == "final":
                collector.final_recv_ts = now
                final_event.set()

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            headers=headers,
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Offer failed with status {resp.status}")
            answer = await resp.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    await asyncio.wait_for(channel_open.wait(), timeout=5.0)
    channel.send(json.dumps({"op": "init", "sid": sid, "sr": sr}))
    await asyncio.wait_for(ready_event.wait(), timeout=5.0)

    samples_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
    stream_start = time.perf_counter()
    samples_sent = 0

    async for chunk in iter_pcm_chunks(audio_bytes, samples_per_chunk):
        try:
            channel.send(chunk)
        except Exception as e:
            # DataChannel closed while sending — surface a clean error and stop streaming.
            set_error(RuntimeError(f"datachannel_send_failed: {e}"))
            break
        if first_audio_sent_ts is None:
            first_audio_sent_ts = time.perf_counter()
        samples_sent += len(chunk) // 2
        last_chunk_sent_ts = time.perf_counter()
        target = stream_start + (samples_sent / sr) / max(rtf, 1e-6)
        sleep_for = target - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            await asyncio.sleep(0)

    channel.send(json.dumps({"op": "close"}))
    linger_timeout = max(0.5, tail_linger_ms / 1000.0 + 1.0)
    try:
        await asyncio.wait_for(final_event.wait(), timeout=linger_timeout)
    except asyncio.TimeoutError:
        pass

    await pc.close()
    if error_future.done():
        raise error_future.result()

    final_ts = collector.final_recv_ts or time.perf_counter()
    wall_s = final_ts - t0
    audio_samples = len(audio_bytes) // 2
    audio_duration_s = audio_samples / float(sr)

    metrics: Dict[str, object] = {
        "text": collector.final_text,
        "audio_s": audio_duration_s,
        "wall_s": wall_s,
        "rtf": (wall_s / audio_duration_s) if audio_duration_s > 0 else float("inf"),
        "xrt": (audio_duration_s / wall_s) if wall_s > 0 else 0.0,
        "partials": len(collector.partial_ts),
        "avg_partial_gap_ms": (
            float(
                (
                    sum(b - a for a, b in zip(collector.partial_ts[:-1], collector.partial_ts[1:]))
                    / (len(collector.partial_ts) - 1)
                )
                * 1000.0
            )
            if len(collector.partial_ts) >= 2
            else 0.0
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
        vals: list[float] = []
        for r in results:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    import statistics as stats

    wall = _vals("wall_s")
    audio = _vals("audio_s")
    rtf = _vals("rtf")
    xrt = _vals("xrt")
    ttfw = [v for v in _vals("ttfw_s") if v is not None]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
        return sorted(values)[idx]

    print(f"\n== {title} ==")
    print(f"n={len(results)}")
    print(f"Wall s      | avg={stats.mean(wall):.4f}  p50={stats.median(wall):.4f}  p95={pct(wall,0.95):.4f}")
    print(f"Audio s     | avg={stats.mean(audio):.4f}")
    print(f"RTF         | avg={stats.mean(rtf):.4f}  p50={stats.median(rtf):.4f}  p95={pct(rtf,0.95):.4f}")
    print(f"xRT         | avg={stats.mean(xrt):.4f}")
    if ttfw:
        print(f"TTFW        | avg={stats.mean(ttfw):.4f}  p50={stats.median(ttfw):.4f}  p95={pct(ttfw,0.95):.4f}")


__all__ = [
    "build_offer_url",
    "run_streaming_session",
    "file_to_pcm16_mono_16k",
    "file_duration_seconds",
    "resolve_sample_path",
    "summarize_results",
]

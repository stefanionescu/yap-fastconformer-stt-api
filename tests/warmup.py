#!/usr/bin/env python3
"""
Warmup and quick health check for FastConformer ASR server (WebSocket).

Designed to be copied into Docker image and run inside the container.
"""
from __future__ import annotations
import argparse
import asyncio
import os
from pathlib import Path

from common import (
    run_streaming_session,
    file_to_pcm16_mono_16k,
    file_duration_seconds,
    resolve_sample_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup via WebSocket streaming")
    default_host = os.getenv("ASR_HOST", "127.0.0.1")
    default_port = os.getenv("ASR_PORT", "8080")
    default_server = f"{default_host}:{default_port}"
    parser.add_argument("--server", type=str, default=default_server, help="host:port or ws://host:port")
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--file", type=str, default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--rtf", type=float, default=10.0, help="Real-time factor (1.0-10.0; 10=faster)")
    parser.add_argument("--debug", action="store_true", help="Print partials")
    parser.add_argument("--full-text", action="store_true", help="Print full final text")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    pcm = file_to_pcm16_mono_16k(str(audio_path))
    duration = file_duration_seconds(str(audio_path))

    res = await run_streaming_session(
        args.server,
        pcm,
        rtf=args.rtf,
        sr=16000,
        chunk_ms=20,
        tail_linger_ms=150,
        secure=args.secure,
        print_partials=args.debug,
    )

    text = str(res.get("text", ""))
    if args.full_text:
        print(f"Text: {text}")
    else:
        print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    print(f"Audio duration: {duration:.4f}s")
    wall = float(res.get("wall_s", 0.0))
    print(f"Transcription time (to last interim): {wall:.4f}s")
    rtf_measured = wall / duration if duration > 0 else 0.0
    print(f"RTF(measured): {rtf_measured:.4f}  xRT: {(1.0/rtf_measured) if rtf_measured>0 else 0.0:.2f}x  (target={args.rtf})")
    ttfw = res.get("ttfw_s")
    if ttfw is not None:
        print(f"TTFW: {float(ttfw)*1000.0:.1f}ms")
    print(f"Partials: {int(res.get('partials', 0))}")
    print(f"Δ(audio): {float(res.get('delta_to_audio_ms', 0.0)):.1f}ms")
    print(f"Flush→Final: {float(res.get('finalize_ms', 0.0)):.1f}ms")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())



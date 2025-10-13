#!/usr/bin/env python3
"""
Simple WebSocket client for FastConformer ASR server.

Streams PCM16 mono @16k to ws://host:port and prints partials/final text.
"""
from __future__ import annotations
import argparse
import asyncio
import os
from pathlib import Path

from common import (
    run_streaming_session,
    file_to_pcm16_mono_16k,
    resolve_sample_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS client for FastConformer server")
    # Resolve server from env; default to ASR_HOST:PORT if present or localhost:8080
    default_host = os.getenv("ASR_HOST", "127.0.0.1")
    default_port = os.getenv("ASR_PORT", "8080")
    default_server = f"{default_host}:{default_port}"
    parser.add_argument("--server", default=default_server, help="host:port or ws://host:port")
    parser.add_argument("--secure", action="store_true", help="Use WSS")
    parser.add_argument("--file", type=str, default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0=realtime, >1 faster)")
    parser.add_argument("--full-text", action="store_true", help="Print full final text (default: first 50 chars)")
    parser.add_argument("--print-partials", action="store_true", help="Print partial updates")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    pcm = file_to_pcm16_mono_16k(str(audio_path))
    res = await run_streaming_session(
        args.server,
        pcm,
        rtf=args.rtf,
        sr=16000,
        chunk_ms=20,
        tail_linger_ms=200,
        secure=args.secure,
        print_partials=args.print_partials,
    )
    text = str(res.get("text", ""))
    if args.full_text:
        print(f"Text: {text}")
    else:
        print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")

    # Basic metrics display
    wall = float(res.get("wall_s", 0.0))
    audio_s = float(res.get("audio_s", 0.0))
    rtf = float(res.get("rtf", 0.0))
    xrt = float(res.get("xrt", 0.0))
    ttfw = res.get("ttfw_s")
    print(f"Wall: {wall:.4f}s  Audio: {audio_s:.4f}s  RTF: {rtf:.4f}  xRT: {xrt:.2f}x  TTFW: {0.0 if ttfw is None else float(ttfw):.4f}s")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())



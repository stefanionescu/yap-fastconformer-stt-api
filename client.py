#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os

from tests.common import load_audio, resolve_sample_path, stream_session

DEFAULT_WS_URL = os.getenv("WS", "ws://127.0.0.1:8080/ws")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Parakeet streaming client")
    parser.add_argument("wav", help="Path to 16kHz mono WAV/FLAC/OGG file")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0 = real time)")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.wav)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    audio = load_audio(audio_path)
    result = await stream_session(
        args.url,
        audio,
        frame_ms=args.frame_ms,
        rtf=args.rtf,
        print_partials=args.print_partials,
    )
    text = str(result.get("text", ""))
    print(text)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

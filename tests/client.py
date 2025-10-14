#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os

from common import load_audio, resolve_sample_path, stream_session

DEFAULT_WS_URL = os.getenv("WS", "ws://127.0.0.1:8080/ws")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parakeet streaming smoke test")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint, e.g. ws://host:8080/ws")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0 = real time)")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses as they arrive")
    parser.add_argument("--full-text", action="store_true", help="Print the full final transcript")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    audio_path = resolve_sample_path(args.file)
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
    if args.full_text:
        print(f"Final: {text}")
    else:
        print(f"Final: {text[:80]}…" if len(text) > 80 else f"Final: {text}")
    wall = float(result.get("wall_s", 0.0))
    audio_s = float(result.get("audio_s", 0.0))
    rtf = float(result.get("rtf", 0.0))
    xrt = float(result.get("xrt", 0.0))
    ttfw = result.get("ttfw_s")
    print(
        "Wall={:.3f}s  Audio={:.3f}s  RTF={:.3f}  xRT={:.2f}x  Partials={}  TTFW={:.1f}ms".format(
            wall,
            audio_s,
            rtf,
            xrt,
            int(result.get("partials", 0)),
            (ttfw * 1000.0) if isinstance(ttfw, (int, float)) else float("nan"),
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())

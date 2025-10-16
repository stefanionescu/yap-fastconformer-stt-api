#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from utils import SAMPLE_RATE, load_audio, resolve_sample_path, stream_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real audio concurrency bench for the Vosk server")
    parser.add_argument("--url", default=os.getenv("WS", "ws://127.0.0.1:8000"), help="WebSocket endpoint")
    parser.add_argument("--streams", type=int, default=16, help="Number of concurrent streams")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor for sending audio")
    parser.add_argument("--print-partials", action="store_true", help="Print partial hypotheses")
    parser.add_argument("--print-finals", action="store_true", help="Print final transcripts from each stream")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    # Load real audio file like warmup.py does
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return
    
    audio = load_audio(audio_path)
    # Ensure contiguous int16 for consistent wire format (s16le)
    if audio.dtype != "int16":
        audio = np.clip(audio.astype("float32", copy=False), -1.0, 1.0)
        audio = (audio * 32767.0).astype("int16")
    else:
        audio = audio.astype("int16", copy=False)
    
    duration = len(audio) / SAMPLE_RATE
    print(f"Using audio: {audio_path}")
    print(f"Audio duration: {duration:.3f}s")
    print(f"Running {args.streams} concurrent streams...")

    results = []
    
    async def session(stream_id: int) -> dict:
        result = await stream_session(
            args.url,
            audio,
            frame_ms=args.frame_ms,
            rtf=args.rtf,
            print_partials=args.print_partials,
        )
        
        if args.print_finals:
            text = str(result.get("text", ""))
            wall = float(result.get("wall_s", 0.0))
            rtf_measured = wall / duration if duration > 0 else 0.0
            print(f"Stream #{stream_id}: Final='{text[:60]}...' wall={wall:.3f}s RTF={rtf_measured:.3f}")
        
        return result

    tasks = [asyncio.create_task(session(i)) for i in range(int(args.streams))]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    
    # Calculate aggregate stats
    total_wall = sum(float(r.get("wall_s", 0.0)) for r in results)
    avg_wall = total_wall / len(results) if results else 0.0
    avg_rtf = avg_wall / duration if duration > 0 else 0.0
    throughput = len(results) * duration / elapsed if elapsed > 0 else 0.0
    
    print(f"\n=== Benchmark Results ===")
    print(f"Streams: {args.streams}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Avg per-stream wall: {avg_wall:.3f}s")
    print(f"Avg RTF: {avg_rtf:.3f}")
    print(f"Throughput: {throughput:.2f}x realtime")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

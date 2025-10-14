#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import statistics as stats
import time

from common import load_audio, resolve_sample_path, stream_session

DEFAULT_WS_URL = os.getenv("WS", "ws://127.0.0.1:8080/ws")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrency benchmark for Parakeet streaming server")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint")
    parser.add_argument("--file", default="mid.wav", help="Audio file (absolute path or under samples/)")
    parser.add_argument("--n", type=int, default=32, help="Total sessions to run")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent sessions")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor for sending audio")
    parser.add_argument("--frame-ms", type=int, default=20, help="Frame size in milliseconds")
    parser.add_argument("--print-partials", action="store_true", help="Emit partial hypotheses while streaming")
    return parser.parse_args()


async def run_benchmark(args: argparse.Namespace) -> tuple[list[dict], int]:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    audio = load_audio(audio_path)

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    results: list[dict] = []

    async def worker(_: int) -> None:
        async with sem:
            res = await stream_session(
                args.url,
                audio,
                frame_ms=args.frame_ms,
                rtf=args.rtf,
                print_partials=args.print_partials,
            )
            results.append(res)

    tasks = [asyncio.create_task(worker(i)) for i in range(int(args.n))]
    await asyncio.gather(*tasks)
    return results, len(tasks)


def summarize(results: list[dict]) -> None:
    if not results:
        print("No results to summarise")
        return

    def collect(key: str) -> list[float]:
        vals = []
        for item in results:
            val = item.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return vals

    wall = collect("wall_s")
    audio = collect("audio_s")
    rtf = collect("rtf")
    ttfw = [v for v in collect("ttfw_s") if v is not None]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
        return sorted(values)[idx]

    print(f"Sessions: {len(results)}")
    print(f"Wall avg={stats.mean(wall):.3f}s  p50={stats.median(wall):.3f}s  p95={pct(wall,0.95):.3f}s")
    print(f"RTF  avg={stats.mean(rtf):.3f}  p50={stats.median(rtf):.3f}  p95={pct(rtf,0.95):.3f}")
    if ttfw:
        print(f"TTFW avg={stats.mean(ttfw)*1000.0:.1f}ms  p50={stats.median(ttfw)*1000.0:.1f}ms  p95={pct(ttfw,0.95)*1000.0:.1f}ms")
    total_audio = sum(audio)
    total_wall = sum(wall)
    if total_wall > 0:
        print(f"Throughput: {total_audio / total_wall:.2f}× real-time ({total_audio:.1f}s audio in {total_wall:.1f}s)")


def main() -> None:
    args = parse_args()
    print(
        "Benchmark → url={} n={} concurrency={} rtf={}".format(
            args.url,
            args.n,
            args.concurrency,
            args.rtf,
        )
    )
    t0 = time.perf_counter()
    results, _ = asyncio.run(run_benchmark(args))
    elapsed = time.perf_counter() - t0
    summarize(results)
    print(f"Total elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()

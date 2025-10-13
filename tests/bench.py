#!/usr/bin/env python3
"""
Benchmark WebSocket streaming for FastConformer server.

Runs N sessions with max concurrency, reports latency/RTF/xRT.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import time

from common import (
    run_streaming_session,
    file_to_pcm16_mono_16k,
    resolve_sample_path,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WebSocket streaming benchmark")
    default_host = os.getenv("ASR_HOST", "127.0.0.1")
    default_port = os.getenv("ASR_PORT", "8000")
    default_server = f"{default_host}:{default_port}"
    ap.add_argument("--server", default=default_server, help="host:port or ws://host:port or full URL")
    ap.add_argument("--secure", action="store_true", help="Use WSS")
    ap.add_argument("--n", type=int, default=20, help="Total sessions")
    ap.add_argument("--concurrency", type=int, default=5, help="Max concurrent sessions")
    ap.add_argument("--file", type=str, default="mid.wav", help="Audio file (under samples/ or absolute)")
    ap.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0-10.0; 10=faster)")
    ap.add_argument("--print-partials", action="store_true", help="Print partials during sessions")
    return ap.parse_args()


async def run_benchmark(args: argparse.Namespace) -> tuple[list[dict], int, int, float]:
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        return [], 0, args.n, 0.0
    pcm = file_to_pcm16_mono_16k(str(audio_path))

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    results: list[dict] = []
    rejected = 0
    errors = 0

    async def worker(i: int) -> None:
        nonlocal rejected, errors
        async with sem:
            try:
                res = await run_streaming_session(
                    args.server,
                    pcm,
                    rtf=args.rtf,
                    sr=16000,
                    chunk_ms=20,
                    tail_linger_ms=150,
                    secure=args.secure,
                    print_partials=args.print_partials,
                )
                results.append(res)
            except Exception as e:
                # No capacity differentiation at server; count all as errors
                errors += 1

    t0 = time.time()
    tasks = [asyncio.create_task(worker(i)) for i in range(int(args.n))]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    return results, rejected, errors, elapsed


def main() -> None:
    args = parse_args()
    print(
        f"Benchmark → WS | n={args.n} | concurrency={args.concurrency} | rtf={args.rtf} | server={args.server}"
    )
    res, rejected, errors, elapsed = asyncio.run(run_benchmark(args))
    summarize_results("WebSocket Streaming", res)
    print(f"Rejected: {rejected}")
    print(f"Errors: {errors}")
    print(f"Total elapsed: {elapsed:.4f}s")
    if res:
        total_audio = sum(float(r.get("audio_s", 0.0)) for r in res)
        print(
            f"Total audio processed: {total_audio:.2f}s | Overall throughput: {total_audio/elapsed*60:.2f} sec/min = {total_audio/elapsed:.2f} min/min"
        )


if __name__ == "__main__":
    main()



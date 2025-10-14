#!/usr/bin/env python3
"""
Benchmark WebRTC streaming for the Moonshine ASR server.

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
    ap = argparse.ArgumentParser(description="Moonshine WebRTC streaming benchmark")
    default_host = os.getenv("ASR_HOST", "127.0.0.1")
    default_port = os.getenv("ASR_PORT", "8000")
    default_server = f"{default_host}:{default_port}"
    ap.add_argument("--server", default=default_server, help="host:port or full http(s) URL")
    ap.add_argument("--path", default=os.getenv("ASR_WEBRTC_PATH", "/webrtc"), help="Offer endpoint path")
    ap.add_argument("--secure", action="store_true", help="Use HTTPS for offer exchange")
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
    errors = 0

    async def worker(_: int) -> None:
        nonlocal errors
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
                    offer_path=args.path,
                )
                results.append(res)
            except Exception:
                errors += 1

    t0 = time.time()
    tasks = [asyncio.create_task(worker(i)) for i in range(int(args.n))]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    return results, 0, errors, elapsed


def main() -> None:
    args = parse_args()
    print(
        "Benchmark → WebRTC | n={} | concurrency={} | rtf={} | server={}".format(
            args.n,
            args.concurrency,
            args.rtf,
            args.server,
        )
    )
    res, rejected, errors, elapsed = asyncio.run(run_benchmark(args))
    summarize_results("WebRTC Streaming", res)
    print(f"Rejected: {rejected}")
    print(f"Errors: {errors}")
    print(f"Total elapsed: {elapsed:.4f}s")
    if res:
        total_audio = sum(float(r.get("audio_s", 0.0)) for r in res)
        print(
            "Total audio processed: {:.2f}s | Overall throughput: {:.2f} sec/min = {:.2f} min/min".format(
                total_audio,
                total_audio / elapsed * 60.0,
                total_audio / elapsed,
            )
        )


if __name__ == "__main__":
    main()

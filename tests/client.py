#!/usr/bin/env python3
"""
Flexible Moonshine client for local or RunPod-hosted ASR endpoints.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from common import (
    run_streaming_session,
    file_to_pcm16_mono_16k,
    resolve_sample_path,
)
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moonshine ASR client")
    parser.add_argument("--server", help="Explicit host:port or full http(s) URL")
    parser.add_argument("--https-url", help="Full HTTPS URL to the offer endpoint (overrides profile/server)")
    parser.add_argument("--tcp-host", help="TCP host to use when constructing the offer URL")
    parser.add_argument("--tcp-port", type=int, help="TCP port to use when constructing the offer URL")
    parser.add_argument("--path", help="Offer endpoint path (defaults to profile or /webrtc)")
    parser.add_argument("--secure", action="store_true", help="Force HTTPS when using host:port")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file with RunPod settings")
    parser.add_argument("--api-key", help="RunPod API key (overrides profile/env)")
    parser.add_argument("--file", type=str, default="mid.wav", help="Audio file (absolute or under samples/)")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor (1.0-10.0; 10=faster)")
    parser.add_argument("--full-text", action="store_true", help="Print full final text")
    parser.add_argument("--print-partials", action="store_true", help="Print partial updates")
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> int:
    env_file_values = load_env_file(args.env_file)
    audio_path = resolve_sample_path(args.file)
    if not audio_path.exists():
        print(f"Audio not found: {audio_path}")
        return 2
    pcm = file_to_pcm16_mono_16k(str(audio_path))

    server, path_hint = resolve_endpoint(args, env_file_values)
    offer_path = resolve_offer_path(args, env_file_values, path_hint)
    headers = resolve_headers(args, env_file_values)
    secure = decide_secure(args, server)

    res = await run_streaming_session(
        server,
        pcm,
        rtf=args.rtf,
        sr=16000,
        chunk_ms=20,
        tail_linger_ms=200,
        secure=secure,
        print_partials=args.print_partials,
        offer_path=offer_path,
        headers=headers,
    )
    text = str(res.get("text", ""))
    if args.full_text:
        print(f"Text: {text}")
    else:
        print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")

    wall = float(res.get("wall_s", 0.0))
    audio_s = float(res.get("audio_s", 0.0))
    rtf = float(res.get("rtf", 0.0))
    xrt = float(res.get("xrt", 0.0))
    ttfw = res.get("ttfw_s")
    print(
        "Wall: {:.4f}s  Audio: {:.4f}s  RTF: {:.4f}  xRT: {:.2f}x  TTFW: {:.4f}s".format(
            wall,
            audio_s,
            rtf,
            xrt,
            0.0 if ttfw is None else float(ttfw),
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_once(args))


def resolve_endpoint(args: argparse.Namespace, env_values: Dict[str, str]) -> Tuple[str, Optional[str]]:
    # Highest priority: explicit https URL
    if args.https_url:
        return split_url(args.https_url)

    if args.server:
        parsed = urlparse(args.server)
        if parsed.scheme in {"http", "https"}:
            return split_url(args.server)
        return args.server, None

    if args.tcp_host and args.tcp_port:
        return f"{args.tcp_host}:{int(args.tcp_port)}", None

    profile_https = env_values.get("RUNPOD_HTTPS_URL") or os.getenv("RUNPOD_HTTPS_URL")
    if profile_https:
        return split_url(profile_https)

    host = args.tcp_host or env_values.get("RUNPOD_TCP_HOST") or os.getenv("RUNPOD_TCP_HOST")
    port = args.tcp_port or env_values.get("RUNPOD_TCP_PORT") or os.getenv("RUNPOD_TCP_PORT")
    if host and port:
        return f"{host}:{port}", None

    host = os.getenv("ASR_HOST", "127.0.0.1")
    port = os.getenv("ASR_PORT", "8000")
    return f"{host}:{port}", None


def resolve_offer_path(args: argparse.Namespace, env_values: Dict[str, str], path_hint: Optional[str]) -> str:
    if args.path:
        return normalise_path(args.path)
    if path_hint:
        return normalise_path(path_hint)
    return normalise_path(env_values.get("RUNPOD_WEBRTC_PATH", "/webrtc"))


def resolve_headers(args: argparse.Namespace, env_values: Dict[str, str]) -> dict[str, str] | None:
    api_key = args.api_key or env_values.get("RUNPOD_API_KEY") or os.getenv("RUNPOD_API_KEY")
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def decide_secure(args: argparse.Namespace, server: str) -> bool:
    if args.secure:
        return True
    return server.startswith("https://")


def split_url(url: str) -> Tuple[str, Optional[str]]:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else url
    path = parsed.path or None
    return base, path


def normalise_path(path: Optional[str]) -> str:
    if not path:
        return "/webrtc"
    return path if path.startswith("/") else "/" + path


def load_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


if __name__ == "__main__":
    raise SystemExit(main())

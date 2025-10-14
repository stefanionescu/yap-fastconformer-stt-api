from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "moonshine/base"
    onnx_providers: Tuple[str, ...] = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    onnx_engine_dir: str | None = None
    onnx_provider_options: dict[str, dict[str, str]] | None = None
    max_batch_size: int = 32
    max_batch_wait_ms: int = 10
    sample_rate: int = 16000
    max_buffer_seconds: float = 90.0
    linger_after_close_ms: int = 250
    http_offer_path: str = "/webrtc"
    log_level: str = "INFO"
    warmup_seconds: float = 1.5


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def load_config() -> Config:
    host = _env("ASR_HOST", "0.0.0.0") or "0.0.0.0"
    port = int(os.environ.get("ASR_PORT", "8000") or 8000)
    model_name = _env("MOONSHINE_MODEL", "moonshine/base") or "moonshine/base"
    providers_raw = _env("MOONSHINE_ONNX_PROVIDERS", "")
    if providers_raw:
        onnx_providers = tuple(p.strip() for p in providers_raw.split(",") if p.strip()) or (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )
    else:
        onnx_providers = (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )
    onnx_engine_dir = _env("MOONSHINE_TRT_ENGINE_DIR", "") or None
    provider_opts_env = _env("MOONSHINE_ONNX_PROVIDER_OPTIONS", "")
    onnx_provider_options: dict[str, dict[str, str]] | None = None
    if provider_opts_env:
        # Expect comma separated provider=key1:val1;key2:val2 entries
        parsed: dict[str, dict[str, str]] = {}
        for segment in provider_opts_env.split(","):
            segment = segment.strip()
            if not segment or "=" not in segment:
                continue
            provider, options_blob = segment.split("=", 1)
            option_pairs = {}
            for opt in options_blob.split(";"):
                opt = opt.strip()
                if not opt or ":" not in opt:
                    continue
                key, value = opt.split(":", 1)
                option_pairs[key.strip()] = value.strip()
            if option_pairs:
                parsed[provider.strip()] = option_pairs
        if parsed:
            onnx_provider_options = parsed
    max_batch_size = max(1, int(os.environ.get("MAX_BATCH_SIZE", "32") or 32))
    max_batch_wait_ms = max(1, int(os.environ.get("MAX_BATCH_WAIT_MS", "10") or 10))
    sample_rate = int(os.environ.get("ASR_SAMPLE_RATE", "16000") or 16000)
    max_buffer_seconds = float(os.environ.get("MAX_BUFFER_SECONDS", "90.0") or 90.0)
    linger_after_close_ms = int(os.environ.get("WEBSOCKET_LINGER_MS", "250") or 250)
    http_offer_path = _env("WEBRTC_OFFER_PATH", "/webrtc") or "/webrtc"
    log_level = _env("ASR_LOG_LEVEL", "INFO") or "INFO"
    warmup_seconds = float(os.environ.get("MODEL_WARMUP_SECONDS", "1.5") or 1.5)

    return Config(
        host=host,
        port=port,
        model_name=model_name,
        onnx_providers=onnx_providers,
        onnx_engine_dir=onnx_engine_dir,
        onnx_provider_options=onnx_provider_options,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        sample_rate=sample_rate,
        max_buffer_seconds=max_buffer_seconds,
        linger_after_close_ms=linger_after_close_ms,
        http_offer_path=http_offer_path,
        log_level=log_level,
        warmup_seconds=warmup_seconds,
    )


__all__ = ["Config", "load_config"]

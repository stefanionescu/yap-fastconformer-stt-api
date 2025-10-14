from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Precision = Literal["fp16", "bf16", "fp32", "int8"]


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    model_id: str = "UsefulSensors/moonshine-base"
    precision: Precision = "fp16"
    max_batch_size: int = 32
    max_batch_wait_ms: int = 10
    sample_rate: int = 16000
    max_buffer_seconds: float = 120.0
    idle_session_timeout_s: float = 30.0
    linger_after_close_ms: int = 250
    http_offer_path: str = "/webrtc"
    log_level: str = "INFO"
    allow_cpu_fallback: bool = True
    warmup_seconds: float = 1.5


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def load_config() -> Config:
    host = _env("ASR_HOST", "0.0.0.0") or "0.0.0.0"
    port = int(os.environ.get("ASR_PORT", "8000") or 8000)
    model_id = _env("MOONSHINE_MODEL_ID", "UsefulSensors/moonshine-base")
    precision = _env("MOONSHINE_PRECISION", "fp16").lower()
    if precision not in {"fp16", "bf16", "fp32", "int8"}:
        precision = "fp16"
    max_batch_size = max(1, int(os.environ.get("MAX_BATCH_SIZE", "32") or 32))
    max_batch_wait_ms = max(1, int(os.environ.get("MAX_BATCH_WAIT_MS", "10") or 10))
    sample_rate = int(os.environ.get("ASR_SAMPLE_RATE", "16000") or 16000)
    max_buffer_seconds = float(os.environ.get("MAX_BUFFER_SECONDS", "120.0") or 120.0)
    idle_session_timeout_s = float(os.environ.get("SESSION_IDLE_TIMEOUT", "30.0") or 30.0)
    linger_after_close_ms = int(os.environ.get("WEBSOCKET_LINGER_MS", "250") or 250)
    http_offer_path = _env("WEBRTC_OFFER_PATH", "/webrtc") or "/webrtc"
    log_level = _env("ASR_LOG_LEVEL", "INFO") or "INFO"
    allow_cpu_fallback = _env("ALLOW_CPU_FALLBACK", "true").lower() in {"1", "true", "yes"}
    warmup_seconds = float(os.environ.get("MODEL_WARMUP_SECONDS", "1.5") or 1.5)

    return Config(
        host=host,
        port=port,
        model_id=model_id,
        precision=precision,  # type: ignore[arg-type]
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        sample_rate=sample_rate,
        max_buffer_seconds=max_buffer_seconds,
        idle_session_timeout_s=idle_session_timeout_s,
        linger_after_close_ms=linger_after_close_ms,
        http_offer_path=http_offer_path,
        log_level=log_level,
        allow_cpu_fallback=allow_cpu_fallback,
        warmup_seconds=warmup_seconds,
    )


__all__ = ["Config", "load_config"]

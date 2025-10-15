"""Configuration helpers for the Vosk streaming server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Application-level configuration resolved from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: Path = Path("/models/en")
    sample_rate: int = 16_000
    concurrency: int = 64
    max_message_size: int = 1 << 23  # 8 MiB
    enable_word_times: bool = True
    vosk_log_level: int = -1
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        model_dir = Path(os.getenv("MODEL_DIR", "/models/en")).expanduser().resolve()
        sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
        concurrency = max(1, int(os.getenv("CONCURRENCY", "64")))
        max_message_size = int(os.getenv("MAX_WS_MESSAGE_SIZE", str(1 << 23)))
        enable_word_times = _env_bool("ENABLE_WORD_TIMES", True)
        vosk_log_level = int(os.getenv("VOSK_LOG_LEVEL", "-1"))
        log_level = os.getenv("LOG_LEVEL", "INFO")
        return cls(
            host=host,
            port=port,
            model_dir=model_dir,
            sample_rate=sample_rate,
            concurrency=concurrency,
            max_message_size=max_message_size,
            enable_word_times=enable_word_times,
            vosk_log_level=vosk_log_level,
            log_level=log_level,
        )

from __future__ import annotations

# Manual configuration defaults (edit here when changing defaults).

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8080

DEFAULT_MODEL_NAME: str = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
DEFAULT_ATT_CONTEXT = (70, 1)  # (lookahead, heads)
DEFAULT_DECODER: str = "rnnt"
DEFAULT_DEVICE: str = "cuda:0"
DEFAULT_STEP_MS: int = 20
DEFAULT_MAX_BATCH: int = 128



#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate
export HF_HOME="${HF_HOME:-$(pwd)/.cache/hf}"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

export STEP_MS="${STEP_MS:-80}"
export RIGHT_CONTEXT_MS="${RIGHT_CONTEXT_MS:-120}"
export MIN_EMIT_CHARS="${MIN_EMIT_CHARS:-1}"
export MAX_INFLIGHT_STEPS="${MAX_INFLIGHT_STEPS:-4}"

echo "[05] Starting server (uvicorn)…"
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000 --ws auto --loop uvloop

#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-parakeet-tdt-streaming}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PORT="${PORT:-8080}"

exec docker run --rm -it \
  --gpus all \
  -p "$PORT:8080" \
  -e STEP_MS="${STEP_MS:-240}" \
  -e RIGHT_CONTEXT_MS="${RIGHT_CONTEXT_MS:-160}" \
  -e MAX_INFLIGHT_STEPS="${MAX_INFLIGHT_STEPS:-1}" \
  -e HF_HOME="${HF_HOME:-/cache/hf}" \
  "$IMAGE_NAME:$IMAGE_TAG"

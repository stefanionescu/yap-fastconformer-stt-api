#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-moonshine-asr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PORT="${ASR_PORT:-8000}"
MODEL_ID="${MOONSHINE_MODEL_ID:-UsefulSensors/moonshine-base}"
PRECISION="${MOONSHINE_PRECISION:-fp16}"
MAX_BATCH="${MAX_BATCH_SIZE:-32}"

exec docker run --rm -it \
  --gpus all \
  -p "$PORT:8000" \
  -e MOONSHINE_MODEL_ID="$MODEL_ID" \
  -e MOONSHINE_PRECISION="$PRECISION" \
  -e MAX_BATCH_SIZE="$MAX_BATCH" \
  "$IMAGE_NAME:$IMAGE_TAG"

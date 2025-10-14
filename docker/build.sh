#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-parakeet-tdt-streaming}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  -f "$ROOT_DIR/docker/Dockerfile" \
  "$ROOT_DIR"

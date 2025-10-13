#!/usr/bin/env bash
set -euo pipefail

export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}

docker run --rm -it \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -p 8000:8000 \
  fastconf-streaming:latest



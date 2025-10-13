#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_USE_CUDA_DSA=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDNN_BENCHMARK=1

exec python /app/server/asr_server.py



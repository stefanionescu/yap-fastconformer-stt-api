#!/usr/bin/env bash
# Centralized environment configuration for non-Docker deployments.
# Copy or export overrides before running other scripts if needed.

# Server bind
export ASR_HOST="0.0.0.0"
export ASR_PORT="8080"

# Model + runtime
export ASR_MODEL="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
export ASR_ATT_CTX="70,1"           # look-ahead: e.g., 70,0 / 70,1 / 70,16 / 70,33
export ASR_DECODER="rnnt"           # rnnt or ctc
export ASR_DEVICE="cuda:0"          # GPU device
export ASR_STEP_MS="20"             # batcher tick period (ms)
export ASR_MAX_BATCH="128"          # max concurrent streams per tick

# Python performance knobs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional: torch/cuDNN flags
export TORCH_USE_CUDA_DSA=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDNN_BENCHMARK=1



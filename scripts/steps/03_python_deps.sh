#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate

echo "[03] Detecting NVIDIA CUDA capability for PyTorch wheel selection…"
CUDA_VER_STR=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $9; exit}')
CUDA_MAJOR=$(echo "${CUDA_VER_STR:-0.0}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VER_STR:-0.0}" | cut -d. -f2)

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
TORCH_VER="2.4.0"
TORCHAUDIO_VER="2.4.0"
TORCHVISION_VER="0.19.0"

# Prefer cu124 with torch 2.5.1+ when driver supports >= CUDA 12.4
if [[ -n "$CUDA_VER_STR" ]]; then
  if (( CUDA_MAJOR > 12 )) || { (( CUDA_MAJOR == 12 )) && (( CUDA_MINOR >= 4 )); }; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    TORCH_VER="2.5.1"
    TORCHAUDIO_VER="2.5.1"
    TORCHVISION_VER="0.20.1"
  fi
fi

echo "[03] Installing PyTorch (torch=${TORCH_VER}, cuda=${TORCH_INDEX_URL##*/})…"
pip install --index-url "$TORCH_INDEX_URL" \
  torch=="$TORCH_VER" torchvision=="$TORCHVISION_VER" torchaudio=="$TORCHAUDIO_VER"

echo "[03] Installing Python deps…"
pip install -r requirements.txt

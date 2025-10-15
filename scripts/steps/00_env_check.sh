#!/usr/bin/env bash
set -euo pipefail

echo "[00] Env check…"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: This installer supports Linux only." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver / CUDA first." >&2
  exit 1
fi
nvidia-smi || true

# Parse driver supported CUDA version from nvidia-smi output
SMI_OUT=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
CUDA_VER_STR=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $9; exit}')
echo "[00] NVIDIA driver=$SMI_OUT CUDA(SMI)=$CUDA_VER_STR"

if [[ -n "$CUDA_VER_STR" ]]; then
  CUDA_MAJOR=$(echo "$CUDA_VER_STR" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_VER_STR" | cut -d. -f2)
  # Require >= 12.4 minimally; recommend >= 12.6 for NeMo 2.5
  if (( CUDA_MAJOR < 12 )) || { (( CUDA_MAJOR == 12 )) && (( CUDA_MINOR < 4 )); }; then
    echo "ERROR: Detected CUDA $CUDA_VER_STR (<12.4). Please upgrade NVIDIA driver to support CUDA >=12.6 for best performance." >&2
    exit 1
  fi
else
  echo "WARNING: Could not detect CUDA version from nvidia-smi. Proceeding cautiously." >&2
fi

if command -v python3.12 >/dev/null 2>&1; then
  PYV=$(python3.12 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
  echo "Python3.12 detected: $PYV"
else
  echo "Python3.12 not found (will install in 01_system_deps.sh)."
fi

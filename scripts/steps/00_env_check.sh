#!/usr/bin/env bash
set -euo pipefail

echo "[00] Env check…"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver / CUDA first." >&2
  exit 1
fi
nvidia-smi || true

if command -v python3.12 >/dev/null 2>&1; then
  PYV=$(python3.12 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
  echo "Python3.12 detected: $PYV"
else
  echo "Python3.12 not found (will install in 01_system_deps.sh)."
fi

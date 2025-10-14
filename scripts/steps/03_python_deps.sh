#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate

echo "[03] Installing PyTorch CUDA wheels…"
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

echo "[03] Installing Python deps…"
pip install -r requirements.txt

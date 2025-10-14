#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"

if pgrep -f "server\.asr_server" >/dev/null 2>&1; then
  echo "Stopping running Moonshine ASR server processes..."
  pkill -f "server\.asr_server" || true
fi

if [[ -d "$VENV_PATH" ]]; then
  echo "Removing virtual environment at $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

echo "Clearing Python __pycache__ directories"
find "$ROOT_DIR" -type d -name '__pycache__' -exec rm -rf {} +

HF_CACHE_DIR="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}}"
if [[ -d "$HF_CACHE_DIR" ]]; then
  echo "Removing Hugging Face cache at $HF_CACHE_DIR"
  rm -rf "$HF_CACHE_DIR"
fi

if [[ -d "$HOME/.cache/hf_transfer" ]]; then
  echo "Removing hf_transfer cache"
  rm -rf "$HOME/.cache/hf_transfer"
fi

if [[ -d "$HOME/.cache/torch" ]]; then
  echo "Removing torch cache"
  rm -rf "$HOME/.cache/torch"
fi

if [[ -d "$HOME/.cache/pip" ]]; then
  echo "Removing pip cache"
  rm -rf "$HOME/.cache/pip"
fi

if [[ -d "$HOME/.cache/aiohttp" ]]; then
  echo "Removing aiohttp cache"
  rm -rf "$HOME/.cache/aiohttp"
fi

if [[ -d "$HOME/.cache/aioice" ]]; then
  echo "Removing aioice cache"
  rm -rf "$HOME/.cache/aioice"
fi

if [[ -d "$HOME/.local/share/aiortc" ]]; then
  echo "Removing aiortc runtime data"
  rm -rf "$HOME/.local/share/aiortc"
fi

echo "Cleanup complete."

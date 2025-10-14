#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCH_CUDA="${TORCH_CUDA:-}"

if [[ ! -d "$VENV_PATH" ]]; then
  "${PYTHON_BIN}" -m venv "$VENV_PATH"
fi
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

pip install --upgrade pip wheel
pip install -r "$ROOT_DIR/requirements.txt"

if [[ -n "$TORCH_CUDA" ]]; then
  if [[ "$TORCH_CUDA" == cpu ]]; then
    pip install --no-cache-dir "torch==${TORCH_VERSION}"
  else
    pip install --no-cache-dir \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      "torch==${TORCH_VERSION}+${TORCH_CUDA}"
  fi
else
  pip install --no-cache-dir "torch==${TORCH_VERSION}"
fi

echo "Environment ready at $VENV_PATH"

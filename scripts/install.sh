#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCH_CUDA="${TORCH_CUDA:-}"

APT_UPDATED=0
PKG_LOG="$ROOT_DIR/.moonshine_installed_packages"

ensure_system_dep() {
  local dep="$1"
  local install_hint="$2"
  if command -v "$dep" >/dev/null 2>&1; then
    return 0
  fi

  echo "Missing dependency '$dep'. Attempting automatic install..." >&2

  if command -v apt-get >/dev/null 2>&1; then
    if [[ ${APT_UPDATED:-0} -eq 0 ]]; then
      apt-get update
      APT_UPDATED=1
    fi
    apt-get install -y $install_hint
    mkdir -p "$(dirname "$PKG_LOG")"
    for pkg in $install_hint; do
      if ! grep -qx "$pkg" "$PKG_LOG" 2>/dev/null; then
        printf '%s\n' "$pkg" >> "$PKG_LOG"
      fi
    done
  elif command -v yum >/dev/null 2>&1; then
    yum install -y $install_hint
    mkdir -p "$(dirname "$PKG_LOG")"
    for pkg in $install_hint; do
      if ! grep -qx "$pkg" "$PKG_LOG" 2>/dev/null; then
        printf '%s\n' "$pkg" >> "$PKG_LOG"
      fi
    done
  elif command -v brew >/dev/null 2>&1; then
    brew install $install_hint
    mkdir -p "$(dirname "$PKG_LOG")"
    for pkg in $install_hint; do
      if ! grep -qx "$pkg" "$PKG_LOG" 2>/dev/null; then
        printf '%s\n' "$pkg" >> "$PKG_LOG"
      fi
    done
  else
    echo "Could not auto-install '$dep'. Please install it manually (hint: $install_hint)." >&2
    exit 1
  fi

  if ! command -v "$dep" >/dev/null 2>&1; then
    echo "Dependency '$dep' still missing after attempted install." >&2
    exit 1
  fi
}

ensure_system_dep "pkg-config" "pkg-config libavformat-dev libavdevice-dev libavcodec-dev libavutil-dev libswscale-dev libopus-dev libvpx-dev"
ensure_system_dep "ffmpeg" "ffmpeg"

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

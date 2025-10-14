#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
PKG_LOG="$ROOT_DIR/.moonshine_installed_packages"

log() {
  printf '[stop] %s\n' "$*"
}

if [[ $EUID -ne 0 ]] && { command -v apt-get >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; }; then
  log "Warning: run as root (or with sudo) to fully remove system packages."
fi

# --- Terminate running server processes ---
if pgrep -f "server\.asr_server" >/dev/null 2>&1; then
  log "Stopping active Moonshine server processes"
  pkill -f "server\.asr_server" || true
fi
if pgrep -f "python .*server/asr_server.py" >/dev/null 2>&1; then
  log "Killing stray python processes"
  pkill -f "python .*server/asr_server.py" || true
fi

# --- Remove virtual environment ---
if [[ -d "$VENV_PATH" ]]; then
  log "Removing virtual environment at $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

# --- Uninstall system packages that install.sh recorded ---
if [[ -f "$PKG_LOG" ]]; then
  mapfile -t RECORDED_PKGS < <(grep -v '^\s*$' "$PKG_LOG" | sort -u)
  if [[ ${#RECORDED_PKGS[@]} -gt 0 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      log "Purging system packages: ${RECORDED_PKGS[*]}"
      DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge "${RECORDED_PKGS[@]}" || true
      DEBIAN_FRONTEND=noninteractive apt-get autoremove -y || true
      apt-get clean || true
    elif command -v yum >/dev/null 2>&1; then
      log "Removing system packages (yum): ${RECORDED_PKGS[*]}"
      yum remove -y "${RECORDED_PKGS[@]}" || true
      yum clean all || true
    elif command -v brew >/dev/null 2>&1; then
      for pkg in "${RECORDED_PKGS[@]}"; do
        log "Uninstalling brew package $pkg"
        brew uninstall --ignore-dependencies "$pkg" || true
      done
    else
      log "Warning: package manager not detected; cannot auto-remove installed system packages"
    fi
  fi
  rm -f "$PKG_LOG"
fi

# --- Clear repository caches & artifacts ---
log "Removing __pycache__ directories and bytecode"
find "$ROOT_DIR" -type f -name '*.pyc' -delete || true
find "$ROOT_DIR" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} + || true

# Remove repo-local caches directories if created
for path in \
  "$ROOT_DIR/.cache" \
  "$ROOT_DIR/.moonshine_cache" \
  "$ROOT_DIR/.huggingface" \
  "$ROOT_DIR/.hf_cache" \
  "$ROOT_DIR/logs" \
  "$ROOT_DIR/.pytest_cache"; do
  if [[ -e "$path" ]]; then
    log "Removing repo cache $path"
    rm -rf "$path"
  fi
done

# --- Clear user-level caches related to this project ---
CACHE_PATHS=(
  "$HOME/.cache/onnxruntime"
  "$HOME/.cache/pip"
  "$HOME/.cache/aiohttp"
  "$HOME/.cache/aioice"
  "$HOME/.cache/aiortc"
  "$HOME/.cache/pyav"
  "$HOME/.local/share/aiortc"
  "$HOME/.local/state/aiortc"
  "$HOME/.config/aiortc"
)
for cache_dir in "${CACHE_PATHS[@]}"; do
  if [[ -e "$cache_dir" ]]; then
    log "Removing cache $cache_dir"
    rm -rf "$cache_dir"
  fi
done

# --- Optional pip cache purge if available ---
if command -v pip >/dev/null 2>&1; then
  log "Clearing pip cache"
  pip cache purge >/dev/null 2>&1 || true
fi

log "Cleanup complete"

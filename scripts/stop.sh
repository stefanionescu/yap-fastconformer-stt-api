#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/.run.pid"
LOG_DIR="$ROOT_DIR/logs"
VENV_PATH="$ROOT_DIR/.venv"

log() {
  printf '[stop] %s\n' "$*"
}

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
    log "Killing pipeline PID $PID"
    kill "$PID" || true
    sleep 1
    if ps -p "$PID" >/dev/null 2>&1; then
      log "Force killing…"
      kill -9 "$PID" || true
    fi
  else
    log "No running pipeline for PID $PID"
  fi
  rm -f "$PID_FILE"
else
  log "No .run.pid found; attempting to stop uvicorn…"
  pkill -f "uvicorn server:app" || true
fi

if [[ -d "$LOG_DIR" ]]; then
  log "Removing logs under $LOG_DIR"
  rm -rf "$LOG_DIR"
fi

if [[ -d "$VENV_PATH" ]]; then
  log "Removing virtual environment $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

CACHE_ROOTS=(
  "$ROOT_DIR/.cache"
  "$ROOT_DIR/.cache/hf"
  "$ROOT_DIR/.huggingface"
  "$ROOT_DIR/.hf_cache"
)

if [[ -n "${HF_HOME:-}" ]]; then
  CACHE_ROOTS+=("$HF_HOME")
fi

CACHE_ROOTS+=(
  "$HOME/.cache/huggingface"
  "$HOME/.cache/hf"
  "$HOME/.cache/pip"
  "$HOME/.cache/torch"
  "$HOME/.cache/transformers"
  "$HOME/.cache/nemo"
  "$HOME/.huggingface"
  "$HOME/.hf_cache"
  "$HOME/.local/share/huggingface"
  "$HOME/.local/state/huggingface"
)

CACHE_PATHS=(
  "${CACHE_ROOTS[@]}"
)
for path in "${CACHE_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    log "Removing cache $path"
    rm -rf "$path"
  fi
done

log "Clearing __pycache__ directories"
find "$ROOT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

log "Clearing Python bytecode"
find "$ROOT_DIR" -type f -name '*.pyc' -delete 2>/dev/null || true

if command -v pip >/dev/null 2>&1; then
  log "Purging pip download cache"
  pip cache purge >/dev/null 2>&1 || true
fi

log "Removing stale PID file"
rm -f "$PID_FILE"

log "Stop complete"

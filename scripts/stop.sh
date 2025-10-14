#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/.run.pid"
LOG_DIR="$ROOT_DIR/logs"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
    echo "Killing pipeline PID $PID"
    kill "$PID" || true
    sleep 1
    if ps -p "$PID" >/dev/null 2>&1; then
      echo "Force killing…"
      kill -9 "$PID" || true
    fi
  else
    echo "No running pipeline for PID $PID"
  fi
  rm -f "$PID_FILE"
else
  echo "No .run.pid found; attempting to stop uvicorn…"
  pkill -f "uvicorn server:app" || true
fi

echo "Stopped. Logs remain under $LOG_DIR"

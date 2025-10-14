#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/moonshine.log}"
PIPELINE_SCRIPT="$ROOT_DIR/scripts/install_and_start.sh"
PID_FILE="$LOG_DIR/moonshine.pid"

mkdir -p "$LOG_DIR"

# Check for existing pipeline
if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${existing_pid}" ]] && ps -p "$existing_pid" >/dev/null 2>&1; then
    echo "Moonshine pipeline already running (pid=${existing_pid}). Logs: $LOG_FILE" >&2
    tail -n +1 -f "$LOG_FILE"
    exit 0
  else
    rm -f "$PID_FILE"
  fi
fi

nohup bash "$PIPELINE_SCRIPT" "$ROOT_DIR" "$LOG_FILE" >/dev/null 2>&1 &
PIPE_PID=$!

echo "$PIPE_PID" > "$PID_FILE"

echo "Started Moonshine pipeline (pid=$PIPE_PID)"
echo "Logs: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Tail command: tail -f $LOG_FILE"
echo "Press Ctrl+C to stop tailing; the pipeline keeps running in the background."

# Wait for log file to appear so tail doesn't exit immediately
timeout=30
while [[ ! -f "$LOG_FILE" && $timeout -gt 0 ]]; do
  sleep 1
  timeout=$((timeout - 1))
done

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file $LOG_FILE not created yet. Check process $PIPE_PID manually."
  exit 1
fi

trap 'echo "\nStopping log tail; pipeline is still running (pid=$PIPE_PID)."' INT TERM

tail -n +1 -f "$LOG_FILE"

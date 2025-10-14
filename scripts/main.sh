#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="$ROOT_DIR/.run.pid"

mkdir -p "$LOG_DIR"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "$existing_pid" ]] && ps -p "$existing_pid" >/dev/null 2>&1; then
    echo "Parakeet streaming pipeline already running (pid=$existing_pid). Logs: $LOG_DIR/latest.log" >&2
    echo "Press Ctrl+C to stop tailing; the pipeline keeps running."
    trap 'echo "\nStopping log tail; pipeline (pid=$existing_pid) continues running."' INT TERM
    tail -n +1 -f "$LOG_DIR/latest.log"
    exit 0
  else
    rm -f "$PID_FILE"
  fi
fi

CHAIN="set -euo pipefail
cd \"$ROOT_DIR\"
./scripts/steps/00_env_check.sh
./scripts/steps/01_system_deps.sh
./scripts/steps/02_python_env.sh
./scripts/steps/03_python_deps.sh
./scripts/steps/04_prefetch_model.sh
exec ./scripts/steps/05_run_server.sh
"

echo "[main] Launching detached setup. Log: $LOG_FILE"
nohup bash -c "$CHAIN" >>"$LOG_FILE" 2>&1 < /dev/null &
PIPE_PID=$!

echo "$PIPE_PID" > "$PID_FILE"

echo "[main] Detached. Tail live logs with: tail -f $LOG_DIR/latest.log"
echo "[main] Stop with: ./scripts/stop.sh"

timeout=30
while [[ ! -f "$LOG_FILE" && $timeout -gt 0 ]]; do
  sleep 1
  timeout=$((timeout - 1))
done

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file $LOG_FILE not created yet. Check process $PIPE_PID manually."
  exit 1
fi

trap 'echo "\nStopping log tail; pipeline (pid=$PIPE_PID) keeps running."' INT TERM

tail -n +1 -f "$LOG_FILE"

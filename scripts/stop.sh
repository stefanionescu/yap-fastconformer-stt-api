#!/usr/bin/env bash
set -euo pipefail

# Gracefully stop local server processes and purge caches while preserving the repo.
# Usage:
#   bash scripts/stop.sh                # stop server, clear caches
#   bash scripts/stop.sh --nuke-venv    # also remove .venv entirely

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

NUKE_VENV=0
if [[ "${1:-}" == "--nuke-venv" ]]; then
  NUKE_VENV=1
fi

echo "[stop] Stopping ASR server"
# Prefer pidfile if present
PID_FILE="${REPO_ROOT}/.run/asr_server.pid"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${PID}" ]]; then
    kill "${PID}" 2>/dev/null || true
    sleep 1
    rm -f "${PID_FILE}" || true
  fi
fi
# Fallback: try to stop any remaining server(s)
pkill -f "python .*server/asr_server.py" || true
sleep 1

echo "[stop] Clearing Hugging Face caches"
rm -rf "${HOME}/.cache/huggingface" || true
rm -rf "${HOME}/.cache/huggingface_hub" || true
rm -rf "${HOME}/.cache/torch/hub" || true

echo "[stop] Clearing pip caches"
pip cache purge || true
rm -rf "${HOME}/.cache/pip" || true

echo "[stop] Clearing torch caches"
rm -rf "${HOME}/.cache/torch" || true

echo "[stop] Removing virtual environment and pip deps"
rm -rf "${REPO_ROOT}/.venv" || true

if [[ ${NUKE_VENV} -eq 1 ]]; then
  echo "[stop] Also clearing system pip cache"
  python3 -m pip cache purge || true
fi

echo "[stop] Done"



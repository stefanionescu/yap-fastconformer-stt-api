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
# Also stop any live tail processes following the server log
pkill -f "tail -n \+1 -F .*asr_server.log" || true
sleep 1

# Remove runtime dirs and logs created by deploy/run
echo "[stop] Removing runtime dirs and logs"
rm -rf "${REPO_ROOT}/.run" || true
rm -rf "${REPO_ROOT}/logs" || true
rm -f  "${REPO_ROOT}/nohup.out" || true

echo "[stop] Clearing Hugging Face caches"
# Respect HF_HOME/HUGGINGFACE_HUB_CACHE if set, plus common locations
HF_BASE="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface}}"
rm -rf "${HF_BASE}" || true
rm -rf "${HOME}/.cache/huggingface" || true
rm -rf "${HOME}/.cache/huggingface_hub" || true
rm -rf "${HOME}/.cache/torch/hub" || true
# In some environments, caches live under the workspace root
WS_CACHE_DIR="$(dirname "${REPO_ROOT}")/.cache"
rm -rf "${WS_CACHE_DIR}/huggingface" || true
rm -rf "${WS_CACHE_DIR}/huggingface_hub" || true
rm -rf "${WS_CACHE_DIR}/datasets" || true

echo "[stop] Clearing pip caches"
pip cache purge || true
rm -rf "${HOME}/.cache/pip" || true

echo "[stop] Clearing torch caches"
rm -rf "${HOME}/.cache/torch" || true

echo "[stop] Removing virtual environment and pip deps"
rm -rf "${REPO_ROOT}/.venv" || true

# Additional heavy caches frequently created during runs
echo "[stop] Clearing additional caches (datasets/numba/wandb/pycache)"
rm -rf "${HOME}/.config/wandb" || true
rm -rf "${HOME}/.cache/wandb" || true
rm -rf "${HOME}/.cache/numba" || true
rm -rf "${HOME}/.cache/datasets" || true
# Remove repo-local caches
rm -rf "${REPO_ROOT}/.cache" || true
# Purge __pycache__ folders in repo
find "${REPO_ROOT}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

if [[ ${NUKE_VENV} -eq 1 ]]; then
  echo "[stop] Also clearing system pip cache"
  python3 -m pip cache purge || true
fi

echo "[stop] Done"



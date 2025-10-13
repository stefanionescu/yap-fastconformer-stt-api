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

echo "[stop] Stopping ASR server processes (python asr_server.py)"
# Try to stop running server(s)
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

if [[ ${NUKE_VENV} -eq 1 ]]; then
  echo "[stop] Removing virtual environment .venv/"
  rm -rf "${REPO_ROOT}/.venv" || true
fi

echo "[stop] Done"



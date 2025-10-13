#!/usr/bin/env bash
set -euo pipefail

# Load env + helpers
# shellcheck source=common/common.sh
source "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/common/common.sh"

activate_venv

SERVER_HOST="${ASR_HOST:-127.0.0.1}"
SERVER_PORT="${ASR_PORT:-8080}"
SERVER="${SERVER_HOST}:${SERVER_PORT}"

FILE_NAME="${1:-mid.wav}"
RTF_RAW="${2:-10}"

# Clamp RTF to [1,10]
RTF_NUM=$(printf '%s' "${RTF_RAW}" | awk '{print ($0+0)}')
if (( $(echo "${RTF_NUM} > 10" | bc -l) )); then
  echo "[warmup] ERROR: RTF must be <= 10" >&2
  exit 2
fi
if (( $(echo "${RTF_NUM} < 1" | bc -l) )); then RTF_NUM=1; fi
RTF="${RTF_NUM}"

echo "[warmup] server=${SERVER} file=${FILE_NAME} rtf=${RTF}"
python "${REPO_ROOT}/tests/warmup.py" --server "${SERVER}" --file "${FILE_NAME}" --rtf "${RTF}" --full-text || true



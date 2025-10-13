#!/usr/bin/env bash
set -euo pipefail

# Load env + helpers
# shellcheck source=common.sh
source "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/common.sh"

activate_venv

SERVER_HOST="${ASR_HOST:-127.0.0.1}"
SERVER_PORT="${ASR_PORT:-8080}"
SERVER="${SERVER_HOST}:${SERVER_PORT}"

FILE_NAME="${1:-mid.wav}"
RTF="${2:-1000}"

echo "[warmup] server=${SERVER} file=${FILE_NAME} rtf=${RTF}"
python "${REPO_ROOT}/tests/warmup.py" --server "${SERVER}" --file "${FILE_NAME}" --rtf "${RTF}" --full-text || true



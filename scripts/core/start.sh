#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
HOST="${ASR_HOST:-0.0.0.0}"
PORT="${ASR_PORT:-8000}"
PATH_ALIAS="${ASR_WEBRTC_PATH:-/webrtc}"

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

export ASR_HOST="$HOST"
export ASR_PORT="$PORT"
export ASR_WEBRTC_PATH="$PATH_ALIAS"

exec python -m server.asr_server

#!/usr/bin/env bash
set -euo pipefail

# Load env + helpers
# shellcheck source=common/common.sh
source "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/common/common.sh"

activate_venv

echo "[server] Starting ASR server on ws://${ASR_HOST}:${ASR_PORT}"
exec python "${REPO_ROOT}/server/asr_server.py"



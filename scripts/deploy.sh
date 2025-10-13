#!/usr/bin/env bash
set -euo pipefail

# Single-command deployment: setup venv, install deps, launch server in background, follow logs.
# Usage: bash scripts/deploy.sh [--warmup]

RUN_WARMUP=0
if [[ "${1:-}" == "--warmup" ]]; then
  RUN_WARMUP=1
fi

# Load env + helpers
# shellcheck source=deps/common.sh
source "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/deps/common.sh"

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
LOG_DIR="${REPO_ROOT}/logs"
RUN_DIR="${REPO_ROOT}/.run"
LOG_FILE="${LOG_DIR}/asr_server.log"
PID_FILE="${RUN_DIR}/asr_server.pid"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# Ensure venv exists
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  echo "[deploy] Creating virtualenv"
  bash "${REPO_ROOT}/scripts/deps/create_venv.sh"
fi

echo "[deploy] Installing/updating dependencies"
bash "${REPO_ROOT}/scripts/deps/install_deps.sh"

# Activate venv for current shell
activate_venv

# Detect existing server
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[deploy] Server already running (pid=${OLD_PID}) on ws://${ASR_HOST:-0.0.0.0}:${ASR_PORT:-8000}"
    echo "[deploy] Following logs at ${LOG_FILE} (Ctrl-C to stop following; server keeps running)"
    tail -n +1 -F "${LOG_FILE}"
    exit 0
  fi
fi

echo "[deploy] Starting ASR server in background on ws://${ASR_HOST:-0.0.0.0}:${ASR_PORT:-8000}"
nohup bash "${REPO_ROOT}/scripts/run/start.sh" >> "${LOG_FILE}" 2>&1 &
NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"
disown "${NEW_PID}" || true

echo "[deploy] pid=${NEW_PID}  log=${LOG_FILE}"

if [[ ${RUN_WARMUP} -eq 1 ]]; then
  echo "[deploy] Running warmup test..."
  sleep 2  # Give server a moment to start
  bash "${REPO_ROOT}/scripts/run/warmup.sh" mid.wav 10 || echo "[deploy] Warmup failed (server may still be starting)"
fi

echo "[deploy] Following logs (Ctrl-C to detach; server continues running)"
tail -n +1 -F "${LOG_FILE}"



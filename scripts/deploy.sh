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

# Establish logging as early as possible so all subsequent output is captured
# Paths derived from REPO_ROOT provided by common.sh
LOG_DIR="${REPO_ROOT}/logs"
RUN_DIR="${REPO_ROOT}/.run"
LOG_FILE="${LOG_DIR}/asr_server.log"
PID_FILE="${RUN_DIR}/asr_server.pid"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
# Ensure log file exists before tailing to avoid transient 'cannot open' noise
: > "${LOG_FILE}"

# Preserve original stdout/stderr for interactive tailing
exec 3>&1 4>&2
# Redirect all subsequent script output to the log file
exec >> "${LOG_FILE}" 2>&1

echo "[deploy] =============================================="
echo "[deploy] Starting deployment (logging to ${LOG_FILE})"
echo "[deploy] You can safely Ctrl-C to stop log following; server continues"
echo "[deploy] To follow logs later: tail -n +1 -F ${LOG_FILE}"
echo "[deploy] =============================================="

# Start tailing logs to the user's terminal from the very beginning
tail -n +1 -F "${LOG_FILE}" >&3 2>&4 &
TAIL_PID=$!
# Clean up tail when this script exits for any reason
trap 'kill -TERM ${TAIL_PID} 2>/dev/null || true' EXIT
# Allow Ctrl-C to stop only the tailing, leaving server running
trap 'echo "[deploy] Log following stopped; server continues. To follow later: tail -n +1 -F ${LOG_FILE}" >&3; kill -TERM ${TAIL_PID} 2>/dev/null || true; exit 0' INT

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Background deployment function
deploy_background() {
  # Self-initialize in the background shell to avoid relying on parent-shell state
  REPO_ROOT="${REPO_ROOT:-$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )}"
  LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
  RUN_DIR="${RUN_DIR:-${REPO_ROOT}/.run}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/asr_server.log}"
  PID_FILE="${PID_FILE:-${RUN_DIR}/asr_server.pid}"
  RUN_WARMUP="${RUN_WARMUP:-0}"

  mkdir -p "${LOG_DIR}" "${RUN_DIR}"

  # Load env + helpers (provides activate_venv and loads env.sh if present)
  # shellcheck source=deps/common.sh
  source "${REPO_ROOT}/scripts/deps/common.sh"

  echo "[deploy] $(date): Background worker started" >> "${LOG_FILE}"

  # Check HF_TOKEN is set
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[deploy] ERROR: HF_TOKEN environment variable is required for model download" >&2
    echo "[deploy] Set it with: export HF_TOKEN=your_token_here" >&2
    echo "[deploy] Get token from: https://huggingface.co/settings/tokens" >&2
    return 1
  fi

  # Ensure venv exists
  if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
    echo "[deploy] $(date): Creating virtualenv" >> "${LOG_FILE}"
    bash "${REPO_ROOT}/scripts/deps/create_venv.sh" >> "${LOG_FILE}" 2>&1
  fi

  # Install or update dependencies
  echo "[deploy] $(date): Installing/updating dependencies" >> "${LOG_FILE}"
  bash "${REPO_ROOT}/scripts/deps/install_deps.sh" >> "${LOG_FILE}" 2>&1

  # Activate venv
  activate_venv

  # Detect existing server
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}" || true)"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
      echo "[deploy] $(date): Server already running (pid=${OLD_PID}) on ws://${ASR_HOST:-0.0.0.0}:${ASR_PORT:-8000}" >> "${LOG_FILE}"
      return 0
    fi
  fi

  echo "[deploy] $(date): Starting ASR server on ws://${ASR_HOST:-0.0.0.0}:${ASR_PORT:-8000}" >> "${LOG_FILE}"

  # Start server and capture its PID
  bash "${REPO_ROOT}/scripts/run/start.sh" >> "${LOG_FILE}" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${PID_FILE}"

  echo "[deploy] $(date): Server started with PID=${SERVER_PID}" >> "${LOG_FILE}"

  if [[ ${RUN_WARMUP} -eq 1 ]]; then
    echo "[deploy] $(date): Waiting 3s for server startup..." >> "${LOG_FILE}"
    sleep 3
    echo "[deploy] $(date): Running warmup test..." >> "${LOG_FILE}"
    bash "${REPO_ROOT}/scripts/run/warmup.sh" mid.wav 10 >> "${LOG_FILE}" 2>&1 || echo "[deploy] $(date): Warmup failed" >> "${LOG_FILE}"
  fi

  # Keep the background process alive to maintain server
  wait "${SERVER_PID}"
}

# Run deployment in background (pass critical vars explicitly to the subshell)
nohup bash -c "RUN_WARMUP=${RUN_WARMUP} REPO_ROOT=${REPO_ROOT} LOG_DIR=${LOG_DIR} RUN_DIR=${RUN_DIR} LOG_FILE=${LOG_FILE} PID_FILE=${PID_FILE}; $(declare -f deploy_background); deploy_background" &
DEPLOY_PID=$!
disown "${DEPLOY_PID}" || true

echo "[deploy] Deployment process started (PID=${DEPLOY_PID})"
echo "[deploy] Following logs (Ctrl-C stops following only, server continues):"
echo ""

# Block here to continue following logs until user interrupts
wait "${TAIL_PID}"



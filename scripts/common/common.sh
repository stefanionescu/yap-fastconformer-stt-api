#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"

# shellcheck source=env.sh
if [[ -f "${REPO_ROOT}/scripts/common/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/scripts/common/env.sh"
fi

VENV_DIR="${REPO_ROOT}/.venv"
VENV_BIN="${VENV_DIR}/bin"

activate_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[common] Missing venv at ${VENV_DIR}. Run scripts/create_venv.sh first." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${VENV_BIN}/activate"
}



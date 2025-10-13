#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

python3 -m venv "${REPO_ROOT}/.venv"
echo "[venv] Created at ${REPO_ROOT}/.venv"
echo "[venv] To activate: source ${REPO_ROOT}/.venv/bin/activate"



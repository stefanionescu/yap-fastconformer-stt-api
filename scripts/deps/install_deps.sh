#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"

if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  echo "[install] Missing venv. Run scripts/common/create_venv.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${REPO_ROOT}/.venv/bin/activate"

pip install --upgrade pip
# Install Cython first (required for youtokentome build)
pip install "Cython>=0.29.0"
pip install -r "${REPO_ROOT}/requirements.txt"
echo "[install] Dependencies installed."



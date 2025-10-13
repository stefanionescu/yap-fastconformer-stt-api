#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"

if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  echo "[install] Missing venv. Run scripts/common/create_venv.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${REPO_ROOT}/.venv/bin/activate"

pip install --upgrade pip setuptools wheel build

# Ensure build tooling is present for native deps built from sdist
pip install "Cython>=0.29.0" "numpy==1.26.4"

# Preinstall youtokentome without build isolation so Cython is available at build-time
# This avoids pip's build isolation which breaks due to missing Cython in the isolated env
pip install --no-build-isolation --no-cache-dir "youtokentome==1.0.6"

# Install project requirements (youtokentome should already be satisfied)
pip install -r "${REPO_ROOT}/requirements.txt"
echo "[install] Dependencies installed."

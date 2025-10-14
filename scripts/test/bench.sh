#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

exec python "$ROOT_DIR/tests/bench.py" "$@"

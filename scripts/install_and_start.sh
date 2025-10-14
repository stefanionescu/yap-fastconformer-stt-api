#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <root_dir> <log_file>" >&2
  exit 1
fi

ROOT_DIR="$1"
LOG_FILE="$2"

mkdir -p "$(dirname "$LOG_FILE")"

exec >>"$LOG_FILE" 2>&1

printf '\n==== [%s] Moonshine install+start pipeline (pid=%s) ====\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$$"

bash "$ROOT_DIR/scripts/core/install.sh"

printf '\n==== [%s] Install complete — launching server ====\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

exec bash "$ROOT_DIR/scripts/core/start.sh"

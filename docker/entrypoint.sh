#!/usr/bin/env bash
set -euo pipefail

cd /app
exec ./scripts/steps/05_run_server.sh

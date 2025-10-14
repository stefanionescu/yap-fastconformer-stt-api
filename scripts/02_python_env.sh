#!/usr/bin/env bash
set -euo pipefail

echo "[02] Creating venv…"
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools

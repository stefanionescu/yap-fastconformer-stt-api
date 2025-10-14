#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate
export HF_HOME="${HF_HOME:-$(pwd)/.cache/hf}"
python scripts/steps/04_prefetch_model.py

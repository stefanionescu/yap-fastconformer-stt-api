#!/usr/bin/env bash
set -euo pipefail

echo "[01] Installing system dependencies…"
APT="apt-get"
SUDO=""
if [ "$EUID" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "ERROR: need root or sudo to install system deps" >&2
    exit 1
  fi
fi

$SUDO $APT update -y
$SUDO $APT install -y --no-install-recommends \
  python3.12 python3.12-venv python3-pip \
  ffmpeg libsndfile1 build-essential git ca-certificates

$SUDO update-ca-certificates || true

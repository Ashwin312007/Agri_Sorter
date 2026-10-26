#!/usr/bin/env bash
# find_cam.sh - USB Webcam Auto-Detection and Configuration Tool for Raspberry Pi 4
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/fruit-venv}"
if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/find_cam.py" "$@"

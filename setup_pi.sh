#!/usr/bin/env bash
# setup_pi.sh - idempotent setup for Raspberry Pi OS 64-bit (Bookworm), Python 3.11
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/fruit-venv}"
ONNXRUNTIME_VERSION="1.19.2"
NUMPY_VERSION="1.26.4"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "Expected aarch64 (64-bit OS), got $(uname -m). Flash the 64-bit Raspberry Pi OS."
  exit 1
fi

echo "[1/4] apt packages (OpenCV with GUI support + V4L tools)"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip python3-opencv python3-numpy v4l-utils

echo "[2/4] venv (inherits apt OpenCV)"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/4] pinned pip packages"
python -m pip install "numpy==${NUMPY_VERSION}" "onnxruntime==${ONNXRUNTIME_VERSION}"

echo "[4/4] camera permissions"
if ! id -nG "$USER" | grep -qw video; then
  sudo usermod -aG video "$USER"
  echo "Added $USER to the video group. Log out and back in (or reboot) once."
fi

python -c "import cv2, numpy, onnxruntime as ort; print('cv2', cv2.__version__, '| numpy', numpy.__version__, '| onnxruntime', ort.__version__)"
echo "Done. Activate with: source $VENV_DIR/bin/activate"

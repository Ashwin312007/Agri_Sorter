#!/usr/bin/env bash
# setup_pi.sh - Idempotent Setup Script for Raspberry Pi 4 (Debian Bookworm 64-bit)
# Sets up Python 3.11 virtualenv with system OpenCV, ONNX Runtime, V4L2 webcam, and GPIO
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/fruit-venv}"
ONNXRUNTIME_VERSION="1.19.2"
NUMPY_VERSION="1.26.4"

echo "==================================================================="
echo "  Setting up Raspberry Pi 4 for Multi-Fruit Damage Detection"
echo "==================================================================="

# 1. Architecture Check
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  echo "[ERROR] Expected aarch64 (64-bit ARM OS), but detected: $ARCH"
  echo "Please flash 64-bit Raspberry Pi OS (Debian Bookworm) to ensure ONNX compatibility."
  exit 1
fi
echo "[1/5] Verified 64-bit OS ($ARCH)"

# 2. System Packages
echo "[2/5] Updating and installing system packages..."
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-opencv \
  python3-numpy \
  python3-gpiozero \
  v4l-utils \
  libcamera-tools \
  libopenblas-dev

# 3. Virtual Environment Creation
echo "[3/5] Setting up isolated Python virtual environment at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 4. Pip Dependencies
echo "[4/5] Installing pinned edge inference packages..."
python3 -m pip install --upgrade pip
python3 -m pip install \
  "numpy==${NUMPY_VERSION}" \
  "onnxruntime==${ONNXRUNTIME_VERSION}" \
  "pillow>=10.0.0"

# 5. Device Permissions
echo "[5/5] Configuring camera & GPIO user permissions..."
for grp in video dialout gpio; do
  if getent group "$grp" > /dev/null 2>&1; then
    if ! id -nG "$USER" | grep -qw "$grp"; then
      sudo usermod -aG "$grp" "$USER"
      echo "  Added $USER to $grp group"
    fi
  fi
done

echo "-------------------------------------------------------------------"
echo "Verification check inside virtual environment:"
python3 -c "
import cv2, numpy as np, onnxruntime as ort
print('  OpenCV version     :', cv2.__version__)
print('  NumPy version      :', np.__version__)
print('  ONNX Runtime version:', ort.__version__, '(', ort.get_available_providers()[0], ')')
"
echo "-------------------------------------------------------------------"
echo "Setup successfully completed!"
echo "To run detection:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 check_env.py"
echo "  python3 detect.py --fruit apple --conf 0.20"
echo "  python3 web_detect.py --fruit pomegranate"
echo "==================================================================="

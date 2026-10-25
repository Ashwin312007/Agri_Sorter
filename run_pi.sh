#!/usr/bin/env bash
# run_pi.sh - Quick Launcher for Raspberry Pi Fruit Damage Sorter
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/fruit-venv}"
if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

FRUIT="${1:-menu}"
CONF="${2:-0.15}"
CAM="${3:-0}"
GPIO_PIN="${4:-17}"

if [ "$FRUIT" = "menu" ]; then
  echo "================================================="
  echo "  Raspberry Pi 4 Fruit Damage Sorter Launcher"
  echo "================================================="
  echo "Select target fruit:"
  echo "  1) Apple        (best.onnx)"
  echo "  2) Mango        (best_mango.onnx)"
  echo "  3) Guava        (best_guava.onnx)"
  echo "  4) Pomegranate  (best_pomegranate.onnx)"
  echo "  5) Live Web GUI (Multi-Fruit Dashboard :5000)"
  echo "  6) Run Pre-flight Environment Check"
  echo "================================================="
  read -rp "Enter choice [1-6]: " choice
  case "$choice" in
    1) FRUIT="apple"; CONF="0.20" ;;
    2) FRUIT="mango"; CONF="0.10" ;;
    3) FRUIT="guava"; CONF="0.15" ;;
    4) FRUIT="pomegranate"; CONF="0.15" ;;
    5)
      echo "Launching Web Dashboard on port 5000..."
      python3 web_detect.py --fruit pomegranate
      exit 0
      ;;
    6)
      python3 check_env.py
      exit 0
      ;;
    *)
      echo "Invalid selection. Defaulting to apple."
      FRUIT="apple"
      ;;
  esac
fi

echo "Starting detector for: $FRUIT (conf=$CONF, cam=$CAM, gpio=$GPIO_PIN)..."
python3 detect.py \
  --fruit "$FRUIT" \
  --conf "$CONF" \
  --cam "$CAM" \
  --gpio-pin "$GPIO_PIN" \
  --csv detections.csv \
  --headless

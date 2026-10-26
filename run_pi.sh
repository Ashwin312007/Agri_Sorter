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
CAM="${3:-auto}"
GPIO_PIN="${4:-17}"

# Read saved camera configuration if present
if [ -f "cam.conf" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    key=$(echo "$key" | tr -d '[:space:]')
    value=$(echo "$value" | tr -d '[:space:]"')
    case "$key" in
      CAM_INDEX) [ "$CAM" = "auto" ] && CAM="$value" ;;
      CAM_DEV)   [ "$CAM" = "auto" ] && CAM="$value" ;;
    esac
  done < cam.conf
fi

if [ "$FRUIT" = "menu" ]; then
  echo "================================================="
  echo "  Raspberry Pi 4 Fruit Damage Sorter Launcher"
  echo "================================================="
  echo "Current Webcam: $CAM"
  echo "Select action:"
  echo "  1) Apple Detector        (best.onnx)"
  echo "  2) Mango Detector        (best_mango.onnx)"
  echo "  3) Guava Detector        (best_guava.onnx)"
  echo "  4) Pomegranate Detector  (best_pomegranate.onnx)"
  echo "  5) Live Web GUI          (Multi-Fruit Dashboard :5000)"
  echo "  6) Run Pre-flight Checks (check_env.py)"
  echo "  7) Find / Configure Cam  (find_cam.sh)"
  echo "================================================="
  read -rp "Enter choice [1-7]: " choice
  case "$choice" in
    1) FRUIT="apple"; CONF="0.20" ;;
    2) FRUIT="mango"; CONF="0.10" ;;
    3) FRUIT="guava"; CONF="0.15" ;;
    4) FRUIT="pomegranate"; CONF="0.15" ;;
    5)
      echo "Launching Web Dashboard on port 5000 (Camera: $CAM)..."
      python3 web_detect.py --fruit pomegranate --cam "$CAM" --no-browser
      exit 0
      ;;
    6)
      python3 check_env.py --cam "$CAM"
      exit 0
      ;;
    7)
      ./find_cam.sh
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

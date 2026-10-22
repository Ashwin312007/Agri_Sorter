#!/usr/bin/env bash
# ==============================================================================
# Agri_Sorter Hailo Dataflow Compiler (DFC) Compilation Script
# Target: Raspberry Pi AI HAT (hailo8l)
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo " Starting Hailo Dataflow Compiler for Raspberry Pi AI HAT (hailo8l)"
echo "=================================================================="

ONNX_MODEL="/home/taiki/Agri_Sorter/models/detector/yolo11n.onnx"
ALLS_SCRIPT="/home/taiki/Agri_Sorter/models/detector/yolo11n_fruit.alls"
OUTPUT_HEF="/home/taiki/Agri_Sorter/models/detector/yolo11n_fruit.hef"
CALIB_COUNT=64

if ! command -v hailo &> /dev/null; then
    echo "[-] hailo CLI not found in PATH."
    echo "    Please activate your Hailo DFC virtualenv:"
    echo "    source /path/to/hailo_virtualenv/bin/activate"
    exit 1
fi

echo "[+] Step 1: Parsing ONNX model to Hailo Archive (HAR)..."
hailo parser onnx "$ONNX_MODEL" --hw-arch hailo8l

HAR_FILE="${ONNX_MODEL%.*}.har"

echo "[+] Step 2: Optimizing with INT8 Post-Training Quantization..."
hailo optimize "$HAR_FILE" --model-script "$ALLS_SCRIPT" --hw-arch hailo8l

echo "[+] Step 3: Compiling to Hailo Executable Format (HEF)..."
hailo compiler "$HAR_FILE" --hw-arch hailo8l --output "$OUTPUT_HEF"

echo "[+] Compilation finished successfully! Output: $OUTPUT_HEF"

# Edge Fruit Sorter

Raspberry Pi 4 starter system for apple grading.

## Architecture

- **Multi-fruit external vision**: YOLO apple detector plus a per-apple freshness classifier executed 100% offline on the Raspberry Pi 4 CPU using static ONNX models via OpenCV DNN (`cv2.dnn`) with ARM NEON vector acceleration (zero PyTorch / Ultralytics runtime bloat).
- **Internal quality**: XGBoost model trained from ultrasonic acoustic measurements.
- **Runtime**: Raspberry Pi 4 (BCM2711 Quad Cortex-A72), standard USB webcam (`/dev/video0`) via OpenCV (or optional Picamera2), ADS1115 I2C ADC, GPIO ultrasonic transmitter, encoder, and sorter ejection servo.

Model training and ONNX export happen on a workstation or Colab. The Raspberry Pi 4 performs 100% offline edge inference only.

## Dataset Reality

The included external training pipeline accepts a YOLO-format dataset. A useful public starting point is the [Apple Defects Dataset](https://github.com/cidis-vision/apple-defects), which contains visible images and masks for rot, bruise, and spot. Public data will not exactly match your camera, lighting, fruit variety, or conveyor view, so add your own images before trusting the sorter.

There is no ready-made dataset for your particular ultrasonic transmitter/receiver readings. Collect `data/internal/measurements.csv`, then cut open each measured apple and fill in `internal_label` (`0=healthy`, `1=damaged`).

## External Baseline Used in This Repository

The included public dataset is split into `Fresh` and `Rotten` apples, so the first trained model is a lightweight image classifier. It answers `FRESH` or `ROTTEN`; it does not locate a wound. Use the segmentation script only after adding a wound-mask dataset.

## Setup on a Training Workstation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt

# 1. Prepare data and train models
python3 scripts/prepare_apple_freshness_data.py --source "data/external/fresh_rotten_source/Fruit Freshness Dataset/Fruit Freshness Dataset/Apple"
python3 scripts/train_external_classifier.py
python3 scripts/train_internal.py --csv data/internal/measurements.csv

# 2. Export PyTorch models to static ONNX for Raspberry Pi 4 OpenCV DNN
python3 scripts/export_to_onnx.py
```

Copy the resulting lightweight models into `models/` on the Raspberry Pi:

```text
models/detector/yolo11n.onnx
models/external/best.onnx
models/internal/xgb_internal.joblib
```

## Raspberry Pi 4 Setup & Edge Execution

The Raspberry Pi 4 runs 100% offline edge inference with OpenCV DNN on CPU. A standard USB webcam (`/dev/video0`) is configured as the primary default camera backend.

```bash
# On Raspberry Pi OS:
sudo bash scripts/setup_rpi4.sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

### Zero-Argument Execution

Both the vision standalone app and the full multi-station conveyor pipeline auto-resolve the `.onnx` models and default to the USB webcam (`/dev/video0`):

```bash
# Run standalone multi-apple vision tracking
python3 app/pi_inference.py

# Run full conveyor pipeline (vision + ultrasonic + spatial tracking + sorter ejection)
python3 app/conveyor_pipeline.py
```

The detector gives every apple a temporary vision track ID. The encoder matches that ID to the ultrasonic fruit record before sorting. Simultaneous visible apples are supported; touching apples are not.

For headless environments (e.g. SSH sessions without an X11 `DISPLAY`), both applications automatically detect the absence of `DISPLAY` or accept an explicit `--headless` flag to bypass `cv2.imshow`.

If using an official Raspberry Pi Camera Module instead of a USB webcam, pass `--camera-backend picamera2` (or set `backend = "picamera2"` in `config/hardware.toml`).

## Model & Hardware Verification

Before running on live fruit, verify the offline model benchmarks and hardware configuration:

```bash
# Verify static ONNX models and OpenCV DNN inference
python3 scripts/verify_models.py

# Benchmark Raspberry Pi 4 CPU cores, memory bandwidth, and peripheral interfaces
python3 scripts/check_rpi4_hardware.py --mock
```

## Sensor CSV Format

Required columns:

```text
travel_time_us,received_amplitude,fruit_diameter_mm,weight_g,temperature_c,internal_label
```

Use at least several hundred labelled apples if possible, with healthy and damaged fruit from the same variety and storage conditions as the machine.

## GPIO & Wiring Defaults

Configured in `config/hardware.toml`:

```text
break_beam_gpio=17
servo_gpio=18
encoder_gpio=23
```

Use a level shifter for any 5 V sensor output, external motor/servo power, and a proper motor driver. Do not power a servo from a Pi GPIO pin.

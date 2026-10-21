# Edge Fruit Sorter

Raspberry Pi 4 starter system for apple grading.

## Architecture

- **Multi-fruit external vision**: Fine-tuned YOLO11n apple defect detector (`bruising`, `crack`, `insect_bite`, `rotten`, `russetting`) supporting single-pass defect classification and spatial tracking. Executed 100% offline on the Raspberry Pi 4 CPU using static ONNX models via OpenCV DNN (`cv2.dnn`) with ARM NEON vector acceleration (zero PyTorch / Ultralytics runtime bloat).
- **Dual-Stage Fallback**: Supports dedicated Stage 1 YOLO apple detector plus Stage 2 freshness classifier when using separate models.
- **Internal quality**: XGBoost model trained from ultrasonic acoustic measurements.
- **Runtime**: Raspberry Pi 4 (BCM2711 Quad Cortex-A72), standard USB webcam (`/dev/video0`) via OpenCV (or optional Picamera2), ADS1115 I2C ADC, GPIO ultrasonic transmitter, encoder, and sorter ejection servo.

Model training and ONNX export happen on a workstation or laptop. The Raspberry Pi 4 performs 100% offline edge inference only.

## Dataset Reality & Roboflow Integration

The sorter uses authentic YOLO-format datasets with normalized bounding boxes (`[class, cx, cy, w, h]`):

1. **Roboflow Universe Integration**:
   Download pre-annotated apple defect datasets directly via Roboflow API or direct URL:
   ```bash
   # Download via Roboflow API key:
   python3 scripts/download_yolo_dataset.py --roboflow-key "YOUR_API_KEY" --project "apple-defects-yafjq"

   # Or download a verified starter dataset (no API key needed):
   python3 scripts/download_yolo_dataset.py --sample
   ```

2. **Ultrasonic Firmness Dataset**:
   Collect `data/internal/measurements.csv` on your machine, then cut open each measured apple and fill in `internal_label` (`0=healthy`, `1=damaged`).

## Training & Edge Export on a Workstation

```bash
# 1. Activate development environment
pip install -r requirements-train.txt

# 2. Train YOLO11n on Apple Defects (Auto-exports to static ONNX with ARM NEON patching)
python3 scripts/train_yolo_defects.py --data data/external/apple_defects/data.yaml --epochs 25 --imgsz 416

# 3. (Optional) Train internal ultrasonic XGBoost model
python3 scripts/train_internal.py --csv data/internal/measurements.csv
```

The training script automatically generates:
- `models/detector/yolo11n.pt` (PyTorch weights)
- `models/detector/yolo11n.onnx` (Static 416x416 ONNX with OpenCV 4.6.0 ARM patch)
- `models/detector/yolo11n.json` (Class metadata: bruising, crack, insect_bite, rotten, russetting)

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

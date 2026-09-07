# Edge Fruit Sorter

Raspberry Pi 5 starter system for apple grading.

## Architecture

- Multi-fruit external vision: YOLO apple detector plus a per-apple freshness classifier
- Internal quality: XGBoost model trained from ultrasonic measurements
- Runtime: Raspberry Pi 5, Picamera2/OpenCV, GPIO break-beam, encoder, and servo

Training happens on a laptop or Colab. The Pi performs inference only.

## Dataset reality

The included external training pipeline accepts a YOLO-format dataset. A useful public starting point is the [Apple Defects Dataset](https://github.com/cidis-vision/apple-defects), which contains visible images and masks for rot, bruise, and spot. Public data will not exactly match your camera, lighting, fruit variety, or conveyor view, so add your own images before trusting the sorter.

There is no ready-made dataset for your particular ultrasonic transmitter/receiver readings. Collect `data/internal/measurements.csv`, then cut open each measured apple and fill in `internal_label` (`0=healthy`, `1=damaged`).

## External baseline used in this repository

The included public dataset is split into `Fresh` and `Rotten` apples, so the first trained model is a lightweight image classifier. It answers `FRESH` or `ROTTEN`; it does not locate a wound. Use the segmentation script only after adding a wound-mask dataset.

## Setup on a training computer

```bash
python -m venv .venv
pip install -r requirements-train.txt
python scripts/prepare_apple_freshness_data.py --source "data/external/fresh_rotten_source/Fruit Freshness Dataset/Fruit Freshness Dataset/Apple"
python scripts/train_external_classifier.py
python scripts/train_internal.py --csv data/internal/measurements.csv
```

Copy the resulting files into `models/` on the Pi:

```text
models/external/best.pt
models/internal/xgb_internal.joblib
```

## Raspberry Pi setup

```bash
python -m venv .venv
pip install -r requirements-pi.txt
python app/pi_inference.py --detector-model models/detector/yolo11n.pt --classifier-model models/external/best.pt
```

The detector gives every apple a temporary vision track ID. The encoder must match that ID to the ultrasonic fruit record before sorting. Simultaneous visible apples are supported; touching apples are not.

For the official Raspberry Pi Camera Module, use the default `--camera-backend auto` (or explicitly use `--camera-backend picamera2`). For a USB webcam, use `--camera-backend opencv --camera 0`.

## Sensor CSV format

Required columns:

```text
travel_time_us,received_amplitude,fruit_diameter_mm,weight_g,temperature_c,internal_label
```

Use at least several hundred labelled apples if possible, with healthy and damaged fruit from the same variety and storage conditions as the machine.

## GPIO defaults

Change these for your wiring:

```text
break_beam_gpio=17
servo_gpio=18
encoder_gpio=23
```

Use a level shifter for any 5 V sensor output, external motor/servo power, and a proper motor driver. Do not power a servo from a Pi GPIO pin.

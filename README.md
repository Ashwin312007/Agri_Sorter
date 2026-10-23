# Fruit Damage Detector: Raspberry Pi 4 (APPLE)

Target: Raspberry Pi 4 (4 GB or 8 GB), Raspberry Pi OS 64-bit Bookworm, Python 3.11, USB webcam on /dev/video0.

## Files
- `setup_pi.sh`: installs packages and the venv (idempotent)
- `check_env.py`: PASS/FAIL checks for board, OS, Python, camera, and model
- `detect.py`: live detection, FRESH/REJECT verdict, CSV log
- `best.onnx`: exported model, copied to the same folder

## Run

```bash
# 1. Setup (once). Log out/in if it adds you to the video group.
chmod +x setup_pi.sh && ./setup_pi.sh
source ~/fruit-venv/bin/activate

# 2. Pre-flight checks
python3 check_env.py --model best.onnx --cam 0

# 3. Live detection with a window (needs a desktop session)
python3 detect.py --model best.onnx --conf 0.25 --cam 0

# 4. Headless (SSH, no monitor): save REJECT frames, log to CSV
python3 detect.py --model best.onnx --conf 0.25 --headless --save --csv detections.csv
```

Press `q` in the window, or Ctrl+C, to stop.

## Notes
- `--conf`: replace `0.25` with the best threshold from the Phase 4 confidence sweep.
- Class names come from the ONNX metadata and must match `--expect` (default: `apple_fresh,apple_rotten,apple_bruised,apple_damaged`). A mismatch stops the program with a list of what is missing or extra. For another fruit, pass its four classes with `--expect`.
- Only `*_fresh` counts as FRESH. Every other class above `--conf` gives REJECT.
- Expected speed: an estimate of about 5-10 FPS (yolov5n, 416, CPU). Not measured yet. Measure on the Pi and update this line.
- `detections.csv` gets one row per detection, or one row per frame when nothing is found.

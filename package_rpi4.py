#!/usr/bin/env python3
"""Build self-contained Raspberry Pi 4 deployment bundle.

Gathers:
- Models: best.onnx, best_mango.onnx, best_guava.onnx, best_pomegranate.onnx
- Edge code: detect.py, web_detect.py, check_env.py
- Setup scripts: setup_pi.sh, run_pi.sh
- Service files: systemd/fruit_sorter.service
- Deployment docs: README_RPI4.md

Produces:
- Directory: deploy_rpi4/
- Archive: fruit_sorter_rpi4.zip
"""
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT / "deploy_rpi4"
ZIP_PATH = ROOT / "fruit_sorter_rpi4.zip"

ITEMS_TO_COPY = [
    # Models
    ("best.onnx", "best.onnx"),
    ("best_mango.onnx", "best_mango.onnx"),
    ("best_guava.onnx", "best_guava.onnx"),
    ("best_pomegranate.onnx", "best_pomegranate.onnx"),

    # Edge code
    ("detect.py", "detect.py"),
    ("web_detect.py", "web_detect.py"),
    ("check_env.py", "check_env.py"),

    # Setup scripts
    ("setup_pi.sh", "setup_pi.sh"),
    ("run_pi.sh", "run_pi.sh"),

    # Systemd
    ("systemd/fruit_sorter.service", "systemd/fruit_sorter.service"),
]


def build_package():
    print(f"[1/4] Preparing deployment directory at {DEPLOY_DIR}...")
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / "systemd").mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / "saved").mkdir(parents=True, exist_ok=True)

    print("[2/4] Copying deployment artifacts...")
    for src_rel, dst_rel in ITEMS_TO_COPY:
        src = ROOT / src_rel
        dst = DEPLOY_DIR / dst_rel
        if not src.exists():
            print(f"  [ERROR] Required file missing: {src}")
            return False
        shutil.copy2(src, dst)
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  Copied {src_rel:30s} -> {dst_rel} ({size_mb:.2f} MB)")

    # Write README_RPI4.md inside the bundle
    readme_path = DEPLOY_DIR / "README_RPI4.md"
    readme_content = """# Raspberry Pi 4 Fruit Damage Sorter - Deployment Guide

## 1. Hardware Requirements
- **Raspberry Pi 4 Model B** (4 GB or 8 GB RAM recommended)
- **Raspberry Pi OS 64-bit** (Debian Bookworm)
- **USB Webcam** (UVC-compliant, 640x480 @ 30 FPS connected to USB 3.0 /dev/video0)
- **MicroSD Card** (16 GB+ Class 10 / A1)
- **5V 3A USB-C Power Supply**
- *(Optional)* **5V Relay Module or 12V Solenoid** connected to **BCM GPIO Pin 17** for mechanical reject actuation.

---

## 2. Transfer Files to Raspberry Pi
Transfer the folder `deploy_rpi4/` or unzip `fruit_sorter_rpi4.zip` to `/home/pi/fruit_sorter/`:

```bash
# Example from host laptop using SCP:
scp -r deploy_rpi4/* pi@<raspberry-pi-ip>:/home/pi/fruit_sorter/
```

Or copy the folder onto a USB drive, plug it into the Pi, and copy to `~/fruit_sorter`.

---

## 3. One-Step Setup
SSH into your Raspberry Pi or open a terminal:

```bash
cd /home/pi/fruit_sorter
chmod +x setup_pi.sh run_pi.sh
./setup_pi.sh
```

This installs system OpenCV with hardware acceleration, configures Python 3.11 venv at `~/fruit-venv`, installs ONNX Runtime, and configures camera/video permissions.

---

## 4. Pre-Flight Verification Check
Run the comprehensive hardware & model validation check:

```bash
source ~/fruit-venv/bin/activate
python3 check_env.py
```

All 4 fruit models (Apple, Mango, Guava, Pomegranate), camera frame grab, and metadata will be tested and report `[PASS]`.

---

## 5. Running the Sorter

### Interactive Launcher:
```bash
./run_pi.sh
```
Presents a fast menu to choose fruit (Apple, Mango, Guava, Pomegranate, Web GUI, or check_env).

### CLI Direct Mode:
```bash
# Pomegranate detection with GPIO 17 reject actuation:
python3 detect.py --fruit pomegranate --conf 0.15 --gpio-pin 17 --headless

# Guava detection:
python3 detect.py --fruit guava --conf 0.15 --gpio-pin 17 --headless

# Mango detection:
python3 detect.py --fruit mango --conf 0.10 --gpio-pin 17 --headless

# Apple detection:
python3 detect.py --fruit apple --conf 0.20 --gpio-pin 17 --headless
```

### Live Dual-Stream Web GUI:
```bash
python3 web_detect.py --fruit pomegranate
```
Open `http://<raspberry-pi-ip>:5000` in any web browser on your phone, tablet, or laptop on the local network to view the live video stream, dynamic sensitivity slider, and switch fruits on the fly.

---

## 6. Autostart on Boot (Systemd Service)
To run the fruit sorter automatically as an appliance when the Raspberry Pi powers on:

```bash
sudo cp systemd/fruit_sorter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fruit_sorter.service
sudo systemctl start fruit_sorter.service

# View status / logs:
sudo systemctl status fruit_sorter.service
journalctl -u fruit_sorter.service -f
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    shutil.copy2(readme_path, ROOT / "README_RPI4.md")
    print(f"  Generated README_RPI4.md")

    # 3. Create Zip file
    print(f"\n[3/4] Packaging into zip archive: {ZIP_PATH}...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in DEPLOY_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(DEPLOY_DIR)
                zipf.write(file_path, arcname)

    zip_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  Created {ZIP_PATH.name} ({zip_mb:.2f} MB)")

    print("\n[4/4] Package verification:")
    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        namelist = zipf.namelist()
        print(f"  Zip contents ({len(namelist)} files):")
        for name in sorted(namelist):
            print(f"    - {name}")

    print("\n=== Raspberry Pi 4 Deployment Package Ready! ===")
    return True


if __name__ == "__main__":
    build_package()

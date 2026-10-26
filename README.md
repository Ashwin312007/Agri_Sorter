# Multi-Fruit Agricultural Damage Sorter (Raspberry Pi 4)

An autonomous real-time computer vision system built for **Raspberry Pi 4 (64-bit OS Bookworm)** to detect surface defects, rot, bruising, and disease across **Apple**, **Mango**, **Guava**, and **Pomegranate** using optimized **YOLOv5 ONNX** edge models with optional **GPIO pneumatic/relay reject actuation** and a **real-time dual-stream web dashboard**.

---

## Table of Contents
1. [Overview & Supported Fruits](#1-overview--supported-fruits)
2. [Project File Structure](#2-project-file-structure)
3. [Hardware Requirements & Pinout](#3-hardware-requirements--pinout)
4. [Step 1: Clone Repository & Switch Branch](#4-step-1-clone-repository--switch-branch)
5. [Step 2: One-Step Automated Setup (`setup_pi.sh`)](#5-step-2-one-step-automated-setup-setup_pish)
6. [Step 3: USB Webcam Auto-Discovery & Port Configuration (`find_cam.sh`)](#6-step-3-usb-webcam-auto-discovery--port-configuration-find_camsh)
7. [Step 4: Run Pre-Flight Environment Checks (`check_env.py`)](#7-step-4-run-pre-flight-environment-checks-check_envpy)
8. [Step 5: Running the Sorter](#8-step-5-running-the-sorter)
   - [Option A: Interactive Menu Launcher (`./run_pi.sh`)](#option-a-interactive-menu-launcher-run_pish)
   - [Option B: Headless Production CLI with GPIO Actuation (`detect.py`)](#option-b-headless-production-cli-with-gpio-actuation-detectpy)
   - [Option C: Live Dual-Stream Web Dashboard (`web_detect.py`)](#option-c-live-dual-stream-web-dashboard-web_detectpy)
9. [Step 6: Autostart on Boot (Systemd Appliance Mode)](#9-step-6-autostart-on-boot-systemd-appliance-mode)
10. [Performance Benchmarks & Conveyor Timing](#10-performance-benchmarks--conveyor-timing)
11. [Troubleshooting & FAQs](#11-troubleshooting--faqs)

---

## 1. Overview & Supported Fruits

The system classifies every incoming fruit into 4 unified categories and outputs a binary **`FRESH`** or **`REJECT`** verdict:
- **`FRESH`**: Fruit is clean, healthy, and passes inspection.
- **`REJECT`**: Defect detected above the calibrated threshold; triggers GPIO actuator.

| Fruit | Model File | Calibrated Confidence (`--conf`) | Unified Classes |
|---|---|---|---|
| 🍎 **Apple** | `best.onnx` | `0.20` | `apple_fresh`, `apple_rotten`, `apple_bruised`, `apple_damaged` |
| 🥭 **Mango** | `best_mango.onnx` | `0.10` | `mango_fresh`, `mango_rotten`, `mango_bruised`, `mango_damaged` |
| 🍈 **Guava** | `best_guava.onnx` | `0.15` | `guava_fresh`, `guava_rotten`, `guava_bruised`, `guava_damaged` |
| 🔴 **Pomegranate** | `best_pomegranate.onnx` | `0.15` | `pomegranate_fresh`, `pomegranate_rotten`, `pomegranate_bruised`, `pomegranate_damaged` |

---

## 2. Project File Structure

```text
Agri_Sorter/
├── best.onnx                  # Apple defect model (YOLOv5n ONNX, 416x416)
├── best_mango.onnx            # Mango defect model (YOLOv5n ONNX, 416x416)
├── best_guava.onnx            # Guava defect model (YOLOv5n ONNX, 416x416)
├── best_pomegranate.onnx      # Pomegranate defect model (YOLOv5n ONNX, 416x416)
├── find_cam.sh                # USB webcam discovery & config shell wrapper
├── find_cam.py                # V4L2 probe, frame grab validation & cam.conf generator
├── run_pi.sh                  # Interactive terminal launcher menu (all fruits + GUI)
├── setup_pi.sh                # Automated 1-step Pi environment installer
├── detect.py                  # Headless production detection & GPIO reject trigger
├── web_detect.py              # Dual-stream real-time browser dashboard (:5000)
├── check_env.py               # Pre-flight diagnostic & hardware check suite
├── package_rpi4.py            # Deployment bundler (generates deploy_rpi4 & zip)
├── systemd/
│   └── fruit_sorter.service   # Systemd service for 24/7 appliance autostart
└── README.md                  # Complete deployment and operating guide
```

---

## 3. Hardware Requirements & Pinout

### Required Hardware
- **Raspberry Pi 4 Model B** (2 GB, 4 GB, or 8 GB RAM)
- **Raspberry Pi OS 64-bit (Debian Bookworm)** on a 16 GB+ MicroSD card (A1/A2 rated)
- **Power Supply**: Official 5.1V / 3.0A USB-C power supply
- **USB Webcam**: Standard UVC-compliant USB webcam (Logitech C270/C920, OV5640, or generic 720p/1080p webcam running at 640x480 MJPG) connected to USB 3.0
- **Cooling**: Active fan or aluminum heatsink case (prevents thermal throttling under continuous inference)

### Optional Sorting Actuator (Relay / Solenoid / Air Flipper)
Connect a 5V single-channel relay module to drive a 12V pneumatic valve or solenoid reject flipper:

```
Raspberry Pi 4 Header             5V Relay Module
---------------------             ---------------
Pin 2  (5V Power)      -------->  VCC
Pin 6  (GND)           -------->  GND
Pin 11 (BCM GPIO 17)   -------->  IN (Signal)
```

---

## 4. Step 1: Clone Repository & Switch Branch

Open a terminal on your Raspberry Pi (or connect via SSH):

```bash
# Navigate to home directory
cd ~

# Clone the repository
git clone https://github.com/Ashwin312007/Agri_Sorter.git

# Enter the repository directory
cd Agri_Sorter

# Switch to the dedicated Raspberry Pi 4 branch
git checkout rpi4-fruit-detection
```

> **Note**: All 4 trained ONNX models (`best.onnx`, `best_mango.onnx`, `best_guava.onnx`, `best_pomegranate.onnx`) are already tracked and present in the repository root. No external downloads are required!

---

## 5. Step 2: One-Step Automated Setup (`setup_pi.sh`)

Run the idempotent setup script:

```bash
chmod +x setup_pi.sh run_pi.sh find_cam.sh
./setup_pi.sh
```

### What `setup_pi.sh` does automatically:
1. Verifies that the OS is 64-bit ARM (`aarch64`).
2. Installs system packages via APT: `python3-opencv` (hardware-accelerated V4L2 backend), `python3-numpy`, `python3-venv`, `v4l-utils`, and `python3-gpiozero`.
3. Creates an isolated Python 3.11 virtual environment at `~/fruit-venv` with `--system-site-packages`.
4. Installs pinned runtime dependencies: `onnxruntime==1.19.2`, `numpy==1.26.4`, and `pillow`.
5. Adds the current user to the `video` and `gpio` permission groups.

> **Important**: If the script added your user to the `video` group for the first time, log out and back in once (or run `sudo reboot`) so the permissions take effect.

---

## 6. Step 3: USB Webcam Auto-Discovery & Port Configuration (`find_cam.sh`)

Raspberry Pi 4 creates several internal hardware video nodes (`bcm2835-codec`, `bcm2835-isp`, etc.) in addition to your external USB webcam. Furthermore, UVC webcams generate two `/dev/video*` devices (one for video capture and one for metadata).

Use `find_cam.sh` to automatically scan, isolate your physical USB webcam, test frame grab, and save the active port into `cam.conf`:

```bash
# Run interactive camera discovery & selector:
./find_cam.sh
```

### Example Output:
```text
=================================================================
  Raspberry Pi 4 USB Webcam Configuration & Test Tool
=================================================================
Scanning available video devices...

Found 1 operational camera(s):
-----------------------------------------------------------------
 #  | Device         | USB   | Resolution   | Name
-----------------------------------------------------------------
  0 | /dev/video0    | YES   | 640x480 BGR  | USB 2.0 PC Camera [RECOMMENDED]
-----------------------------------------------------------------

Detected default recommended camera: Index 0 (USB 2.0 PC Camera)
Enter camera index to use [0]: 0
[OK] Saved camera configuration to cam.conf
```

### Useful Camera Commands:
```bash
# 1. Just list operational cameras without changing config:
./find_cam.sh --list

# 2. Automatically pick and save the first working USB webcam:
./find_cam.sh --auto

# 3. Explicitly set a specific camera index or device path:
./find_cam.sh --set 0
./find_cam.sh --set /dev/video2

# 4. Grab and save a test frame snapshot to verify video feed:
./find_cam.sh --test 0
```
> **Auto-Resolution**: Once saved to `cam.conf`, all scripts (`detect.py`, `web_detect.py`, `run_pi.sh`, and `fruit_sorter.service`) automatically use this configured camera without requiring manual port arguments!

---

## 7. Step 4: Run Pre-Flight Environment Checks (`check_env.py`)

Before starting the conveyor, verify all hardware interfaces and models:

```bash
# Activate virtual environment
source ~/fruit-venv/bin/activate

# Run pre-flight diagnostic suite
python3 check_env.py
```

### Expected Output:
```text
=================================================================
  Raspberry Pi 4 Fruit Damage Sorter - Pre-Flight Check
=================================================================
[PASS] Board Architecture - Raspberry Pi 4 Model B Rev 1.4
[PASS] 64-bit OS (aarch64/x86_64) - aarch64
[PASS] Python 3.11+ Runtime - v3.11.2
[PASS] Camera Device (/dev/video0) - /dev/video0
[PASS] v4l2-ctl utility - available
[PASS] Camera Stream & Frame Grab - 640x480 BGR
[PASS] ONNX Runtime Engine - v1.19.2 (CPUExecutionProvider)
[PASS] Model Load [APPLE] - best.onnx (shape [1, 3, 416, 416])
[PASS] Classes Match [APPLE] - ['apple_fresh', 'apple_rotten', 'apple_bruised', 'apple_damaged']
[PASS] Model Load [MANGO] - best_mango.onnx (shape [1, 3, 416, 416])
[PASS] Classes Match [MANGO] - ['mango_fresh', 'mango_rotten', 'mango_bruised', 'mango_damaged']
[PASS] Model Load [GUAVA] - best_guava.onnx (shape [1, 3, 416, 416])
[PASS] Classes Match [GUAVA] - ['guava_fresh', 'guava_rotten', 'guava_bruised', 'guava_damaged']
[PASS] Model Load [POMEGRANATE] - best_pomegranate.onnx (shape [1, 3, 416, 416])
[PASS] Classes Match [POMEGRANATE] - ['pomegranate_fresh', 'pomegranate_rotten', 'pomegranate_bruised', 'pomegranate_damaged']
[PASS] GPIO Hardware Actuation (gpiozero) - v2.0
=================================================================
Summary: 15/15 checks passed.
```

---

## 8. Step 5: Running the Sorter

You have 3 easy ways to run the sorter depending on your operational needs:

### Option A: Interactive Menu Launcher (`./run_pi.sh`)
```bash
./run_pi.sh
```
Presents a simple interactive terminal menu with auto-camera detection:
```text
=================================================
  Raspberry Pi 4 Fruit Damage Sorter Launcher
=================================================
Current Webcam: 0
Select action:
  1) Apple Detector        (best.onnx)
  2) Mango Detector        (best_mango.onnx)
  3) Guava Detector        (best_guava.onnx)
  4) Pomegranate Detector  (best_pomegranate.onnx)
  5) Live Web GUI          (Multi-Fruit Dashboard :5000)
  6) Run Pre-flight Checks (check_env.py)
  7) Find / Configure Cam  (find_cam.sh)
=================================================
Enter choice [1-7]:
```

---

### Option B: Headless Production CLI with GPIO Actuation (`detect.py`)
Run headless over SSH without needing a monitor attached to the Pi:

```bash
# Pomegranate sorting with GPIO 17 reject actuation (uses cam.conf or auto-probed webcam):
python3 detect.py --fruit pomegranate --conf 0.15 --gpio-pin 17 --headless

# Guava sorting:
python3 detect.py --fruit guava --conf 0.15 --gpio-pin 17 --headless

# Mango sorting:
python3 detect.py --fruit mango --conf 0.10 --gpio-pin 17 --headless

# Apple sorting:
python3 detect.py --fruit apple --conf 0.20 --gpio-pin 17 --headless
```

#### CLI Flags Reference:
| Flag | Default | Description |
|---|---|---|
| `--fruit` | `auto` | Target fruit: `apple`, `mango`, `guava`, `pomegranate`, or `auto` |
| `--model` | `best.onnx` | Custom path to ONNX model (auto-resolved if `--fruit` is set) |
| `--conf` | `0.25` | Confidence threshold for defect triggers (see table above) |
| `--cam` | `auto` | Camera index (e.g. `0`), path (`/dev/video0`), or `auto` (reads `cam.conf` / probes) |
| `--gpio-pin` | `None` | BCM GPIO pin number for reject relay/solenoid trigger |
| `--gpio-pulse`| `0.2` | Actuation pulse duration in seconds |
| `--headless` | `False` | Run without GUI window (ideal for SSH) |
| `--save` | `False` | Save annotated JPEG snapshots of rejected fruits into `saved/` |
| `--csv` | `detections.csv`| CSV telemetry log file |

---

### Option C: Live Dual-Stream Web Dashboard (`web_detect.py`)
Run the web server directly on the Raspberry Pi:

```bash
# Starts dashboard (auto-detects webcam and avoids opening browser in headless mode):
python3 web_detect.py --fruit pomegranate --no-browser
```

1. Open any web browser on your phone, tablet, or laptop connected to the same Wi-Fi / local network:
   ```
   http://<raspberry-pi-ip>:5000
   ```
2. **Dual Stream Display**:
   - Left Card: **[1] Raw Camera View** (real-time 640x480 webcam input).
   - Right Card: **[2] YOLO Vision Inference** (bounding boxes, class labels, and `FRESH` / `REJECT` verdict banner).
3. **Live Controls**:
   - **Fruit Switcher Buttons**: Toggle instantly between `🍎 APPLE`, `🥭 MANGO`, `🍈 GUAVA`, and `🔴 POMEGRANATE` on the fly without restarting the process.
   - **Detection Sensitivity Slider**: Adjust the confidence threshold dynamically in real time.

---

## 9. Step 6: Autostart on Boot (Systemd Appliance Mode)

To make your Raspberry Pi boot up directly as an automated sorting appliance:

```bash
# 1. Copy the systemd service file
sudo cp systemd/fruit_sorter.service /etc/systemd/system/

# 2. Reload systemd daemon
sudo systemctl daemon-reload

# 3. Enable and start the service
sudo systemctl enable fruit_sorter.service
sudo systemctl start fruit_sorter.service
```

### Useful Service Commands:
```bash
# Check service status:
sudo systemctl status fruit_sorter.service

# View live real-time detection logs:
journalctl -u fruit_sorter.service -f

# Stop the sorter service:
sudo systemctl stop fruit_sorter.service

# Restart the sorter service:
sudo systemctl restart fruit_sorter.service
```

---

## 10. Performance Benchmarks & Conveyor Timing

### Benchmarked Throughput on Raspberry Pi 4 (Quad-Core Cortex-A72 @ 1.5GHz)
- **Inference Latency**: **70 – 75 ms** per frame (using 4 CPU threads and ARM NEON SIMD acceleration).
- **End-to-End Throughput**: **12 – 14 FPS** (including V4L2 MJPG decoding, letterbox resizing, inference, NMS, and GPIO trigger).
- **RAM Footprint**: **~85 MB** (leaves >95% of Raspberry Pi RAM free for OS and networking).

### Conveyor Belt Actuation Timing Formula:
To ensure the pneumatic flipper pushes the rejected fruit exactly when it arrives at the reject chute:

$$\text{Actuation Delay (ms)} = \left(\frac{\text{Distance from Camera to Flipper (cm)}}{\text{Conveyor Speed (cm/s)}} \times 1000\right) - 75\text{ ms (latency)}$$

*Example*: If camera is 30 cm before the reject arm and belt moves at 30 cm/s:
$$\text{Delay} = \left(\frac{30}{30} \times 1000\right) - 75 = 925\text{ ms}$$

---

## 11. Troubleshooting & FAQs

### Q1: `Camera index 0 failed to grab` or `Camera 0 not available`
- Run `./find_cam.sh` to automatically scan all `/dev/video*` ports and isolate your USB webcam from the Pi's internal ISP/codec nodes.
- Save the active camera index: `./find_cam.sh --auto` or `./find_cam.sh --set <index>`.
- Ensure the USB webcam is plugged into a blue **USB 3.0** port.
- Check that the device exists: `ls -l /dev/video*`.
- Check device capabilities: `v4l2-ctl --list-devices`.
- If permission denied: `sudo usermod -aG video $USER`, then log out and back in.

### Q2: Frame rate drops or Pi becomes very hot
- Sustained 4-core inference requires active cooling. Fit an inexpensive 5V fan or aluminum armor case to prevent thermal throttling from 1.5GHz down to 1.0GHz.
- Run `vcgencmd measure_temp` to verify that CPU temperature stays below 65°C.

### Q3: How to test without a physical camera or on sample photos?
You can run `detect.py` directly on single photos or directories:
```bash
python3 detect.py --fruit pomegranate --source test_fruit.jpg --conf 0.15
```

### Q4: Relay triggers in reverse (stays ON and turns OFF on defect)?
Some relay modules are **active-low**. In `detect.py`, change `active_high=True` to `active_high=False` in `OutputDevice(pin, active_high=False)`.

---

## License
MIT License. Developed for automated agricultural sorting applications.

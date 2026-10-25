# Raspberry Pi 4 Fruit Damage Sorter - Deployment Guide

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

# Raspberry Pi 5 Live Deployment & Production Runbook

**System:** Agri_Sorter Automated Fruit Inspection & Conveyor Sorter  
**Target Hardware:** Raspberry Pi 5 (RP1 Southbridge, Broadcom BCM2712, 4GB/8GB)  
**Primary Interfaces:** CSI-2 (Picamera2 / Camera Module 3), I2C1 (ADS1115 ADC @ 400kHz), RP1 GPIO (Pin 18 40kHz Ultrasonic Transmitter Burst)  
**Target Event:** Tuesday Live Sorting Demonstration  

---

## 1. Quick Start (3 Commands)

Execute these three commands directly from the terminal on the Raspberry Pi 5 to set up dependencies, run pre-flight diagnostics, and launch the real-time sorting pipeline:

```bash
# 1. Clone repository and navigate to project root
git clone https://github.com/taiki-repo/Agri_Sorter.git && cd Agri_Sorter

# 2. Initialize virtualenv with Debian Bookworm system-site-packages and install dependencies
python3 -m venv --system-site-packages venv && source venv/bin/activate && pip install -r requirements-pi.txt

# 3. Launch live sorting pipeline in headless production mode at recommended conveyor speed (0.12 m/s)
python -m app.conveyor_pipeline --speed 0.12 --headless
```

*(Note: For local development laptops or CI benches without physical sensors attached, see [Section 6](#6-production-running-commands) for `--mock` and `--feed-mock-tracks` flags.)*

---

## 2. Hardware Wiring Matrix & Pinout

The Raspberry Pi 5 interfaces with two separate analog/digital subsystems:
1. **Transmitter Drive**: Direct hardware burst from RP1 GPIO line 18 (Pin 12) through driver stage to the 40 kHz acoustic transducer.
2. **Receiver Conditioning & Digitisation**: Analog echo envelope demodulator (LM741 / BAT54S clamping) sampled via Texas Instruments ADS1115 16-bit ADC over I2C bus 1 at address `0x48`.

### 2.1 Raspberry Pi 5 40-Pin Header Pinout Mapping

```text
                           Raspberry Pi 5 Header
                                3.3V Power [01] [02] 5V Power
             (ADS1115 SDA) BCM 2 (I2C1 SDA) [03] [04] 5V Power
             (ADS1115 SCL) BCM 3 (I2C1 SCL) [05] [06] GND (Star Ground)
                           BCM 4 (GPCLK0)  [07] [08] BCM 14 (TXD)
                             Star GND Ref  [09] [10] BCM 15 (RXD)
                           BCM 17          [11] [12] BCM 18 (TX 40kHz Burst)
                           BCM 27          [13] [14] GND
                           BCM 22          [15] [16] BCM 23
                                3.3V Power [17] [18] BCM 24
                           BCM 10 (MOSI)   [19] [20] GND
                           BCM 9 (MISO)    [21] [22] BCM 25
                           BCM 11 (SCLK)   [23] [24] BCM 8 (CE0)
                                     GND   [25] [26] BCM 7 (CE1)
                                      ...   ...  ...
```

### 2.2 Point-to-Point Wiring Interconnect Table

| Net Name | From (Source Pin) | To (Destination Pin) | Signal Function / Voltage Level |
| :--- | :--- | :--- | :--- |
| **I2C_SDA** | Pi 5 Pin 3 (BCM 2) | ADS1115 Pin SDA | I2C Bus 1 Serial Data (3.3V Logic, 400 kHz) |
| **I2C_SCL** | Pi 5 Pin 5 (BCM 3) | ADS1115 Pin SCL | I2C Bus 1 Serial Clock (3.3V Logic, 400 kHz) |
| **VDD_3V3** | Pi 5 Pin 1 (3.3V DC) | ADS1115 Pin VDD | Clean 3.3V logic & ADC reference rail |
| **ADDR_GND** | ADS1115 Pin ADDR | ADS1115 Pin GND | Hardwires I2C slave address to `0x48` |
| **A0_INPUT** | Clamping Diode Node | ADS1115 Pin A0 (AIN0)| Conditioned fruit acoustic envelope (0.0V - 3.3V) |
| **TX_DRIVE** | Pi 5 Pin 12 (BCM 18) | Transducer TX Driver IN | 8-cycle 40 kHz square wave burst (3.3V logic) |
| **STAR_GND** | Pi 5 Pin 6 or Pin 9 | Central Bus Star Point | Common star ground reference (Pi, PSU 0V, ADC, TX/RX shields) |
| **AFE_V+** | External PSU +12V | LM741 Pin 7 ($V_{+}$) | Positive op-amp rail for acoustic amplifier |
| **AFE_V-** | External PSU −12V | LM741 Pin 4 ($V_{-}$) | Negative op-amp rail for acoustic amplifier |
| **AFE_0V** | External PSU 0V/GND | Central Bus Star Point | Op-amp reference return |

### 2.3 Common Star Grounding Architecture

> [!CAUTION]
> **Ground Loop Prevention**: The high-gain acoustic analog front-end (AFE) detects microvolt-level signals from the piezo receiver. If the external $\pm 12\text{V}$ power supply ground, Raspberry Pi logic ground, and ADS1115 ground are connected in a daisy chain, switching return currents from the Raspberry Pi 5 RP1 processor and camera CSI bus will inject $50\text{ Hz} / 100\text{ Hz}$ and switching spikes into the envelope line.
>
> **Mandatory Rule**: All ground leads must meet at a single physical brass binding post or central terminal strip ("Star Ground"). The shield braid of the receiver transducer coaxial cable must terminate at this star point only.

```text
    Pi 5 Pin 6 (Logic GND) ────┐
    ADS1115 GND Pin        ────┤
    PSU 0V Rail (Dual Rail)────┼─── [ CENTRAL STAR GROUND POINT ]
    Transducer RX Shield   ────┤
    Transducer TX Return   ────┘
```

---

## 3. Sensor Physical Layout & Mounting

The physical sorting bench consists of three discrete stations arranged sequentially along the linear conveyor belt:

```text
[ Conveyor Feed ] ===>  [ Station 0: Camera ]  ===>  [ Station 1: Ultrasonic ]  ===>  [ Station 2: Ejector ]
                        X = 0.00 m                   X = 0.30 m                      X = 0.55 m
                        (YOLO11 Detection)           (40 kHz Acoustic Probe)         (Pneumatic/Servo Gate)
```

### 3.1 Transducer Mounting Height Calculation & Blind Zone Margin

Piezoelectric ceramic transducers (such as the GU16D01L01-A27 40 kHz unit) experience mechanical resonance and electrical ringing following excitation. The transducer cannot distinguish incoming fruit echoes during this decay period.

* **Measured Ring-Down Period ($t_{\text{blind}}$)**: $\le 1.20\text{ ms}$ ($1200\ \mu\text{s}$).
* **Speed of Sound in Air ($c$ at $20^\circ\text{C}$)**: $343\text{ m/s} = 0.343\text{ mm/\mu s}$.
* **Round-Trip Distance Traveled in $1.20\text{ ms}$**:
  $$d_{\text{two-way}} = c \times t_{\text{blind}} = 343\text{ m/s} \times 0.0012\text{ s} = 0.4116\text{ m} = 411.6\text{ mm}$$
* **Minimum One-Way Clearance to Specimen Top**:
  $$h_{\text{clearance}} \ge \frac{d_{\text{two-way}}}{2} = \frac{411.6\text{ mm}}{2} = 205.8\text{ mm}$$

To accommodate standard commercial apples (calibre diameter $70\text{ mm} - 85\text{ mm}$ resting on the belt) and provide mechanical safety margin against fruit jams:

$$\text{Transducer Face Height above Belt Surface } (H) \ge 240\text{ mm}$$

At $H = 240\text{ mm}$ and apple crown height $= 75\text{ mm}$, the air gap is $165\text{ mm}$.  
Round-trip acoustic travel time:
$$t_{\text{arrival}} = \frac{2 \times 165\text{ mm}}{0.343\text{ mm/\mu s}} \approx 962\ \mu\text{s} + \Delta t_{\text{internal}} \approx 1350\ \mu\text{s} - 1450\ \mu\text{s}$$
This places the echo envelope comfortably past the $1.20\text{ ms}$ ring-down boundary, preventing false triggers.

### 3.2 Conveyor Belt Speed Limit & Spatial Sampling Density

The ADS1115 ADC has a maximum conversion rate of **860 samples per second** (SPS), meaning consecutive samples are spaced by:
$$\Delta t_{\text{sample}} = \frac{1}{860\text{ Hz}} \approx 1.163\text{ ms} = 1163\ \mu\text{s}$$

The critical acoustic evaluation occurs across the fruit apex crown, defined in configuration as an **apex sampling window of $35.0\text{ mm}$**.

* **Hard Maximum Safe Speed ($v_{\text{max}}$)**: **$0.20\text{ m/s}$**  
  At $0.20\text{ m/s}$, the fruit traverses the $35\text{ mm}$ window in:
  $$t_{\text{window}} = \frac{0.035\text{ m}}{0.20\text{ m/s}} = 175\text{ ms}$$
  Number of acquired ADC acoustic samples:
  $$N_{\text{samples}} = 0.175\text{ s} \times 860\text{ SPS} \approx 150\text{ samples}$$
  Speeds exceeding $0.20\text{ m/s}$ cause severe spatial smearing and violate tracking latency safety guards.
* **Recommended Demonstration Speed ($v_{\text{rec}}$)**: **$0.12\text{ m/s}$**  
  At $0.12\text{ m/s}$, the window duration is $292\text{ ms}$, providing $\approx 251$ samples for high-fidelity envelope peak detection and sub-sample interpolation.

---

## 4. Critical Pi 5 Hardware Gotchas

The Raspberry Pi 5 introduces substantial architectural changes over the Raspberry Pi 4 that must be observed during deployment:

### 4.1 22-Pin MIPI CSI-2 Camera Cable Requirement
* **The Pitfall**: The Raspberry Pi 5 has two 22-pin, 0.5 mm-pitch MIPI connectors (labeled `CAM/DISP 0` and `CAM/DISP 1`). The legacy 15-pin 1.0 mm-pitch ribbon cable from Raspberry Pi 3/4 **does not fit**.
* **The Action**: Use the official 22-pin-to-15-pin Raspberry Pi camera adapter cable or native 22-pin Camera Module 3 cable.
* **Orientation**: The cable contacts must face **inward towards the USB/Ethernet ports** (metal pads facing the board's circuit traces; plastic stiffener facing the connector locking clip).

### 4.2 400 kHz Fast-Mode I2C Bus Configuration
* **The Pitfall**: Standard Raspberry Pi OS defaults the I2C bus to 100 kHz standard mode. Reading the ADS1115 conversion registers and continuously polling flags at 100 kHz saturates the bus and causes scheduling delays.
* **The Action**: On Debian Bookworm on Pi 5, configuration settings live in `/boot/firmware/config.txt` (not `/boot/config.txt`).  
  Ensure the following lines exist:
  ```ini
  # /boot/firmware/config.txt
  dtparam=i2c_arm=on
  dtparam=i2c_arm_baudrate=400000
  camera_auto_detect=1
  ```
  After editing, reboot: `sudo reboot`.

### 4.3 Debian Bookworm PEP 668 & System Packages Policy
* **The Pitfall**: Debian 12 (Bookworm) marks the system Python environment as `EXTERNALLY-MANAGED` under PEP 668. Running `sudo pip install ...` will fail with an error. Furthermore, hardware libraries like `picamera2` and `libcamera` depend on compiled C++ APT system packages (`python3-picamera2`, `python3-gpiod`).
* **The Action**: Always create a virtual environment with the `--system-site-packages` flag:
  ```bash
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -r requirements-pi.txt
  ```
  This allows the virtualenv to access system-installed `picamera2` and `gpiod` while managing pip-installed ML packages (`ultralytics`, `xgboost`, `adafruit-circuitpython-ads1x15`) cleanly.

---

## 5. Pre-Flight Hardware Verification

Before turning on conveyor motors or running live fruit, run the built-in diagnostic utility:

```bash
# On live Raspberry Pi 5:
python scripts/check_rpi5_hardware.py

# On developer laptop / bench simulation:
python scripts/check_rpi5_hardware.py --mock
```

### Expected Output Dashboard

When all subsystems are operational, the script produces the following verification table:

```text
======================================================================================
                    Agri_Sorter RPi5 Hardware Diagnostic Dashboard                    
======================================================================================
 Execution Mode : REAL HARDWARE (Raspberry Pi 5)
 Timestamp      : 2026-10-20 09:15:00
 Repository     : /home/taiki/Documents/Agri_Sorter
--------------------------------------------------------------------------------------
 #  | Subsystem              | Status | Details                                       
--------------------------------------------------------------------------------------
 1  | System Platform        | [PASS] | Raspberry Pi 5 Model B Rev 1.0 (aarch64)      
 2  | Camera Backend         | [PASS] | Picamera2 captured 1280x720 frame             
 3  | I2C ADS1115 ADC        | [PASS] | ADS1115 on /dev/i2c-1 (0x48): raw=12450...   
 4  | RP1 GPIO Transmitter   | [PASS] | RP1 /dev/gpiochip4 pin 18 triggered 40kHz ... 
 5  | AI Models (YOLO11)     | [PASS] | Det: 15.2ms (yolo11n) | Cls: 3.0ms (best.pt)  
--------------------------------------------------------------------------------------
 OVERALL STATUS: ALL CHECKS PASSED (5/5)
======================================================================================
```

If any check returns `[FAIL]`, consult the [Troubleshooting](#7-troubleshooting--faq) section below before starting the conveyor.

---

## 6. Production Running Commands

All production scripts reside in the top-level `app` package. Always run them from the repository root with the virtual environment activated (`source venv/bin/activate`).

### 6.1 Live Production Sorting (Camera + Ultrasonic + Ejector)
Runs live Picamera2 video feed, YOLO11 detection and defect classification, ultrasonic acoustic burst interrogation, and automated ejection grading:
```bash
python -m app.conveyor_pipeline --speed 0.12 --headless --csv-out runtime/live_run.csv
```
* Key flags:
  * `--speed 0.12`: Sets belt velocity to the calibrated demonstration speed ($0.12\text{ m/s}$).
  * `--headless`: Disables OpenCV GUI display windows for robust operation via SSH or Wayland terminals.
  * `--csv-out <path>`: Exports complete tracking logs with per-fruit ToF, amplitude, visual labels, and final quality grades.

### 6.2 Bench Bring-Up (Real Sensors with Synthetic Fruit Tracks)
Use this mode when the physical sensors and Raspberry Pi are wired on the bench, but the physical conveyor belt is stationary or fruit has not yet arrived. It feeds synthetic tracking tracks into the pipeline while firing real ultrasonic bursts and reading the ADS1115 ADC:
```bash
python -m app.conveyor_pipeline --feed-mock-tracks --headless
```

### 6.3 Full Software Emulation (Laptop Verification)
Executes the full spatial conveyor coordinator using synthetic camera images and simulated ADC responses. Ideal for algorithm verification without hardware:
```bash
python -m app.conveyor_pipeline --mock --duration 5 --csv-out runtime/mock_summary.csv
```

---

## 7. Troubleshooting & FAQ

### Issue 1: `Permission denied: '/dev/gpiochip4'`
* **Cause**: On the Raspberry Pi 5, the 40-pin GPIO lines are controlled by the RP1 southbridge PCIe chip, mapped as `/dev/gpiochip4`. If the user is not in the `gpio` group or udev rules are missing, accessing the character device fails.
* **Resolution**:
  ```bash
  sudo usermod -aG gpio,i2c,video $USER
  sudo chmod 660 /dev/gpiochip4
  ```
  Log out and log back in (or reboot) for group permissions to take effect.

### Issue 2: `Camera not detected` or `No cameras available`
* **Cause 1**: The 22-pin ribbon cable is loose, inserted backwards, or inserted into the wrong port.
* **Cause 2**: Camera autodetection disabled in firmware.
* **Resolution**:
  1. Inspect physical cable: Ensure the contacts face toward the USB/Ethernet jacks and the retention collar is locked flat.
  2. Verify system detection using the native libcamera CLI:
     ```bash
     rpicam-hello --list-cameras
     ```
     Should output: `0 : imx708 [4608x2592] (/base/axi/pcie2_brcm/.../imx708@1a)`
  3. Ensure `camera_auto_detect=1` is present in `/boot/firmware/config.txt`.

### Issue 3: `I2C address 0x48 missing` on `i2cdetect -y 1`
* **Cause 1**: Missing pull-up resistors or loose 3.3V/GND connection on the ADS1115 breakout.
* **Cause 2**: The `ADDR` pin on the ADS1115 is left floating or connected to 3.3V (which shifts the address to `0x49`).
* **Resolution**:
  1. Verify with I2C scanner:
     ```bash
     i2cdetect -y 1
     ```
     Address `48` should appear in the matrix.
  2. Ensure `ADDR` pin on the ADS1115 board is wired directly to `GND`.
  3. Confirm 3.3V with a multimeter across ADS1115 `VDD` and `GND`.

### Issue 4: Wayland Display Warnings or OpenCV GUI Crashes
* **Cause**: Raspberry Pi OS Bookworm uses Wayland (Wayfire/Labwc) rather than X11. Running OpenCV `cv2.imshow()` across remote SSH or without an active Wayland display session triggers:
  `Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. QFontDatabase: Cannot find font directory...`
* **Resolution**:
  * **Recommended**: Always pass `--headless` in production or SSH sessions:
    ```bash
    python -m app.conveyor_pipeline --speed 0.12 --headless
    ```
  * **If GUI display on local HDMI monitor is required**: Export the Qt platform variable prior to execution:
    ```bash
    export QT_QPA_PLATFORM=xcb
    export DISPLAY=:0
    python -m app.conveyor_pipeline --speed 0.12
    ```

---

## 8. Summary Checklist for Tuesday Morning Demonstration

- [ ] **Physical**: Sensor mounting bar locked at $H = 240\text{ mm}$ above conveyor belt.
- [ ] **Physical**: Common star ground continuity checked ($< 0.5\ \Omega$ between Pi GND, PSU 0V, ADC GND).
- [ ] **Cabling**: 22-pin camera cable locked firmly in `CAM/DISP 1` port.
- [ ] **Power**: Op-amp dual supply energized at $\pm 12\text{V DC}$.
- [ ] **Software**: Run `python scripts/check_rpi5_hardware.py` $\rightarrow$ Confirm `5/5 PASS`.
- [ ] **Conveyor**: Speed controller dialed to $0.12\text{ m/s}$.
- [ ] **Launch**: Run `python -m app.conveyor_pipeline --speed 0.12 --headless --csv-out runtime/demo_results.csv`.

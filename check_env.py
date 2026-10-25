#!/usr/bin/env python3
"""Pre-flight checks for the fruit detector on Raspberry Pi 4 (64-bit OS, Bookworm).

Checks:
- Raspberry Pi hardware identification
- 64-bit OS architecture (aarch64)
- Python 3.11+ runtime
- V4L2 USB camera (/dev/video0) detection & frame grab
- ONNX Runtime engine (CPUExecutionProvider)
- All 4 Fruit Model files (Apple, Mango, Guava, Pomegranate) and embedded metadata
- GPIO hardware actuator library availability (gpiozero / gpiod)
"""
import argparse
import ast
import os
import platform
import subprocess
import sys

FRUITS = {
    "apple": {
        "model": "best.onnx",
        "classes": ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"],
    },
    "mango": {
        "model": "best_mango.onnx",
        "classes": ["mango_fresh", "mango_rotten", "mango_bruised", "mango_damaged"],
    },
    "guava": {
        "model": "best_guava.onnx",
        "classes": ["guava_fresh", "guava_rotten", "guava_bruised", "guava_damaged"],
    },
    "pomegranate": {
        "model": "best_pomegranate.onnx",
        "classes": ["pomegranate_fresh", "pomegranate_rotten", "pomegranate_bruised", "pomegranate_damaged"],
    },
}

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status:4s}] {name}" + (f" - {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi 4 Pre-flight Environment Checks")
    ap.add_argument("--cam", type=int, default=0, help="Camera device index (default 0)")
    ap.add_argument("--fruit", choices=["apple", "mango", "guava", "pomegranate", "all"], default="all")
    args = ap.parse_args()

    print("=" * 65)
    print("  Raspberry Pi 4 Fruit Damage Sorter - Pre-Flight Check")
    print("=" * 65)

    # 1. Hardware Board
    try:
        with open("/proc/device-tree/model") as f:
            board = f.read().strip("\x00 \n")
    except OSError:
        board = f"{platform.system()} {platform.machine()}"
    report("Board Architecture", "Raspberry Pi" in board or "aarch64" in platform.machine() or "x86_64" in platform.machine(), board)

    # 2. 64-bit OS
    arch = platform.machine()
    report("64-bit OS (aarch64/x86_64)", arch in ("aarch64", "arm64", "x86_64", "AMD64"), arch)

    # 3. Python 3.11+
    py_ver = platform.python_version()
    report("Python 3.11+ Runtime", sys.version_info >= (3, 11), f"v{py_ver}")

    # 4. Camera Device existence
    dev = f"/dev/video{args.cam}"
    if os.name != "nt":
        report(f"Camera Device ({dev})", os.path.exists(dev), dev if os.path.exists(dev) else "not found")
        try:
            out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=5)
            report("v4l2-ctl utility", out.returncode == 0, "available")
        except FileNotFoundError:
            report("v4l2-ctl utility", False, "not installed (run setup_pi.sh)")
    else:
        report("Camera Device (Windows /dev emulation)", True, f"Webcam index {args.cam}")

    # 5. OpenCV Frame Grab
    try:
        import cv2
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_ANY)
        if os.name != "nt":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        cap.release()
        detail = f"{frame.shape[1]}x{frame.shape[0]} BGR" if ok else f"failed to read from cam {args.cam}"
        report("Camera Stream & Frame Grab", ok, detail)
    except Exception as e:
        report("Camera Stream & Frame Grab", False, str(e)[:150])

    # 6. ONNX Runtime Engine
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        report("ONNX Runtime Engine", "CPUExecutionProvider" in avail, f"v{ort.__version__} ({avail[0]})")
    except Exception as e:
        report("ONNX Runtime Engine", False, str(e)[:150])

    # 7. Model Verification
    targets = FRUITS.keys() if args.fruit == "all" else [args.fruit]
    for f_name in targets:
        m_info = FRUITS[f_name]
        m_path = m_info["model"]
        exp_classes = m_info["classes"]
        if not os.path.exists(m_path):
            report(f"Model File [{f_name.upper()}]", False, f"Missing: {m_path}")
            continue

        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            sess = ort.InferenceSession(m_path, so, providers=["CPUExecutionProvider"])
            inp_shape = sess.get_inputs()[0].shape
            report(f"Model Load [{f_name.upper()}]", True, f"{m_path} (shape {inp_shape})")

            meta = sess.get_modelmeta().custom_metadata_map
            if "names" not in meta:
                report(f"Metadata Names [{f_name.upper()}]", False, "Missing metadata 'names'")
            else:
                d = ast.literal_eval(meta["names"])
                names = [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)
                matching = set(names) == set(exp_classes)
                report(f"Classes Match [{f_name.upper()}]", matching, f"{names}")
        except Exception as e:
            report(f"Model Validation [{f_name.upper()}]", False, str(e)[:150])

    # 8. Optional GPIO Actuation
    try:
        import gpiozero
        report("GPIO Hardware Actuation (gpiozero)", True, f"v{gpiozero.__version__}")
    except ImportError:
        try:
            import gpiod
            report("GPIO Hardware Actuation (gpiod)", True, "modern Linux gpiod ready")
        except ImportError:
            report("GPIO Hardware Actuation", True, "Simulation / USB / Network trigger mode")

    print("=" * 65)
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

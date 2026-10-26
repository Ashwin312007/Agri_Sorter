#!/usr/bin/env python3
"""USB Webcam Discovery, Testing, and Configuration Tool.

Designed for Raspberry Pi 4 and Linux / Windows edge deployments:
- Detects USB webcams vs Raspberry Pi hardware codec / ISP nodes
- Probes V4L2 device nodes and verifies frame grab in OpenCV
- Identifies persistent device paths (/dev/v4l/by-id/*)
- Saves active camera selection to 'cam.conf' for use across all tools
"""
import argparse
import contextlib
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONF_FILE = "cam.conf"


@contextlib.contextmanager
def suppress_c_stderr():
    """Suppress C-level OpenCV VideoCapture backend probing warnings."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        try:
            yield
        finally:
            os.dup2(save_stderr, 2)
            os.close(save_stderr)
            os.close(null_fd)
    except Exception:
        yield


def load_saved_cam_config(conf_path=CONF_FILE):
    """Read saved camera configuration if present."""
    cfg = {}
    p = Path(conf_path)
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[WARN] Failed reading {conf_path}: {e}")
    return cfg


def save_cam_config(cam_index, cam_dev=None, cam_name="USB Webcam", width=640, height=480, conf_path=CONF_FILE):
    """Write camera configuration to cam.conf."""
    content = [
        "# Raspberry Pi 4 Fruit Damage Sorter - Camera Configuration",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"CAM_INDEX={cam_index}",
    ]
    if cam_dev:
        content.append(f"CAM_DEV={cam_dev}")
    content.extend([
        f"CAM_NAME={cam_name}",
        f"CAM_WIDTH={width}",
        f"CAM_HEIGHT={height}",
        "",
    ])
    Path(conf_path).write_text("\n".join(content), encoding="utf-8")
    print(f"[OK] Saved camera configuration to {conf_path}")


def get_v4l2_devices_info():
    """Parse v4l2-ctl --list-devices output to identify USB webcams vs platform nodes."""
    devices = {}
    if os.name != "posix":
        return devices

    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return devices
    except Exception:
        return devices

    current_name = None
    for raw_line in out.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if not raw_line.startswith("\t") and not raw_line.startswith(" "):
            # Header line: e.g. "USB 2.0 Camera (usb-0000:01:00.0-1.2):"
            current_name = line.strip().rstrip(":")
        elif current_name and "/dev/video" in line:
            dev_path = line.strip()
            # Capture the first device node for this camera
            if dev_path not in devices:
                is_usb = any(tok in current_name.lower() for tok in ["usb", "webcam", "uvc", "camera"]) and not any(
                    tok in current_name.lower() for tok in ["bcm2835", "unicam", "codec", "isp"]
                )
                devices[dev_path] = {
                    "name": current_name,
                    "is_usb": is_usb,
                }
    return devices


def probe_single_camera(dev_or_idx, width=640, height=480):
    """Test if a camera index or path can be opened and successfully grabs a frame."""
    try:
        import cv2
    except ImportError:
        return False, "OpenCV (cv2) not installed", None

    # On Windows cv2.CAP_ANY opens standard UVC directly without DSHOW backend index mismatch
    api = cv2.CAP_V4L2 if (os.name == "posix" and hasattr(cv2, "CAP_V4L2")) else cv2.CAP_ANY
    
    # If dev_or_idx is digit string, convert to int
    target = int(dev_or_idx) if isinstance(dev_or_idx, str) and dev_or_idx.isdigit() else dev_or_idx

    with suppress_c_stderr():
        cap = cv2.VideoCapture(target, api)

    if not cap.isOpened():
        return False, "Failed to open video capture", None

    try:
        if os.name == "posix":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Allow camera sensor to warm up
        for _ in range(3):
            with suppress_c_stderr():
                ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                cap.release()
                return True, f"{w}x{h} BGR", frame
        cap.release()
        return False, "Device opened but read() returned no frame (likely metadata/control node)", None
    except Exception as e:
        if cap:
            cap.release()
        return False, str(e), None


def scan_cameras(max_probe=10):
    """Scan and return list of working cameras."""
    v4l_info = get_v4l2_devices_info()
    found = []

    if os.name == "posix":
        # Check /dev/video* devices
        video_devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.search(r"\d+", p).group() if re.search(r"\d+", p) else 999))
        if not video_devs:
            video_devs = [f"/dev/video{i}" for i in range(max_probe)]
    else:
        video_devs = [i for i in range(max_probe)]

    consecutive_fails = 0
    for dev in video_devs:
        # Extract index number
        if isinstance(dev, str):
            m = re.search(r"/dev/video(\d+)", dev)
            idx = int(m.group(1)) if m else None
        else:
            idx = dev
            dev = f"Webcam #{idx}"

        dev_str = str(dev)
        v_meta = v4l_info.get(dev_str, {})
        dev_name = v_meta.get("name", f"Camera {idx}")
        is_usb = v_meta.get("is_usb", True if os.name == "nt" else False)

        ok, detail, test_frame = probe_single_camera(idx if idx is not None else dev_str)
        if ok:
            consecutive_fails = 0
            found.append({
                "index": idx if idx is not None else 0,
                "dev": dev_str,
                "name": dev_name,
                "is_usb": is_usb,
                "resolution": detail,
                "frame": test_frame,
            })
        else:
            consecutive_fails += 1
            if os.name == "nt" and consecutive_fails >= 2 and found:
                break

    # Sort USB cameras to the top
    found.sort(key=lambda x: (not x["is_usb"], x["index"]))
    return found


def resolve_camera_source(cam_arg=None, conf_path=CONF_FILE):
    """Universal fast camera resolver for detect.py and web_detect.py.
    
    Priority:
    1. cam_arg (if explicitly provided and not None / 'auto')
    2. Environment variable FRUIT_CAM or CAM_INDEX
    3. cam.conf file
    4. Fast probe of primary indices (0, 1, 2)
    5. Fallback: 0
    """
    if cam_arg is not None and str(cam_arg).lower() not in ("auto", "none", ""):
        # Check if integer string
        if str(cam_arg).isdigit():
            return int(cam_arg)
        return str(cam_arg)

    # Check env vars
    env_cam = os.environ.get("FRUIT_CAM") or os.environ.get("CAM_INDEX")
    if env_cam:
        return int(env_cam) if env_cam.isdigit() else env_cam

    # Check cam.conf
    cfg = load_saved_cam_config(conf_path)
    if "CAM_INDEX" in cfg and cfg["CAM_INDEX"].isdigit():
        return int(cfg["CAM_INDEX"])
    if "CAM_DEV" in cfg and cfg["CAM_DEV"]:
        return cfg["CAM_DEV"]

    # Fast probe (avoids multi-second timeouts)
    for probe_idx in (0, 1, 2):
        ok, _, _ = probe_single_camera(probe_idx)
        if ok:
            return probe_idx

    return 0


def main():
    parser = argparse.ArgumentParser(description="Find, test, and configure USB webcam for Fruit Sorter")
    parser.add_argument("--list", "-l", action="store_true", help="List all detected cameras and exit")
    parser.add_argument("--auto", "-a", action="store_true", help="Automatically select and save the first working webcam")
    parser.add_argument("--set", "-s", type=str, default=None, help="Set active camera index or /dev/video* path")
    parser.add_argument("--test", "-t", type=str, default=None, help="Test capture frame from specified camera and save test image")
    parser.add_argument("--out", default=CONF_FILE, help=f"Path to config file (default: {CONF_FILE})")
    args = parser.parse_args()

    print("=" * 65)
    print("  Raspberry Pi 4 USB Webcam Configuration & Test Tool")
    print("=" * 65)

    # Test single camera
    if args.test is not None:
        target = int(args.test) if args.test.isdigit() else args.test
        print(f"Testing capture on camera: {target}...")
        ok, detail, frame = probe_single_camera(target)
        if ok and frame is not None:
            import cv2
            out_img = "cam_test.jpg"
            cv2.imwrite(out_img, frame)
            print(f"[PASS] Successfully grabbed test frame: {detail}")
            print(f"[PASS] Saved test snapshot to {out_img}")
            return 0
        else:
            print(f"[FAIL] Camera test failed: {detail}")
            return 1

    # Explicitly set camera
    if args.set is not None:
        target = int(args.set) if args.set.isdigit() else args.set
        dev_path = f"/dev/video{target}" if isinstance(target, int) and os.name == "posix" else str(target)
        save_cam_config(target, cam_dev=dev_path, conf_path=args.out)
        return 0

    print("Scanning available video devices...")
    cams = scan_cameras(max_probe=10)

    if not cams:
        print("[FAIL] No operational cameras detected!")
        print("Troubleshooting:")
        print("  1. Ensure USB webcam is securely plugged into a blue USB 3.0 port.")
        print("  2. Run 'lsusb' and 'v4l2-ctl --list-devices' to check Linux recognition.")
        print("  3. Ensure user is in 'video' group: sudo usermod -aG video $USER")
        return 1

    print(f"\nFound {len(cams)} operational camera(s):")
    print("-" * 65)
    print(f" {'#':2s} | {'Device':14s} | {'USB':5s} | {'Resolution':12s} | {'Name'}")
    print("-" * 65)
    for i, c in enumerate(cams):
        usb_mark = "YES" if c["is_usb"] else "NO"
        rec = " [RECOMMENDED]" if i == 0 and c["is_usb"] else ""
        print(f" {c['index']:2d} | {c['dev']:14s} | {usb_mark:5s} | {c['resolution']:12s} | {c['name']}{rec}")
    print("-" * 65)

    if args.list:
        cfg = load_saved_cam_config(args.out)
        curr = cfg.get("CAM_INDEX", "Not set (defaults to 0)")
        print(f"\nCurrent configured camera in {args.out}: {curr}")
        return 0

    if args.auto:
        best = cams[0]
        print(f"\nAuto-selecting best webcam: Index {best['index']} ({best['name']})")
        save_cam_config(best["index"], cam_dev=best["dev"], cam_name=best["name"], conf_path=args.out)
        return 0

    # Interactive prompt if multiple or single
    print(f"\nDetected default recommended camera: Index {cams[0]['index']} ({cams[0]['name']})")
    try:
        ans = input(f"Enter camera index to use [{cams[0]['index']}]: ").strip()
        selected_idx = int(ans) if ans.isdigit() else cams[0]["index"]
    except (EOFError, KeyboardInterrupt):
        selected_idx = cams[0]["index"]

    # Find matching camera
    chosen = next((c for c in cams if c["index"] == selected_idx), None)
    if chosen:
        save_cam_config(chosen["index"], cam_dev=chosen["dev"], cam_name=chosen["name"], conf_path=args.out)
    else:
        dev_path = f"/dev/video{selected_idx}" if os.name == "posix" else f"Webcam #{selected_idx}"
        save_cam_config(selected_idx, cam_dev=dev_path, conf_path=args.out)

    print("\nYou can verify live capture at any time with:")
    print(f"  python3 find_cam.py --test {selected_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pre-flight checks for the fruit detector on Raspberry Pi. Prints PASS/FAIL per check."""
import argparse
import ast
import os
import platform
import subprocess
import sys

APPLE_CLASSES = ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"]
results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.onnx")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--expect", default=",".join(APPLE_CLASSES),
                    help="comma-separated expected class names (default: apple classes)")
    args = ap.parse_args()
    expected = [n.strip() for n in args.expect.split(",") if n.strip()]

    try:
        with open("/proc/device-tree/model") as f:
            board = f.read().strip("\x00 \n")
    except OSError:
        board = "unknown"
    report("Board is Raspberry Pi", "Raspberry Pi" in board, board)

    arch = platform.machine()
    report("64-bit OS (aarch64)", arch == "aarch64", arch)

    report("Python 3.11+", sys.version_info >= (3, 11), platform.python_version())

    dev = f"/dev/video{args.cam}"
    report(f"{dev} exists", os.path.exists(dev))
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=10)
        listing = " | ".join(out.stdout.split())[:200]
        report("v4l2-ctl lists devices", out.returncode == 0, listing)
    except FileNotFoundError:
        report("v4l2-ctl installed", False, "run setup_pi.sh")

    try:
        import cv2
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        cap.release()
        detail = f"{frame.shape[1]}x{frame.shape[0]}" if ok else f"index {args.cam} failed to grab"
        report("Camera opens and grabs a frame", ok, detail)
    except Exception as e:
        report("Camera opens and grabs a frame", False, str(e)[:200])

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        report("ONNX model loads", True, f"{args.model} input {sess.get_inputs()[0].shape}")

        meta = sess.get_modelmeta().custom_metadata_map
        if "names" not in meta:
            report("Class names in model metadata", False, "missing - pass --names to detect.py")
        else:
            d = ast.literal_eval(meta["names"])
            names = [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)
            report("Class names in model metadata", True, str(names)[:150])
            missing = set(expected) - set(names)
            extra = set(names) - set(expected)
            report("Class names match expected", not missing and not extra,
                   "OK" if not missing and not extra else f"missing {sorted(missing)}, extra {sorted(extra)}")
    except Exception as e:
        report("ONNX model loads", False, str(e)[:200])

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

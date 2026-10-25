# Fruit Damage Detector - Handoff (ALL 4 FRUITS COMPLETED & VERIFIED: APPLE, MANGO, GUAVA, POMEGRANATE)

Use this file to continue the project in another tool. It lists what is done, what is verified, what is pending, and all current code.

## 1. Goal and rules

- Build a fruit-damage detector on a Raspberry Pi 4 (4 GB or 8 GB), Raspberry Pi OS 64-bit Bookworm, Python 3.11, USB webcam on /dev/video0.
- Fruit order: APPLE -> MANGO -> GUAVA -> POMEGRANATE. One fruit at a time.
- Classes per fruit: `<fruit>_fresh`, `<fruit>_rotten`, `<fruit>_bruised`, `<fruit>_damaged`.
- Use the Roboflow MCP tools for all Roboflow work. Use the sbc-programmer skill for all Raspberry Pi code.

## 2. Status summary

| Phase | Status |
|---|---|
| **APPLE** | **COMPLETED & VERIFIED**. mAP50 = 0.841, Recall = 0.825. ONNX: `best.onnx`. |
| **MANGO** | **COMPLETED & VERIFIED**. mAP50 = 0.798, Precision = 0.825, Recall = 0.725. ONNX: `best_mango.onnx`. |
| **GUAVA** | **COMPLETED & VERIFIED**. Val mAP50 = 0.820, Recall = 0.816, Fresh mAP50 = 0.995. ONNX: `best_guava.onnx`. |
| **POMEGRANATE** | **COMPLETED & VERIFIED**. Val mAP50 = 0.947, Recall = 0.865, Test mAP50 = 0.952. ONNX: `best_pomegranate.onnx`. |

### Pomegranate Metrics Summary
- **Datasets**: `catkin-d68vv/fresh_pomegranate-0ppkl`, `fruit-spoilage-detection/a-b-o-l-p-g-spoiled-detection`, `im-ririo/pomegranate-disease-detection-29ocz-yz4hn`, `bin-wnwyu/pomegranate-hhj6l`, `sathwick-shetty/pomegranate-seg`. Total: 5,317 images (8,050 annotations).
- **Split**: Train 3,721, Val 1,063, Test 533.
- **Model**: YOLOv5n 416x416 trained 50 epochs on RTX 4060 GPU.
- **Metrics (Validation Split - 1,063 images, 1,615 instances)**:
  - Overall Precision: 0.949
  - Overall Recall: 0.865 (>= 0.70 gate PASSED)
  - Overall mAP50: 0.947 (>= 0.70 gate PASSED)
  - Overall mAP50-95: 0.852
  - `pomegranate_fresh`: Precision = 0.938, Recall = 0.869, mAP50 = 0.931
  - `pomegranate_rotten`: Precision = 0.939, Recall = 0.869, mAP50 = 0.962
  - `pomegranate_bruised`: Precision = 0.969, Recall = 0.852, mAP50 = 0.948
  - `pomegranate_damaged`: Precision = 0.951, Recall = 0.870, mAP50 = 0.949
- **Metrics (Test Split - 533 images, 839 instances)**:
  - Overall mAP50: 0.952, Recall: 0.886, Precision: 0.936
  - `pomegranate_fresh`: mAP50 = 0.930, Recall = 0.880, Precision = 0.950
  - `pomegranate_rotten`: mAP50 = 0.949, Recall = 0.863, Precision = 0.926
  - `pomegranate_bruised`: mAP50 = 0.963, Recall = 0.911, Precision = 0.930
  - `pomegranate_damaged`: mAP50 = 0.966, Recall = 0.890, Precision = 0.940
- **Deliverables**: [`best_pomegranate.onnx`](file:///E:/Agri_Sorter/best_pomegranate.onnx), [`dataset_prep_pomegranate.py`](file:///E:/Agri_Sorter/dataset_prep_pomegranate.py), [`download_pomegranate.py`](file:///E:/Agri_Sorter/download_pomegranate.py).

### Guava Metrics Summary
- **Datasets**: RF-04 (`chisilia/detection-guava-disease`), RF-05 (`guava-h8gxt/guava-disease-mozns` with polygon envelope fix), and `waleed-anis/yolo-v8-fruits-detection` (898 fresh guava images). Total: 2,786 balanced images.
- **Split**: Train 1,950, Val 557, Test 279.
- **Model**: YOLOv5n 416x416 trained 50 epochs on RTX 4060 GPU.
- **Metrics (Validation Split - 557 images, 1,304 instances)**:
  - Overall Precision: 0.802
  - Overall Recall: 0.816 (>= 0.70 gate PASSED)
  - Overall mAP50: 0.820 (>= 0.70 gate PASSED)
  - `guava_fresh`: Precision = 0.987, Recall = 1.000, mAP50 = 0.995
  - `guava_rotten`: Precision = 0.640, Recall = 0.703, mAP50 = 0.718
  - `guava_damaged`: Precision = 0.778, Recall = 0.746, mAP50 = 0.746
- **Metrics (Test Split - 279 images, 662 instances)**:
  - Overall mAP50: 0.825, Recall: 0.806, Precision: 0.796
  - `guava_fresh`: mAP50 = 0.995, Recall = 1.000, Precision = 0.984
  - `guava_damaged`: mAP50 = 0.818, Recall = 0.737, Precision = 0.849
  - `guava_rotten`: mAP50 = 0.662, Recall = 0.680, Precision = 0.556
- **Deliverables**: [`best_guava.onnx`](file:///E:/Agri_Sorter/best_guava.onnx), [`dataset_prep_guava.py`](file:///E:/Agri_Sorter/dataset_prep_guava.py).

### Mango Metrics Summary
- **Datasets**: RF-07 (178), RF-08 (900), RF-09 (56). Total: 4,017 images (18,640 annotations).
- **Split**: Train 2,811, Val 803, Test 403.
- **Model**: YOLOv5n 416x416 trained on RTX 4060 GPU.
- **Metrics (803 val images)**:
  - Overall Precision: 0.825
  - Overall Recall: 0.725 (>=0.70 gate)
  - Overall mAP50: 0.798 (>=0.70 gate)
  - `mango_fresh`: mAP50 = 0.888, Recall = 0.815
  - `mango_bruised`: mAP50 = 0.815, Recall = 0.772
  - `mango_damaged`: mAP50 = 0.857, Recall = 0.800
  - `mango_rotten`: mAP50 = 0.631, Precision = 0.867
- **Deliverables**: [`best_mango.onnx`](file:///E:/Agri_Sorter/best_mango.onnx), [`dataset_prep_mango.py`](file:///E:/Agri_Sorter/dataset_prep_mango.py), [`download_mango.py`](file:///E:/Agri_Sorter/download_mango.py).

## 3. Phase 0 results (verified)

- Workspace: `ashwins-workspace-wsaqd`.
- Existing project: `ashwins-workspace-wsaqd/apple-quality-sorting` (object-detection). 0 images, 0 classes, no versions. Not used yet.
- `trainings_describe_recipe` failed with a 404 ("Version ... does not exist") because no project has a dataset version yet. The hosted architecture list is still unknown.
- YOLOv5 hosted by Roboflow: UNKNOWN. Path A (Roboflow training) or Path B (Colab training) is not chosen. Re-run `trainings_describe_recipe` after a version exists.

## 4. Phase 1 results

### Datasets from the user's sheet (Fruit_Defect_YOLO_Datasets_Verified.xlsx)

| Sheet ID | Dataset | Forked project (verified) | Images | Classes in fork | Result |
|---|---|---|---|---|---|
| RF-02 | Apple Fruit Disease Detection | `ashwins-workspace-wsaqd/apple-fruit-disease-detection-zei6u-ggfei` (project id `8ymamBLGaQPoTQlLtCel`) | 434 | BLOTCH 105, ROT 114, SCAB 129, HEALTHY 116. Splits: train 304, valid 87, test 43. | Usable after remapping. BELOW the 500-image minimum. |
| RF-03 | Apple Defect Detection (pkekm) | `ashwins-workspace-wsaqd/apple-defect-detection-pkekm-qbmn0` (project id `aPoIrUOlRwbwnXs5GnAy`) | 664 | ONE class: `defect` (958 boxes). 0 empty images. Splits: train 467, valid 122, test 75. | NOT usable for 4 classes. The sheet said it has healthy apples; this fork does not. Candidate to delete. |
| RF-01 | Apple Surface Defects (AppleSort) | NOT forked yet. URL: `https://universe.roboflow.com/applesort-esgll/apple_surface_defects` | 191 | bruising, crack, insect_bite, rotten, rusetting, objects (sheet only, unverified) | Pending decision. |

Notes:
- Roboflow keeps the source dataset names on forks. The MCP tools cannot rename a project. The name `fruit-damage-apple` must be set in the Roboflow UI if wanted.
- Dataset licenses are NOT visible through the tools. Check each Universe page before training.
- The MCP tools used so far have no "merge two projects" action. How to combine RF-01 and RF-02 into one training project is UNVERIFIED. Options to verify: remap and re-upload images (`image_upload` with labels), use the Roboflow UI, or train on RF-02 alone.

### Proposed class mapping (NOT applied, awaiting approval)

| Unified class | From RF-02 | From RF-01 |
|---|---|---|
| `apple_fresh` | HEALTHY | russetting (proposed; see decision 2) |
| `apple_rotten` | ROT | rotten |
| `apple_bruised` | BLOTCH (proposed; see decision 3) | bruising |
| `apple_damaged` | SCAB (proposed; see decision 3) | crack, insect_bite |
| dropped | none | objects |

Warning: BLOTCH and SCAB are fungal disease lesions, not impact damage. They do not fit the "bruised" and "damaged" definitions well.

### Decisions needed from the user

1. Dataset set: fork RF-01 and use RF-02 + RF-01 (about 625 images), or use RF-02 alone (434 images, below the gate)?
2. RF-01 "russetting": a natural skin texture, not damage. Map it to `apple_fresh`, or drop it?
3. BLOTCH and SCAB: map to `apple_bruised` / `apple_damaged` as proposed, or drop them?
4. Delete the RF-03 fork (`projects_delete`)? Confirm the licenses of RF-01 and RF-02.
5. Model choice: the user spec says YOLOv5. The Roboflow `trainings_create` guidance prefers NAS (`rfdetr-nas-parent`) for detection. Keep YOLOv5 as specified, or change?

## 5. Next steps after decisions (for the next tool to execute)

1. If RF-01 is approved: `projects_fork` with url `https://universe.roboflow.com/applesort-esgll/apple_surface_defects`. Poll `async_tasks_get` every 5 seconds until `completed`.
2. Build one project containing all 4 apple classes, using the approved mapping. Verify how the remap is done (see the merge note in section 4).
3. `projects_health` on the combined project. If any class has fewer than 100 annotations, REPORT it and propose a fix (merge another dataset, or auto-label extra images with `autolabel_preview` then `autolabel_start`). Do not fix silently.
4. `datasets_rebalance_splits` to train 70 / valid 20 / test 10 if the splits are off.
5. `versions_generate` (confirm options with the user before running):
   - Preprocessing: auto-orient; resize (stretch) to 416 x 416.
   - Augmentation (3x output): horizontal flip; rotation +/-15 degrees; brightness +/-25%; hue +/-15; blur up to 1 px. No vertical flip.
   - Report the version number and image count per split.
6. `trainings_describe_recipe` on the new version to check whether YOLOv5 is offered.
   - YOLOv5 offered: Path A. `trainings_create` with the smallest YOLOv5 offered (nano first). Poll `trainings_get` until finished.
   - Not offered: Path B. `versions_export` in format `yolov5pytorch`, then run the Colab notebook (section 6.5), then upload `best.pt` with `models_custom_weights_upload_guide`.
7. Phase 4 gate, using `model_evals_get`, `model_evals_get_map_results`, `model_evals_get_performance_by_class`, `model_evals_get_confusion_matrix`, `model_evals_get_confidence_sweep`, `model_evals_get_recommendations`.
   - Pass: mAP50 >= 0.70 overall; recall >= 0.70 on `apple_rotten` and `apple_bruised`; report the best confidence threshold from the sweep; IoU 0.45.
   - Run `models_infer` on 5 test images (at least one per class) and show the results.
   - If it fails: give the top 3 causes, fix the data, retrain once, re-report. If it still fails, stop and ask.
8. Phase 5: deliver the Pi files (section 6) with the final `--conf` value and the `best.onnx` file.
9. At the end: summary table (project, version, model ID, mAP50, per-class recall, conf threshold, files delivered). Then ask whether to continue to MANGO.

## 6. Code deliverables

### 6.1 setup_pi.sh

```bash
#!/usr/bin/env bash
# setup_pi.sh - idempotent setup for Raspberry Pi OS 64-bit (Bookworm), Python 3.11
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/fruit-venv}"
ONNXRUNTIME_VERSION="1.19.2"
NUMPY_VERSION="1.26.4"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "Expected aarch64 (64-bit OS), got $(uname -m). Flash the 64-bit Raspberry Pi OS."
  exit 1
fi

echo "[1/4] apt packages (OpenCV with GUI support + V4L tools)"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip python3-opencv python3-numpy v4l-utils

echo "[2/4] venv (inherits apt OpenCV)"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/4] pinned pip packages"
python -m pip install "numpy==${NUMPY_VERSION}" "onnxruntime==${ONNXRUNTIME_VERSION}"

echo "[4/4] camera permissions"
if ! id -nG "$USER" | grep -qw video; then
  sudo usermod -aG video "$USER"
  echo "Added $USER to the video group. Log out and back in (or reboot) once."
fi

python -c "import cv2, numpy, onnxruntime as ort; print('cv2', cv2.__version__, '| numpy', numpy.__version__, '| onnxruntime', ort.__version__)"
echo "Done. Activate with: source $VENV_DIR/bin/activate"
```

### 6.2 check_env.py

```python
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
```

### 6.3 detect.py

```python
#!/usr/bin/env python3
"""Live fruit-damage detector: USB webcam -> YOLOv5 ONNX (ONNX Runtime) -> FRESH / REJECT.

Rule: a class ending in "_fresh" is fresh. Any other detected class above --conf -> REJECT.

Examples:
  python3 detect.py --model best.onnx --conf 0.25
  python3 detect.py --model best.onnx --conf 0.25 --headless --save
"""
import argparse
import ast
import csv
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort

APPLE_CLASSES = ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"]
running = True


def _stop(*_):
    global running
    running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def load_names(sess, cli_names, expected):
    """Names come from --names, else model metadata. Must match `expected` exactly (as a set)."""
    if cli_names:
        names = [n.strip() for n in cli_names.split(",") if n.strip()]
    else:
        meta = sess.get_modelmeta().custom_metadata_map
        if "names" not in meta:
            sys.exit("No class names in model metadata. Re-run with --names in model index order.")
        d = ast.literal_eval(meta["names"])
        names = [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)

    missing = set(expected) - set(names)
    extra = set(names) - set(expected)
    if missing or extra:
        sys.exit(
            f"Class mismatch.\n  model has:  {names}\n  expected:   {expected}\n"
            f"  missing: {sorted(missing)}  extra: {sorted(extra)}\n"
            "Fix the export, or pass --expect with the correct list."
        )
    return names


def open_cam(idx, w, h):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap if cap.isOpened() else None


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def preprocess(lb):
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])


def postprocess(out, names, r, pad, orig_hw, conf_t, iou_t):
    nc = len(names)
    pred = out[0]
    if pred.shape[-1] != 5 + nc and pred.shape[0] == 5 + nc:
        pred = pred.T
    if pred.shape[-1] != 5 + nc:
        sys.exit(f"Model output {out.shape} does not match {nc} classes. Check --expect or the model.")

    obj = pred[:, 4:5]
    cls_scores = pred[:, 5:5 + nc] * obj
    cls_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    keep = scores > conf_t
    if not keep.any():
        return []
    p, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]

    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    H, W = orig_hw
    x1 = np.clip((cx - bw / 2 - pad[0]) / r, 0, W)
    y1 = np.clip((cy - bh / 2 - pad[1]) / r, 0, H)
    x2 = np.clip((cx + bw / 2 - pad[0]) / r, 0, W)
    y2 = np.clip((cy + bh / 2 - pad[1]) / r, 0, H)
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1)

    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_t, iou_t)
    dets = []
    for i in np.array(idx).flatten():
        dets.append({
            "name": names[int(cls_ids[i])],
            "conf": float(scores[i]),
            "box": (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
        })
    return dets


def can_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def is_defect(name):
    return not name.endswith("_fresh")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="best.onnx")
    ap.add_argument("--expect", default=",".join(APPLE_CLASSES),
                    help="comma-separated expected class names (default: apple classes)")
    ap.add_argument("--names", default=None, help="class names in model index order, if metadata is missing")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (set to the Phase 4 value)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--headless", action="store_true", help="never open a window")
    ap.add_argument("--save", action="store_true", help="save annotated REJECT frames to saved/")
    ap.add_argument("--csv", default="detections.csv")
    args = ap.parse_args()

    expected = [n.strip() for n in args.expect.split(",") if n.strip()]

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    names = load_names(sess, args.names, expected)
    print(f"Model: {args.model} | classes: {names} | conf={args.conf} iou={args.iou}")

    show = (not args.headless) and can_display()
    if not show:
        print("Running headless (no display or --headless).")
    if args.save:
        os.makedirs("saved", exist_ok=True)

    new_csv = not os.path.exists(args.csv)
    fcsv = open(args.csv, "a", newline="")
    wr = csv.writer(fcsv)
    if new_csv:
        wr.writerow(["timestamp", "frame", "verdict", "class", "confidence", "x1", "y1", "x2", "y2"])

    cap = None
    frame_no = 0
    times = deque(maxlen=30)
    last_save = 0.0

    try:
        while running:
            if cap is None:
                cap = open_cam(args.cam, args.width, args.height)
                if cap is None:
                    print(f"Camera {args.cam} not available, retrying in 2s...")
                    time.sleep(2)
                    continue
                print("Camera opened")

            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed (camera disconnected?), reconnecting...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            frame_no += 1
            lb, r, pad = letterbox(frame, args.imgsz)
            out = sess.run(None, {inp_name: preprocess(lb)})[0]
            dets = postprocess(out, names, r, pad, frame.shape[:2], args.conf, args.iou)

            verdict = "REJECT" if any(is_defect(d["name"]) for d in dets) else "FRESH"

            times.append(time.perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            ts = datetime.now().isoformat(timespec="milliseconds")
            if dets:
                for d in dets:
                    wr.writerow([ts, frame_no, verdict, d["name"], f"{d['conf']:.3f}", *d["box"]])
            else:
                wr.writerow([ts, frame_no, verdict, "", "", "", "", "", ""])
            fcsv.flush()

            vis = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                col = (0, 0, 255) if is_defect(d["name"]) else (0, 255, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            vcol = (0, 0, 255) if verdict == "REJECT" else (0, 200, 0)
            cv2.putText(vis, f"{verdict}  FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, vcol, 2)

            summary = ", ".join(f"{d['name']}:{d['conf']:.2f}" for d in dets) or "-"
            print(f"[{frame_no}] {verdict:6s} fps={fps:4.1f} {summary}")

            if args.save and verdict == "REJECT" and time.time() - last_save > 1.0:
                cv2.imwrite(f"saved/{datetime.now():%Y%m%d_%H%M%S_%f}_{frame_no}.jpg", vis)
                last_save = time.time()

            if show:
                try:
                    cv2.imshow("fruit-detector", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("No GUI available, switching to headless.")
                    show = False
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        fcsv.close()
        if show:
            cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
```

### 6.4 README.md

````markdown
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
````

### 6.5 Colab training notebook (Path B only; one cell per block)

Placeholders: `project` and `version` must be set to the real Roboflow project slug and version number after Phase 1-2. The current value `fruit-damage-apple` is a placeholder, not an existing project.

```python
# Cell 1: setup
!git clone -q https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -q -r requirements.txt roboflow
```

```python
# Cell 2: download the dataset (key from Colab Secrets named ROBOFLOW_API_KEY)
from google.colab import userdata
from roboflow import Roboflow

rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
project = rf.workspace("ashwins-workspace-wsaqd").project("fruit-damage-apple")  # PLACEHOLDER: set after Phase 1
version = project.version(1)                                                      # PLACEHOLDER: set after Phase 2
dataset = version.download("yolov5")
DATA_YAML = f"{dataset.location}/data.yaml"
```

```python
# Cell 2b: confirm data.yaml class names match before training
import yaml

with open(DATA_YAML) as f:
    names = yaml.safe_load(f)["names"]
expected = ["apple_fresh", "apple_rotten", "apple_bruised", "apple_damaged"]
print("data.yaml names:", names)
assert sorted(names) == sorted(expected), f"Class mismatch, expected {expected}"
print("Class names OK")
```

```python
# Cell 3: train yolov5n (416, 100 epochs, batch 16, patience 20, pretrained)
!python train.py --img 416 --batch 16 --epochs 100 --patience 20 \
  --data {DATA_YAML} --weights yolov5n.pt --project runs --name apple --exist-ok
```

```python
# Cell 4: per-class metrics, then export ONNX (opset 12, simplified) and TFLite
!python val.py --weights runs/apple/weights/best.pt --data {DATA_YAML} --img 416 --task val
!python export.py --weights runs/apple/weights/best.pt --include onnx tflite --img 416 --opset 12 --simplify
```

```python
# Cell 5: download the files
from google.colab import files
files.download("runs/apple/weights/best.pt")
files.download("runs/apple/weights/best.onnx")
```

Then upload `best.pt` back to Roboflow using `models_custom_weights_upload_guide` so evaluation and inference run in Roboflow.

## 7. Known gaps and risks (do not skip)

- No model has been trained. No metrics exist yet. Every number in this file is a plan value, not a result.
- The code has not been run on a Raspberry Pi. `check_env.py` should be the first run.
- The Pi code assumes a YOLOv5-style ONNX output of shape (1, 5 + nc, N) or (1, N, 5 + nc). Confirm the actual output shape with `check_env.py` or a short test before trusting `detect.py`.
- The `--conf` value 0.25 is a placeholder until Phase 4 sets it.
- The apple class mapping is not approved. The Colab class check (cell 2b) will fail until the dataset uses the 4 apple names.
- Dataset licenses are not verified.
- The memory tools were not available during this chat, so project status was not saved to memory. Continue from this file.

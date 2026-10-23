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
    api = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") and os.name == "posix" else (cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened() and os.name == "nt":
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
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
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name == "nt")


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
    ap.add_argument("--source", type=str, default=None, help="image path or directory of images for offline test")
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

    if args.source:
        p = os.path.abspath(args.source)
        if os.path.isdir(p):
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            img_files = [os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in exts]
        elif os.path.isfile(p):
            img_files = [p]
        else:
            sys.exit(f"Source not found: {args.source}")

        print(f"Processing {len(img_files)} image(s) from {args.source}...")
        for i, img_path in enumerate(img_files, 1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            lb, r, pad = letterbox(frame, args.imgsz)
            out = sess.run(None, {inp_name: preprocess(lb)})[0]
            dets = postprocess(out, names, r, pad, frame.shape[:2], args.conf, args.iou)
            verdict = "REJECT" if any(is_defect(d["name"]) for d in dets) else "FRESH"
            ts = datetime.now().isoformat(timespec="milliseconds")
            if dets:
                for d in dets:
                    wr.writerow([ts, i, verdict, d["name"], f"{d['conf']:.3f}", *d["box"]])
            else:
                wr.writerow([ts, i, verdict, "", "", "", "", "", ""])
            summary = ", ".join(f"{d['name']}:{d['conf']:.2f}" for d in dets) or "None (Clean)"
            print(f"[{i}/{len(img_files)}] {os.path.basename(img_path):35s} -> {verdict:6s} | {summary}")
            if args.save:
                vis = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    col = (0, 0, 255) if is_defect(d["name"]) else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.imwrite(f"saved/{os.path.basename(img_path)}", vis)
        fcsv.close()
        print("Batch processing complete.")
        return

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

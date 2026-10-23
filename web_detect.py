#!/usr/bin/env python3
"""Dual-View Live Fruit Damage Detector Dashboard.

Streams side-by-side:
- Left: Live Raw Webcam View
- Right: YOLO Vision Model View (Bounding Boxes, Confidences, FRESH/REJECT)
"""
import ast
import os
import signal
import sys
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort

from detect import APPLE_CLASSES, is_defect, letterbox, load_names, open_cam, postprocess, preprocess

running = True
raw_frame_bytes = None
model_frame_bytes = None
latest_stats = {
    "fps": 0.0,
    "verdict": "FRESH",
    "detections": "Clean",
    "count": 0,
}


def stop_handler(*_):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


class DualStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global raw_frame_bytes, model_frame_bytes
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>Apple Defect Detector - Live Dual View</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0f1117;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            width: 100%;
            max-width: 1300px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 15px;
            border-bottom: 1px solid #2d3342;
            margin-bottom: 20px;
        }
        h1 { font-size: 1.6rem; color: #4ade80; }
        .meta { font-size: 0.9rem; color: #94a3b8; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            width: 100%;
            max-width: 1300px;
        }
        .card {
            background: #1e2433;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #334155;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            display: flex;
            flex-direction: column;
        }
        .card-header {
            padding: 12px 16px;
            font-weight: 600;
            background: #181d29;
            border-bottom: 1px solid #2d3748;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stream-view {
            width: 100%;
            aspect-ratio: 4 / 3;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stream-view img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .classes-bar {
            margin-top: 20px;
            width: 100%;
            max-width: 1300px;
            padding: 14px 20px;
            background: #1e2433;
            border-radius: 8px;
            border: 1px solid #334155;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-around;
        }
        .badge-fresh { color: #4ade80; font-weight: bold; }
        .badge-defect { color: #f87171; font-weight: bold; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Apple Damage Detector</h1>
            <div class="meta">YOLOv5 ONNX Edge Engine | Raspberry Pi 4 Target</div>
        </div>
        <div class="meta">
            Target: 416x416 | Classes: 4 | Status: <span style="color:#4ade80;">Active</span>
        </div>
    </header>

    <div class="grid">
        <div class="card">
            <div class="card-header">
                <span>[1] Raw Camera View</span>
                <span style="color:#94a3b8; font-size:0.85rem;">Source: Webcam 0</span>
            </div>
            <div class="stream-view">
                <img src="/video_raw" alt="Raw Webcam View">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>[2] YOLO Vision Model Inference</span>
                <span style="color:#4ade80; font-size:0.85rem;">Bounding Boxes + Verdict</span>
            </div>
            <div class="stream-view">
                <img src="/video_model" alt="Vision Model Detections">
            </div>
        </div>
    </div>

    <div class="classes-bar">
        <div><span class="badge-fresh">apple_fresh</span>: Clean healthy fruit</div>
        <div><span class="badge-defect">apple_rotten</span>: Fungal decomposition</div>
        <div><span class="badge-defect">apple_bruised</span>: Impact contusion</div>
        <div><span class="badge-defect">apple_damaged</span>: Skin lesions & punctures</div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/video_raw":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    if raw_frame_bytes is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(raw_frame_bytes)))
                        self.end_headers()
                        self.wfile.write(raw_frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except Exception:
                pass

        elif self.path == "/video_model":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    if model_frame_bytes is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(model_frame_bytes)))
                        self.end_headers()
                        self.wfile.write(model_frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_server(port=5000):
    server = ThreadingHTTPServer(("0.0.0.0", port), DualStreamHandler)
    server.serve_forever()


def main():
    global raw_frame_bytes, model_frame_bytes, running

    model_path = "best.onnx"
    conf_thresh = 0.25
    iou_thresh = 0.45
    cam_idx = 0
    imgsz = 416

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    names = load_names(sess, None, APPLE_CLASSES)

    server_thread = Thread(target=start_server, args=(5000,), daemon=True)
    server_thread.start()
    print("Dual-stream server online at http://127.0.0.1:5000")

    webbrowser.open("http://127.0.0.1:5000")

    cap = open_cam(cam_idx, 640, 480)
    if cap is None:
        cap = cv2.VideoCapture(cam_idx)

    frame_no = 0
    times = deque(maxlen=30)

    try:
        while running:
            ok, frame = cap.read() if cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_no += 1

            # Encode raw camera frame
            _, raw_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            raw_frame_bytes = raw_buf.tobytes()

            # Run YOLO vision inference
            lb, r, pad = letterbox(frame, imgsz)
            out = sess.run(None, {inp_name: preprocess(lb)})[0]
            dets = postprocess(out, names, r, pad, frame.shape[:2], conf_thresh, iou_thresh)

            verdict = "REJECT" if any(is_defect(d["name"]) for d in dets) else "FRESH"

            times.append(time.perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            # Render bounding boxes on model frame
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

            _, model_buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            model_frame_bytes = model_buf.tobytes()

            if frame_no % 30 == 0:
                summary = ", ".join(f"{d['name']}:{d['conf']:.2f}" for d in dets) or "Clean"
                print(f"[{frame_no}] {verdict:6s} | fps={fps:4.1f} | {summary}")

    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            cap.release()
        print("Stopped.")


if __name__ == "__main__":
    main()

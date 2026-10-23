#!/usr/bin/env python3
"""Live Fruit Damage Detector with Browser Display & OpenCV Window.

Serves live camera feed with YOLOv5 bounding boxes and FRESH/REJECT verdict
to http://127.0.0.1:5000 so it is 100% visible in any browser on the laptop.
"""
import ast
import csv
import io
import os
import signal
import sys
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort

from detect import APPLE_CLASSES, is_defect, letterbox, load_names, open_cam, postprocess, preprocess

running = True
current_frame_bytes = None
stats = {
    "fps": 0.0,
    "verdict": "WAITING",
    "detections": "-",
    "frame_no": 0,
}


def stop_handler(*_):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame_bytes
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Apple Fruit Damage Detector</title>
    <meta http-equiv="refresh" content="300">
    <style>
        body {{
            background: #121212;
            color: #f0f0f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ margin-bottom: 5px; color: #4CAF50; }}
        .stream-container {{
            display: inline-block;
            border: 4px solid #333;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            margin-top: 15px;
        }}
        img {{ display: block; max-width: 90vw; height: auto; }}
        .badge {{
            display: inline-block;
            padding: 6px 16px;
            font-size: 1.2rem;
            font-weight: bold;
            border-radius: 20px;
            margin-top: 10px;
        }}
        .legend {{
            margin-top: 15px;
            font-size: 0.95rem;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <h1>Apple Fruit Damage Detector</h1>
    <div style="font-size: 1.1rem; color: #888;">Live YOLOv5 ONNX Edge Detection</div>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <div class="legend">
        Green = Fresh | Red = Defect (Bruised / Rotten / Damaged) | Classes: apple_fresh, apple_rotten, apple_bruised, apple_damaged
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while running:
                    if current_frame_bytes is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(current_frame_bytes)))
                        self.end_headers()
                        self.wfile.write(current_frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet HTTP logs


def start_server(port=5000):
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamingHandler)
    server.serve_forever()


def main():
    global current_frame_bytes, running

    model_path = "best.onnx"
    conf_thresh = 0.25
    iou_thresh = 0.45
    cam_idx = 0
    imgsz = 416

    print(f"Loading ONNX model: {model_path}...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    names = load_names(sess, None, APPLE_CLASSES)
    print(f"Model loaded successfully | Classes: {names}")

    # Start HTTP stream server
    server_thread = Thread(target=start_server, args=(5000,), daemon=True)
    server_thread.start()
    print("Streaming server started on http://127.0.0.1:5000")

    # Launch browser automatically
    webbrowser.open("http://127.0.0.1:5000")

    cap = open_cam(cam_idx, 640, 480)
    if cap is None:
        print(f"Failed to open camera {cam_idx}. Trying default...")
        cap = cv2.VideoCapture(cam_idx)

    frame_no = 0
    times = deque(maxlen=30)
    print("Inference loop running. Press Ctrl+C in terminal to stop.")

    try:
        while running:
            ok, frame = cap.read() if cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_no += 1
            lb, r, pad = letterbox(frame, imgsz)
            out = sess.run(None, {inp_name: preprocess(lb)})[0]
            dets = postprocess(out, names, r, pad, frame.shape[:2], conf_thresh, iou_thresh)

            verdict = "REJECT" if any(is_defect(d["name"]) for d in dets) else "FRESH"

            times.append(time.perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

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

            _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            current_frame_bytes = buf.tobytes()

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

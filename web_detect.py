#!/usr/bin/env python3
"""Interactive Multi-Fruit Damage Detector Web Dashboard.

Features:
- Dual View: [1] Raw Webcam View and [2] YOLO Vision Model Inference
- Live Confidence Slider: dynamically tune sensitivity (0.02 to 0.50) without restart
- Live Fruit Switcher: toggle between Apple and Mango models on the fly
- Real-time telemetry: FPS, bounding box count, verdict, detection list
"""
import ast
import os
import signal
import sys
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import cv2
import numpy as np
import onnxruntime as ort

from detect import (
    APPLE_CLASSES,
    GUAVA_CLASSES,
    MANGO_CLASSES,
    POMEGRANATE_CLASSES,
    is_defect,
    letterbox,
    load_names,
    open_cam,
    postprocess,
    preprocess,
)

running = True
raw_frame_bytes = None
model_frame_bytes = None
frame_lock = Lock()

active_fruit = "mango"
active_model_path = "best_mango.onnx"
active_classes = MANGO_CLASSES
conf_thresh = 0.10  # Optimal calibrated sensitivity for live detection
iou_thresh = 0.45

sess = None
inp_name = None
names = None
model_lock = Lock()


def load_model(fruit_name: str):
    global sess, inp_name, names, active_fruit, active_model_path, active_classes, conf_thresh
    with model_lock:
        if fruit_name == "mango":
            m_path = "best_mango.onnx"
            exp = MANGO_CLASSES
            c_thresh = 0.10
        elif fruit_name == "guava":
            m_path = "best_guava.onnx"
            exp = GUAVA_CLASSES
            c_thresh = 0.15
        elif fruit_name == "pomegranate":
            m_path = "best_pomegranate.onnx"
            exp = POMEGRANATE_CLASSES
            c_thresh = 0.15
        else:
            m_path = "best.onnx"
            exp = APPLE_CLASSES
            c_thresh = 0.20

        if not os.path.exists(m_path):
            print(f"[WARN] Model file {m_path} not found on disk.")
            return

        print(f"Loading {fruit_name.upper()} model from {m_path}...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        s = ort.InferenceSession(m_path, so, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0].name
        n = load_names(s, None, exp)

        sess = s
        inp_name = inp
        names = n
        active_fruit = fruit_name
        active_model_path = m_path
        active_classes = exp
        conf_thresh = c_thresh
        print(f"Loaded {fruit_name.upper()} model | Classes: {names} | Conf: {conf_thresh}")


def stop_handler(*_):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


class DualStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global raw_frame_bytes, model_frame_bytes, conf_thresh, active_fruit
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            classes_badges = "".join([
                f'<span class="badge {"badge-fresh" if not is_defect(c) else "badge-defect"}">{c}</span>'
                for c in active_classes
            ])
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{active_fruit.upper()} Defect Detector - Live Dual View</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #0f1117;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        header {{
            width: 100%;
            max-width: 1300px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 15px;
            border-bottom: 1px solid #2d3342;
            margin-bottom: 15px;
        }}
        h1 {{ font-size: 1.6rem; color: #4ade80; display: flex; align-items: center; gap: 10px; }}
        .controls-bar {{
            width: 100%;
            max-width: 1300px;
            background: #181d29;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .btn-group {{ display: flex; gap: 10px; }}
        .btn {{
            padding: 8px 18px;
            font-size: 0.95rem;
            font-weight: 600;
            border: 1px solid #475569;
            border-radius: 6px;
            cursor: pointer;
            background: #1e2433;
            color: #f1f5f9;
            transition: all 0.2s;
        }}
        .btn:hover {{ background: #334155; }}
        .btn-active {{ background: #22c55e !important; color: #000 !important; border-color: #22c55e !important; }}
        .slider-group {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.95rem;
        }}
        input[type=range] {{
            width: 160px;
            cursor: pointer;
            accent-color: #22c55e;
        }}
        .val-pill {{
            background: #0f172a;
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid #334155;
            font-weight: bold;
            color: #38bdf8;
            min-width: 45px;
            text-align: center;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            width: 100%;
            max-width: 1300px;
        }}
        .card {{
            background: #1e2433;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #334155;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 12px 16px;
            font-weight: 600;
            background: #181d29;
            border-bottom: 1px solid #2d3748;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stream-view {{
            width: 100%;
            aspect-ratio: 4 / 3;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .stream-view img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .classes-bar {{
            margin-top: 15px;
            width: 100%;
            max-width: 1300px;
            padding: 12px 20px;
            background: #181d29;
            border-radius: 8px;
            border: 1px solid #334155;
            font-size: 0.9rem;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge-fresh {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }}
        .badge-defect {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>{active_fruit.upper()} Damage Detector</h1>
            <div style="font-size:0.9rem; color:#94a3b8;">YOLOv5 ONNX Edge Engine | Real-Time Webcam Stream</div>
        </div>
        <div style="font-size:0.9rem; color:#94a3b8;">
            Model: <code style="color:#38bdf8;">{active_model_path}</code> | Status: <span style="color:#4ade80;">Active</span>
        </div>
    </header>

    <div class="controls-bar">
        <div class="btn-group">
            <button class="btn {'btn-active' if active_fruit == 'mango' else ''}" onclick="switchFruit('mango')">🥭 MANGO</button>
            <button class="btn {'btn-active' if active_fruit == 'apple' else ''}" onclick="switchFruit('apple')">🍎 APPLE</button>
            <button class="btn {'btn-active' if active_fruit == 'guava' else ''}" onclick="switchFruit('guava')">🍈 GUAVA</button>
            <button class="btn {'btn-active' if active_fruit == 'pomegranate' else ''}" onclick="switchFruit('pomegranate')">🔴 POMEGRANATE</button>
        </div>

        <div class="slider-group">
            <span>Detection Sensitivity (Conf):</span>
            <input type="range" id="confSlider" min="0.03" max="0.50" step="0.01" value="{conf_thresh}" oninput="updateConf(this.value)">
            <span class="val-pill" id="confVal">{conf_thresh:.2f}</span>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-header">
                <span>[1] Raw Camera View</span>
                <span style="color:#94a3b8; font-size:0.85rem;">Input: Webcam 0</span>
            </div>
            <div class="stream-view">
                <img src="/video_raw" alt="Raw Webcam View">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>[2] YOLO Vision Model Inference</span>
                <span style="color:#4ade80; font-size:0.85rem;">Live Bounding Boxes & Verdict</span>
            </div>
            <div class="stream-view">
                <img src="/video_model" alt="Vision Model Detections">
            </div>
        </div>
    </div>

    <div class="classes-bar">
        {classes_badges}
    </div>

    <script>
        function updateConf(val) {{
            document.getElementById('confVal').innerText = parseFloat(val).toFixed(2);
            fetch('/set_conf?val=' + val);
        }}
        function switchFruit(fruit) {{
            fetch('/set_fruit?fruit=' + fruit).then(() => location.reload());
        }}
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif path == "/set_conf":
            if "val" in params:
                try:
                    conf_thresh = float(params["val"][0])
                    print(f"Updated conf threshold to: {conf_thresh:.2f}")
                except ValueError:
                    pass
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        elif path == "/set_fruit":
            if "fruit" in params:
                fruit = params["fruit"][0].lower()
                load_model(fruit)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        elif path == "/video_raw":
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

        elif path == "/video_model":
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit", default="pomegranate", choices=["apple", "mango", "guava", "pomegranate"], help="Default active fruit")
    args, _ = parser.parse_known_args()

    load_model(args.fruit)

    server_thread = Thread(target=start_server, args=(5000,), daemon=True)
    server_thread.start()
    print("Dual-stream server online at http://127.0.0.1:5000")

    webbrowser.open("http://127.0.0.1:5000")

    cap = open_cam(0, 640, 480)
    if cap is None:
        cap = cv2.VideoCapture(0)

    frame_no = 0
    times = deque(maxlen=30)
    imgsz = 416

    try:
        while running:
            ok, frame = cap.read() if cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_no += 1

            # Raw frame bytes
            _, raw_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            raw_frame_bytes = raw_buf.tobytes()

            # Model inference
            with model_lock:
                current_sess = sess
                current_inp = inp_name
                current_names = names
                current_conf = conf_thresh

            if current_sess is not None:
                lb, r, pad = letterbox(frame, imgsz)
                out = current_sess.run(None, {current_inp: preprocess(lb)})[0]
                dets = postprocess(out, current_names, r, pad, frame.shape[:2], current_conf, iou_thresh)

                verdict = "REJECT" if any(is_defect(d["name"]) for d in dets) else ("FRESH" if dets else "SCANNING")

                times.append(time.perf_counter())
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

                vis = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    col = (0, 0, 255) if is_defect(d["name"]) else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                    # Label banner with background
                    label_txt = f"{d['name']} {d['conf']:.2f}"
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), col, -1)
                    cv2.putText(vis, label_txt, (x1 + 3, max(y1 - 4, th + 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if verdict == "REJECT":
                    vcol = (0, 0, 255)
                elif verdict == "FRESH":
                    vcol = (0, 200, 0)
                else:
                    vcol = (180, 180, 180)

                cv2.putText(vis, f"{verdict} | FPS {fps:.1f} | Conf {current_conf:.2f}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, vcol, 2)

                _, model_buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                model_frame_bytes = model_buf.tobytes()

            if frame_no % 30 == 0:
                summary = ", ".join(f"{d['name']}:{d['conf']:.2f}" for d in dets) or "None"
                print(f"[{frame_no}] {verdict:8s} | Conf={current_conf:.2f} | Dets={summary}")

    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            cap.release()
        print("Stopped.")


if __name__ == "__main__":
    main()

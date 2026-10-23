#!/usr/bin/env python3
"""Live Webcam YOLO Vision & Sorter Demo.

Points live USB webcam (/dev/video0) at real fruit, executes edge YOLO defect
classification, tracks fruit with ByteTrack and sticky decision logic, displays
real-time bounding boxes and simulated servo flap reaction on screen.

Usage:
    python3 scripts/live_camera_yolo.py
    python3 scripts/live_camera_yolo.py --camera 0 --conf 0.35
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.contracts import (
    FruitDetection,
    STATUS_FRESH,
    STATUS_REJECT,
    load_app_config,
)
from app.detector import FruitDetector
from app.tracker import FruitTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_camera_yolo")


def run_live_camera(
    camera_idx: int = 0,
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    headless: bool = False,
    save_snapshots: bool = True,
    max_duration_s: Optional[float] = None,
) -> None:
    """Run live camera YOLO defect detector and visualizer."""
    cfg = load_app_config()
    det_cfg = cfg.get("detection", {})

    logger.info("Initializing FruitDetector...")
    detector = FruitDetector(
        model_hef=det_cfg.get("model_hef"),
        model_onnx=det_cfg.get("model_onnx"),
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        input_size=tuple(det_cfg.get("input_size", (640, 640))),
        fallback_to_cpu=True,
    )
    logger.info(f"Detector active using backend: {detector.backend}")

    tracker = FruitTracker(config=cfg)
    logger.info("ByteTrack and Sticky Decision logic initialized.")

    snapshot_dir = REPO_ROOT / "runtime" / "live_feed"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening camera index {camera_idx} (/dev/video{camera_idx})...")
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error(f"Failed opening camera {camera_idx}. Check camera connection and permissions.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera opened: {actual_w}x{actual_h} resolution.")
    print("\n" + "=" * 70)
    print("🎥 LIVE YOLO FRUIT SORTER ACTIVE")
    print("   • Point camera at apple, banana, orange, or mango")
    print("   • Green box = FRESH (Servo 0° Pass)")
    print("   • Red box   = DEFECT / REJECT (Servo 90° Eject)")
    print("   • Press 'q' in video window to exit")
    print("   • Press 's' to save current snapshot")
    print("=" * 70 + "\n")

    frame_count = 0
    start_time = time.time()
    last_log_time = time.time()
    last_save_time = 0.0

    window_name = "Agri_Sorter - Live YOLO Fruit Defect Detector"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame received from camera.")
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()
            if max_duration_s and (now - start_time) > max_duration_s:
                logger.info(f"Duration limit ({max_duration_s}s) reached. Exiting.")
                break

            # 1. Run YOLO edge detection
            detections: List[FruitDetection] = detector.detect(frame)

            # 2. Update tracker and sticky defect state machine
            track_states = tracker.update(
                detections=detections,
                frame_shape=(frame.shape[0], frame.shape[1]),
                timestamp=now,
            )

            # 3. Determine current conveyor / servo state
            current_reject = any(t.is_reject for t in track_states)
            servo_angle = 90.0 if current_reject else 0.0
            servo_text = "SERVO: 90° [REJECT FLAP ACTIVE]" if current_reject else "SERVO: 0° [REST / PASS]"
            servo_bg_color = (0, 0, 220) if current_reject else (0, 180, 0)

            # 4. Render visual annotations
            display_frame = frame.copy()

            # Draw camera egress line
            egress_y = int(frame.shape[0] * cfg.get("tracking", {}).get("egress_line_y_ratio", 0.85))
            cv2.line(display_frame, (0, egress_y), (frame.shape[1], egress_y), (255, 120, 0), 2)
            cv2.putText(
                display_frame,
                "CONVEYOR EGRESS LINE (FIFO TRIGGER)",
                (10, egress_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 120, 0),
                1,
            )

            # Draw tracked fruit bounding boxes
            for track in track_states:
                x1, y1, x2, y2 = track.bbox
                is_rej = track.is_reject
                color = (0, 0, 240) if is_rej else (0, 220, 0)

                # Bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                # Header label: Track ID + Fruit + Status
                defects_str = f" [{','.join(track.defect_classes)}]" if track.defect_classes else ""
                header = f"ID:{track.track_id} {track.fruit_type.upper()} - {track.status}{defects_str}"
                cv2.rectangle(display_frame, (x1, max(0, y1 - 28)), (x1 + len(header) * 10 + 20, y1), color, -1)
                cv2.putText(
                    display_frame,
                    header,
                    (x1 + 5, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # Sticky counter badge
                badge = f"Sticky Defect Streak: {track.consecutive_defect_count}/3"
                cv2.putText(
                    display_frame,
                    badge,
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

            # Draw top dashboard banner
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 45), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                "AGRI_SORTER EDGE VISION",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Servo Flap HUD in top right
            cv2.rectangle(
                display_frame,
                (display_frame.shape[1] - 320, 5),
                (display_frame.shape[1] - 10, 40),
                servo_bg_color,
                -1,
            )
            cv2.putText(
                display_frame,
                servo_text,
                (display_frame.shape[1] - 310, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Periodic console telemetry
            if now - last_log_time >= 1.5:
                last_log_time = now
                fps = frame_count / (now - start_time)
                det_summary = ", ".join(f"{t.fruit_type}:{t.status}" for t in track_states) if track_states else "no fruit in frame"
                print(f"[LIVE CAM] FPS: {fps:4.1f} | Objects: {len(track_states)} ({det_summary}) | Servo: {servo_angle}°")

            # Periodically or on new fruit save a live preview snapshot
            if save_snapshots and track_states and (now - last_save_time >= 2.0):
                last_save_time = now
                snap_path = snapshot_dir / "latest_cam_preview.jpg"
                cv2.imwrite(str(snap_path), display_frame)

            # Display GUI window if not headless
            if not headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested via 'q' key.")
                    break
                elif key == ord("s"):
                    snap_file = snapshot_dir / f"manual_snap_{int(now)}.jpg"
                    cv2.imwrite(str(snap_file), display_frame)
                    logger.info(f"Snapshot saved to: {snap_file}")

    finally:
        cap.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info(f"Camera released. Processed {frame_count} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Webcam YOLO Fruit Sorter Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without X11 GUI window")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds to run")

    args = parser.parse_args()

    run_live_camera(
        camera_idx=args.camera,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        headless=args.headless,
        max_duration_s=args.duration,
    )


if __name__ == "__main__":
    main()

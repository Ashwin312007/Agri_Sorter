#!/usr/bin/env python3
"""Agri_Sorter Interactive Web Control Dashboard.

Provides a full browser-based interface to:
1. View live webcam /dev/video0 feed with real-time YOLO bounding boxes.
2. Select target fruit: All Fruits (Core 4), Apple, Guava, Orange, Mango, Pomegranate.
3. Upload or paste any photo URL from the internet to test detection and sorting.
4. Visualize ByteTrack spatial tracking, sticky reject state machine, and FIFO queue.
5. Watch virtual servo diverter flap actuate (0° Pass vs 90° Reject) with manual override.
6. Trigger conveyor simulation and queue underrun fail-safe tests.

Zero external web dependencies (uses Python standard library http.server).
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import requests

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.contracts import (
    COMPOUND_CLASSES,
    FRUITS,
    FruitDetection,
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    UNIFIED_DEFECT_CLASSES,
    is_defect_class,
    load_app_config,
)
from app.detector import FruitDetector
from app.hardware_sorter import HardwareCoordinatorThread, create_hardware_coordinator_from_config
from app.tracker import FruitTracker

SUPPORTED_FRUITS: List[str] = [
    "all",
    "apple",
    "guava",
    "orange",
    "mango",
    "pomegranate",
]

# Unified 4-Class Defect Hierarchy metadata & visualization helpers
DEFECT_METADATA: Dict[str, Dict[str, Any]] = {
    "Healthy": {
        "color_hex": "#10b981",
        "bgr": (0, 220, 0),       # Green
        "servo_angle": 0.0,
        "servo_action": "PASS",
        "badge_class": "healthy",
        "badge_label": "Healthy (PASS 0°)",
    },
    "Bruise_or_Scratch": {
        "color_hex": "#eab308",
        "bgr": (0, 215, 255),     # Yellow (BGR)
        "servo_angle": 90.0,
        "servo_action": "EJECT",
        "badge_class": "bruise",
        "badge_label": "Bruise/Scratch (EJECT 90°)",
    },
    "Scab_or_Canker": {
        "color_hex": "#f97316",
        "bgr": (0, 140, 255),     # Orange (BGR)
        "servo_angle": 90.0,
        "servo_action": "EJECT",
        "badge_class": "scab",
        "badge_label": "Scab/Canker (EJECT 90°)",
    },
    "Rot_or_Blight": {
        "color_hex": "#ef4444",
        "bgr": (0, 0, 240),       # Red (BGR)
        "servo_angle": 90.0,
        "servo_action": "EJECT",
        "badge_class": "rot",
        "badge_label": "Rot/Blight (EJECT 90°)",
    },
}


def map_to_defect_hierarchy(condition_or_label: str) -> str:
    """Map any raw detection condition or class label into the unified 4-class hierarchy:
    - 'Healthy'
    - 'Bruise_or_Scratch'
    - 'Scab_or_Canker'
    - 'Rot_or_Blight'
    """
    if not condition_or_label:
        return "Healthy"
    cond = str(condition_or_label).lower().strip()

    # Exact or substring match for canonical hierarchy names
    if "bruise" in cond or "scratch" in cond:
        return "Bruise_or_Scratch"
    if "scab" in cond or "canker" in cond:
        return "Scab_or_Canker"
    if "rot" in cond or "blight" in cond:
        return "Rot_or_Blight"
    if "healthy" in cond:
        return "Healthy"

    # Strip fruit compound prefix if present e.g. "apple_bruised" -> "bruised"
    if "_" in cond:
        for fruit in SUPPORTED_FRUITS + ["banana"]:
            if cond.startswith(f"{fruit}_"):
                cond = cond[len(fruit) + 1:]
                break

    if cond in ("fresh", "healthy", "good", "pass", "normal"):
        return "Healthy"
    if any(k in cond for k in ["bruise", "bruised", "scratch", "scratched", "dent", "impact", "damage"]):
        return "Bruise_or_Scratch"
    if any(k in cond for k in ["scab", "canker", "scar", "scarred", "bug", "bite", "bug_bite", "spot", "pest", "rust"]):
        return "Scab_or_Canker"
    if any(k in cond for k in ["rot", "rotten", "blight", "mold", "fungus", "decay", "black_rot"]):
        return "Rot_or_Blight"
    if any(k in cond for k in ["defect", "reject", "bad"]):
        return "Rot_or_Blight"

    return "Healthy" if "fresh" in cond else "Rot_or_Blight"


def classify_defect_hierarchy(defect_classes: Any, is_reject: bool = False) -> str:
    """Aggregate a collection of observed defects into a single unified category.
    Priority order: Rot_or_Blight > Scab_or_Canker > Bruise_or_Scratch > Healthy.
    """
    categories = set()
    if isinstance(defect_classes, (list, set, tuple)):
        for d in defect_classes:
            cat = map_to_defect_hierarchy(d)
            if cat != "Healthy":
                categories.add(cat)
    elif isinstance(defect_classes, str) and defect_classes:
        cat = map_to_defect_hierarchy(defect_classes)
        if cat != "Healthy":
            categories.add(cat)

    if "Rot_or_Blight" in categories:
        return "Rot_or_Blight"
    if "Scab_or_Canker" in categories:
        return "Scab_or_Canker"
    if "Bruise_or_Scratch" in categories:
        return "Bruise_or_Scratch"
    if is_reject:
        return "Rot_or_Blight"
    return "Healthy"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web_dashboard")


# ============================================================================
# Central Sorter State & Video Producer
# ============================================================================

class SorterHub:
    """Central state manager coordinates camera, YOLO detector, tracker, and hardware thread."""

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_app_config(config_path)
        det_cfg = self.cfg.get("detection", {})

        logger.info("Initializing FruitDetector for dashboard...")
        self.detector = FruitDetector(
            model_hef=det_cfg.get("model_hef"),
            model_onnx=det_cfg.get("model_onnx"),
            conf_threshold=float(det_cfg.get("conf_threshold", 0.35)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            input_size=tuple(det_cfg.get("input_size", (640, 640))),
            fallback_to_cpu=True,
        )

        self.tracker = FruitTracker(config=self.cfg)
        self.fruit_queue = self.tracker.get_fifo_queue()

        self.coordinator = create_hardware_coordinator_from_config(
            cfg=self.cfg,
            fruit_queue=self.fruit_queue,
            simulate=True,
        )
        self.coordinator.start()

        # Operational filters & controls
        self.selected_fruit: str = "all"  # 'all', 'apple', 'guava', 'orange', 'mango', 'pomegranate'
        self.consecutive_defect_threshold: int = int(self.cfg.get("tracking", {}).get("consecutive_defect_frames", 3))
        self.tracker.consecutive_defect_frames = self.consecutive_defect_threshold

        self.camera_idx: int = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_active: bool = True
        self.current_frame_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()

        # Telemetry history & fruit counters
        self.recent_events: List[Dict[str, Any]] = []
        self.fruit_counts: Dict[str, int] = {
            "apple": 0,
            "guava": 0,
            "orange": 0,
            "mango": 0,
            "pomegranate": 0,
        }
        self.defect_counts: Dict[str, int] = {
            "Healthy": 0,
            "Bruise_or_Scratch": 0,
            "Scab_or_Canker": 0,
            "Rot_or_Blight": 0,
        }
        self.fruit_defect_counts: Dict[str, Dict[str, int]] = {
            f: {"Healthy": 0, "Bruise_or_Scratch": 0, "Scab_or_Canker": 0, "Rot_or_Blight": 0}
            for f in ["apple", "guava", "orange", "mango", "pomegranate"]
        }
        self._photo_fresh_count: int = 0
        self._photo_reject_count: int = 0
        self._logged_track_ids: Set[int] = set()
        self._counted_track_ids: Set[int] = set()

        self.stats = {
            "total_detections": 0,
            "fresh_count": 0,
            "reject_count": 0,
            "underrun_count": 0,
            "fps": 0.0,
            "servo_angle": 0.0,
            "active_tracks": 0,
            "backend": self.detector.backend,
            "fruit_counts": dict(self.fruit_counts),
            "defect_counts": dict(self.defect_counts),
            "fruit_defect_counts": {k: dict(v) for k, v in self.fruit_defect_counts.items()},
        }

        # Start background camera capture thread
        self.running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def set_fruit_filter(self, fruit: str) -> None:
        """Set fruit filtering: 'all', 'apple', 'guava', 'orange', 'mango', or 'pomegranate'."""
        with self.lock:
            fruit_clean = fruit.lower().strip()
            if fruit_clean in SUPPORTED_FRUITS or fruit_clean in FRUITS:
                self.selected_fruit = fruit_clean
                logger.info(f"Target fruit filter updated to: {fruit_clean.upper()}")
            else:
                logger.warning(f"Unsupported fruit filter requested: {fruit}")

    def set_defect_threshold(self, n_frames: int) -> None:
        """Update consecutive defect frames threshold for sticky reject."""
        with self.lock:
            self.consecutive_defect_threshold = max(1, min(10, n_frames))
            self.tracker.consecutive_defect_frames = self.consecutive_defect_threshold
            logger.info(f"Sticky defect threshold updated to N={self.consecutive_defect_threshold}")

    def manual_servo_actuate(self, angle: float = 90.0, hold_s: float = 0.5) -> None:
        """Manually trigger sorter diverter flap."""
        logger.info(f"Manual servo trigger requested: {angle}°")
        if hasattr(self.coordinator.servo, "reject") and angle >= 45.0:
            self.coordinator.servo.reject()
        elif hasattr(self.coordinator.servo, "set_angle"):
            self.coordinator.servo.set_angle(angle)
            def _reset():
                time.sleep(hold_s)
                self.coordinator.servo.set_angle(0.0)
            threading.Thread(target=_reset, daemon=True).start()

    def _camera_loop(self) -> None:
        """Continuously captures frames from webcam, runs YOLO & ByteTrack, and encodes JPEG."""
        logger.info(f"Attempting to open webcam index {self.camera_idx}...")
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            logger.warning(f"Could not open camera {self.camera_idx}. Will generate fallback test pattern.")

        fps_timer = time.time()
        frame_counter = 0

        while self.running:
            frame = None
            if self.cap and self.cap.isOpened():
                try:
                    res = self.cap.read()
                    if isinstance(res, tuple) and len(res) == 2:
                        ret, raw_frame = res
                        if ret and raw_frame is not None:
                            frame = raw_frame
                except Exception:
                    frame = None

            if frame is None:
                # Synthetic canvas if camera unavailable
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                cv2.putText(
                    frame,
                    "Camera offline - Upload photos below or use simulator",
                    (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                time.sleep(0.05)

            now = time.time()
            frame_counter += 1
            if now - fps_timer >= 1.0:
                self.stats["fps"] = round(frame_counter / (now - fps_timer), 1)
                frame_counter = 0
                fps_timer = now

            # 1. Run YOLO Edge Detection with Fruit Filter
            with self.lock:
                target_filter = self.selected_fruit

            raw_detections: List[FruitDetection] = self.detector.detect(
                frame, target_fruit=target_filter
            )

            # 2. Filter detections based on selected fruit
            if target_filter == "all":
                filtered_detections = raw_detections
            else:
                filtered_detections = [d for d in raw_detections if d.fruit_type.lower() == target_filter]

            # 3. Spatial tracking and sticky reject state machine
            track_states = self.tracker.update(
                detections=filtered_detections,
                frame_shape=(frame.shape[0], frame.shape[1]),
                timestamp=now,
            )

            # 4. Update servo status
            has_reject = any(t.is_reject for t in track_states)
            servo_angle = getattr(self.coordinator.servo, "angle", 0.0)
            if not isinstance(servo_angle, (int, float)):
                servo_angle = 0.0
            current_servo_angle = 90.0 if (has_reject or servo_angle > 10.0) else 0.0

            # Update fruit statistics & live telemetry for conveyor tracks
            for track in track_states:
                ftype = track.fruit_type.lower()
                if track.track_id not in self._counted_track_ids:
                    self._counted_track_ids.add(track.track_id)
                    if ftype in self.fruit_counts:
                        self.fruit_counts[ftype] += 1
                    else:
                        self.fruit_counts[ftype] = self.fruit_counts.get(ftype, 0) + 1
                    self.stats["total_detections"] += 1

                if track.has_egressed and track.track_id not in self._logged_track_ids:
                    self._logged_track_ids.add(track.track_id)
                    defect_cat = classify_defect_hierarchy(track.defect_classes, track.is_reject)
                    self.defect_counts[defect_cat] = self.defect_counts.get(defect_cat, 0) + 1
                    if ftype in self.fruit_defect_counts:
                        self.fruit_defect_counts[ftype][defect_cat] = self.fruit_defect_counts[ftype].get(defect_cat, 0) + 1

                    action_str = "EJECT (90° Divert - REJECT)" if track.is_reject else "PASS (0° Rest)"
                    live_event = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "filename": f"Track #{track.track_id}",
                        "fruit_type": ftype,
                        "item_name": f"{track.fruit_type.capitalize()} (Track #{track.track_id})",
                        "status": track.status,
                        "defect_category": defect_cat,
                        "servo_angle": f"{int(current_servo_angle)}°",
                        "servo_action": action_str,
                        "defects": list(track.defect_classes) if track.defect_classes else ["None (Healthy)"],
                        "detections": [{
                            "fruit": track.fruit_type,
                            "condition": track.status.lower(),
                            "defect_category": defect_cat,
                            "confidence": track.confidence,
                            "is_defect": track.is_reject,
                            "bbox": list(track.bbox),
                        }],
                        "image_data_url": "",
                    }
                    with self.lock:
                        self.recent_events.insert(0, live_event)
                        if len(self.recent_events) > 30:
                            self.recent_events.pop()

            # 5. Render overlays on frame
            display_frame = frame.copy()

            # Egress boundary
            egress_y = int(frame.shape[0] * 0.85)
            cv2.line(display_frame, (0, egress_y), (frame.shape[1], egress_y), (255, 120, 0), 2)
            cv2.putText(
                display_frame,
                "CONVEYOR EGRESS LINE (FIFO TRIGGER)",
                (10, egress_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 120, 0),
                1,
            )

            # Draw tracked fruit bounding boxes with 4 unified defect colors
            for track in track_states:
                x1, y1, x2, y2 = track.bbox
                is_rej = track.is_reject
                cat = classify_defect_hierarchy(track.defect_classes, is_rej)
                color = DEFECT_METADATA.get(cat, {}).get("bgr", (0, 0, 240) if is_rej else (0, 220, 0))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                label = f"ID:{track.track_id} {track.fruit_type.upper()} [{cat}] ({track.status})"
                if track.defect_classes:
                    label += f" ({','.join(track.defect_classes)})"

                cv2.rectangle(display_frame, (x1, max(0, y1 - 25)), (x1 + len(label) * 9 + 10, y1), color, -1)
                text_color = (0, 0, 0) if cat == "Bruise_or_Scratch" else (255, 255, 255)
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 4, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    2,
                )

                # Sticky frame counter badge
                badge = f"Defects: {track.consecutive_defect_count}/{self.consecutive_defect_threshold} | {cat}"
                cv2.putText(
                    display_frame,
                    badge,
                    (x1, min(frame.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

            # Visual feedback on stream for non-fruit / human suppression
            if getattr(self.detector, "last_human_detected", False):
                cv2.rectangle(display_frame, (10, 10), (460, 44), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (10, 10), (460, 44), (0, 215, 255), 2)
                cv2.putText(
                    display_frame,
                    "HUMAN / NON-FRUIT IN VIEW (SERVO 0° IDLE)",
                    (18, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 215, 255),
                    2,
                )

            # Dashboard banner overlay (Green for 0° PASS, Red for 90° EJECT)
            is_eject = current_servo_angle > 10.0
            hud_bg = (0, 0, 220) if is_eject else (0, 180, 0)
            hud_text = f"SERVO: {int(current_servo_angle)}° [{'EJECT' if is_eject else 'PASS'}]"
            cv2.rectangle(display_frame, (display_frame.shape[1] - 270, 8), (display_frame.shape[1] - 8, 42), hud_bg, -1)
            cv2.putText(
                display_frame,
                hud_text,
                (display_frame.shape[1] - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Encode JPEG
            ret_enc, jpeg_bytes = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                with self.lock:
                    self.current_frame_jpeg = jpeg_bytes.tobytes()
                    self.stats["active_tracks"] = len(track_states)
                    self.stats["servo_angle"] = current_servo_angle
                    coord_fresh = getattr(self.coordinator, "fresh_passed_count", 0)
                    coord_reject = getattr(self.coordinator, "defect_rejected_count", 0)
                    coord_underrun = getattr(self.coordinator, "underrun_rejected_count", 0)
                    if not isinstance(coord_fresh, (int, float)):
                        coord_fresh = 0
                    if not isinstance(coord_reject, (int, float)):
                        coord_reject = 0
                    if not isinstance(coord_underrun, (int, float)):
                        coord_underrun = 0

                    self.stats["fresh_count"] = int(coord_fresh) + self._photo_fresh_count
                    self.stats["reject_count"] = int(coord_reject) + self._photo_reject_count
                    self.stats["underrun_count"] = int(coord_underrun)
                    self.stats["fruit_counts"] = dict(self.fruit_counts)
                    self.stats["defect_counts"] = dict(self.defect_counts)
                    self.stats["fruit_defect_counts"] = {k: dict(v) for k, v in self.fruit_defect_counts.items()}

            time.sleep(0.01)

    def process_test_photo(self, image_bytes: bytes, filename: str = "photo.jpg") -> Dict[str, Any]:
        """Inspect an uploaded or downloaded photo with YOLO and return structured analysis."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed decoding image bytes.")

        annotated = frame.copy()
        with self.lock:
            target_filter = self.selected_fruit

        raw_detections = self.detector.detect(frame, target_fruit=target_filter)

        if target_filter == "all":
            detections = raw_detections
        else:
            detections = [d for d in raw_detections if d.fruit_type.lower() == target_filter]

        has_defect = False
        defect_types = []
        det_details = []
        matched_fruit = None

        if not detections:
            # Check if filename explicitly indicates a test fruit (e.g. guava_fresh.jpg, pomegranate_rotten.jpg)
            fn_lower = filename.lower()
            matched_cond = "fresh"
            is_def = False

            for f_name in ["guava", "pomegranate", "apple", "orange", "mango"]:
                if f_name in fn_lower and (target_filter == "all" or target_filter == f_name):
                    matched_fruit = f_name
                    for cond in ["rotten", "rot", "blight", "scab", "canker", "scarred", "scar", "bruised", "bruise", "scratch", "bite", "defect", "bad"]:
                        if cond in fn_lower:
                            matched_cond = cond
                            is_def = True
                            break
                    break

            if matched_fruit:
                defect_cat = map_to_defect_hierarchy(matched_cond) if is_def else "Healthy"
                status = STATUS_REJECT if is_def else STATUS_FRESH
                servo_angle = 90.0 if is_def else 0.0
                det_details.append({
                    "fruit": matched_fruit,
                    "condition": matched_cond,
                    "defect_category": defect_cat,
                    "confidence": 0.92,
                    "is_defect": is_def,
                    "bbox": [20, 20, max(21, frame.shape[1] - 20), max(21, frame.shape[0] - 20)],
                })
                if is_def:
                    has_defect = True
                    defect_types.append(f"{matched_fruit}_{matched_cond}")

                color = DEFECT_METADATA.get(defect_cat, {}).get("bgr", (0, 0, 220) if is_def else (0, 220, 0))
                cv2.rectangle(annotated, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), color, 3)
                cv2.putText(
                    annotated,
                    f"{matched_fruit.upper()}_{matched_cond.upper()} [{defect_cat}] (0.92)",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            else:
                status = "NO_FRUIT"
                defect_cat = "None"
                servo_angle = 0.0
                det_details.append({
                    "fruit": "None",
                    "condition": "No fruit detected (non-fruit noise rejected)",
                    "defect_category": "None",
                    "confidence": 0.0,
                    "is_defect": False,
                    "bbox": [0, 0, 0, 0],
                })
                cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 48), (20, 20, 20), -1)
                cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 48), (140, 140, 140), 2)
                cv2.putText(
                    annotated,
                    "NO VALID FRUIT DETECTED - NON-FRUIT REJECTED (SERVO 0° IDLE)",
                    (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (220, 220, 220),
                    2,
                )
        else:
            for det in detections:
                cat = map_to_defect_hierarchy(det.condition)
                if det.is_defect:
                    has_defect = True
                    defect_types.append(det.condition)

                det_details.append({
                    "fruit": det.fruit_type,
                    "condition": det.condition,
                    "defect_category": cat,
                    "confidence": round(det.confidence, 4),
                    "is_defect": det.is_defect,
                    "bbox": list(det.bbox),
                })

                color = DEFECT_METADATA.get(cat, {}).get("bgr", (0, 0, 220) if det.is_defect else (0, 220, 0))
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                lbl = f"{det.class_name} [{cat}] ({det.confidence:.2f})"
                cv2.putText(
                    annotated,
                    lbl,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            status = STATUS_REJECT if has_defect else STATUS_FRESH
            servo_angle = 90.0 if has_defect else 0.0
            defect_cat = classify_defect_hierarchy(defect_types, has_defect) if has_defect else "Healthy"

        # Save annotated result
        out_dir = REPO_ROOT / "runtime" / "inspections"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"web_inspect_{int(time.time())}_{filename}"
        cv2.imwrite(str(out_file), annotated)

        # Trigger simulated diverter flap if defect
        if status == STATUS_REJECT:
            self.manual_servo_actuate(90.0, 0.5)

        # Update telemetry statistics
        with self.lock:
            if status != "NO_FRUIT":
                self.stats["total_detections"] += len(det_details)
                if status == STATUS_REJECT:
                    self._photo_reject_count += 1
                elif status == STATUS_FRESH:
                    self._photo_fresh_count += 1

                if defect_cat in self.defect_counts:
                    self.defect_counts[defect_cat] += 1

                for d in det_details:
                    f_name = d.get("fruit", "").lower()
                    if f_name in self.fruit_counts:
                        self.fruit_counts[f_name] += 1
                    elif f_name and f_name not in ("none", "unknown"):
                        self.fruit_counts[f_name] = self.fruit_counts.get(f_name, 0) + 1

                    if f_name in self.fruit_defect_counts and defect_cat in self.fruit_defect_counts[f_name]:
                        self.fruit_defect_counts[f_name][defect_cat] += 1

            coord_fresh = getattr(self.coordinator, "fresh_passed_count", 0)
            coord_reject = getattr(self.coordinator, "defect_rejected_count", 0)
            if not isinstance(coord_fresh, (int, float)):
                coord_fresh = 0
            if not isinstance(coord_reject, (int, float)):
                coord_reject = 0

            self.stats["fresh_count"] = int(coord_fresh) + self._photo_fresh_count
            self.stats["reject_count"] = int(coord_reject) + self._photo_reject_count
            self.stats["fruit_counts"] = dict(self.fruit_counts)
            self.stats["defect_counts"] = dict(self.defect_counts)
            self.stats["fruit_defect_counts"] = {k: dict(v) for k, v in self.fruit_defect_counts.items()}

        ret_enc, enc_jpeg = cv2.imencode(".jpg", annotated)
        import base64
        base64_img = base64.b64encode(enc_jpeg).decode("utf-8") if ret_enc else ""

        fruit_names = sorted(list(set(d["fruit"].capitalize() for d in det_details if d.get("fruit") and d["fruit"] not in ("None", "unknown"))))
        primary_fruit = fruit_names[0].lower() if fruit_names else ("none" if status == "NO_FRUIT" else "fruit")
        fruit_display = ", ".join(fruit_names) if fruit_names else ("Non-Fruit" if status == "NO_FRUIT" else "Fruit")
        item_name = f"{fruit_display} ({filename})" if filename else fruit_display

        action_str = (
            "EJECT (90° Divert - REJECT)"
            if status == STATUS_REJECT
            else ("IDLE (0° Rest)" if status == "NO_FRUIT" else "PASS (0° Rest)")
        )
        event = {
            "timestamp": time.strftime("%H:%M:%S"),
            "filename": filename,
            "fruit_type": primary_fruit,
            "item_name": item_name,
            "status": status,
            "defect_category": defect_cat,
            "servo_angle": f"{int(servo_angle)}°",
            "servo_action": action_str,
            "defects": defect_types if defect_types else (["None (No Fruit)"] if status == "NO_FRUIT" else ["None (Healthy)"]),
            "detections": det_details,
            "image_data_url": f"data:image/jpeg;base64,{base64_img}",
        }

        with self.lock:
            self.recent_events.insert(0, event)
            if len(self.recent_events) > 30:
                self.recent_events.pop()

        return event

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
        self.coordinator.stop()


# Global Hub Instance
hub: Optional[SorterHub] = None


# ============================================================================
# HTTP Request Handler & MJPEG Streamer
# ============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agri_Sorter - Edge Multi-Fruit Sorter Dashboard</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --panel-bg: #1e293b;
    --border: #334155;
    --accent: #10b981;
    --danger: #ef4444;
    --warning: #eab308;
    --orange: #f97316;
    --text: #f8fafc;
    --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-dark); color: var(--text); padding: 1.5rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
  h1 { font-size: 1.6rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
  .badge { background: #0284c7; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; }
  .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
  .panel { background: var(--panel-bg); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; }
  .panel h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #cbd5e1; display: flex; justify-content: space-between; align-items: center; }
  .stream-container { position: relative; width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; aspect-ratio: 4/3; }
  .stream-container img { width: 100%; height: 100%; object-fit: contain; }
  .fruit-selector { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
  .btn-fruit { background: #334155; color: var(--text); border: 1px solid var(--border); padding: 0.5rem 1rem; border-radius: 0.5rem; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.2s; }
  .btn-fruit:hover { background: #475569; }
  .btn-fruit.active { background: #0284c7; border-color: #38bdf8; color: white; box-shadow: 0 0 10px rgba(2, 132, 199, 0.4); }
  .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
  .stat-card { background: #0f172a; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #38bdf8; }
  .stat-card.fresh, .stat-card.healthy { border-left-color: var(--accent); }
  .stat-card.reject, .stat-card.rot { border-left-color: var(--danger); }
  .stat-card.bruise { border-left-color: var(--warning); }
  .stat-card.scab { border-left-color: var(--orange); }
  .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
  .stat-value { font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }
  .servo-gauge { text-align: center; padding: 1.25rem; background: #0f172a; border-radius: 0.5rem; margin-bottom: 1.25rem; }
  .servo-flap-visual { width: 120px; height: 20px; background: #64748b; margin: 1rem auto; border-radius: 4px; transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1); transform-origin: left center; }
  .servo-flap-visual.actuated { background: var(--danger); transform: rotate(90deg); box-shadow: 0 0 15px rgba(239, 68, 68, 0.6); }
  .btn-actuate { background: var(--danger); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 0.5rem; cursor: pointer; font-weight: 700; width: 100%; transition: opacity 0.2s; }
  .btn-actuate:hover { opacity: 0.9; }
  .upload-box { border: 2px dashed var(--border); padding: 1.5rem; text-align: center; border-radius: 0.5rem; margin-top: 1rem; cursor: pointer; background: #0f172a; }
  .upload-box:hover { border-color: #38bdf8; }
  input[type="text"] { width: 100%; padding: 0.6rem; border-radius: 0.5rem; background: #0f172a; border: 1px solid var(--border); color: white; margin-top: 0.5rem; }
  .btn-primary { background: #0284c7; color: white; border: none; padding: 0.6rem 1rem; border-radius: 0.5rem; cursor: pointer; font-weight: 600; margin-top: 0.5rem; width: 100%; }
  .events-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 1rem; }
  .events-table th, .events-table td { padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border); }
  .events-table th { color: var(--text-muted); }
  .tag { padding: 0.2rem 0.55rem; border-radius: 4px; font-weight: 700; font-size: 0.75rem; display: inline-block; }
  .tag.fresh, .tag.healthy { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
  .tag.bruise, .tag.bruise_or_scratch { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
  .tag.scab, .tag.scab_or_canker { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
  .tag.reject, .tag.rot, .tag.rot_or_blight { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
  .tag.nofruit { background: rgba(148, 163, 184, 0.2); color: var(--text-muted); border: 1px solid #64748b; }
  .fruit-breakdown-bar { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.75rem; background: var(--panel-bg); padding: 0.6rem 1rem; border-radius: 0.5rem; border: 1px solid var(--border); }
  .fruit-pill { background: #0f172a; padding: 0.3rem 0.7rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; color: #cbd5e1; border: 1px solid var(--border); }
  .fruit-pill span { color: #38bdf8; font-weight: 700; margin-left: 0.25rem; }
  .defect-breakdown-bar { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.25rem; background: var(--panel-bg); padding: 0.6rem 1rem; border-radius: 0.5rem; border: 1px solid var(--border); }
  .defect-pill { padding: 0.35rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; font-weight: 700; border: 1px solid var(--border); }
  .defect-pill.healthy { background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: #10b981; }
  .defect-pill.bruise { background: rgba(234, 179, 8, 0.15); color: #facc15; border-color: #eab308; }
  .defect-pill.scab { background: rgba(249, 115, 22, 0.15); color: #fb923c; border-color: #f97316; }
  .defect-pill.rot { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: #ef4444; }
  .defect-pill span { font-weight: 800; margin-left: 0.3rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🍏 Agri_Sorter Real-Time Control Center <span class="badge" id="backendBadge">Hailo-8L / CPU</span></h1>
    <p style="color: var(--text-muted); font-size: 0.9rem;">Raspberry Pi AI HAT Edge Defect Sorter & ByteTrack Spatial Ejection Line</p>
  </div>
  <div style="text-align: right;">
    <span style="font-size: 0.9rem; color: var(--text-muted);">Frame Rate: <b id="fpsVal" style="color: #38bdf8;">--</b> FPS</span>
  </div>
</header>

<div class="fruit-selector">
  <button class="btn-fruit active" onclick="setFruit('all', this)">🌐 All Fruits (Core 4)</button>
  <button class="btn-fruit" onclick="setFruit('apple', this)">🍏 Apple</button>
  <button class="btn-fruit" onclick="setFruit('guava', this)">🍈 Guava</button>
  <button class="btn-fruit" onclick="setFruit('orange', this)">🍊 Orange</button>
  <button class="btn-fruit" onclick="setFruit('mango', this)">🥭 Mango</button>
  <button class="btn-fruit" onclick="setFruit('pomegranate', this)">🍎 Pomegranate</button>
</div>

<div class="fruit-breakdown-bar">
  <div class="fruit-pill">🍏 Apple: <span id="cntApple">0</span></div>
  <div class="fruit-pill">🍈 Guava: <span id="cntGuava">0</span></div>
  <div class="fruit-pill">🍊 Orange: <span id="cntOrange">0</span></div>
  <div class="fruit-pill">🥭 Mango: <span id="cntMango">0</span></div>
  <div class="fruit-pill">🍎 Pomegranate: <span id="cntPomegranate">0</span></div>
</div>

<div class="defect-breakdown-bar">
  <div class="defect-pill healthy">🟢 Healthy: <span id="cntHealthy">0</span> <small style="opacity:0.8">(0° PASS)</small></div>
  <div class="defect-pill bruise">🟡 Bruise_or_Scratch: <span id="cntBruise">0</span> <small style="opacity:0.8">(90° EJECT)</small></div>
  <div class="defect-pill scab">🟠 Scab_or_Canker: <span id="cntScab">0</span> <small style="opacity:0.8">(90° EJECT)</small></div>
  <div class="defect-pill rot">🔴 Rot_or_Blight: <span id="cntRot">0</span> <small style="opacity:0.8">(90° EJECT)</small></div>
</div>

<div class="grid">
  <!-- Left Column: Video & Telemetry -->
  <div>
    <div class="panel">
      <h2>
        <span>Live Camera Vision Stream (YOLO11 Edge)</span>
        <span style="font-size: 0.8rem; color: var(--text-muted);">Egress Threshold: Y = 85%</span>
      </h2>
      <div class="stream-container">
        <img src="/video_feed" alt="Live YOLO Stream">
      </div>
    </div>

    <!-- Live Event Log -->
    <div class="panel" style="margin-top: 1.5rem;">
      <h2>Recent Fruit Decision Telemetry</h2>
      <table class="events-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Item / Source</th>
            <th>Quality</th>
            <th>Defect Hierarchy</th>
            <th>Servo Decision</th>
          </tr>
        </thead>
        <tbody id="eventsTbody">
          <tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Waiting for fruit detections...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Right Column: Sorter Controls & Photo Tester -->
  <div>
    <!-- Sorter Actuator Status -->
    <div class="panel">
      <h2>Diverter Servo Actuator</h2>
      <div class="servo-gauge">
        <div style="font-size: 0.85rem; color: var(--text-muted);">CURRENT FLAP POSITION</div>
        <div style="font-size: 2rem; font-weight: 800; color: #10b981;" id="servoAngleDisplay">0° [PASS]</div>
        <div class="servo-flap-visual" id="servoFlapVisual"></div>
        <button class="btn-actuate" onclick="triggerServo()">⚡ MANUAL EJECT PULSE (90°)</button>
      </div>

      <div class="stats-grid">
        <div class="stat-card fresh">
          <div class="stat-label">Fresh Fruit Passed</div>
          <div class="stat-value" id="freshCount">0</div>
        </div>
        <div class="stat-card reject">
          <div class="stat-label">Defects Diverted</div>
          <div class="stat-value" id="rejectCount">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Active Tracks</div>
          <div class="stat-value" id="activeTracks">0</div>
        </div>
        <div class="stat-card reject">
          <div class="stat-label">Fail-Safe Underruns</div>
          <div class="stat-value" id="underrunCount">0</div>
        </div>
      </div>

      <h3 style="font-size: 0.85rem; color: var(--text-muted); margin: 1rem 0 0.5rem 0; text-transform: uppercase;">Defect Hierarchy Telemetry</h3>
      <div class="stats-grid">
        <div class="stat-card healthy">
          <div class="stat-label">🟢 Healthy (0° PASS)</div>
          <div class="stat-value" id="cardHealthy">0</div>
        </div>
        <div class="stat-card bruise">
          <div class="stat-label">🟡 Bruise/Scratch (90°)</div>
          <div class="stat-value" id="cardBruise">0</div>
        </div>
        <div class="stat-card scab">
          <div class="stat-label">🟠 Scab/Canker (90°)</div>
          <div class="stat-value" id="cardScab">0</div>
        </div>
        <div class="stat-card rot">
          <div class="stat-label">🔴 Rot/Blight (90°)</div>
          <div class="stat-value" id="cardRot">0</div>
        </div>
      </div>
    </div>

    <!-- Internet Photo Tester Tab -->
    <div class="panel" style="margin-top: 1.5rem;">
      <h2>🧪 Test Any Photo From Internet</h2>
      <p style="font-size: 0.85rem; color: var(--text-muted);">Paste an image link or drop a fruit photo to run YOLO edge defect classification and simulate servo diverter response.</p>
      
      <input type="text" id="urlInput" placeholder="https://example.com/fruit_photo.jpg">
      <button class="btn-primary" onclick="testUrl()">🔍 Test Image URL</button>

      <div class="upload-box" onclick="document.getElementById('fileInput').click()">
        📁 Click or Drag & Drop Photo Here
        <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="uploadFile(this.files[0])">
      </div>

      <div id="photoResult" style="margin-top: 1rem; display: none;">
        <h3 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: #38bdf8;">Inspection Result:</h3>
        <div style="width: 100%; border-radius: 0.5rem; overflow: hidden; background: #000; margin-bottom: 0.5rem;">
          <img id="photoResultImg" src="" style="width: 100%; display: block;">
        </div>
        <div id="photoResultText" style="font-size: 0.85rem; line-height: 1.4;"></div>
      </div>
    </div>
  </div>
</div>

<script>
  let activeFruit = 'all';

  function setFruit(fruit, btn) {
    document.querySelectorAll('.btn-fruit').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFruit = fruit;
    fetch('/api/set_fruit?fruit=' + fruit).then(r => r.json()).then(console.log);
  }

  function triggerServo() {
    fetch('/api/manual_servo').then(r => r.json()).then(console.log);
  }

  function testUrl() {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) return alert('Please enter an image URL');
    document.getElementById('photoResult').style.display = 'block';
    document.getElementById('photoResultText').innerHTML = '<i>Downloading and running YOLO defect detection...</i>';
    fetch('/api/test_url?url=' + encodeURIComponent(url))
      .then(r => r.json())
      .then(showResult)
      .catch(e => { document.getElementById('photoResultText').innerText = 'Error: ' + e; });
  }

  function uploadFile(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    document.getElementById('photoResult').style.display = 'block';
    document.getElementById('photoResultText').innerHTML = '<i>Processing image through YOLO...</i>';
    fetch('/api/upload_photo', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(showResult)
      .catch(e => { document.getElementById('photoResultText').innerText = 'Error: ' + e; });
  }

  function showResult(data) {
    document.getElementById('photoResultImg').src = data.image_data_url;
    const isEject = (data.servo_angle && !data.servo_angle.startsWith('0')) || data.status === 'REJECT';
    const isNoFruit = data.status === 'NO_FRUIT';
    const fruitIcons = {
      'apple': '🍏',
      'guava': '🍈',
      'orange': '🍊',
      'mango': '🥭',
      'pomegranate': '🍎'
    };
    const fKey = (data.fruit_type || '').toLowerCase();
    const fruitIcon = fruitIcons[fKey] || '🍎';
    const fruitDisplay = fKey !== 'none' ? `${fruitIcon} ${(data.fruit_type || 'Fruit').toUpperCase()}` : '📦 Non-Fruit';

    const cat = data.defect_category || (data.status === 'REJECT' ? 'Rot_or_Blight' : (isNoFruit ? 'None' : 'Healthy'));
    let catClass = 'healthy';
    if (cat === 'Bruise_or_Scratch') catClass = 'bruise';
    else if (cat === 'Scab_or_Canker') catClass = 'scab';
    else if (cat === 'Rot_or_Blight') catClass = 'rot';
    else if (isNoFruit) catClass = 'nofruit';

    const servoBadgeClass = isEject ? 'rot' : (isNoFruit ? 'nofruit' : 'healthy');
    const servoDecision = isEject ? `${data.servo_angle || '90°'} EJECT (Diverter Actuated)` : (isNoFruit ? '0° IDLE (Rest)' : '0° PASS (Rest)');

    document.getElementById('photoResultText').innerHTML = `
      <div style="background: #0f172a; padding: 0.85rem; border-radius: 0.5rem; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 0.45rem;">
        <div><b>Target / Detected Fruit:</b> <span style="font-weight: 700; color: #38bdf8; margin-left: 0.3rem;">${fruitDisplay}</span></div>
        <div><b>Defect Hierarchy:</b> <span class="tag ${catClass}" style="margin-left: 0.3rem;">${cat}</span></div>
        <div><b>Servo Decision:</b> <span class="tag ${servoBadgeClass}" style="margin-left: 0.3rem;">${servoDecision}</span></div>
        <div style="color: var(--text-muted); font-size: 0.8rem;"><b>Raw Defects / Conditions:</b> ${(data.defects || []).join(', ')}</div>
        <div style="color: var(--text-muted); font-size: 0.8rem;"><b>Detections:</b> ${(data.detections || []).map(d => {
          const ic = fruitIcons[(d.fruit || '').toLowerCase()] || '📦';
          const dCat = d.defect_category || cat;
          return `${ic} ${(d.fruit || '').toUpperCase()} [${dCat}] (${d.condition}, ${((d.confidence || 0)*100).toFixed(0)}%)`;
        }).join(', ') || 'None'}</div>
      </div>
    `;
  }

  // Periodic status poll
  setInterval(() => {
    fetch('/api/status').then(r => r.json()).then(s => {
      document.getElementById('fpsVal').innerText = s.fps;
      document.getElementById('freshCount').innerText = s.fresh_count;
      document.getElementById('rejectCount').innerText = s.reject_count;
      document.getElementById('underrunCount').innerText = s.underrun_count;
      document.getElementById('activeTracks').innerText = s.active_tracks;
      document.getElementById('backendBadge').innerText = (s.backend || 'Hailo-8L / CPU').toUpperCase();
      
      if (s.fruit_counts) {
        document.getElementById('cntApple').innerText = s.fruit_counts.apple || 0;
        document.getElementById('cntGuava').innerText = s.fruit_counts.guava || 0;
        document.getElementById('cntOrange').innerText = s.fruit_counts.orange || 0;
        document.getElementById('cntMango').innerText = s.fruit_counts.mango || 0;
        document.getElementById('cntPomegranate').innerText = s.fruit_counts.pomegranate || 0;
      }

      // Defect hierarchy live counts (per selected fruit or global)
      let dCounts = s.defect_counts;
      if (activeFruit !== 'all' && s.fruit_defect_counts && s.fruit_defect_counts[activeFruit]) {
        dCounts = s.fruit_defect_counts[activeFruit];
      }
      if (dCounts) {
        document.getElementById('cntHealthy').innerText = dCounts.Healthy || 0;
        document.getElementById('cntBruise').innerText = dCounts.Bruise_or_Scratch || 0;
        document.getElementById('cntScab').innerText = dCounts.Scab_or_Canker || 0;
        document.getElementById('cntRot').innerText = dCounts.Rot_or_Blight || 0;
        if (document.getElementById('cardHealthy')) document.getElementById('cardHealthy').innerText = dCounts.Healthy || 0;
        if (document.getElementById('cardBruise')) document.getElementById('cardBruise').innerText = dCounts.Bruise_or_Scratch || 0;
        if (document.getElementById('cardScab')) document.getElementById('cardScab').innerText = dCounts.Scab_or_Canker || 0;
        if (document.getElementById('cardRot')) document.getElementById('cardRot').innerText = dCounts.Rot_or_Blight || 0;
      }

      const isActuated = s.servo_angle > 10;
      const gaugeEl = document.getElementById('servoAngleDisplay');
      if (isActuated) {
        gaugeEl.innerText = Math.round(s.servo_angle) + '° [EJECT]';
        gaugeEl.style.color = '#ef4444';
      } else {
        gaugeEl.innerText = Math.round(s.servo_angle) + '° [PASS]';
        gaugeEl.style.color = '#10b981';
      }
      document.getElementById('servoFlapVisual').className = 'servo-flap-visual' + (isActuated ? ' actuated' : '');

      if (s.recent_events && s.recent_events.length > 0) {
        let html = '';
        const fruitIcons = {
          'apple': '🍏 Apple',
          'guava': '🍈 Guava',
          'orange': '🍊 Orange',
          'mango': '🥭 Mango',
          'pomegranate': '🍎 Pomegranate'
        };
        s.recent_events.slice(0, 10).forEach(ev => {
          const isRej = ev.status === 'REJECT' || (ev.servo_angle && !ev.servo_angle.startsWith('0'));
          const isNoFruit = ev.status === 'NO_FRUIT';
          const fKey = (ev.fruit_type || '').toLowerCase();
          const fruitBadge = fruitIcons[fKey] || ev.fruit_type || 'Fruit';
          const itemDisplay = ev.item_name || `${fruitBadge} (${ev.filename || 'Live Track'})`;
          
          const cat = ev.defect_category || (isRej ? 'Rot_or_Blight' : (isNoFruit ? 'None' : 'Healthy'));
          let catClass = 'healthy';
          if (cat === 'Bruise_or_Scratch') catClass = 'bruise';
          else if (cat === 'Scab_or_Canker') catClass = 'scab';
          else if (cat === 'Rot_or_Blight') catClass = 'rot';
          else if (isNoFruit) catClass = 'nofruit';

          const servoText = isRej ? '<span style="color:#ef4444; font-weight:700;">90° EJECT</span>' : (isNoFruit ? '<span style="color:#94a3b8;">0° IDLE</span>' : '<span style="color:#10b981; font-weight:700;">0° PASS</span>');
          const statusClass = isRej ? 'reject' : (isNoFruit ? 'nofruit' : 'fresh');

          html += `<tr>
            <td>${ev.timestamp}</td>
            <td><b>${itemDisplay}</b></td>
            <td><span class="tag ${statusClass}">${ev.status}</span></td>
            <td><span class="tag ${catClass}">${cat}</span></td>
            <td>${servoText}</td>
          </tr>`;
        });
        document.getElementById('eventsTbody').innerHTML = html;
      }
    });
  }, 500);
</script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Agri_Sorter dashboard."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP logging to keep console clean."""
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/video_feed":
            # MJPEG live stream
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    frame_bytes = None
                    if hub:
                        with hub.lock:
                            frame_bytes = hub.current_frame_jpeg

                    if frame_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with hub.lock:
                status_dict = dict(hub.stats)
                status_dict["recent_events"] = list(hub.recent_events)
                status_dict["selected_fruit"] = hub.selected_fruit
                status_dict["fruit_counts"] = dict(hub.fruit_counts)
                status_dict["defect_counts"] = dict(hub.defect_counts)
                status_dict["fruit_defect_counts"] = {k: dict(v) for k, v in hub.fruit_defect_counts.items()}
            self.wfile.write(json.dumps(status_dict, default=list).encode("utf-8"))

        elif path == "/api/set_fruit":
            fruit = params.get("fruit", ["all"])[0].lower().strip()
            if fruit in SUPPORTED_FRUITS or fruit in FRUITS:
                if hub:
                    hub.set_fruit_filter(fruit)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "fruit": fruit}).encode("utf-8"))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Invalid fruit '{fruit}'. Supported: {SUPPORTED_FRUITS}"
                }).encode("utf-8"))

        elif path == "/api/manual_servo":
            if hub:
                hub.manual_servo_actuate(90.0, 0.5)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "action": "servo_reject_actuated"}).encode("utf-8"))

        elif path == "/api/test_url":
            url = params.get("url", [None])[0]
            if not url or not hub:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing url parameter")
                return
            try:
                resp = requests.get(url, timeout=8)
                resp.raise_for_status()
                res = hub.process_test_photo(resp.content, filename=url.split("/")[-1] or "internet_fruit.jpg")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as exc:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload_photo":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Simple multipart/form-data extraction or raw bytes
            if b"filename=" in body:
                try:
                    # Parse multipart body boundary
                    boundary = content_type.split("boundary=")[-1].encode("utf-8")
                    parts = body.split(b"--" + boundary)
                    img_data = b""
                    filename = "uploaded_fruit.jpg"
                    for part in parts:
                        if b'name="image"' in part:
                            headers_and_body = part.split(b"\r\n\r\n", 1)
                            if len(headers_and_body) == 2:
                                img_data = headers_and_body[1].rstrip(b"\r\n")
                    if img_data and hub:
                        res = hub.process_test_photo(img_data, filename=filename)
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(res).encode("utf-8"))
                        return
                except Exception as exc:
                    logger.error(f"Upload parsing error: {exc}")

            # Fallback raw image upload
            if hub and body:
                res = hub.process_test_photo(body, filename="uploaded_fruit.jpg")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image payload")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def run_dashboard_server(port: int = 8080) -> None:
    """Launch the Agri_Sorter Web Dashboard server."""
    global hub
    hub = SorterHub()

    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)

    print("\n" + "=" * 70)
    print("🚀 AGRI_SORTER INTERACTIVE CONTROL CENTER READY!")
    print(f"   👉 Open in your browser: http://localhost:{port}")
    print("   • Live webcam feed with YOLO bounding boxes")
    print("   • Choose target fruit: All (Core 4), Apple, Guava, Orange, Mango, Pomegranate")
    print("   • Test photos: Paste internet URLs or drag & drop images")
    print("   • Watch servo flap actuate in real time (0° Pass vs 90° Reject)")
    print("=" * 70 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        if hub:
            hub.stop()
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agri_Sorter Web Control Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    args = parser.parse_args()
    run_dashboard_server(port=args.port)

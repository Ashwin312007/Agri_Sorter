"""Conveyor Pipeline & Spatial Tracking Coordinator.

Orchestrates conveyor belt tracking of fruit from camera detection (Station 0)
to ultrasonic acoustic sensing (Station 1) and automated sorter ejection (Station 2).
"""
import argparse
import csv
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    cv2.setNumThreads(3)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from app.hardware.camera import CameraBackend, MockCamera, RealCamera, ThreadedCamera
from app.hardware.config import load_config
from app.hardware.pipeline import UltrasonicPipeline
from app.pi_inference import (
    classify_apple,
    crop_box,
    find_class_id,
    is_defect_detector,
    load_vision_engine,
)

logger = logging.getLogger("conveyor_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BAD_LABELS_REJECT = {"rotten", "rot", "mold", "mould", "crack"}
BAD_LABELS_SECOND = {"bruise", "bruising", "spot", "damage", "damaged", "insect_bite", "russetting"}
BAD_LABELS = BAD_LABELS_REJECT | BAD_LABELS_SECOND

DEFAULT_DETECTOR_MODEL = (
    "models/detector/yolo11n.onnx"
    if Path("models/detector/yolo11n.onnx").is_file()
    else "models/detector/yolo11n.pt"
)
DEFAULT_CLASSIFIER_MODEL = (
    "models/external/best.onnx"
    if Path("models/external/best.onnx").is_file()
    else "models/external/best.pt"
)

CSV_FIELDNAMES = [
    "track_id",
    "entry_time",
    "bbox",
    "visual_label",
    "visual_confidence",
    "ultrasonic_travel_time_us",
    "ultrasonic_amplitude",
    "internal_label_pred",
    "final_grade",
    "status",
    "internal_fallback",
]


@dataclass
class FruitTrack:
    """Represents a single fruit being tracked along the conveyor belt."""

    track_id: int
    entry_time: float
    bbox: Tuple[int, int, int, int]
    visual_label: str
    visual_confidence: float
    ultrasonic_travel_time_us: Optional[float] = None
    ultrasonic_amplitude: Optional[int] = None
    internal_label_pred: Optional[int] = None
    final_grade: Optional[str] = None  # 'FIRST_QUALITY', 'SECOND_QUALITY', 'REJECT'
    status: str = "IN_TRANSIT"  # 'IN_TRANSIT', 'MEASURED', 'EJECTED'
    internal_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert track state to dictionary representation."""
        return {
            "track_id": self.track_id,
            "entry_time": self.entry_time,
            "bbox": f"{self.bbox[0]}:{self.bbox[1]}:{self.bbox[2]}:{self.bbox[3]}",
            "visual_label": self.visual_label,
            "visual_confidence": self.visual_confidence,
            "ultrasonic_travel_time_us": self.ultrasonic_travel_time_us,
            "ultrasonic_amplitude": self.ultrasonic_amplitude,
            "internal_label_pred": self.internal_label_pred,
            "final_grade": self.final_grade,
            "status": self.status,
            "internal_fallback": self.internal_fallback,
        }


class ConveyorCoordinator:
    """Coordinates spatial tracking and quality classification of fruit along conveyor."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        mock: bool = False,
        speed_override: Optional[float] = None,
        belt_speed_mps: Optional[float] = None,
        strict_safety: bool = False,
        internal_model_path: Optional[Union[str, Path]] = None,
        detector_model_path: Optional[Union[str, Path]] = None,
        classifier_model_path: Optional[Union[str, Path]] = None,
        headless: bool = False,
        feed_mock_tracks: bool = False,
        detect_cadence: int = 2,
    ) -> None:
        """Initialize conveyor coordinator.

        Args:
            config: Hardware configuration dictionary or path to hardware.toml.
            mock: If True, uses mock hardware backends.
            speed_override: Optional belt speed override in m/s (or belt_speed_mps).
            belt_speed_mps: Alias for speed_override.
            strict_safety: If True, raises ValueError when speed exceeds max safe speed.
            internal_model_path: Optional path to internal classifier joblib model.
            detector_model_path: Path to YOLO detector model (.onnx or .pt).
            classifier_model_path: Path to external visual classifier model (.onnx or .pt).
            headless: If True, disables cv2 GUI display windows.
            feed_mock_tracks: If True, allows bench testing sensors with mock tracks.
            detect_cadence: Run full YOLO detection every N frames (interpolating intermediate frames).
        """
        if belt_speed_mps is not None and speed_override is None:
            speed_override = belt_speed_mps

        if config is None or isinstance(config, (str, Path)):
            self.config = load_config(config, mock=mock)
        else:
            self.config = config
            if mock:
                self.config["mock"] = True

        self.mock = mock or self.config.get("mock", False)
        self.strict_safety = strict_safety
        self.headless = headless or not bool(os.environ.get("DISPLAY"))
        self.feed_mock_tracks = feed_mock_tracks
        self.detect_cadence = max(1, int(detect_cadence))
        self._frame_count = 0
        self.detector_model_path = detector_model_path or DEFAULT_DETECTOR_MODEL
        self.classifier_model_path = classifier_model_path or DEFAULT_CLASSIFIER_MODEL

        conveyor_cfg = self.config.get("conveyor", {})
        self.belt_speed_mps = float(
            speed_override
            if speed_override is not None
            else conveyor_cfg.get("belt_speed_mps", 0.12)
        )
        self.max_safe_speed_mps = float(conveyor_cfg.get("max_safe_speed_mps", 0.20))
        self.camera_to_ultrasonic_distance_m = float(
            conveyor_cfg.get("camera_to_ultrasonic_distance_m", 0.30)
        )
        self.ultrasonic_to_sorter_distance_m = float(
            conveyor_cfg.get("ultrasonic_to_sorter_distance_m", 0.25)
        )
        self.apex_sampling_window_mm = float(
            conveyor_cfg.get("apex_sampling_window_mm", 35.0)
        )

        # Check conveyor speed safety
        self._check_speed_safety()

        # Initialize Camera backend
        cam_cfg = self.config.get("camera", {})
        resolution = cam_cfg.get("resolution", (1280, 720))
        threaded = cam_cfg.get("threaded", False)
        if self.mock:
            self.camera: CameraBackend = MockCamera(
                width=int(resolution[0]),
                height=int(resolution[1]),
            )
        else:
            self.camera = RealCamera(
                backend=cam_cfg.get("backend", "auto"),
                device_index=cam_cfg.get("device_index", 0),
                resolution=resolution,
                threaded=threaded,
            )
        if threaded:
            self.camera = ThreadedCamera(self.camera)

        # Initialize Ultrasonic pipeline
        self.ultrasonic = UltrasonicPipeline(config=self.config, mock=self.mock)

        # Initialize internal classification model if present
        self.internal_model = None
        self.internal_features = None
        if internal_model_path:
            self._load_internal_model(internal_model_path)

        # Initialize YOLO vision detector and classifier (if not in mock-only mode)
        self.detector = None
        self.classifier = None
        self.apple_class_id = None
        if not self.mock:
            self._load_vision_models()

        # Spatial tracking queue and track archive
        self.spatial_queue: List[FruitTrack] = []
        self.all_tracks: List[FruitTrack] = []
        self.items: Dict[int, FruitTrack] = {}
        self._next_track_id = 1
        self._is_started = False

    def _load_vision_models(self) -> None:
        """Load vision detector and classification models using lightweight OpenCV DNN engine."""
        det_path = Path(self.detector_model_path)
        cls_path = Path(self.classifier_model_path)

        logger.info("Loading vision detector from %s", det_path)
        self.detector = load_vision_engine(det_path, task="detect")
        self.single_pass = is_defect_detector(self.detector)

        if self.single_pass:
            logger.info("Detector %s contains defect classes; running in single-pass mode", det_path)
            self.classifier = None
            self.apple_class_id = None
        else:
            self.apple_class_id = find_class_id(self.detector, "apple")
            logger.info("Loading vision classifier from %s", cls_path)
            self.classifier = load_vision_engine(cls_path, task="classify")

    def _check_speed_safety(self) -> None:
        """Validate conveyor speed against hardware limits."""
        if self.belt_speed_mps <= 0:
            raise ValueError(f"Conveyor belt speed must be positive (> 0), got: {self.belt_speed_mps}")

        if self.belt_speed_mps > self.max_safe_speed_mps:
            msg = (
                f"Conveyor speed ({self.belt_speed_mps:.3f} m/s) exceeds maximum safe speed "
                f"({self.max_safe_speed_mps:.3f} m/s)! ADS1115 sampling rate (860 SPS) "
                f"may miss the apex measurement window."
            )
            if self.strict_safety:
                raise ValueError(msg)
            logger.warning("SAFETY ALERT: %s", msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            print("\n" + "=" * 70, file=sys.stderr)
            print("⚠️  CONVEYOR SPEED SAFETY WARNING:", file=sys.stderr)
            print(f"   Belt speed:     {self.belt_speed_mps:.3f} m/s", file=sys.stderr)
            print(f"   Max safe limit: {self.max_safe_speed_mps:.3f} m/s", file=sys.stderr)
            print("   ADS1115 sampling rate may cause missed apex readings!", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)

    def _load_internal_model(self, model_path: Union[str, Path]) -> None:
        """Attempt to load trained internal classifier."""
        path = Path(model_path)
        if not path.is_file():
            logger.warning("Internal model path not found: %s", path)
            return
        try:
            from app.ultrasonic_capture import load_prediction_model

            self.internal_model, self.internal_features = load_prediction_model(str(path))
            logger.info("Loaded internal quality prediction model from %s", path)
        except Exception as exc:
            logger.warning("Could not load internal model from %s: %s", path, exc)
            self.internal_model = None

    def start(self) -> None:
        """Start hardware subsystems."""
        if not self._is_started:
            self.camera.start()
            self._is_started = True

    def stop(self) -> None:
        """Stop hardware subsystems."""
        if self._is_started:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.ultrasonic.close()
            except Exception:
                pass
            self._is_started = False
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "ConveyorCoordinator":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def get_arrival_time(self, track: FruitTrack) -> float:
        """Compute arrival timestamp at Station 1 (Ultrasonic)."""
        return track.entry_time + (self.camera_to_ultrasonic_distance_m / self.belt_speed_mps)

    def get_ejection_time(self, track: FruitTrack) -> float:
        """Compute arrival timestamp at Station 2 (Sorter / Ejector)."""
        total_dist = self.camera_to_ultrasonic_distance_m + self.ultrasonic_to_sorter_distance_m
        return track.entry_time + (total_dist / self.belt_speed_mps)

    def register_fruit(
        self,
        bbox: Tuple[int, int, int, int],
        visual_label: str,
        visual_confidence: float,
        entry_time: Optional[float] = None,
        track_id: Optional[int] = None,
    ) -> FruitTrack:
        """Register a detected fruit into the conveyor spatial tracking queue."""
        if entry_time is None:
            entry_time = time.time()
        if track_id is None:
            track_id = self._next_track_id
            self._next_track_id += 1

        track = FruitTrack(
            track_id=track_id,
            entry_time=entry_time,
            bbox=bbox,
            visual_label=visual_label,
            visual_confidence=visual_confidence,
            status="IN_TRANSIT",
        )
        self.add_track(track)
        return track

    def add_track(self, track: FruitTrack) -> FruitTrack:
        """Add an existing FruitTrack object to queue."""
        self.spatial_queue.append(track)
        if track not in self.all_tracks:
            self.all_tracks.append(track)
        self.items[track.track_id] = track
        if track.track_id >= self._next_track_id:
            self._next_track_id = track.track_id + 1
        return track

    def _evaluate_grade(self, track: FruitTrack) -> None:
        """Determine final grade from visual inspection and ultrasonic measurements."""
        label_lower = track.visual_label.strip().lower()

        # Step 1: External visual defect check
        if label_lower in BAD_LABELS:
            if label_lower in BAD_LABELS_REJECT:
                track.final_grade = "REJECT"
            else:
                track.final_grade = "SECOND_QUALITY"
            track.internal_fallback = False
            return

        # Step 2: Internal acoustic defect prediction if model available
        if self.internal_model is not None:
            try:
                from app.ultrasonic_capture import predict_row

                pred = predict_row(
                    model=self.internal_model,
                    features=self.internal_features,
                    travel_time_us=track.ultrasonic_travel_time_us or 0.0,
                    received_amplitude=track.ultrasonic_amplitude or 0,
                )
                track.internal_label_pred = pred
                track.final_grade = "FIRST_QUALITY" if pred == 0 else "SECOND_QUALITY"
                track.internal_fallback = False
                return
            except Exception as exc:
                logger.warning(
                    "Internal model prediction failed for track #%d (%s); falling back to FIRST_QUALITY",
                    track.track_id,
                    exc,
                )
                track.internal_fallback = True
                track.final_grade = "FIRST_QUALITY"
                return

        # Step 3: Graceful fallback when internal model is absent
        track.internal_fallback = True
        track.final_grade = "FIRST_QUALITY"

    def step(
        self,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[FruitTrack]:
        """Perform a single step of spatial tracking and station evaluation.

        Args:
            frame: Optional video frame. If None, captured from camera if started.
            timestamp: Optional current timestamp (seconds). Defaults to time.time().

        Returns:
            List[FruitTrack]: Current spatial queue of active tracks.
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_count += 1
        is_cadence = (self._frame_count % self.detect_cadence == 0)

        if frame is None and self._is_started:
            try:
                frame = self.camera.capture()
            except Exception as exc:
                logger.debug("Frame capture skipped: %s", exc)

        # Station 0: Live camera detection & tracking
        if not self.mock and not self.feed_mock_tracks:
            if frame is not None and self.detector is not None:
                track_classes = None if getattr(self, "single_pass", False) else [self.apple_class_id]
                result = self.detector.track(
                    frame,
                    persist=True,
                    classes=track_classes,
                    imgsz=416,
                    conf=0.30,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    skip_detect=(not is_cadence),
                )[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    num_tracks = len(track_ids)
                    class_ids = [0] * num_tracks
                    confidences = [1.0] * num_tracks
                    try:
                        if hasattr(result.boxes, "cls") and result.boxes.cls is not None:
                            c = result.boxes.cls.int().cpu().tolist()
                            if isinstance(c, (list, tuple)) and len(c) == num_tracks:
                                class_ids = [int(v) for v in c]
                    except Exception:
                        pass
                    try:
                        if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                            cf = result.boxes.conf.float().cpu().tolist()
                            if isinstance(cf, (list, tuple)) and len(cf) == num_tracks:
                                confidences = [float(v) for v in cf]
                    except Exception:
                        pass

                    for box, track_id, cid, conf in zip(boxes, track_ids, class_ids, confidences):
                        track_id = int(track_id)
                        if track_id not in self.items:
                            if getattr(self, "single_pass", False):
                                visual_label = self.detector.names.get(cid, str(cid)).lower()
                                visual_conf = float(conf)
                            else:
                                crop = crop_box(frame, box)
                                if crop.size > 0 and self.classifier is not None:
                                    visual_label, visual_conf = classify_apple(self.classifier, crop)
                                else:
                                    visual_label, visual_conf = "apple", 1.0
                            new_track = FruitTrack(
                                track_id=track_id,
                                entry_time=timestamp,
                                bbox=tuple(int(v) for v in box),
                                visual_label=visual_label,
                                visual_confidence=visual_conf,
                            )
                            self.items[track_id] = new_track
                            self.add_track(new_track)
                            logger.info(
                                "Station 0: detected new apple track #%d (%s, conf=%.2f) at %s",
                                track_id,
                                visual_label,
                                visual_conf,
                                new_track.bbox,
                            )
                        else:
                            self.items[track_id].bbox = tuple(int(v) for v in box)

                        if not self.headless:
                            x1, y1, x2, y2 = (int(v) for v in box)
                            item = self.items.get(track_id)
                            lbl = item.visual_label if item else "apple"
                            conf = item.visual_confidence if item else 1.0
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"#{track_id} {lbl} {conf:.0%}",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2,
                            )

        for track in list(self.spatial_queue):
            t_arrive = self.get_arrival_time(track)
            t_eject = self.get_ejection_time(track)

            # Station 1: Ultrasonic measurement window
            if track.status == "IN_TRANSIT" and timestamp >= t_arrive:
                meas = self.ultrasonic.measure()
                track.ultrasonic_travel_time_us = float(meas["travel_time_us"])
                track.ultrasonic_amplitude = int(meas["received_amplitude"])
                self._evaluate_grade(track)
                track.status = "MEASURED"
                logger.info(
                    "Track #%d MEASURED at Station 1: ToF=%.1f us, Amp=%d -> Grade=%s (fallback=%s)",
                    track.track_id,
                    track.ultrasonic_travel_time_us,
                    track.ultrasonic_amplitude,
                    track.final_grade,
                    track.internal_fallback,
                )

            # Station 2: Sorter ejection
            if track.status == "MEASURED" and timestamp >= t_eject:
                track.status = "EJECTED"
                logger.info(
                    "Track #%d EJECTED at Station 2: Grade=%s",
                    track.track_id,
                    track.final_grade,
                )

        # Visual display
        if frame is not None and not self.headless:
            cv2.imshow("Conveyor Vision", frame)
            cv2.waitKey(1)

        return list(self.spatial_queue)

    def save_csv(self, csv_path: Union[str, Path]) -> None:
        """Export all processed track records to CSV."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for t in self.all_tracks:
                writer.writerow({
                    "track_id": t.track_id,
                    "entry_time": f"{t.entry_time:.4f}",
                    "bbox": f"{t.bbox[0]}:{t.bbox[1]}:{t.bbox[2]}:{t.bbox[3]}",
                    "visual_label": t.visual_label,
                    "visual_confidence": f"{t.visual_confidence:.4f}",
                    "ultrasonic_travel_time_us": (
                        f"{t.ultrasonic_travel_time_us:.1f}"
                        if t.ultrasonic_travel_time_us is not None
                        else ""
                    ),
                    "ultrasonic_amplitude": (
                        str(t.ultrasonic_amplitude)
                        if t.ultrasonic_amplitude is not None
                        else ""
                    ),
                    "internal_label_pred": (
                        str(t.internal_label_pred)
                        if t.internal_label_pred is not None
                        else ""
                    ),
                    "final_grade": t.final_grade or "",
                    "status": t.status,
                    "internal_fallback": str(t.internal_fallback),
                })


def main() -> None:
    """CLI entrypoint for conveyor pipeline."""
    parser = argparse.ArgumentParser(
        description="Conveyor Pipeline & Spatial Tracking Coordinator"
    )
    parser.add_argument("--mock", action="store_true", help="Run with mock hardware")
    parser.add_argument("--config", type=str, default=None, help="Path to hardware.toml config file")
    parser.add_argument("--speed", type=float, default=None, help="Belt speed override in m/s")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds")
    parser.add_argument("--csv-out", type=str, default=None, help="Path to save output CSV")
    parser.add_argument(
        "--strict-safety",
        action="store_true",
        help="Raise error if speed exceeds maximum safe speed",
    )
    parser.add_argument(
        "--internal-model",
        type=str,
        default=None,
        help="Path to trained internal classifier joblib file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI visual display window (no cv2.imshow)",
    )
    parser.add_argument(
        "--feed-mock-tracks",
        action="store_true",
        default=False,
        help="Feed mock fruit tracks for bench testing real sensors without physical fruit",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default=DEFAULT_DETECTOR_MODEL,
        help="Path to YOLO detector model (.onnx or .pt)",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=DEFAULT_CLASSIFIER_MODEL,
        help="Path to external visual classifier model (.onnx or .pt)",
    )
    parser.add_argument(
        "--detect-cadence",
        type=int,
        default=2,
        help="Run full YOLO detection every N frames (interpolating intermediate frames, default: 2)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target frame rate cap (default: None for unthrottled maximum FPS)",
    )
    args = parser.parse_args()

    coordinator = ConveyorCoordinator(
        config=args.config,
        mock=args.mock,
        speed_override=args.speed,
        strict_safety=args.strict_safety,
        internal_model_path=args.internal_model,
        detector_model_path=args.detector_model,
        classifier_model_path=args.classifier_model,
        headless=args.headless,
        feed_mock_tracks=args.feed_mock_tracks,
        detect_cadence=args.detect_cadence,
    )

    coordinator.start()

    start_time = time.time()
    speed = coordinator.belt_speed_mps
    t_to_us = coordinator.camera_to_ultrasonic_distance_m / speed
    t_to_sorter = (
        coordinator.camera_to_ultrasonic_distance_m
        + coordinator.ultrasonic_to_sorter_distance_m
    ) / speed

    # In mock mode or feed-mock-tracks mode, pre-seed representative fruit tracks so full spatial cycle is exercised
    if args.mock or args.feed_mock_tracks:
        # Track 1: Sorter boundary track (pre-measured, arrives at sorter within 0.7s)
        t1 = coordinator.register_fruit(
            bbox=(100, 150, 220, 270),
            visual_label="bruise",
            visual_confidence=0.88,
            entry_time=start_time - (t_to_sorter - 0.7),
            track_id=1,
        )

        # Track 2: Ultrasonic arrival track (arrives at ultrasonic station in ~0.4s)
        coordinator.register_fruit(
            bbox=(250, 140, 370, 260),
            visual_label="fresh",
            visual_confidence=0.96,
            entry_time=start_time - (t_to_us - 0.4),
            track_id=2,
        )

        # Track 3: Rotten defect track (arrives at ultrasonic station in ~1.2s)
        coordinator.register_fruit(
            bbox=(400, 160, 520, 280),
            visual_label="rotten",
            visual_confidence=0.93,
            entry_time=start_time - (t_to_us - 1.2),
            track_id=3,
        )

        # Track 4: Fresh entry track (enters now, in transit)
        coordinator.register_fruit(
            bbox=(550, 150, 670, 270),
            visual_label="fresh",
            visual_confidence=0.98,
            entry_time=start_time,
            track_id=4,
        )

    print(
        f"Conveyor pipeline started (mock={args.mock}, speed={speed:.3f} m/s, "
        f"max_safe={coordinator.max_safe_speed_mps:.3f} m/s)"
    )

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if args.duration is not None and elapsed >= args.duration:
                break
            coordinator.step(timestamp=now)
            if args.target_fps is not None and args.target_fps > 0:
                target_delay = (1.0 / args.target_fps) - (time.time() - now)
                if target_delay > 0:
                    time.sleep(target_delay)
            elif args.mock:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        coordinator.stop()
        if args.csv_out:
            coordinator.save_csv(args.csv_out)
            print(f"Saved {len(coordinator.all_tracks)} track records to {args.csv_out}")

    # Display summary table
    print("\n" + "=" * 80)
    print("CONVEYOR PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    header = f"{'ID':<4} {'Visual':<10} {'Conf':<6} {'ToF (us)':<10} {'Amp':<6} {'Grade':<16} {'Status':<12} {'Fallback'}"
    print(header)
    print("-" * 80)
    for t in coordinator.all_tracks:
        tof_str = f"{t.ultrasonic_travel_time_us:.1f}" if t.ultrasonic_travel_time_us is not None else "-"
        amp_str = str(t.ultrasonic_amplitude) if t.ultrasonic_amplitude is not None else "-"
        grade_str = t.final_grade or "-"
        print(
            f"{t.track_id:<4} {t.visual_label:<10} {t.visual_confidence:<6.2f} "
            f"{tof_str:<10} {amp_str:<6} {grade_str:<16} {t.status:<12} {t.internal_fallback}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

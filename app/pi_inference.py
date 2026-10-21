"""Live multi-apple vision for the Raspberry Pi 4.

Stage 1 uses a YOLO detector to find and track every apple in a frame.
Stage 2 runs the trained fresh/rotten classifier on the crop for each tracked apple.
The resulting vision track ID is the ID that the encoder/ultrasonic integration must
associate with the upstream ultrasonic measurement before sorting.

Standard Lightweight Architecture for Raspberry Pi 4:
- OpenCV DNN (cv2.dnn) for ONNX model execution (ARM NEON accelerated).
- Zero PyTorch, torchvision, or Ultralytics runtime bloat.
- Pure-Python / NumPy ConveyorTracker (zero C++ compilation, zero lap/lapx).
"""
import argparse
import csv
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    cv2.setNumThreads(3)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

logger = logging.getLogger("pi_inference")

BAD_LABELS = {"rotten", "rot", "bruise", "spot", "mold", "mould", "damage", "damaged"}

# Standard 80-class COCO mapping (Class 47 is 'apple')
COCO_CLASSES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}


# ==============================================================================
# Lightweight Tensor / Result Wrappers (Ultralytics API Compatibility)
# ==============================================================================
class _ArrayWrapper:
    """Wraps sequences with .cpu().tolist() and .int().cpu().tolist() compatibility."""

    def __init__(self, data: Sequence[Any]):
        self._data = list(data)

    def cpu(self) -> "_ArrayWrapper":
        return self

    def tolist(self) -> list:
        return list(self._data)

    def int(self) -> "_ArrayWrapper":
        return _ArrayWrapper([int(x) for x in self._data])

    def float(self) -> "_ArrayWrapper":
        return _ArrayWrapper([float(x) for x in self._data])

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __repr__(self) -> str:
        return f"_ArrayWrapper({self._data})"


class BoxesResult:
    """Ultralytics-compatible boxes container."""

    def __init__(
        self,
        xyxy: Sequence[Sequence[Union[int, float]]],
        ids: Optional[Sequence[int]] = None,
        confidences: Optional[Sequence[float]] = None,
        class_ids: Optional[Sequence[int]] = None,
    ):
        self._xyxy = [list(box) for box in xyxy]
        self._ids = list(ids) if ids is not None else None
        self._confidences = list(confidences) if confidences is not None else [1.0] * len(self._xyxy)
        self._class_ids = list(class_ids) if class_ids is not None else [0] * len(self._xyxy)

    @property
    def xyxy(self) -> _ArrayWrapper:
        return _ArrayWrapper(self._xyxy)

    @property
    def id(self) -> Optional[_ArrayWrapper]:
        return _ArrayWrapper(self._ids) if self._ids is not None else None

    @property
    def conf(self) -> _ArrayWrapper:
        return _ArrayWrapper(self._confidences)

    @property
    def cls(self) -> _ArrayWrapper:
        return _ArrayWrapper(self._class_ids)

    def __len__(self) -> int:
        return len(self._xyxy)


class DetectionsResult:
    """Ultralytics-compatible detection result wrapper."""

    def __init__(self, boxes: BoxesResult):
        self.boxes = boxes


class _ProbsWrapper:
    """Ultralytics-compatible classifier probabilities container."""

    def __init__(self, top1: int, top1conf: float, data: Optional[Dict[int, float]] = None):
        self.top1 = int(top1)
        self.top1conf = float(top1conf)
        self.data = data if data is not None else {top1: top1conf}


class ClassificationResult:
    """Ultralytics-compatible classification result wrapper."""

    def __init__(self, names: Dict[int, str], top1: int, top1conf: float, data: Optional[Dict[int, float]] = None):
        self.names = dict(names)
        self.probs = _ProbsWrapper(top1=top1, top1conf=top1conf, data=data)


# ==============================================================================
# Pure-Python / NumPy Conveyor Belt Tracker
# ==============================================================================
@dataclass
class TrackedObject:
    track_id: int
    box: List[int]
    cx: float
    cy: float
    last_seen: float
    missed_frames: int = 0
    confidence: float = 1.0
    class_id: int = 0


class ConveyorTracker:
    """Pure-Python / NumPy spatial tracker optimized for conveyor belt sorting.

    Associates bounding boxes across frames using predicted centroid displacement
    based on conveyor velocity and Euclidean spatial gating.
    Requires ZERO C++ compilers, zero lap/lapx, and zero ByteTrack dependencies.
    """

    def __init__(
        self,
        max_distance: float = 120.0,
        max_missed: int = 15,
        velocity_px_s: Tuple[float, float] = (0.0, 0.0),
    ):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.velocity_px_s = velocity_px_s  # (vx, vy) in pixels/sec
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 1
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear all active tracks and reset counter."""
        self.tracks.clear()
        self._next_track_id = 1
        self._last_timestamp = None

    def update(
        self,
        boxes: Sequence[Sequence[Union[int, float]]],
        confidences: Optional[Sequence[float]] = None,
        class_ids: Optional[Sequence[int]] = None,
        timestamp: Optional[float] = None,
    ) -> List[Tuple[List[int], int]]:
        """Update tracker with detections from current frame.

        Args:
            boxes: Sequence of [x1, y1, x2, y2] bounding boxes.
            confidences: Optional confidence score for each detection.
            class_ids: Optional class ID for each detection.
            timestamp: Frame timestamp in seconds.

        Returns:
            List of (box, track_id) tuples for all active tracks in this frame.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = (now - self._last_timestamp) if self._last_timestamp is not None else 0.0
        dt = max(0.0, min(dt, 1.0))
        self._last_timestamp = now

        num_dets = len(boxes)
        if confidences is None:
            confs = [1.0] * num_dets
        else:
            confs = list(confidences)

        if class_ids is None:
            cids = [0] * num_dets
        else:
            cids = list(class_ids)

        det_centroids: List[Tuple[float, float]] = []
        clean_boxes: List[List[int]] = []
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            clean_boxes.append([x1, y1, x2, y2])
            det_centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        vx, vy = self.velocity_px_s
        matched_track_ids = set()
        matched_det_indices = set()
        results: List[Tuple[List[int], int]] = []

        if self.tracks and num_dets > 0:
            track_ids = list(self.tracks.keys())
            pred_centroids = [
                (self.tracks[tid].cx + vx * dt, self.tracks[tid].cy + vy * dt)
                for tid in track_ids
            ]

            costs: List[Tuple[float, int, int]] = []
            for t_idx, (t_cx, t_cy) in enumerate(pred_centroids):
                for d_idx, (d_cx, d_cy) in enumerate(det_centroids):
                    dist = float(np.hypot(t_cx - d_cx, t_cy - d_cy))
                    costs.append((dist, t_idx, d_idx))

            costs.sort(key=lambda x: x[0])

            for dist, t_idx, d_idx in costs:
                tid = track_ids[t_idx]
                if tid in matched_track_ids or d_idx in matched_det_indices:
                    continue
                if dist <= self.max_distance:
                    matched_track_ids.add(tid)
                    matched_det_indices.add(d_idx)
                    trk = self.tracks[tid]
                    trk.box = clean_boxes[d_idx]
                    trk.cx, trk.cy = det_centroids[d_idx]
                    trk.confidence = float(confs[d_idx])
                    trk.class_id = int(cids[d_idx])
                    trk.last_seen = now
                    trk.missed_frames = 0
                    results.append((trk.box, tid))

        for d_idx in range(num_dets):
            if d_idx not in matched_det_indices:
                new_id = self._next_track_id
                self._next_track_id += 1
                bx = clean_boxes[d_idx]
                cx, cy = det_centroids[d_idx]
                trk = TrackedObject(
                    track_id=new_id,
                    box=bx,
                    cx=cx,
                    cy=cy,
                    last_seen=now,
                    missed_frames=0,
                    confidence=float(confs[d_idx]),
                    class_id=int(cids[d_idx]),
                )
                self.tracks[new_id] = trk
                results.append((bx, new_id))

        dead_tracks = []
        for tid, trk in self.tracks.items():
            if tid not in matched_track_ids and tid not in [r[1] for r in results]:
                trk.missed_frames += 1
                if trk.missed_frames > self.max_missed:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.tracks[tid]

        return results

    def propagate(self, timestamp: Optional[float] = None) -> List[Tuple[List[int], int]]:
        """Kinematically update all active tracks based on elapsed time and conveyor velocity.

        Args:
            timestamp: Frame timestamp in seconds. Defaults to time.monotonic().

        Returns:
            List of (box, track_id) tuples for all updated tracks.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = (now - self._last_timestamp) if self._last_timestamp is not None else 0.0
        dt = max(0.0, min(dt, 1.0))
        self._last_timestamp = now

        vx, vy = self.velocity_px_s
        dx = vx * dt
        dy = vy * dt

        for trk in self.tracks.values():
            trk.cx += dx
            trk.cy += dy
            x1, y1, x2, y2 = trk.box
            trk.box = [int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)]
            trk.last_seen = now

        return [(trk.box, tid) for tid, trk in self.tracks.items()]


# ==============================================================================
# OpenCV DNN Lightweight Inference Engine (ARM NEON Accelerated)
# ==============================================================================
class OpenCVInferenceEngine:
    """Lightweight OpenCV DNN inference engine for YOLO detection and classification.

    Loads ONNX models via cv2.dnn.readNetFromONNX() and executes forward passes
    without any PyTorch or Ultralytics dependencies.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        task: str = "auto",
        input_size: Optional[Tuple[int, int]] = None,
        conf_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Determine task
        name_lower = str(self.model_path).lower()
        if task == "auto":
            if "best" in name_lower or "classify" in name_lower or "cls" in name_lower:
                self.task = "classify"
            else:
                self.task = "detect"
        else:
            self.task = task.lower()

        # Set default input size
        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size = (224, 224) if self.task == "classify" else (416, 416)

        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

        # Class names dictionary
        if class_names is not None:
            self.names = dict(class_names)
        elif self.task == "classify":
            self.names = {0: "fresh", 1: "rotten"}
        else:
            self.names = dict(COCO_CLASSES)

        logger.info("Initializing OpenCVInferenceEngine from %s (task=%s, size=%s)", self.model_path, self.task, self.input_size)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.tracker = ConveyorTracker(max_distance=120.0, max_missed=15)

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        classes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[List[int]], List[float], List[int]]:
        """Run YOLO object detection on a frame using OpenCV DNN.

        Returns:
            Tuple of (boxes, confidences, class_ids), where boxes are [x1, y1, x2, y2].
        """
        conf_thresh = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        nms_thresh = self.nms_threshold if nms_threshold is None else float(nms_threshold)

        orig_h, orig_w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Handle output dimensions: YOLOv8 / YOLOv11 output is typically (1, 4 + nc, N)
        if outputs.ndim == 3:
            if outputs.shape[1] < outputs.shape[2]:
                outputs = np.transpose(outputs, (0, 2, 1))
            preds = outputs[0]  # shape (N, 4 + nc)
        elif outputs.ndim == 2:
            preds = outputs
        else:
            return [], [], []

        if preds.shape[0] == 0:
            return [], [], []

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        scores_matrix = preds[:, 4:]

        num_classes = scores_matrix.shape[1]
        if classes is not None:
            valid_classes = [c for c in classes if 0 <= c < num_classes]
            if not valid_classes:
                return [], [], []
            if len(valid_classes) == 1:
                target_c = valid_classes[0]
                scores = scores_matrix[:, target_c]
                class_ids = np.full(len(scores), target_c, dtype=int)
            else:
                sub_scores = scores_matrix[:, valid_classes]
                sub_best = np.argmax(sub_scores, axis=1)
                scores = np.max(sub_scores, axis=1)
                class_ids = np.array(valid_classes)[sub_best]
        else:
            class_ids = np.argmax(scores_matrix, axis=1)
            scores = np.max(scores_matrix, axis=1)

        mask = scores >= conf_thresh
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return [], [], []

        scale_x = orig_w / float(self.input_size[0])
        scale_y = orig_h / float(self.input_size[1])

        nms_boxes: List[List[int]] = []
        nms_scores: List[float] = []
        nms_cids: List[int] = []

        for idx in valid_indices:
            box_cx, box_cy, box_w, box_h = cx[idx], cy[idx], w[idx], h[idx]
            bx1 = int((box_cx - box_w / 2.0) * scale_x)
            by1 = int((box_cy - box_h / 2.0) * scale_y)
            bw = int(box_w * scale_x)
            bh = int(box_h * scale_y)
            nms_boxes.append([bx1, by1, bw, bh])
            nms_scores.append(float(scores[idx]))
            nms_cids.append(int(class_ids[idx]))

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thresh, nms_thresh)
        if len(indices) == 0:
            return [], [], []

        flat_indices = np.array(indices).flatten().tolist()
        final_boxes: List[List[int]] = []
        final_scores: List[float] = []
        final_classes: List[int] = []

        for idx in flat_indices:
            b = nms_boxes[idx]
            x1 = max(0, b[0])
            y1 = max(0, b[1])
            x2 = min(orig_w, b[0] + b[2])
            y2 = min(orig_h, b[1] + b[3])
            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(nms_scores[idx])
            final_classes.append(nms_cids[idx])

        return final_boxes, final_scores, final_classes

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """Run classification on a cropped fruit image using OpenCV DNN.

        Returns:
            Tuple of (predicted_label_lower, confidence_float).
        """
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        logits = outputs.flatten()
        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        top1_id = int(np.argmax(probs))
        top1_conf = float(probs[top1_id])

        label = self.names.get(top1_id, str(top1_id)).lower()
        return label, top1_conf

    def predict(self, source: np.ndarray, imgsz: Optional[int] = None, verbose: bool = False, **kwargs) -> list:
        """Ultralytics-compatible predict interface."""
        if imgsz is not None:
            self.input_size = (imgsz, imgsz)

        if self.task == "classify":
            label, conf = self.classify(source)
            top1_id = 0
            for cid, name in self.names.items():
                if name.lower() == label:
                    top1_id = cid
                    break
            probs_dict = {top1_id: conf}
            return [ClassificationResult(names=self.names, top1=top1_id, top1conf=conf, data=probs_dict)]

        # Detection task
        boxes, scores, class_ids = self.detect(source, **kwargs)
        boxes_res = BoxesResult(xyxy=boxes, ids=None, confidences=scores, class_ids=class_ids)
        return [DetectionsResult(boxes=boxes_res)]

    def track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        classes: Optional[Sequence[int]] = None,
        imgsz: Optional[int] = None,
        conf: float = 0.30,
        tracker: Optional[str] = None,
        verbose: bool = False,
        skip_detect: bool = False,
        **kwargs,
    ) -> list:
        """Ultralytics-compatible track interface using pure-Python ConveyorTracker."""
        if imgsz is not None:
            self.input_size = (imgsz, imgsz)

        ts = kwargs.get("timestamp", None)
        if skip_detect:
            tracked = self.tracker.propagate(timestamp=ts)
        else:
            boxes, scores, class_ids = self.detect(frame, conf_threshold=conf, classes=classes)
            tracked = self.tracker.update(boxes, confidences=scores, class_ids=class_ids, timestamp=ts)

        tracked_boxes = [t[0] for t in tracked]
        tracked_ids = [t[1] for t in tracked]

        # Extract matching confidences and class IDs for tracked boxes
        tracked_confs = []
        tracked_cids = []
        for tid in tracked_ids:
            if tid in self.tracker.tracks:
                trk = self.tracker.tracks[tid]
                tracked_confs.append(trk.confidence)
                tracked_cids.append(trk.class_id)
            else:
                tracked_confs.append(1.0)
                tracked_cids.append(0)

        boxes_res = BoxesResult(
            xyxy=tracked_boxes,
            ids=tracked_ids,
            confidences=tracked_confs,
            class_ids=tracked_cids,
        )
        return [DetectionsResult(boxes=boxes_res)]

    def __call__(self, frame: np.ndarray, **kwargs) -> list:
        return self.predict(frame, **kwargs)


# ==============================================================================
# Lazy Ultralytics Fallback Engine (PC / Workstation Only)
# ==============================================================================
class UltralyticsInferenceEngine:
    """Lazy fallback wrapper for Ultralytics YOLO models (PC / workstation only)."""

    def __init__(self, model_path: Union[str, Path]):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. To run on Raspberry Pi 4 without PyTorch,\n"
                "export your model to ONNX using 'python scripts/export_to_onnx.py'\n"
                "and load with OpenCVInferenceEngine."
            ) from exc

        self.model = YOLO(str(model_path))
        self.names = self.model.names

    def track(self, *args, **kwargs):
        return self.model.track(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        res = self.model.predict(crop, imgsz=224, verbose=False)[0]
        label = res.names[res.probs.top1].lower()
        return label, float(res.probs.top1conf)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_vision_engine(
    model_path: Union[str, Path],
    task: str = "auto",
    classes: Optional[Dict[int, str]] = None,
    prefer_onnx: bool = True,
):
    """Factory function: load OpenCV DNN engine (preferred on Pi 4) or lazy Ultralytics fallback."""
    p = Path(model_path)

    # 1. Direct ONNX model path
    if p.suffix.lower() == ".onnx":
        if not p.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {p}")
        return OpenCVInferenceEngine(p, task=task, class_names=classes)

    # 2. PyTorch .pt model path: check for sibling .onnx first
    onnx_sibling = p.with_suffix(".onnx")
    if prefer_onnx and onnx_sibling.is_file():
        logger.info("Found sibling ONNX model %s for %s; using OpenCV DNN engine", onnx_sibling, p)
        return OpenCVInferenceEngine(onnx_sibling, task=task, class_names=classes)

    # 3. If .pt file exists and no .onnx: try lazy Ultralytics
    if p.is_file():
        try:
            return UltralyticsInferenceEngine(p)
        except ImportError:
            raise FileNotFoundError(
                f"Model file '{p}' is a PyTorch weight file (.pt), but 'ultralytics' is not installed.\n"
                f"For Raspberry Pi 4 deployment without PyTorch bloatware, export this model to ONNX:\n"
                f"  python scripts/export_to_onnx.py --detector models/detector/yolo11n.pt --classifier models/external/best.pt\n"
                f"Or place '{onnx_sibling.name}' in the model directory."
            )

    raise FileNotFoundError(f"Model file not found: {p}")


# ==============================================================================
# Helper Functions & Utilities
# ==============================================================================
def find_class_id(model: Any, class_name: str) -> int:
    """Find integer class ID for a given class name from model.names."""
    names_dict = getattr(model, "names", {})
    if isinstance(names_dict, dict):
        for class_id, name in names_dict.items():
            if str(name).lower() == class_name.lower():
                return int(class_id)
    elif isinstance(names_dict, (list, tuple)):
        for class_id, name in enumerate(names_dict):
            if str(name).lower() == class_name.lower():
                return class_id

    # Fallback default for standard COCO apple class
    if class_name.lower() == "apple":
        return 47

    raise ValueError(f"Detector does not contain the class '{class_name}'. Available: {names_dict}")


def crop_box(frame: np.ndarray, box: Sequence[Union[int, float]], padding: int = 8) -> np.ndarray:
    """Extract a cropped sub-image bounded by [x1, y1, x2, y2] with optional padding."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    return frame[y1:y2, x1:x2]


def classify_apple(model: Any, crop: np.ndarray) -> Tuple[str, float]:
    """Classify a single apple crop using the loaded classifier model."""
    if isinstance(model, OpenCVInferenceEngine):
        return model.classify(crop)

    # Ultralytics or Mock model predict()
    result = model.predict(crop, imgsz=224, verbose=False)[0]
    label = result.names[result.probs.top1].lower()
    return label, float(result.probs.top1conf)


@dataclass
class TrackState:
    labels: Counter = field(default_factory=Counter)
    confidences: list = field(default_factory=list)
    emitted: bool = False
    last_seen: float = 0.0


class TrackDecisions:
    """Stabilise classifications across several frames of the same apple."""

    def __init__(self, minimum_frames: int):
        self.minimum_frames = minimum_frames
        self.items: Dict[int, TrackState] = {}

    def observe(self, track_id: int, label: str, confidence: float) -> Tuple[str, float, bool]:
        state = self.items.setdefault(track_id, TrackState())
        state.labels[label] += 1
        state.confidences.append(confidence)
        state.last_seen = time.monotonic()
        final_label, votes = state.labels.most_common(1)[0]
        mean_confidence = sum(state.confidences) / len(state.confidences)
        return final_label, mean_confidence, votes >= self.minimum_frames and not state.emitted

    def mark_emitted(self, track_id: int) -> None:
        if track_id in self.items:
            self.items[track_id].emitted = True

    def purge_old(self, seconds: float = 5.0) -> None:
        now = time.monotonic()
        for track_id in list(self.items):
            if now - self.items[track_id].last_seen > seconds:
                del self.items[track_id]


def open_camera(args: Any):
    """Initialize camera device using Picamera2 or OpenCV backend."""
    if args.camera_backend in ("auto", "picamera2"):
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
            picam.configure(config)
            picam.start()
            print("Using Raspberry Pi Camera Module via Picamera2")
            return picam, None
        except ImportError:
            if args.camera_backend == "picamera2":
                raise RuntimeError("Picamera2 is not installed. Use a USB camera or install Picamera2.")
        except Exception as error:
            if args.camera_backend == "picamera2":
                raise RuntimeError(f"Could not start Pi camera: {error}") from error
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open USB camera")
    print("Using USB webcam via OpenCV")
    return None, camera


def main() -> None:
    default_detector = (
        "models/detector/yolo11n.onnx"
        if Path("models/detector/yolo11n.onnx").is_file()
        else "models/detector/yolo11n.pt"
    )
    default_classifier = (
        "models/external/best.onnx"
        if Path("models/external/best.onnx").is_file()
        else "models/external/best.pt"
    )

    parser = argparse.ArgumentParser(description="Live multi-apple vision for Raspberry Pi 4.")
    parser.add_argument("--detector-model", default=default_detector, help="Path to detector weights (.onnx or .pt).")
    parser.add_argument("--classifier-model", default=default_classifier, help="Path to classifier weights (.onnx or .pt).")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI display (no cv2.imshow).")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "opencv", "picamera2"), default="auto")
    parser.add_argument("--minimum-frames", type=int, default=3, help="Matching classifications required before decision.")
    parser.add_argument("--detect-cadence", type=int, default=2, help="Run full detection every N frames (default: 2).")
    parser.add_argument("--log", default="runtime/vision_results.csv")
    args = parser.parse_args()

    # Auto-detect headless if DISPLAY is absent/empty or explicitly requested
    is_headless = args.headless or not os.environ.get("DISPLAY")
    if is_headless and not args.headless:
        logger.info("No DISPLAY detected in environment. Running in headless mode.")

    detector = load_vision_engine(args.detector_model, task="detect")
    classifier = load_vision_engine(args.classifier_model, task="classify")
    apple_class = find_class_id(detector, "apple")

    decisions = TrackDecisions(args.minimum_frames)
    picam, camera = open_camera(args)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()

    try:
        with open(log_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("timestamp", "vision_track_id", "external_label", "confidence", "vision_decision"))
            if is_new:
                writer.writeheader()
            print("Running multi-apple vision. Press " + ("Ctrl+C" if is_headless else "q or Ctrl+C") + " to stop.")
            frame_idx = 0
            while True:
                frame_idx += 1
                if picam is not None:
                    frame = cv2.cvtColor(picam.capture_array(), cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = camera.read()
                    if not ok:
                        time.sleep(0.01)
                        continue

                skip_detect = (args.detect_cadence > 1) and (frame_idx % args.detect_cadence != 0)
                result = detector.track(
                    frame,
                    persist=True,
                    classes=[apple_class],
                    imgsz=416,
                    conf=0.30,
                    verbose=False,
                    skip_detect=skip_detect,
                )[0]

                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist()):
                        crop = crop_box(frame, box)
                        if crop.size == 0:
                            continue
                        label, confidence = classify_apple(classifier, crop)
                        final_label, mean_confidence, ready = decisions.observe(track_id, label, confidence)
                        vision_decision = "SECOND_QUALITY" if final_label in BAD_LABELS else "SIZE_GRADING"
                        x1, y1, x2, y2 = (int(value) for value in box)
                        color = (0, 0, 255) if vision_decision == "SECOND_QUALITY" else (0, 200, 0)
                        if not is_headless:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                f"#{track_id} {final_label} {mean_confidence:.0%}",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                color,
                                2,
                            )
                        if ready:
                            writer.writerow({
                                "timestamp": time.time(),
                                "vision_track_id": track_id,
                                "external_label": final_label,
                                "confidence": f"{mean_confidence:.4f}",
                                "vision_decision": vision_decision,
                            })
                            handle.flush()
                            decisions.mark_emitted(track_id)
                            print(f"Vision track #{track_id}: {vision_decision} ({final_label}, {mean_confidence:.0%})")
                decisions.purge_old()
                if not is_headless:
                    cv2.imshow("Fruit sorter: multi-apple vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("\nMulti-apple vision stopped by user.")
    finally:
        if picam is not None:
            picam.stop()
        if camera is not None:
            camera.release()
        if not is_headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

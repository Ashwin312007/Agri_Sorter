"""conveyor-aware ByteTrack and sticky decision state machine tracker for Agri_Sorter.

Maintains persistent track IDs for fruits traversing the conveyor belt,
implements a sticky defect state machine (FRESH -> REJECT with consecutive-frame filtering),
and manages an arrival-order FIFO queue upon camera egress.
"""

from collections import Counter
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.contracts import (
    FruitDetection,
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    TrackState,
    is_defect_class,
    is_fresh_class,
    load_app_config,
    parse_compound_label,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Bounding Box Geometry & Cost Utilities
# ============================================================================

def bbox_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes_a: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes_b: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        Array of shape (N, M) containing IoU values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (N, M, 2)
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (N, M, 2)
    wh = np.clip(rb - lt, a_min=0, a_max=None)                   # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]                            # (N, M)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Solve minimum cost bipartite matching with a gating threshold.

    Args:
        cost_matrix: Cost matrix of shape (N, M).
        thresh: Maximum allowed cost for an acceptable match.

    Returns:
        matched_indices: Array of shape (K, 2) where each row is (row_idx, col_idx).
        unmatched_rows: List of row indices not matched.
        unmatched_cols: List of column indices not matched.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matched.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    matched_arr = np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)
    return matched_arr, sorted(list(unmatched_rows)), sorted(list(unmatched_cols))


# ============================================================================
# Kalman Filter for Bounding Box Tracking
# ============================================================================

class KalmanBoxFilter:
    """8D Kalman Filter for tracking bounding boxes [cx, cy, w, h, vx, vy, vw, vh].

    Provides kinematic state propagation and measurement updates optimized for
    moving objects such as conveyor fruit.
    """

    def __init__(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement projection matrix H (maps 8D state to [cx, cy, w, h])
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Motion and observation noise scales
        std_pos = [max(1.0, w * 0.05), max(1.0, h * 0.05), max(1.0, w * 0.05), max(1.0, h * 0.05)]
        std_vel = [max(1.0, w * 0.01), max(1.0, h * 0.01), max(1.0, w * 0.01), max(1.0, h * 0.01)]

        # Covariance matrices
        self.P = np.diag(np.square(std_pos + [s * 10.0 for s in std_vel])).astype(np.float32)
        self.Q = np.diag(np.square(std_pos + std_vel)).astype(np.float32)
        self.R = np.diag(np.square(std_pos)).astype(np.float32)

    def predict(self) -> Tuple[int, int, int, int]:
        """Propagate state estimate forward one time step."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # Clamp dimensions to valid positive numbers
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.get_bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Update state estimate with new measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Innovation
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.get_bbox()

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Extract [x1, y1, x2, y2] bounding box from current state."""
        cx, cy, w, h = self.x[:4]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return (x1, y1, x2, y2)


# ============================================================================
# Internal Track Entity & Sticky Decision State Machine
# ============================================================================

class _FruitTrack:
    """Internal fruit track entity maintaining trajectory and classification state."""

    def __init__(
        self,
        track_id: int,
        detection: FruitDetection,
        timestamp: float,
        consecutive_defect_threshold: int = 3,
    ):
        self.track_id = track_id
        self.kf = KalmanBoxFilter(detection.bbox)
        self.bbox = detection.bbox
        self.predicted_bbox = detection.bbox

        # Sticky Decision State Machine variables
        self.status: str = STATUS_FRESH
        self.defect_classes: Set[str] = set()
        self.consecutive_defect_count: int = 0
        self.consecutive_defect_threshold = max(1, consecutive_defect_threshold)

        # Observations and confidence tracking
        self.fruit_type_votes: Counter[str] = Counter()
        self.confidences: List[float] = []
        self.current_condition: str = "fresh"
        self.total_frames: int = 0
        self.missed_frames: int = 0

        # Egress latch
        self.has_egressed: bool = False
        self.first_seen: float = timestamp
        self.last_seen: float = timestamp

        # Apply initial detection observation
        self.apply_detection(detection, timestamp)

    def predict(self) -> Tuple[int, int, int, int]:
        """Propagate motion model and record predicted bounding box."""
        self.predicted_bbox = self.kf.predict()
        self.bbox = self.predicted_bbox
        return self.predicted_bbox

    def apply_detection(self, detection: FruitDetection, timestamp: float) -> None:
        """Update track with a newly matched detection and advance the sticky state machine."""
        self.bbox = self.kf.update(detection.bbox)
        self.predicted_bbox = self.bbox
        self.confidences.append(float(detection.confidence))
        self.last_seen = timestamp
        self.missed_frames = 0
        self.total_frames += 1

        # Track fruit type voting (e.g. apple, banana)
        f_type = detection.fruit_type.strip().lower() if detection.fruit_type else "unknown"
        if f_type and f_type != "unknown":
            self.fruit_type_votes[f_type] += 1
        elif "_" in detection.class_name:
            inferred_fruit, _ = parse_compound_label(detection.class_name)
            self.fruit_type_votes[inferred_fruit] += 1

        # Extract defect condition
        cond_raw = detection.condition.strip().lower() if detection.condition else ""
        if not cond_raw or cond_raw == "unknown":
            _, cond_raw = parse_compound_label(detection.class_name)
        self.current_condition = cond_raw

        # Determine if detection is defective
        is_defect = (
            detection.is_defect
            or is_defect_class(cond_raw)
            or is_defect_class(detection.class_name)
        )

        # ====================================================================
        # Sticky Decision State Machine Logic:
        # 1. Once status == "REJECT", it CAN NEVER go back to "FRESH".
        # 2. Defect class must appear in N consecutive frames to flip FRESH -> REJECT.
        # 3. Accumulate all observed defect classes across transit.
        # ====================================================================
        if is_defect:
            # Register specific defect class condition (e.g., 'bruised', 'bug_bite')
            clean_defect = cond_raw
            if clean_defect.endswith("_fresh") or clean_defect == "fresh":
                clean_defect = "defect"
            elif "_" in clean_defect:
                _, clean_defect = parse_compound_label(clean_defect)

            if clean_defect and clean_defect != "fresh":
                self.defect_classes.add(clean_defect)

            self.consecutive_defect_count += 1
            if self.consecutive_defect_count >= self.consecutive_defect_threshold:
                self.status = STATUS_REJECT
        else:
            # Detection is FRESH
            self.consecutive_defect_count = 0
            # Sticky rule: If already REJECT, preserve REJECT permanently!
            # If still FRESH, it remains FRESH.

    @property
    def fruit_type(self) -> str:
        """Dominant fruit type determined by accumulated voting."""
        if self.fruit_type_votes:
            return self.fruit_type_votes.most_common(1)[0][0]
        return "fruit"

    @property
    def mean_confidence(self) -> float:
        """Arithmetic mean of confidences observed for this track."""
        if not self.confidences:
            return 1.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center coordinate (cx, cy) of current bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def to_track_state(self) -> TrackState:
        """Construct user-facing TrackState snapshot."""
        return TrackState(
            track_id=self.track_id,
            bbox=self.bbox,
            fruit_type=self.fruit_type,
            status=self.status,
            defect_classes=set(self.defect_classes),
            confidence=round(self.mean_confidence, 4),
            consecutive_defect_count=self.consecutive_defect_count,
            has_egressed=self.has_egressed,
            missed_frames=self.missed_frames,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            current_condition=self.current_condition,
            total_frames=self.total_frames,
        )

    def to_fruit_event(self, egress_time: Optional[float] = None) -> FruitEvent:
        """Construct final immutable FruitEvent for FIFO queue insertion."""
        ts = egress_time if egress_time is not None else time.time()
        return FruitEvent(
            track_id=self.track_id,
            fruit_type=self.fruit_type,
            final_status=self.status,
            defect_classes=sorted(list(self.defect_classes)),
            confidence=round(self.mean_confidence, 4),
            egress_timestamp=float(ts),
        )


# ============================================================================
# Main Conveyor Fruit Tracker
# ============================================================================

class FruitTracker:
    """Multi-fruit conveyor tracker combining ByteTrack association,

    sticky defect state machine, and FIFO egress event queuing.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracker_type: Optional[str] = None,
        consecutive_defect_frames: Optional[int] = None,
        egress_line_y_ratio: Optional[float] = None,
        track_buffer: Optional[int] = None,
        match_thresh: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        egress_direction: str = "y",
    ):
        """Initialize FruitTracker.

        Args:
            config: Full configuration dictionary. Defaults to config/config.yaml.
            tracker_type: Tracking backend ("bytetrack" or "conveyor").
            consecutive_defect_frames: N consecutive defect frames required to flip FRESH -> REJECT.
            egress_line_y_ratio: Frame Y ratio threshold denoting camera exit (default 0.85).
            track_buffer: Maximum frames to retain lost tracks before purging (default 30).
            match_thresh: IoU association threshold for matching (default 0.7).
            conf_threshold: High/low detection confidence partition threshold (default 0.40).
            egress_direction: Direction of conveyor movement ("y" or "x").
        """
        # Load default configuration if not explicitly provided
        if config is None:
            try:
                cfg = load_app_config()
            except Exception:
                cfg = {}
        else:
            cfg = config

        trk_cfg = cfg.get("tracking", {})
        det_cfg = cfg.get("detection", {})

        self.tracker_type = tracker_type or trk_cfg.get("tracker_type", "bytetrack")
        self.consecutive_defect_frames = int(
            consecutive_defect_frames
            if consecutive_defect_frames is not None
            else trk_cfg.get("consecutive_defect_frames", 3)
        )
        self.egress_line_y_ratio = float(
            egress_line_y_ratio
            if egress_line_y_ratio is not None
            else trk_cfg.get("egress_line_y_ratio", 0.85)
        )
        self.track_buffer = int(
            track_buffer if track_buffer is not None else trk_cfg.get("track_buffer", 30)
        )
        self.match_thresh = float(
            match_thresh if match_thresh is not None else trk_cfg.get("match_thresh", 0.7)
        )
        self.conf_threshold = float(
            conf_threshold if conf_threshold is not None else det_cfg.get("conf_threshold", 0.40)
        )
        self.egress_direction = egress_direction.lower()

        # Track storage
        self.tracks: Dict[int, _FruitTrack] = {}
        self._next_track_id: int = 1
        self._frame_count: int = 0
        self._lock = threading.Lock()

        # Thread-safe FIFO queue holding FruitEvent records in arrival order
        self.fifo_queue: queue.Queue[FruitEvent] = queue.Queue()

        logger.info(
            f"FruitTracker initialized: type={self.tracker_type}, "
            f"consecutive_defect_frames={self.consecutive_defect_frames}, "
            f"egress_line_y_ratio={self.egress_line_y_ratio}, "
            f"track_buffer={self.track_buffer}"
        )

    def get_fifo_queue(self) -> queue.Queue[FruitEvent]:
        """Return the thread-safe arrival-order FIFO queue of FruitEvents."""
        return self.fifo_queue

    def reset(self) -> None:
        """Reset tracker state, active tracks, and ID counter."""
        with self._lock:
            self.tracks.clear()
            self._next_track_id = 1
            self._frame_count = 0
            # Drain FIFO queue
            while not self.fifo_queue.empty():
                try:
                    self.fifo_queue.get_nowait()
                except queue.Empty:
                    break

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Retrieve current state of a specific track ID if active."""
        with self._lock:
            track = self.tracks.get(track_id)
            return track.to_track_state() if track else None

    def get_active_tracks(self) -> List[TrackState]:
        """Return snapshot list of all currently active track states."""
        with self._lock:
            return [trk.to_track_state() for trk in self.tracks.values()]

    def update(
        self,
        detections: List[FruitDetection],
        frame_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        timestamp: Optional[float] = None,
    ) -> List[TrackState]:
        """Update tracker with detections from current camera frame.

        Executes 2-stage ByteTrack bipartite association, applies sticky defect
        logic to all matched tracks, triggers camera egress pushes into FIFO queue
        for fruits crossing the threshold or leaving view, and cleans up dead tracks.

        Args:
            detections: List of FruitDetection instances from the detector.
            frame_shape: Frame dimensions (height, width) or (H, W, C).
            timestamp: Frame timestamp in seconds (defaults to time.time()).

        Returns:
            List of active TrackState objects in this frame.
        """
        now = time.time() if timestamp is None else float(timestamp)
        h, w = frame_shape[0], frame_shape[1]

        with self._lock:
            self._frame_count += 1

            # 1. Predict next locations for all existing tracks
            for trk in self.tracks.values():
                trk.predict()

            # 2. Partition detections into high and low confidence pools
            high_dets: List[Tuple[int, FruitDetection]] = []
            low_dets: List[Tuple[int, FruitDetection]] = []
            for idx, det in enumerate(detections):
                if det.confidence >= self.conf_threshold:
                    high_dets.append((idx, det))
                else:
                    low_dets.append((idx, det))

            active_track_ids = list(self.tracks.keys())
            matched_track_ids: Set[int] = set()

            # ----------------------------------------------------------------
            # First Association: Active Tracks <-> High Confidence Detections
            # ----------------------------------------------------------------
            if active_track_ids and high_dets:
                track_boxes = np.array([self.tracks[tid].bbox for tid in active_track_ids])
                det_boxes = np.array([det.bbox for _, det in high_dets])

                ious = bbox_ious(track_boxes, det_boxes)
                # Cost matrix: 1 - IoU (gated with centroid fallback for fast conveyor motion)
                cost_matrix = 1.0 - ious

                # Allow spatial distance gating if IoU is near 0
                max_dist = max(h, w) * 0.6
                for t_i, tid in enumerate(active_track_ids):
                    t_cx, t_cy = self.tracks[tid].centroid
                    for d_i, (_, det) in enumerate(high_dets):
                        if ious[t_i, d_i] < 0.1:
                            d_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                            d_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                            dist = float(np.hypot(t_cx - d_cx, t_cy - d_cy))
                            if dist < max_dist:
                                cost_matrix[t_i, d_i] = min(
                                    cost_matrix[t_i, d_i],
                                    0.7 + 0.2 * (dist / max_dist),
                                )

                # Association threshold (allow matching when cost <= 0.85)
                thresh = max(0.8, 1.0 - max(0.1, 1.0 - self.match_thresh))
                matched_pairs, unmatched_t_indices, unmatched_d_indices = linear_assignment(
                    cost_matrix, thresh=thresh
                )

                for t_idx, d_idx in matched_pairs:
                    tid = active_track_ids[t_idx]
                    _, det = high_dets[d_idx]
                    self.tracks[tid].apply_detection(det, now)
                    matched_track_ids.add(tid)

                unmatched_track_ids = [active_track_ids[i] for i in unmatched_t_indices]
                unmatched_high_dets = [high_dets[i] for i in unmatched_d_indices]
            else:
                unmatched_track_ids = active_track_ids
                unmatched_high_dets = high_dets

            # ----------------------------------------------------------------
            # Second Association: Remaining Tracks <-> Low Confidence Detections
            # ----------------------------------------------------------------
            if unmatched_track_ids and low_dets:
                track_boxes = np.array([self.tracks[tid].bbox for tid in unmatched_track_ids])
                det_boxes = np.array([det.bbox for _, det in low_dets])

                ious = bbox_ious(track_boxes, det_boxes)
                cost_matrix = 1.0 - ious

                matched_pairs, remaining_unmatched_t_indices, _ = linear_assignment(
                    cost_matrix, thresh=0.65
                )

                for t_idx, d_idx in matched_pairs:
                    tid = unmatched_track_ids[t_idx]
                    _, det = low_dets[d_idx]
                    self.tracks[tid].apply_detection(det, now)
                    matched_track_ids.add(tid)

                still_unmatched_track_ids = [
                    unmatched_track_ids[i] for i in remaining_unmatched_t_indices
                ]
            else:
                still_unmatched_track_ids = unmatched_track_ids

            # ----------------------------------------------------------------
            # Unmatched Tracks: Increment missed frame counter
            # ----------------------------------------------------------------
            for tid in still_unmatched_track_ids:
                self.tracks[tid].missed_frames += 1

            # ----------------------------------------------------------------
            # Unmatched High Detections: Initialize New Tracks
            # ----------------------------------------------------------------
            for _, det in unmatched_high_dets:
                new_id = self._next_track_id
                self._next_track_id += 1
                new_track = _FruitTrack(
                    track_id=new_id,
                    detection=det,
                    timestamp=now,
                    consecutive_defect_threshold=self.consecutive_defect_frames,
                )
                self.tracks[new_id] = new_track
                matched_track_ids.add(new_id)

            # ----------------------------------------------------------------
            # Egress Line Evaluation for Active Tracks
            # ----------------------------------------------------------------
            egress_threshold = (
                h * self.egress_line_y_ratio
                if self.egress_direction == "y"
                else w * self.egress_line_y_ratio
            )

            for tid, trk in self.tracks.items():
                if not trk.has_egressed:
                    cx, cy = trk.centroid
                    pos = cy if self.egress_direction == "y" else cx
                    bound = trk.bbox[3] if self.egress_direction == "y" else trk.bbox[2]
                    frame_limit = h if self.egress_direction == "y" else w

                    # Crosses egress line or boundary
                    if pos >= egress_threshold or bound >= frame_limit:
                        self._emit_egress(trk, now)

            # ----------------------------------------------------------------
            # Dead Tracks Removal (Leaves camera view)
            # ----------------------------------------------------------------
            dead_track_ids = []
            for tid, trk in self.tracks.items():
                # Culled if missed frames exceed buffer or completely left frame bounds
                completely_out = (
                    trk.bbox[1] >= h
                    if self.egress_direction == "y"
                    else trk.bbox[0] >= w
                )
                if trk.missed_frames > self.track_buffer or completely_out:
                    dead_track_ids.append(tid)
                    # When leaving camera view, if not already egressed, push FruitEvent
                    if not trk.has_egressed:
                        self._emit_egress(trk, now)

            for tid in dead_track_ids:
                del self.tracks[tid]

            # Return list of current active track states
            return [trk.to_track_state() for trk in self.tracks.values()]

    def _emit_egress(self, track: _FruitTrack, egress_time: float) -> FruitEvent:
        """Safely latch track.has_egressed = True and push event to FIFO queue."""
        track.has_egressed = True
        event = track.to_fruit_event(egress_time=egress_time)
        self.fifo_queue.put(event)
        logger.debug(
            f"Track {event.track_id} egressed: type={event.fruit_type}, "
            f"status={event.final_status}, defects={event.defect_classes}"
        )
        return event

    def flush_remaining(self, timestamp: Optional[float] = None) -> List[FruitEvent]:
        """Force-egress all remaining active tracks that have not yet egressed.

        Useful for pipeline drain or conveyor shutdown.
        """
        now = time.time() if timestamp is None else float(timestamp)
        emitted = []
        with self._lock:
            for trk in self.tracks.values():
                if not trk.has_egressed:
                    event = self._emit_egress(trk, now)
                    emitted.append(event)
        return emitted


# Aliases for backward compatibility and alternate naming preferences
Tracker = FruitTracker
ConveyorFruitTracker = FruitTracker
ConveyorTracker = FruitTracker

__all__ = [
    "FruitTracker",
    "Tracker",
    "ConveyorFruitTracker",
    "ConveyorTracker",
    "KalmanBoxFilter",
    "TrackState",
    "FruitDetection",
    "FruitEvent",
    "bbox_ious",
    "linear_assignment",
]

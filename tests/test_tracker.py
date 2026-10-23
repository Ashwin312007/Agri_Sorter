"""Unit tests for app.tracker: ByteTrack association, sticky defect state machine, and FIFO egress."""

import queue
import time
import pytest

from app.contracts import (
    FruitDetection,
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    TrackState,
)
from app.tracker import FruitTracker, KalmanBoxFilter, bbox_ious, linear_assignment


def make_detection(
    bbox=(100, 100, 150, 150),
    confidence=0.9,
    class_id=0,
    class_name="apple_fresh",
    fruit_type="apple",
    condition="fresh",
    is_defect=False,
) -> FruitDetection:
    """Helper to build FruitDetection test instances."""
    return FruitDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        fruit_type=fruit_type,
        condition=condition,
        is_defect=is_defect,
    )


class TestKalmanAndBbox:
    """Tests for Kalman filter and geometric bounding box utilities."""

    def test_bbox_ious(self):
        boxes1 = [(0, 0, 10, 10)]
        boxes2 = [(0, 0, 10, 10), (10, 10, 20, 20)]
        ious = bbox_ious(boxes1, boxes2)
        assert ious.shape == (1, 2)
        assert pytest.approx(ious[0, 0], 0.01) == 1.0
        assert pytest.approx(ious[0, 1], 0.01) == 0.0

    def test_kalman_filter_predict_update(self):
        kf = KalmanBoxFilter((100, 100, 150, 150))
        pred_box = kf.predict()
        assert len(pred_box) == 4
        updated_box = kf.update((105, 105, 155, 155))
        assert len(updated_box) == 4


class TestTrackingAssociation:
    """Tests for persistent track_id assignment across conveyor frames."""

    def test_persistent_track_id_across_frames(self):
        tracker = FruitTracker(tracker_type="bytetrack", egress_line_y_ratio=0.95)
        frame_shape = (640, 480)

        # Frame 1: Fruit at y=100
        det1 = make_detection(bbox=(100, 100, 150, 150))
        states1 = tracker.update([det1], frame_shape, timestamp=1.0)
        assert len(states1) == 1
        tid1 = states1[0].track_id

        # Frame 2: Fruit moved to y=115
        det2 = make_detection(bbox=(100, 115, 150, 165))
        states2 = tracker.update([det2], frame_shape, timestamp=1.1)
        assert len(states2) == 1
        assert states2[0].track_id == tid1  # Track ID persists!

        # Frame 3: Fruit moved to y=130
        det3 = make_detection(bbox=(100, 130, 150, 180))
        states3 = tracker.update([det3], frame_shape, timestamp=1.2)
        assert len(states3) == 1
        assert states3[0].track_id == tid1

    def test_multiple_fruits_maintain_distinct_ids(self):
        tracker = FruitTracker(egress_line_y_ratio=0.95)
        frame_shape = (640, 480)

        fruit1 = make_detection(bbox=(50, 100, 90, 140), fruit_type="apple")
        fruit2 = make_detection(bbox=(250, 100, 290, 140), fruit_type="orange")

        states = tracker.update([fruit1, fruit2], frame_shape, timestamp=1.0)
        assert len(states) == 2
        tids = {s.track_id for s in states}
        assert len(tids) == 2


class TestStickyDecisionStateMachine:
    """Tests for requirement 2: Sticky decision state machine and defect tracking."""

    def test_fresh_start_and_consecutive_defect_trigger(self):
        # Configured for N=3 consecutive defect frames
        tracker = FruitTracker(consecutive_defect_frames=3, egress_line_y_ratio=0.95)
        frame_shape = (640, 480)

        # Frame 1: Fresh detection -> status FRESH
        states = tracker.update(
            [make_detection(bbox=(100, 100, 140, 140), is_defect=False, condition="fresh")],
            frame_shape,
            timestamp=1.0,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 0
        assert len(states[0].defect_classes) == 0

        # Frame 2: 1st defect detection ("bruised") -> status remains FRESH (1 < 3)
        states = tracker.update(
            [make_detection(bbox=(100, 110, 140, 150), is_defect=True, condition="bruised")],
            frame_shape,
            timestamp=1.1,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 1
        assert "bruised" in states[0].defect_classes

        # Frame 3: 2nd defect detection ("bruised") -> status remains FRESH (2 < 3)
        states = tracker.update(
            [make_detection(bbox=(100, 120, 140, 160), is_defect=True, condition="bruised")],
            frame_shape,
            timestamp=1.2,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 2

        # Frame 4: 3rd consecutive defect detection ("bug_bite") -> status flips to REJECT!
        states = tracker.update(
            [make_detection(bbox=(100, 130, 140, 170), is_defect=True, condition="bug_bite")],
            frame_shape,
            timestamp=1.3,
        )
        assert states[0].status == STATUS_REJECT
        assert states[0].consecutive_defect_count == 3
        # Both defect classes recorded
        assert "bruised" in states[0].defect_classes
        assert "bug_bite" in states[0].defect_classes

    def test_intermittent_defect_resets_counter_before_threshold(self):
        # N=3
        tracker = FruitTracker(consecutive_defect_frames=3, egress_line_y_ratio=0.95)
        frame_shape = (640, 480)

        # Frame 1: Defect 1
        states = tracker.update(
            [make_detection(bbox=(100, 100, 140, 140), is_defect=True, condition="bruised")],
            frame_shape,
            timestamp=1.0,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 1

        # Frame 2: Defect 2
        states = tracker.update(
            [make_detection(bbox=(100, 110, 140, 150), is_defect=True, condition="bruised")],
            frame_shape,
            timestamp=1.1,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 2

        # Frame 3: Fresh -> consecutive count resets to 0, status remains FRESH
        states = tracker.update(
            [make_detection(bbox=(100, 120, 140, 160), is_defect=False, condition="fresh")],
            frame_shape,
            timestamp=1.2,
        )
        assert states[0].status == STATUS_FRESH
        assert states[0].consecutive_defect_count == 0
        # However, defect_classes accumulated the observed defect
        assert "bruised" in states[0].defect_classes

    def test_sticky_rule_once_reject_never_returns_to_fresh(self):
        # Once REJECT, it CAN NEVER go back to FRESH
        tracker = FruitTracker(consecutive_defect_frames=2, egress_line_y_ratio=0.95)
        frame_shape = (640, 480)

        # 2 consecutive defect frames -> flips to REJECT
        tracker.update(
            [make_detection(bbox=(100, 100, 140, 140), is_defect=True, condition="rotten")],
            frame_shape,
            timestamp=1.0,
        )
        states = tracker.update(
            [make_detection(bbox=(100, 110, 140, 150), is_defect=True, condition="rotten")],
            frame_shape,
            timestamp=1.1,
        )
        assert states[0].status == STATUS_REJECT

        # Followed by 5 consecutive FRESH frames
        for step in range(5):
            states = tracker.update(
                [make_detection(bbox=(100, 120 + step * 10, 140, 160 + step * 10), is_defect=False, condition="fresh")],
                frame_shape,
                timestamp=1.2 + step * 0.1,
            )
            # Must permanently remain REJECT
            assert states[0].status == STATUS_REJECT
            assert states[0].is_reject is True
            assert states[0].is_fresh is False


class TestCameraEgressAndFifoQueue:
    """Tests for requirement 3: Camera egress threshold and arrival-order FIFO queue."""

    def test_egress_crossing_pushes_single_event_to_fifo(self):
        # Frame height = 640. Egress ratio = 0.85 -> egress_y = 544
        tracker = FruitTracker(egress_line_y_ratio=0.85, consecutive_defect_frames=2)
        fifo_q = tracker.get_fifo_queue()
        assert fifo_q.empty()

        frame_shape = (640, 480)

        # Frame 1: y=480 (centroid y=500, well before egress line 544)
        tracker.update(
            [make_detection(bbox=(100, 480, 140, 520), fruit_type="apple", condition="fresh")],
            frame_shape,
            timestamp=1.0,
        )
        assert fifo_q.empty()

        # Frame 2: y=530 (centroid y=550 >= 544, crosses egress line!)
        states2 = tracker.update(
            [make_detection(bbox=(100, 530, 140, 570), fruit_type="apple", condition="fresh")],
            frame_shape,
            timestamp=2.0,
        )
        assert states2[0].has_egressed is True
        assert fifo_q.qsize() == 1

        event: FruitEvent = fifo_q.get_nowait()
        assert event.track_id == states2[0].track_id
        assert event.fruit_type == "apple"
        assert event.final_status == STATUS_FRESH
        assert event.defect_classes == []
        assert event.egress_timestamp == 2.0

        # Frame 3: Fruit still visible at y=580 (still past egress line)
        states3 = tracker.update(
            [make_detection(bbox=(100, 580, 140, 620), fruit_type="apple", condition="fresh")],
            frame_shape,
            timestamp=2.1,
        )
        assert states3[0].has_egressed is True
        # Must NOT push duplicate event!
        assert fifo_q.empty()

    def test_arrival_order_fifo_with_multiple_fruits(self):
        # Verify strict arrival-order in FIFO queue
        tracker = FruitTracker(egress_line_y_ratio=0.80)
        fifo_q = tracker.get_fifo_queue()
        frame_shape = (600, 600)  # egress line at y = 480

        # Fruit 1 ahead (y=400), Fruit 2 behind (y=100)
        f1_step1 = make_detection(bbox=(50, 400, 90, 440), fruit_type="apple")
        f2_step1 = make_detection(bbox=(200, 100, 240, 140), fruit_type="mango")
        tracker.update([f1_step1, f2_step1], frame_shape, timestamp=1.0)
        assert fifo_q.empty()

        # Step 2: Fruit 1 crosses egress line (y=500), Fruit 2 is at y=250
        f1_step2 = make_detection(bbox=(50, 500, 90, 540), fruit_type="apple")
        f2_step2 = make_detection(bbox=(200, 250, 240, 290), fruit_type="mango")
        tracker.update([f1_step2, f2_step2], frame_shape, timestamp=2.0)
        assert fifo_q.qsize() == 1

        # Step 3: Fruit 2 crosses egress line (y=500)
        f2_step3 = make_detection(bbox=(200, 500, 240, 540), fruit_type="mango")
        tracker.update([f2_step3], frame_shape, timestamp=3.0)
        assert fifo_q.qsize() == 2

        # Verify FIFO arrival order: Apple arrived first, Mango second
        event1 = fifo_q.get_nowait()
        event2 = fifo_q.get_nowait()
        assert event1.fruit_type == "apple"
        assert event2.fruit_type == "mango"
        assert event1.egress_timestamp == 2.0
        assert event2.egress_timestamp == 3.0

    def test_leave_camera_view_triggers_egress_if_not_already_egressed(self):
        # If a track leaves camera view without explicitly crossing egress line,
        # it is emitted when purged from active tracks.
        tracker = FruitTracker(track_buffer=2, egress_line_y_ratio=0.90)
        fifo_q = tracker.get_fifo_queue()
        frame_shape = (600, 600)

        # Tracked at y=300 for 1 frame
        det = make_detection(bbox=(100, 300, 150, 350), fruit_type="orange", is_defect=True, condition="scarred")
        tracker.update([det], frame_shape, timestamp=1.0)
        assert fifo_q.empty()

        # Next 3 frames: no detections (fruit disappears / leaves camera view)
        tracker.update([], frame_shape, timestamp=1.1)
        tracker.update([], frame_shape, timestamp=1.2)
        tracker.update([], frame_shape, timestamp=1.3)

        # Track exceeded track_buffer=2 and was purged -> event pushed
        assert fifo_q.qsize() == 1
        event = fifo_q.get_nowait()
        assert event.fruit_type == "orange"
        assert "scarred" in event.defect_classes


class TestTrackerCleanMethodsAndConfig:
    """Tests for public methods, config integration, and edge cases."""

    def test_flush_remaining(self):
        tracker = FruitTracker(egress_line_y_ratio=0.99)
        frame_shape = (640, 480)
        det = make_detection(bbox=(100, 100, 140, 140))
        tracker.update([det], frame_shape)
        fifo_q = tracker.get_fifo_queue()
        assert fifo_q.empty()

        # Drain pipeline
        flushed = tracker.flush_remaining()
        assert len(flushed) == 1
        assert fifo_q.qsize() == 1

        # Second flush does nothing because already egressed
        flushed2 = tracker.flush_remaining()
        assert len(flushed2) == 0

    def test_reset(self):
        tracker = FruitTracker()
        frame_shape = (640, 480)
        tracker.update([make_detection(bbox=(100, 600, 140, 640))], frame_shape)
        assert tracker.get_fifo_queue().qsize() == 1

        tracker.reset()
        assert tracker.get_fifo_queue().empty()
        assert len(tracker.get_active_tracks()) == 0

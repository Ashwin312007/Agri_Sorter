"""End-to-End Integration Tests for Agri_Sorter.

Validates the full integrated pipeline:
1. FruitDetector -> FruitTracker -> Sticky Decision -> FIFO Queue -> HardwareCoordinatorThread -> ServoActuator -> CSV Telemetry.
2. Sticky defect state machine resilience (consecutive defect threshold N=3, permanent REJECT).
3. FIFO queue arrival-order discipline.
4. Fail-safe queue underrun rejection & warning logging.
5. Ultrasonic debounce and refractory lockout.
6. Complete 20 compound classes end-to-end traversal.
7. Subprocess CLI execution of `app/main_sorter.py --simulate`.
"""

import csv
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import List

import numpy as np
import pytest

from app.contracts import (
    COMPOUND_CLASSES,
    FRUITS,
    CONDITIONS,
    REPO_ROOT,
    FruitDetection,
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    is_defect_class,
    is_fresh_class,
    parse_compound_label,
)
from app.detector import FruitDetector
from app.hardware_sorter import (
    HardwareCoordinatorThread,
    MockHCSR04Backend,
    MockServoBackend,
    ServoActuator,
    UltrasonicSensor,
    create_hardware_coordinator_from_config,
)
from app.tracker import FruitTracker


def _create_detection(
    bbox=(200, 100, 280, 180),
    confidence=0.95,
    class_name="apple_fresh",
) -> FruitDetection:
    fruit, cond = parse_compound_label(class_name)
    defect = is_defect_class(class_name)
    class_id = COMPOUND_CLASSES.index(class_name) if class_name in COMPOUND_CLASSES else 0
    return FruitDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        fruit_type=fruit,
        condition=cond,
        is_defect=defect,
    )


class TestSorterIntegration:
    """End-to-end integration test suite."""

    def test_end_to_end_pipeline_fresh_defect_and_underrun(self, tmp_path: Path):
        """Full pipeline test:

        - Fruit 1: Apple fresh -> exits camera -> FRESH event in FIFO -> Ultrasonic triggers -> PASS (0°).
        - Fruit 2: Orange rotten -> 3 consecutive defects -> exits camera -> REJECT event in FIFO -> Ultrasonic triggers -> REJECT (90°).
        - Fruit 3: Untracked fruit at sensor with empty FIFO -> Fail-safe REJECT (90°) + warning log.
        - Telemetry CSV verification of all events.
        """
        csv_path = tmp_path / "integration_events.csv"

        # 1. Setup Tracker
        tracker_config = {
            "tracking": {
                "tracker_type": "bytetrack",
                "consecutive_defect_frames": 3,
                "track_buffer": 10,
                "match_thresh": 0.5,
                "egress_line_y_ratio": 0.80,  # 80% of 500 = 400
            }
        }
        tracker = FruitTracker(config=tracker_config)
        fruit_fifo = tracker.get_fifo_queue()

        # 2. Setup Hardware Coordinator
        ultrasonic = UltrasonicSensor(
            trigger_pin=23,
            echo_pin=24,
            trigger_distance_cm=10.0,
            debounce_samples=1,
            refractory_lockout_s=0.05,
            simulate=True,
        )
        servo = ServoActuator(
            pin=18,
            rest_angle=0.0,
            reject_angle=90.0,
            pulse_hold_time_s=0.1,  # Short hold for fast testing
            simulate=True,
        )
        coordinator = HardwareCoordinatorThread(
            fruit_queue=fruit_fifo,
            ultrasonic=ultrasonic,
            servo=servo,
            event_log_path=csv_path,
            poll_interval_s=0.01,
            reject_on_empty_queue=True,
        )
        coordinator.start()

        try:
            # -------------------------------------------------------------
            # Stage A: Ingest Fruit 1 (Fresh Apple)
            # Starts at y=100, moves down to y=420 (crosses egress at y=400)
            # -------------------------------------------------------------
            now = 1000.0
            for y in [100, 200, 300, 420]:
                now += 0.05
                det = _create_detection(bbox=(200, y, 280, y + 80), class_name="apple_fresh")
                tracker.update([det], frame_shape=(500, 500), timestamp=now)

            # Fruit 1 should have egressed to FIFO
            assert fruit_fifo.qsize() == 1
            event1 = fruit_fifo.queue[0]
            assert event1.fruit_type == "apple"
            assert event1.final_status == STATUS_FRESH
            assert event1.defect_classes == []

            # Simulate ultrasonic trigger for Fruit 1 (< 10 cm)
            ultrasonic.set_simulated_distance(5.0)
            coordinator.handle_trigger()
            ultrasonic.set_simulated_distance(50.0)

            # Flap should be PASS (rest angle 0°)
            assert servo.angle == pytest.approx(0.0)
            assert coordinator.fresh_passed_count == 1
            assert coordinator.defect_rejected_count == 0
            assert fruit_fifo.empty()

            # -------------------------------------------------------------
            # Stage B: Ingest Fruit 2 (Rotten Orange)
            # Starts at y=100, moves down to y=420 with rotten defect for 3+ frames
            # -------------------------------------------------------------
            for y in [100, 200, 300, 420]:
                now += 0.05
                det = _create_detection(bbox=(200, y, 280, y + 80), class_name="orange_rotten")
                tracker.update([det], frame_shape=(500, 500), timestamp=now)

            # Fruit 2 should have egressed to FIFO
            assert fruit_fifo.qsize() == 1
            event2 = fruit_fifo.queue[0]
            assert event2.fruit_type == "orange"
            assert event2.final_status == STATUS_REJECT
            assert "rotten" in event2.defect_classes

            # Simulate ultrasonic trigger for Fruit 2 (< 10 cm)
            ultrasonic.set_simulated_distance(6.0)
            coordinator.handle_trigger()
            ultrasonic.set_simulated_distance(50.0)

            # Flap should be actuated to REJECT (90°)
            assert servo.angle == pytest.approx(90.0)
            assert coordinator.defect_rejected_count == 1
            assert fruit_fifo.empty()

            # Wait for servo auto-reset to rest position
            time.sleep(0.15)
            assert servo.angle == pytest.approx(0.0)

            # -------------------------------------------------------------
            # Stage C: Ingest Fruit 3 (Untracked fruit / Queue Underrun)
            # Sensor triggers while FIFO queue is empty
            # -------------------------------------------------------------
            assert fruit_fifo.empty()
            ultrasonic.set_simulated_distance(4.0)
            coordinator.handle_trigger()
            ultrasonic.set_simulated_distance(50.0)

            # Flap must actuate fail-safe REJECT (90°)
            assert servo.angle == pytest.approx(90.0)
            assert coordinator.underrun_rejected_count == 1

            time.sleep(0.15)
            assert servo.angle == pytest.approx(0.0)

        finally:
            coordinator.stop(timeout=1.0)

        # -------------------------------------------------------------
        # Stage D: Verify CSV Telemetry Log
        # -------------------------------------------------------------
        assert csv_path.is_file()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        # Row 1: Apple Fresh PASS
        assert rows[0]["fruit_type"] == "apple"
        assert rows[0]["final_status"] == STATUS_FRESH
        assert rows[0]["action"] == "PASS"
        assert rows[0]["defect_classes"] == "none"

        # Row 2: Orange Rotten REJECT
        assert rows[1]["fruit_type"] == "orange"
        assert rows[1]["final_status"] == STATUS_REJECT
        assert rows[1]["action"] == "REJECT"
        assert "rotten" in rows[1]["defect_classes"]

        # Row 3: Queue Underrun fail-safe REJECT
        assert rows[2]["track_id"] == "-1"
        assert rows[2]["final_status"] == "UNDERRUN_REJECT"
        assert rows[2]["action"] == "REJECT"

    def test_sticky_defect_state_machine_strict_invariance(self):
        """Verify:

        1. Transient defect of <3 consecutive frames does NOT flip status to REJECT.
        2. Exactly 3 consecutive defect frames flips status to REJECT.
        3. Once status is REJECT, subsequent FRESH detections NEVER revert it to FRESH.
        4. Multiple defect types are accumulated in defect_classes set.
        """
        tracker = FruitTracker(consecutive_defect_frames=3, egress_line_y_ratio=0.99)
        t = 100.0

        # Sub-test 1: 2 defect frames then fresh -> remains FRESH
        t += 0.1
        tracker.update(
            [_create_detection(bbox=(50, 50, 100, 100), class_name="apple_bruised")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        t += 0.1
        tracker.update(
            [_create_detection(bbox=(50, 55, 100, 105), class_name="apple_bruised")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        t += 0.1
        active = tracker.update(
            [_create_detection(bbox=(50, 60, 100, 110), class_name="apple_fresh")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        assert len(active) == 1
        assert active[0].status == STATUS_FRESH
        assert active[0].consecutive_defect_count == 0

        # Sub-test 2: 3 consecutive defect frames (apple_scarred) -> flips to REJECT
        t += 0.1
        tracker.update(
            [_create_detection(bbox=(50, 65, 100, 115), class_name="apple_scarred")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        t += 0.1
        tracker.update(
            [_create_detection(bbox=(50, 70, 100, 120), class_name="apple_scarred")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        t += 0.1
        active = tracker.update(
            [_create_detection(bbox=(50, 75, 100, 125), class_name="apple_scarred")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        assert len(active) == 1
        assert active[0].status == STATUS_REJECT

        # Sub-test 3: Follow with 10 consecutive FRESH frames -> STAYS REJECT!
        for i in range(10):
            t += 0.1
            active = tracker.update(
                [_create_detection(bbox=(50, 80 + i * 5, 100, 130 + i * 5), class_name="apple_fresh")],
                frame_shape=(500, 500),
                timestamp=t,
            )
            assert active[0].status == STATUS_REJECT, f"Reverted to FRESH on frame {i}!"

        # Sub-test 4: Additional defect (bug_bite) is accumulated
        t += 0.1
        active = tracker.update(
            [_create_detection(bbox=(50, 140, 100, 190), class_name="apple_bug_bite")],
            frame_shape=(500, 500),
            timestamp=t,
        )
        assert "bruised" in active[0].defect_classes
        assert "scarred" in active[0].defect_classes
        assert "bug_bite" in active[0].defect_classes

    def test_conveyor_arrival_order_fifo_discipline(self):
        """Verify arrival-order FIFO queue discipline:

        Fruits egressing the camera view are queued and popped strictly in arrival order.
        """
        tracker = FruitTracker(consecutive_defect_frames=2, egress_line_y_ratio=0.7)
        fifo = tracker.get_fifo_queue()
        now = 100.0

        # Fruit 1 moves and egresses first
        for y in [200, 300, 450]:
            now += 0.05
            tracker.update(
                [_create_detection(bbox=(100, y, 160, y + 60), class_name="apple_fresh")],
                frame_shape=(500, 500),
                timestamp=now,
            )
        assert fifo.qsize() == 1

        # Fruit 2 moves and egresses second
        for y in [200, 300, 450]:
            now += 0.05
            tracker.update(
                [_create_detection(bbox=(200, y, 260, y + 60), class_name="banana_rotten")],
                frame_shape=(500, 500),
                timestamp=now,
            )
        assert fifo.qsize() == 2

        # Fruit 3 moves and egresses third
        for y in [200, 300, 450]:
            now += 0.05
            tracker.update(
                [_create_detection(bbox=(300, y, 360, y + 60), class_name="mango_bruised")],
                frame_shape=(500, 500),
                timestamp=now,
            )
        assert fifo.qsize() == 3

        # Pop events and verify FIFO arrival order
        e1 = fifo.get_nowait()
        e2 = fifo.get_nowait()
        e3 = fifo.get_nowait()

        assert e1.fruit_type == "apple" and e1.final_status == STATUS_FRESH
        assert e2.fruit_type == "banana" and e2.final_status == STATUS_REJECT
        assert e3.fruit_type == "mango" and e3.final_status == STATUS_REJECT
        assert fifo.empty()

    def test_all_20_compound_classes_consistency(self):
        """Verify all 20 compound classes are consistently handled across contracts, tracker, and classification."""
        assert len(COMPOUND_CLASSES) == 20
        fresh_count = 0
        defect_count = 0

        for compound in COMPOUND_CLASSES:
            fruit, cond = parse_compound_label(compound)
            assert fruit in FRUITS
            assert cond in CONDITIONS

            is_fresh = is_fresh_class(compound)
            is_defect = is_defect_class(compound)
            assert is_fresh != is_defect  # Mutually exclusive

            if is_fresh:
                fresh_count += 1
                assert cond == "fresh"
            else:
                defect_count += 1
                assert cond in ["rotten", "bruised", "scarred", "bug_bite"]

            # Tracker update check
            tracker = FruitTracker(consecutive_defect_frames=1, egress_line_y_ratio=0.5)
            det = _create_detection(bbox=(50, 50, 100, 100), class_name=compound)
            active = tracker.update([det], frame_shape=(400, 400), timestamp=1.0)
            assert len(active) == 1
            if is_fresh:
                assert active[0].status == STATUS_FRESH
            else:
                assert active[0].status == STATUS_REJECT
                assert cond in active[0].defect_classes

        assert fresh_count == 4
        assert defect_count == 16

    def test_ultrasonic_debounce_and_refractory_lockout(self):
        """Verify ultrasonic sensor median filtering, trigger threshold (< 10cm), and refractory lockout."""
        triggered = []

        def on_trig():
            triggered.append(time.time())

        sensor = UltrasonicSensor(
            trigger_distance_cm=10.0,
            debounce_samples=3,
            refractory_lockout_s=0.20,
            simulate=True,
            on_trigger=on_trig,
        )

        now = 100.0
        # Readings above threshold (> 10cm) -> no trigger
        sensor.set_simulated_distance(15.0)
        assert not sensor.check_trigger(current_time=now)
        assert len(triggered) == 0

        # Reading below threshold (< 10cm) -> triggers
        sensor.set_simulated_distance(7.0)
        now += 0.01
        assert sensor.check_trigger(current_time=now)
        assert len(triggered) == 1

        # Immediate reading still below threshold within refractory lockout (0.20s) -> blocked
        now += 0.05
        assert not sensor.check_trigger(current_time=now)
        assert len(triggered) == 1

        # Distance clears (object departs beam)
        now += 0.10
        sensor.set_simulated_distance(50.0)
        assert not sensor.check_trigger(current_time=now)
        assert len(triggered) == 1

        # Next object arrives after refractory lockout elapsed (> 0.20s since last trigger) -> fires again
        now += 0.15  # Total delta = 0.25s > 0.20s
        sensor.set_simulated_distance(6.5)
        assert sensor.check_trigger(current_time=now)
        assert len(triggered) == 2

    def test_cli_simulate_subprocess(self, tmp_path: Path):
        """Run `app/main_sorter.py --simulate` via CLI subprocess and verify clean execution and log output."""
        log_path = tmp_path / "cli_fruit_events.csv"
        custom_cfg = tmp_path / "test_config.yaml"

        # Create temporary config pointing to log_path
        with open(REPO_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
            cfg_text = f.read()
        cfg_text = cfg_text.replace("runtime/fruit_events.csv", str(log_path))
        custom_cfg.write_text(cfg_text, encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "app.main_sorter",
            "--simulate",
            "--config",
            str(custom_cfg),
            "--frames",
            "120",
            "--fps",
            "150",
            "--headless",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, f"CLI exited with {result.returncode}: {result.stderr}"
        assert "AGRI_SORTER PIPELINE EXECUTION SUMMARY" in output
        assert "frames_processed" in output
        assert log_path.is_file()

        with open(log_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2  # Header + at least one event

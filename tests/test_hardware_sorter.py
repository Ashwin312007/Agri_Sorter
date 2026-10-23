"""Unit tests for Hardware Sorter (Requirement 6 & 7):

HC-SR04 ultrasonic triggering, servo actuation thread, event logging, and simulate mode.
"""

import csv
import logging
from pathlib import Path
import queue
import time
from unittest.mock import MagicMock, patch
import pytest

from app.contracts import (
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
)
from app.hardware_sorter import (
    HardwareCoordinatorThread,
    MockHCSR04Backend,
    MockServoBackend,
    RealHCSR04Backend,
    RealServoBackend,
    ServoActuator,
    UltrasonicSensor,
    create_hardware_coordinator_from_config,
)
from app.main_sorter import run_sorter


# ============================================================================
# 1. UltrasonicSensor Tests
# ============================================================================

def test_ultrasonic_median_filter_and_timeout():
    """Verify median-of-3 distance filtering and timeout handling."""
    sensor = UltrasonicSensor(
        trigger_pin=23,
        echo_pin=24,
        trigger_distance_cm=10.0,
        echo_timeout_ms=30.0,
        debounce_samples=3,
        refractory_lockout_s=0.25,
        simulate=True,
    )

    # 1. Constant reading: median of [15.0, 15.0, 15.0] = 15.0
    sensor.set_simulated_distance(15.0)
    assert sensor.get_filtered_distance_cm() == pytest.approx(15.0)

    # 2. Timeout handling: backend returns None (simulating echo timeout > 30ms)
    sensor.set_simulated_distance(None)
    assert sensor.measure_raw_distance_cm() is None
    assert sensor.get_filtered_distance_cm() is None
    # Timed-out readings must not trigger
    assert sensor.check_trigger(current_time=100.0) is False

    # 3. Dynamic samples with outlier / median behavior
    sensor._backend.measure_once = MagicMock(side_effect=[8.0, 20.0, 9.0])
    # Median of [8.0, 9.0, 20.0] = 9.0
    assert sensor.get_filtered_distance_cm() == pytest.approx(9.0)

    sensor.close()


def test_ultrasonic_debounce_and_refractory_lockout():
    """Verify debounce logic and 250ms refractory lockout ensure one fruit = one trigger."""
    triggered_count = 0

    def on_trigger_cb():
        nonlocal triggered_count
        triggered_count += 1

    sensor = UltrasonicSensor(
        trigger_distance_cm=10.0,
        refractory_lockout_s=0.25,
        simulate=True,
        on_trigger=on_trigger_cb,
    )

    # Idle state: distance 50cm (above 10cm threshold)
    sensor.set_simulated_distance(50.0)
    assert sensor.check_trigger(current_time=1.0) is False
    assert triggered_count == 0

    # Fruit arrives: distance drops to 6cm <= 10.0cm -> FIRST TRIGGER
    sensor.set_simulated_distance(6.0)
    assert sensor.check_trigger(current_time=1.05) is True
    assert triggered_count == 1

    # Fruit continues passing under sensor for 100ms: distance still 6cm
    # Must NOT trigger again!
    assert sensor.check_trigger(current_time=1.15) is False
    assert sensor.check_trigger(current_time=1.25) is False
    assert triggered_count == 1

    # Even after 300ms (> 250ms refractory lockout), while fruit remains present:
    # Debounce must prevent re-triggering on same fruit
    assert sensor.check_trigger(current_time=1.40) is False
    assert triggered_count == 1

    # Fruit clears sensor: distance rises to 45cm
    sensor.set_simulated_distance(45.0)
    assert sensor.check_trigger(current_time=1.45) is False
    assert triggered_count == 1

    # Second fruit arrives at t=1.70s (> 0.25s after last trigger):
    sensor.set_simulated_distance(5.5)
    assert sensor.check_trigger(current_time=1.70) is True
    assert triggered_count == 2

    sensor.close()


def test_ultrasonic_refractory_lockout_window():
    """Verify that a second fruit cannot trigger before refractory_lockout_s expires."""
    sensor = UltrasonicSensor(
        trigger_distance_cm=10.0,
        refractory_lockout_s=0.25,
        simulate=True,
    )

    # Fruit 1 triggers at t=10.0
    sensor.set_simulated_distance(5.0)
    assert sensor.check_trigger(current_time=10.0) is True

    # Fruit 1 leaves quickly at t=10.05
    sensor.set_simulated_distance(40.0)
    assert sensor.check_trigger(current_time=10.05) is False

    # Glitch / bounce at t=10.15 (< 0.25s lockout from t=10.0)
    sensor.set_simulated_distance(5.0)
    assert sensor.check_trigger(current_time=10.15) is False

    # Fruit 2 valid arrival at t=10.30 (> 0.25s lockout)
    sensor.set_simulated_distance(5.0)
    assert sensor.check_trigger(current_time=10.30) is True

    sensor.close()


# ============================================================================
# 2. ServoActuator Tests
# ============================================================================

def test_servo_actuation_non_blocking_and_angles():
    """Verify Rest=0°, Reject=90°, non-blocking execution, and hold-time auto reset."""
    servo = ServoActuator(
        pin=18,
        rest_angle=0.0,
        reject_angle=90.0,
        pulse_hold_time_s=0.1,  # Fast hold time for unit testing
        simulate=True,
    )

    # 1. Initial position is rest angle 0°
    assert servo.angle == 0.0
    assert servo.is_rejecting is False
    assert servo.rejection_count == 0

    # 2. Actuate reject stroke (non-blocking)
    t0 = time.time()
    servo.reject()
    t_elapsed = time.time() - t0
    # Must return immediately (< 20ms) without blocking
    assert t_elapsed < 0.05
    assert servo.angle == 90.0
    assert servo.is_rejecting is True
    assert servo.rejection_count == 1

    # 3. Wait for pulse_hold_time_s (0.1s) -> auto reset to 0°
    time.sleep(0.15)
    assert servo.angle == 0.0
    assert servo.is_rejecting is False

    # 4. Explicit rest() cancels pending timer immediately
    servo.reject()
    assert servo.angle == 90.0
    servo.rest()
    assert servo.angle == 0.0
    assert servo.is_rejecting is False

    # 5. Overlapping rejects extend hold window safely
    servo.reject()
    time.sleep(0.06)
    servo.reject()  # Re-trigger before 0.1s expired
    assert servo.rejection_count == 4
    time.sleep(0.06)
    # Flap should still be rejecting because timer was extended
    assert servo.angle == 90.0
    assert servo.is_rejecting is True
    time.sleep(0.07)
    # Now expired
    assert servo.angle == 0.0

    servo.close()


# ============================================================================
# 3. HardwareCoordinatorThread Tests
# ============================================================================

def test_hardware_coordinator_fresh_fruit(tmp_path: Path):
    """Verify FRESH fruit leaves servo at 0° and logs PASS event."""
    fruit_queue: queue.Queue[FruitEvent] = queue.Queue()
    log_file = tmp_path / "test_events.csv"

    ultrasonic = UltrasonicSensor(simulate=True)
    servo = ServoActuator(rest_angle=0.0, reject_angle=90.0, pulse_hold_time_s=0.1, simulate=True)

    coordinator = HardwareCoordinatorThread(
        fruit_queue=fruit_queue,
        ultrasonic=ultrasonic,
        servo=servo,
        event_log_path=log_file,
    )

    # Put FRESH fruit in queue
    event = FruitEvent(
        track_id=101,
        fruit_type="apple",
        final_status=STATUS_FRESH,
        defect_classes=[],
        confidence=0.98,
        egress_timestamp=100.0,
    )
    fruit_queue.put(event)

    # Trigger arrival
    handled = coordinator.handle_trigger()
    assert handled == event
    assert servo.angle == 0.0
    assert servo.is_rejecting is False
    assert coordinator.fresh_passed_count == 1
    assert coordinator.defect_rejected_count == 0
    assert coordinator.underrun_rejected_count == 0

    # Verify CSV record
    assert log_file.is_file()
    with open(log_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "timestamp", "track_id", "fruit_type", "final_status",
        "action", "defect_classes", "confidence", "egress_timestamp"
    ]
    assert rows[1][1] == "101"
    assert rows[1][2] == "apple"
    assert rows[1][3] == STATUS_FRESH
    assert rows[1][4] == "PASS"
    assert rows[1][5] == "none"

    coordinator.stop()


def test_hardware_coordinator_reject_fruit(tmp_path: Path):
    """Verify REJECT fruit actuates servo flap to 90° and logs REJECT event."""
    fruit_queue: queue.Queue[FruitEvent] = queue.Queue()
    log_file = tmp_path / "test_events.csv"

    ultrasonic = UltrasonicSensor(simulate=True)
    servo = ServoActuator(rest_angle=0.0, reject_angle=90.0, pulse_hold_time_s=0.1, simulate=True)

    coordinator = HardwareCoordinatorThread(
        fruit_queue=fruit_queue,
        ultrasonic=ultrasonic,
        servo=servo,
        event_log_path=log_file,
    )

    # Put REJECT fruit in queue
    event = FruitEvent(
        track_id=102,
        fruit_type="orange",
        final_status=STATUS_REJECT,
        defect_classes=["rotten", "scarred"],
        confidence=0.94,
        egress_timestamp=105.0,
    )
    fruit_queue.put(event)

    # Trigger arrival
    handled = coordinator.handle_trigger()
    assert handled == event
    assert servo.angle == 90.0
    assert servo.is_rejecting is True
    assert coordinator.defect_rejected_count == 1
    assert coordinator.fresh_passed_count == 0

    # Verify CSV record
    with open(log_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1][1] == "102"
    assert rows[1][2] == "orange"
    assert rows[1][3] == STATUS_REJECT
    assert rows[1][4] == "REJECT"
    assert "rotten" in rows[1][5]

    coordinator.stop()


def test_hardware_coordinator_queue_underrun_failsafe(tmp_path: Path):
    """Verify untracked fruit (empty queue) logs required warning and triggers fail-safe reject."""
    import app.hardware_sorter as hw_mod

    fruit_queue: queue.Queue[FruitEvent] = queue.Queue()
    log_file = tmp_path / "test_events.csv"

    ultrasonic = UltrasonicSensor(simulate=True)
    servo = ServoActuator(rest_angle=0.0, reject_angle=90.0, pulse_hold_time_s=0.1, simulate=True)

    coordinator = HardwareCoordinatorThread(
        fruit_queue=fruit_queue,
        ultrasonic=ultrasonic,
        servo=servo,
        event_log_path=log_file,
        reject_on_empty_queue=True,
    )

    expected_msg = "QUEUE_UNDERRUN_REJECT: Untracked fruit detected at sorter flap. Fail-safe REJECT activated!"

    # Trigger when queue is EMPTY
    with patch.object(hw_mod.logger, "warning") as mock_warn:
        handled = coordinator.handle_trigger()
        mock_warn.assert_called_with(expected_msg)

    assert handled is None

    # Servo must actuate to 90° for fail-safe reject
    assert servo.angle == 90.0
    assert servo.is_rejecting is True
    assert coordinator.underrun_rejected_count == 1

    # Verify CSV record has fail-safe details
    with open(log_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1][1] == "-1"
    assert rows[1][3] == "UNDERRUN_REJECT"
    assert rows[1][4] == "REJECT"
    assert rows[1][5] == "untracked_fruit"

    coordinator.stop()


def test_create_hardware_coordinator_from_config(tmp_path: Path):
    """Verify factory builder initializes coordinator with correct config parameters."""
    cfg = {
        "hardware": {
            "ultrasonic": {
                "trigger_pin": 23,
                "echo_pin": 24,
                "trigger_distance_cm": 12.5,
                "echo_timeout_ms": 25.0,
                "debounce_samples": 4,
                "refractory_lockout_s": 0.3,
            },
            "servo": {
                "pin": 18,
                "rest_angle": 5.0,
                "reject_angle": 85.0,
                "pulse_hold_time_s": 0.4,
            },
            "fail_safe": {
                "reject_on_empty_queue": True,
            },
        },
        "logging": {
            "event_log_path": str(tmp_path / "factory_events.csv"),
        },
    }

    q: queue.Queue[FruitEvent] = queue.Queue()
    coord = create_hardware_coordinator_from_config(cfg=cfg, fruit_queue=q, simulate=True)

    assert coord.ultrasonic.trigger_distance_cm == 12.5
    assert coord.ultrasonic.debounce_samples == 4
    assert coord.ultrasonic.refractory_lockout_s == 0.3
    assert coord.servo.rest_angle == 5.0
    assert coord.servo.reject_angle == 85.0
    assert coord.servo.pulse_hold_time_s == 0.4
    assert coord.reject_on_empty_queue is True

    coord.stop()


# ============================================================================
# 4. Simulation Pipeline & CLI Integration Tests
# ============================================================================

def test_main_sorter_simulate_pipeline(tmp_path: Path):
    """Verify run_sorter in simulate mode processes frames, triggers, and writes log."""
    log_file = tmp_path / "sim_events.csv"

    with patch("app.main_sorter.load_app_config") as mock_cfg:
        mock_cfg.return_value = {
            "tracking": {
                "tracker_type": "bytetrack",
                "consecutive_defect_frames": 2,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "egress_line_y_ratio": 0.85,
            },
            "hardware": {
                "ultrasonic": {
                    "trigger_pin": 23,
                    "echo_pin": 24,
                    "trigger_distance_cm": 10.0,
                    "echo_timeout_ms": 30.0,
                    "debounce_samples": 3,
                    "refractory_lockout_s": 0.25,
                },
                "servo": {
                    "pin": 18,
                    "rest_angle": 0.0,
                    "reject_angle": 90.0,
                    "pulse_hold_time_s": 0.1,
                },
                "fail_safe": {
                    "reject_on_empty_queue": True,
                },
            },
            "logging": {
                "event_log_path": str(log_file),
            },
        }

        summary = run_sorter(
            simulate=True,
            max_frames=80,
            fps=200.0,
            test_underrun=True,
            headless=True,
        )

        assert summary["frames_processed"] == 80
        assert summary["total_triggers"] >= 1
        assert summary["underrun_rejected_count"] >= 1
        assert log_file.is_file()

        with open(log_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2

"""Unit tests for Arduino HC-SR04 USB serial bridge and integration."""

import json
import time
from unittest.mock import MagicMock, patch
import pytest

from app.hardware.arduino_bridge import (
    ArduinoBridge,
    UltrasonicReading,
    discover_serial_ports,
)
from app.conveyor_pipeline import ConveyorCoordinator, FruitTrack


def test_ultrasonic_reading_dataclass():
    """Verify UltrasonicReading fields, types, and dictionary conversion."""
    now = time.time()
    reading = UltrasonicReading(
        tof_us=4210.0,
        dist_mm=72.0,
        is_fruit=True,
        timestamp=now,
    )
    assert reading.tof_us == 4210.0
    assert reading.dist_mm == 72.0
    assert reading.is_fruit is True
    assert reading.timestamp == now

    d = reading.to_dict()
    assert d == {
        "tof_us": 4210.0,
        "dist_mm": 72.0,
        "is_fruit": True,
        "timestamp": now,
    }


def test_parse_line_valid():
    """Verify parse_line correctly parses valid Arduino JSON output."""
    bridge = ArduinoBridge(mock=True)

    # Standard reading with fruit detected
    line1 = '{"tof_us": 4210, "dist_mm": 72, "fruit": 1}\n'
    r1 = bridge.parse_line(line1, timestamp=100.0)
    assert r1 is not None
    assert r1.tof_us == 4210.0
    assert r1.dist_mm == 72.0
    assert r1.is_fruit is True
    assert r1.timestamp == 100.0

    # Baseline reading without fruit
    line2 = '{"tof_us": 13994.1, "dist_mm": 240.0, "fruit": 0}'
    r2 = bridge.parse_line(line2, timestamp=200.0)
    assert r2 is not None
    assert r2.tof_us == pytest.approx(13994.1)
    assert r2.dist_mm == 240.0
    assert r2.is_fruit is False
    assert r2.timestamp == 200.0


def test_parse_line_invalid_and_ignored():
    """Verify parse_line safely returns None on non-JSON or unrelated packets."""
    bridge = ArduinoBridge(mock=True)

    # Empty / whitespace
    assert bridge.parse_line("") is None
    assert bridge.parse_line("   \r\n") is None

    # Corrupt JSON / truncated text
    assert bridge.parse_line("{'tof_us': 4210, ") is None
    assert bridge.parse_line("NOT_A_JSON") is None
    assert bridge.parse_line("--- Booting HC-SR04 ---") is None

    # Control responses without sensor readings
    assert bridge.parse_line('{"res":"PONG"}') is None
    assert bridge.parse_line('{"res":"CALIB_OK","baseline_mm":240}') is None
    assert bridge.parse_line('{"err":"UNKNOWN_COMMAND"}') is None


def test_mock_reading_generator():
    """Verify mock mode generates realistic 40Hz continuous readings."""
    bridge = ArduinoBridge(mock=True)
    assert bridge.get_latest_reading() is None

    bridge.start()
    try:
        # Give worker a moment to generate at least one reading
        time.sleep(0.06)
        r = bridge.get_latest_reading()
        assert r is not None
        assert isinstance(r, UltrasonicReading)
        assert r.tof_us > 0.0
        assert r.dist_mm > 0.0
        assert isinstance(r.is_fruit, bool)
    finally:
        bridge.stop()

    assert bridge._running is False


def test_set_mock_reading_override():
    """Verify manual override injection for testing edge conditions."""
    bridge = ArduinoBridge(mock=True)
    custom = UltrasonicReading(tof_us=9999.0, dist_mm=123.0, is_fruit=True, timestamp=1234.0)
    bridge.set_mock_reading(custom)

    r = bridge.get_latest_reading()
    assert r is not None
    assert r.tof_us == 9999.0
    assert r.dist_mm == 123.0
    assert r.is_fruit is True


def test_context_manager():
    """Verify ArduinoBridge lifecycle with context manager protocol."""
    with ArduinoBridge(mock=True) as bridge:
        assert bridge._running is True
        time.sleep(0.04)
        assert bridge.get_latest_reading() is not None

    assert bridge._running is False


def test_send_command_ping_and_calibrate_in_mock():
    """Verify PING and CALIB commands in mock mode."""
    with ArduinoBridge(mock=True, baseline_distance_mm=240.0) as bridge:
        assert bridge.ping() is True
        resp = bridge.send_command("PING")
        assert resp == '{"res":"PONG"}'

        calib_baseline = bridge.calibrate_baseline()
        assert calib_baseline == 240
        assert bridge.baseline_distance_mm == 240.0


def test_auto_discover_ports():
    """Verify serial port discovery filtering and fallback."""
    # Test when serial port list contains ACM/USB devices
    fake_port = MagicMock()
    fake_port.device = "/dev/ttyACM0"
    fake_port.description = "Arduino Uno"
    fake_port.hwid = "USB VID:PID=2341:0043"

    with patch("serial.tools.list_ports.comports", return_value=[fake_port]):
        ports = discover_serial_ports()
        assert "/dev/ttyACM0" in ports


def test_fallback_to_mock_when_port_unavailable():
    """Verify that specifying a non-existent port falls back gracefully to mock mode."""
    bridge = ArduinoBridge(port="/dev/nonexistent_serial_device_xyz123", mock=False)
    bridge.start()
    try:
        assert bridge.is_mock is True
        assert bridge._running is True
        time.sleep(0.04)
        reading = bridge.get_latest_reading()
        assert reading is not None
    finally:
        bridge.stop()


def test_conveyor_pipeline_integration_with_arduino():
    """Verify ConveyorCoordinator Station 1 integrates with ArduinoBridge readings."""
    coord = ConveyorCoordinator(
        mock=True,
        speed_override=0.10,
        use_arduino=True,
    )
    assert coord.use_arduino is True
    assert coord.arduino_bridge is not None

    coord.start()
    try:
        # Pre-seed a known reading in the arduino bridge
        custom_reading = UltrasonicReading(
            tof_us=4250.0,
            dist_mm=75.0,
            is_fruit=True,
            timestamp=time.time(),
        )
        coord.arduino_bridge.set_mock_reading(custom_reading)

        track = FruitTrack(
            track_id=1,
            entry_time=10.0,
            bbox=(0, 0, 50, 50),
            visual_label="fresh",
            visual_confidence=0.99,
        )
        coord.add_track(track)

        # Arrival at Station 1 is at 10.0 + 0.30 / 0.10 = 13.0s
        t_arrive = coord.get_arrival_time(track)
        coord.step(timestamp=t_arrive)

        assert track.status == "MEASURED"
        assert track.ultrasonic_travel_time_us == pytest.approx(4250.0)
        assert track.ultrasonic_amplitude == 75
        assert track.final_grade == "FIRST_QUALITY"
    finally:
        coord.stop()

    assert coord.arduino_bridge._running is False

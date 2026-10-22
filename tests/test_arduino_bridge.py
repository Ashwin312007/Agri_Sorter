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
        fruit_diameter_mm=68.0,
    )
    assert reading.tof_us == 4210.0
    assert reading.dist_mm == 72.0
    assert reading.is_fruit is True
    assert reading.timestamp == now
    assert reading.fruit_diameter_mm == 68.0

    d = reading.to_dict()
    assert d == {
        "tof_us": 4210.0,
        "dist_mm": 72.0,
        "is_fruit": True,
        "timestamp": now,
        "fruit_diameter_mm": 68.0,
    }


def test_parse_line_valid():
    """Verify parse_line correctly parses valid Arduino Uno JSON output."""
    bridge = ArduinoBridge(mock=True, baseline_distance_mm=240.0)

    # Arduino Uno reading with explicit fruit_diameter_mm
    line_arduino_diam = '{"tof_us": 4210, "dist_mm": 72, "fruit_diameter_mm": 68, "fruit": 1}\n'
    r_diam = bridge.parse_line(line_arduino_diam, timestamp=100.0)
    assert r_diam is not None
    assert r_diam.tof_us == 4210.0
    assert r_diam.dist_mm == 72.0
    assert r_diam.is_fruit is True
    assert r_diam.fruit_diameter_mm == 68.0
    assert r_diam.timestamp == 100.0

    # Arduino reading without explicit fruit_diameter_mm (calculated dynamically from baseline)
    line_arduino_fallback = '{"tof_us": 4210, "dist_mm": 72, "fruit": 1}\n'
    r_fallback = bridge.parse_line(line_arduino_fallback, timestamp=150.0)
    assert r_fallback is not None
    assert r_fallback.tof_us == 4210.0
    assert r_fallback.dist_mm == 72.0
    assert r_fallback.is_fruit is True
    assert r_fallback.fruit_diameter_mm == 168.0  # 240.0 - 72.0
    assert r_fallback.timestamp == 150.0

    # Baseline reading without fruit
    line2 = '{"tof_us": 13994.1, "dist_mm": 240.0, "fruit_diameter_mm": 0, "fruit": 0}'
    r2 = bridge.parse_line(line2, timestamp=200.0)
    assert r2 is not None
    assert r2.tof_us == pytest.approx(13994.1)
    assert r2.dist_mm == 240.0
    assert r2.is_fruit is False
    assert r2.fruit_diameter_mm == 0.0
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
    """Verify mock mode generates realistic 40Hz continuous readings with fruit diameter."""
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
        assert r.fruit_diameter_mm is not None
        assert r.fruit_diameter_mm >= 0.0
    finally:
        bridge.stop()

    assert bridge._running is False


def test_set_mock_reading_override():
    """Verify manual override injection for testing edge conditions."""
    bridge = ArduinoBridge(mock=True)
    custom = UltrasonicReading(
        tof_us=9999.0,
        dist_mm=123.0,
        is_fruit=True,
        timestamp=1234.0,
        fruit_diameter_mm=71.5,
    )
    bridge.set_mock_reading(custom)

    r = bridge.get_latest_reading()
    assert r is not None
    assert r.tof_us == 9999.0
    assert r.dist_mm == 123.0
    assert r.is_fruit is True
    assert r.fruit_diameter_mm == 71.5


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


def test_esp32_c6_port_discovery():
    """Verify auto-discovery detects ESP32-C6 native USB CDC / JTAG serial device."""
    # ESP32-C6 with Espressif VID 0x303A and PID 0x1001
    esp_port = MagicMock()
    esp_port.device = "/dev/ttyACM1"
    esp_port.description = "USB JTAG/serial debug unit"
    esp_port.hwid = "USB VID:PID=303a:1001 SER=123456"
    esp_port.vid = 0x303A
    esp_port.pid = 0x1001

    with patch("serial.tools.list_ports.comports", return_value=[esp_port]):
        ports = discover_serial_ports()
        assert "/dev/ttyACM1" in ports

    # ESP32 CP210x or CH340 UART bridge
    esp_uart = MagicMock()
    esp_uart.device = "/dev/ttyUSB0"
    esp_uart.description = "CP2102 USB to UART Bridge Controller - ESP32"
    esp_uart.hwid = "USB VID:PID=10c4:ea60"
    esp_uart.vid = 0x10C4
    esp_uart.pid = 0xEA60

    with patch("serial.tools.list_ports.comports", return_value=[esp_uart]):
        ports = discover_serial_ports()
        assert "/dev/ttyUSB0" in ports


def test_arduino_uno_port_discovery():
    """Verify auto-discovery detects Arduino Uno with VID 0x2341 (Uno R3 0x0043 and Uno R1/R2 0x0001)."""
    # Arduino Uno R3: VID 0x2341, PID 0x0043
    uno_r3_port = MagicMock()
    uno_r3_port.device = "/dev/ttyACM0"
    uno_r3_port.description = "Arduino Uno (CDC ACM)"
    uno_r3_port.hwid = "USB VID:PID=2341:0043"
    uno_r3_port.vid = 0x2341
    uno_r3_port.pid = 0x0043

    with patch("serial.tools.list_ports.comports", return_value=[uno_r3_port]):
        ports = discover_serial_ports()
        assert "/dev/ttyACM0" in ports

    # Arduino Uno R1/R2: VID 0x2341, PID 0x0001
    uno_r1_port = MagicMock()
    uno_r1_port.device = "/dev/ttyACM1"
    uno_r1_port.description = "Arduino Uno"
    uno_r1_port.hwid = "USB VID:PID=2341:0001"
    uno_r1_port.vid = 0x2341
    uno_r1_port.pid = 0x0001

    with patch("serial.tools.list_ports.comports", return_value=[uno_r1_port]):
        ports = discover_serial_ports()
        assert "/dev/ttyACM1" in ports

    # Arduino clone with CH340 chip
    ch340_port = MagicMock()
    ch340_port.device = "/dev/ttyUSB0"
    ch340_port.description = "USB-SERIAL CH340"
    ch340_port.hwid = "USB VID:PID=1a86:7523"
    ch340_port.vid = 0x1A86
    ch340_port.pid = 0x7523

    with patch("serial.tools.list_ports.comports", return_value=[ch340_port]):
        ports = discover_serial_ports()
        assert "/dev/ttyUSB0" in ports


def test_bridge_config_arduino_section():
    """Verify ArduinoBridge loads settings from primary [arduino] config section."""
    cfg = {
        "arduino": {
            "port": "/dev/ttyACM2",
            "baud_rate": 115200,
            "baseline_distance_mm": 240.0,
            "fruit_threshold_mm": 30.0,
            "timeout_s": 1.5,
        }
    }
    bridge = ArduinoBridge(mock=True, config=cfg)
    assert bridge.configured_port == "/dev/ttyACM2"
    assert bridge.baud_rate == 115200
    assert bridge.baseline_distance_mm == 240.0
    assert bridge.fruit_threshold_mm == 30.0
    assert bridge.timeout_s == 1.5


def test_bridge_config_esp32_section():
    """Verify ArduinoBridge loads settings from [esp32] config section."""
    cfg = {
        "esp32": {
            "port": "/dev/ttyACM5",
            "baud_rate": 115200,
            "baseline_distance_mm": 260.0,
            "fruit_threshold_mm": 35.0,
            "timeout_s": 2.0,
        }
    }
    bridge = ArduinoBridge(mock=True, config=cfg)
    assert bridge.configured_port == "/dev/ttyACM5"
    assert bridge.baud_rate == 115200
    assert bridge.baseline_distance_mm == 260.0
    assert bridge.fruit_threshold_mm == 35.0
    assert bridge.timeout_s == 2.0


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
            fruit_diameter_mm=68.5,
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
        assert track.fruit_diameter_mm == pytest.approx(68.5)
        assert track.final_grade == "FIRST_QUALITY"
    finally:
        coord.stop()

    assert coord.arduino_bridge._running is False

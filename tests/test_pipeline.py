"""Tests for ultrasonic pipeline orchestration, signal conditioning, and CLI capture."""
import csv
import sys
from pathlib import Path
import pytest

from app.hardware.pipeline import UltrasonicPipeline
from app.ultrasonic_capture import run_capture


def test_ultrasonic_pipeline_mock_mode():
    with UltrasonicPipeline(mock=True) as pipeline:
        res = pipeline.measure()
        assert isinstance(res, dict)
        assert "travel_time_us" in res
        assert "received_amplitude" in res
        assert res["travel_time_us"] == pytest.approx(4200.0)
        assert res["received_amplitude"] == 62

        # capture() tuple convenience
        tof, amp = pipeline.capture()
        assert tof == pytest.approx(4200.0)
        assert amp == 62

        # Conveyor configuration parsed with expected defaults
        assert "conveyor" in pipeline.config
        conveyor = pipeline.config["conveyor"]
        assert conveyor["belt_speed_mps"] == pytest.approx(0.12)
        assert conveyor["max_safe_speed_mps"] == pytest.approx(0.20)
        assert conveyor["camera_to_ultrasonic_distance_m"] == pytest.approx(0.30)
        assert conveyor["ultrasonic_to_sorter_distance_m"] == pytest.approx(0.25)
        assert conveyor["sensor_mounting_height_mm"] == 240
        assert conveyor["apex_sampling_window_mm"] == 35


def test_run_capture_csv_output(tmp_path: Path):
    csv_file = tmp_path / "test_output.csv"
    rows = run_capture(
        count=3,
        csv_out=str(csv_file),
        mock=True,
        delay=0.0,
    )
    assert len(rows) == 3
    assert csv_file.is_file()

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert "travel_time_us" in reader[0]
        assert "received_amplitude" in reader[0]
        assert float(reader[0]["travel_time_us"]) == pytest.approx(4200.0)
        assert int(reader[0]["received_amplitude"]) == 62


def test_run_capture_with_auxiliary_features(tmp_path: Path):
    csv_file = tmp_path / "aux_output.csv"
    rows = run_capture(
        count=2,
        csv_out=str(csv_file),
        mock=True,
        delay=0.0,
        diameter=75.0,
        weight=182.5,
        temperature=24.8,
    )
    assert len(rows) == 2

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert float(reader[0]["fruit_diameter_mm"]) == pytest.approx(75.0)
        assert float(reader[0]["weight_g"]) == pytest.approx(182.5)
        assert float(reader[0]["temperature_c"]) == pytest.approx(24.8)


def test_ultrasonic_signal_conditioning_and_capture(tmp_path: Path):
    """Verify acoustic signal conditioning and capture without XGBoost dependencies."""
    csv_file = tmp_path / "conditioned_output.csv"
    rows = run_capture(
        count=2,
        csv_out=str(csv_file),
        mock=True,
        delay=0.0,
        diameter=74.0,
        weight=180.0,
        temperature=25.0,
    )
    assert len(rows) == 2
    assert csv_file.is_file()

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        for r in reader:
            assert float(r["travel_time_us"]) == pytest.approx(4200.0)
            assert int(r["received_amplitude"]) == 62
            assert float(r["fruit_diameter_mm"]) == pytest.approx(74.0)
            assert float(r["weight_g"]) == pytest.approx(180.0)
            assert float(r["temperature_c"]) == pytest.approx(25.0)


def test_run_capture_with_arduino_bridge(tmp_path: Path):
    """Verify run_capture integration with Arduino Uno microcontroller bridge in mock mode."""
    csv_file = tmp_path / "arduino_capture.csv"
    rows = run_capture(
        count=2,
        csv_out=str(csv_file),
        mock=True,
        delay=0.0,
        use_arduino=True,
    )
    assert len(rows) == 2
    assert "travel_time_us" in rows[0]
    assert "received_amplitude" in rows[0]
    assert "fruit_diameter_mm" in rows[0]
    assert rows[0]["fruit_diameter_mm"] is not None

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert "fruit_diameter_mm" in reader[0]


def test_run_capture_with_esp32_bridge(tmp_path: Path):
    """Verify run_capture integration with ESP32 microcontroller bridge in mock mode."""
    csv_file = tmp_path / "esp32_capture.csv"
    rows = run_capture(
        count=2,
        csv_out=str(csv_file),
        mock=True,
        delay=0.0,
        use_esp32=True,
    )
    assert len(rows) == 2
    assert "travel_time_us" in rows[0]
    assert "received_amplitude" in rows[0]
    assert "fruit_diameter_mm" in rows[0]
    assert rows[0]["fruit_diameter_mm"] is not None

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert "fruit_diameter_mm" in reader[0]


def test_ultrasonic_capture_cli_flags(tmp_path: Path, monkeypatch):
    """Verify CLI flags --use-arduino, --use-mcu, and --use-esp32 execute cleanly."""
    from app.ultrasonic_capture import main as capture_main

    csv_file = tmp_path / "cli_arduino_capture.csv"
    test_args = [
        "ultrasonic_capture.py",
        "--mock",
        "--count",
        "1",
        "--use-arduino",
        "--csv-out",
        str(csv_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    capture_main()

    assert csv_file.is_file()
    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert "fruit_diameter_mm" in reader[0]


def test_ultrasonic_capture_cli_flags_esp32_alias(tmp_path: Path, monkeypatch):
    """Verify CLI flag --use-esp32 alias executes cleanly."""
    from app.ultrasonic_capture import main as capture_main

    csv_file = tmp_path / "cli_mcu_capture.csv"
    test_args = [
        "ultrasonic_capture.py",
        "--mock",
        "--count",
        "1",
        "--use-esp32",
        "--csv-out",
        str(csv_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    capture_main()

    assert csv_file.is_file()
    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert "fruit_diameter_mm" in reader[0]


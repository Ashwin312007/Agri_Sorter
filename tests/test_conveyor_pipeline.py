"""Unit tests for ConveyorCoordinator and spatial tracking pipeline."""
import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.conveyor_pipeline import (
    BAD_LABELS,
    BAD_LABELS_REJECT,
    BAD_LABELS_SECOND,
    ConveyorCoordinator,
    FruitTrack,
    main,
)


def test_fruit_track_dataclass():
    track = FruitTrack(
        track_id=10,
        entry_time=100.0,
        bbox=(10, 20, 30, 40),
        visual_label="fresh",
        visual_confidence=0.95,
    )
    assert track.track_id == 10
    assert track.entry_time == 100.0
    assert track.bbox == (10, 20, 30, 40)
    assert track.visual_label == "fresh"
    assert track.visual_confidence == 0.95
    assert track.ultrasonic_travel_time_us is None
    assert track.ultrasonic_amplitude is None
    assert track.internal_label_pred is None
    assert track.final_grade is None
    assert track.status == "IN_TRANSIT"
    assert track.internal_fallback is False

    d = track.to_dict()
    assert d["track_id"] == 10
    assert d["status"] == "IN_TRANSIT"
    assert d["bbox"] == "10:20:30:40"


def test_conveyor_speed_safety_checks():
    # Normal speed within limit (0.12 <= 0.20)
    coord = ConveyorCoordinator(mock=True, speed_override=0.15)
    assert coord.belt_speed_mps == 0.15

    # Excessive speed with warning (default non-strict)
    with pytest.warns(UserWarning, match="exceeds maximum safe speed"):
        coord_warn = ConveyorCoordinator(mock=True, speed_override=0.25, strict_safety=False)
        assert coord_warn.belt_speed_mps == 0.25

    # Excessive speed with strict mode -> ValueError
    with pytest.raises(ValueError, match="exceeds maximum safe speed"):
        ConveyorCoordinator(mock=True, speed_override=0.25, strict_safety=True)

    # Invalid non-positive speed -> ValueError
    with pytest.raises(ValueError, match="must be positive"):
        ConveyorCoordinator(mock=True, speed_override=0.0)


def test_timing_and_station_arrival():
    # camera_to_ultrasonic = 0.30 m, ultrasonic_to_sorter = 0.25 m
    coord = ConveyorCoordinator(mock=True, speed_override=0.10)
    track = FruitTrack(
        track_id=1,
        entry_time=10.0,
        bbox=(0, 0, 50, 50),
        visual_label="fresh",
        visual_confidence=0.99,
    )
    coord.add_track(track)

    t_arrive = coord.get_arrival_time(track)
    # 10.0 + 0.30 / 0.10 = 13.0
    assert t_arrive == pytest.approx(13.0)

    t_eject = coord.get_ejection_time(track)
    # 10.0 + (0.30 + 0.25) / 0.10 = 10.0 + 5.5 = 15.5
    assert t_eject == pytest.approx(15.5)

    # Step before arrival (t = 12.0)
    coord.step(timestamp=12.0)
    assert track.status == "IN_TRANSIT"
    assert track.ultrasonic_travel_time_us is None

    # Step at arrival (t = 13.0)
    coord.step(timestamp=13.0)
    assert track.status == "MEASURED"
    assert track.ultrasonic_travel_time_us == pytest.approx(4200.0)
    assert track.ultrasonic_amplitude == 62
    assert track.final_grade == "FIRST_QUALITY"
    assert track.internal_fallback is True

    # Step at ejection (t = 15.5)
    coord.step(timestamp=15.5)
    assert track.status == "EJECTED"


def test_grading_logic_defects():
    coord = ConveyorCoordinator(mock=True)

    # Rotten defect -> REJECT
    t_rot = coord.register_fruit(
        bbox=(0, 0, 10, 10),
        visual_label="rotten",
        visual_confidence=0.95,
        entry_time=0.0,
    )
    coord._evaluate_grade(t_rot)
    assert t_rot.final_grade == "REJECT"
    assert t_rot.internal_fallback is False

    # Bruise defect -> SECOND_QUALITY
    t_bruise = coord.register_fruit(
        bbox=(0, 0, 10, 10),
        visual_label="bruise",
        visual_confidence=0.90,
        entry_time=0.0,
    )
    coord._evaluate_grade(t_bruise)
    assert t_bruise.final_grade == "SECOND_QUALITY"
    assert t_bruise.internal_fallback is False


def test_grading_with_internal_model(tmp_path: Path, monkeypatch):
    stub_healthy = MagicMock()
    stub_healthy.predict.return_value = [0]
    mock_joblib = MagicMock()
    mock_joblib.load.return_value = {
        "model": stub_healthy,
        "features": ["travel_time_us", "received_amplitude"],
    }
    monkeypatch.setitem(sys.modules, "joblib", mock_joblib)

    dummy_model_file = tmp_path / "dummy_model.joblib"
    dummy_model_file.touch()

    coord = ConveyorCoordinator(mock=True, internal_model_path=str(dummy_model_file))

    # Healthy internal prediction
    t_fresh = FruitTrack(
        track_id=5,
        entry_time=0.0,
        bbox=(0, 0, 10, 10),
        visual_label="fresh",
        visual_confidence=0.99,
        ultrasonic_travel_time_us=4200.0,
        ultrasonic_amplitude=60,
    )
    coord._evaluate_grade(t_fresh)
    assert t_fresh.final_grade == "FIRST_QUALITY"
    assert t_fresh.internal_label_pred == 0
    assert t_fresh.internal_fallback is False

    # Defect internal prediction
    stub_damaged = MagicMock()
    stub_damaged.predict.return_value = [1]
    coord.internal_model = stub_damaged
    t_damaged = FruitTrack(
        track_id=6,
        entry_time=0.0,
        bbox=(0, 0, 10, 10),
        visual_label="fresh",
        visual_confidence=0.99,
        ultrasonic_travel_time_us=4200.0,
        ultrasonic_amplitude=30,
    )
    coord._evaluate_grade(t_damaged)
    assert t_damaged.final_grade == "SECOND_QUALITY"
    assert t_damaged.internal_label_pred == 1
    assert t_damaged.internal_fallback is False



def test_csv_export(tmp_path: Path):
    coord = ConveyorCoordinator(mock=True)
    coord.register_fruit(
        bbox=(10, 20, 30, 40),
        visual_label="fresh",
        visual_confidence=0.95,
        entry_time=1.0,
    )
    csv_file = tmp_path / "tracks.csv"
    coord.save_csv(csv_file)
    assert csv_file.is_file()

    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["track_id"] == "1"
        assert reader[0]["visual_label"] == "fresh"
        assert reader[0]["status"] == "IN_TRANSIT"


def test_cli_execution_with_csv(tmp_path: Path, monkeypatch):
    csv_file = tmp_path / "cli_tracks.csv"
    test_args = [
        "conveyor_pipeline.py",
        "--mock",
        "--duration",
        "0.2",
        "--csv-out",
        str(csv_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert csv_file.is_file()
    with open(csv_file, "r", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 4

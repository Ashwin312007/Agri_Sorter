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


def test_conveyor_coordinator_vision_init_options():
    coord = ConveyorCoordinator(
        mock=True,
        headless=True,
        feed_mock_tracks=True,
        detector_model_path="models/detector/yolo11n.pt",
        classifier_model_path="models/external/best.pt",
    )
    assert coord.mock is True
    assert coord.headless is True
    assert coord.feed_mock_tracks is True
    assert coord.detector is None
    assert coord.classifier is None
    assert coord.apple_class_id is None
    assert coord.items == {}


def test_conveyor_step_vision_tracking_integration(monkeypatch):
    coord = ConveyorCoordinator(mock=True, headless=True, feed_mock_tracks=False)
    coord.mock = False  # simulate live mode for step()

    # Mock detector and its tracking result
    mock_box = [10.0, 20.0, 110.0, 120.0]
    mock_track_id = 42

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu().tolist.return_value = [mock_box]
    mock_boxes.id.int().cpu().tolist.return_value = [mock_track_id]

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_detector = MagicMock()
    mock_detector.track.return_value = [mock_result]

    # Mock classifier
    mock_classifier = MagicMock()
    mock_cls_result = MagicMock()
    mock_cls_result.names = {0: "fresh"}
    mock_cls_result.probs.top1 = 0
    mock_cls_result.probs.top1conf = 0.94
    mock_classifier.predict.return_value = [mock_cls_result]

    coord.detector = mock_detector
    coord.classifier = mock_classifier
    coord.apple_class_id = 47

    # Create dummy frame
    import numpy as np
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # First step: should register new track
    t0 = 1000.0
    tracks = coord.step(frame=dummy_frame, timestamp=t0)

    mock_detector.track.assert_called_once()
    assert mock_track_id in coord.items
    track = coord.items[mock_track_id]
    assert track.track_id == 42
    assert track.entry_time == 1000.0
    assert track.bbox == (10, 20, 110, 120)
    assert track.visual_label == "fresh"
    assert track.visual_confidence == pytest.approx(0.94)
    assert track in coord.spatial_queue

    # Second step with same track: classifier should not be called again for this track
    call_count = mock_classifier.predict.call_count
    coord.step(frame=dummy_frame, timestamp=t0 + 0.1)
    assert mock_classifier.predict.call_count == call_count


def test_conveyor_step_visual_display(monkeypatch):
    import numpy as np
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_imshow = MagicMock()
    mock_waitkey = MagicMock()
    monkeypatch.setattr("cv2.imshow", mock_imshow)
    monkeypatch.setattr("cv2.waitKey", mock_waitkey)

    # When headless is True: imshow should NOT be called
    coord_headless = ConveyorCoordinator(mock=True, headless=True)
    coord_headless.step(frame=dummy_frame, timestamp=1.0)
    mock_imshow.assert_not_called()
    mock_waitkey.assert_not_called()

    # When headless is False: imshow and waitKey SHOULD be called
    coord_gui = ConveyorCoordinator(mock=True, headless=False)
    coord_gui.step(frame=dummy_frame, timestamp=1.0)
    mock_imshow.assert_called_once_with("Conveyor Vision", dummy_frame)
    mock_waitkey.assert_called_once_with(1)


def test_cli_flags_headless_and_feed_mock(tmp_path: Path, monkeypatch):
    csv_file = tmp_path / "cli_headless.csv"
    test_args = [
        "conveyor_pipeline.py",
        "--mock",
        "--duration",
        "0.1",
        "--headless",
        "--feed-mock-tracks",
        "--detector-model",
        "models/detector/yolo11n.pt",
        "--classifier-model",
        "models/external/best.pt",
        "--csv-out",
        str(csv_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert csv_file.is_file()


def test_conveyor_coordinator_detect_cadence():
    import numpy as np

    # Test initialization clamping
    coord = ConveyorCoordinator(mock=True, detect_cadence=3)
    assert coord.detect_cadence == 3
    assert coord._frame_count == 0

    coord_clamped = ConveyorCoordinator(mock=True, detect_cadence=0)
    assert coord_clamped.detect_cadence == 1

    # Test cadence alternating behavior during step()
    coord_live = ConveyorCoordinator(mock=True, headless=True, detect_cadence=2)
    coord_live.mock = False  # Enable vision step path

    mock_detector = MagicMock()
    mock_detector.track.return_value = [MagicMock(boxes=None)]
    coord_live.detector = mock_detector

    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)

    # Frame 1: count=1, 1 % 2 != 0 -> skip_detect=True
    coord_live.step(frame=dummy_frame, timestamp=1.0)
    assert mock_detector.track.call_args[1]["skip_detect"] is True

    # Frame 2: count=2, 2 % 2 == 0 -> skip_detect=False
    coord_live.step(frame=dummy_frame, timestamp=1.05)
    assert mock_detector.track.call_args[1]["skip_detect"] is False

    # Frame 3: count=3, 3 % 2 != 0 -> skip_detect=True
    coord_live.step(frame=dummy_frame, timestamp=1.10)
    assert mock_detector.track.call_args[1]["skip_detect"] is True


def test_cli_flags_detect_cadence_and_target_fps(tmp_path: Path, monkeypatch):
    csv_file = tmp_path / "cli_fps.csv"
    test_args = [
        "conveyor_pipeline.py",
        "--mock",
        "--duration",
        "0.1",
        "--headless",
        "--detect-cadence",
        "3",
        "--target-fps",
        "60.0",
        "--csv-out",
        str(csv_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert csv_file.is_file()



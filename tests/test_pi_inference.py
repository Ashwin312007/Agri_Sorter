"""Unit tests for lightweight RPi4 vision inference, tracking, and acoustic classification."""
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.pi_inference import (
    BoxesResult,
    ClassificationResult,
    ConveyorTracker,
    OpenCVInferenceEngine,
    TrackDecisions,
    _ArrayWrapper,
    crop_box,
    find_class_id,
    is_defect_detector,
    load_vision_engine,
)
from app.ultrasonic_capture import (
    RuleBasedAcousticClassifier,
    load_prediction_model,
    predict_row,
)


def test_array_wrapper_cpu_and_int():
    wrap = _ArrayWrapper([1.0, 2.7, 3.2])
    assert wrap.cpu() is wrap
    assert wrap.tolist() == [1.0, 2.7, 3.2]
    assert wrap.int().tolist() == [1, 2, 3]
    assert len(wrap) == 3
    assert wrap[1] == 2.7


def test_boxes_result_interface():
    boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
    ids = [1, 2]
    confs = [0.95, 0.88]
    cids = [47, 47]

    res = BoxesResult(xyxy=boxes, ids=ids, confidences=confs, class_ids=cids)
    assert len(res) == 2
    assert res.xyxy.cpu().tolist() == boxes
    assert res.id.int().cpu().tolist() == [1, 2]
    assert res.conf.tolist() == confs
    assert res.cls.int().tolist() == [47, 47]


def test_conveyor_tracker_spatial_association():
    tracker = ConveyorTracker(max_distance=50.0, max_missed=5, velocity_px_s=(10.0, 0.0))

    # Frame 0: detection at (100, 100) -> [80, 80, 120, 120]
    box0 = [80, 80, 120, 120]
    results0 = tracker.update(boxes=[box0], timestamp=0.0)
    assert len(results0) == 1
    assert results0[0][1] == 1  # track ID 1
    assert results0[0][0] == box0

    # Frame 1: 0.1s later, belt moves +1px in x. Detection at [81, 80, 121, 120]
    box1 = [81, 80, 121, 120]
    results1 = tracker.update(boxes=[box1], timestamp=0.1)
    assert len(results1) == 1
    assert results1[0][1] == 1  # Should retain track ID 1!

    # Frame 2: new fruit enters at [20, 20, 60, 60] alongside fruit 1
    box2 = [20, 20, 60, 60]
    results2 = tracker.update(boxes=[box1, box2], timestamp=0.2)
    assert len(results2) == 2
    track_ids = {r[1] for r in results2}
    assert 1 in track_ids
    assert 2 in track_ids


def test_conveyor_tracker_track_expiration():
    tracker = ConveyorTracker(max_distance=40.0, max_missed=2)
    tracker.update(boxes=[[10, 10, 30, 30]], timestamp=0.0)
    assert 1 in tracker.tracks

    # Miss frame 1
    tracker.update(boxes=[], timestamp=0.1)
    assert 1 in tracker.tracks
    assert tracker.tracks[1].missed_frames == 1

    # Miss frame 2
    tracker.update(boxes=[], timestamp=0.2)
    assert 1 in tracker.tracks
    assert tracker.tracks[1].missed_frames == 2

    # Miss frame 3 -> exceed max_missed (2)
    tracker.update(boxes=[], timestamp=0.3)
    assert 1 not in tracker.tracks


def test_find_class_id_and_crop_box():
    mock_model = MagicMock()
    mock_model.names = {0: "person", 47: "apple", 49: "orange"}
    assert find_class_id(mock_model, "apple") == 47
    assert find_class_id(mock_model, "APPLE") == 47

    # Default fallback for apple when not found
    mock_empty = MagicMock()
    mock_empty.names = {0: "dummy"}
    assert find_class_id(mock_empty, "apple") == 47

    with pytest.raises(ValueError, match="Detector does not contain the class"):
        find_class_id(mock_empty, "banana")

    # Crop box
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_box(frame, [10, 20, 30, 40], padding=0)
    assert crop.shape == (20, 20, 3)


def test_track_decisions_logic():
    decisions = TrackDecisions(minimum_frames=3)

    # Observation 1
    lbl, conf, ready = decisions.observe(track_id=1, label="fresh", confidence=0.90)
    assert lbl == "fresh"
    assert ready is False

    # Observation 2
    lbl, conf, ready = decisions.observe(track_id=1, label="fresh", confidence=0.92)
    assert ready is False

    # Observation 3 -> ready
    lbl, conf, ready = decisions.observe(track_id=1, label="fresh", confidence=0.94)
    assert ready is True
    assert conf == pytest.approx(0.92)

    # Mark emitted
    decisions.mark_emitted(1)
    lbl, conf, ready = decisions.observe(track_id=1, label="fresh", confidence=0.95)
    assert ready is False  # Already emitted


def test_rule_based_acoustic_classifier():
    clf = RuleBasedAcousticClassifier(amplitude_threshold=45.0, min_travel_time_us=3000.0, max_travel_time_us=6000.0)

    # Healthy apple: amp=62, ToF=4200 -> 0
    pred_healthy = clf.predict([[4200.0, 62.0]])
    assert pred_healthy == [0]

    # Attenuated defect: amp=30, ToF=4200 -> 1
    pred_defect_amp = clf.predict([[4200.0, 30.0]])
    assert pred_defect_amp == [1]

    # Cavity defect: amp=60, ToF=1500 (too fast) -> 1
    pred_defect_tof = clf.predict([[1500.0, 60.0]])
    assert pred_defect_tof == [1]

    # Serialization test
    d = clf.to_dict()
    clf2 = RuleBasedAcousticClassifier.from_dict(d)
    assert clf2.amplitude_threshold == clf.amplitude_threshold
    assert clf2.min_travel_time_us == clf.min_travel_time_us


def test_load_prediction_model_json(tmp_path: Path):
    json_path = tmp_path / "acoustic_rule.json"
    rule_data = {
        "type": "RuleBasedAcousticClassifier",
        "amplitude_threshold": 48.0,
        "min_travel_time_us": 2800.0,
        "max_travel_time_us": 6200.0,
        "features": ["travel_time_us", "received_amplitude"],
    }
    with open(json_path, "w") as f:
        json.dump(rule_data, f)

    model, features = load_prediction_model(str(json_path))
    assert isinstance(model, RuleBasedAcousticClassifier)
    assert model.amplitude_threshold == 48.0
    assert features == ["travel_time_us", "received_amplitude"]

    # Predict row using JSON model
    pred = predict_row(model, features, travel_time_us=4200.0, received_amplitude=60)
    assert pred == 0

    pred_def = predict_row(model, features, travel_time_us=4200.0, received_amplitude=25)
    assert pred_def == 1


def test_load_vision_engine_informative_error_on_pt_without_ultralytics(tmp_path: Path):
    pt_file = tmp_path / "dummy_weights.pt"
    pt_file.touch()

    # When .pt file exists but no .onnx and no ultralytics
    with pytest.raises(FileNotFoundError, match="export this model to ONNX"):
        load_vision_engine(pt_file)


def test_conveyor_tracker_propagate():
    tracker = ConveyorTracker(velocity_px_s=(100.0, 50.0))
    # Initial detection at t = 1.0s
    initial_box = [10, 20, 50, 60]
    tracker.update(boxes=[initial_box], timestamp=1.0)
    assert 1 in tracker.tracks
    trk = tracker.tracks[1]
    assert trk.box == initial_box
    assert trk.cx == 30.0
    assert trk.cy == 40.0

    # Propagate at t = 1.1s (dt = 0.1s)
    # dx = 100 * 0.1 = 10, dy = 50 * 0.1 = 5
    propagated = tracker.propagate(timestamp=1.1)
    assert len(propagated) == 1
    box, tid = propagated[0]
    assert tid == 1
    assert box == [20, 25, 60, 65]
    assert trk.cx == pytest.approx(40.0)
    assert trk.cy == pytest.approx(45.0)
    assert trk.last_seen == 1.1


def test_opencv_inference_engine_track_skip_detect():
    engine = object.__new__(OpenCVInferenceEngine)
    engine.input_size = (416, 416)
    engine.task = "detect"
    engine.tracker = ConveyorTracker(velocity_px_s=(50.0, 0.0))

    # Pre-seed tracker with an apple track
    engine.tracker.update(
        boxes=[[10, 20, 30, 40]],
        confidences=[0.92],
        class_ids=[47],
        timestamp=1.0,
    )

    engine.detect = MagicMock()
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Calling track with skip_detect=True should NOT call detect()
    results = engine.track(dummy_frame, skip_detect=True, timestamp=1.1)

    engine.detect.assert_not_called()
    assert len(results) == 1
    boxes_res = results[0].boxes
    assert len(boxes_res) == 1
    assert boxes_res.id.int().cpu().tolist() == [1]
    # dt = 0.1s, dx = 50 * 0.1 = 5px -> [15, 20, 35, 40]
    assert boxes_res.xyxy.cpu().tolist() == [[15, 20, 35, 40]]
    assert boxes_res.conf.tolist() == [pytest.approx(0.92)]
    assert boxes_res.cls.int().tolist() == [47]


def test_is_defect_detector():
    # 1. COCO detector (no defect classes)
    coco_mock = MagicMock()
    coco_mock.names = {0: "person", 47: "apple", 49: "orange"}
    assert is_defect_detector(coco_mock) is False

    # 2. Defect detector with fresh_apple, rot, bruise
    defect_mock = MagicMock()
    defect_mock.names = {0: "fresh_apple", 1: "rot", 2: "bruise"}
    assert is_defect_detector(defect_mock) is True

    # 3. Defect detector with fresh/rotten
    cls_mock = MagicMock()
    cls_mock.names = {0: "fresh", 1: "rotten"}
    assert is_defect_detector(cls_mock) is True

    # 4. Empty or invalid model
    empty_mock = MagicMock()
    empty_mock.names = {}
    assert is_defect_detector(empty_mock) is False


def test_find_class_id_with_defect_classes():
    # Model with fresh_apple (has substring apple)
    mock_defect = MagicMock()
    mock_defect.names = {0: "fresh_apple", 1: "rot", 2: "bruise"}
    assert find_class_id(mock_defect, "apple") == 0
    assert find_class_id(mock_defect, "rot") == 1
    assert find_class_id(mock_defect, "bruise") == 2

    # Model with fresh / rotten (no apple in name)
    mock_generic = MagicMock()
    mock_generic.names = {0: "fresh", 1: "rotten"}
    assert find_class_id(mock_generic, "apple") == 0
    assert find_class_id(mock_generic, "rotten") == 1


def test_dataset_sample_creation_and_validation(tmp_path):
    from scripts.download_yolo_dataset import create_sample_dataset, validate_dataset

    ds_dir = tmp_path / "sample_dataset"
    classes = {0: "fresh_apple", 1: "rot", 2: "bruise"}
    yaml_path = create_sample_dataset(ds_dir, classes=classes)

    assert yaml_path.is_file()
    valid, stats = validate_dataset(ds_dir)
    assert valid is True
    assert stats["total_images"] > 0
    assert stats["total_boxes"] > 0
    assert len(stats["errors"]) == 0



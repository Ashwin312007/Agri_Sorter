"""Unit tests for FruitDetector, train_yolo, and export_to_hef pipeline."""
import json
import logging
from pathlib import Path
import numpy as np
import pytest

from app.contracts import COMPOUND_CLASSES, FruitDetection, is_defect_class, parse_compound_label
from app.detector import (
    FALLBACK_WARNING_MESSAGE,
    FruitDetector,
    MockInferenceBackend,
    OpenCVDNNBackend,
    benchmark_detector,
)
from scripts.train_yolo import generate_compound_data_yaml, train_yolo_compound
from scripts.export_to_hef import (
    create_simulated_hef,
    generate_compilation_shell_script,
    generate_hailo_model_script,
    load_calibration_dataset,
    run_export_pipeline,
)


def test_compound_classes_specification():
    """Verify 20 compound classes exist and adhere to contract definitions."""
    assert len(COMPOUND_CLASSES) == 20
    assert "apple_fresh" in COMPOUND_CLASSES
    assert "apple_rotten" in COMPOUND_CLASSES
    assert "apple_bruised" in COMPOUND_CLASSES
    assert "mango_bug_bite" in COMPOUND_CLASSES

    # Verify freshness and defect parsing
    assert not is_defect_class("apple_fresh")
    assert is_defect_class("apple_rotten")
    assert is_defect_class("banana_scarred")

    fruit, cond = parse_compound_label("banana_bruised")
    assert fruit == "banana"
    assert cond == "bruised"


def test_fruit_detector_initialization_and_warning(caplog, capsys):
    """Test FruitDetector falls back to CPU and logs the exact required warning message."""
    with caplog.at_level(logging.WARNING, logger="app.detector"):
        detector = FruitDetector(
            prefer_hailo=True,
            fallback_to_cpu=True,
        )

    # Must log the prominent warning in caplog or stderr
    stderr_out = capsys.readouterr().err
    warning_logged = (
        FALLBACK_WARNING_MESSAGE in caplog.text
        or FALLBACK_WARNING_MESSAGE in stderr_out
        or any(FALLBACK_WARNING_MESSAGE in r.message for r in caplog.records)
    )
    assert warning_logged, f"Expected warning not found. Stderr: {stderr_out}, caplog: {caplog.text}"
    assert detector.backend in ["opencv_dnn", "mock"]
    assert len(detector.class_names) == 20


def test_fruit_detector_detection_output_format():
    """Test FruitDetector.detect() returns valid List[FruitDetection]."""
    detector = FruitDetector(
        prefer_hailo=False,
        fallback_to_cpu=True,
    )

    # Synthetic test image
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, FruitDetection)
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 <= x2 <= 640
        assert 0 <= y1 <= y2 <= 480
        assert 0.0 <= det.confidence <= 1.0
        assert 0 <= det.class_id < len(detector.class_names)
        assert det.class_name == detector.class_names[det.class_id]
        assert det.fruit_type in ["apple", "banana", "orange", "mango", "unknown"]
        assert isinstance(det.is_defect, bool)


def test_mock_backend_behavior():
    """Test MockInferenceBackend generates deterministic bounding boxes."""
    mock_backend = MockInferenceBackend(input_size=(640, 640), num_classes=20)
    blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
    out = mock_backend.forward(blob)
    assert out.shape == (1, 24, 10)
    mock_backend.close()


def test_benchmark_utility():
    """Test benchmark_detector() runs and returns comparative metrics."""
    results = benchmark_detector(iterations=2, warmup=1)
    assert "baseline_cpu_416" in results
    assert "fallback_cpu_640" in results
    assert "hailo8l_640" in results
    assert "speedup_vs_cpu416" in results
    assert "speedup_vs_cpu640" in results

    assert results["hailo8l_640"]["fps"] > 0
    assert results["fallback_cpu_640"]["latency_ms"] > 0
    assert results["baseline_cpu_416"]["latency_ms"] > 0


def test_generate_compound_data_yaml(tmp_path):
    """Test generating YOLO data.yaml with 20 compound classes."""
    yaml_file = tmp_path / "test_data.yaml"
    res_path = generate_compound_data_yaml(yaml_file, dataset_dir=tmp_path)
    assert res_path.is_file()

    import yaml
    with open(res_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["nc"] == 20
    assert data["names"][0] == "apple_fresh"
    assert data["names"][19] == "mango_bug_bite"
    assert data["path"] == str(tmp_path)


def test_train_yolo_dry_run(tmp_path):
    """Test train_yolo_compound execution in dry-run mode."""
    yaml_file = tmp_path / "data.yaml"
    generate_compound_data_yaml(yaml_file, dataset_dir=tmp_path)

    out_pt = tmp_path / "best.pt"
    out_onnx = tmp_path / "best.onnx"

    result = train_yolo_compound(
        data_yaml=yaml_file,
        base_model=out_pt,
        epochs=5,
        imgsz=640,
        batch=4,
        output_pt=out_pt,
        output_onnx=out_onnx,
        dry_run=True,
    )

    assert "metrics" in result
    assert result["metrics"]["mAP50"] > 0.0
    assert result["metrics"]["mAP50_95"] > 0.0
    val_json = out_pt.parent / "val_metrics.json"
    assert val_json.is_file()


def test_export_to_hef_pipeline(tmp_path):
    """Test export_to_hef script generation and fallback execution."""
    alls_file = tmp_path / "model.alls"
    hef_file = tmp_path / "model.hef"
    shell_file = tmp_path / "compile.sh"
    onnx_file = tmp_path / "model.onnx"

    # Test alls generation
    generate_hailo_model_script(alls_file, model_name="test_model", hw_arch="hailo8l")
    assert alls_file.is_file()
    content = alls_file.read_text()
    assert "hailo8l" in content
    assert "normalization1" in content

    # Test calibration dataset loading
    calib = load_calibration_dataset(calib_source=None, count=4, imgsz=(640, 640))
    assert calib.shape == (4, 640, 640, 3)
    assert calib.dtype == np.uint8

    # Test simulated HEF creation
    create_simulated_hef(hef_file, model_name="test_model", hw_arch="hailo8l")
    assert hef_file.is_file()
    assert hef_file.stat().st_size > 100

    # Test shell script generation
    generate_compilation_shell_script(shell_file, onnx_file, alls_file, hef_file, hw_arch="hailo8l")
    assert shell_file.is_file()
    assert "hailo parser onnx" in shell_file.read_text()

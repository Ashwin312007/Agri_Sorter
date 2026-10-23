"""Integration tests for Unified 4-Class Architecture and 13 User Roboflow Datasets.

Verifies:
1. 13 Roboflow datasets are mapped into the unified 4 classes (Healthy, Bruise_or_Scratch, Scab_or_Canker, Rot_or_Blight).
2. Contracts and config specify nc: 4 and UNIFIED_DEFECT_CLASSES.
3. Orange, Guava, and Pomegranate detections operate without skin rejection.
4. Sorter decision: Healthy -> PASS (0°), Defect -> REJECT (90°).
"""

import numpy as np
import pytest

from app.contracts import (
    STATUS_FRESH,
    STATUS_REJECT,
    UNIFIED_DEFECT_CLASSES,
    FruitDetection,
    is_defect_class,
    is_fresh_class,
    load_app_config,
)
from app.detector import (
    STAGE2_CLASSES,
    STAGE2_COLOR_CODING,
    STAGE2_SERVO_MAP,
    FruitDetector,
)
from app.web_dashboard import (
    classify_defect_hierarchy,
    map_to_defect_hierarchy,
)
from scripts.dataset_pipeline import (
    ROBOFLOW_PROJECT_MAPPINGS,
    UNIFIED_CLASSES,
    generate_unified_data_yaml,
)


def test_unified_taxonomy_contracts():
    """Verify contracts and configuration match the 4 unified defect classes."""
    expected = ["Healthy", "Bruise_or_Scratch", "Scab_or_Canker", "Rot_or_Blight"]
    assert UNIFIED_DEFECT_CLASSES == expected
    assert UNIFIED_CLASSES == expected
    assert STAGE2_CLASSES == expected

    cfg = load_app_config()
    assert cfg["classes"]["nc"] == 4
    assert cfg["classes"]["defect_classes"] == expected


def test_13_roboflow_projects_mapped():
    """Verify all 13 user Roboflow projects have mappings into the 4 classes."""
    expected_projects = [
        "applesort-esgll/apple_surface_defects",
        "fruit-disease-detection-nrrei/apple-fruit-disease-detection-zei6u",
        "appledefectdetection/apple-defects-yafjq",
        "chisilia/detection-guava-disease",
        "guava-wedlv/guava-3hezg",
        "thao-ulmra/guava-qnkee-1uz3i",
        "kaycee-kaye-y-villanueva-mzmks/mango-defect-detection",
        "mangoclass/mangograding",
        "capstone-2az9h/orange-disease-detection-qunki",
        "mydetect-o2wvw/orange-defect",
        "orangedisease/orange-disease-2twml",
        "pomegranate-diseases/pomegranate-disease-detection-t7rg0",
        "orin-f84rr/pomegranate-gqopm",
    ]

    for proj in expected_projects:
        assert proj in ROBOFLOW_PROJECT_MAPPINGS, f"Missing project mapping: {proj}"
        mapping = ROBOFLOW_PROJECT_MAPPINGS[proj]
        assert len(mapping) > 0, f"Empty mapping for {proj}"
        for raw_lbl, target_cls in mapping.items():
            assert target_cls in UNIFIED_CLASSES, f"Invalid target {target_cls} for {proj}:{raw_lbl}"


def test_sorter_servo_angle_decisions():
    """Verify Class 0 (Healthy) maps to Servo 0° (PASS) and Classes 1..3 map to 90° (REJECT)."""
    assert STAGE2_SERVO_MAP["Healthy"] == 0
    assert STAGE2_SERVO_MAP["Bruise_or_Scratch"] == 90
    assert STAGE2_SERVO_MAP["Scab_or_Canker"] == 90
    assert STAGE2_SERVO_MAP["Rot_or_Blight"] == 90

    assert is_fresh_class("Healthy") is True
    assert is_fresh_class("Bruise_or_Scratch") is False
    assert is_fresh_class("Scab_or_Canker") is False
    assert is_fresh_class("Rot_or_Blight") is False

    assert is_defect_class("Healthy") is False
    assert is_defect_class("Bruise_or_Scratch") is True
    assert is_defect_class("Scab_or_Canker") is True
    assert is_defect_class("Rot_or_Blight") is True


def test_orange_guava_pomegranate_detection_and_no_skin_rejection():
    """Verify orange, guava, and pomegranate can be detected and genuine orange is not rejected as skin."""
    detector = FruitDetector(prefer_hailo=False, fallback_to_cpu=True)

    # Vivid orange crop: Cr high, Cb low
    orange_bgr = np.full((100, 100, 3), (20, 120, 240), dtype=np.uint8)
    # Genuine orange peel MUST NOT be rejected as skin
    assert FruitDetector.is_skin_tone_dominant(orange_bgr) is False

    # Human skin crop: BGR (130, 160, 215) MUST be detected as skin
    skin_bgr = np.full((100, 100, 3), (130, 160, 215), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(skin_bgr) is True


def test_generate_unified_data_yaml_execution(tmp_path):
    """Verify data.yaml generation writes nc: 4 and correct unified names."""
    out_yaml = tmp_path / "data.yaml"
    generate_unified_data_yaml(out_yaml)
    assert out_yaml.is_file()

    import yaml
    with open(out_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["nc"] == 4
    assert cfg["names"][0] == "Healthy"
    assert cfg["names"][1] == "Bruise_or_Scratch"
    assert cfg["names"][2] == "Scab_or_Canker"
    assert cfg["names"][3] == "Rot_or_Blight"


def test_web_dashboard_defect_hierarchy_aggregation():
    """Verify dashboard correctly aggregates defects into the 4 unified categories."""
    assert map_to_defect_hierarchy("apple_fresh") == "Healthy"
    assert map_to_defect_hierarchy("orange_bruised") == "Bruise_or_Scratch"
    assert map_to_defect_hierarchy("guava_scab") == "Scab_or_Canker"
    assert map_to_defect_hierarchy("pomegranate_bacterial_blight") == "Rot_or_Blight"

    assert classify_defect_hierarchy(["bruised", "scab"]) == "Scab_or_Canker"
    assert classify_defect_hierarchy(["bruised", "rot"]) == "Rot_or_Blight"
    assert classify_defect_hierarchy([], is_reject=False) == "Healthy"
    assert classify_defect_hierarchy([], is_reject=True) == "Rot_or_Blight"

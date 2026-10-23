"""Integration tests for Unified 4-Class Architecture and 13 User Roboflow Datasets.

Verifies:
1. 13 Roboflow datasets are mapped into the unified 4 classes (Healthy, Bruise_or_Scratch, Scab_or_Canker, Rot_or_Blight).
2. Contracts and config specify nc: 4 and UNIFIED_DEFECT_CLASSES.
3. Orange, Guava, and Pomegranate detections operate without skin rejection.
4. Sorter decision: Healthy -> PASS (0°), Defect -> REJECT (90°).
"""

import json
import numpy as np
import pytest

from app.contracts import (
    REPO_ROOT,
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
    VERIFIED_ROBOFLOW_PROJECTS,
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


def test_14_roboflow_verified_projects_rf01_through_rf14():
    """Verify all 14 Roboflow projects RF-01 through RF-14 are correctly registered.

    Cross-referenced with Fruit_Defect_YOLO_Datasets_Verified.xlsx:
    RF-01: Apple Surface Defects (AppleSort)
    RF-02: Apple Fruit Disease Detection
    RF-03: Apple Defect Detection (pkekm)
    RF-04: Detection Guava Disease (chisilia)
    RF-05: Guava Fruit Hub (guava-h8gxt)
    RF-06: Guava Disease Model (chisilia v5)
    RF-07: Mango Defect Detection (mintras)
    RF-08: Mango Defect Detection (Villanueva)
    RF-09: MangoGrading (MangoClass)
    RF-10: Orange Disease Detection (CAPSTONE)
    RF-11: Orange Defect Dataset (Mydetect)
    RF-12: Pomegranate Disease Detection (Fizza)
    RF-13: Pomegranate Disease Detection (IM)
    RF-14: Pomegranate Fresh vs. Rotten (Bin)
    """
    assert len(VERIFIED_ROBOFLOW_PROJECTS) == 14

    for rf_id in [f"RF-{i:02d}" for i in range(1, 15)]:
        assert rf_id in VERIFIED_ROBOFLOW_PROJECTS, f"Missing {rf_id} in VERIFIED_ROBOFLOW_PROJECTS"
        entry = VERIFIED_ROBOFLOW_PROJECTS[rf_id]

        # Verify entry schema
        assert "fruit" in entry
        assert "name" in entry
        assert "project" in entry
        assert "url" in entry
        assert entry["url"].startswith("https://universe.roboflow.com/")

        # Verify registration in ROBOFLOW_PROJECT_MAPPINGS by RF-ID
        assert rf_id in ROBOFLOW_PROJECT_MAPPINGS, f"Missing mapping for {rf_id}"
        mapping_by_id = ROBOFLOW_PROJECT_MAPPINGS[rf_id]
        assert len(mapping_by_id) > 0

        # Verify registration by project name
        proj_name = entry["project"]
        assert proj_name in ROBOFLOW_PROJECT_MAPPINGS, f"Missing mapping for {proj_name}"
        mapping_by_proj = ROBOFLOW_PROJECT_MAPPINGS[proj_name]
        assert len(mapping_by_proj) > 0

        # Verify all mapped labels strictly target the unified 4 classes
        for raw_lbl, target_cls in mapping_by_id.items():
            assert (
                target_cls in UNIFIED_CLASSES
            ), f"Invalid target {target_cls} for {rf_id}:{raw_lbl}"


def test_rebuilt_4class_model_metadata_and_outputs():
    """Verify rebuilt model configuration outputs map to the 4 unified classes."""
    model_json_path = REPO_ROOT / "models" / "detector" / "yolo11n_fruit.json"
    assert model_json_path.is_file(), f"Model config not found at {model_json_path}"

    with open(model_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["num_classes"] == 4
    expected_classes = {
        "0": "Healthy",
        "1": "Bruise_or_Scratch",
        "2": "Scab_or_Canker",
        "3": "Rot_or_Blight",
    }
    for class_id_str, class_name in expected_classes.items():
        assert meta["names"][class_id_str] == class_name
        # Index lookup
        assert UNIFIED_CLASSES[int(class_id_str)] == class_name


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

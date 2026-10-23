"""Tests for app/contracts.py and config contracts."""

import pytest
from app.contracts import (
    COMPOUND_CLASSES,
    CONDITIONS,
    EXTENDED_COMPOUND_CLASSES,
    EXTENDED_FRUITS,
    FRUITS,
    LEGACY_FRUITS,
    STATUS_FRESH,
    STATUS_REJECT,
    UNIFIED_DEFECT_CLASSES,
    is_defect_class,
    is_fresh_class,
    load_app_config,
    parse_compound_label,
)


def test_unified_defect_classes():
    expected = ["Healthy", "Bruise_or_Scratch", "Scab_or_Canker", "Rot_or_Blight"]
    assert UNIFIED_DEFECT_CLASSES == expected


def test_status_constants():
    assert STATUS_FRESH == "FRESH"
    assert STATUS_REJECT == "REJECT"


def test_fresh_and_defect_helpers():
    # Fresh / healthy variants
    assert is_fresh_class("Healthy") is True
    assert is_fresh_class("healthy") is True
    assert is_fresh_class("fresh") is True
    assert is_fresh_class("apple_fresh") is True
    assert is_fresh_class("orange_fresh") is True

    # Defect variants
    assert is_fresh_class("Bruise_or_Scratch") is False
    assert is_fresh_class("Scab_or_Canker") is False
    assert is_fresh_class("Rot_or_Blight") is False
    assert is_fresh_class("apple_rotten") is False
    assert is_fresh_class("mango_bruised") is False

    # Defect helper is exact negation
    for label in [
        "Healthy",
        "healthy",
        "fresh",
        "apple_fresh",
        "Bruise_or_Scratch",
        "Scab_or_Canker",
        "Rot_or_Blight",
        "apple_rotten",
        "guava_bug_bite",
    ]:
        assert is_defect_class(label) == (not is_fresh_class(label))


def test_fruits_and_compound_classes_compatibility():
    assert "apple" in FRUITS
    assert "pomegranate" in EXTENDED_FRUITS
    assert len(COMPOUND_CLASSES) == len(FRUITS) * len(CONDITIONS)
    assert len(EXTENDED_COMPOUND_CLASSES) == len(EXTENDED_FRUITS) * len(CONDITIONS)


def test_load_app_config_schema():
    cfg = load_app_config()
    classes_cfg = cfg.get("classes", {})
    assert classes_cfg.get("defect_classes") == [
        "Healthy",
        "Bruise_or_Scratch",
        "Scab_or_Canker",
        "Rot_or_Blight",
    ]
    assert classes_cfg.get("nc") == 4
    assert classes_cfg.get("extended_fruits") == [
        "apple",
        "guava",
        "orange",
        "mango",
        "pomegranate",
    ]


def test_unified_4class_model_outputs_and_actuation_contracts():
    """Verify rebuilt 4-class unified model output mapping and actuation contracts.

    Mapping:
      Class 0 -> Healthy -> Actuation 0° (PASS)
      Class 1 -> Bruise_or_Scratch -> Actuation 90° (REJECT)
      Class 2 -> Scab_or_Canker -> Actuation 90° (REJECT)
      Class 3 -> Rot_or_Blight -> Actuation 90° (REJECT)
    """
    cfg = load_app_config()
    servo_cfg = cfg.get("hardware", {}).get("servo", {})
    rest_angle = servo_cfg.get("rest_angle", 0.0)
    reject_angle = servo_cfg.get("reject_angle", 90.0)

    assert rest_angle == 0.0
    assert reject_angle == 90.0

    unified_classes = UNIFIED_DEFECT_CLASSES
    assert len(unified_classes) == 4
    assert unified_classes[0] == "Healthy"
    assert unified_classes[1] == "Bruise_or_Scratch"
    assert unified_classes[2] == "Scab_or_Canker"
    assert unified_classes[3] == "Rot_or_Blight"

    # Map class index to expected actuation angle
    expected_angles = {
        0: rest_angle,     # Healthy -> 0° PASS
        1: reject_angle,   # Bruise_or_Scratch -> 90° REJECT
        2: reject_angle,   # Scab_or_Canker -> 90° REJECT
        3: reject_angle,   # Rot_or_Blight -> 90° REJECT
    }

    for idx, expected_angle in expected_angles.items():
        class_name = unified_classes[idx]
        if idx == 0:
            assert is_fresh_class(class_name) is True
            assert is_defect_class(class_name) is False
            assert expected_angle == 0.0
        else:
            assert is_fresh_class(class_name) is False
            assert is_defect_class(class_name) is True
            assert expected_angle == 90.0

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

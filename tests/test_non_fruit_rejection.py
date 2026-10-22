"""Comprehensive unit and integration tests for Two-Stage Non-Fruit & Noise Rejection.

Verifies:
  1. Bounding boxes under 60x60 (e.g. eye of size 30x30, area < 3600) are rejected as non-fruit.
  2. Verified fruit crops with defects (rot, bruise, scar, bug bite) are classified as REJECT.
  3. Verified clean fruit crops are classified as FRESH.
  4. All 4 fruit types (apple, banana, orange, mango) can be selected, detected, and classified.
  5. Human skin tone and facial regions produce ZERO fruit detections.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from app.contracts import (
    COMPOUND_CLASSES,
    FRUITS,
    FruitDetection,
    is_defect_class,
    is_fresh_class,
    parse_compound_label,
)
from app.detector import (
    MIN_FRUIT_AREA,
    MIN_FRUIT_HEIGHT,
    MIN_FRUIT_WIDTH,
    FruitDetector,
)


@pytest.fixture(scope="module")
def detector() -> FruitDetector:
    """Instantiate FruitDetector once for test suite."""
    return FruitDetector(prefer_hailo=False, fallback_to_cpu=True)


# ============================================================================
# Test 1: Bounding boxes under 60x60 rejected as non-fruit noise
# ============================================================================

def test_under_60x60_geometry_guard():
    """Verify validate_geometry strictly rejects width < 60, height < 60, or area < 3600."""
    # 30x30 pupil/eye box
    assert not FruitDetector.validate_geometry((10, 10, 40, 40))
    # 59x59 box (area 3481 < 3600)
    assert not FruitDetector.validate_geometry((0, 0, 59, 59))
    # 20x100 elongated sliver (width 20 < 60)
    assert not FruitDetector.validate_geometry((0, 0, 20, 100))
    # 100x10 elongated line (height 10 < 60)
    assert not FruitDetector.validate_geometry((0, 0, 100, 10))
    # 10x400 extreme aspect ratio
    assert not FruitDetector.validate_geometry((0, 0, 10, 400))

    # Valid fruit boxes: >= 60x60 and area >= 3600
    assert FruitDetector.validate_geometry((0, 0, 60, 60))
    assert FruitDetector.validate_geometry((100, 100, 220, 220))
    assert FruitDetector.validate_geometry((50, 50, 150, 200))


def test_small_sub_60_image_produces_zero_detections(detector: FruitDetector):
    """Verify input frames under 60x60 produce ZERO fruit detections."""
    # 30x30 eye simulator image
    eye_img = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.circle(eye_img, (15, 15), 10, (255, 255, 255), -1)  # sclera
    cv2.circle(eye_img, (15, 15), 5, (20, 30, 50), -1)     # pupil

    detections = detector.detect(eye_img)
    assert len(detections) == 0, f"Expected 0 detections for 30x30 eye, got {len(detections)}"

    # 45x45 noise button
    button_img = np.full((45, 45, 3), 100, dtype=np.uint8)
    detections = detector.detect(button_img)
    assert len(detections) == 0


def test_human_skin_and_face_suppression(detector: FruitDetector):
    """Verify human skin tone dominance is rejected and produces ZERO fruit detections."""
    # Typical skin chromaticity: BGR (130, 160, 215)
    skin_crop = np.full((120, 120, 3), (130, 160, 215), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(skin_crop) is True

    # High saturation fruit patches MUST NOT be rejected as skin
    orange_fruit = np.full((120, 120, 3), (0, 140, 255), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(orange_fruit) is False

    red_apple = np.full((120, 120, 3), (20, 20, 220), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(red_apple) is False

    # Detection on skin canvas must yield zero fruit detections
    skin_canvas = np.full((480, 640, 3), (135, 165, 215), dtype=np.uint8)
    # Add a small 25x25 dark feature resembling an eye/pupil
    cv2.circle(skin_canvas, (320, 240), 12, (20, 20, 20), -1)

    detections = detector.detect(skin_canvas)
    assert len(detections) == 0, "Human face/skin canvas must produce ZERO fruit detections"


# ============================================================================
# Test 2: Verified fruit crops with defects classified as REJECT
# ============================================================================

def test_defective_crops_classified_as_reject(detector: FruitDetector):
    """Verify crops with dark necrotic rot spots or bruises are classified as REJECT."""
    for fruit in FRUITS:
        # Create fruit background
        if fruit == "apple":
            crop = np.full((140, 140, 3), (25, 25, 220), dtype=np.uint8)
        elif fruit == "banana":
            crop = np.full((140, 140, 3), (10, 210, 235), dtype=np.uint8)
        elif fruit == "orange":
            crop = np.full((140, 140, 3), (0, 135, 255), dtype=np.uint8)
        else:  # mango
            crop = np.full((140, 140, 3), (20, 180, 240), dtype=np.uint8)

        # Inject necrotic rot defect lesion
        cv2.circle(crop, (70, 70), 22, (15, 20, 25), -1)

        condition, compound, conf, is_defect = detector.classify_crop_defect(crop, fruit)
        assert is_defect is True, f"Defective {fruit} crop should have is_defect=True"
        assert is_defect_class(compound) is True
        assert condition in ["rotten", "bruised", "scarred", "bug_bite"]
        assert compound == f"{fruit}_{condition}"
        assert conf > 0.5


def test_bruised_fruit_crop_defect_classification(detector: FruitDetector):
    """Verify bruised discoloration on fruit crop is classified as REJECT."""
    # Banana with brownish bruise patch
    banana = np.full((130, 130, 3), (0, 220, 245), dtype=np.uint8)
    # Brown bruise patch in HSV: H in [10, 25], S in [80, 160], V in [40, 100]
    # BGR approx (30, 60, 95)
    cv2.circle(banana, (65, 65), 24, (30, 65, 95), -1)

    condition, compound, conf, is_defect = detector.classify_crop_defect(banana, "banana")
    assert is_defect is True
    assert is_defect_class(compound) is True
    assert compound in ["banana_bruised", "banana_rotten", "banana_scarred"]


# ============================================================================
# Test 3: Verified clean fruit crops classified as FRESH
# ============================================================================

def test_clean_fruit_crops_classified_as_fresh(detector: FruitDetector):
    """Verify healthy, clean fruit crops without defects are classified as FRESH."""
    fruit_colors = {
        "apple": (25, 25, 220),       # Red
        "banana": (10, 220, 240),     # Bright yellow
        "orange": (0, 140, 255),      # Vivid orange
        "mango": (20, 185, 240),      # Yellow-orange
    }

    for fruit, bgr in fruit_colors.items():
        clean_crop = np.full((140, 140, 3), bgr, dtype=np.uint8)
        condition, compound, conf, is_defect = detector.classify_crop_defect(clean_crop, fruit)

        assert is_defect is False, f"Clean {fruit} should be fresh (is_defect=False)"
        assert is_fresh_class(compound) is True
        assert condition == "fresh"
        assert compound == f"{fruit}_fresh"
        assert conf >= 0.85


# ============================================================================
# Test 4: All 4 fruit types can be selected and detected
# ============================================================================

def test_mango_chromatic_contour_convexity_detection(detector: FruitDetector):
    """Verify mango detection via chromatic yellow/orange/green contour & convexity."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw oval convex mango: size 140x90, area ~10,000 > 3600
    cv2.ellipse(canvas, (320, 240), (70, 45), 25, 0, 360, (20, 180, 240), -1)

    detections = detector.detect(canvas, target_fruit="mango")
    assert len(detections) >= 1, "Expected at least 1 mango detection"
    det = detections[0]
    assert det.fruit_type == "mango"
    assert det.class_name in ["mango_fresh", "mango_rotten", "mango_bruised"]
    assert det.bbox[2] - det.bbox[0] >= 60
    assert det.bbox[3] - det.bbox[1] >= 60


def test_fruit_filter_selection(detector: FruitDetector):
    """Verify target_fruit filter filters detections for Apple, Banana, Orange, Mango."""
    # Multi-fruit canvas with Apple and Orange
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Apple (red circle)
    cv2.circle(canvas, (180, 200), 50, (20, 20, 220), -1)
    # Orange (orange circle)
    cv2.circle(canvas, (460, 200), 50, (0, 140, 255), -1)

    # Filter by Apple only
    apple_dets = detector.detect(canvas, target_fruit="apple")
    for d in apple_dets:
        assert d.fruit_type == "apple", f"Expected only apple when filtered, got {d.fruit_type}"

    # Filter by Orange only
    orange_dets = detector.detect(canvas, target_fruit="orange")
    for d in orange_dets:
        assert d.fruit_type == "orange", f"Expected only orange when filtered, got {d.fruit_type}"

    # Filter by All fruits
    all_dets = detector.detect(canvas, target_fruit="all")
    fruit_types = {d.fruit_type for d in all_dets}
    # Should contain detected fruits
    assert len(all_dets) >= 1


def test_all_20_compound_classes_supported(detector: FruitDetector):
    """Verify all 20 compound classes are recognized across all 4 fruits."""
    assert len(detector.class_names) == 20
    for fruit in ["apple", "banana", "orange", "mango"]:
        for cond in ["fresh", "rotten", "bruised", "scarred", "bug_bite"]:
            expected_compound = f"{fruit}_{cond}"
            assert expected_compound in detector.class_names


# ============================================================================
# Test 5: Guava & Pomegranate Detection and Stage 2 Defect Evaluation
# ============================================================================

def test_guava_candidate_detection(detector: FruitDetector):
    """Verify guava detection via spherical geometry & green/yellow-green chromatic profile."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Guava color in HSV: H=50, S=160, V=180 -> BGR (approx (35, 180, 120))
    hsv_guava = np.uint8([[[50, 160, 180]]])
    bgr_guava = tuple(int(c) for c in cv2.cvtColor(hsv_guava, cv2.COLOR_HSV2BGR)[0, 0])

    # Draw spherical guava: radius 50 (100x100 >= 60x60, area ~7854 >= 3600)
    cv2.circle(canvas, (320, 240), 50, bgr_guava, -1)

    # 1. Direct candidate detection
    candidates = detector._detect_guava_candidates(canvas)
    assert len(candidates) >= 1, "Expected at least 1 guava candidate"
    fruit, conf, box = candidates[0]
    assert fruit == "guava"
    assert conf >= 0.70
    assert box[2] - box[0] >= 60
    assert box[3] - box[1] >= 60

    # 2. End-to-end detect() with target_fruit="guava"
    detections = detector.detect(canvas, target_fruit="guava")
    assert len(detections) >= 1
    assert detections[0].fruit_type == "guava"
    assert detections[0].class_name == "guava_fresh"
    assert detections[0].condition == "fresh"
    assert detections[0].is_defect is False


def test_pomegranate_candidate_detection(detector: FruitDetector):
    """Verify pomegranate detection via globose geometry & red/ruby-pink chromatic profile."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Pomegranate color in HSV: H=5 (red), S=180, V=170 -> BGR
    hsv_pom = np.uint8([[[5, 180, 170]]])
    bgr_pom = tuple(int(c) for c in cv2.cvtColor(hsv_pom, cv2.COLOR_HSV2BGR)[0, 0])

    # Draw globose pomegranate: radius 50 (100x100 >= 60x60, area ~7854 >= 3600)
    cv2.circle(canvas, (320, 240), 50, bgr_pom, -1)

    # 1. Direct candidate detection
    candidates = detector._detect_pomegranate_candidates(canvas)
    assert len(candidates) >= 1, "Expected at least 1 pomegranate candidate"
    fruit, conf, box = candidates[0]
    assert fruit == "pomegranate"
    assert conf >= 0.70
    assert box[2] - box[0] >= 60
    assert box[3] - box[1] >= 60

    # 2. End-to-end detect() with target_fruit="pomegranate"
    detections = detector.detect(canvas, target_fruit="pomegranate")
    assert len(detections) >= 1
    assert detections[0].fruit_type == "pomegranate"
    assert detections[0].class_name == "pomegranate_fresh"
    assert detections[0].condition == "fresh"
    assert detections[0].is_defect is False


def test_guava_and_pomegranate_defect_evaluation(detector: FruitDetector):
    """Verify Stage 2 defect evaluation for guava and pomegranate crops."""
    # Guava crop with necrotic rot lesion
    hsv_guava = np.uint8([[[50, 160, 180]]])
    bgr_guava = tuple(int(c) for c in cv2.cvtColor(hsv_guava, cv2.COLOR_HSV2BGR)[0, 0])
    guava_crop = np.full((140, 140, 3), bgr_guava, dtype=np.uint8)
    cv2.circle(guava_crop, (70, 70), 22, (15, 20, 25), -1)  # rot lesion

    cond, comp, conf, is_def = detector.classify_crop_defect(guava_crop, "guava")
    assert is_def is True
    assert cond in ["rotten", "bruised", "scarred", "bug_bite"]
    assert comp == f"guava_{cond}"

    # Pomegranate crop with necrotic rot lesion
    hsv_pom = np.uint8([[[5, 180, 170]]])
    bgr_pom = tuple(int(c) for c in cv2.cvtColor(hsv_pom, cv2.COLOR_HSV2BGR)[0, 0])
    pom_crop = np.full((140, 140, 3), bgr_pom, dtype=np.uint8)
    cv2.circle(pom_crop, (70, 70), 22, (15, 20, 25), -1)  # rot lesion

    cond, comp, conf, is_def = detector.classify_crop_defect(pom_crop, "pomegranate")
    assert is_def is True
    assert cond in ["rotten", "bruised", "scarred", "bug_bite"]
    assert comp == f"pomegranate_{cond}"

    # Clean guava crop
    clean_guava = np.full((140, 140, 3), bgr_guava, dtype=np.uint8)
    cond, comp, conf, is_def = detector.classify_crop_defect(clean_guava, "guava")
    assert is_def is False
    assert cond == "fresh"
    assert comp == "guava_fresh"

    # Clean pomegranate crop
    clean_pom = np.full((140, 140, 3), bgr_pom, dtype=np.uint8)
    cond, comp, conf, is_def = detector.classify_crop_defect(clean_pom, "pomegranate")
    assert is_def is False
    assert cond == "fresh"
    assert comp == "pomegranate_fresh"

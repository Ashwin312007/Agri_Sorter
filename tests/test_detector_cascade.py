"""Test suite for calibrated fruit detector cascade and Stage 2 defect classification.

Validates:
  1. Orange detection under normal indoor webcam lighting (relaxed saturation bound >= 75)
     and skin suppression bypass for genuine orange fruit (H in [5, 25], S >= 80, V >= 90).
  2. Guava detection with relaxed circularity (>= 0.38) and expanded green hue range ([28, 88]).
  3. Pomegranate detection with relaxed circularity (>= 0.40) and dual red hue bands ([0, 15], [160, 180]).
  4. Unified Stage 2 Defect Classification to 4 classes:
     - Class 0: 'Healthy' -> is_defect = False -> PASS (Servo 0°) -> Green
     - Class 1: 'Bruise_or_Scratch' -> is_defect = True -> REJECT (Servo 90°) -> Yellow
     - Class 2: 'Scab_or_Canker' -> is_defect = True -> REJECT (Servo 90°) -> Orange
     - Class 3: 'Rot_or_Blight' -> is_defect = True -> REJECT (Servo 90°) -> Red
  5. Human face, eye, and skin suppression strictly produces ZERO fruit detections.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from app.detector import (
    FruitDetector,
    STAGE2_CLASSES,
    STAGE2_CLASS_MAP,
    STAGE2_COLOR_CODING,
    STAGE2_SERVO_MAP,
    DefectCondition,
    get_defect_color,
    make_defect_condition,
)


@pytest.fixture(scope="module")
def detector() -> FruitDetector:
    """Instantiate FruitDetector once for cascade tests."""
    return FruitDetector(prefer_hailo=False, fallback_to_cpu=True)


# ==============================================================================
# Test 1: Orange Detection & Skin Suppression Exemption
# ==============================================================================

def test_orange_skin_suppression_exemption():
    """Verify solid orange fruit (H in [5, 25], S >= 80, V >= 90) is NOT treated as skin."""
    # Vivid orange fruit
    vivid_orange = np.full((100, 100, 3), (0, 140, 255), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(vivid_orange) is False
    assert FruitDetector.is_skin_tone_dominant(vivid_orange, "orange") is False

    # Indoor webcam orange with BGR (20, 110, 210) -> H=14, S=230, V=210, low blue
    indoor_orange = np.full((100, 100, 3), (20, 110, 210), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(indoor_orange) is False
    assert FruitDetector.is_skin_tone_dominant(indoor_orange, "orange") is False

    # Human skin must still be suppressed
    human_skin = np.full((100, 100, 3), (130, 160, 215), dtype=np.uint8)
    assert FruitDetector.is_skin_tone_dominant(human_skin) is True


def test_orange_color_geometry_indoor_lighting(detector: FruitDetector):
    """Verify orange detection under indoor webcam lighting with saturation down to 75."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Create orange with HSV: H=15, S=80, V=190
    hsv_orange = np.uint8([[[15, 80, 190]]])
    bgr_orange = tuple(int(c) for c in cv2.cvtColor(hsv_orange, cv2.COLOR_HSV2BGR)[0, 0])

    # Draw circular orange: radius 55 (110x110 >= 60x60, area ~9500 >= 3600)
    cv2.circle(canvas, (320, 240), 55, bgr_orange, -1)

    # Detect via complementary color geometry
    candidates = detector._detect_color_geometry(canvas, target="orange")
    assert len(candidates) >= 1, "Orange in normal indoor lighting must be detected"
    fruit, conf, box = candidates[0]
    assert fruit == "orange"
    assert box[2] - box[0] >= 60
    assert box[3] - box[1] >= 60

    # End-to-end detection
    dets = detector.detect(canvas, target_fruit="orange")
    assert len(dets) >= 1
    assert dets[0].fruit_type == "orange"


# ==============================================================================
# Test 2: Guava Detection (Relaxed circularity >= 0.38 & Hue [28, 88])
# ==============================================================================

def test_guava_globose_and_oval_detection(detector: FruitDetector):
    """Verify guava detection with relaxed circularity >= 0.38 for oval / hand-held guavas."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Green-yellow guava HSV: H=55, S=120, V=170
    hsv_guava = np.uint8([[[55, 120, 170]]])
    bgr_guava = tuple(int(c) for c in cv2.cvtColor(hsv_guava, cv2.COLOR_HSV2BGR)[0, 0])

    # Draw slightly oval guava: axes (60, 45), angle 20
    cv2.ellipse(canvas, (320, 240), (60, 45), 20, 0, 360, bgr_guava, -1)

    # 1. Alias and candidate check
    candidates = detector._detect_guava_globose(canvas)
    assert len(candidates) >= 1, "Oval guava with circularity >= 0.38 should be detected"
    fruit, conf, box = candidates[0]
    assert fruit == "guava"
    assert conf >= 0.70

    # 2. End-to-end detection
    dets = detector.detect(canvas, target_fruit="guava")
    assert len(dets) >= 1
    assert dets[0].fruit_type == "guava"
    assert dets[0].condition == "fresh"
    assert dets[0].is_defect is False


def test_guava_hue_expansion(detector: FruitDetector):
    """Verify guava green hue range [28, 88] is detected."""
    for test_h in [30, 45, 60, 80]:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv_color = np.uint8([[[test_h, 110, 180]]])
        bgr = tuple(int(c) for c in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(canvas, (320, 240), 50, bgr, -1)

        candidates = detector._detect_guava_candidates(canvas)
        assert len(candidates) >= 1, f"Guava with Hue={test_h} should be detected"


# ==============================================================================
# Test 3: Pomegranate Detection (Relaxed circularity >= 0.40 & Red H in [0, 15], [160, 180])
# ==============================================================================

def test_pomegranate_calyx_crown_detection(detector: FruitDetector):
    """Verify pomegranate detection with calyx crown and uneven hand-held angles (C >= 0.40)."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Low red HSV: H=6, S=160, V=180
    hsv_pom = np.uint8([[[6, 160, 180]]])
    bgr_pom = tuple(int(c) for c in cv2.cvtColor(hsv_pom, cv2.COLOR_HSV2BGR)[0, 0])

    # Draw globose body + small calyx crown polygon
    cv2.circle(canvas, (320, 240), 50, bgr_pom, -1)
    # Calyx crown triangles at top
    crown_pts = np.array([[310, 190], [320, 175], [330, 190]], np.int32)
    cv2.fillPoly(canvas, [crown_pts], bgr_pom)

    candidates = detector._detect_pomegranate_calyx(canvas)
    assert len(candidates) >= 1, "Pomegranate with calyx crown should be detected"
    fruit, conf, box = candidates[0]
    assert fruit == "pomegranate"
    assert conf >= 0.70

    dets = detector.detect(canvas, target_fruit="pomegranate")
    assert len(dets) >= 1
    assert dets[0].fruit_type == "pomegranate"


def test_pomegranate_dual_hue_bands(detector: FruitDetector):
    """Verify pomegranate detection across both low red (0-15) and high red (160-180)."""
    for red_h in [4, 12, 165, 175]:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv_color = np.uint8([[[red_h, 150, 180]]])
        bgr = tuple(int(c) for c in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(canvas, (320, 240), 50, bgr, -1)

        candidates = detector._detect_pomegranate_candidates(canvas)
        assert len(candidates) >= 1, f"Pomegranate with Red Hue={red_h} should be detected"


# ==============================================================================
# Test 4: Stage 2 Defect Classification (Unified 4 Classes)
# ==============================================================================

def test_stage2_unified_4_classes_mapping(detector: FruitDetector):
    """Verify 4 unified Stage 2 defect classes, servo angles, and color codings."""
    # Class 0: Healthy
    c0 = make_defect_condition("fresh")
    assert c0.class_id == 0
    assert c0.unified_name == "Healthy"
    assert c0.is_defect is False
    assert c0.servo_angle == 0
    assert c0.color_bgr == (0, 220, 0)
    assert c0 == "Healthy"
    assert c0 == "fresh"

    # Class 1: Bruise_or_Scratch
    c1 = make_defect_condition("bruised")
    assert c1.class_id == 1
    assert c1.unified_name == "Bruise_or_Scratch"
    assert c1.is_defect is True
    assert c1.servo_angle == 90
    assert c1.color_bgr == (0, 220, 255)  # Yellow
    assert c1 == "Bruise_or_Scratch"
    assert c1 == "bruised"

    c1_alt = make_defect_condition("scarred")
    assert c1_alt.class_id == 1
    assert c1_alt.unified_name == "Bruise_or_Scratch"
    assert c1_alt.is_defect is True
    assert c1_alt.servo_angle == 90

    # Class 2: Scab_or_Canker
    c2 = make_defect_condition("bug_bite")
    assert c2.class_id == 2
    assert c2.unified_name == "Scab_or_Canker"
    assert c2.is_defect is True
    assert c2.servo_angle == 90
    assert c2.color_bgr == (0, 140, 255)  # Orange
    assert c2 == "Scab_or_Canker"
    assert c2 == "bug_bite"

    # Class 3: Rot_or_Blight
    c3 = make_defect_condition("rotten")
    assert c3.class_id == 3
    assert c3.unified_name == "Rot_or_Blight"
    assert c3.is_defect is True
    assert c3.servo_angle == 90
    assert c3.color_bgr == (0, 0, 220)  # Red
    assert c3 == "Rot_or_Blight"
    assert c3 == "rotten"


def test_classify_crop_defect_returns_unified_condition(detector: FruitDetector):
    """Verify classify_crop_defect returns DefectCondition with unified 4-class metadata."""
    # 1. Clean orange crop -> Healthy
    clean_orange = np.full((120, 120, 3), (0, 140, 255), dtype=np.uint8)
    cond, comp, conf, is_defect = detector.classify_crop_defect(clean_orange, "orange")
    assert is_defect is False
    assert cond == "Healthy"
    assert cond == "fresh"
    assert cond.servo_angle == 0
    assert comp == "orange_fresh"

    # 2. Rotten apple crop -> Rot_or_Blight
    rotten_apple = np.full((120, 120, 3), (25, 25, 220), dtype=np.uint8)
    cv2.circle(rotten_apple, (60, 60), 20, (15, 20, 25), -1)  # necrotic decay
    cond, comp, conf, is_defect = detector.classify_crop_defect(rotten_apple, "apple")
    assert is_defect is True
    assert cond == "Rot_or_Blight"
    assert cond == "rotten"
    assert cond.servo_angle == 90
    assert comp == "apple_rotten"

    # 3. Bruised guava crop -> Bruise_or_Scratch
    hsv_guava = np.uint8([[[50, 160, 180]]])
    bgr_guava = tuple(int(c) for c in cv2.cvtColor(hsv_guava, cv2.COLOR_HSV2BGR)[0, 0])
    bruised_guava = np.full((120, 120, 3), bgr_guava, dtype=np.uint8)
    cv2.circle(bruised_guava, (60, 60), 22, (30, 65, 95), -1)  # brown bruise
    cond, comp, conf, is_defect = detector.classify_crop_defect(bruised_guava, "guava")
    assert is_defect is True
    assert cond.class_id in [1, 3]  # Bruise_or_Scratch or Rot_or_Blight
    assert cond.servo_angle == 90

    # 4. Punctured pomegranate crop -> Scab_or_Canker
    hsv_pom = np.uint8([[[5, 180, 170]]])
    bgr_pom = tuple(int(c) for c in cv2.cvtColor(hsv_pom, cv2.COLOR_HSV2BGR)[0, 0])
    punctured_pom = np.full((120, 120, 3), bgr_pom, dtype=np.uint8)
    cv2.circle(punctured_pom, (60, 60), 6, (10, 15, 20), -1)  # small bug bite puncture
    cond, comp, conf, is_defect = detector.classify_crop_defect(punctured_pom, "pomegranate")
    assert is_defect is True
    assert cond.class_id in [1, 2, 3]
    assert cond.servo_angle == 90


def test_frame_annotation_color_coding(detector: FruitDetector):
    """Verify annotate_frame draws bounding boxes with designated colors."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a healthy apple detection
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(canvas, (320, 240), 50, (25, 25, 220), -1)
    dets = detector.detect(canvas, target_fruit="apple")
    assert len(dets) >= 1

    annotated = FruitDetector.annotate_frame(canvas, dets)
    assert annotated is not None
    assert annotated.shape == canvas.shape
    # Border pixels around detection box should have Healthy Green (0, 220, 0)
    color = FruitDetector.get_stage2_defect_color(dets[0].condition)
    assert color == STAGE2_COLOR_CODING["Healthy"]


# ==============================================================================
# Test 5: Face / Eye Suppression Security Guard
# ==============================================================================

def test_face_and_eye_suppression_produces_zero_fruit(detector: FruitDetector):
    """Ensure facial images, eyes, and skin patches produce ZERO fruit detections."""
    # 1. 30x30 eye simulation
    eye = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.circle(eye, (15, 15), 12, (255, 255, 255), -1)
    cv2.circle(eye, (15, 15), 6, (20, 30, 50), -1)
    assert len(detector.detect(eye)) == 0

    # 2. 50x50 button / mole noise
    noise = np.full((50, 50, 3), (80, 80, 80), dtype=np.uint8)
    assert len(detector.detect(noise)) == 0

    # 3. Synthetic face canvas
    face_canvas = np.full((480, 640, 3), (130, 160, 215), dtype=np.uint8)
    # Eyes
    cv2.circle(face_canvas, (260, 200), 15, (255, 255, 255), -1)
    cv2.circle(face_canvas, (260, 200), 7, (20, 20, 20), -1)
    cv2.circle(face_canvas, (380, 200), 15, (255, 255, 255), -1)
    cv2.circle(face_canvas, (380, 200), 7, (20, 20, 20), -1)
    # Mouth
    cv2.ellipse(face_canvas, (320, 320), (50, 15), 0, 0, 180, (70, 70, 180), -1)

    dets = detector.detect(face_canvas)
    assert len(dets) == 0, f"Human face canvas must produce ZERO fruit detections, got {len(dets)}"


def test_unified_4class_detector_actuation_angles():
    """Verify rebuilt 4-class unified detector classes match strict actuation angles.

    Healthy: 0° (PASS)
    Bruise_or_Scratch: 90° (REJECT)
    Scab_or_Canker: 90° (REJECT)
    Rot_or_Blight: 90° (REJECT)
    """
    for cls_name in ["Healthy", "Bruise_or_Scratch", "Scab_or_Canker", "Rot_or_Blight"]:
        cond = make_defect_condition(cls_name)
        if cls_name == "Healthy":
            assert cond.servo_angle == 0
            assert cond.is_defect is False
            assert STAGE2_SERVO_MAP[cls_name] == 0
        else:
            assert cond.servo_angle == 90
            assert cond.is_defect is True
            assert STAGE2_SERVO_MAP[cls_name] == 90

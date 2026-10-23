"""Unit and integration tests for Agri_Sorter Web Dashboard.

Tests:
1. HTML Fruit Selector Buttons (All Fruits (Core 4), Apple, Guava, Orange, Mango, Pomegranate).
2. Backend fruit filter support for ["all", "apple", "guava", "orange", "mango", "pomegranate"].
3. Accurate telemetry table and statistics tracking for Guava and Pomegranate.
4. HTTP API endpoints /api/set_fruit, /api/status.
"""

import json
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.web_dashboard import (
    HTML_PAGE,
    SUPPORTED_FRUITS,
    SorterHub,
)


def test_html_fruit_selector_buttons():
    """Verify HTML contains all 6 required buttons with exact labels and callbacks."""
    expected_buttons = [
        '<button class="btn-fruit active" onclick="setFruit(\'all\', this)">🌐 All Fruits (Core 4)</button>',
        '<button class="btn-fruit" onclick="setFruit(\'apple\', this)">🍏 Apple</button>',
        '<button class="btn-fruit" onclick="setFruit(\'guava\', this)">🍈 Guava</button>',
        '<button class="btn-fruit" onclick="setFruit(\'orange\', this)">🍊 Orange</button>',
        '<button class="btn-fruit" onclick="setFruit(\'mango\', this)">🥭 Mango</button>',
        '<button class="btn-fruit" onclick="setFruit(\'pomegranate\', this)">🍎 Pomegranate</button>',
    ]
    for btn in expected_buttons:
        assert btn in HTML_PAGE, f"Missing button in HTML_PAGE: {btn}"

    # Banana should no longer be a selector button
    assert "🍌 Bananas Only" not in HTML_PAGE
    assert "setFruit('banana'" not in HTML_PAGE


def test_supported_fruits_constant():
    """Verify SUPPORTED_FRUITS contains the required 6 options."""
    expected = ["all", "apple", "guava", "orange", "mango", "pomegranate"]
    assert SUPPORTED_FRUITS == expected


def test_fruit_breakdown_bar_in_html():
    """Verify fruit statistics pills for Guava and Pomegranate exist in HTML."""
    assert 'id="cntGuava"' in HTML_PAGE
    assert 'id="cntPomegranate"' in HTML_PAGE
    assert 'id="cntApple"' in HTML_PAGE
    assert 'id="cntOrange"' in HTML_PAGE
    assert 'id="cntMango"' in HTML_PAGE


@patch("app.web_dashboard.FruitDetector")
@patch("app.web_dashboard.FruitTracker")
@patch("app.web_dashboard.create_hardware_coordinator_from_config")
@patch("cv2.VideoCapture")
def test_sorter_hub_fruit_filters(mock_cap, mock_coord, mock_tracker, mock_detector):
    """Verify set_fruit_filter supports all 6 targets and ignores invalid ones."""
    hub = SorterHub()
    # Stop background thread loop for clean testing
    hub.running = False

    targets = ["all", "apple", "guava", "orange", "mango", "pomegranate"]
    for fruit in targets:
        hub.set_fruit_filter(fruit)
        assert hub.selected_fruit == fruit

    # Unsupported fruit should be rejected
    hub.set_fruit_filter("dragonfruit")
    assert hub.selected_fruit == "pomegranate"  # remains at previous value


@patch("app.web_dashboard.FruitDetector")
@patch("app.web_dashboard.FruitTracker")
@patch("app.web_dashboard.create_hardware_coordinator_from_config")
@patch("cv2.VideoCapture")
def test_telemetry_and_statistics_guava(mock_cap, mock_coord, mock_tracker, mock_detector):
    """Verify telemetry table and statistics accurately track Guava."""
    mock_detector.return_value.detect.return_value = []
    mock_cap.return_value.isOpened.return_value = False
    hub = SorterHub()
    hub.running = False

    # Create dummy image
    test_img = np.full((300, 300, 3), 100, dtype=np.uint8)
    _, enc_img = cv2.imencode(".jpg", test_img)
    img_bytes = enc_img.tobytes()

    # Test fresh guava photo
    event_fresh = hub.process_test_photo(img_bytes, filename="guava_fresh_test.jpg")
    assert event_fresh["fruit_type"] == "guava"
    assert "Guava" in event_fresh["item_name"]
    assert event_fresh["status"] == "FRESH"
    assert "PASS" in event_fresh["servo_action"]
    assert hub.fruit_counts["guava"] == 1
    assert hub.stats["fruit_counts"]["guava"] == 1
    assert hub.stats["fresh_count"] >= 1
    assert hub.recent_events[0]["item_name"] == event_fresh["item_name"]

    # Test rotten guava photo
    event_reject = hub.process_test_photo(img_bytes, filename="guava_rotten_sample.jpg")
    assert event_reject["fruit_type"] == "guava"
    assert "Guava" in event_reject["item_name"]
    assert event_reject["status"] == "REJECT"
    assert "REJECT" in event_reject["servo_action"]
    assert hub.fruit_counts["guava"] == 2
    assert hub.stats["fruit_counts"]["guava"] == 2
    assert hub.stats["reject_count"] >= 1
    assert hub.recent_events[0]["status"] == "REJECT"


@patch("app.web_dashboard.FruitDetector")
@patch("app.web_dashboard.FruitTracker")
@patch("app.web_dashboard.create_hardware_coordinator_from_config")
@patch("cv2.VideoCapture")
def test_telemetry_and_statistics_pomegranate(mock_cap, mock_coord, mock_tracker, mock_detector):
    """Verify telemetry table and statistics accurately track Pomegranate."""
    mock_detector.return_value.detect.return_value = []
    mock_cap.return_value.isOpened.return_value = False
    hub = SorterHub()
    hub.running = False

    test_img = np.full((300, 300, 3), 120, dtype=np.uint8)
    _, enc_img = cv2.imencode(".jpg", test_img)
    img_bytes = enc_img.tobytes()

    # Test defect pomegranate photo
    event_reject = hub.process_test_photo(img_bytes, filename="pomegranate_defect_01.jpg")
    assert event_reject["fruit_type"] == "pomegranate"
    assert "Pomegranate" in event_reject["item_name"]
    assert event_reject["status"] == "REJECT"
    assert "REJECT" in event_reject["servo_action"]
    assert hub.fruit_counts["pomegranate"] == 1
    assert hub.stats["fruit_counts"]["pomegranate"] == 1
    assert hub.stats["reject_count"] >= 1
    assert hub.recent_events[0]["item_name"] == event_reject["item_name"]

    # Test fresh pomegranate photo
    event_fresh = hub.process_test_photo(img_bytes, filename="pomegranate_fresh_02.jpg")
    assert event_fresh["fruit_type"] == "pomegranate"
    assert "Pomegranate" in event_fresh["item_name"]
    assert event_fresh["status"] == "FRESH"
    assert "PASS" in event_fresh["servo_action"]
    assert hub.fruit_counts["pomegranate"] == 2
    assert hub.stats["fruit_counts"]["pomegranate"] == 2


@patch("app.web_dashboard.FruitDetector")
@patch("app.web_dashboard.FruitTracker")
@patch("app.web_dashboard.create_hardware_coordinator_from_config")
@patch("cv2.VideoCapture")
def test_detector_output_tracking_guava_and_pomegranate(mock_cap, mock_coord, mock_tracker, mock_detector):
    """Verify when detector returns FruitDetection objects for Guava & Pomegranate, stats and telemetry are populated."""
    from app.contracts import FruitDetection
    mock_cap.return_value.isOpened.return_value = False
    hub = SorterHub()
    hub.running = False

    # Mock detector returning a guava detection
    mock_detector.return_value.detect.return_value = [
        FruitDetection(
            bbox=(50, 50, 200, 200),
            confidence=0.89,
            class_id=1,
            class_name="guava_fresh",
            fruit_type="guava",
            condition="fresh",
            is_defect=False,
        )
    ]

    test_img = np.full((300, 300, 3), 100, dtype=np.uint8)
    _, enc_img = cv2.imencode(".jpg", test_img)
    img_bytes = enc_img.tobytes()

    ev = hub.process_test_photo(img_bytes, filename="guava_camera_snap.jpg")
    assert ev["fruit_type"] == "guava"
    assert "Guava" in ev["item_name"]
    assert ev["status"] == "FRESH"
    assert hub.fruit_counts["guava"] == 1
    assert hub.stats["fruit_counts"]["guava"] == 1

    # Mock detector returning a pomegranate defect detection
    mock_detector.return_value.detect.return_value = [
        FruitDetection(
            bbox=(60, 60, 220, 220),
            confidence=0.91,
            class_id=2,
            class_name="pomegranate_rotten",
            fruit_type="pomegranate",
            condition="rotten",
            is_defect=True,
        )
    ]

    ev2 = hub.process_test_photo(img_bytes, filename="pom_camera_snap.jpg")
    assert ev2["fruit_type"] == "pomegranate"
    assert "Pomegranate" in ev2["item_name"]
    assert ev2["status"] == "REJECT"
    assert hub.fruit_counts["pomegranate"] == 1
    assert hub.stats["fruit_counts"]["pomegranate"] == 1


def test_dashboard_api_set_fruit():
    """Verify HTTP API endpoint /api/set_fruit accepts guava & pomegranate and updates filter."""
    import app.web_dashboard as wd
    from io import BytesIO

    mock_hub = MagicMock()
    mock_hub.selected_fruit = "all"
    mock_hub.stats = {"fruit_counts": {}}
    mock_hub.recent_events = []
    mock_hub.fruit_counts = {"guava": 3, "pomegranate": 2}
    mock_hub.lock = MagicMock()
    mock_hub.lock.__enter__.return_value = None
    mock_hub.lock.__exit__.return_value = None

    wd.hub = mock_hub

    # Helper to simulate GET requests
    class DummyHandler(wd.DashboardRequestHandler):
        def __init__(self, path):
            self.path = path
            self.request_version = "HTTP/1.1"
            self.command = "GET"
            self.wfile = BytesIO()
            self.headers = {}
            self.rfile = BytesIO()
            self._response_code = None
            self._headers = {}

        def send_response(self, code, message=None):
            self._response_code = code

        def send_header(self, keyword, value):
            self._headers[keyword] = value

        def end_headers(self):
            pass

    for fruit in ["all", "apple", "guava", "orange", "mango", "pomegranate"]:
        handler = DummyHandler(f"/api/set_fruit?fruit={fruit}")
        handler.do_GET()
        assert handler._response_code == 200
        resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert resp["success"] is True
        assert resp["fruit"] == fruit
        mock_hub.set_fruit_filter.assert_called_with(fruit)

    # Test /api/status endpoint
    status_handler = DummyHandler("/api/status")
    status_handler.do_GET()
    assert status_handler._response_code == 200
    status_resp = json.loads(status_handler.wfile.getvalue().decode("utf-8"))
    assert "fruit_counts" in status_resp
    assert status_resp["fruit_counts"]["guava"] == 3
    assert status_resp["fruit_counts"]["pomegranate"] == 2


def test_unified_defect_hierarchy_helpers():
    """Verify map_to_defect_hierarchy and classify_defect_hierarchy mapping rules."""
    from app.web_dashboard import classify_defect_hierarchy, map_to_defect_hierarchy

    # Healthy
    assert map_to_defect_hierarchy("fresh") == "Healthy"
    assert map_to_defect_hierarchy("healthy") == "Healthy"
    assert map_to_defect_hierarchy("apple_fresh") == "Healthy"
    assert map_to_defect_hierarchy("good") == "Healthy"

    # Bruise_or_Scratch
    assert map_to_defect_hierarchy("bruised") == "Bruise_or_Scratch"
    assert map_to_defect_hierarchy("scratch") == "Bruise_or_Scratch"
    assert map_to_defect_hierarchy("guava_bruised") == "Bruise_or_Scratch"
    assert map_to_defect_hierarchy("Bruise_or_Scratch") == "Bruise_or_Scratch"

    # Scab_or_Canker
    assert map_to_defect_hierarchy("scarred") == "Scab_or_Canker"
    assert map_to_defect_hierarchy("bug_bite") == "Scab_or_Canker"
    assert map_to_defect_hierarchy("scab") == "Scab_or_Canker"
    assert map_to_defect_hierarchy("canker") == "Scab_or_Canker"
    assert map_to_defect_hierarchy("orange_scarred") == "Scab_or_Canker"

    # Rot_or_Blight
    assert map_to_defect_hierarchy("rotten") == "Rot_or_Blight"
    assert map_to_defect_hierarchy("blight") == "Rot_or_Blight"
    assert map_to_defect_hierarchy("decay") == "Rot_or_Blight"
    assert map_to_defect_hierarchy("apple_rotten") == "Rot_or_Blight"

    # Hierarchy ranking priority: Rot > Scab > Bruise > Healthy
    assert classify_defect_hierarchy(["fresh"]) == "Healthy"
    assert classify_defect_hierarchy(["fresh", "bruised"]) == "Bruise_or_Scratch"
    assert classify_defect_hierarchy(["bruised", "scab"]) == "Scab_or_Canker"
    assert classify_defect_hierarchy(["bruised", "scab", "rotten"]) == "Rot_or_Blight"
    assert classify_defect_hierarchy([], is_reject=True) == "Rot_or_Blight"


def test_html_defect_hierarchy_elements():
    """Verify HTML contains all 4 defect badges, pills, and servo gauge displays."""
    # Defect breakdown pills
    assert 'id="cntHealthy"' in HTML_PAGE
    assert 'id="cntBruise"' in HTML_PAGE
    assert 'id="cntScab"' in HTML_PAGE
    assert 'id="cntRot"' in HTML_PAGE

    # Defect cards in right telemetry panel
    assert 'id="cardHealthy"' in HTML_PAGE
    assert 'id="cardBruise"' in HTML_PAGE
    assert 'id="cardScab"' in HTML_PAGE
    assert 'id="cardRot"' in HTML_PAGE

    # Servo display and gauge
    assert 'id="servoAngleDisplay"' in HTML_PAGE
    assert 'id="servoFlapVisual"' in HTML_PAGE
    assert '0° [PASS]' in HTML_PAGE

    # Defect hierarchy CSS classes
    assert '.defect-pill.healthy' in HTML_PAGE
    assert '.defect-pill.bruise' in HTML_PAGE
    assert '.defect-pill.scab' in HTML_PAGE
    assert '.defect-pill.rot' in HTML_PAGE


@patch("app.web_dashboard.FruitDetector")
@patch("app.web_dashboard.FruitTracker")
@patch("app.web_dashboard.create_hardware_coordinator_from_config")
@patch("cv2.VideoCapture")
def test_process_test_photo_all_4_defect_classes(mock_cap, mock_coord, mock_tracker, mock_detector):
    """Verify process_test_photo correctly identifies and sets servo angle for all 4 defect classes."""
    mock_detector.return_value.detect.return_value = []
    mock_cap.return_value.isOpened.return_value = False
    hub = SorterHub()
    hub.running = False

    test_img = np.full((300, 300, 3), 110, dtype=np.uint8)
    _, enc_img = cv2.imencode(".jpg", test_img)
    img_bytes = enc_img.tobytes()

    # 1. Healthy -> 0° PASS
    ev_healthy = hub.process_test_photo(img_bytes, filename="apple_fresh_sample.jpg")
    assert ev_healthy["defect_category"] == "Healthy"
    assert ev_healthy["status"] == "FRESH"
    assert ev_healthy["servo_angle"] == "0°"
    assert "PASS" in ev_healthy["servo_action"]

    # 2. Bruise_or_Scratch -> 90° EJECT
    ev_bruise = hub.process_test_photo(img_bytes, filename="apple_bruised_sample.jpg")
    assert ev_bruise["defect_category"] == "Bruise_or_Scratch"
    assert ev_bruise["status"] == "REJECT"
    assert ev_bruise["servo_angle"] == "90°"
    assert "EJECT" in ev_bruise["servo_action"]

    # 3. Scab_or_Canker -> 90° EJECT
    ev_scab = hub.process_test_photo(img_bytes, filename="orange_scab_sample.jpg")
    assert ev_scab["defect_category"] == "Scab_or_Canker"
    assert ev_scab["status"] == "REJECT"
    assert ev_scab["servo_angle"] == "90°"
    assert "EJECT" in ev_scab["servo_action"]

    # 4. Rot_or_Blight -> 90° EJECT
    ev_rot = hub.process_test_photo(img_bytes, filename="guava_rotten_sample.jpg")
    assert ev_rot["defect_category"] == "Rot_or_Blight"
    assert ev_rot["status"] == "REJECT"
    assert ev_rot["servo_angle"] == "90°"
    assert "EJECT" in ev_rot["servo_action"]

    # Verify cumulative defect counters
    assert hub.defect_counts["Healthy"] >= 1
    assert hub.defect_counts["Bruise_or_Scratch"] >= 1
    assert hub.defect_counts["Scab_or_Canker"] >= 1
    assert hub.defect_counts["Rot_or_Blight"] >= 1


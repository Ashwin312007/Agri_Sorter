"""Tests for camera backends and MockCamera."""
import numpy as np
import pytest
from app.hardware.camera import CameraBackend, MockCamera


def test_mock_camera_frame_shape_and_dtype():
    cam = MockCamera(width=640, height=480, color=(10, 20, 30))
    assert not cam.is_started

    with pytest.raises(RuntimeError, match="not started"):
        cam.capture()

    cam.start()
    assert cam.is_started

    frame = cam.capture()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert (frame[0, 0] == [10, 20, 30]).all()
    assert cam.frame_count == 1

    cam.stop()
    assert not cam.is_started
    with pytest.raises(RuntimeError, match="not started"):
        cam.capture()


def test_mock_camera_default_resolution():
    cam = MockCamera()
    cam.start()
    frame = cam.capture()
    assert frame.shape == (720, 1280, 3)
    cam.stop()

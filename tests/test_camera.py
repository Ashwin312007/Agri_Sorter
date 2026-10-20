"""Tests for camera backends, MockCamera, and ThreadedCamera."""
import time
import numpy as np
import pytest
from app.hardware.camera import CameraBackend, MockCamera, RealCamera, ThreadedCamera


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


def test_threaded_camera_lifecycle_and_capture():
    mock_cam = MockCamera(width=640, height=360, color=(100, 150, 200))
    threaded = ThreadedCamera(backend=mock_cam, max_queue_size=1)
    assert not threaded.is_started

    with pytest.raises(RuntimeError, match="not started"):
        threaded.capture()

    threaded.start()
    assert threaded.is_started
    assert mock_cam.is_started

    # Wait briefly for worker thread to capture at least one frame
    time.sleep(0.02)
    frame = threaded.capture(timeout=1.0)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert (frame[0, 0] == [100, 150, 200]).all()

    threaded.stop()
    assert not threaded.is_started
    assert not mock_cam.is_started

    with pytest.raises(RuntimeError, match="not started"):
        threaded.capture()


def test_threaded_camera_drop_oldest_when_full():
    class SequentialMockCamera(CameraBackend):
        def __init__(self):
            self.count = 0
            self.started = False

        def start(self):
            self.started = True

        def capture(self):
            if not self.started:
                raise RuntimeError("not started")
            self.count += 1
            frame = np.full((10, 10, 3), self.count, dtype=np.uint8)
            return frame

        def stop(self):
            self.started = False

    seq_cam = SequentialMockCamera()
    threaded = ThreadedCamera(backend=seq_cam, max_queue_size=1)
    threaded.start()

    # Allow worker to run multiple captures to fill queue and drop oldest
    time.sleep(0.05)
    frame = threaded.capture(timeout=1.0)
    # The value in the frame should be greater than 1 because older frames were dropped
    assert frame[0, 0, 0] > 1

    threaded.stop()


def test_real_camera_threaded_parameter():
    cam = RealCamera(backend="opencv", threaded=True)
    assert cam.threaded is True

    cam_default = RealCamera(backend="opencv")
    assert cam_default.threaded is False


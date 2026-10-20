"""Camera backend interfaces and implementations for Picamera2, OpenCV, and Mocking."""
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union
import numpy as np

__all__ = ["CameraBackend", "RealCamera", "MockCamera", "ThreadedCamera"]


class CameraBackend(ABC):
    """Abstract base class for camera capture backends."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and start video capture stream."""
        pass

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture a single frame as a BGR numpy array.

        Returns:
            np.ndarray: BGR image frame of shape (height, width, 3).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capture stream and release camera hardware."""
        pass


class RealCamera(CameraBackend):
    """Real camera wrapper supporting Raspberry Pi Camera Module (Picamera2) and USB OpenCV."""

    def __init__(
        self,
        backend: str = "auto",
        device_index: int = 0,
        resolution: Union[Sequence[int], Tuple[int, int]] = (1280, 720),
        threaded: bool = False,
    ) -> None:
        """Initialize RealCamera backend.

        Args:
            backend: 'auto', 'picamera2', or 'opencv'.
            device_index: Device index for USB video capture (default 0).
            resolution: Frame resolution (width, height), e.g. (1280, 720).
            threaded: Whether background threaded frame acquisition is requested.
        """
        self.backend = backend.lower()
        self.device_index = device_index
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.threaded = threaded
        self._picam = None
        self._cap = None
        self._is_started = False

    def start(self) -> None:
        """Open camera hardware according to backend preference."""
        if self._is_started:
            return

        import cv2

        if self.backend in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2

                picam = Picamera2()
                config = picam.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                picam.configure(config)
                picam.start()
                self._picam = picam
                self._is_started = True
                return
            except ImportError:
                if self.backend == "picamera2":
                    raise RuntimeError("Picamera2 is not installed. Install with apt or use backend='opencv'.")
            except Exception as exc:
                if self.backend == "picamera2":
                    raise RuntimeError(f"Could not start Picamera2: {exc}") from exc

        try:
            cap = cv2.VideoCapture(self.device_index)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open USB camera on /dev/video{self.device_index}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap = cap
            self._is_started = True
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize camera backend: {exc}") from exc

    def capture(self) -> np.ndarray:
        """Capture single BGR frame.

        Returns:
            np.ndarray: BGR image frame.
        """
        if not self._is_started:
            raise RuntimeError("Camera has not been started. Call start() before capture().")

        import cv2

        if self._picam is not None:
            rgb = self._picam.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from video capture device.")
            return frame

        raise RuntimeError("No active camera backend available.")

    def stop(self) -> None:
        """Release camera hardware."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._is_started = False


class MockCamera(CameraBackend):
    """Mock camera backend generating synthetic solid-color or pattern frames."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Initialize MockCamera.

        Args:
            width: Frame width in pixels (default 1280).
            height: Frame height in pixels (default 720).
            color: BGR tuple for solid color frame (default green).
        """
        self.width = width
        self.height = height
        self.color = color
        self.is_started = False
        self.frame_count = 0

    def start(self) -> None:
        """Start mock camera."""
        self.is_started = True

    def capture(self) -> np.ndarray:
        """Capture synthetic BGR frame.

        Returns:
            np.ndarray: Array of shape (height, width, 3) and dtype uint8.
        """
        if not self.is_started:
            raise RuntimeError("MockCamera is not started. Call start() before capture().")

        self.frame_count += 1
        frame = np.full((self.height, self.width, 3), self.color, dtype=np.uint8)
        return frame

    def stop(self) -> None:
        """Stop mock camera."""
        self.is_started = False


class ThreadedCamera(CameraBackend):
    """Threaded camera wrapper that performs background capture to overlap I/O with inference."""

    def __init__(self, backend: CameraBackend, max_queue_size: int = 1) -> None:
        """Initialize ThreadedCamera wrapper.

        Args:
            backend: CameraBackend instance to wrap.
            max_queue_size: Capacity of queue (default 1 to keep only freshest frame).
        """
        self.backend = backend
        self.max_queue_size = max(1, int(max_queue_size))
        self._queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_started = False

    @property
    def is_started(self) -> bool:
        """Return whether threaded capture is active."""
        return self._is_started

    def _worker(self) -> None:
        """Worker thread function continuously capturing frames."""
        while not self._stop_event.is_set():
            try:
                frame = self.backend.capture()
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(0.005)
                continue

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put(frame, timeout=0.01)
            except queue.Full:
                pass
            time.sleep(0.001)

    def start(self) -> None:
        """Start underlying camera backend and spawn worker thread."""
        if self._is_started:
            return
        self.backend.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._is_started = True

    def capture(self, timeout: float = 1.0) -> np.ndarray:
        """Retrieve the latest captured frame.

        Args:
            timeout: Timeout in seconds waiting for next frame.

        Returns:
            np.ndarray: BGR image frame.

        Raises:
            RuntimeError: If camera is not started or capture timed out.
        """
        if not self._is_started:
            raise RuntimeError("ThreadedCamera is not started.")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("ThreadedCamera capture timed out.")

    def stop(self, timeout: float = 0.5) -> None:
        """Stop background worker thread and backend camera.

        Args:
            timeout: Maximum time to wait for thread join.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        try:
            self.backend.stop()
        except Exception:
            pass
        self._is_started = False


"""Arduino HC-SR04 Ultrasonic USB Serial Bridge.

Communicates with Arduino running ultrasonic_sorter.ino over USB serial.
Runs a background reader thread with thread-safe access to the latest
ultrasonic time-of-flight and distance readings. Falls back to mock mode
if hardware is disconnected or unavailable.
"""

from dataclasses import dataclass
import glob
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    SERIAL_AVAILABLE = False

logger = logging.getLogger("arduino_bridge")


@dataclass
class UltrasonicReading:
    """Represents a single reading from the ultrasonic sensor."""

    tof_us: float
    dist_mm: float
    is_fruit: bool
    timestamp: float
    fruit_diameter_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary representation."""
        return {
            "tof_us": self.tof_us,
            "dist_mm": self.dist_mm,
            "is_fruit": self.is_fruit,
            "timestamp": self.timestamp,
            "fruit_diameter_mm": self.fruit_diameter_mm,
        }


def discover_serial_ports() -> List[str]:
    """Auto-discover potential Arduino Uno / microcontroller USB serial ports.

    Checks /dev/ttyACM*, /dev/ttyUSB* on Linux/macOS and COM* on Windows.
    Matches Arduino Uno (VID 0x2341, PID 0x0043 / 0x0001), CH340, FTDI, CP210x, ESP32, etc.

    Returns:
        List[str]: List of matching serial port paths.
    """
    ports: List[str] = []

    # 1. Pyserial list_ports check
    if SERIAL_AVAILABLE:
        try:
            for p in serial.tools.list_ports.comports():
                dev = p.device
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                vid = getattr(p, "vid", None)
                pid = getattr(p, "pid", None)

                # Arduino Uno: VID 0x2341, PID 0x0043 (Uno R3) / 0x0001 (Uno R1/R2)
                is_arduino = (
                    vid == 0x2341
                    or ("2341:0043" in hwid)
                    or ("2341:0001" in hwid)
                    or ("2341" in hwid and ("0043" in hwid or "0001" in hwid))
                    or (vid is not None and vid == 0x2341 and (pid in (0x0043, 0x0001) if pid is not None else True))
                    or "arduino" in desc
                    or "uno" in desc
                )

                # ESP32-C6 USB Serial/JTAG: VID 0x303a, PID 0x1001
                is_esp32_c6 = (
                    (vid == 0x303A and pid == 0x1001)
                    or ("303a:1001" in hwid)
                    or ("303a" in hwid and "1001" in hwid)
                )

                keywords = ("arduino", "uno", "ch340", "ftdi", "cp210", "esp32", "espressif", "jtag")
                matches_keyword = any(k in desc or k in hwid for k in keywords)

                if (
                    is_arduino
                    or is_esp32_c6
                    or matches_keyword
                    or "acm" in dev.lower()
                    or "usb" in dev.lower()
                    or dev.upper().startswith("COM")
                ):
                    if dev not in ports:
                        ports.append(dev)
        except Exception as exc:
            logger.debug("pyserial port scan error: %s", exc)

    # 2. Filesystem glob fallback for POSIX systems
    if sys.platform != "win32":
        acm_ports = sorted(glob.glob("/dev/ttyACM*"))
        usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
        for p in acm_ports + usb_ports:
            if p not in ports:
                ports.append(p)

    return ports


class ArduinoBridge:
    """Thread-safe USB serial bridge for Arduino / ESP32-C6 HC-SR04 ultrasonic sensor."""

    def __init__(
        self,
        port: str = "auto",
        baud_rate: int = 115200,
        timeout_s: float = 1.0,
        mock: bool = False,
        config: Optional[Dict[str, Any]] = None,
        baseline_distance_mm: float = 240.0,
        fruit_threshold_mm: float = 30.0,
    ) -> None:
        """Initialize Arduino / ESP32 serial bridge.

        Args:
            port: Serial port path (e.g. '/dev/ttyACM0', 'COM3') or 'auto'.
            baud_rate: Serial baud rate (default: 115200).
            timeout_s: Serial communication read timeout in seconds.
            mock: If True, operates in simulated mock mode without hardware.
            config: Optional configuration dict containing [esp32] or [arduino] settings.
            baseline_distance_mm: Belt baseline distance in mm.
            fruit_threshold_mm: Height delta in mm indicating fruit presence.
        """
        # Load config parameters if provided (checks [arduino] first, then falls back to [esp32])
        cfg = {}
        if config:
            if "arduino" in config:
                cfg = config["arduino"]
            elif "esp32" in config:
                cfg = config["esp32"]
        if cfg:
            port = cfg.get("port", port)
            baud_rate = cfg.get("baud_rate", baud_rate)
            timeout_s = cfg.get("timeout_s", timeout_s)
            baseline_distance_mm = cfg.get("baseline_distance_mm", baseline_distance_mm)
            fruit_threshold_mm = cfg.get("fruit_threshold_mm", fruit_threshold_mm)

        self.configured_port = port
        self.baud_rate = int(baud_rate)
        self.timeout_s = float(timeout_s)
        self.baseline_distance_mm = float(baseline_distance_mm)
        self.fruit_threshold_mm = float(fruit_threshold_mm)
        self.explicit_mock = bool(mock)

        self.port: Optional[str] = None
        self._serial: Optional[Any] = None
        self.is_mock: bool = self.explicit_mock

        self._lock = threading.Lock()
        self._latest_reading: Optional[UltrasonicReading] = None
        self._reading_queue: queue.Queue = queue.Queue(maxsize=128)

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Simulated fruit generator state for mock mode
        self._mock_fruit_active = False
        self._mock_custom_override: Optional[UltrasonicReading] = None

    def start(self) -> "ArduinoBridge":
        """Connect to hardware serial or fallback to mock mode, and start reader thread."""
        if self._running:
            return self

        if not self.explicit_mock:
            self._connect()
        else:
            self.is_mock = True
            logger.info("ArduinoBridge initialized in explicit mock mode.")

        self._running = True
        target = self._mock_worker if self.is_mock else self._serial_worker
        self._thread = threading.Thread(target=target, name="ArduinoBridgeWorker", daemon=True)
        self._thread.start()
        return self

    def _connect(self) -> None:
        """Attempt connection to Arduino serial port."""
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial is not installed; falling back to mock mode.")
            self.is_mock = True
            return

        candidates = [self.configured_port] if self.configured_port != "auto" else discover_serial_ports()

        if not candidates:
            logger.warning("No Arduino serial ports discovered; falling back to mock mode.")
            self.is_mock = True
            return

        for p in candidates:
            try:
                logger.info("Attempting connection to Arduino on %s (%d baud)...", p, self.baud_rate)
                ser = serial.Serial(
                    port=p,
                    baudrate=self.baud_rate,
                    timeout=self.timeout_s,
                    write_timeout=self.timeout_s,
                )
                self.port = p
                self._serial = ser
                self.is_mock = False
                logger.info("Successfully connected to Arduino on %s", p)
                return
            except Exception as exc:
                logger.debug("Failed opening port %s: %s", p, exc)

        logger.warning(
            "Could not connect to any Arduino serial port (%s); falling back to mock mode.",
            candidates,
        )
        self.is_mock = True

    def stop(self) -> None:
        """Stop background worker thread and close serial port."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:
                logger.debug("Error closing serial port: %s", exc)
            finally:
                self._serial = None

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "ArduinoBridge":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def parse_line(self, line: str, timestamp: Optional[float] = None) -> Optional[UltrasonicReading]:
        """Parse a JSON string from Arduino into an UltrasonicReading.

        Args:
            line: Raw line string from serial.
            timestamp: Optional timestamp override.

        Returns:
            Optional[UltrasonicReading]: Parsed reading or None if line is invalid or a non-reading payload.
        """
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        # Check for expected measurement fields: {"tof_us": ..., "dist_mm": ..., "fruit": ...}
        if "tof_us" in data and "dist_mm" in data:
            tof = float(data["tof_us"])
            dist = float(data["dist_mm"])
            fruit = bool(data.get("fruit", 0))
            ts = timestamp if timestamp is not None else time.time()

            if "fruit_diameter_mm" in data and data["fruit_diameter_mm"] is not None:
                fruit_diameter = float(data["fruit_diameter_mm"])
            elif fruit:
                fruit_diameter = max(0.0, float(self.baseline_distance_mm - dist))
            else:
                fruit_diameter = 0.0

            return UltrasonicReading(
                tof_us=tof,
                dist_mm=dist,
                is_fruit=fruit,
                timestamp=ts,
                fruit_diameter_mm=fruit_diameter,
            )

        return None

    def get_latest_reading(self) -> Optional[UltrasonicReading]:
        """Retrieve the most recent UltrasonicReading in a thread-safe manner.

        Returns:
            Optional[UltrasonicReading]: Most recent reading, or None if no reading available yet.
        """
        with self._lock:
            return self._latest_reading

    def set_mock_reading(self, reading: Optional[UltrasonicReading]) -> None:
        """Manually inject a mock reading for testing."""
        with self._lock:
            self._mock_custom_override = reading
            if reading is not None:
                self._latest_reading = reading
                try:
                    self._reading_queue.put_nowait(reading)
                except queue.Full:
                    pass

    def send_command(self, cmd: str) -> Optional[str]:
        """Send a serial command and read single-line response.

        Args:
            cmd: Command string (e.g. 'PING', 'CALIB').

        Returns:
            Optional[str]: Response line, or None if failed/mocked.
        """
        if self.is_mock or self._serial is None:
            if cmd.strip().upper() == "PING":
                return '{"res":"PONG"}'
            if cmd.strip().upper() == "CALIB":
                return f'{{"res":"CALIB_OK","baseline_mm":{int(self.baseline_distance_mm)}}}'
            return '{"err":"MOCK_UNKNOWN"}'

        with self._lock:
            try:
                line_to_send = (cmd.strip() + "\n").encode("utf-8")
                self._serial.write(line_to_send)
                self._serial.flush()
                resp = self._serial.readline().decode("utf-8", errors="replace").strip()
                return resp
            except Exception as exc:
                logger.warning("Error communicating command '%s': %s", cmd, exc)
                return None

    def ping(self) -> bool:
        """Send PING command to Arduino and verify PONG response."""
        resp = self.send_command("PING")
        if not resp:
            return False
        try:
            data = json.loads(resp)
            return data.get("res") == "PONG"
        except Exception:
            return False

    def calibrate_baseline(self) -> Optional[int]:
        """Send CALIB command to Arduino and return updated baseline distance."""
        resp = self.send_command("CALIB")
        if not resp:
            return None
        try:
            data = json.loads(resp)
            if data.get("res") == "CALIB_OK" and "baseline_mm" in data:
                new_baseline = int(data["baseline_mm"])
                self.baseline_distance_mm = float(new_baseline)
                return new_baseline
        except Exception:
            pass
        return None

    def _serial_worker(self) -> None:
        """Background thread reading serial lines from hardware."""
        logger.info("Arduino serial worker thread started.")
        while self._running:
            if self._serial is None:
                time.sleep(0.1)
                continue

            try:
                raw_bytes = self._serial.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="replace").strip()
                reading = self.parse_line(line)
                if reading is not None:
                    with self._lock:
                        self._latest_reading = reading
                        try:
                            self._reading_queue.put_nowait(reading)
                        except queue.Full:
                            # Discard oldest to keep fresh
                            try:
                                self._reading_queue.get_nowait()
                                self._reading_queue.put_nowait(reading)
                            except Exception:
                                pass
            except serial.SerialException as exc:
                logger.warning("Serial error: %s. Switching to mock mode.", exc)
                self.is_mock = True
                self._mock_worker()
                break
            except Exception as exc:
                logger.debug("Unexpected error in serial worker: %s", exc)
                time.sleep(0.01)

    def _mock_worker(self) -> None:
        """Background thread synthesizing mock readings at ~40Hz."""
        logger.info("Arduino mock reading worker thread started (40Hz).")
        cycle = 0
        while self._running:
            now = time.time()

            with self._lock:
                override = self._mock_custom_override

            if override is not None:
                diam = (
                    override.fruit_diameter_mm
                    if override.fruit_diameter_mm is not None
                    else (68.0 if override.is_fruit else 0.0)
                )
                reading = UltrasonicReading(
                    tof_us=override.tof_us,
                    dist_mm=override.dist_mm,
                    is_fruit=override.is_fruit,
                    timestamp=now,
                    fruit_diameter_mm=diam,
                )
            else:
                cycle = (cycle + 1) % 160  # ~4 second full repeating cycle at 40Hz
                # Simulate fruit passing during cycle 40..80
                is_fruit = (40 <= cycle <= 80)
                if is_fruit:
                    # Apex distance ~ 72 mm, ToF ~ 4210 us, realistic diameter ~ 68 mm (65-75 mm range)
                    dist_mm = 72.0
                    tof_us = 4210.0
                    fruit_diameter_mm = 68.0
                else:
                    # Empty belt baseline ~ 240 mm, ToF ~ 14000 us, 0.0 mm diameter
                    dist_mm = self.baseline_distance_mm
                    tof_us = (self.baseline_distance_mm * 2000.0) / 343.0
                    fruit_diameter_mm = 0.0

                reading = UltrasonicReading(
                    tof_us=tof_us,
                    dist_mm=dist_mm,
                    is_fruit=is_fruit,
                    timestamp=now,
                    fruit_diameter_mm=fruit_diameter_mm,
                )

            with self._lock:
                self._latest_reading = reading
                try:
                    self._reading_queue.put_nowait(reading)
                except queue.Full:
                    try:
                        self._reading_queue.get_nowait()
                        self._reading_queue.put_nowait(reading)
                    except Exception:
                        pass

            time.sleep(0.025)  # 25 ms interval (40Hz)

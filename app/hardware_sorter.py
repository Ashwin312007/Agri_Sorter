"""Hardware actuation and ultrasonic sensing subsystem for Agri_Sorter.

Implements Requirement 6 & 7:
1. UltrasonicSensor: HC-SR04 distance sensing with median-of-3 filtering,
   30ms timeout, debounce logic, and refractory lockout (250ms).
2. ServoActuator: Sorting diverter flap control via gpiozero.AngularServo
   (or mock backend under simulation / non-Pi OS), non-blocking actuation (0° rest, 90° reject).
3. HardwareCoordinatorThread: Dedicated background daemon reading from
   queue.Queue[FruitEvent], actuating servo on reject, passing on fresh, and
   triggering fail-safe reject on queue underrun with event logging to CSV.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from app.contracts import (
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    load_app_config,
)

logger = logging.getLogger("hardware_sorter")

# Try importing gpiozero; if unavailable (e.g. running on x86_64 host), fallback to mock
try:
    import gpiozero
    from gpiozero import AngularServo, DigitalInputDevice, DigitalOutputDevice
    HAS_GPIOZERO = True
except (ImportError, Exception):
    gpiozero = None
    AngularServo = None
    DigitalInputDevice = None
    DigitalOutputDevice = None
    HAS_GPIOZERO = False


# ============================================================================
# HC-SR04 Hardware & Mock Backends
# ============================================================================

class _IUltrasonicBackend:
    """Interface for HC-SR04 ultrasonic sensor backends."""

    def measure_once(self) -> Optional[float]:
        """Perform a single pulse measurement and return distance in cm, or None if timed out."""
        raise NotImplementedError

    def close(self) -> None:
        """Release underlying hardware resources."""
        pass


class RealHCSR04Backend(_IUltrasonicBackend):
    """Direct GPIO bitbanging backend for HC-SR04 on Raspberry Pi."""

    def __init__(self, trigger_pin: int, echo_pin: int, echo_timeout_s: float = 0.030):
        if not HAS_GPIOZERO or DigitalOutputDevice is None or DigitalInputDevice is None:
            raise RuntimeError("gpiozero is required for RealHCSR04Backend")
        self.trigger = DigitalOutputDevice(trigger_pin, initial_value=False)
        self.echo = DigitalInputDevice(echo_pin)
        self.echo_timeout_s = echo_timeout_s

    def measure_once(self) -> Optional[float]:
        # Send 10 microsecond HIGH pulse to Trigger
        self.trigger.off()
        time.sleep(0.000002)
        self.trigger.on()
        time.sleep(0.000010)
        self.trigger.off()

        # Wait for Echo to go HIGH (start of pulse)
        t_start = time.time()
        timeout = t_start + self.echo_timeout_s
        while not self.echo.is_active:
            if time.time() > timeout:
                return None

        pulse_start = time.time()

        # Wait for Echo to go LOW (end of pulse)
        timeout = pulse_start + self.echo_timeout_s
        while self.echo.is_active:
            if time.time() > timeout:
                return None

        pulse_end = time.time()
        pulse_duration = pulse_end - pulse_start
        # Speed of sound in air ≈ 343 m/s = 34,300 cm/s (round trip / 2)
        distance_cm = (pulse_duration * 34300.0) / 2.0
        return distance_cm

    def close(self) -> None:
        try:
            self.trigger.close()
            self.echo.close()
        except Exception:
            pass


class MockHCSR04Backend(_IUltrasonicBackend):
    """Thread-safe mock backend for HC-SR04 simulation and testing."""

    def __init__(self, default_distance_cm: float = 50.0):
        self._distance_cm: Optional[float] = default_distance_cm
        self._lock = threading.Lock()

    def set_distance(self, distance_cm: Optional[float]) -> None:
        with self._lock:
            self._distance_cm = distance_cm

    def measure_once(self) -> Optional[float]:
        with self._lock:
            return self._distance_cm

    def close(self) -> None:
        pass


# ============================================================================
# UltrasonicSensor
# ============================================================================

class UltrasonicSensor:
    """HC-SR04 ultrasonic distance sensor with median-of-3 filtering and debounce."""

    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        trigger_distance_cm: float = 10.0,
        echo_timeout_ms: float = 30.0,
        debounce_samples: int = 3,
        refractory_lockout_s: float = 0.25,
        simulate: bool = False,
        on_trigger: Optional[Callable[[], None]] = None,
    ):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.trigger_distance_cm = float(trigger_distance_cm)
        self.echo_timeout_s = float(echo_timeout_ms) / 1000.0
        self.debounce_samples = max(1, int(debounce_samples))
        self.refractory_lockout_s = float(refractory_lockout_s)
        self.simulate = simulate
        self.on_trigger = on_trigger

        self._lock = threading.Lock()
        self._last_trigger_time: float = 0.0
        self._is_fruit_present: bool = False

        # Select real or mock backend
        self._backend: _IUltrasonicBackend
        if simulate or not HAS_GPIOZERO:
            self._backend = MockHCSR04Backend()
        else:
            try:
                self._backend = RealHCSR04Backend(
                    trigger_pin=trigger_pin,
                    echo_pin=echo_pin,
                    echo_timeout_s=self.echo_timeout_s,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize RealHCSR04Backend ({e}). Falling back to MockHCSR04Backend."
                )
                self._backend = MockHCSR04Backend()
                self.simulate = True

    def set_simulated_distance(self, distance_cm: Optional[float]) -> None:
        """Inject simulated distance for testing / simulate mode."""
        if isinstance(self._backend, MockHCSR04Backend):
            self._backend.set_distance(distance_cm)

    def measure_raw_distance_cm(self) -> Optional[float]:
        """Perform a single raw measurement. Returns distance in cm or None if timed out."""
        return self._backend.measure_once()

    def get_filtered_distance_cm(self) -> Optional[float]:
        """Collect median-of-N distance samples (default 3) discarding timeouts."""
        samples: List[float] = []
        for _ in range(self.debounce_samples):
            reading = self.measure_raw_distance_cm()
            if reading is not None and reading > 0.0:
                samples.append(reading)
            if self.debounce_samples > 1 and not self.simulate:
                time.sleep(0.002)  # Tiny pause between ultrasonic bursts

        if not samples:
            return None
        return float(statistics.median(samples))

    def check_trigger(self, current_time: Optional[float] = None) -> bool:
        """Evaluate distance reading against trigger threshold and refractory lockout.

        Returns True if a new fruit arrival trigger event occurred, False otherwise.
        """
        now = time.time() if current_time is None else current_time
        dist = self.get_filtered_distance_cm()

        with self._lock:
            # Check if distance drops below threshold (fruit present)
            if dist is not None and dist <= self.trigger_distance_cm:
                time_since_last = now - self._last_trigger_time
                if not self._is_fruit_present:
                    if time_since_last >= self.refractory_lockout_s:
                        self._is_fruit_present = True
                        self._last_trigger_time = now
                        logger.debug(
                            f"Ultrasonic trigger: dist={dist:.2f}cm <= {self.trigger_distance_cm:.2f}cm "
                            f"(lockout delta={time_since_last:.3f}s)"
                        )
                        if self.on_trigger is not None:
                            try:
                                self.on_trigger()
                            except Exception as e:
                                logger.error(f"Error in on_trigger callback: {e}", exc_info=True)
                        return True
                    else:
                        # Suppressed by refractory lockout; do not latch fruit present
                        return False
                # Fruit is already latched present; debounce prevents re-triggering
                return False
            else:
                # Distance is above threshold; fruit has cleared sensor zone
                self._is_fruit_present = False
                return False

    def close(self) -> None:
        """Release backend hardware resources."""
        self._backend.close()


# ============================================================================
# ServoActuator Hardware & Mock Backends
# ============================================================================

class _IServoBackend:
    """Interface for Servo actuator backends."""

    def set_angle(self, angle: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RealServoBackend(_IServoBackend):
    """Hardware PWM / GPIO backend using gpiozero.AngularServo on Raspberry Pi."""

    def __init__(
        self,
        pin: int,
        min_angle: float,
        max_angle: float,
        min_pulse_width: float,
        max_pulse_width: float,
    ):
        if not HAS_GPIOZERO or AngularServo is None:
            raise RuntimeError("gpiozero is required for RealServoBackend")
        self.servo = AngularServo(
            pin=pin,
            min_angle=min_angle,
            max_angle=max_angle,
            min_pulse_width=min_pulse_width,
            max_pulse_width=max_pulse_width,
        )

    def set_angle(self, angle: float) -> None:
        self.servo.angle = angle

    def close(self) -> None:
        try:
            self.servo.close()
        except Exception:
            pass


class MockServoBackend(_IServoBackend):
    """Thread-safe mock backend for Servo actuation."""

    def __init__(self, min_angle: float = 0.0, max_angle: float = 90.0):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle: float = min_angle
        self._lock = threading.Lock()

    def set_angle(self, angle: float) -> None:
        with self._lock:
            self.current_angle = max(self.min_angle, min(self.max_angle, float(angle)))

    def close(self) -> None:
        pass


# ============================================================================
# ServoActuator
# ============================================================================

class ServoActuator:
    """Sorting diverter flap servo actuator with non-blocking stroke execution."""

    def __init__(
        self,
        pin: int = 18,
        rest_angle: float = 0.0,
        reject_angle: float = 90.0,
        pulse_hold_time_s: float = 0.5,
        min_pulse_width: float = 0.0005,
        max_pulse_width: float = 0.0025,
        simulate: bool = False,
    ):
        self.pin = pin
        self.rest_angle = float(rest_angle)
        self.reject_angle = float(reject_angle)
        self.pulse_hold_time_s = float(pulse_hold_time_s)
        self.simulate = simulate

        self._lock = threading.Lock()
        self._angle: float = self.rest_angle
        self._is_rejecting: bool = False
        self._rejection_count: int = 0
        self._reset_timer: Optional[threading.Timer] = None

        self._backend: _IServoBackend
        if simulate or not HAS_GPIOZERO:
            self._backend = MockServoBackend(
                min_angle=min(self.rest_angle, self.reject_angle),
                max_angle=max(self.rest_angle, self.reject_angle),
            )
        else:
            try:
                self._backend = RealServoBackend(
                    pin=pin,
                    min_angle=min(self.rest_angle, self.reject_angle),
                    max_angle=max(self.rest_angle, self.reject_angle),
                    min_pulse_width=min_pulse_width,
                    max_pulse_width=max_pulse_width,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize RealServoBackend ({e}). Falling back to MockServoBackend."
                )
                self._backend = MockServoBackend(
                    min_angle=min(self.rest_angle, self.reject_angle),
                    max_angle=max(self.rest_angle, self.reject_angle),
                )
                self.simulate = True

        # Initialize servo at rest position
        self.rest()

    @property
    def angle(self) -> float:
        """Current flap angle in degrees."""
        with self._lock:
            return self._angle

    @property
    def is_rejecting(self) -> bool:
        """True if diverter flap is currently in reject position."""
        with self._lock:
            return self._is_rejecting

    @property
    def rejection_count(self) -> int:
        """Total count of reject strokes initiated."""
        with self._lock:
            return self._rejection_count

    def _set_angle_internal(self, angle: float) -> None:
        self._angle = angle
        try:
            self._backend.set_angle(angle)
        except Exception as e:
            logger.error(f"Error communicating angle {angle}° to servo backend: {e}")

    def set_angle(self, angle: float) -> None:
        """Explicitly set servo angle (degrees)."""
        with self._lock:
            self._set_angle_internal(angle)

    def rest(self) -> None:
        """Move flap to rest position (0°) and cancel pending reset timers."""
        with self._lock:
            if self._reset_timer is not None:
                self._reset_timer.cancel()
                self._reset_timer = None
            self._set_angle_internal(self.rest_angle)
            self._is_rejecting = False

    def reject(self) -> None:
        """Actuate flap to reject position (90°) asynchronously (non-blocking).

        Schedules an automatic reset to rest position after `pulse_hold_time_s`.
        If called while already rejecting, the hold window is extended.
        """
        with self._lock:
            # Cancel any pending auto-reset timer
            if self._reset_timer is not None:
                self._reset_timer.cancel()
                self._reset_timer = None

            self._set_angle_internal(self.reject_angle)
            self._is_rejecting = True
            self._rejection_count += 1
            logger.info(
                f"Servo actuated -> REJECT ({self.reject_angle}°) [Stroke #{self._rejection_count}]"
            )

            # Schedule return to rest position asynchronously
            self._reset_timer = threading.Timer(self.pulse_hold_time_s, self._auto_reset)
            self._reset_timer.daemon = True
            self._reset_timer.start()

    def _auto_reset(self) -> None:
        """Timer callback to restore rest position after hold time."""
        with self._lock:
            self._set_angle_internal(self.rest_angle)
            self._is_rejecting = False
            self._reset_timer = None
            logger.debug(f"Servo returned -> REST ({self.rest_angle}°)")

    def close(self) -> None:
        """Release servo resources and cancel timers."""
        with self._lock:
            if self._reset_timer is not None:
                self._reset_timer.cancel()
                self._reset_timer = None
            self._backend.close()


# ============================================================================
# HardwareCoordinatorThread
# ============================================================================

class HardwareCoordinatorThread(threading.Thread):
    """Background thread coordinating HC-SR04 triggers with FIFO fruit queue and servo flap.

    1. Pops oldest FruitEvent when HC-SR04 triggers.
    2. Activates fail-safe reject if queue underrun occurs (untracked fruit).
    3. Actuates diverter flap for REJECT fruits, passes FRESH fruits.
    4. Appends audit records to CSV event log.
    """

    def __init__(
        self,
        fruit_queue: queue.Queue[FruitEvent],
        ultrasonic: UltrasonicSensor,
        servo: ServoActuator,
        event_log_path: Union[str, Path] = "runtime/fruit_events.csv",
        reject_on_empty_queue: bool = True,
        poll_interval_s: float = 0.005,
    ):
        super().__init__(name="HardwareCoordinatorThread", daemon=True)
        self.fruit_queue = fruit_queue
        self.ultrasonic = ultrasonic
        self.servo = servo
        self.event_log_path = Path(event_log_path)
        self.reject_on_empty_queue = bool(reject_on_empty_queue)
        self.poll_interval_s = float(poll_interval_s)

        self._stop_event = threading.Event()
        self._trigger_lock = threading.Lock()
        self._csv_lock = threading.Lock()

        # Telemetry / counters
        self.total_triggers: int = 0
        self.fresh_passed_count: int = 0
        self.defect_rejected_count: int = 0
        self.underrun_rejected_count: int = 0

        self._init_csv_log()

        # Wire trigger callback from ultrasonic sensor to handle_trigger
        if self.ultrasonic.on_trigger is None:
            self.ultrasonic.on_trigger = self.handle_trigger

    def _init_csv_log(self) -> None:
        """Ensure parent directory exists and CSV header is written."""
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_lock:
            if not self.event_log_path.is_file() or self.event_log_path.stat().st_size == 0:
                with open(self.event_log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "track_id",
                        "fruit_type",
                        "final_status",
                        "action",
                        "defect_classes",
                        "confidence",
                        "egress_timestamp",
                    ])
                    f.flush()

    def _log_event(
        self,
        event: Optional[FruitEvent],
        action: str,
        is_underrun: bool = False,
    ) -> None:
        """Thread-safe write of fruit sorting decision to CSV."""
        now_ts = time.time()
        with self._csv_lock:
            try:
                with open(self.event_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if is_underrun or event is None:
                        writer.writerow([
                            f"{now_ts:.4f}",
                            "-1",
                            "untracked",
                            "UNDERRUN_REJECT",
                            action,
                            "untracked_fruit",
                            "0.0000",
                            f"{now_ts:.4f}",
                        ])
                    else:
                        defects_str = (
                            ",".join(event.defect_classes)
                            if event.defect_classes
                            else "none"
                        )
                        writer.writerow([
                            f"{now_ts:.4f}",
                            str(event.track_id),
                            str(event.fruit_type),
                            str(event.final_status),
                            action,
                            defects_str,
                            f"{event.confidence:.4f}",
                            f"{event.egress_timestamp:.4f}",
                        ])
                    f.flush()
            except Exception as e:
                logger.error(f"Failed writing event to {self.event_log_path}: {e}")

    def handle_trigger(self) -> Optional[FruitEvent]:
        """Process fruit arrival at diverter flap when ultrasonic sensor triggers.

        Can be called by ultrasonic polling loop or triggered programmatically in simulation.
        """
        with self._trigger_lock:
            self.total_triggers += 1

            # 1. Pop oldest FruitEvent from FIFO queue
            try:
                event = self.fruit_queue.get_nowait()
            except queue.Empty:
                event = None

            # 2. Handle Queue Underrun (Untracked fruit fail-safe)
            if event is None:
                logger.warning(
                    "QUEUE_UNDERRUN_REJECT: Untracked fruit detected at sorter flap. Fail-safe REJECT activated!"
                )
                self.underrun_rejected_count += 1
                if self.reject_on_empty_queue:
                    self.servo.reject()
                self._log_event(None, action="REJECT", is_underrun=True)
                return None

            # 3. Handle popped FruitEvent
            if event.final_status == STATUS_REJECT:
                logger.info(
                    f"Fruit #{event.track_id} ({event.fruit_type}) final_status={event.final_status} "
                    f"defects={event.defect_classes} -> Diverter Flap Actuated (REJECT 90°)"
                )
                self.defect_rejected_count += 1
                self.servo.reject()
                self._log_event(event, action="REJECT", is_underrun=False)
            else:
                logger.info(
                    f"Fruit #{event.track_id} ({event.fruit_type}) final_status={event.final_status} "
                    f"-> Flap at Rest (PASS 0°)"
                )
                self.fresh_passed_count += 1
                self.servo.rest()
                self._log_event(event, action="PASS", is_underrun=False)

            return event

    def run(self) -> None:
        """Continuous background ultrasonic polling loop."""
        logger.info("HardwareCoordinatorThread started.")
        while not self._stop_event.is_set():
            try:
                self.ultrasonic.check_trigger()
            except Exception as e:
                logger.error(f"Error during ultrasonic poll loop: {e}", exc_info=True)
            time.sleep(self.poll_interval_s)
        logger.info("HardwareCoordinatorThread exiting.")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal thread shutdown and release hardware resources."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        self.ultrasonic.close()
        self.servo.close()


# ============================================================================
# Factory Initializer
# ============================================================================

def create_hardware_coordinator_from_config(
    cfg: Optional[Dict[str, Any]] = None,
    fruit_queue: Optional[queue.Queue[FruitEvent]] = None,
    simulate: bool = False,
    event_log_path: Optional[Union[str, Path]] = None,
) -> HardwareCoordinatorThread:
    """Build and return a HardwareCoordinatorThread from unified config.yaml."""
    if cfg is None:
        cfg = load_app_config()

    if fruit_queue is None:
        fruit_queue = queue.Queue()

    hw_cfg = cfg.get("hardware", {})
    us_cfg = hw_cfg.get("ultrasonic", {})
    sv_cfg = hw_cfg.get("servo", {})
    fs_cfg = hw_cfg.get("fail_safe", {})
    log_cfg = cfg.get("logging", {})

    log_path = event_log_path or log_cfg.get("event_log_path", "runtime/fruit_events.csv")

    ultrasonic = UltrasonicSensor(
        trigger_pin=int(us_cfg.get("trigger_pin", 23)),
        echo_pin=int(us_cfg.get("echo_pin", 24)),
        trigger_distance_cm=float(us_cfg.get("trigger_distance_cm", 10.0)),
        echo_timeout_ms=float(us_cfg.get("echo_timeout_ms", 30.0)),
        debounce_samples=int(us_cfg.get("debounce_samples", 3)),
        refractory_lockout_s=float(us_cfg.get("refractory_lockout_s", 0.25)),
        simulate=simulate,
    )

    servo = ServoActuator(
        pin=int(sv_cfg.get("pin", 18)),
        rest_angle=float(sv_cfg.get("rest_angle", 0.0)),
        reject_angle=float(sv_cfg.get("reject_angle", 90.0)),
        pulse_hold_time_s=float(sv_cfg.get("pulse_hold_time_s", 0.5)),
        min_pulse_width=float(sv_cfg.get("min_pulse_width", 0.0005)),
        max_pulse_width=float(sv_cfg.get("max_pulse_width", 0.0025)),
        simulate=simulate,
    )

    coordinator = HardwareCoordinatorThread(
        fruit_queue=fruit_queue,
        ultrasonic=ultrasonic,
        servo=servo,
        event_log_path=log_path,
        reject_on_empty_queue=bool(fs_cfg.get("reject_on_empty_queue", True)),
    )
    return coordinator

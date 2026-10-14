"""Transmitter driver interfaces and implementations for 40 kHz tone burst generation."""
from abc import ABC, abstractmethod
import time
from typing import Optional

try:
    import gpiod
    _HAS_GPIOD = True
except ImportError:
    _HAS_GPIOD = False


class BurstTransmitter(ABC):
    """Abstract base class for ultrasonic burst transmitters."""

    @abstractmethod
    def fire(self) -> None:
        """Emit an N-cycle burst at the configured ultrasonic frequency."""
        pass

    def close(self) -> None:
        """Release transmitter hardware resources."""
        pass


class RealTransmitter(BurstTransmitter):
    """Real GPIO transmitter using gpiod v2 / v1 on Raspberry Pi 5 (RP1 south chip)."""

    def __init__(
        self,
        gpio_chip: str = "/dev/gpiochip4",
        gpio_line: int = 18,
        frequency_hz: int = 40000,
        burst_cycles: int = 8,
    ) -> None:
        """Initialize RealTransmitter.

        Args:
            gpio_chip: Path to GPIO chip device (Pi 5 default is /dev/gpiochip4 for RP1).
            gpio_line: GPIO line / BCM pin number (e.g. 18).
            frequency_hz: Burst frequency in Hertz (default 40000 Hz).
            burst_cycles: Number of square wave cycles in tone burst (default 8).

        Raises:
            RuntimeError: If gpiod library is not available or GPIO line request fails.
        """
        if not _HAS_GPIOD:
            raise RuntimeError(
                "gpiod is not installed or GPIO hardware is unavailable. "
                "Install with: sudo apt install python3-gpiod or pip install gpiod"
            )

        self.gpio_chip = gpio_chip
        self.gpio_line = gpio_line
        self.frequency_hz = frequency_hz
        self.burst_cycles = burst_cycles

        # Calculate half-period in nanoseconds (for 40 kHz -> 12,500 ns)
        self.half_period_ns = int((1.0 / (2.0 * float(frequency_hz))) * 1e9)

        self._request = None
        self._chip_v1 = None
        self._line_v1 = None
        self._api_version = 2

        self._init_gpio()

    def _init_gpio(self) -> None:
        """Initialize GPIO output line with fallback between gpiod v2 and v1."""
        try:
            # Try gpiod v2 API first (Bookworm default)
            if hasattr(gpiod, "request_lines") and hasattr(gpiod, "line"):
                from gpiod.line import Direction, Value  # type: ignore

                self._request = gpiod.request_lines(
                    self.gpio_chip,
                    consumer="agri_sorter_tx",
                    config={
                        self.gpio_line: gpiod.LineSettings(
                            direction=Direction.OUTPUT,
                            output_value=Value.INACTIVE,
                        )
                    },
                )
                self._api_version = 2
                return
        except Exception:
            pass

        try:
            # Fallback to gpiod v1 API
            self._chip_v1 = gpiod.Chip(self.gpio_chip)
            self._line_v1 = self._chip_v1.get_line(self.gpio_line)
            self._line_v1.request(consumer="agri_sorter_tx", type=gpiod.LINE_REQ_DIR_OUT)
            self._line_v1.set_value(0)
            self._api_version = 1
        except Exception as exc:
            raise RuntimeError(
                f"Failed to request GPIO line {self.gpio_line} on chip {self.gpio_chip}: {exc}"
            ) from exc

    def _set_level(self, high: bool) -> None:
        """Set output line level."""
        if self._api_version == 2 and self._request is not None:
            from gpiod.line import Value  # type: ignore

            self._request.set_value(self.gpio_line, Value.ACTIVE if high else Value.INACTIVE)
        elif self._api_version == 1 and self._line_v1 is not None:
            self._line_v1.set_value(1 if high else 0)

    def fire(self) -> None:
        """Emit tone burst using tight nanosecond spin loop."""
        half_ns = self.half_period_ns
        for _ in range(self.burst_cycles):
            # Phase HIGH
            self._set_level(True)
            target = time.perf_counter_ns() + half_ns
            while time.perf_counter_ns() < target:
                pass

            # Phase LOW
            self._set_level(False)
            target = time.perf_counter_ns() + half_ns
            while time.perf_counter_ns() < target:
                pass

        # Guarantee pin is driven low after burst
        self._set_level(False)

    def close(self) -> None:
        """Release GPIO lines."""
        if self._request is not None:
            try:
                self._request.release()
            except Exception:
                pass
            self._request = None

        if self._line_v1 is not None:
            try:
                self._line_v1.release()
            except Exception:
                pass
            self._line_v1 = None

        if self._chip_v1 is not None:
            try:
                self._chip_v1.close()
            except Exception:
                pass
            self._chip_v1 = None


class MockTransmitter(BurstTransmitter):
    """Mock ultrasonic transmitter for offline testing and benchmarking."""

    def __init__(
        self,
        gpio_chip: str = "/dev/gpiochip4",
        gpio_line: int = 18,
        frequency_hz: int = 40000,
        burst_cycles: int = 8,
    ) -> None:
        self.gpio_chip = gpio_chip
        self.gpio_line = gpio_line
        self.frequency_hz = frequency_hz
        self.burst_cycles = burst_cycles
        self.fired_count = 0
        self.last_fired_time: Optional[float] = None

    def fire(self) -> None:
        """Record fire event and update timestamp."""
        self.fired_count += 1
        self.last_fired_time = time.monotonic()

    def reset(self) -> None:
        """Reset fire count."""
        self.fired_count = 0
        self.last_fired_time = None

    def close(self) -> None:
        """No-op cleanup."""
        pass

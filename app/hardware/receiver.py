"""Envelope receiver interfaces and implementations for ultrasonic ToF and peak detection."""
from abc import ABC, abstractmethod
import math
import random
import time
from typing import Optional, Tuple

from app.hardware.adc import ADCDriver


def interpolate_crossing(
    t_prev: float,
    v_prev: float,
    t_curr: float,
    v_curr: float,
    threshold: float,
) -> float:
    """Calculate sub-sample interpolated crossing time using linear interpolation.

    Args:
        t_prev: Timestamp of sample immediately before threshold crossing (in microseconds).
        v_prev: ADC value of sample immediately before threshold crossing.
        t_curr: Timestamp of first sample at or above threshold (in microseconds).
        v_curr: ADC value of first sample at or above threshold.
        threshold: Envelope threshold value.

    Returns:
        float: Estimated sub-sample timestamp (in microseconds) where the signal
               crossed the threshold.
    """
    if v_curr == v_prev:
        return t_curr
    fraction = (threshold - v_prev) / (v_curr - v_prev)
    fraction = max(0.0, min(1.0, fraction))
    return t_prev + fraction * (t_curr - t_prev)


class EnvelopeReceiver(ABC):
    """Abstract base class for ultrasonic envelope receivers."""

    @abstractmethod
    def capture(self) -> Tuple[float, int]:
        """Capture echo envelope and detect time-of-flight and peak amplitude.

        Returns:
            Tuple[float, int]: (travel_time_us, peak_amplitude)
        """
        pass

    def close(self) -> None:
        """Release receiver resources."""
        pass


class RealReceiver(EnvelopeReceiver):
    """Real envelope receiver polling an ADC driver with sub-sample linear interpolation."""

    def __init__(
        self,
        adc_driver: ADCDriver,
        threshold: int = 200,
        timeout_us: int = 30000,
        raise_on_timeout: bool = False,
        peak_search_window_us: int = 5000,
    ) -> None:
        """Initialize RealReceiver.

        Args:
            adc_driver: ADC driver instance providing read_raw() method.
            threshold: ADC raw counts above noise floor to qualify as signal arrival.
            timeout_us: Maximum listen window in microseconds (default 30,000 us = 30 ms).
            raise_on_timeout: If True, raises TimeoutError on timeout; if False, returns (-1.0, 0).
            peak_search_window_us: Window after first crossing to continue searching for peak amplitude.
        """
        self.adc = adc_driver
        self.threshold = threshold
        self.timeout_us = timeout_us
        self.raise_on_timeout = raise_on_timeout
        self.peak_search_window_us = peak_search_window_us

    def capture(self) -> Tuple[float, int]:
        """Poll ADC, detect threshold crossing with sub-sample interpolation, and find peak.

        Returns:
            Tuple[float, int]: (travel_time_us, peak_amplitude).
                               If no signal detected within timeout, returns (-1.0, 0)
                               unless raise_on_timeout is True.
        """
        start_ns = time.perf_counter_ns()
        timeout_ns = int(self.timeout_us * 1000)

        t_prev: Optional[float] = None
        v_prev: Optional[float] = None
        crossing_time_us: Optional[float] = None
        peak_amplitude = 0
        search_end_time_us: Optional[float] = None

        while True:
            elapsed_ns = time.perf_counter_ns() - start_ns
            elapsed_us = elapsed_ns / 1000.0

            if elapsed_ns >= timeout_ns:
                break

            if search_end_time_us is not None and elapsed_us >= search_end_time_us:
                break

            val = self.adc.read_raw()

            if val > peak_amplitude:
                peak_amplitude = val

            if crossing_time_us is None and val >= self.threshold:
                if t_prev is not None and v_prev is not None and v_prev < self.threshold:
                    crossing_time_us = interpolate_crossing(
                        t_prev, v_prev, elapsed_us, float(val), float(self.threshold)
                    )
                else:
                    crossing_time_us = elapsed_us

                search_end_time_us = min(
                    float(self.timeout_us),
                    crossing_time_us + self.peak_search_window_us,
                )

            t_prev = elapsed_us
            v_prev = float(val)

        if crossing_time_us is None:
            if self.raise_on_timeout:
                raise TimeoutError(f"No ultrasonic echo detected within {self.timeout_us} us timeout.")
            return -1.0, 0

        return round(crossing_time_us, 2), int(peak_amplitude)

    def close(self) -> None:
        """Close associated ADC driver."""
        self.adc.close()


class MockReceiver(EnvelopeReceiver):
    """Mock ultrasonic receiver returning synthetic or deterministic measurements."""

    def __init__(
        self,
        travel_time_us: float = 4200.0,
        amplitude: int = 62,
        noise_std: float = 0.0,
        adc_driver: Optional[ADCDriver] = None,
        threshold: int = 200,
    ) -> None:
        """Initialize MockReceiver.

        Args:
            travel_time_us: Nominal time-of-flight in microseconds.
            amplitude: Nominal peak amplitude count.
            noise_std: Standard deviation of Gaussian noise added to measurements.
            adc_driver: If provided, can read from mock ADC driver instead.
            threshold: Threshold for ADC crossing if adc_driver is supplied.
        """
        self.travel_time_us = travel_time_us
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.adc_driver = adc_driver
        self.threshold = threshold
        self.capture_count = 0

    def capture(self) -> Tuple[float, int]:
        """Generate simulated time-of-flight and amplitude.

        Returns:
            Tuple[float, int]: (travel_time_us, peak_amplitude)
        """
        self.capture_count += 1

        if self.adc_driver is not None:
            real_rec = RealReceiver(self.adc_driver, threshold=self.threshold)
            return real_rec.capture()

        tof = self.travel_time_us
        amp = self.amplitude

        if self.noise_std > 0.0:
            tof += random.gauss(0.0, self.noise_std)
            amp_noise = random.gauss(0.0, self.noise_std * 0.1)
            amp = max(0, int(round(amp + amp_noise)))

        return round(float(tof), 2), int(amp)

    def set_nominal(self, travel_time_us: float, amplitude: int) -> None:
        """Update nominal measurement values."""
        self.travel_time_us = travel_time_us
        self.amplitude = amplitude

    def close(self) -> None:
        """No-op cleanup."""
        if self.adc_driver is not None:
            self.adc_driver.close()

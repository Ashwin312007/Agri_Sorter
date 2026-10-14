"""ADC driver interfaces and implementations for ADS1115 and mock testing."""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

# Attempt to import Adafruit ADS1x15 library; graceful fallback if not present
try:
    import board
    import busio
    import adafruit_ads1x15.ads1x15 as ads
    from adafruit_ads1x15.ads1115 import ADS1115
    from adafruit_ads1x15.analog_in import AnalogIn

    _HAS_ADS1115 = True
except (ImportError, NotImplementedError):
    # NotImplementedError can occur on non-ARM boards when importing board
    _HAS_ADS1115 = False


class ADCDriver(ABC):
    """Abstract base class for analog-to-digital converter drivers."""

    @abstractmethod
    def read_raw(self) -> int:
        """Read a single raw ADC count value.

        Returns:
            int: Raw ADC reading.
        """
        pass

    def read_voltage(self) -> float:
        """Read voltage approximation.

        Returns:
            float: Voltage in volts.
        """
        return 0.0

    def close(self) -> None:
        """Release any underlying hardware resources."""
        pass


class RealADC(ADCDriver):
    """Hardware driver for ADS1115 16-bit I2C ADC using CircuitPython libraries."""

    def __init__(
        self,
        bus: int = 1,
        address: int = 0x48,
        channel: int = 0,
        gain: int = 1,
        data_rate: int = 860,
    ) -> None:
        """Initialize ADS1115 driver.

        Args:
            bus: I2C bus index (typically 1 on Raspberry Pi).
            address: I2C device address (default 0x48).
            channel: ADC channel index (0 to 3).
            gain: Programmable gain amplifier setting (default 1 for +/-4.096V).
            data_rate: Sampling rate in samples per second (default 860).

        Raises:
            RuntimeError: If Adafruit CircuitPython ADS1x15 libraries are missing
                          or hardware initialization fails.
        """
        if not _HAS_ADS1115:
            raise RuntimeError(
                "adafruit-circuitpython-ads1x15 is not installed or platform is unsupported. "
                "Install with: pip install adafruit-circuitpython-ads1x15 adafruit-blinka"
            )

        self.bus_index = bus
        self.address = address
        self.channel_index = channel
        self.gain = gain
        self.data_rate = data_rate

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._ads = ADS1115(self._i2c, address=self.address, data_rate=self.data_rate, gain=self.gain)

            channel_map = {
                0: ads.P0,
                1: ads.P1,
                2: ads.P2,
                3: ads.P3,
            }
            if channel not in channel_map:
                raise ValueError(f"Invalid ADC channel: {channel}. Must be 0, 1, 2, or 3.")

            self._chan = AnalogIn(self._ads, channel_map[channel])
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ADS1115 on I2C bus {bus} address 0x{address:02x}: {exc}") from exc

    def read_raw(self) -> int:
        """Read raw 16-bit signed integer count from ADS1115."""
        return int(self._chan.value)

    def read_voltage(self) -> float:
        """Read converted voltage from ADS1115 channel."""
        return float(self._chan.voltage)

    def close(self) -> None:
        """Deinitialize I2C bus if held."""
        if hasattr(self, "_i2c") and self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None


# Alias for backward compatibility / explicit naming
ADS1115Driver = RealADC


class MockADC(ADCDriver):
    """Mock ADC driver for deterministic offline testing and simulation."""

    def __init__(
        self,
        values: Optional[Sequence[int]] = None,
        default_value: int = 0,
        fsr_voltage: float = 4.096,
        max_counts: int = 32767,
    ) -> None:
        """Initialize MockADC.

        Args:
            values: Optional sequence of integer values to cycle through.
            default_value: Value returned if `values` is empty or None.
            fsr_voltage: Full-scale range voltage for voltage conversion.
            max_counts: Maximum positive counts for 16-bit ADC (32767).
        """
        self._values = list(values) if values is not None else []
        self._default_value = default_value
        self._fsr_voltage = fsr_voltage
        self._max_counts = max_counts
        self._index = 0
        self.read_count = 0

    def set_values(self, values: Sequence[int]) -> None:
        """Update value sequence and reset index."""
        self._values = list(values)
        self._index = 0

    def reset(self) -> None:
        """Reset index and call count to zero."""
        self._index = 0
        self.read_count = 0

    def read_raw(self) -> int:
        """Return the next raw count in the configured sequence, cycling if needed."""
        self.read_count += 1
        if not self._values:
            return self._default_value

        val = self._values[self._index]
        self._index = (self._index + 1) % len(self._values)
        return int(val)

    def read_voltage(self) -> float:
        """Return voltage corresponding to current raw count."""
        raw = self.read_raw()
        return (raw / self._max_counts) * self._fsr_voltage

    def close(self) -> None:
        """No-op cleanup for mock driver."""
        pass

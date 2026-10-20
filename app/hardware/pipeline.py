"""Ultrasonic hardware orchestration pipeline."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from app.hardware.adc import ADCDriver, MockADC, RealADC
from app.hardware.config import load_config
from app.hardware.receiver import EnvelopeReceiver, MockReceiver, RealReceiver
from app.hardware.transmitter import BurstTransmitter, MockTransmitter, RealTransmitter


class UltrasonicPipeline:
    """Orchestrates burst transmission and envelope reception for fruit quality sensing."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        mock: bool = False,
    ) -> None:
        """Initialize ultrasonic pipeline.

        Args:
            config: Configuration dictionary or path to hardware.toml.
                    If None, loads default config/hardware.toml.
            mock: If True, uses mock hardware drivers regardless of platform.
        """
        if config is None or isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.mock = mock or self.config.get("mock", False)
        self._init_drivers()

    def _init_drivers(self) -> None:
        """Initialize transmitter, ADC, and receiver drivers based on mock mode."""
        tx_cfg = self.config.get("transmitter", {})
        rx_cfg = self.config.get("receiver", {})
        adc_cfg = self.config.get("adc", {})

        if self.mock:
            self.transmitter: BurstTransmitter = MockTransmitter(
                gpio_chip=tx_cfg.get("gpio_chip", "/dev/gpiochip0"),
                gpio_line=tx_cfg.get("gpio_line", 18),
                frequency_hz=tx_cfg.get("frequency_hz", 40000),
                burst_cycles=tx_cfg.get("burst_cycles", 8),
            )
            self.adc: ADCDriver = MockADC(values=None, default_value=0)
            self.receiver: EnvelopeReceiver = MockReceiver(
                travel_time_us=4200.0,
                amplitude=62,
                noise_std=0.0,
            )
        else:
            self.transmitter = RealTransmitter(
                gpio_chip=tx_cfg.get("gpio_chip", "/dev/gpiochip0"),
                gpio_line=tx_cfg.get("gpio_line", 18),
                frequency_hz=tx_cfg.get("frequency_hz", 40000),
                burst_cycles=tx_cfg.get("burst_cycles", 8),
            )
            self.adc = RealADC(
                bus=adc_cfg.get("i2c_bus", 1),
                address=adc_cfg.get("address", 0x48),
                channel=adc_cfg.get("channel", 0),
                gain=adc_cfg.get("gain", 1),
                data_rate=adc_cfg.get("data_rate", 860),
            )
            self.receiver = RealReceiver(
                adc_driver=self.adc,
                threshold=rx_cfg.get("envelope_threshold", 200),
                timeout_us=rx_cfg.get("timeout_us", 30000),
            )

    def measure(self) -> Dict[str, Union[float, int]]:
        """Trigger burst and capture envelope response.

        Returns:
            dict: Measurement dictionary containing:
                  - 'travel_time_us': Estimated time-of-flight in microseconds.
                  - 'received_amplitude': Peak received signal amplitude count.
        """
        self.transmitter.fire()
        travel_time_us, received_amplitude = self.receiver.capture()
        return {
            "travel_time_us": travel_time_us,
            "received_amplitude": received_amplitude,
        }

    def capture(self) -> Tuple[float, int]:
        """Convenience method returning (travel_time_us, received_amplitude) tuple."""
        res = self.measure()
        return float(res["travel_time_us"]), int(res["received_amplitude"])

    def close(self) -> None:
        """Release all hardware driver resources."""
        if hasattr(self, "receiver") and self.receiver is not None:
            self.receiver.close()
        if hasattr(self, "transmitter") and self.transmitter is not None:
            self.transmitter.close()

    def __enter__(self) -> "UltrasonicPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

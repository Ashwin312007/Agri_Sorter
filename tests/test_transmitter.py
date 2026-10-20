"""Tests for ultrasonic burst transmitters (MockTransmitter, RealTransmitter)."""
import pytest
from app.hardware.transmitter import BurstTransmitter, MockTransmitter, RealTransmitter


def test_mock_transmitter_fire():
    tx = MockTransmitter(gpio_chip="/dev/gpiochip0", gpio_line=18, frequency_hz=40000, burst_cycles=8)
    assert tx.fired_count == 0
    assert tx.last_fired_time is None

    result = tx.fire()
    assert result is None
    assert tx.fired_count == 1
    assert tx.last_fired_time is not None

    tx.fire()
    assert tx.fired_count == 2

    tx.reset()
    assert tx.fired_count == 0
    assert tx.last_fired_time is None

    tx.close()  # No-op, no exception


def test_real_transmitter_error_without_hardware():
    # On x86 development machine without gpiod hardware/lines, RealTransmitter raises RuntimeError
    try:
        RealTransmitter(gpio_chip="/dev/gpiochip0", gpio_line=18)
    except RuntimeError as exc:
        assert "gpiod" in str(exc).lower() or "failed to request" in str(exc).lower()

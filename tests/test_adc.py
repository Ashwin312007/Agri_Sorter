"""Tests for ADC drivers (MockADC, RealADC interface, voltage conversion)."""
import pytest
from app.hardware.adc import ADCDriver, MockADC, RealADC, ADS1115Driver


def test_mock_adc_cycles_through_values():
    adc = MockADC([100, 200, 300])
    assert adc.read_raw() == 100
    assert adc.read_raw() == 200
    assert adc.read_raw() == 300
    # Wraps around
    assert adc.read_raw() == 100
    assert adc.read_raw() == 200
    assert adc.read_count == 5


def test_mock_adc_default_value_when_empty():
    adc = MockADC(values=[], default_value=42)
    assert adc.read_raw() == 42
    assert adc.read_raw() == 42


def test_mock_adc_voltage_conversion():
    adc = MockADC([32767, 16383, 0], fsr_voltage=4.096, max_counts=32767)
    v_max = adc.read_voltage()
    assert v_max == pytest.approx(4.096, rel=1e-3)
    v_half = adc.read_voltage()
    assert v_half == pytest.approx(2.048, rel=1e-3)
    v_zero = adc.read_voltage()
    assert v_zero == pytest.approx(0.0, abs=1e-3)


def test_mock_adc_set_values_and_reset():
    adc = MockADC([1, 2, 3])
    assert adc.read_raw() == 1
    adc.set_values([10, 20])
    assert adc.read_raw() == 10
    assert adc.read_raw() == 20
    assert adc.read_raw() == 10
    adc.reset()
    assert adc.read_count == 0
    assert adc.read_raw() == 10


def test_mock_adc_close():
    adc = MockADC([10])
    adc.close()  # No exception


def test_driver_alias():
    assert ADS1115Driver is RealADC


def test_real_adc_error_without_hardware():
    # On x86 development machine without Adafruit hardware/libraries, RealADC raises RuntimeError
    try:
        RealADC(bus=1, address=0x48, channel=0)
    except (RuntimeError, ValueError) as exc:
        assert "adafruit" in str(exc).lower() or "failed to initialize" in str(exc).lower()

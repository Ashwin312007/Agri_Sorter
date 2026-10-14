"""Tests for ultrasonic receiver, sub-sample interpolation, and MockReceiver."""
import pytest
from app.hardware.adc import MockADC
from app.hardware.receiver import (
    EnvelopeReceiver,
    MockReceiver,
    RealReceiver,
    interpolate_crossing,
)


def test_interpolate_crossing_exact_midpoint():
    # t: 100 -> 200 us, v: 10 -> 30, threshold: 20
    t = interpolate_crossing(
        t_prev=100.0,
        v_prev=10.0,
        t_curr=200.0,
        v_curr=30.0,
        threshold=20.0,
    )
    assert t == pytest.approx(150.0)


def test_interpolate_crossing_quarter_fraction():
    # t: 0 -> 100 us, v: 0 -> 100, threshold: 25
    t = interpolate_crossing(
        t_prev=0.0,
        v_prev=0.0,
        t_curr=100.0,
        v_curr=100.0,
        threshold=25.0,
    )
    assert t == pytest.approx(25.0)


def test_interpolate_crossing_flat_signal():
    # Identical values: prevent zero division
    t = interpolate_crossing(
        t_prev=100.0,
        v_prev=50.0,
        t_curr=200.0,
        v_curr=50.0,
        threshold=50.0,
    )
    assert t == 200.0


def test_mock_receiver_capture_returns_tuple():
    rx = MockReceiver(travel_time_us=4200.0, amplitude=62)
    tof, amp = rx.capture()
    assert isinstance(tof, float)
    assert isinstance(amp, int)
    assert tof == pytest.approx(4200.0)
    assert amp == 62
    assert rx.capture_count == 1


def test_mock_receiver_set_nominal():
    rx = MockReceiver(travel_time_us=4200.0, amplitude=62)
    rx.set_nominal(4375.0, 41)
    tof, amp = rx.capture()
    assert tof == pytest.approx(4375.0)
    assert amp == 41


def test_mock_receiver_with_noise():
    rx = MockReceiver(travel_time_us=4200.0, amplitude=62, noise_std=5.0)
    tof, amp = rx.capture()
    assert 4150.0 < tof < 4250.0
    assert 50 < amp < 75


def test_real_receiver_with_mock_adc_crossing_and_peak():
    # Sequence simulates baseline noise -> pulse rising above threshold -> peak -> decay
    adc = MockADC([50, 60, 55, 250, 450, 350, 100])
    rx = RealReceiver(
        adc_driver=adc,
        threshold=200,
        timeout_us=20000,
        peak_search_window_us=2000,
    )
    tof, peak_amp = rx.capture()
    assert tof > 0.0
    assert peak_amp >= 250
    rx.close()


def test_real_receiver_timeout_sentinel():
    # Values never reach threshold of 200
    adc = MockADC([50, 60, 55, 70])
    rx = RealReceiver(
        adc_driver=adc,
        threshold=200,
        timeout_us=1000,  # 1 ms timeout for fast test
        raise_on_timeout=False,
    )
    tof, peak_amp = rx.capture()
    assert tof == -1.0
    assert peak_amp == 70 or peak_amp == 0 or peak_amp <= 70


def test_real_receiver_timeout_raises_when_requested():
    adc = MockADC([10, 20])
    rx = RealReceiver(
        adc_driver=adc,
        threshold=200,
        timeout_us=500,
        raise_on_timeout=True,
    )
    with pytest.raises(TimeoutError):
        rx.capture()

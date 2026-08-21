"""The control engine: block-size invariance, composition, and compilation."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.controls import (
    ConstantControl,
    ExpressionControl,
    ExternalControl,
    KeyframeControl,
    LfoControl,
    MapRangeControl,
    ProductControl,
    RampControl,
    RandomWalkControl,
    SmoothedControl,
    SumControl,
    StepSequenceControl,
    compile_control,
    control_from_dict,
    instantaneous_frequency_deviation_hz,
    linear_transition_control,
)
from src.audio.sam_workbench.waveforms import WAVEFORMS, evaluate_waveform

SAMPLE_RATE = 48_000


def _split_render(control, frames: int, splits: tuple[int, ...]) -> np.ndarray:
    blocks = []
    position = 0
    for span in splits:
        blocks.append(control.render(position, span, SAMPLE_RATE))
        position += span
    if position < frames:
        blocks.append(control.render(position, frames - position, SAMPLE_RATE))
    return np.concatenate(blocks)


STATELESS_CONTROLS = [
    ConstantControl(value=0.25),
    LfoControl(rate_hz=3.7, depth=2.0, offset=-0.5, phase_rad=0.2),
    LfoControl(rate_hz=1.5, depth=1.0, waveform="triangle"),
    LfoControl(rate_hz=0.75, depth=1.0, waveform="square"),
    LfoControl(rate_hz=2.25, depth=1.0, waveform="sawtooth"),
    RampControl(slope_per_s=12.5, intercept=-3.0),
    KeyframeControl(keyframes=((0.0, 100.0), (0.01, 250.0), (0.02, 180.0))),
    KeyframeControl(keyframes=((0.0, 0.0), (0.015, 1.0)), interpolation="smooth"),
    KeyframeControl(keyframes=((0.0, 0.0), (0.01, 1.0)), interpolation="step"),
    SumControl(terms=(ConstantControl(value=1.0), LfoControl(rate_hz=5.0, depth=0.5))),
    ProductControl(factors=(ConstantControl(value=2.0), LfoControl(rate_hz=3.0, depth=1.0))),
    MapRangeControl(source=LfoControl(rate_hz=4.0, depth=1.0), to_low=100.0, to_high=200.0),
    StepSequenceControl(values=(1.0, 2.0, -1.0), step_duration_s=0.003),
    RandomWalkControl(step_interval_s=0.004, step_size=0.2, seed=73),
    ExpressionControl(expression="2 + sin(2 * pi * 3 * t)"),
    ExternalControl(samples=((0.0, 1.0), (0.01, 2.0), (0.02, -1.0)), fallback=0.5),
]


@pytest.mark.parametrize("control", STATELESS_CONTROLS, ids=lambda control: type(control).__name__ + repr(control)[:24])
def test_stateless_controls_are_block_size_invariant(control):
    frames = 1000
    whole = control.render(0, frames, SAMPLE_RATE)
    np.testing.assert_array_equal(whole, _split_render(control, frames, (137, 863)))
    np.testing.assert_array_equal(whole, _split_render(control, frames, (1, 2, 3, 500, 494)))


def test_controls_render_the_requested_shape_and_dtype():
    values = LfoControl(rate_hz=2.0).render(0, 64, SAMPLE_RATE)
    assert values.shape == (64,)
    assert values.dtype == np.float64
    assert LfoControl(rate_hz=2.0).render(0, 0, SAMPLE_RATE).shape == (0,)


def test_lfo_matches_its_closed_form():
    control = LfoControl(rate_hz=3.0, depth=0.5, offset=1.0, phase_rad=0.25)
    times = np.arange(256) / SAMPLE_RATE
    expected = 1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * times + 0.25)
    np.testing.assert_allclose(control.render(0, 256, SAMPLE_RATE), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_waveforms_stay_within_unit_amplitude(waveform):
    phase = np.linspace(-40.0, 40.0, 4096)
    values = evaluate_waveform(waveform, phase)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) <= 1.0 + 1e-12


def test_bounds_are_applied_after_evaluation():
    control = LfoControl(rate_hz=1.0, depth=5.0, minimum=-1.0, maximum=2.0)
    values = control.render(0, SAMPLE_RATE, SAMPLE_RATE)
    assert values.min() >= -1.0
    assert values.max() <= 2.0


def test_keyframes_hold_before_the_first_and_after_the_last():
    control = KeyframeControl(keyframes=((0.01, 5.0), (0.02, 9.0)))
    values = control.render(0, 1500, SAMPLE_RATE)
    assert values[0] == pytest.approx(5.0)
    assert values[-1] == pytest.approx(9.0)


def test_step_interpolation_does_not_ramp():
    control = KeyframeControl(keyframes=((0.0, 0.0), (0.01, 1.0)), interpolation="step")
    values = control.render(0, 960, SAMPLE_RATE)
    assert set(np.unique(values)) == {0.0, 1.0}


def test_transition_pairs_compile_into_keyframes():
    control = linear_transition_control(100.0, 200.0, start_s=0.0, duration_s=0.01)
    values = control.render(0, 960, SAMPLE_RATE)
    assert values[0] == pytest.approx(100.0)
    assert values[-1] == pytest.approx(200.0, abs=0.5)
    assert np.all(np.diff(values) >= 0.0)


def test_unchanging_transition_pairs_fold_into_a_constant():
    assert isinstance(linear_transition_control(4.0, 4.0, duration_s=1.0), ConstantControl)
    assert isinstance(linear_transition_control(1.0, 4.0, duration_s=0.0), ConstantControl)


def test_compile_control_folds_constants_and_disabled_controls():
    assert compile_control(3.5) == ConstantControl(value=3.5)
    assert compile_control(LfoControl(rate_hz=5.0, depth=0.0, offset=2.0)).constant_value() == 2.0
    assert compile_control(LfoControl(rate_hz=5.0, depth=1.0, enabled=False)).constant_value() == 0.0
    lfo = LfoControl(rate_hz=5.0, depth=1.0)
    assert compile_control(lfo) is lfo


def test_smoothing_is_stateful_and_declares_it():
    control = SmoothedControl(source=KeyframeControl(keyframes=((0.0, 0.0), (0.001, 1.0)), interpolation="step"), smoothing_ms=5.0)
    assert control.is_stateful is True
    control.reset()
    smoothed = control.render(0, 480, SAMPLE_RATE)
    assert np.all(np.diff(smoothed) >= -1e-12)
    assert smoothed[-1] < 1.0  # a one-pole approaches its target rather than jumping


def test_smoothing_removes_the_step_discontinuity():
    step = KeyframeControl(keyframes=((0.0, 0.0), (0.005, 1.0)), interpolation="step")
    raw = step.render(0, 480, SAMPLE_RATE)
    smoothed = SmoothedControl(source=step, smoothing_ms=10.0).render(0, 480, SAMPLE_RATE)
    assert np.max(np.abs(np.diff(raw))) == pytest.approx(1.0)
    assert np.max(np.abs(np.diff(smoothed))) < 0.05


@pytest.mark.parametrize("control", STATELESS_CONTROLS, ids=lambda control: type(control).__name__)
def test_controls_round_trip_through_their_serialized_form(control):
    restored = control_from_dict(control.to_dict())
    np.testing.assert_array_equal(
        restored.render(0, 128, SAMPLE_RATE), control.render(0, 128, SAMPLE_RATE)
    )


def test_unknown_control_kind_is_rejected():
    with pytest.raises(ValueError):
        control_from_dict({"kind": "quantum"})


@pytest.mark.parametrize(
    "expression",
    ("__import__('os')", "t.__class__", "open('/tmp/nope')", "missing + 1"),
)
def test_expression_control_rejects_unsafe_or_unknown_syntax(expression):
    with pytest.raises(ValueError):
        ExpressionControl(expression=expression)


def test_external_control_uses_a_safe_fallback_before_a_capture():
    control = ExternalControl(samples=((0.01, 1.0), (0.02, 2.0)), fallback=-0.5)
    values = control.render(0, 480, SAMPLE_RATE)
    assert values[0] == pytest.approx(-0.5)
    assert values[-1] == pytest.approx(-0.5)


def test_random_walk_is_reproducible_and_seeded():
    first = RandomWalkControl(seed=7).render(0, SAMPLE_RATE, SAMPLE_RATE)
    second = RandomWalkControl(seed=7).render(0, SAMPLE_RATE, SAMPLE_RATE)
    different = RandomWalkControl(seed=8).render(0, SAMPLE_RATE, SAMPLE_RATE)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_peak_frequency_deviation_formula():
    assert instantaneous_frequency_deviation_hz(1.2, 4.0) == pytest.approx(4.8)
    assert instantaneous_frequency_deviation_hz(-1.2, 4.0) == pytest.approx(4.8)

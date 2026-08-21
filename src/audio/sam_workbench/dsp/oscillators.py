"""Carrier oscillators.

An oscillator turns a control-driven frequency into accumulated phase and then
into a waveform. Phase 1 renders the sine carrier of the SAM equations; the
same accumulation feeds the band-limited and wavetable carriers of the
generalized signal model later.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..controls import ControlBase, as_control
from ..waveforms import evaluate_waveform
from .phase import PhaseAccumulator, integrate_phase, phase_from_constant_frequency

__all__ = ["carrier_phase", "render_oscillator"]


def carrier_phase(
    frequency: ControlBase | float,
    start_sample: int,
    frames: int,
    sample_rate: float,
    initial_phase_rad: float = 0.0,
    *,
    accumulator: PhaseAccumulator | None = None,
    inclusive: bool = False,
) -> NDArray[np.float64]:
    """Accumulated carrier phase for one block.

    A constant frequency uses the closed form, which is exact and stateless. A
    time-varying frequency is integrated - from ``accumulator`` when one is
    supplied, otherwise from the absolute sample index.
    """

    control = as_control(frequency)
    constant = control.constant_value()
    if constant is not None and accumulator is None and not inclusive:
        return phase_from_constant_frequency(constant, start_sample, frames, sample_rate, initial_phase_rad)

    frequency_curve = control.render(start_sample, frames, sample_rate)
    if accumulator is not None:
        return accumulator.advance(frequency_curve, sample_rate)
    return integrate_phase(frequency_curve, sample_rate, initial_phase_rad, inclusive=inclusive)


def render_oscillator(
    frequency: ControlBase | float,
    start_sample: int,
    frames: int,
    sample_rate: float,
    *,
    waveform: str = "sine",
    initial_phase_rad: float = 0.0,
    amplitude: ControlBase | float = 1.0,
) -> NDArray[np.float64]:
    """Render a mono oscillator block."""

    phase = carrier_phase(frequency, start_sample, frames, sample_rate, initial_phase_rad)
    signal = evaluate_waveform(waveform, phase)
    gain = as_control(amplitude)
    constant_gain = gain.constant_value()
    if constant_gain is not None:
        return float(constant_gain) * signal
    return gain.render(start_sample, frames, sample_rate) * signal

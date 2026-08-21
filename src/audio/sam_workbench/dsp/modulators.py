"""Modulators: the ``q_k(psi_k(t))`` terms of the SAM equations.

A modulator is an independent oscillator whose output is scaled by a per-ear
depth in radians and summed into the carrier's phase argument:

.. code-block:: text

    s_e(t) = a_e(t) * g[ theta_c(t) + sum_k beta_{e,k}(t) q_k(psi_k(t)) + phi_e(t) ]

The exact symmetric SAM of the patent is the case of one sinusoidal modulator
with ``beta_L = beta_R = beta`` and opposed signs, which
:func:`modulation_sum` expresses through a per-ear polarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.controls import ControlBase, as_control
from src.audio.sam_workbench.waveforms import evaluate_waveform
from src.audio.sam_workbench.dsp.oscillators import carrier_phase
from src.audio.sam_workbench.dsp.phase import PhaseAccumulator

__all__ = ["ModulatorSpec", "render_modulator", "modulation_sum"]


@dataclass(frozen=True)
class ModulatorSpec:
    """One independent modulator.

    ``depth_rad`` is the modulation index in radians. ``polarity`` is applied
    per ear by :func:`modulation_sum`, so a single specification can drive both
    ears in opposition without duplicating its state.
    """

    rate_hz: ControlBase | float = 4.0
    depth_rad: ControlBase | float = 1.0
    phase_rad: float = 0.0
    waveform: str = "sine"
    bias_rad: ControlBase | float = 0.0
    enabled: bool = True

    def render(
        self,
        start_sample: int,
        frames: int,
        sample_rate: float,
        *,
        accumulator: PhaseAccumulator | None = None,
    ) -> NDArray[np.float64]:
        """Render this modulator's contribution in radians (before ear polarity)."""

        if not self.enabled or frames == 0:
            return np.zeros(frames, dtype=np.float64)
        phase = carrier_phase(
            self.rate_hz,
            start_sample,
            frames,
            sample_rate,
            float(self.phase_rad),
            accumulator=accumulator,
        )
        shape = evaluate_waveform(self.waveform, phase)
        depth = as_control(self.depth_rad)
        constant_depth = depth.constant_value()
        if constant_depth is None:
            shape = depth.render(start_sample, frames, sample_rate) * shape
        else:
            shape = float(constant_depth) * shape
        bias = as_control(self.bias_rad)
        constant_bias = bias.constant_value()
        if constant_bias is None:
            return shape + bias.render(start_sample, frames, sample_rate)
        if constant_bias:
            return shape + float(constant_bias)
        return shape


def render_modulator(
    spec: ModulatorSpec,
    start_sample: int,
    frames: int,
    sample_rate: float,
    *,
    accumulator: PhaseAccumulator | None = None,
) -> NDArray[np.float64]:
    """Render one modulator block in radians."""

    return spec.render(start_sample, frames, sample_rate, accumulator=accumulator)


def modulation_sum(
    modulators: tuple[ModulatorSpec, ...],
    start_sample: int,
    frames: int,
    sample_rate: float,
    *,
    accumulators: dict[int, PhaseAccumulator] | None = None,
) -> NDArray[np.float64]:
    """Sum every modulator's contribution for one block, in radians."""

    total = np.zeros(frames, dtype=np.float64)
    for index, modulator in enumerate(modulators):
        accumulator = None if accumulators is None else accumulators.get(index)
        total += modulator.render(start_sample, frames, sample_rate, accumulator=accumulator)
    return total

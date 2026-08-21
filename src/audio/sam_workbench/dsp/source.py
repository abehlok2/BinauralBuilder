"""Compiled sources and the abstract phase-modulation equations.

This module holds the one implementation of the SAM equations:

.. code-block:: text

    s_L(t) = a_L(t) * g[ theta_c(t) + sum_k beta_{L,k}(t) q_k(psi_k(t)) + phi_L(t) ]
    s_R(t) = a_R(t) * g[ theta_c(t) - sum_k beta_{R,k}(t) q_k(psi_k(t)) + phi_R(t) ]

The exact symmetric case of the patent is one sinusoidal modulator with equal
depths, which gives opposed instantaneous frequencies
``f_L,R(t) = f_c +/- beta * f_m * cos(2*pi*f_m*t)`` and an interaural phase
difference of ``2*beta*sin(2*pi*f_m*t)``.

Everything renders from an absolute ``start_sample``, so a chunked render is
identical to a whole render.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from ..controls import ControlBase, as_control
from ..conventions import AUDIO_DTYPE, CHANNEL_COUNT, CHANNEL_LEFT, CHANNEL_RIGHT
from ..waveforms import evaluate_waveform
from .blocks import iter_blocks
from .modulators import ModulatorSpec, modulation_sum
from .oscillators import carrier_phase
from .phase import PhaseAccumulator

__all__ = [
    "CompiledSource",
    "EAR_POLARITY_CANONICAL",
    "EAR_POLARITY_LEGACY",
    "EAR_POLARITY_SAME",
    "compile_source",
    "render_source",
    "render_source_block",
    "instantaneous_frequencies",
    "interaural_phase_difference",
]

#: The canonical exact mode: modulation is added at the left ear and
#: subtracted at the right ear.
EAR_POLARITY_CANONICAL = "left_plus_right_minus"
#: The polarity of unversioned legacy SAM2 voices, preserved for old presets.
EAR_POLARITY_LEGACY = "left_minus_right_plus"
#: Both ears modulated in the same direction: no interaural phase difference
#: from the modulator itself. Kept because one legacy tree rendered this way.
EAR_POLARITY_SAME = "left_plus_right_plus"

_POLARITY_SIGNS = {
    EAR_POLARITY_CANONICAL: (1.0, -1.0),
    EAR_POLARITY_LEGACY: (-1.0, 1.0),
    EAR_POLARITY_SAME: (1.0, 1.0),
}


@dataclass(frozen=True)
class CompiledSource:
    """A source reduced to controls and modulators, ready to render.

    Per-ear depth scales, phases, and gains are separate so the engine can
    render the exact symmetric case and deliberate deviations from it -
    asymmetric depth or rate, extra modulators, per-ear bias - without a
    special case for each.
    """

    amplitude: ControlBase | float = 0.5
    carrier_frequency_hz: ControlBase | float = 220.0
    carrier_phase_rad: float = 0.0
    modulators: tuple[ModulatorSpec, ...] = ()
    ear_polarity: str = EAR_POLARITY_CANONICAL
    left_depth_scale: ControlBase | float = 1.0
    right_depth_scale: ControlBase | float = 1.0
    left_phase_rad: ControlBase | float = 0.0
    right_phase_rad: ControlBase | float = 0.0
    left_gain: ControlBase | float = 1.0
    right_gain: ControlBase | float = 1.0
    waveform: str = "sine"
    #: Optional callables ``(start_sample, frames, sample_rate) -> ndarray`` that
    #: supply phase in radians directly. The legacy compatibility path uses them
    #: to hand the engine a path-derived interaural phase and an analytically
    #: integrated ramped carrier, neither of which is a plain oscillator, while
    #: still going through this one implementation of the SAM equations.
    modulation_provider: object | None = field(default=None, repr=False)
    carrier_phase_provider: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.ear_polarity not in _POLARITY_SIGNS:
            raise ValueError(
                f"unknown ear polarity {self.ear_polarity!r}; "
                f"expected one of {', '.join(sorted(_POLARITY_SIGNS))}"
            )

    @property
    def polarity_signs(self) -> tuple[float, float]:
        """``(left_sign, right_sign)`` applied to the modulation sum."""

        return _POLARITY_SIGNS[self.ear_polarity]

    def with_polarity(self, ear_polarity: str) -> "CompiledSource":
        return replace(self, ear_polarity=ear_polarity)


def compile_source(source, *, ear_polarity: str = EAR_POLARITY_CANONICAL) -> CompiledSource:
    """Compile a :class:`~src.audio.sam_workbench.model.Source` for rendering."""

    return CompiledSource(
        amplitude=float(source.amplitude_linear),
        carrier_frequency_hz=float(source.signal.carrier_frequency_hz),
        carrier_phase_rad=float(source.signal.phase_rad),
        modulators=(
            ModulatorSpec(
                rate_hz=float(source.sam.rate_hz),
                depth_rad=float(source.sam.depth_rad),
                phase_rad=float(source.sam.phase_rad),
            ),
        ),
        ear_polarity=ear_polarity,
    )


def _control_values(
    control: ControlBase | float, start_sample: int, frames: int, sample_rate: float
) -> NDArray[np.float64] | float:
    compiled = as_control(control)
    constant = compiled.constant_value()
    if constant is not None:
        return float(constant)
    return compiled.render(start_sample, frames, sample_rate)


def render_source_block(
    source: CompiledSource,
    start_sample: int,
    frames: int,
    sample_rate: float,
    *,
    carrier_accumulator: PhaseAccumulator | None = None,
    modulator_accumulators: dict[int, PhaseAccumulator] | None = None,
) -> NDArray[np.float64]:
    """Render one ``(2, frames)`` block of a compiled source, in float64."""

    if frames <= 0:
        return np.zeros((CHANNEL_COUNT, 0), dtype=np.float64)

    if source.carrier_phase_provider is not None:
        theta = np.asarray(
            source.carrier_phase_provider(start_sample, frames, sample_rate), dtype=np.float64
        )
        if theta.shape != (frames,):
            raise ValueError(
                f"carrier phase provider returned shape {theta.shape}, expected ({frames},)"
            )
        theta = theta + source.carrier_phase_rad
    else:
        theta = carrier_phase(
            source.carrier_frequency_hz,
            start_sample,
            frames,
            sample_rate,
            source.carrier_phase_rad,
            accumulator=carrier_accumulator,
        )

    if source.modulation_provider is not None:
        modulation = np.asarray(
            source.modulation_provider(start_sample, frames, sample_rate), dtype=np.float64
        )
        if modulation.shape != (frames,):
            raise ValueError(
                f"modulation provider returned shape {modulation.shape}, expected ({frames},)"
            )
    else:
        modulation = modulation_sum(
            source.modulators,
            start_sample,
            frames,
            sample_rate,
            accumulators=modulator_accumulators,
        )

    left_sign, right_sign = source.polarity_signs
    left_depth = _control_values(source.left_depth_scale, start_sample, frames, sample_rate)
    right_depth = _control_values(source.right_depth_scale, start_sample, frames, sample_rate)
    left_phase = _control_values(source.left_phase_rad, start_sample, frames, sample_rate)
    right_phase = _control_values(source.right_phase_rad, start_sample, frames, sample_rate)

    left_argument = theta + left_sign * left_depth * modulation + left_phase
    right_argument = theta + right_sign * right_depth * modulation + right_phase

    amplitude = _control_values(source.amplitude, start_sample, frames, sample_rate)
    left_gain = _control_values(source.left_gain, start_sample, frames, sample_rate)
    right_gain = _control_values(source.right_gain, start_sample, frames, sample_rate)

    block = np.empty((CHANNEL_COUNT, frames), dtype=np.float64)
    block[CHANNEL_LEFT] = amplitude * left_gain * evaluate_waveform(source.waveform, left_argument)
    block[CHANNEL_RIGHT] = amplitude * right_gain * evaluate_waveform(source.waveform, right_argument)
    return block


def render_source(
    source,
    sample_rate: float,
    frames: int,
    block_size: int | None = None,
    *,
    start_sample: int = 0,
    ear_polarity: str = EAR_POLARITY_CANONICAL,
    dtype=AUDIO_DTYPE,
) -> NDArray[np.floating]:
    """Render a source to ``(2, frames)`` channel-major audio.

    ``source`` may be a :class:`CompiledSource` or a project
    :class:`~src.audio.sam_workbench.model.Source`. ``block_size`` only changes
    how the work is divided: the samples are identical either way, which is the
    property the chunked BinauralBuilder path depends on.
    """

    compiled = source if isinstance(source, CompiledSource) else compile_source(source, ear_polarity=ear_polarity)
    frames = int(frames)
    if frames < 0:
        raise ValueError(f"frames must not be negative, got {frames}")

    audio = np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
    span = frames if block_size is None else int(block_size)
    for block in iter_blocks(frames, max(1, span), start_sample=int(start_sample)):
        offset = block.start_sample - int(start_sample)
        audio[:, offset : offset + block.frames] = render_source_block(
            compiled, block.start_sample, block.frames, sample_rate
        )
    return audio.astype(dtype, copy=False)


# --- analysis helpers -------------------------------------------------------


def instantaneous_frequencies(
    source: CompiledSource, start_sample: int, frames: int, sample_rate: float
) -> NDArray[np.float64]:
    """Per-ear instantaneous frequency in Hz, as ``(2, frames)``.

    Derived from the rendered phase arguments, so it stays correct for any
    modulator combination rather than only the closed-form symmetric case.
    """

    if frames < 2:
        raise ValueError("instantaneous frequency needs at least two frames")
    theta = (
        np.asarray(source.carrier_phase_provider(start_sample, frames, sample_rate), dtype=np.float64)
        + source.carrier_phase_rad
        if source.carrier_phase_provider is not None
        else carrier_phase(
            source.carrier_frequency_hz, start_sample, frames, sample_rate, source.carrier_phase_rad
        )
    )
    modulation = (
        np.asarray(source.modulation_provider(start_sample, frames, sample_rate), dtype=np.float64)
        if source.modulation_provider is not None
        else modulation_sum(source.modulators, start_sample, frames, sample_rate)
    )
    left_sign, right_sign = source.polarity_signs
    left_depth = _control_values(source.left_depth_scale, start_sample, frames, sample_rate)
    right_depth = _control_values(source.right_depth_scale, start_sample, frames, sample_rate)
    left_phase = _control_values(source.left_phase_rad, start_sample, frames, sample_rate)
    right_phase = _control_values(source.right_phase_rad, start_sample, frames, sample_rate)
    arguments = np.vstack(
        (
            theta + left_sign * left_depth * modulation + left_phase,
            theta + right_sign * right_depth * modulation + right_phase,
        )
    )
    return np.gradient(arguments, 1.0 / float(sample_rate), axis=1) / (2.0 * np.pi)


def interaural_phase_difference(
    source: CompiledSource, start_sample: int, frames: int, sample_rate: float
) -> NDArray[np.float64]:
    """Instantaneous IPD in radians (left argument minus right argument)."""

    modulation = (
        np.asarray(source.modulation_provider(start_sample, frames, sample_rate), dtype=np.float64)
        if source.modulation_provider is not None
        else modulation_sum(source.modulators, start_sample, frames, sample_rate)
    )
    left_sign, right_sign = source.polarity_signs
    left_depth = _control_values(source.left_depth_scale, start_sample, frames, sample_rate)
    right_depth = _control_values(source.right_depth_scale, start_sample, frames, sample_rate)
    left_phase = _control_values(source.left_phase_rad, start_sample, frames, sample_rate)
    right_phase = _control_values(source.right_phase_rad, start_sample, frames, sample_rate)
    return (
        left_sign * left_depth * modulation
        - right_sign * right_depth * modulation
        + (left_phase - right_phase)
    )

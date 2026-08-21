"""The abstract phase-modulation renderer and its required presets.

This engine is the patent's opposed-ear phase modulation and its deliberate
generalizations. It generates its own carrier, so ``process`` ignores the
incoming mono signal; the argument is kept so the engine satisfies the same
:class:`~src.audio.sam_workbench.render.base.SpatialRenderer` contract as the
geometric and HRTF engines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..controls import RampControl, as_control, instantaneous_frequency_deviation_hz
from ..conventions import CHANNEL_COUNT
from ..dsp.blocks import RenderBlock, RenderContext
from ..dsp.modulators import ModulatorSpec
from ..dsp.source import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    CompiledSource,
    instantaneous_frequencies,
    interaural_phase_difference,
    render_source_block,
)

__all__ = [
    "AbstractPMRenderer",
    "PRESET_BUILDERS",
    "preset_asymmetric",
    "preset_binaural_beat",
    "preset_diotic",
    "preset_discontinuous",
    "preset_exact_symmetric",
    "preset_multi_modulator",
    "peak_frequency_deviation_hz",
]

RENDERER_NAME = "exact_symmetric_sam"


class AbstractPMRenderer:
    """Render a :class:`CompiledSource` block by block.

    The renderer is stateless for constant carrier and modulator frequencies -
    every block is computed from its absolute sample index - so blocks may be
    rendered in any order and at any size. It only holds state when a control
    makes a frequency time-varying, in which case the phase is integrated.
    """

    def __init__(self, source: CompiledSource | None = None) -> None:
        self._source = source
        self._context: RenderContext | None = None
        self._next_sample = 0
        self._last_block_frames = 0

    # --- SpatialRenderer ---------------------------------------------------

    def prepare(self, context: RenderContext, source: CompiledSource) -> None:
        self._context = context
        self._source = source
        self._next_sample = 0

    def reset(self, sample_index: int = 0) -> None:
        self._next_sample = int(sample_index)

    def process(
        self, mono: NDArray[np.floating] | None, block: RenderBlock
    ) -> NDArray[np.float64]:
        """Render one block. ``mono`` is unused: this engine is its own source."""

        if self._source is None or self._context is None:
            raise RuntimeError("prepare() must be called before process()")
        audio = render_source_block(
            self._source, block.start_sample, block.frames, self._context.sample_rate_hz
        )
        self._next_sample = block.end_sample
        self._last_block_frames = block.frames
        return audio

    def latency_samples(self) -> int:
        """Abstract phase modulation is generated in place and adds no latency."""

        return 0

    def diagnostics(self) -> dict[str, Any]:
        if self._source is None or self._context is None:
            return {}
        sample_rate = self._context.sample_rate_hz
        frames = max(2, self._last_block_frames)
        start = max(0, self._next_sample - frames)
        frequencies = instantaneous_frequencies(self._source, start, frames, sample_rate)
        ipd = interaural_phase_difference(self._source, start, frames, sample_rate)
        return {
            "renderer": RENDERER_NAME,
            "ear_polarity": self._source.ear_polarity,
            "instantaneous_frequency_hz_min": float(np.min(frequencies)),
            "instantaneous_frequency_hz_max": float(np.max(frequencies)),
            "ipd_rad_min": float(np.min(ipd)),
            "ipd_rad_max": float(np.max(ipd)),
            "latency_samples": self.latency_samples(),
            "channels": CHANNEL_COUNT,
        }


def peak_frequency_deviation_hz(source: CompiledSource) -> float:
    """Peak instantaneous-frequency deviation summed over the modulators.

    Uses the closed form ``|beta * f_m|`` per sinusoidal modulator, which is
    what the validator compares against Nyquist and the configured carrier
    range.
    """

    total = 0.0
    for modulator in source.modulators:
        depth = as_control(modulator.depth_rad).constant_value()
        rate = as_control(modulator.rate_hz).constant_value()
        if depth is None or rate is None:
            continue
        total += instantaneous_frequency_deviation_hz(depth, rate)
    return total


# --- required presets -------------------------------------------------------


def preset_exact_symmetric(
    carrier_hz: float = 220.0, modulation_hz: float = 4.0, depth_rad: float = 1.0, amplitude: float = 0.5
) -> CompiledSource:
    """The patent's symmetric case: one modulator, equal depth, opposed ears."""

    return CompiledSource(
        amplitude=amplitude,
        carrier_frequency_hz=carrier_hz,
        modulators=(ModulatorSpec(rate_hz=modulation_hz, depth_rad=depth_rad),),
        ear_polarity=EAR_POLARITY_CANONICAL,
    )


def preset_asymmetric(
    carrier_hz: float = 220.0,
    modulation_hz: float = 4.0,
    left_depth_rad: float = 1.0,
    right_depth_rad: float = 0.4,
    amplitude: float = 0.5,
) -> CompiledSource:
    """Independent per-ear depth, the simplest deliberate break from symmetry."""

    return CompiledSource(
        amplitude=amplitude,
        carrier_frequency_hz=carrier_hz,
        modulators=(ModulatorSpec(rate_hz=modulation_hz, depth_rad=1.0),),
        left_depth_scale=left_depth_rad,
        right_depth_scale=right_depth_rad,
    )


def preset_multi_modulator(
    carrier_hz: float = 220.0,
    rates_hz: tuple[float, ...] = (4.0, 11.0),
    depths_rad: tuple[float, ...] = (0.8, 0.3),
    amplitude: float = 0.5,
) -> CompiledSource:
    """Several independent modulators summed into the phase argument."""

    if len(rates_hz) != len(depths_rad):
        raise ValueError("rates_hz and depths_rad must have the same length")
    return CompiledSource(
        amplitude=amplitude,
        carrier_frequency_hz=carrier_hz,
        modulators=tuple(
            ModulatorSpec(rate_hz=rate, depth_rad=depth) for rate, depth in zip(rates_hz, depths_rad)
        ),
    )


def preset_discontinuous(
    carrier_hz: float = 220.0, modulation_hz: float = 2.0, depth_rad: float = 1.0, amplitude: float = 0.5
) -> CompiledSource:
    """A stepped spatial phase, for jump behaviour and crossfade testing."""

    return CompiledSource(
        amplitude=amplitude,
        carrier_frequency_hz=carrier_hz,
        modulators=(ModulatorSpec(rate_hz=modulation_hz, depth_rad=depth_rad, waveform="square"),),
    )


def preset_binaural_beat(
    carrier_hz: float = 220.0, beat_hz: float = 4.0, amplitude: float = 0.5
) -> CompiledSource:
    """A traditional binaural beat, for comparison against phase modulation.

    The two ears run at ``carrier_hz +/- beat_hz / 2``, expressed as opposed
    linear phase ramps on one shared carrier. The interaural phase difference
    therefore grows without bound instead of oscillating - exactly the contrast
    with SAM that a listener is meant to be able to hear.
    """

    half_beat_rad_per_s = np.pi * float(beat_hz)
    return CompiledSource(
        amplitude=amplitude,
        carrier_frequency_hz=carrier_hz,
        modulators=(),
        left_phase_rad=RampControl(slope_per_s=half_beat_rad_per_s, unit="rad"),
        right_phase_rad=RampControl(slope_per_s=-half_beat_rad_per_s, unit="rad"),
    )


def preset_diotic(carrier_hz: float = 220.0, amplitude: float = 0.5) -> CompiledSource:
    """An identical tone in both ears: the static control condition."""

    return CompiledSource(amplitude=amplitude, carrier_frequency_hz=carrier_hz, modulators=())


#: Named presets required of the abstract engine.
PRESET_BUILDERS = {
    "exact_symmetric": preset_exact_symmetric,
    "asymmetric": preset_asymmetric,
    "multi_modulator": preset_multi_modulator,
    "discontinuous": preset_discontinuous,
    "binaural_beat": preset_binaural_beat,
    "diotic": preset_diotic,
}

# Re-exported so callers can name the legacy polarity without reaching into dsp.
LEGACY_POLARITY = EAR_POLARITY_LEGACY

"""Signal-processing primitives for the SAM workbench.

The layering is strict and one-directional: `waveforms` and `controls` sit
below this package, `dsp` composes them into oscillators, modulators,
envelopes, mixing, and limiting, and `render` builds renderers on top.
"""

from __future__ import annotations

from .blocks import RenderBlock, RenderContext, iter_blocks
from .envelopes import (
    DEFAULT_CROSSFADE_MS,
    apply_fade_in,
    apply_fade_out,
    crossfade_blocks,
    equal_power_crossfade,
    linear_fade,
    milliseconds_to_frames,
)
from .limiter import LimiterReport, limit_lookahead, safety_clamp, true_peak_estimate_db
from .mixer import apply_shared_gain, mix_stems, peak_level, shared_normalization_gain
from .modulators import ModulatorSpec, modulation_sum, render_modulator
from .oscillators import carrier_phase, render_oscillator
from .phase import PhaseAccumulator, integrate_phase, phase_from_constant_frequency, wrap_phase
from .source import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
    CompiledSource,
    compile_source,
    instantaneous_frequencies,
    interaural_phase_difference,
    render_source,
    render_source_block,
)

__all__ = [
    "DEFAULT_CROSSFADE_MS",
    "EAR_POLARITY_CANONICAL",
    "EAR_POLARITY_LEGACY",
    "EAR_POLARITY_SAME",
    "CompiledSource",
    "LimiterReport",
    "ModulatorSpec",
    "PhaseAccumulator",
    "RenderBlock",
    "RenderContext",
    "apply_fade_in",
    "apply_fade_out",
    "apply_shared_gain",
    "carrier_phase",
    "compile_source",
    "crossfade_blocks",
    "equal_power_crossfade",
    "instantaneous_frequencies",
    "integrate_phase",
    "interaural_phase_difference",
    "iter_blocks",
    "limit_lookahead",
    "linear_fade",
    "milliseconds_to_frames",
    "mix_stems",
    "modulation_sum",
    "peak_level",
    "phase_from_constant_frequency",
    "render_modulator",
    "render_oscillator",
    "render_source",
    "render_source_block",
    "safety_clamp",
    "shared_normalization_gain",
    "true_peak_estimate_db",
    "wrap_phase",
]

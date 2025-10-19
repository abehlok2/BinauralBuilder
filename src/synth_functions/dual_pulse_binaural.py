"""Dual Pulse Binaural synth function.

This module combines the binaural beat and isochronic tone synthesis
approaches so that the carriers maintain a binaural frequency offset
while a synchronous isochronic style pulse envelope modulates the level
of both channels.  The implementation reuses the existing binaural beat
engine for carrier generation and then applies channel-specific pulse
shaping inspired by the isochronic tone voice.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import butter, filtfilt

from .binaural_beat import binaural_beat, binaural_beat_transition
from .common import trapezoid_envelope_vectorized, calculate_transition_alpha


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range [0, 1]."""

    return max(0.0, min(1.0, value))


def _resolve_pulse_rate(requested_rate: float, fallback_rate: float) -> float:
    """Return a valid pulse rate using ``fallback_rate`` when needed."""

    rate = float(requested_rate)
    if rate <= 0.0:
        return float(fallback_rate)
    return rate


def _generate_channel_envelope(
    num_samples: int,
    sample_rate: float,
    pulse_rate: float,
    ramp_percent: float,
    gap_percent: float,
    phase_offset_cycles: float,
    shape: str,
    depth: float,
) -> np.ndarray:
    """Create a per-channel pulse envelope."""

    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    depth = _clamp01(depth)
    if pulse_rate <= 0.0 or depth == 0.0:
        return np.full(num_samples, 1.0, dtype=np.float32)

    ramp_percent = _clamp01(ramp_percent)
    gap_percent = _clamp01(gap_percent)

    cycle_len = 1.0 / pulse_rate
    t = np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    phase_offset_time = (phase_offset_cycles % 1.0) * cycle_len
    t_in_cycle = np.mod(t + phase_offset_time, cycle_len)

    cycle_arr = np.full_like(t_in_cycle, cycle_len)
    env = trapezoid_envelope_vectorized(
        t_in_cycle,
        cycle_arr,
        np.full_like(t_in_cycle, ramp_percent),
        np.full_like(t_in_cycle, gap_percent),
    )

    shape_key = shape.lower().strip()
    if shape_key in {"square", "pulse"}:
        env = (env > 0.0).astype(np.float64)
    elif shape_key in {"raised_cosine", "raised-cosine", "cosine", "hann"}:
        env = 0.5 - 0.5 * np.cos(np.pi * np.clip(env, 0.0, 1.0))
    else:  # default "trapezoid" keeps the generated ramps
        env = np.clip(env, 0.0, 1.0)

    if depth < 1.0:
        env = (1.0 - depth) + depth * env

    return env.astype(np.float32, copy=False)


def dual_pulse_binaural(duration, sample_rate=44100, **params):
    """Generate a dual pulse binaural signal."""

    # Allow callers to collapse the binaural difference if desired.
    binaural_kwargs: Dict[str, float] = dict(params)
    if bool(params.get("forceMono", False)):
        binaural_kwargs["beatFreq"] = 0.0

    audio = binaural_beat(duration, sample_rate, **binaural_kwargs)
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    if audio.shape[0] == 0:
        return audio.astype(np.float32)

    num_samples = audio.shape[0]
    beat_freq = float(params.get("beatFreq", 4.0))
    base_pulse_rate = float(params.get("pulseRate", 0.0))
    pulse_rate = _resolve_pulse_rate(base_pulse_rate, beat_freq)

    pulse_rate_l = _resolve_pulse_rate(params.get("pulseRateL", pulse_rate), pulse_rate)
    pulse_rate_r = _resolve_pulse_rate(params.get("pulseRateR", pulse_rate), pulse_rate)

    ramp_percent = float(params.get("rampPercent", 0.2))
    gap_percent = float(params.get("gapPercent", 0.15))

    ramp_percent_l = float(params.get("rampPercentL", ramp_percent))
    ramp_percent_r = float(params.get("rampPercentR", ramp_percent))
    gap_percent_l = float(params.get("gapPercentL", gap_percent))
    gap_percent_r = float(params.get("gapPercentR", gap_percent))

    pulse_phase_offset = float(params.get("pulsePhaseOffset", 0.0))
    pulse_phase_offset_l = float(params.get("pulsePhaseOffsetL", pulse_phase_offset))
    pulse_phase_offset_r = float(params.get("pulsePhaseOffsetR", pulse_phase_offset))

    pulse_shape = str(params.get("pulseShape", "trapezoid"))
    pulse_shape_l = str(params.get("pulseShapeL", pulse_shape))
    pulse_shape_r = str(params.get("pulseShapeR", pulse_shape))

    pulse_depth = float(params.get("pulseDepth", 1.0))
    pulse_depth_l = float(params.get("pulseDepthL", pulse_depth))
    pulse_depth_r = float(params.get("pulseDepthR", pulse_depth))

    env_l = _generate_channel_envelope(
        num_samples,
        sample_rate,
        pulse_rate_l,
        ramp_percent_l,
        gap_percent_l,
        pulse_phase_offset_l,
        pulse_shape_l,
        pulse_depth_l,
    )
    env_r = _generate_channel_envelope(
        num_samples,
        sample_rate,
        pulse_rate_r,
        ramp_percent_r,
        gap_percent_r,
        pulse_phase_offset_r,
        pulse_shape_r,
        pulse_depth_r,
    )

    audio = audio.astype(np.float32, copy=False)
    audio[:, 0] *= env_l
    audio[:, 1] *= env_r

    if bool(params.get("harmonicSuppression", False)):
        base_freq = float(params.get("baseFreq", 200.0))
        cutoff = min(base_freq * 1.5, 0.5 * sample_rate - 1.0)
        if cutoff > 0.0:
            b, a = butter(4, cutoff / (0.5 * sample_rate), btype="low")
            audio = filtfilt(b, a, audio, axis=0).astype(np.float32, copy=False)

    return audio.astype(np.float32, copy=False)


def dual_pulse_binaural_transition(
    duration,
    sample_rate=44100,
    initial_offset=0.0,
    transition_duration=None,
    **params,
):
    """Transition-capable variant of :func:`dual_pulse_binaural`."""

    binaural_kwargs = dict(params)
    binaural_kwargs.setdefault("transition_curve", params.get("transition_curve", "linear"))
    if bool(params.get("startForceMono", params.get("forceMono", False))) or bool(
        params.get("endForceMono", params.get("forceMono", False))
    ):
        binaural_kwargs.setdefault("startBeatFreq", params.get("startBeatFreq", params.get("beatFreq", 4.0)))
        binaural_kwargs.setdefault("endBeatFreq", params.get("endBeatFreq", binaural_kwargs["startBeatFreq"]))
        binaural_kwargs["startBeatFreq"] = 0.0
        binaural_kwargs["endBeatFreq"] = 0.0
        binaural_kwargs["beatFreq"] = 0.0

    audio = binaural_beat_transition(
        duration,
        sample_rate=sample_rate,
        initial_offset=initial_offset,
        transition_duration=transition_duration,
        **binaural_kwargs,
    )
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    if audio.shape[0] == 0:
        return audio.astype(np.float32)

    num_samples = audio.shape[0]

    alpha = calculate_transition_alpha(
        duration,
        sample_rate,
        initial_offset=initial_offset,
        duration=transition_duration,
        curve=params.get("transition_curve", "linear"),
    )
    if alpha.size != num_samples:
        alpha = np.linspace(0.0, 1.0, num_samples, endpoint=False)
    alpha = alpha.astype(np.float32)

    start_beat_freq = float(params.get("startBeatFreq", params.get("beatFreq", 4.0)))
    end_beat_freq = float(params.get("endBeatFreq", start_beat_freq))

    start_pulse_rate = _resolve_pulse_rate(
        params.get("startPulseRate", params.get("pulseRate", start_beat_freq)),
        start_beat_freq,
    )
    end_pulse_rate = _resolve_pulse_rate(
        params.get("endPulseRate", params.get("pulseRate", end_beat_freq)),
        end_beat_freq,
    )

    start_pulse_rate_l = _resolve_pulse_rate(
        params.get("startPulseRateL", params.get("pulseRateL", start_pulse_rate)),
        start_pulse_rate,
    )
    end_pulse_rate_l = _resolve_pulse_rate(
        params.get("endPulseRateL", params.get("pulseRateL", end_pulse_rate)),
        end_pulse_rate,
    )
    start_pulse_rate_r = _resolve_pulse_rate(
        params.get("startPulseRateR", params.get("pulseRateR", start_pulse_rate)),
        start_pulse_rate,
    )
    end_pulse_rate_r = _resolve_pulse_rate(
        params.get("endPulseRateR", params.get("pulseRateR", end_pulse_rate)),
        end_pulse_rate,
    )

    start_ramp = float(params.get("startRampPercent", params.get("rampPercent", 0.2)))
    end_ramp = float(params.get("endRampPercent", start_ramp))
    start_gap = float(params.get("startGapPercent", params.get("gapPercent", 0.15)))
    end_gap = float(params.get("endGapPercent", start_gap))

    start_ramp_l = float(params.get("startRampPercentL", params.get("rampPercentL", start_ramp)))
    end_ramp_l = float(params.get("endRampPercentL", params.get("rampPercentL", end_ramp)))
    start_ramp_r = float(params.get("startRampPercentR", params.get("rampPercentR", start_ramp)))
    end_ramp_r = float(params.get("endRampPercentR", params.get("rampPercentR", end_ramp)))

    start_gap_l = float(params.get("startGapPercentL", params.get("gapPercentL", start_gap)))
    end_gap_l = float(params.get("endGapPercentL", params.get("gapPercentL", end_gap)))
    start_gap_r = float(params.get("startGapPercentR", params.get("gapPercentR", start_gap)))
    end_gap_r = float(params.get("endGapPercentR", params.get("gapPercentR", end_gap)))

    start_phase_offset = float(params.get("startPulsePhaseOffset", params.get("pulsePhaseOffset", 0.0)))
    end_phase_offset = float(params.get("endPulsePhaseOffset", start_phase_offset))
    start_phase_offset_l = float(
        params.get("startPulsePhaseOffsetL", params.get("pulsePhaseOffsetL", start_phase_offset))
    )
    end_phase_offset_l = float(
        params.get("endPulsePhaseOffsetL", params.get("pulsePhaseOffsetL", end_phase_offset))
    )
    start_phase_offset_r = float(
        params.get("startPulsePhaseOffsetR", params.get("pulsePhaseOffsetR", start_phase_offset))
    )
    end_phase_offset_r = float(
        params.get("endPulsePhaseOffsetR", params.get("pulsePhaseOffsetR", end_phase_offset))
    )

    start_pulse_shape = str(params.get("startPulseShape", params.get("pulseShape", "trapezoid")))
    end_pulse_shape = str(params.get("endPulseShape", start_pulse_shape))
    start_pulse_shape_l = str(params.get("startPulseShapeL", params.get("pulseShapeL", start_pulse_shape)))
    end_pulse_shape_l = str(params.get("endPulseShapeL", params.get("pulseShapeL", end_pulse_shape)))
    start_pulse_shape_r = str(params.get("startPulseShapeR", params.get("pulseShapeR", start_pulse_shape)))
    end_pulse_shape_r = str(params.get("endPulseShapeR", params.get("pulseShapeR", end_pulse_shape)))

    start_depth = float(params.get("startPulseDepth", params.get("pulseDepth", 1.0)))
    end_depth = float(params.get("endPulseDepth", start_depth))
    start_depth_l = float(params.get("startPulseDepthL", params.get("pulseDepthL", start_depth)))
    end_depth_l = float(params.get("endPulseDepthL", params.get("pulseDepthL", end_depth)))
    start_depth_r = float(params.get("startPulseDepthR", params.get("pulseDepthR", start_depth)))
    end_depth_r = float(params.get("endPulseDepthR", params.get("pulseDepthR", end_depth)))

    env_start_l = _generate_channel_envelope(
        num_samples,
        sample_rate,
        start_pulse_rate_l,
        start_ramp_l,
        start_gap_l,
        start_phase_offset_l,
        start_pulse_shape_l,
        start_depth_l,
    )
    env_end_l = _generate_channel_envelope(
        num_samples,
        sample_rate,
        end_pulse_rate_l,
        end_ramp_l,
        end_gap_l,
        end_phase_offset_l,
        end_pulse_shape_l,
        end_depth_l,
    )
    env_start_r = _generate_channel_envelope(
        num_samples,
        sample_rate,
        start_pulse_rate_r,
        start_ramp_r,
        start_gap_r,
        start_phase_offset_r,
        start_pulse_shape_r,
        start_depth_r,
    )
    env_end_r = _generate_channel_envelope(
        num_samples,
        sample_rate,
        end_pulse_rate_r,
        end_ramp_r,
        end_gap_r,
        end_phase_offset_r,
        end_pulse_shape_r,
        end_depth_r,
    )

    blend = alpha.astype(np.float32, copy=False)
    one = np.float32(1.0)
    env_l = env_start_l * (one - blend) + env_end_l * blend
    env_r = env_start_r * (one - blend) + env_end_r * blend
    env_l = env_l.astype(np.float32, copy=False)
    env_r = env_r.astype(np.float32, copy=False)

    audio = audio.astype(np.float32, copy=False)
    audio[:, 0] *= env_l
    audio[:, 1] *= env_r

    start_harm = bool(params.get("startHarmonicSuppression", params.get("harmonicSuppression", False)))
    end_harm = bool(params.get("endHarmonicSuppression", start_harm))
    if start_harm or end_harm:
        base_start = float(params.get("startBaseFreq", params.get("baseFreq", 200.0)))
        base_end = float(params.get("endBaseFreq", base_start))
        avg_base = max(0.0, 0.5 * (base_start + base_end))
        cutoff = min(avg_base * 1.5, 0.5 * sample_rate - 1.0)
        if cutoff > 0.0:
            b, a = butter(4, cutoff / (0.5 * sample_rate), btype="low")
            audio = filtfilt(b, a, audio, axis=0).astype(np.float32, copy=False)

    return audio.astype(np.float32, copy=False)

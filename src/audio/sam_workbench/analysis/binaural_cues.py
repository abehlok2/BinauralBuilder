"""Binaural cue analysis measured from rendered audio.

`dsp.source` derives instantaneous frequency and IPD from the *model*; this
module measures them from a rendered buffer instead. The two must agree - a
disagreement means the renderer is not doing what the specification says - and
the analysis views show what a listener would actually receive.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from ..conventions import CHANNEL_COUNT, SILENCE_DB, linear_to_db
from ..waveforms import TWO_PI

__all__ = [
    "analytic_phase",
    "instantaneous_frequency_hz",
    "interaural_phase_difference_rad",
    "interaural_level_difference_db",
    "estimate_itd_seconds",
    "summarize_cues",
    "unwrap_phase",
]

_EPSILON = 1e-12


def _as_channel_major(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected stereo audio, got shape {array.shape}")
    if array.shape[0] == CHANNEL_COUNT:
        return array
    if array.shape[1] == CHANNEL_COUNT:
        return np.ascontiguousarray(array.T)
    raise ValueError(f"expected {CHANNEL_COUNT} channels, got shape {array.shape}")


def analytic_phase(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    """Unwrapped analytic-signal phase per channel, in radians."""

    block = _as_channel_major(audio)
    if block.shape[1] < 2:
        return np.zeros_like(block)
    analytic = scipy_signal.hilbert(block, axis=1)
    return np.unwrap(np.angle(analytic), axis=1)


def unwrap_phase(phase_rad: NDArray[np.floating]) -> NDArray[np.float64]:
    """Unwrap a phase array along its last axis."""

    return np.unwrap(np.asarray(phase_rad, dtype=np.float64), axis=-1)


def instantaneous_frequency_hz(
    audio: NDArray[np.floating], sample_rate: float, *, smoothing_frames: int = 0
) -> NDArray[np.float64]:
    """Per-ear instantaneous frequency in Hz, measured from the audio.

    Optional smoothing is a plain moving average; it is off by default so the
    measurement stays faithful to what was rendered.
    """

    phase = analytic_phase(audio)
    if phase.shape[1] < 2:
        return np.zeros_like(phase)
    frequency = np.gradient(phase, 1.0 / float(sample_rate), axis=1) / TWO_PI
    if smoothing_frames and smoothing_frames > 1:
        kernel = np.ones(int(smoothing_frames)) / float(smoothing_frames)
        frequency = np.vstack(
            [np.convolve(channel, kernel, mode="same") for channel in frequency]
        )
    return frequency


def interaural_phase_difference_rad(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    """Instantaneous IPD in radians, left minus right, wrapped to ``(-pi, pi]``."""

    phase = analytic_phase(audio)
    if phase.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    difference = phase[0] - phase[1]
    return np.angle(np.exp(1j * difference))


def interaural_level_difference_db(
    audio: NDArray[np.floating], sample_rate: float, *, window_ms: float = 20.0
) -> NDArray[np.float64]:
    """Interaural level difference over time in dB, left minus right.

    Computed from short-window RMS, because an instantaneous sample ratio is
    meaningless wherever either channel crosses zero.
    """

    block = _as_channel_major(audio)
    frames = block.shape[1]
    if frames == 0:
        return np.zeros(0, dtype=np.float64)
    window_frames = max(1, int(round(window_ms * sample_rate / 1000.0)))
    kernel = np.ones(window_frames) / float(window_frames)
    energy = np.vstack([np.convolve(np.square(channel), kernel, mode="same") for channel in block])
    levels = np.sqrt(np.maximum(energy, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(levels[1] > _EPSILON, levels[0] / np.maximum(levels[1], _EPSILON), np.nan)
    difference = 20.0 * np.log10(np.maximum(ratio, _EPSILON))
    return np.nan_to_num(difference, nan=0.0, posinf=-SILENCE_DB, neginf=SILENCE_DB)


def estimate_itd_seconds(
    audio: NDArray[np.floating], sample_rate: float, *, maximum_ms: float = 1.5
) -> float:
    """Estimate the broadband interaural time difference by cross-correlation.

    A positive result means the left channel leads. The search is bounded to a
    physically plausible range so an unrelated correlation peak cannot win.
    """

    block = _as_channel_major(audio)
    if block.shape[1] < 4:
        return 0.0
    left = block[0] - block[0].mean()
    right = block[1] - block[1].mean()
    if np.allclose(left, 0.0) or np.allclose(right, 0.0):
        return 0.0

    maximum_lag = max(1, int(round(maximum_ms * sample_rate / 1000.0)))
    correlation = scipy_signal.correlate(left, right, mode="full")
    lags = scipy_signal.correlation_lags(left.size, right.size, mode="full")
    inside = np.abs(lags) <= maximum_lag
    if not np.any(inside):
        return 0.0
    best = int(lags[inside][int(np.argmax(correlation[inside]))])
    return best / float(sample_rate)


def summarize_cues(
    audio: NDArray[np.floating], sample_rate: float, *, edge_trim_frames: int = 0
) -> dict[str, float]:
    """A compact numeric summary for a report or a status line.

    ``edge_trim_frames`` drops that many samples from each end before
    measuring. The analytic signal rings at a buffer's edges, so without a trim
    the reported frequency range is dominated by an artefact rather than by the
    signal.
    """

    block = _as_channel_major(audio)
    trim = int(edge_trim_frames)
    if trim > 0 and block.shape[1] > 2 * trim:
        block = block[:, trim:-trim]
    ipd = interaural_phase_difference_rad(block)
    ild = interaural_level_difference_db(block, sample_rate)
    frequency = instantaneous_frequency_hz(block, sample_rate)
    peaks = np.max(np.abs(block), axis=1) if block.shape[1] else np.zeros(2)
    return {
        "ipd_rad_min": float(np.min(ipd)) if ipd.size else 0.0,
        "ipd_rad_max": float(np.max(ipd)) if ipd.size else 0.0,
        "ild_db_min": float(np.min(ild)) if ild.size else 0.0,
        "ild_db_max": float(np.max(ild)) if ild.size else 0.0,
        "itd_s": estimate_itd_seconds(block, sample_rate),
        "frequency_hz_min": float(np.min(frequency)) if frequency.size else 0.0,
        "frequency_hz_max": float(np.max(frequency)) if frequency.size else 0.0,
        "peak_left_dbfs": linear_to_db(float(peaks[0])),
        "peak_right_dbfs": linear_to_db(float(peaks[1])),
    }

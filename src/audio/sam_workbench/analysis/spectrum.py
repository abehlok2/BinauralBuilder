"""Spectrum and spectrogram analysis of a rendered buffer."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from ..conventions import SILENCE_DB

__all__ = ["magnitude_spectrum_db", "spectrogram_db"]

_EPSILON = 1e-12


def _as_channel_major(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim == 2 and array.shape[0] != 2 and array.shape[1] == 2:
        return np.ascontiguousarray(array.T)
    return array


def magnitude_spectrum_db(
    audio: NDArray[np.floating], sample_rate: float, *, floor_db: float = -120.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(frequencies_hz, magnitude_db)`` with one row per channel.

    A Hann window is applied and the result is normalized by the window's sum,
    so a full-scale sine reads near 0 dBFS regardless of the analysis length.
    """

    block = _as_channel_major(audio)
    channels, frames = block.shape
    if frames < 2:
        return np.zeros(0, dtype=np.float64), np.zeros((channels, 0), dtype=np.float64)

    window = np.hanning(frames)
    scale = np.sum(window) / 2.0
    spectrum = np.fft.rfft(block * window, axis=1)
    magnitude = np.abs(spectrum) / max(scale, _EPSILON)
    decibels = 20.0 * np.log10(np.maximum(magnitude, _EPSILON))
    frequencies = np.fft.rfftfreq(frames, d=1.0 / float(sample_rate))
    return frequencies, np.maximum(decibels, max(floor_db, SILENCE_DB))


def spectrogram_db(
    audio: NDArray[np.floating],
    sample_rate: float,
    *,
    window_frames: int = 1024,
    overlap: float = 0.5,
    floor_db: float = -120.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(frequencies_hz, times_s, magnitude_db)`` for the summed channels."""

    block = _as_channel_major(audio)
    mono = block.mean(axis=0)
    window_frames = int(min(max(16, window_frames), max(16, mono.size)))
    if mono.size < window_frames:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )
    noverlap = int(window_frames * float(np.clip(overlap, 0.0, 0.95)))
    frequencies, times, values = scipy_signal.spectrogram(
        mono,
        fs=float(sample_rate),
        window="hann",
        nperseg=window_frames,
        noverlap=noverlap,
        mode="magnitude",
    )
    decibels = 20.0 * np.log10(np.maximum(values, _EPSILON))
    return frequencies, times, np.maximum(decibels, max(floor_db, SILENCE_DB))

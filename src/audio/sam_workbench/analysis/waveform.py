"""Waveform and level analysis of a rendered buffer.

These functions summarize audio for display: a plot has a few hundred pixels,
not a few hundred thousand samples, and drawing every sample is both slow and
misleading. The envelope keeps the extremes of each bucket so a transient
cannot disappear between pixels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, linear_to_db

__all__ = ["peak_envelope", "rms_envelope", "peak_level_db", "channel_peaks"]


def _as_channel_major(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"expected mono or stereo audio, got shape {array.shape}")
    if array.shape[0] == CHANNEL_COUNT:
        return array
    if array.shape[1] == CHANNEL_COUNT:
        return np.ascontiguousarray(array.T)
    raise ValueError(f"expected {CHANNEL_COUNT} channels, got shape {array.shape}")


def peak_envelope(audio: NDArray[np.floating], buckets: int = 512) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(minimum, maximum)`` per bucket, each ``(channels, buckets)``.

    Keeping both extremes is what makes a downsampled waveform honest: a single
    loud sample still reaches the top of its bucket.
    """

    block = _as_channel_major(audio)
    channels, frames = block.shape
    buckets = max(1, int(buckets))
    if frames == 0:
        empty = np.zeros((channels, 0), dtype=np.float64)
        return empty, empty.copy()
    if frames <= buckets:
        return block.copy(), block.copy()

    edges = np.linspace(0, frames, buckets + 1).astype(int)
    minimum = np.empty((channels, buckets), dtype=np.float64)
    maximum = np.empty((channels, buckets), dtype=np.float64)
    for index in range(buckets):
        start, stop = edges[index], max(edges[index] + 1, edges[index + 1])
        window = block[:, start:stop]
        minimum[:, index] = window.min(axis=1)
        maximum[:, index] = window.max(axis=1)
    return minimum, maximum


def rms_envelope(audio: NDArray[np.floating], buckets: int = 512) -> NDArray[np.float64]:
    """Root-mean-square level per bucket, ``(channels, buckets)``."""

    block = _as_channel_major(audio)
    channels, frames = block.shape
    buckets = max(1, int(buckets))
    if frames == 0:
        return np.zeros((channels, 0), dtype=np.float64)
    edges = np.linspace(0, frames, buckets + 1).astype(int)
    levels = np.empty((channels, buckets), dtype=np.float64)
    for index in range(buckets):
        start, stop = edges[index], max(edges[index] + 1, edges[index + 1])
        window = block[:, start:stop]
        levels[:, index] = np.sqrt(np.mean(np.square(window), axis=1))
    return levels


def channel_peaks(audio: NDArray[np.floating]) -> NDArray[np.float64]:
    """Peak absolute value per channel."""

    block = _as_channel_major(audio)
    if block.shape[1] == 0:
        return np.zeros(block.shape[0], dtype=np.float64)
    return np.max(np.abs(block), axis=1)


def peak_level_db(audio: NDArray[np.floating]) -> float:
    """Overall peak in dBFS."""

    peaks = channel_peaks(audio)
    return linear_to_db(float(peaks.max()) if peaks.size else 0.0)

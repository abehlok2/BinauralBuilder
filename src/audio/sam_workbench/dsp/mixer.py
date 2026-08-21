"""Stem mixing and shared output gain.

One rule dominates this module: **both ears always share one gain**. A per-ear
or per-direction normalization would change the very interaural cues the
workbench exists to control, so no function here accepts a per-channel gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, db_to_linear

__all__ = ["mix_stems", "apply_shared_gain", "peak_level", "shared_normalization_gain"]


def mix_stems(stems: tuple[NDArray[np.floating], ...], frames: int) -> NDArray[np.float64]:
    """Sum channel-major stereo stems into one ``(2, frames)`` buffer."""

    mixed = np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
    for stem in stems:
        block = np.asarray(stem, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != CHANNEL_COUNT:
            raise ValueError(f"expected ({CHANNEL_COUNT}, frames) stems, got {block.shape}")
        length = min(frames, block.shape[1])
        mixed[:, :length] += block[:, :length]
    return mixed


def apply_shared_gain(audio: NDArray[np.floating], gain_db: float) -> NDArray[np.float64]:
    """Apply one gain to every channel."""

    return np.asarray(audio, dtype=np.float64) * db_to_linear(gain_db)


def peak_level(audio: NDArray[np.floating]) -> float:
    """Largest absolute sample value across all channels."""

    array = np.asarray(audio)
    return float(np.max(np.abs(array))) if array.size else 0.0


def shared_normalization_gain(audio: NDArray[np.floating], target_peak: float) -> float:
    """Return the single gain that brings the loudest channel to ``target_peak``.

    Returned rather than applied, so the caller can record it in the manifest.
    """

    peak = peak_level(audio)
    if peak <= 0.0 or target_peak <= 0.0:
        return 1.0
    return float(target_peak) / peak

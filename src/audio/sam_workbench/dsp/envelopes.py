"""Amplitude envelopes and click-free crossfades.

A jump that is intentional at the control level still needs a short audio
crossfade, otherwise it arrives as a click. Equal-power curves are used for
crossfades between two correlated signals; linear curves are used for fades to
and from silence.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.conventions import CHANNEL_COUNT

__all__ = [
    "DEFAULT_CROSSFADE_MS",
    "equal_power_crossfade",
    "linear_fade",
    "apply_fade_in",
    "apply_fade_out",
    "crossfade_blocks",
    "milliseconds_to_frames",
]

#: Short enough to stay inaudible as a fade, long enough to hide a jump.
DEFAULT_CROSSFADE_MS = 8.0


def milliseconds_to_frames(milliseconds: float, sample_rate: float) -> int:
    """Convert a duration in milliseconds to a whole number of frames."""

    return max(0, int(round(max(0.0, float(milliseconds)) * float(sample_rate) / 1000.0)))


def linear_fade(frames: int, *, rising: bool = True) -> NDArray[np.float64]:
    """A linear gain ramp over ``frames`` samples."""

    if frames <= 0:
        return np.zeros(0, dtype=np.float64)
    if frames == 1:
        return np.ones(1, dtype=np.float64) if rising else np.zeros(1, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    return ramp if rising else 1.0 - ramp


def equal_power_crossfade(frames: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(fade_out, fade_in)`` gains whose squares sum to one."""

    if frames <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy()
    position = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    angle = position * (math.pi / 2.0)
    return np.cos(angle), np.sin(angle)


def _fade_view(audio: NDArray[np.floating], frames: int) -> slice:
    length = audio.shape[-1]
    return slice(0, min(frames, length))


def apply_fade_in(audio: NDArray[np.floating], frames: int) -> NDArray[np.floating]:
    """Fade ``audio`` (channel-major) in over its first ``frames`` samples, in place."""

    window = _fade_view(audio, frames)
    span = window.stop - window.start
    if span > 0:
        audio[..., window] *= linear_fade(span, rising=True)
    return audio


def apply_fade_out(audio: NDArray[np.floating], frames: int) -> NDArray[np.floating]:
    """Fade ``audio`` (channel-major) out over its last ``frames`` samples, in place."""

    length = audio.shape[-1]
    span = min(frames, length)
    if span > 0:
        audio[..., length - span :] *= linear_fade(span, rising=False)
    return audio


def crossfade_blocks(
    outgoing: NDArray[np.floating], incoming: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Equal-power crossfade between two aligned channel-major blocks."""

    if outgoing.shape != incoming.shape:
        raise ValueError(f"shapes differ: {outgoing.shape} vs {incoming.shape}")
    if outgoing.ndim != 2 or outgoing.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected ({CHANNEL_COUNT}, frames) blocks, got {outgoing.shape}")
    fade_out, fade_in = equal_power_crossfade(outgoing.shape[1])
    return outgoing * fade_out + incoming * fade_in

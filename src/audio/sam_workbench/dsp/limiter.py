"""Output safety: a look-ahead limiter for export and a cheap safety clamp.

Both limiters apply one gain envelope to every channel simultaneously. Limiting
each ear independently would change ILD, so the gain is always computed from
the loudest channel and shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d

from ..conventions import CHANNEL_COUNT, DEFAULT_LIMITER_CEILING_DBFS, db_to_linear, linear_to_db

__all__ = ["LimiterReport", "limit_lookahead", "safety_clamp", "true_peak_estimate_db"]

DEFAULT_LOOKAHEAD_MS = 5.0
DEFAULT_RELEASE_MS = 50.0


@dataclass(frozen=True)
class LimiterReport:
    """What the limiter did, for the render manifest."""

    enabled: bool
    ceiling_dbfs: float
    lookahead_ms: float
    release_ms: float
    max_gain_reduction_db: float
    peak_dbfs_before: float
    peak_dbfs_after: float
    clipped_samples: int


def _shared_envelope(audio: NDArray[np.float64], ceiling_linear: float) -> NDArray[np.float64]:
    magnitude = np.max(np.abs(audio), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        required = np.where(magnitude > ceiling_linear, ceiling_linear / magnitude, 1.0)
    return required


def limit_lookahead(
    audio: NDArray[np.floating],
    sample_rate: float,
    *,
    ceiling_dbfs: float = DEFAULT_LIMITER_CEILING_DBFS,
    lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
) -> tuple[NDArray[np.float64], LimiterReport]:
    """Look-ahead limit a channel-major buffer and report what it cost.

    The gain envelope is the running minimum over the look-ahead window,
    smoothed by a one-pole release, so gain never rises faster than the release
    time and never falls late.
    """

    block = np.asarray(audio, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected ({CHANNEL_COUNT}, frames) audio, got {block.shape}")

    ceiling_linear = db_to_linear(ceiling_dbfs)
    peak_before = float(np.max(np.abs(block))) if block.size else 0.0
    if block.size == 0 or peak_before <= ceiling_linear:
        return block, LimiterReport(
            enabled=True,
            ceiling_dbfs=float(ceiling_dbfs),
            lookahead_ms=float(lookahead_ms),
            release_ms=float(release_ms),
            max_gain_reduction_db=0.0,
            peak_dbfs_before=linear_to_db(peak_before),
            peak_dbfs_after=linear_to_db(peak_before),
            clipped_samples=0,
        )

    required = _shared_envelope(block, ceiling_linear)
    lookahead_frames = max(1, int(round(max(0.0, lookahead_ms) * sample_rate / 1000.0)))
    # Forward-looking running minimum: each sample already carries the tightest
    # gain demanded within the next `lookahead_frames` samples, so the limiter
    # never arrives late at a transient.
    window = lookahead_frames + 1
    envelope = minimum_filter1d(required, size=window, origin=-(window // 2), mode="nearest")

    release_coefficient = float(np.exp(-1.0 / max(1e-9, (release_ms / 1000.0) * sample_rate)))
    smoothed = np.empty_like(envelope)
    gain = 1.0
    for index in range(envelope.size):
        target = envelope[index]
        gain = target if target < gain else target + release_coefficient * (gain - target)
        smoothed[index] = gain

    limited = block * smoothed
    peak_after = float(np.max(np.abs(limited)))
    return limited, LimiterReport(
        enabled=True,
        ceiling_dbfs=float(ceiling_dbfs),
        lookahead_ms=float(lookahead_ms),
        release_ms=float(release_ms),
        max_gain_reduction_db=abs(linear_to_db(float(np.min(smoothed)))),
        peak_dbfs_before=linear_to_db(peak_before),
        peak_dbfs_after=linear_to_db(peak_after),
        clipped_samples=int(np.count_nonzero(np.abs(limited) > 1.0)),
    )


def safety_clamp(
    audio: NDArray[np.floating], ceiling_dbfs: float = DEFAULT_LIMITER_CEILING_DBFS
) -> tuple[NDArray[np.float64], LimiterReport]:
    """Hard-clamp a preview buffer to the ceiling with one shared gain per sample."""

    block = np.asarray(audio, dtype=np.float64)
    ceiling_linear = db_to_linear(ceiling_dbfs)
    peak_before = float(np.max(np.abs(block))) if block.size else 0.0
    envelope = _shared_envelope(block, ceiling_linear) if block.size else np.ones(0)
    clamped = block * envelope if block.size else block
    peak_after = float(np.max(np.abs(clamped))) if clamped.size else 0.0
    return clamped, LimiterReport(
        enabled=True,
        ceiling_dbfs=float(ceiling_dbfs),
        lookahead_ms=0.0,
        release_ms=0.0,
        max_gain_reduction_db=abs(linear_to_db(float(np.min(envelope)))) if envelope.size else 0.0,
        peak_dbfs_before=linear_to_db(peak_before),
        peak_dbfs_after=linear_to_db(peak_after),
        clipped_samples=0,
    )


def true_peak_estimate_db(audio: NDArray[np.floating], oversampling: int = 4) -> float:
    """Estimate true peak by linear oversampling.

    An approximation, deliberately: it is reported as a warning aid, not as a
    certified inter-sample peak measurement.
    """

    block = np.asarray(audio, dtype=np.float64)
    if block.size == 0:
        return linear_to_db(0.0)
    frames = block.shape[-1]
    if frames < 2 or oversampling <= 1:
        return linear_to_db(float(np.max(np.abs(block))))
    positions = np.linspace(0.0, frames - 1, (frames - 1) * int(oversampling) + 1)
    peak = 0.0
    for channel in range(block.shape[0]):
        interpolated = np.interp(positions, np.arange(frames), block[channel])
        peak = max(peak, float(np.max(np.abs(interpolated))))
    return linear_to_db(peak)

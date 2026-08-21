"""Numerical trajectory and geometric-cue series for analysis plots."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.audio.sam_workbench.conventions import DEFAULT_SPEED_OF_SOUND_M_S
from src.audio.sam_workbench.render.geometric import geometric_cues
from src.audio.sam_workbench.trajectory.transforms import ListenerTransform

__all__ = ["sample_trajectory", "trajectory_metrics"]


def sample_trajectory(trajectory: object, times_s: ArrayLike) -> NDArray[np.float64]:
    """Evaluate any canonical trajectory for plotting on a declared time axis."""

    times = np.asarray(times_s, dtype=np.float64)
    positions = trajectory(times) if callable(trajectory) else trajectory.evaluate(times)
    result = np.asarray(positions, dtype=np.float64)
    if result.shape != times.shape + (3,):
        raise ValueError(f"trajectory returned shape {result.shape}, expected {times.shape + (3,)}")
    return result


def trajectory_metrics(
    positions_m: ArrayLike,
    sample_rate_hz: float,
    listener: ListenerTransform | None = None,
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
) -> dict[str, NDArray[np.float64]]:
    """Calculate position, velocity, speed, distance, ITD, and geometric ILD.

    ILD follows the renderer's inverse-distance amplitude law and is reported
    as left minus right: ``20 log10(right_distance / left_distance)``.
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape (frames, 3), got {positions.shape}")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    velocity = (
        np.gradient(positions, 1.0 / float(sample_rate_hz), axis=0)
        if len(positions) > 1
        else np.zeros_like(positions)
    )
    cues = geometric_cues(
        positions,
        listener or ListenerTransform(),
        speed_of_sound_m_s,
    )
    distances = cues["distance_m"]
    cues.update(
        {
            "position_m": positions,
            "velocity_m_s": velocity,
            "speed_m_s": np.linalg.norm(velocity, axis=-1),
            "ild_db": 20.0
            * np.log10(
                np.maximum(distances[..., 1], np.finfo(float).tiny)
                / np.maximum(distances[..., 0], np.finfo(float).tiny)
            ),
        }
    )
    return cues

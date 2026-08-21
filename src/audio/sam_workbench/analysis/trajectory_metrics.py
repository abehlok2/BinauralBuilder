"""Metrics for a trajectory and the binaural cues it implies.

Two kinds of number live here:

* **path metrics** - length, speed, acceleration, curvature - which say whether
  a path is smooth, how fast a source crosses it, and where it turns hardest;
* **predicted cues** - ITD, ILD, distance, azimuth - computed from the geometry
  *before* anything is rendered.

The predicted cues are what make the analysis views trustworthy: a rendered
buffer can be measured with :mod:`binaural_cues`, and the two must agree. A
disagreement means the renderer is not following the geometry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, cartesian_to_spherical_deg, linear_to_db

__all__ = [
    "path_length_m",
    "speeds_m_s",
    "curvature_1_m",
    "predicted_ear_distances_m",
    "predicted_itd_s",
    "predicted_ild_db",
    "trajectory_summary",
]

_EPSILON = 1e-12


def _as_positions(positions_m: NDArray[np.floating]) -> NDArray[np.float64]:
    block = np.asarray(positions_m, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != 3:
        raise ValueError(f"expected (3, n) positions in metres, got shape {block.shape}")
    return block


def path_length_m(positions_m: NDArray[np.floating]) -> float:
    """Total distance travelled along a sampled path, in metres."""

    block = _as_positions(positions_m)
    if block.shape[1] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(block, axis=1), axis=0)))


def speeds_m_s(positions_m: NDArray[np.floating], sample_rate: float) -> NDArray[np.float64]:
    """Instantaneous speed at each sample, in metres per second."""

    block = _as_positions(positions_m)
    if block.shape[1] < 2:
        return np.zeros(block.shape[1], dtype=np.float64)
    velocity = np.gradient(block, 1.0 / float(sample_rate), axis=1)
    return np.linalg.norm(velocity, axis=0)


def curvature_1_m(positions_m: NDArray[np.floating], sample_rate: float) -> NDArray[np.float64]:
    """Curvature at each sample in inverse metres, using the cross-product form.

    A straight line reads zero; a circle of radius ``r`` reads ``1 / r``. Large
    values mark the corners a listener will hear as sudden direction changes.
    """

    block = _as_positions(positions_m)
    if block.shape[1] < 3:
        return np.zeros(block.shape[1], dtype=np.float64)
    step = 1.0 / float(sample_rate)
    velocity = np.gradient(block, step, axis=1)
    acceleration = np.gradient(velocity, step, axis=1)
    cross = np.cross(velocity.T, acceleration.T).T
    speed = np.linalg.norm(velocity, axis=0)
    numerator = np.linalg.norm(cross, axis=0)
    return np.where(speed > _EPSILON, numerator / np.maximum(speed**3, _EPSILON), 0.0)


def predicted_ear_distances_m(
    positions_m: NDArray[np.floating], head_radius_m: float = 0.0875
) -> NDArray[np.float64]:
    """Distance from each ear to the source, ``(2, n)``, left ear first."""

    block = _as_positions(positions_m)
    ears = np.array(
        [[0.0, 0.0], [float(head_radius_m), -float(head_radius_m)], [0.0, 0.0]], dtype=np.float64
    )
    return np.vstack(
        [np.linalg.norm(block - ears[:, ear][:, None], axis=0) for ear in range(CHANNEL_COUNT)]
    )


def predicted_itd_s(
    positions_m: NDArray[np.floating],
    *,
    head_radius_m: float = 0.0875,
    speed_of_sound_m_s: float = 343.0,
) -> NDArray[np.float64]:
    """ITD implied by the geometry, in seconds; positive when the left ear leads."""

    distances = predicted_ear_distances_m(positions_m, head_radius_m)
    return (distances[1] - distances[0]) / float(speed_of_sound_m_s)


def predicted_ild_db(
    positions_m: NDArray[np.floating],
    *,
    head_radius_m: float = 0.0875,
    distance_exponent: float = 1.0,
    minimum_distance_m: float = 0.15,
) -> NDArray[np.float64]:
    """ILD implied by the distance law alone, in dB, left minus right.

    Distance is the only mechanism here: a geometric renderer with no head
    shadow produces exactly this, and the difference between this and a
    measured ILD is what the shadow filter contributes.
    """

    distances = np.maximum(
        predicted_ear_distances_m(positions_m, head_radius_m), float(minimum_distance_m)
    )
    ratio = (distances[1] / distances[0]) ** float(distance_exponent)
    return 20.0 * np.log10(np.maximum(ratio, _EPSILON))


def trajectory_summary(
    positions_m: NDArray[np.floating],
    sample_rate: float,
    *,
    head_radius_m: float = 0.0875,
    speed_of_sound_m_s: float = 343.0,
) -> dict[str, float]:
    """A compact numeric summary for a status line or a render report."""

    block = _as_positions(positions_m)
    if block.shape[1] == 0:
        return {}
    spherical = np.array(
        [cartesian_to_spherical_deg(block[:, index]) for index in range(block.shape[1])],
        dtype=np.float64,
    ).T
    speeds = speeds_m_s(block, sample_rate)
    itd = predicted_itd_s(block, head_radius_m=head_radius_m, speed_of_sound_m_s=speed_of_sound_m_s)
    ild = predicted_ild_db(block, head_radius_m=head_radius_m)
    return {
        "path_length_m": path_length_m(block),
        "speed_m_s_mean": float(np.mean(speeds)),
        "speed_m_s_max": float(np.max(speeds)),
        "distance_m_min": float(np.min(spherical[2])),
        "distance_m_max": float(np.max(spherical[2])),
        "azimuth_deg_min": float(np.min(spherical[0])),
        "azimuth_deg_max": float(np.max(spherical[0])),
        "elevation_deg_min": float(np.min(spherical[1])),
        "elevation_deg_max": float(np.max(spherical[1])),
        "itd_s_min": float(np.min(itd)),
        "itd_s_max": float(np.max(itd)),
        "ild_db_min": float(np.min(ild)),
        "ild_db_max": float(np.max(ild)),
        "level_db_at_nearest": linear_to_db(1.0 / max(float(np.min(spherical[2])), _EPSILON)),
    }

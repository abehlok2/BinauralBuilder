"""Numeric and coordinate conventions shared by all SAM renderers."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

DEFAULT_SAMPLE_RATE_HZ: Final = 44_100
DEFAULT_PREVIEW_BLOCK_FRAMES: Final = 512
DEFAULT_OFFLINE_BLOCK_FRAMES: Final = 4096
DEFAULT_SOUND_SPEED_M_S: Final = 343.0
DEFAULT_MASTER_GAIN_DB: Final = -6.0
DEFAULT_LIMITER_CEILING_DBFS: Final = -1.0

# Internal Cartesian coordinates are right-handed: +x front, +y left, +z up.
LEFT_EAR_AXIS: Final = np.array((0.0, 1.0, 0.0), dtype=np.float64)
RIGHT_EAR_AXIS: Final = np.array((0.0, -1.0, 0.0), dtype=np.float64)


def spherical_to_cartesian_m(
    azimuth_deg: float, elevation_deg: float, radius_m: float
) -> npt.NDArray[np.float64]:
    """Convert front-zero, left-positive spherical coordinates to Cartesian."""
    if not all(math.isfinite(value) for value in (azimuth_deg, elevation_deg, radius_m)):
        raise ValueError("coordinates must be finite")
    if radius_m < 0.0:
        raise ValueError("radius_m must be non-negative")
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)
    horizontal_m = radius_m * math.cos(elevation_rad)
    return np.array(
        (
            horizontal_m * math.cos(azimuth_rad),
            horizontal_m * math.sin(azimuth_rad),
            radius_m * math.sin(elevation_rad),
        ),
        dtype=np.float64,
    )


def cartesian_to_spherical_deg(
    position_m: npt.ArrayLike,
) -> tuple[float, float, float]:
    """Return ``(azimuth_deg, elevation_deg, radius_m)`` in app convention."""
    vector = np.asarray(position_m, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("position_m must contain three finite coordinates")
    radius_m = float(np.linalg.norm(vector))
    if radius_m == 0.0:
        return 0.0, 0.0, 0.0
    azimuth_deg = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    elevation_deg = math.degrees(math.asin(float(vector[2]) / radius_m))
    return azimuth_deg, elevation_deg, radius_m


def db_to_linear(gain_db: float) -> float:
    """Convert an amplitude gain in dB to a linear multiplier."""
    return 10.0 ** (gain_db / 20.0)


def linear_to_db(gain_linear: float) -> float:
    """Convert a positive linear amplitude multiplier to dB."""
    if not math.isfinite(gain_linear) or gain_linear <= 0.0:
        raise ValueError("gain_linear must be finite and greater than zero")
    return 20.0 * math.log10(gain_linear)


def wrap_phase_rad(phase_rad: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap phase to the half-open interval ``[-pi, pi)`` using float64."""
    phase = np.asarray(phase_rad, dtype=np.float64)
    return (phase + math.pi) % (2.0 * math.pi) - math.pi

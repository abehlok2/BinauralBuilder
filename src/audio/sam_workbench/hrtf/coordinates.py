"""SOFA coordinate conversion into the canonical +x forward/+y left/+z up frame."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _unit_scale(unit: str) -> float:
    value = unit.strip().lower()
    if value in {"metre", "meter", "metres", "meters", "m"}:
        return 1.0
    if value in {"centimetre", "centimeter", "cm"}:
        return 0.01
    if value in {"millimetre", "millimeter", "mm"}:
        return 0.001
    raise ValueError(f"unsupported SOFA distance unit {unit!r}")


def sofa_positions_to_cartesian(
    positions: ArrayLike, coordinate_type: str, units: str
) -> NDArray[np.float64]:
    """Convert SOFA positions to canonical Cartesian metres.

    SOFA spherical coordinates are azimuth (positive left), elevation, and
    radius. Cartesian SimpleFreeFieldHRIR coordinates share the canonical axes.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("SOFA positions must have shape (measurements, 3) and be finite")
    parts = [part.strip() for part in units.replace(";", ",").split(",")]
    kind = coordinate_type.strip().lower()
    if kind == "spherical":
        if len(parts) != 3 or not parts[0].lower().startswith("degree") or not parts[1].lower().startswith("degree"):
            raise ValueError("spherical SOFA positions require degree, degree, distance units")
        azimuth, elevation = np.radians(values[:, 0]), np.radians(values[:, 1])
        radius = values[:, 2] * _unit_scale(parts[2])
        cos_elevation = np.cos(elevation)
        return np.column_stack((
            radius * cos_elevation * np.cos(azimuth),
            radius * cos_elevation * np.sin(azimuth),
            radius * np.sin(elevation),
        ))
    if kind == "cartesian":
        scale = _unit_scale(parts[0] if parts else units)
        return values * scale
    raise ValueError(f"unsupported SOFA coordinate type {coordinate_type!r}")


def cartesian_to_sofa_spherical(points_m: ArrayLike) -> NDArray[np.float64]:
    """Return azimuth degrees, elevation degrees, radius metres."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("points must end in three coordinates")
    radius = np.linalg.norm(points, axis=-1)
    safe = np.where(radius > 0.0, radius, 1.0)
    azimuth = np.degrees(np.arctan2(points[..., 1], points[..., 0]))
    elevation = np.degrees(np.arcsin(np.clip(points[..., 2] / safe, -1.0, 1.0)))
    return np.stack((azimuth, elevation, radius), axis=-1)

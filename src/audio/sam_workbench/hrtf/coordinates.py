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


def unit_vectors(points_m: ArrayLike) -> NDArray[np.float64]:
    """Project ``(measurements, 3)`` positions onto the unit sphere.

    Direction lookup happens on the sphere, not in metres: two measurements at
    different radii in the same direction must be recognised as the same
    direction.
    """

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (measurements, 3) points, got shape {points.shape}")
    norms = np.linalg.norm(points, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return points / safe[:, None]


def angular_distance_deg(first: ArrayLike, second: ArrayLike) -> NDArray[np.float64]:
    """Great-circle angle in degrees between unit vectors.

    ``first`` is ``(3,)`` or ``(n, 3)`` and ``second`` is ``(m, 3)``; the result
    is ``(m,)`` or ``(n, m)``. Measuring on the sphere rather than differencing
    azimuths is what keeps the lookup well behaved near the poles and across
    the azimuth wrap.
    """

    left = np.atleast_2d(np.asarray(first, dtype=np.float64))
    right = np.asarray(second, dtype=np.float64)
    if left.shape[1] != 3 or right.ndim != 2 or right.shape[1] != 3:
        raise ValueError("angular distance operates on (n, 3) and (m, 3) vectors")
    dot = np.clip(left @ right.T, -1.0, 1.0)
    result = np.degrees(np.arccos(dot))
    return result[0] if np.asarray(first).ndim == 1 else result


#: Canonical Cartesian metres to azimuth/elevation/radius degrees. The canonical
#: frame and the SOFA spherical frame share axes, so this is the same mapping
#: main already exposes under its SOFA-facing name.
canonical_to_spherical_deg = cartesian_to_sofa_spherical

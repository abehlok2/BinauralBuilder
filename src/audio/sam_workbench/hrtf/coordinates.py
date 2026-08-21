"""SOFA coordinate conversion.

SOFA files declare their own coordinate type and units, and they do not all
agree: a file may store spherical degrees, spherical radians, or Cartesian
metres, and its azimuth may run the opposite way from ours. The declared
metadata is authoritative - never the file name, never a guess - so every
conversion in this module starts from what the file says about itself.

The canonical frame is the one the rest of the workbench uses: ``+x`` forward,
``+y`` left, ``+z`` up, azimuth ``0`` front and positive toward the left.
SOFA's ``spherical`` type uses the same handedness, so azimuth carries across
directly; the work is in units, radius, and the elevation-versus-colatitude
distinction.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..conventions import cartesian_to_spherical_deg, spherical_to_cartesian_m

__all__ = [
    "SOFA_POSITION_TYPES",
    "positions_to_canonical",
    "unit_vectors",
    "angular_distance_deg",
    "canonical_to_spherical_deg",
]

SOFA_POSITION_TYPES = ("spherical", "cartesian", "spherical harmonics")


def _unit_scale(units: str) -> tuple[float, bool]:
    """Return ``(angle_scale_to_degrees, radius_in_metres)`` for a units string.

    SOFA writes units as a comma-separated triple such as
    ``"degree, degree, metre"``. Anything naming radians is converted; anything
    naming centimetres is scaled.
    """

    text = str(units or "").lower()
    angle_scale = math.degrees(1.0) if "rad" in text else 1.0
    radius_scale = 0.01 if ("centimet" in text or "cm" in text) else 1.0
    return angle_scale, radius_scale


def positions_to_canonical(
    positions: NDArray[np.floating], position_type: str, units: str
) -> NDArray[np.float64]:
    """Convert SOFA ``SourcePosition`` values to canonical ``(3, n)`` metres.

    ``position_type`` and ``units`` come from the file's own attributes.
    """

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected (measurements, 3) positions, got shape {values.shape}")

    kind = str(position_type or "spherical").strip().lower()
    if kind.startswith("cart"):
        # SOFA cartesian is +x front, +y left, +z up: the canonical frame.
        _, radius_scale = _unit_scale(units)
        return np.ascontiguousarray(values.T) * radius_scale

    if not kind.startswith("spherical"):
        raise ValueError(f"unsupported SOFA position type {position_type!r}")

    angle_scale, radius_scale = _unit_scale(units)
    azimuth_deg = values[:, 0] * angle_scale
    elevation_deg = values[:, 1] * angle_scale
    radius_m = values[:, 2] * radius_scale
    # A file with no usable radius still has a direction; place it at one metre
    # so the direction survives and the distance is obviously nominal.
    radius_m = np.where(radius_m > 0.0, radius_m, 1.0)

    return np.vstack(
        [spherical_to_cartesian_m(azimuth, elevation, radius)
         for azimuth, elevation, radius in zip(azimuth_deg, elevation_deg, radius_m)]
    ).T


def canonical_to_spherical_deg(points_m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert canonical ``(3, n)`` metres to ``(3, n)`` azimuth/elevation/radius."""

    block = np.asarray(points_m, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != 3:
        raise ValueError(f"expected (3, n) points, got shape {block.shape}")
    return np.array(
        [cartesian_to_spherical_deg(block[:, index]) for index in range(block.shape[1])],
        dtype=np.float64,
    ).T


def unit_vectors(points_m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Project ``(3, n)`` positions onto the unit sphere.

    Direction lookup happens on the sphere, not in metres: two measurements at
    different radii in the same direction must be recognized as the same
    direction.
    """

    block = np.asarray(points_m, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != 3:
        raise ValueError(f"expected (3, n) points, got shape {block.shape}")
    norms = np.linalg.norm(block, axis=0)
    safe = np.where(norms > 0.0, norms, 1.0)
    return block / safe


def angular_distance_deg(
    first: NDArray[np.floating], second: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Great-circle angle in degrees between unit vectors.

    ``first`` is ``(3,)`` or ``(3, n)``; ``second`` is ``(3, m)``. The result
    broadcasts to ``(n, m)`` or ``(m,)``. Using the sphere rather than a
    difference of azimuths is what makes the lookup behave near the poles and
    across the azimuth wrap.
    """

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim == 1:
        left = left[:, None]
    dot = np.clip(left.T @ right, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

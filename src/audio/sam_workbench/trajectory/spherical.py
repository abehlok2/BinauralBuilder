"""Spherical entry for people who think in azimuth, elevation and distance.

The canonical representation of a source position is Cartesian: a trajectory is
``p(t) = [x(t), y(t), z(t)]`` in metres, and every renderer derives azimuth,
elevation and distance from it.  Spherical values are an *input convenience*,
converted at the boundary, so there is only ever one path format to maintain.

Axis naming
-----------
This package uses the frame documented in
:mod:`src.audio.sam_workbench.conventions`: ``+x`` forward, ``+y`` left,
``+z`` up, right-handed, azimuth ``0 deg`` in front and increasing toward the
listener's left.  Descriptions that name the axes "left/right, front/back,
height" are the same geometry with ``x`` and ``y`` exchanged; the derived
quantities agree term for term:

===============  ==============================  =========================
quantity         this package                    left/right-first naming
===============  ==============================  =========================
azimuth          ``atan2(y, x)``                 ``atan2(x, y)``
elevation        ``atan2(z, hypot(x, y))``       ``atan2(z, hypot(x, y))``
distance         ``norm(p)``                     ``norm(p)``
===============  ==============================  =========================

Converting at the adapter boundary rather than storing a second format is what
keeps a saved project unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SphericalPosition",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "spherical_array_to_cartesian",
    "cartesian_array_to_spherical",
]


@dataclass(frozen=True)
class SphericalPosition:
    """One position entered as azimuth, elevation and distance.

    Azimuth ``0 deg`` is directly in front and increases toward the listener's
    left; elevation is positive upward; distance is in metres.
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    distance_m: float = 1.0

    def __post_init__(self) -> None:
        for name in ("azimuth_deg", "elevation_deg", "distance_m"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.distance_m) < 0.0:
            raise ValueError("distance_m must not be negative")
        if not -90.0 <= float(self.elevation_deg) <= 90.0:
            raise ValueError(
                f"elevation_deg must lie in [-90, 90], got {self.elevation_deg!r}"
            )

    def to_cartesian(self) -> tuple[float, float, float]:
        """Return the canonical ``(x, y, z)`` metres for this direction."""

        x, y, z = spherical_to_cartesian(
            self.azimuth_deg, self.elevation_deg, self.distance_m
        )
        return (float(x), float(y), float(z))

    @classmethod
    def from_cartesian(cls, point_m: Sequence[float]) -> "SphericalPosition":
        azimuth, elevation, distance = cartesian_to_spherical(point_m)
        return cls(azimuth, elevation, distance)

    def describe(self) -> dict[str, float]:
        """The GUI's JSON form for a spherical entry."""

        return {
            "azimuthDegrees": float(self.azimuth_deg),
            "elevationDegrees": float(self.elevation_deg),
            "distanceMetres": float(self.distance_m),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SphericalPosition":
        """Accept either the documented long keys or their short aliases."""

        def pick(*names: str, default: float) -> float:
            for name in names:
                if name in data and data[name] is not None:
                    return float(data[name])
            return default

        return cls(
            azimuth_deg=pick("azimuthDegrees", "azimuthDeg", "azimuth", default=0.0),
            elevation_deg=pick(
                "elevationDegrees", "elevationDeg", "elevation", default=0.0
            ),
            distance_m=pick("distanceMetres", "distanceM", "distance", default=1.0),
        )


def spherical_to_cartesian(
    azimuth_deg: ArrayLike, elevation_deg: ArrayLike, distance_m: ArrayLike = 1.0
) -> NDArray[np.float64]:
    """Vectorized spherical-to-canonical-Cartesian conversion, in metres."""

    azimuth = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    elevation = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    radius = np.asarray(distance_m, dtype=np.float64)
    azimuth, elevation, radius = np.broadcast_arrays(azimuth, elevation, radius)
    horizontal = radius * np.cos(elevation)
    return np.stack(
        (
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            radius * np.sin(elevation),
        ),
        axis=-1,
    )


def cartesian_to_spherical(
    point_m: ArrayLike,
) -> tuple[float, float, float]:
    """Return ``(azimuth_deg, elevation_deg, distance_m)`` for one point.

    The origin maps to ``(0, 0, 0)`` rather than raising, so a trajectory that
    passes exactly through the listener stays finite.
    """

    values = np.asarray(point_m, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected a 3-element point, got shape {values.shape}")
    result = cartesian_array_to_spherical(values[None])[0]
    return (float(result[0]), float(result[1]), float(result[2]))


def spherical_array_to_cartesian(values: ArrayLike) -> NDArray[np.float64]:
    """Convert an ``(n, 3)`` array of azimuth/elevation/distance to metres."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"expected (n, 3) azimuth/elevation/distance rows, got {array.shape}"
        )
    return spherical_to_cartesian(array[:, 0], array[:, 1], array[:, 2])


def cartesian_array_to_spherical(points_m: ArrayLike) -> NDArray[np.float64]:
    """Convert ``(..., 3)`` metres to ``(..., 3)`` azimuth/elevation/distance.

    Azimuth is wrapped to ``(-180, 180]`` so a value read back into a numeric
    editor matches what the editor would have produced.
    """

    values = np.asarray(points_m, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise ValueError(f"points must end in three coordinates, got {values.shape}")
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    distance = np.linalg.norm(values, axis=-1)
    safe = np.maximum(distance, np.finfo(np.float64).tiny)
    azimuth = np.degrees(np.arctan2(y, x))
    # Wrap to (-180, 180]: atan2 returns [-pi, pi], and exactly -180 should
    # read back as +180 so the two ends of the range are not both reachable.
    azimuth = np.where(azimuth <= -180.0, azimuth + 360.0, azimuth)
    elevation = np.degrees(np.arcsin(np.clip(z / safe, -1.0, 1.0)))
    at_origin = distance <= 0.0
    return np.stack(
        (
            np.where(at_origin, 0.0, azimuth),
            np.where(at_origin, 0.0, elevation),
            distance,
        ),
        axis=-1,
    )

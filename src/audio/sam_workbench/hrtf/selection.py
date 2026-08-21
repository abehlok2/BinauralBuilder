"""Direction lookup on the measurement sphere.

Every interpolation mode needs the same primitive: given a direction, which
measurements are near it and how much does each count? Doing that on the unit
sphere - rather than on azimuth and elevation separately - is what keeps the
answer sane at the poles and across the azimuth wrap, where a naive angular
distance says two neighbouring directions are 350 degrees apart.

Three strategies, in the order they were built:

* :meth:`DirectionIndex.nearest` - one measurement, no blending;
* :meth:`DirectionIndex.nearest_k` - the k closest, weighted by unit-sphere
  distance;
* :meth:`DirectionIndex.barycentric` - the spherical triangle containing the
  direction, with barycentric weights.

The triangulation is built from the convex hull of the measurement points,
which for a set of directions on a sphere is its Delaunay triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .coordinates import angular_distance_deg, unit_vectors

__all__ = ["DirectionIndex", "DirectionWeights"]

_EPSILON = 1e-12
#: Below this a query is treated as sitting exactly on a measurement.
_COINCIDENT_DEG = 1e-6


def _as_direction(direction: NDArray[np.floating]) -> NDArray[np.float64]:
    """Normalise one direction to a unit ``(3,)`` vector."""

    return unit_vectors(np.asarray(direction, dtype=np.float64).reshape(1, 3))[0]


@dataclass(frozen=True)
class DirectionWeights:
    """Which measurements to use for one direction, and in what proportion."""

    indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    #: Angular distance to the nearest measurement used, in degrees.
    distance_deg: float
    #: True when the query fell outside the measured grid and had to be
    #: approximated from its edge rather than surrounded by it.
    extrapolated: bool = False

    def __post_init__(self) -> None:
        total = float(np.sum(self.weights))
        if total > _EPSILON:
            object.__setattr__(self, "weights", np.asarray(self.weights, dtype=np.float64) / total)


@dataclass(frozen=True)
class DirectionIndex:
    """A searchable set of measurement directions.

    Directions are stored ``(measurements, 3)`` to match the orientation of
    :attr:`~..hrtf.sofa_io.HRTFDataset.positions_m`, so an index can be built
    straight from a loaded dataset without transposing.
    """

    directions: NDArray[np.float64]  # (measurements, 3) unit vectors

    def __post_init__(self) -> None:
        block = np.asarray(self.directions, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != 3:
            raise ValueError(f"expected (measurements, 3) directions, got shape {block.shape}")
        if block.shape[0] == 0:
            raise ValueError("a direction index needs at least one measurement")
        object.__setattr__(self, "directions", unit_vectors(block))

    @property
    def measurements(self) -> int:
        return int(self.directions.shape[0])

    # --- nearest ------------------------------------------------------------

    def nearest(self, direction: NDArray[np.floating]) -> DirectionWeights:
        """The single closest measurement.

        The starting point of the whole progression: no blending, so no chance
        of smearing a transient or weakening an interaural time difference.
        """

        distances = angular_distance_deg(_as_direction(direction), self.directions)
        index = int(np.argmin(distances))
        return DirectionWeights(
            indices=np.array([index], dtype=np.int64),
            weights=np.array([1.0]),
            distance_deg=float(distances[index]),
        )

    # --- k nearest ----------------------------------------------------------

    def nearest_k(self, direction: NDArray[np.floating], count: int = 3) -> DirectionWeights:
        """The ``count`` closest measurements, weighted by unit-sphere distance.

        Weights are inverse-distance on the sphere, so a query sitting exactly
        on a measurement returns that measurement alone rather than a blend
        that would blur it.
        """

        distances = angular_distance_deg(_as_direction(direction), self.directions)
        count = int(max(1, min(count, self.measurements)))
        order = np.argsort(distances)[:count]
        chosen = distances[order]

        if chosen[0] <= _COINCIDENT_DEG:
            return DirectionWeights(
                indices=order[:1].astype(np.int64),
                weights=np.array([1.0]),
                distance_deg=float(chosen[0]),
            )

        weights = 1.0 / np.maximum(chosen, _EPSILON)
        return DirectionWeights(
            indices=order.astype(np.int64), weights=weights, distance_deg=float(chosen[0])
        )

    # --- spherical triangle -------------------------------------------------

    @cached_property
    def _triangulation(self) -> NDArray[np.int64] | None:
        """Triangles of the measurement sphere, or ``None`` when impossible.

        A horizontal-only set has no spherical triangulation - every point is
        coplanar - so the lookup falls back rather than inventing one.
        """

        if self.measurements < 4:
            return None
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(self.directions)
        except Exception:
            return None
        return np.asarray(hull.simplices, dtype=np.int64)

    @property
    def supports_triangulation(self) -> bool:
        return self._triangulation is not None

    def barycentric(self, direction: NDArray[np.floating]) -> DirectionWeights:
        """The spherical triangle containing ``direction``, with its weights.

        The direction is a ray from the origin; the triangle it pierces gives
        three measurements and three barycentric weights. When no triangle
        contains it - a sparse or planar grid - the lookup degrades to the
        three nearest and says so by marking the result extrapolated.
        """

        triangles = self._triangulation
        query = _as_direction(direction)
        if triangles is None:
            fallback = self.nearest_k(query, 3)
            return DirectionWeights(
                fallback.indices, fallback.weights, fallback.distance_deg, extrapolated=True
            )

        for simplex in triangles:
            vertices = self.directions[simplex]  # (3, 3), one vertex per row
            try:
                weights = np.linalg.solve(vertices.T, query)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate triangle
                continue
            if np.all(weights >= -1e-9):
                distances = angular_distance_deg(query, vertices)
                return DirectionWeights(
                    indices=np.asarray(simplex, dtype=np.int64),
                    weights=np.maximum(weights, 0.0),
                    distance_deg=float(np.min(distances)),
                )

        fallback = self.nearest_k(query, 3)
        return DirectionWeights(
            fallback.indices, fallback.weights, fallback.distance_deg, extrapolated=True
        )

    # --- coverage -----------------------------------------------------------

    def median_spacing_deg(self) -> float:
        """Median nearest-neighbour spacing, a proxy for how dense the grid is."""

        if self.measurements < 2:
            return 180.0
        distances = angular_distance_deg(self.directions, self.directions)
        np.fill_diagonal(distances, np.inf)
        return float(np.median(np.min(distances, axis=1)))

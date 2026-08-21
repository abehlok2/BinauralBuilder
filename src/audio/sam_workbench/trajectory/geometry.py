"""Vectorized canonical path geometry, independent of traversal timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from .transforms import Transform

Point3 = tuple[float, float, float]


def _parameter(value: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("path parameter must be finite")
    return result


def _points(value: Sequence[Sequence[float]], *, minimum: int = 1) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < minimum:
        raise ValueError(f"expected at least {minimum} three-dimensional points")
    if not np.all(np.isfinite(result)):
        raise ValueError("geometry points must be finite")
    return result


@runtime_checkable
class Geometry(Protocol):
    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Line:
    start_m: Point3
    end_m: Point3

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)[..., None]
        start = np.asarray(self.start_m, dtype=np.float64)
        return start + (np.asarray(self.end_m, dtype=np.float64) - start) * progress


@dataclass(frozen=True)
class Arc:
    radius_m: float = 1.0
    start_deg: float = -45.0
    end_deg: float = 45.0
    center_m: Point3 = (0.0, 0.0, 0.0)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = np.radians(self.start_deg + (self.end_deg - self.start_deg) * _parameter(u))
        return _polar_points(angle, np.full_like(angle, self.radius_m), self.center_m)


@dataclass(frozen=True)
class Circle:
    radius_m: float = 1.0
    center_m: Point3 = (0.0, 0.0, 0.0)
    start_deg: float = 0.0

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = np.radians(self.start_deg) + 2.0 * np.pi * _parameter(u)
        return _polar_points(angle, np.full_like(angle, self.radius_m), self.center_m)


@dataclass(frozen=True)
class Ellipse:
    radii_m: tuple[float, float] = (1.0, 0.5)
    center_m: Point3 = (0.0, 0.0, 0.0)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = 2.0 * np.pi * _parameter(u)
        center = np.asarray(self.center_m)
        return np.stack(
            (
                center[0] + self.radii_m[0] * np.cos(angle),
                center[1] + self.radii_m[1] * np.sin(angle),
                np.full_like(angle, center[2]),
            ),
            axis=-1,
        )


@dataclass(frozen=True)
class Spiral:
    start_radius_m: float = 0.1
    end_radius_m: float = 1.0
    turns: float = 2.0
    center_m: Point3 = (0.0, 0.0, 0.0)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)
        radius = self.start_radius_m + (self.end_radius_m - self.start_radius_m) * progress
        return _polar_points(2.0 * np.pi * self.turns * progress, radius, self.center_m)


@dataclass(frozen=True)
class Helix:
    radius_m: float = 1.0
    height_m: float = 1.0
    turns: float = 1.0
    center_m: Point3 = (0.0, 0.0, 0.0)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)
        result = _polar_points(2.0 * np.pi * self.turns * progress, np.full_like(progress, self.radius_m), self.center_m)
        result[..., 2] += self.height_m * (progress - 0.5)
        return result


@dataclass(frozen=True)
class Lissajous:
    amplitudes_m: Point3 = (1.0, 1.0, 0.0)
    frequencies: Point3 = (1.0, 2.0, 1.0)
    phases_deg: Point3 = (0.0, 90.0, 0.0)
    center_m: Point3 = (0.0, 0.0, 0.0)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)[..., None]
        phase = np.radians(self.phases_deg)
        return np.asarray(self.center_m) + np.asarray(self.amplitudes_m) * np.sin(
            2.0 * np.pi * progress * np.asarray(self.frequencies) + phase
        )


@dataclass(frozen=True)
class Bezier:
    control_points_m: tuple[Point3, ...]

    def __post_init__(self) -> None:
        _points(self.control_points_m, minimum=2)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)
        values = np.broadcast_to(_points(self.control_points_m), progress.shape + (len(self.control_points_m), 3)).copy()
        for remaining in range(len(self.control_points_m) - 1, 0, -1):
            values[..., :remaining, :] = (
                (1.0 - progress[..., None, None]) * values[..., :remaining, :]
                + progress[..., None, None] * values[..., 1 : remaining + 1, :]
            )
        return values[..., 0, :]


@dataclass(frozen=True)
class Spline:
    """Interpolating cubic spline through three or more control points."""

    control_points_m: tuple[Point3, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        _points(self.control_points_m, minimum=3)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        points = _points(self.control_points_m, minimum=3)
        if self.closed and not np.allclose(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        parameter = np.linspace(0.0, 1.0, len(points))
        boundary = "periodic" if self.closed else "natural"
        spline = CubicSpline(parameter, points, axis=0, bc_type=boundary)
        return np.asarray(spline(np.clip(_parameter(u), 0.0, 1.0)), dtype=np.float64)


@dataclass(frozen=True)
class Polyline:
    points_m: tuple[Point3, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        _points(self.points_m, minimum=2)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        return sample_polyline_arclength(self.points_m, u, self.closed)


@dataclass(frozen=True)
class Polygon:
    """Closed straight-edged path through the supplied vertices."""

    points_m: tuple[Point3, ...]

    def __post_init__(self) -> None:
        _points(self.points_m, minimum=3)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        return sample_polyline_arclength(self.points_m, u, closed=True)


@dataclass(frozen=True)
class PointCloud:
    """Deterministic traversal through an ordered collection of points."""

    points_m: tuple[Point3, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        _points(self.points_m, minimum=2)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        return sample_polyline_arclength(self.points_m, u, self.closed)


@dataclass(frozen=True)
class TransformedGeometry:
    geometry: Geometry
    transform: Transform = Transform()

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        return self.transform.apply(self.geometry.evaluate(u))


def _polar_points(angle: NDArray[np.float64], radius: NDArray[np.float64], center: Point3) -> NDArray[np.float64]:
    origin = np.asarray(center, dtype=np.float64)
    return np.stack(
        (origin[0] + radius * np.cos(angle), origin[1] + radius * np.sin(angle), np.full_like(angle, origin[2])),
        axis=-1,
    )


def sample_polyline_arclength(points: Sequence[Sequence[float]], u: ArrayLike, closed: bool = False) -> NDArray[np.float64]:
    """Sample a polyline by physical distance rather than vertex index."""

    vertices = _points(points, minimum=2)
    if closed and not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))
    distance = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distance)))
    progress = np.clip(_parameter(u), 0.0, 1.0)
    if cumulative[-1] <= np.finfo(np.float64).eps:
        return np.broadcast_to(vertices[0], progress.shape + (3,)).copy()
    target = progress * cumulative[-1]
    result = np.empty(progress.shape + (3,), dtype=np.float64)
    for axis in range(3):
        result[..., axis] = np.interp(target, cumulative, vertices[:, axis])
    return result


def arclength_table(geometry: Geometry, samples: int = 2049) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a monotonic ``(parameter, cumulative metres)`` lookup table."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    parameter = np.linspace(0.0, 1.0, int(samples))
    positions = np.asarray(geometry.evaluate(parameter), dtype=np.float64)
    if positions.shape != (samples, 3):
        raise ValueError(f"geometry returned invalid shape {positions.shape}")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
    return parameter, cumulative


def evaluate_arclength(geometry: Geometry, progress: ArrayLike, samples: int = 2049) -> NDArray[np.float64]:
    """Evaluate arbitrary geometry at normalized physical distance."""

    parameter, cumulative = arclength_table(geometry, samples)
    requested = np.clip(_parameter(progress), 0.0, 1.0)
    if cumulative[-1] <= np.finfo(np.float64).eps:
        return geometry.evaluate(np.zeros_like(requested))
    remapped = np.interp(requested * cumulative[-1], cumulative, parameter)
    return geometry.evaluate(remapped)

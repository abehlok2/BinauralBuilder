"""Path geometry: *where* a source can be, independent of *when* it is there.

A geometry maps a normalized parameter ``u`` in ``[0, 1]`` to points in metres
in the canonical listener frame (``+x`` forward, ``+y`` left, ``+z`` up). It
knows nothing about time, speed, or direction - those belong to the traversal -
so one path can be reused with different timing laws, which is exactly the
separation the specification asks for.

Arc-length parameterization is provided by every geometry, so a traversal can
ask for constant *speed* rather than constant parameter rate. The two differ
sharply on an ellipse or a spline, where equal steps in ``u`` are not equal
steps in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArcGeometry",
    "CircleGeometry",
    "EllipseGeometry",
    "GeometrySpec",
    "HelixGeometry",
    "LineGeometry",
    "LissajousGeometry",
    "PointCloudGeometry",
    "PolylineGeometry",
    "SpiralGeometry",
    "SplineGeometry",
    "geometry_from_dict",
]

_ARC_LENGTH_SAMPLES = 1024


def _as_points(values: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Normalize a point list to ``(3, n)`` metres, accepting 2-D input."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a list of points, got shape {array.shape}")
    if array.shape[1] == 2:
        array = np.column_stack((array, np.zeros(array.shape[0])))
    if array.shape[1] != 3:
        raise ValueError(f"expected 2-D or 3-D points, got shape {array.shape}")
    return np.ascontiguousarray(array.T)


@dataclass(frozen=True)
class GeometrySpec:
    """Base class: a curve in metres, parameterized by ``u`` in ``[0, 1]``."""

    kind: str = "geometry"

    # --- interface ----------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return False

    def evaluate(self, u: NDArray[np.floating] | float) -> NDArray[np.float64]:
        """Return ``(3, n)`` points in metres for parameter values ``u``."""

        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    # --- arc length ---------------------------------------------------------

    def _parameter_domain(self, samples: int) -> NDArray[np.float64]:
        return np.linspace(0.0, 1.0, int(samples))

    def arc_length_table(
        self, samples: int = _ARC_LENGTH_SAMPLES
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(u, cumulative_length_m)`` sampled along the curve."""

        u = self._parameter_domain(samples)
        points = self.evaluate(u)
        steps = np.linalg.norm(np.diff(points, axis=1), axis=0)
        return u, np.concatenate(([0.0], np.cumsum(steps)))

    def length_m(self, samples: int = _ARC_LENGTH_SAMPLES) -> float:
        """Total path length in metres."""

        return float(self.arc_length_table(samples)[1][-1])

    def u_at_arc_length(
        self, normalized_distance: NDArray[np.floating] | float, samples: int = _ARC_LENGTH_SAMPLES
    ) -> NDArray[np.float64]:
        """Map normalized distance travelled to the parameter ``u``.

        This is what makes constant-speed traversal possible: without it, a
        source crossing an ellipse's flat side moves visibly faster than one
        rounding its end.
        """

        u_table, lengths = self.arc_length_table(samples)
        total = lengths[-1]
        distance = np.atleast_1d(np.asarray(normalized_distance, dtype=np.float64))
        if total <= 0.0:
            return np.zeros_like(distance)
        return np.interp(np.clip(distance, 0.0, 1.0) * total, lengths, u_table)

    def evaluate_by_arc_length(
        self, normalized_distance: NDArray[np.floating] | float
    ) -> NDArray[np.float64]:
        """Evaluate at a normalized *distance* rather than a raw parameter."""

        return self.evaluate(self.u_at_arc_length(normalized_distance))


@dataclass(frozen=True)
class LineGeometry(GeometrySpec):
    """A straight segment between two points, in metres."""

    start_m: tuple[float, float, float] = (1.0, -1.0, 0.0)
    end_m: tuple[float, float, float] = (1.0, 1.0, 0.0)
    kind: str = "line"

    def evaluate(self, u):
        fraction = np.atleast_1d(np.asarray(u, dtype=np.float64))
        start = np.asarray(self.start_m, dtype=np.float64)[:, None]
        end = np.asarray(self.end_m, dtype=np.float64)[:, None]
        return start + (end - start) * fraction[None, :]

    def to_dict(self):
        return {"kind": self.kind, "start_m": list(self.start_m), "end_m": list(self.end_m)}


@dataclass(frozen=True)
class ArcGeometry(GeometrySpec):
    """A horizontal (or elevated) circular arc around the listener.

    Angles are canonical azimuths in degrees: ``0`` is front and positive turns
    toward the listener's left.
    """

    radius_m: float = 1.0
    start_deg: float = -90.0
    end_deg: float = 90.0
    elevation_deg: float = 0.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kind: str = "arc"

    @property
    def is_closed(self) -> bool:
        return abs((float(self.end_deg) - float(self.start_deg)) % 360.0) < 1e-9 and (
            float(self.end_deg) != float(self.start_deg)
        )

    def evaluate(self, u):
        fraction = np.atleast_1d(np.asarray(u, dtype=np.float64))
        azimuth = np.radians(float(self.start_deg) + (float(self.end_deg) - float(self.start_deg)) * fraction)
        elevation = math.radians(float(self.elevation_deg))
        horizontal = float(self.radius_m) * math.cos(elevation)
        points = np.vstack(
            (
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                np.full(azimuth.shape, float(self.radius_m) * math.sin(elevation)),
            )
        )
        return points + np.asarray(self.centre_m, dtype=np.float64)[:, None]

    def to_dict(self):
        return {
            "kind": self.kind,
            "radius_m": float(self.radius_m),
            "start_deg": float(self.start_deg),
            "end_deg": float(self.end_deg),
            "elevation_deg": float(self.elevation_deg),
            "centre_m": list(self.centre_m),
        }


@dataclass(frozen=True)
class CircleGeometry(ArcGeometry):
    """A full horizontal circle: an arc that closes on itself."""

    start_deg: float = 0.0
    end_deg: float = 360.0
    kind: str = "circle"

    @property
    def is_closed(self) -> bool:
        return True


@dataclass(frozen=True)
class EllipseGeometry(GeometrySpec):
    """An ellipse in the horizontal plane, optionally rotated about ``+z``."""

    forward_radius_m: float = 1.5
    lateral_radius_m: float = 1.0
    rotation_deg: float = 0.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    elevation_m: float = 0.0
    kind: str = "ellipse"

    @property
    def is_closed(self) -> bool:
        return True

    def evaluate(self, u):
        angle = 2.0 * math.pi * np.atleast_1d(np.asarray(u, dtype=np.float64))
        forward = float(self.forward_radius_m) * np.cos(angle)
        lateral = float(self.lateral_radius_m) * np.sin(angle)
        rotation = math.radians(float(self.rotation_deg))
        points = np.vstack(
            (
                forward * math.cos(rotation) - lateral * math.sin(rotation),
                forward * math.sin(rotation) + lateral * math.cos(rotation),
                np.full(angle.shape, float(self.elevation_m)),
            )
        )
        return points + np.asarray(self.centre_m, dtype=np.float64)[:, None]

    def to_dict(self):
        return {
            "kind": self.kind,
            "forward_radius_m": float(self.forward_radius_m),
            "lateral_radius_m": float(self.lateral_radius_m),
            "rotation_deg": float(self.rotation_deg),
            "centre_m": list(self.centre_m),
            "elevation_m": float(self.elevation_m),
        }


@dataclass(frozen=True)
class SpiralGeometry(GeometrySpec):
    """A horizontal spiral from one radius to another over a number of turns."""

    start_radius_m: float = 0.5
    end_radius_m: float = 2.0
    turns: float = 2.0
    start_deg: float = 0.0
    elevation_deg: float = 0.0
    kind: str = "spiral"

    def evaluate(self, u):
        fraction = np.atleast_1d(np.asarray(u, dtype=np.float64))
        azimuth = np.radians(float(self.start_deg) + 360.0 * float(self.turns) * fraction)
        radius = float(self.start_radius_m) + (float(self.end_radius_m) - float(self.start_radius_m)) * fraction
        elevation = math.radians(float(self.elevation_deg))
        horizontal = radius * math.cos(elevation)
        return np.vstack(
            (
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                radius * math.sin(elevation),
            )
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "start_radius_m": float(self.start_radius_m),
            "end_radius_m": float(self.end_radius_m),
            "turns": float(self.turns),
            "start_deg": float(self.start_deg),
            "elevation_deg": float(self.elevation_deg),
        }


@dataclass(frozen=True)
class HelixGeometry(GeometrySpec):
    """A circle in the horizontal plane that rises as it turns."""

    radius_m: float = 1.0
    turns: float = 2.0
    start_deg: float = 0.0
    start_height_m: float = -0.5
    end_height_m: float = 0.5
    kind: str = "helix"

    def evaluate(self, u):
        fraction = np.atleast_1d(np.asarray(u, dtype=np.float64))
        azimuth = np.radians(float(self.start_deg) + 360.0 * float(self.turns) * fraction)
        height = float(self.start_height_m) + (float(self.end_height_m) - float(self.start_height_m)) * fraction
        return np.vstack(
            (float(self.radius_m) * np.cos(azimuth), float(self.radius_m) * np.sin(azimuth), height)
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "radius_m": float(self.radius_m),
            "turns": float(self.turns),
            "start_deg": float(self.start_deg),
            "start_height_m": float(self.start_height_m),
            "end_height_m": float(self.end_height_m),
        }


@dataclass(frozen=True)
class LissajousGeometry(GeometrySpec):
    """A Lissajous figure in the horizontal plane."""

    forward_scale_m: float = 1.5
    lateral_scale_m: float = 1.0
    forward_ratio: float = 3.0
    lateral_ratio: float = 2.0
    phase_deg: float = 90.0
    elevation_m: float = 0.0
    kind: str = "lissajous"

    @property
    def is_closed(self) -> bool:
        return float(self.forward_ratio).is_integer() and float(self.lateral_ratio).is_integer()

    def evaluate(self, u):
        angle = 2.0 * math.pi * np.atleast_1d(np.asarray(u, dtype=np.float64))
        phase = math.radians(float(self.phase_deg))
        return np.vstack(
            (
                float(self.forward_scale_m) * np.sin(float(self.forward_ratio) * angle + phase),
                float(self.lateral_scale_m) * np.sin(float(self.lateral_ratio) * angle),
                np.full(angle.shape, float(self.elevation_m)),
            )
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "forward_scale_m": float(self.forward_scale_m),
            "lateral_scale_m": float(self.lateral_scale_m),
            "forward_ratio": float(self.forward_ratio),
            "lateral_ratio": float(self.lateral_ratio),
            "phase_deg": float(self.phase_deg),
            "elevation_m": float(self.elevation_m),
        }


@dataclass(frozen=True)
class PolylineGeometry(GeometrySpec):
    """Straight segments through a point list, open or closed."""

    points_m: tuple = ()
    closed: bool = False
    kind: str = "polyline"

    @property
    def is_closed(self) -> bool:
        return bool(self.closed)

    def _vertices(self) -> NDArray[np.float64]:
        points = _as_points(self.points_m)
        if points.shape[1] < 2:
            raise ValueError("a polyline needs at least two points")
        if self.closed:
            points = np.hstack((points, points[:, :1]))
        return points

    def evaluate(self, u):
        vertices = self._vertices()
        fraction = np.clip(np.atleast_1d(np.asarray(u, dtype=np.float64)), 0.0, 1.0)
        knots = np.linspace(0.0, 1.0, vertices.shape[1])
        return np.vstack([np.interp(fraction, knots, vertices[axis]) for axis in range(3)])

    def to_dict(self):
        return {
            "kind": self.kind,
            "points_m": [list(point) for point in _as_points(self.points_m).T],
            "closed": bool(self.closed),
        }


@dataclass(frozen=True)
class SplineGeometry(GeometrySpec):
    """A Catmull-Rom spline through a point list, open or closed.

    The same curve family the legacy custom-path editor draws, so a translated
    profile follows the shape the user drew rather than an approximation of it.
    """

    points_m: tuple = ()
    closed: bool = False
    samples_per_segment: int = 24
    kind: str = "spline"

    @property
    def is_closed(self) -> bool:
        return bool(self.closed)

    def _sampled(self) -> NDArray[np.float64]:
        points = _as_points(self.points_m)
        count = points.shape[1]
        if count < 2:
            raise ValueError("a spline needs at least two points")
        if count == 2:
            return points

        segments = count if self.closed else count - 1
        steps = max(2, int(self.samples_per_segment))
        sampled = []
        for index in range(segments):
            if self.closed:
                p0 = points[:, (index - 1) % count]
                p1 = points[:, index]
                p2 = points[:, (index + 1) % count]
                p3 = points[:, (index + 2) % count]
            else:
                p0 = points[:, max(0, index - 1)]
                p1 = points[:, index]
                p2 = points[:, min(count - 1, index + 1)]
                p3 = points[:, min(count - 1, index + 2)]
            t = np.linspace(0.0, 1.0, steps, endpoint=False)
            t2, t3 = t * t, t * t * t
            segment = 0.5 * (
                (2.0 * p1)[:, None]
                + (-p0 + p2)[:, None] * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[:, None] * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[:, None] * t3
            )
            sampled.append(segment)
        sampled.append(points[:, :1] if self.closed else points[:, -1:])
        return np.hstack(sampled)

    def evaluate(self, u):
        sampled = self._sampled()
        fraction = np.clip(np.atleast_1d(np.asarray(u, dtype=np.float64)), 0.0, 1.0)
        knots = np.linspace(0.0, 1.0, sampled.shape[1])
        return np.vstack([np.interp(fraction, knots, sampled[axis]) for axis in range(3)])

    def to_dict(self):
        return {
            "kind": self.kind,
            "points_m": [list(point) for point in _as_points(self.points_m).T],
            "closed": bool(self.closed),
            "samples_per_segment": int(self.samples_per_segment),
        }


@dataclass(frozen=True)
class PointCloudGeometry(GeometrySpec):
    """Discrete positions with no path between them.

    Traversing a point cloud is a sequence of jumps, which is what the
    discontinuity handling and its crossfade exist for.
    """

    points_m: tuple = ()
    kind: str = "point_cloud"

    @property
    def is_closed(self) -> bool:
        return True

    def evaluate(self, u):
        points = _as_points(self.points_m)
        if points.shape[1] == 0:
            raise ValueError("a point cloud needs at least one point")
        fraction = np.atleast_1d(np.asarray(u, dtype=np.float64))
        index = np.clip((np.mod(fraction, 1.0) * points.shape[1]).astype(int), 0, points.shape[1] - 1)
        return points[:, index]

    def to_dict(self):
        return {"kind": self.kind, "points_m": [list(point) for point in _as_points(self.points_m).T]}


_GEOMETRY_KINDS = {
    "line": LineGeometry,
    "arc": ArcGeometry,
    "circle": CircleGeometry,
    "ellipse": EllipseGeometry,
    "spiral": SpiralGeometry,
    "helix": HelixGeometry,
    "lissajous": LissajousGeometry,
    "polyline": PolylineGeometry,
    "spline": SplineGeometry,
    "point_cloud": PointCloudGeometry,
}


def geometry_from_dict(data) -> GeometrySpec:
    """Rebuild a geometry from its serialized form."""

    kind = str(data.get("kind", "")).lower()
    geometry_type = _GEOMETRY_KINDS.get(kind)
    if geometry_type is None:
        raise ValueError(f"unknown geometry kind {kind!r}")
    fields = {key: value for key, value in data.items() if key != "kind"}
    for key in ("start_m", "end_m", "centre_m"):
        if key in fields:
            fields[key] = tuple(float(value) for value in fields[key])
    for key in ("points_m",):
        if key in fields:
            fields[key] = tuple(tuple(float(value) for value in point) for point in fields[key])
    return geometry_type(**fields)

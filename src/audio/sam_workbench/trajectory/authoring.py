"""Version-three, opt-in path constraints and directional interpolation.

Constraints act in listener-relative metres after transforms. Stored control
points remain Cartesian. Legacy paths without constraints retain their output.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from .spherical import cartesian_array_to_spherical, spherical_array_to_cartesian


def spherical_segments(points, parameter, u):
    """Shortest great-circle segments with independently linear radii.

    Antipodal endpoints have no unique great circle: reject them rather than
    selecting an arbitrary overhead/underhead route. An intermediate key makes
    the author's chosen plane explicit. Zero-radius keys have no direction.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(
            "Spherical control points must be finite three-dimensional positions"
        )
    radii = np.linalg.norm(points, axis=-1)
    if np.any(radii <= 1e-9):
        raise ValueError(
            "Spherical interpolation needs nonzero distance at every point"
        )
    directions = points / radii[:, None]
    dot = np.clip(np.sum(directions[:-1] * directions[1:], axis=-1), -1, 1)
    if np.any(dot < -1 + 1e-8):
        raise ValueError(
            "Opposite spherical points need an intermediate point to choose the arc"
        )
    values = np.clip(np.asarray(u, dtype=float), parameter[0], parameter[-1])
    index = np.clip(
        np.searchsorted(parameter, values, side="right") - 1, 0, len(points) - 2
    )
    fraction = (values - parameter[index]) / (parameter[index + 1] - parameter[index])
    angle = np.arccos(dot[index])
    sine = np.sin(angle)
    safe = np.maximum(sine, 1e-12)
    a = np.where(angle < 1e-7, 1 - fraction, np.sin((1 - fraction) * angle) / safe)
    b = np.where(angle < 1e-7, fraction, np.sin(fraction * angle) / safe)
    direction = a[..., None] * directions[index] + b[..., None] * directions[index + 1]
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    radius = radii[index] + fraction * (radii[index + 1] - radii[index])
    return direction * radius[..., None]


@dataclass(frozen=True)
class SphericalPolyline:
    points_m: tuple
    closed: bool = False

    def __post_init__(self):
        if len(self.points_m) < 2:
            raise ValueError("Spherical paths need at least two points")
        self.evaluate(np.array([0.0, 1.0]))

    def evaluate(self, u):
        points = np.asarray(self.points_m, dtype=float)
        if self.closed and not np.allclose(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        return spherical_segments(points, np.linspace(0, 1, len(points)), u)


@dataclass(frozen=True)
class PathConstraints:
    distance_m: float | None = None
    elevation_deg: float | None = None
    azimuth_deg: float | None = None
    minimum_height_m: float | None = None
    maximum_height_m: float | None = None
    clearance_m: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.distance_m is not None and self.distance_m <= 0:
            raise ValueError("Locked distance must be positive")
        if self.elevation_deg is not None and not -90 <= self.elevation_deg <= 90:
            raise ValueError("Locked elevation must be between -90 and 90 degrees")
        if self.clearance_m < 0:
            raise ValueError("Listener clearance must be nonnegative")
        if (
            self.minimum_height_m is not None
            and self.maximum_height_m is not None
            and self.minimum_height_m > self.maximum_height_m
        ):
            raise ValueError("Minimum height exceeds maximum height")
        if self.distance_m is not None:
            low = max(
                -self.distance_m,
                (
                    self.minimum_height_m
                    if self.minimum_height_m is not None
                    else -self.distance_m
                ),
            )
            high = min(
                self.distance_m,
                (
                    self.maximum_height_m
                    if self.maximum_height_m is not None
                    else self.distance_m
                ),
            )
            if low > high:
                raise ValueError(
                    "Height bounds do not intersect the locked distance sphere"
                )
            if self.elevation_deg is not None:
                z = self.distance_m * np.sin(np.radians(self.elevation_deg))
                if not low - 1e-9 <= z <= high + 1e-9:
                    raise ValueError(
                        "Locked elevation and distance conflict with height bounds"
                    )
        elif self.elevation_deg is not None and (
            self.minimum_height_m is not None or self.maximum_height_m is not None
        ):
            raise ValueError(
                "Combine elevation and height bounds only with a locked distance"
            )

    def apply(self, points):
        points = np.asarray(points, dtype=float)
        if not any(
            value is not None
            for name, value in vars(self).items()
            if name != "clearance_m"
        ):
            return points
        spherical = cartesian_array_to_spherical(points)
        if np.any(spherical[..., 2] < 1e-9):
            raise ValueError(
                "Constrained paths cannot pass through the listener; add an intermediate point"
            )
        if self.distance_m is not None:
            spherical[..., 2] = self.distance_m
        if self.elevation_deg is not None:
            spherical[..., 1] = self.elevation_deg
        if self.azimuth_deg is not None:
            spherical[..., 0] = self.azimuth_deg
        result = spherical_array_to_cartesian(spherical.reshape(-1, 3)).reshape(
            points.shape
        )
        if self.minimum_height_m is not None or self.maximum_height_m is not None:
            z = np.clip(result[..., 2], self.minimum_height_m, self.maximum_height_m)
            if self.distance_m is not None:
                z = np.clip(z, -self.distance_m, self.distance_m)
                horizontal = np.sqrt(np.maximum(self.distance_m**2 - z * z, 0))
                azimuth = np.radians(spherical[..., 0])
                result[..., 0] = horizontal * np.cos(azimuth)
                result[..., 1] = horizontal * np.sin(azimuth)
            result[..., 2] = z
        return result

    @classmethod
    def from_mapping(cls, data):
        return cls(
            **{
                key: value
                for key, value in dict(data or {}).items()
                if key in cls.__dataclass_fields__
            }
        )

    def describe(self):
        return {key: value for key, value in vars(self).items() if value is not None}


@dataclass(frozen=True)
class EvaluatedGeometry:
    """Effective, listener-relative shape, including constraints."""

    model: object

    def evaluate(self, u):
        model = self.model
        points = model.transform.apply(model.geometry.evaluate(u))
        if not model.is_listener_relative:
            points = model.listener.world_to_listener(points)
        return model.constraints.apply(points)


@dataclass(frozen=True)
class AngularGeometry:
    geometry: object
    samples: int = 4097

    @cached_property
    def table(self):
        u = np.linspace(0, 1, self.samples)
        points = np.asarray(self.geometry.evaluate(u))
        radii = np.linalg.norm(points, axis=-1)
        if np.any(radii <= 1e-9):
            raise ValueError("Constant angular speed is undefined at the listener")
        directions = points / radii[:, None]
        increments = np.arctan2(
            np.linalg.norm(np.cross(directions[:-1], directions[1:]), axis=-1),
            np.sum(directions[:-1] * directions[1:], axis=-1),
        )
        if np.any(
            (increments <= 1e-12)
            & (np.linalg.norm(np.diff(points, axis=0), axis=-1) > 1e-9)
        ):
            raise ValueError(
                "Constant angular speed cannot preserve radial-only motion; choose parameter speed"
            )
        lengths = np.r_[0.0, np.cumsum(increments)]
        keep = np.r_[True, np.diff(lengths) > 1e-12]
        return u[keep], lengths[keep]

    def evaluate(self, progress):
        u, lengths = self.table
        if lengths[-1] <= 1e-12:
            raise ValueError(
                "Constant angular speed needs a path with changing direction"
            )
        mapped = np.interp(np.asarray(progress) * lengths[-1], lengths, u)
        return self.geometry.evaluate(mapped)


def loop_findings(model):
    if model is None or model.traversal.mode != "loop":
        return []
    shape = EvaluatedGeometry(model)
    points = shape.evaluate(np.array([0, 1e-4, 1 - 1e-4, 1]))
    gap = float(np.linalg.norm(points[-1] - points[0]))
    if gap > 1e-4:
        return [
            f"Loop jumps {gap:.3f} m at its seam. Choose ping-pong, close the path, or use explicit discontinuous traversal."
        ]
    a, b = points[1] - points[0], points[-1] - points[-2]
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm > 1e-12 and np.dot(a, b) / norm < 0.999:
        return [
            "Loop changes direction abruptly at its seam. Adjust endpoint tangents or choose eased ping-pong motion."
        ]
    return []

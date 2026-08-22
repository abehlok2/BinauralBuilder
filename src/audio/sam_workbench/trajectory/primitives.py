"""Three-dimensional path primitives.

The existing geometry module covers flat circles, arcs and a handful of curves.
Those are enough to move a source around a listener but not above or below one,
and height is the whole point of a 3-D path.  This module adds the primitives
that put a source overhead, under the chin, or spiralling between the two.

Every primitive here is ordinary geometry: it implements ``evaluate(u)`` over
normalized progress in ``[0, 1]`` and returns ``(..., 3)`` metres in the
canonical listener frame.  None of them knows anything about time.  That is
deliberate - a primitive plus any traversal compiles to the same timestamped
trajectory, so a rising spiral traversed at constant speed and the same spiral
driven from an envelope are one geometry with two time laws rather than two
separate path types.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spherical import spherical_to_cartesian

__all__ = [
    "HorizontalOrbit",
    "VerticalOrbit",
    "TiltedOrbit",
    "RisingArc",
    "OverheadSweep",
    "ElevationSweep",
    "SphericalOrbit",
    "DomeTraversal",
    "FigureEight3D",
    "Pendulum",
    "RandomWalkVolume",
    "Torus",
    "PRIMITIVE_TYPES",
]


def _parameter(value: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("path parameter must be finite")
    return result


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _centre(value: Sequence[float]) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("centre_m must contain three finite values")
    return result


def _elevation(value: float, name: str) -> float:
    number = _finite(value, name)
    if not -90.0 <= number <= 90.0:
        raise ValueError(f"{name} must lie in [-90, 90], got {number}")
    return number


# --- orbits ----------------------------------------------------------------


@dataclass(frozen=True)
class HorizontalOrbit:
    """A ring around the listener at one height.

    The familiar flat circle, expressed in the spatial-audio terms an author
    thinks in: a radius, a starting azimuth, and an elevation the whole ring
    sits at.  Elevation of zero is the ear-height plane.
    """

    radius_m: float = 1.5
    elevation_deg: float = 0.0
    start_azimuth_deg: float = 0.0
    turns: float = 1.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _elevation(self.elevation_deg, "elevation_deg")
        _finite(self.turns, "turns")
        _centre(self.centre_m)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        azimuth = self.start_azimuth_deg + 360.0 * self.turns * _parameter(u)
        return (
            spherical_to_cartesian(
                azimuth, self.elevation_deg, self.radius_m
            )
            + _centre(self.centre_m)
        )


@dataclass(frozen=True)
class VerticalOrbit:
    """A ring in a vertical plane: front, over the head, behind, underneath.

    ``plane_azimuth_deg`` turns the whole circle about the vertical axis, so
    zero sweeps through the median plane (front to overhead to back) and 90
    sweeps left-to-right over the head.
    """

    radius_m: float = 1.5
    plane_azimuth_deg: float = 0.0
    start_angle_deg: float = 0.0
    turns: float = 1.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _finite(self.plane_azimuth_deg, "plane_azimuth_deg")
        _finite(self.start_angle_deg, "start_angle_deg")
        _finite(self.turns, "turns")
        _centre(self.centre_m)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = np.radians(self.start_angle_deg + 360.0 * self.turns * _parameter(u))
        plane = math.radians(self.plane_azimuth_deg)
        horizontal = self.radius_m * np.cos(angle)
        return (
            np.stack(
                (
                    horizontal * math.cos(plane),
                    horizontal * math.sin(plane),
                    self.radius_m * np.sin(angle),
                ),
                axis=-1,
            )
            + _centre(self.centre_m)
        )


@dataclass(frozen=True)
class TiltedOrbit:
    """A ring tilted out of the horizontal plane by ``tilt_deg``.

    At zero tilt it is a horizontal orbit and at 90 a vertical one, so this is
    the general case both of those specialize.  ``tilt_axis_azimuth_deg`` names
    the horizontal direction the ring is tipped up toward.
    """

    radius_m: float = 1.5
    tilt_deg: float = 30.0
    tilt_axis_azimuth_deg: float = 0.0
    start_azimuth_deg: float = 0.0
    turns: float = 1.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _finite(self.tilt_deg, "tilt_deg")
        _finite(self.tilt_axis_azimuth_deg, "tilt_axis_azimuth_deg")
        _finite(self.turns, "turns")
        _centre(self.centre_m)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = np.radians(self.start_azimuth_deg + 360.0 * self.turns * _parameter(u))
        flat = np.stack(
            (
                self.radius_m * np.cos(angle),
                self.radius_m * np.sin(angle),
                np.zeros_like(angle),
            ),
            axis=-1,
        )
        # Tip the ring about a horizontal axis pointing along the tilt azimuth:
        # rotate that axis onto +y, pitch, then rotate back.
        from .transforms import rotation_matrix_ypr

        to_axis = rotation_matrix_ypr(yaw_deg=-self.tilt_axis_azimuth_deg)
        pitch = rotation_matrix_ypr(pitch_deg=self.tilt_deg)
        matrix = to_axis.T @ pitch @ to_axis
        return flat @ matrix.T + _centre(self.centre_m)


@dataclass(frozen=True)
class SphericalOrbit:
    """A ring at constant distance whose radius breathes in and out.

    Distance moves between ``start_distance_m`` and ``end_distance_m`` while
    azimuth advances, which is the expanding and contracting orbit an author
    asks for by name.  With the two distances equal it is a constant-distance
    orbit on the measurement sphere - the case an HRTF dataset represents best.
    """

    start_distance_m: float = 0.6
    end_distance_m: float = 2.0
    elevation_deg: float = 0.0
    turns: float = 2.0
    start_azimuth_deg: float = 0.0
    cycles: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.start_distance_m, "start_distance_m")
        _positive(self.end_distance_m, "end_distance_m")
        _elevation(self.elevation_deg, "elevation_deg")
        _finite(self.turns, "turns")
        _positive(self.cycles, "cycles")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)
        azimuth = self.start_azimuth_deg + 360.0 * self.turns * progress
        # A raised cosine breathes out and back in over each cycle, so a looped
        # traversal returns to its starting radius without a jump.
        breath = 0.5 - 0.5 * np.cos(2.0 * np.pi * self.cycles * progress)
        distance = (
            self.start_distance_m
            + (self.end_distance_m - self.start_distance_m) * breath
        )
        return spherical_to_cartesian(azimuth, self.elevation_deg, distance)


# --- sweeps ----------------------------------------------------------------


@dataclass(frozen=True)
class RisingArc:
    """An arc that changes azimuth and elevation together.

    This is the floor-to-overhead sweep and its falling counterpart: set
    ``start_elevation_deg`` low and ``end_elevation_deg`` high, and give the
    azimuths the same value to rise straight up in front of the listener.
    """

    start_azimuth_deg: float = -45.0
    end_azimuth_deg: float = 45.0
    start_elevation_deg: float = -30.0
    end_elevation_deg: float = 60.0
    start_distance_m: float = 1.5
    end_distance_m: float = 1.5

    def __post_init__(self) -> None:
        _finite(self.start_azimuth_deg, "start_azimuth_deg")
        _finite(self.end_azimuth_deg, "end_azimuth_deg")
        _elevation(self.start_elevation_deg, "start_elevation_deg")
        _elevation(self.end_elevation_deg, "end_elevation_deg")
        _positive(self.start_distance_m, "start_distance_m")
        _positive(self.end_distance_m, "end_distance_m")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)

        def between(start: float, end: float) -> NDArray[np.float64]:
            return start + (end - start) * progress

        return spherical_to_cartesian(
            between(self.start_azimuth_deg, self.end_azimuth_deg),
            between(self.start_elevation_deg, self.end_elevation_deg),
            between(self.start_distance_m, self.end_distance_m),
        )


@dataclass(frozen=True)
class OverheadSweep:
    """Front-centre to directly overhead and on to the back, at one distance.

    The path an author means by "take it over my head": azimuth is held while
    elevation climbs to the zenith and, if ``pass_over`` is set, continues down
    behind the listener by flipping azimuth through 180 degrees.
    """

    azimuth_deg: float = 0.0
    start_elevation_deg: float = 0.0
    distance_m: float = 1.5
    pass_over: bool = True
    end_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.azimuth_deg, "azimuth_deg")
        _elevation(self.start_elevation_deg, "start_elevation_deg")
        _elevation(self.end_elevation_deg, "end_elevation_deg")
        _positive(self.distance_m, "distance_m")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)
        if not self.pass_over:
            elevation = self.start_elevation_deg + (
                90.0 - self.start_elevation_deg
            ) * progress
            return spherical_to_cartesian(self.azimuth_deg, elevation, self.distance_m)
        # Rise to the zenith over the first half, descend behind over the
        # second. Azimuth flips at the top, where it is undefined anyway, so
        # the position stays continuous through the crossing.
        rising = progress <= 0.5
        elevation = np.where(
            rising,
            self.start_elevation_deg + (90.0 - self.start_elevation_deg) * (2.0 * progress),
            90.0 - (90.0 - self.end_elevation_deg) * (2.0 * progress - 1.0),
        )
        azimuth = np.where(rising, self.azimuth_deg, self.azimuth_deg + 180.0)
        return spherical_to_cartesian(azimuth, elevation, self.distance_m)


@dataclass(frozen=True)
class ElevationSweep:
    """A front-to-back sweep at a chosen elevation, or a pure elevation ramp.

    Holding ``start_azimuth_deg`` at 0 and ``end_azimuth_deg`` at 180 walks the
    source over or under the listener from front to back; holding both equal
    and varying the elevations ramps height in place.
    """

    start_azimuth_deg: float = 0.0
    end_azimuth_deg: float = 180.0
    elevation_deg: float = 45.0
    distance_m: float = 1.5
    end_elevation_deg: float | None = None

    def __post_init__(self) -> None:
        _finite(self.start_azimuth_deg, "start_azimuth_deg")
        _finite(self.end_azimuth_deg, "end_azimuth_deg")
        _elevation(self.elevation_deg, "elevation_deg")
        if self.end_elevation_deg is not None:
            _elevation(self.end_elevation_deg, "end_elevation_deg")
        _positive(self.distance_m, "distance_m")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)
        end_elevation = (
            self.elevation_deg
            if self.end_elevation_deg is None
            else self.end_elevation_deg
        )
        return spherical_to_cartesian(
            self.start_azimuth_deg
            + (self.end_azimuth_deg - self.start_azimuth_deg) * progress,
            self.elevation_deg + (end_elevation - self.elevation_deg) * progress,
            self.distance_m,
        )


@dataclass(frozen=True)
class DomeTraversal:
    """A spiral over a hemisphere, from its rim to its pole.

    Set ``start_elevation_deg`` to 0 and ``end_elevation_deg`` to 90 for a dome
    over the listener; use negative elevations for the lower hemisphere.  This
    is the spiral-rising-around-the-listener path, and unlike a helix it stays
    at a constant distance so an HRTF dataset can follow it.
    """

    distance_m: float = 1.5
    start_elevation_deg: float = 0.0
    end_elevation_deg: float = 85.0
    turns: float = 3.0
    start_azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.distance_m, "distance_m")
        _elevation(self.start_elevation_deg, "start_elevation_deg")
        _elevation(self.end_elevation_deg, "end_elevation_deg")
        _finite(self.turns, "turns")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)
        return spherical_to_cartesian(
            self.start_azimuth_deg + 360.0 * self.turns * progress,
            self.start_elevation_deg
            + (self.end_elevation_deg - self.start_elevation_deg) * progress,
            self.distance_m,
        )


# --- figures ---------------------------------------------------------------


@dataclass(frozen=True)
class FigureEight3D:
    """A figure-eight traced on a shell around the listener.

    Drawn in azimuth and elevation at a constant distance rather than as a
    Cartesian lemniscate.  A Cartesian figure-eight centred on the listener
    crosses through the head twice per cycle, where distance is zero, gain is
    unbounded and azimuth is undefined; on a shell the crossing is a direction
    like any other, and the whole path stays on the surface an HRTF dataset
    actually measured.

    ``tilt_deg`` rotates the figure within the azimuth/elevation plane.  At 0
    the lobes are left and right and the source alternates above and below
    within each lobe; near 45 the figure runs diagonally, giving the
    above-left to below-right alternation this application asks for.
    """

    azimuth_extent_deg: float = 60.0
    elevation_extent_deg: float = 30.0
    distance_m: float = 1.5
    tilt_deg: float = 0.0
    centre_azimuth_deg: float = 0.0
    centre_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.distance_m, "distance_m")
        _finite(self.azimuth_extent_deg, "azimuth_extent_deg")
        _finite(self.elevation_extent_deg, "elevation_extent_deg")
        _finite(self.tilt_deg, "tilt_deg")
        _finite(self.centre_azimuth_deg, "centre_azimuth_deg")
        _elevation(self.centre_elevation_deg, "centre_elevation_deg")

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        angle = 2.0 * np.pi * _parameter(u)
        # Gerono lemniscate: the doubled frequency on one axis is what closes
        # the curve into two lobes rather than one loop.
        across = self.azimuth_extent_deg * np.sin(angle)
        along = self.elevation_extent_deg * np.sin(2.0 * angle)
        tilt = math.radians(self.tilt_deg)
        azimuth = self.centre_azimuth_deg + across * math.cos(tilt) - along * math.sin(tilt)
        elevation = self.centre_elevation_deg + across * math.sin(tilt) + along * math.cos(tilt)
        # Elevation is a polar angle, not a free axis: a tilted figure with a
        # large extent would otherwise ask for directions past the zenith.
        return spherical_to_cartesian(
            azimuth, np.clip(elevation, -90.0, 90.0), self.distance_m
        )


@dataclass(frozen=True)
class Pendulum:
    """A swing along an arc, with the pendulum's own rise at the extremes.

    The height comes from the geometry rather than from a separate height
    curve: a bob on a string is highest at the ends of its swing and lowest in
    the middle, which is what makes this read as a pendulum instead of as a
    left-right pan.
    """

    length_m: float = 1.5
    swing_deg: float = 60.0
    #: The horizontal direction the swing plane runs along.
    plane_azimuth_deg: float = 90.0
    pivot_m: tuple[float, float, float] = (0.0, 0.0, 1.0)
    swings: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.length_m, "length_m")
        _finite(self.swing_deg, "swing_deg")
        _finite(self.plane_azimuth_deg, "plane_azimuth_deg")
        _positive(self.swings, "swings")
        _centre(self.pivot_m)

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)
        angle = np.radians(self.swing_deg) * np.sin(
            2.0 * np.pi * self.swings * progress
        )
        plane = math.radians(self.plane_azimuth_deg)
        along = self.length_m * np.sin(angle)
        return (
            np.stack(
                (
                    along * math.cos(plane),
                    along * math.sin(plane),
                    -self.length_m * np.cos(angle),
                ),
                axis=-1,
            )
            + _centre(self.pivot_m)
        )


@dataclass(frozen=True)
class Torus:
    """A small circle carried around a large one - a path around the head.

    The source winds ``minor_turns`` times through a tube of radius
    ``minor_radius_m`` while that tube is carried once around the listener at
    ``major_radius_m``.  With a few minor turns per major one it weaves above
    and below ear height as it orbits, which is the toroidal path this
    application wants.
    """

    major_radius_m: float = 1.5
    minor_radius_m: float = 0.4
    major_turns: float = 1.0
    minor_turns: float = 6.0
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _positive(self.major_radius_m, "major_radius_m")
        _positive(self.minor_radius_m, "minor_radius_m")
        _finite(self.major_turns, "major_turns")
        _finite(self.minor_turns, "minor_turns")
        _centre(self.centre_m)
        if self.minor_radius_m >= self.major_radius_m:
            raise ValueError(
                "minor_radius_m must be smaller than major_radius_m, or the "
                "path passes through the listener"
            )

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = _parameter(u)
        major = 2.0 * np.pi * self.major_turns * progress
        minor = 2.0 * np.pi * self.minor_turns * progress
        radius = self.major_radius_m + self.minor_radius_m * np.cos(minor)
        return (
            np.stack(
                (
                    radius * np.cos(major),
                    radius * np.sin(major),
                    self.minor_radius_m * np.sin(minor),
                ),
                axis=-1,
            )
            + _centre(self.centre_m)
        )


@dataclass(frozen=True)
class RandomWalkVolume:
    """A seeded wander confined to a box around the listener.

    Determinism matters more here than it looks: an export has to match its
    preview, and blocks are rendered out of order, so the walk is built from a
    fixed number of seeded steps and interpolated rather than accumulated
    sample by sample.  The same seed always produces the same path.
    """

    extent_m: tuple[float, float, float] = (2.0, 2.0, 1.0)
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int = 16
    seed: int = 0
    #: Keeps the walk outside a sphere of this radius, so it never lands inside
    #: the listener's head.
    minimum_distance_m: float = 0.3
    smooth: bool = True

    def __post_init__(self) -> None:
        extent = _centre(self.extent_m)
        if np.any(extent <= 0.0):
            raise ValueError("extent_m components must be positive")
        _centre(self.centre_m)
        if int(self.steps) < 2:
            raise ValueError("steps must be at least two")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if float(self.minimum_distance_m) < 0.0:
            raise ValueError("minimum_distance_m must not be negative")

    def _waypoints(self) -> NDArray[np.float64]:
        generator = np.random.default_rng(int(self.seed))
        count = int(self.steps)
        extent = _centre(self.extent_m)
        # A random walk rather than independent draws: successive points are
        # near each other, which is what makes it read as motion.
        increments = generator.normal(size=(count, 3)) * (extent / 4.0)
        points = np.cumsum(increments, axis=0)
        # Reflect back into the box instead of clipping, so the walk turns at
        # the wall rather than sliding along it.
        half = extent / 2.0
        folded = np.abs((points + half) % (4.0 * half) - 2.0 * half) - half
        points = folded + _centre(self.centre_m)
        radius = np.linalg.norm(points - _centre(self.centre_m), axis=-1, keepdims=True)
        minimum = float(self.minimum_distance_m)
        if minimum > 0.0:
            scale = np.where(radius < minimum, minimum / np.maximum(radius, 1e-9), 1.0)
            points = _centre(self.centre_m) + (points - _centre(self.centre_m)) * scale
        # Close the loop so a looping traversal does not jump on wrap.
        return np.vstack((points, points[:1]))

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(_parameter(u), 0.0, 1.0)
        points = self._waypoints()
        knots = np.linspace(0.0, 1.0, len(points))
        if not self.smooth:
            result = np.stack(
                [np.interp(progress, knots, points[:, axis]) for axis in range(3)],
                axis=-1,
            )
        else:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(knots, points, axis=0, bc_type="periodic")
            result = np.asarray(spline(progress), dtype=np.float64)
        # Enforced on the curve, not just on the waypoints it passes through:
        # a spline between two points that each clear the minimum can still bow
        # inside it, and a path that dips into the listener's head is exactly
        # what the minimum exists to prevent.
        return self._enforce_minimum(result)

    def _enforce_minimum(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        minimum = float(self.minimum_distance_m)
        if minimum <= 0.0:
            return points
        centre = _centre(self.centre_m)
        offset = points - centre
        radius = np.linalg.norm(offset, axis=-1, keepdims=True)
        scale = np.where(radius < minimum, minimum / np.maximum(radius, 1e-9), 1.0)
        return centre + offset * scale


#: Serialized names for the 3-D primitives, in the order the editor lists them.
PRIMITIVE_TYPES: tuple[str, ...] = (
    "horizontal_orbit",
    "vertical_orbit",
    "tilted_orbit",
    "spherical_orbit",
    "rising_arc",
    "overhead_sweep",
    "elevation_sweep",
    "dome_traversal",
    "figure_eight_3d",
    "pendulum",
    "torus",
    "random_walk_volume",
)

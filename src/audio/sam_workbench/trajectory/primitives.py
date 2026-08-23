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

Each primitive's mathematics lives in a module-level function taking explicit
parameter values rather than in the dataclass body.  The frozen dataclasses are
the stored, validated form; the functions are the evaluated form, and they also
serve :mod:`.dynamic`, which drives the same numbers with time-varying values.
One formula per shape means a modulated orbit and a static one can never drift
apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spherical import spherical_to_cartesian
from .transforms import apply_stacked, rotation_matrices_ypr

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


def horizontal_orbit_points(
    u: ArrayLike,
    *,
    radius_m: ArrayLike,
    elevation_deg: ArrayLike,
    start_azimuth_deg: ArrayLike,
    turns: ArrayLike,
    centre_m: ArrayLike,
) -> NDArray[np.float64]:
    azimuth = start_azimuth_deg + 360.0 * turns * _parameter(u)
    return (
        spherical_to_cartesian(azimuth, elevation_deg, radius_m)
        + np.asarray(centre_m, dtype=np.float64)
    )


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
        return horizontal_orbit_points(
            u,
            radius_m=self.radius_m,
            elevation_deg=self.elevation_deg,
            start_azimuth_deg=self.start_azimuth_deg,
            turns=self.turns,
            centre_m=self.centre_m,
        )


def vertical_orbit_points(
    u: ArrayLike,
    *,
    radius_m: ArrayLike,
    plane_azimuth_deg: ArrayLike,
    start_angle_deg: ArrayLike,
    turns: ArrayLike,
    centre_m: ArrayLike,
) -> NDArray[np.float64]:
    angle = np.radians(start_angle_deg + 360.0 * turns * _parameter(u))
    plane = np.radians(plane_azimuth_deg)
    horizontal = radius_m * np.cos(angle)
    return (
        np.stack(
            (
                horizontal * np.cos(plane),
                horizontal * np.sin(plane),
                radius_m * np.sin(angle),
            ),
            axis=-1,
        )
        + np.asarray(centre_m, dtype=np.float64)
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
        return vertical_orbit_points(
            u,
            radius_m=self.radius_m,
            plane_azimuth_deg=self.plane_azimuth_deg,
            start_angle_deg=self.start_angle_deg,
            turns=self.turns,
            centre_m=self.centre_m,
        )


def tilted_orbit_points(
    u: ArrayLike,
    *,
    radius_m: ArrayLike,
    tilt_deg: ArrayLike,
    tilt_axis_azimuth_deg: ArrayLike,
    start_azimuth_deg: ArrayLike,
    turns: ArrayLike,
    centre_m: ArrayLike,
) -> NDArray[np.float64]:
    angle = np.radians(start_azimuth_deg + 360.0 * turns * _parameter(u))
    flat = np.stack(
        (
            radius_m * np.cos(angle),
            radius_m * np.sin(angle),
            np.zeros_like(angle),
        ),
        axis=-1,
    )
    # Tip the ring about a horizontal axis pointing along the tilt azimuth:
    # rotate that axis onto +y, pitch, then rotate back.
    matrix = rotation_matrices_ypr(
        yaw_deg=-tilt_axis_azimuth_deg, pitch_deg=tilt_deg
    )
    if matrix.ndim == 2:
        tilted = flat @ matrix.T
    else:
        tilted = apply_stacked(flat, matrix)
    return tilted + np.asarray(centre_m, dtype=np.float64)


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
        return tilted_orbit_points(
            u,
            radius_m=self.radius_m,
            tilt_deg=self.tilt_deg,
            tilt_axis_azimuth_deg=self.tilt_axis_azimuth_deg,
            start_azimuth_deg=self.start_azimuth_deg,
            turns=self.turns,
            centre_m=self.centre_m,
        )


def spherical_orbit_points(
    u: ArrayLike,
    *,
    start_distance_m: ArrayLike,
    end_distance_m: ArrayLike,
    elevation_deg: ArrayLike,
    turns: ArrayLike,
    start_azimuth_deg: ArrayLike,
    cycles: ArrayLike,
) -> NDArray[np.float64]:
    progress = _parameter(u)
    azimuth = start_azimuth_deg + 360.0 * turns * progress
    # A raised cosine breathes out and back in over each cycle, so a looped
    # traversal returns to its starting radius without a jump.
    breath = 0.5 - 0.5 * np.cos(2.0 * np.pi * cycles * progress)
    distance = start_distance_m + (end_distance_m - start_distance_m) * breath
    return spherical_to_cartesian(azimuth, elevation_deg, distance)


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
        return spherical_orbit_points(
            u,
            start_distance_m=self.start_distance_m,
            end_distance_m=self.end_distance_m,
            elevation_deg=self.elevation_deg,
            turns=self.turns,
            start_azimuth_deg=self.start_azimuth_deg,
            cycles=self.cycles,
        )


# --- sweeps ----------------------------------------------------------------


def rising_arc_points(
    u: ArrayLike,
    *,
    start_azimuth_deg: ArrayLike,
    end_azimuth_deg: ArrayLike,
    start_elevation_deg: ArrayLike,
    end_elevation_deg: ArrayLike,
    start_distance_m: ArrayLike,
    end_distance_m: ArrayLike,
) -> NDArray[np.float64]:
    progress = np.clip(_parameter(u), 0.0, 1.0)

    def between(start: ArrayLike, end: ArrayLike) -> NDArray[np.float64]:
        return start + (end - start) * progress

    return spherical_to_cartesian(
        between(start_azimuth_deg, end_azimuth_deg),
        between(start_elevation_deg, end_elevation_deg),
        between(start_distance_m, end_distance_m),
    )


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
        return rising_arc_points(
            u,
            start_azimuth_deg=self.start_azimuth_deg,
            end_azimuth_deg=self.end_azimuth_deg,
            start_elevation_deg=self.start_elevation_deg,
            end_elevation_deg=self.end_elevation_deg,
            start_distance_m=self.start_distance_m,
            end_distance_m=self.end_distance_m,
        )


def overhead_sweep_points(
    u: ArrayLike,
    *,
    azimuth_deg: ArrayLike,
    start_elevation_deg: ArrayLike,
    distance_m: ArrayLike,
    pass_over: bool,
    end_elevation_deg: ArrayLike,
) -> NDArray[np.float64]:
    progress = np.clip(_parameter(u), 0.0, 1.0)
    if not pass_over:
        elevation = start_elevation_deg + (90.0 - start_elevation_deg) * progress
        return spherical_to_cartesian(azimuth_deg, elevation, distance_m)
    # Rise to the zenith over the first half, descend behind over the
    # second. Azimuth flips at the top, where it is undefined anyway, so
    # the position stays continuous through the crossing.
    rising = progress <= 0.5
    elevation = np.where(
        rising,
        start_elevation_deg + (90.0 - start_elevation_deg) * (2.0 * progress),
        90.0 - (90.0 - end_elevation_deg) * (2.0 * progress - 1.0),
    )
    azimuth = np.where(rising, azimuth_deg, azimuth_deg + 180.0)
    return spherical_to_cartesian(azimuth, elevation, distance_m)


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
        return overhead_sweep_points(
            u,
            azimuth_deg=self.azimuth_deg,
            start_elevation_deg=self.start_elevation_deg,
            distance_m=self.distance_m,
            pass_over=self.pass_over,
            end_elevation_deg=self.end_elevation_deg,
        )


def elevation_sweep_points(
    u: ArrayLike,
    *,
    start_azimuth_deg: ArrayLike,
    end_azimuth_deg: ArrayLike,
    elevation_deg: ArrayLike,
    distance_m: ArrayLike,
    end_elevation_deg: ArrayLike | None,
) -> NDArray[np.float64]:
    progress = np.clip(_parameter(u), 0.0, 1.0)
    resolved_end = elevation_deg if end_elevation_deg is None else end_elevation_deg
    return spherical_to_cartesian(
        start_azimuth_deg + (end_azimuth_deg - start_azimuth_deg) * progress,
        elevation_deg + (resolved_end - elevation_deg) * progress,
        distance_m,
    )


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
        return elevation_sweep_points(
            u,
            start_azimuth_deg=self.start_azimuth_deg,
            end_azimuth_deg=self.end_azimuth_deg,
            elevation_deg=self.elevation_deg,
            distance_m=self.distance_m,
            end_elevation_deg=self.end_elevation_deg,
        )


def dome_traversal_points(
    u: ArrayLike,
    *,
    distance_m: ArrayLike,
    start_elevation_deg: ArrayLike,
    end_elevation_deg: ArrayLike,
    turns: ArrayLike,
    start_azimuth_deg: ArrayLike,
) -> NDArray[np.float64]:
    progress = np.clip(_parameter(u), 0.0, 1.0)
    return spherical_to_cartesian(
        start_azimuth_deg + 360.0 * turns * progress,
        start_elevation_deg + (end_elevation_deg - start_elevation_deg) * progress,
        distance_m,
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
        return dome_traversal_points(
            u,
            distance_m=self.distance_m,
            start_elevation_deg=self.start_elevation_deg,
            end_elevation_deg=self.end_elevation_deg,
            turns=self.turns,
            start_azimuth_deg=self.start_azimuth_deg,
        )


# --- figures ---------------------------------------------------------------


def figure_eight_3d_points(
    u: ArrayLike,
    *,
    azimuth_extent_deg: ArrayLike,
    elevation_extent_deg: ArrayLike,
    distance_m: ArrayLike,
    tilt_deg: ArrayLike,
    centre_azimuth_deg: ArrayLike,
    centre_elevation_deg: ArrayLike,
) -> NDArray[np.float64]:
    angle = 2.0 * np.pi * _parameter(u)
    # Gerono lemniscate: the doubled frequency on one axis is what closes
    # the curve into two lobes rather than one loop.
    across = azimuth_extent_deg * np.sin(angle)
    along = elevation_extent_deg * np.sin(2.0 * angle)
    tilt = np.radians(tilt_deg)
    azimuth = centre_azimuth_deg + across * np.cos(tilt) - along * np.sin(tilt)
    elevation = centre_elevation_deg + across * np.sin(tilt) + along * np.cos(tilt)
    # Elevation is a polar angle, not a free axis: a tilted figure with a
    # large extent would otherwise ask for directions past the zenith.
    return spherical_to_cartesian(
        azimuth, np.clip(elevation, -90.0, 90.0), distance_m
    )


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
        return figure_eight_3d_points(
            u,
            azimuth_extent_deg=self.azimuth_extent_deg,
            elevation_extent_deg=self.elevation_extent_deg,
            distance_m=self.distance_m,
            tilt_deg=self.tilt_deg,
            centre_azimuth_deg=self.centre_azimuth_deg,
            centre_elevation_deg=self.centre_elevation_deg,
        )


def pendulum_points(
    u: ArrayLike,
    *,
    length_m: ArrayLike,
    swing_deg: ArrayLike,
    plane_azimuth_deg: ArrayLike,
    pivot_m: ArrayLike,
    swings: ArrayLike,
) -> NDArray[np.float64]:
    progress = _parameter(u)
    angle = np.radians(swing_deg) * np.sin(2.0 * np.pi * swings * progress)
    plane = np.radians(plane_azimuth_deg)
    along = length_m * np.sin(angle)
    return (
        np.stack(
            (
                along * np.cos(plane),
                along * np.sin(plane),
                -length_m * np.cos(angle),
            ),
            axis=-1,
        )
        + np.asarray(pivot_m, dtype=np.float64)
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
        return pendulum_points(
            u,
            length_m=self.length_m,
            swing_deg=self.swing_deg,
            plane_azimuth_deg=self.plane_azimuth_deg,
            pivot_m=self.pivot_m,
            swings=self.swings,
        )


def torus_points(
    u: ArrayLike,
    *,
    major_radius_m: ArrayLike,
    minor_radius_m: ArrayLike,
    major_turns: ArrayLike,
    minor_turns: ArrayLike,
    centre_m: ArrayLike,
) -> NDArray[np.float64]:
    progress = _parameter(u)
    major = 2.0 * np.pi * major_turns * progress
    minor = 2.0 * np.pi * minor_turns * progress
    radius = major_radius_m + minor_radius_m * np.cos(minor)
    return (
        np.stack(
            (
                radius * np.cos(major),
                radius * np.sin(major),
                minor_radius_m * np.sin(minor),
            ),
            axis=-1,
        )
        + np.asarray(centre_m, dtype=np.float64)
    )


@dataclass(frozen=True)
class Torus:
    """A small circle carried around a large one - a path around the head.

    The source winds ``minor_turns`` times through a tube of radius
    ``minor_radius_m`` while that tube is carried ``major_turns`` times around
    the listener at ``major_radius_m``.  With a few minor turns per major one it
    weaves above and below ear height as it orbits, which is the toroidal path
    this application wants.

    ``major_turns`` is how many complete orbits of the listener the traversal
    makes, not a drawing resolution: raising it multiplies the angular speed for
    the same traversal duration.  Twenty-one turns in one second sweeps about
    22 degrees per control interval, far past what an HRTF grid resolves, and
    the render is heard as smeared rather than as motion.  For a single orbit
    that winds twenty-one times vertically, put the twenty-one on
    ``minor_turns`` instead; to keep twenty-one orbits, lengthen the traversal
    in proportion.
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
        return torus_points(
            u,
            major_radius_m=self.major_radius_m,
            minor_radius_m=self.minor_radius_m,
            major_turns=self.major_turns,
            minor_turns=self.minor_turns,
            centre_m=self.centre_m,
        )


def random_walk_points(
    u: ArrayLike,
    *,
    extent_m: ArrayLike,
    centre_m: ArrayLike,
    steps: int,
    seed: int,
    minimum_distance_m: ArrayLike,
    smooth: bool,
) -> NDArray[np.float64]:
    progress = np.clip(_parameter(u), 0.0, 1.0)

    extent_array = np.asarray(extent_m, dtype=np.float64)
    centre_array = np.asarray(centre_m, dtype=np.float64)
    minimum_array = np.asarray(minimum_distance_m, dtype=np.float64)
    if extent_array.ndim == 1 and centre_array.ndim == 1 and minimum_array.ndim == 0:
        return _random_walk_row(
            progress,
            extent=extent_array,
            centre=centre_array,
            steps=int(steps),
            seed=int(seed),
            minimum=minimum_array,
            smooth=bool(smooth),
        )
    # Time-varying volume parameters rebuild the seeded waypoint set per query,
    # so each time keeps its own box. Called once per control-grid window, the
    # row-wise cost is bounded by the window's grid resolution.
    flat = np.asarray(progress).reshape(-1)
    rows = [
        _random_walk_row(
            flat[index : index + 1],
            extent=extent_array.reshape(-1)[index * 3 : index * 3 + 3]
            if extent_array.ndim == 2
            else extent_array,
            centre=centre_array.reshape(-1)[index * 3 : index * 3 + 3]
            if centre_array.ndim == 2
            else centre_array,
            steps=int(steps),
            seed=int(seed),
            minimum=minimum_array.reshape(-1)[index]
            if minimum_array.ndim
            else minimum_array,
            smooth=bool(smooth),
        )
        for index in range(len(flat))
    ]
    return np.concatenate(rows, axis=0).reshape(*np.shape(progress), 3)


def _random_walk_row(
    progress: NDArray[np.float64],
    *,
    extent: NDArray[np.float64],
    centre: NDArray[np.float64],
    steps: int,
    seed: int,
    minimum: NDArray[np.float64],
    smooth: bool,
) -> NDArray[np.float64]:
    points = _random_walk_waypoints(
        extent=extent, centre=centre, steps=steps, seed=seed, minimum=minimum
    )
    knots = np.linspace(0.0, 1.0, len(points))
    if not smooth:
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
    return _enforce_random_walk_minimum(result, centre=centre, minimum=minimum)


def _random_walk_waypoints(
    *,
    extent: NDArray[np.float64],
    centre: NDArray[np.float64],
    steps: int,
    seed: int,
    minimum: NDArray[np.float64],
) -> NDArray[np.float64]:
    generator = np.random.default_rng(int(seed))
    count = int(steps)
    # A random walk rather than independent draws: successive points are
    # near each other, which is what makes it read as motion.
    increments = generator.normal(size=(count, 3)) * (extent / 4.0)
    points = np.cumsum(increments, axis=0)
    # Reflect back into the box instead of clipping, so the walk turns at
    # the wall rather than sliding along it.
    half = extent / 2.0
    folded = np.abs((points + half) % (4.0 * half) - 2.0 * half) - half
    points = folded + centre
    radius = np.linalg.norm(points - centre, axis=-1, keepdims=True)
    limit = float(np.max(minimum))
    if limit > 0.0:
        scale = np.where(radius < limit, limit / np.maximum(radius, 1e-9), 1.0)
        points = centre + (points - centre) * scale
    # Close the loop so a looping traversal does not jump on wrap.
    return np.vstack((points, points[:1]))


def _enforce_random_walk_minimum(
    points: NDArray[np.float64],
    *,
    centre: NDArray[np.float64],
    minimum: NDArray[np.float64],
) -> NDArray[np.float64]:
    minimum_value = float(np.max(minimum))
    if minimum_value <= 0.0:
        return points
    offset = points - centre
    radius = np.linalg.norm(offset, axis=-1, keepdims=True)
    scale = np.where(radius < minimum_value, minimum_value / np.maximum(radius, 1e-9), 1.0)
    return centre + offset * scale


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
        return _random_walk_waypoints(
            extent=_centre(self.extent_m),
            centre=_centre(self.centre_m),
            steps=int(self.steps),
            seed=int(self.seed),
        )

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        return random_walk_points(
            u,
            extent_m=self.extent_m,
            centre_m=self.centre_m,
            steps=int(self.steps),
            seed=int(self.seed),
            minimum_distance_m=float(self.minimum_distance_m),
            smooth=bool(self.smooth),
        )

    def _enforce_minimum(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return _enforce_random_walk_minimum(
            points,
            centre=_centre(self.centre_m),
            minimum=np.asarray(float(self.minimum_distance_m)),
        )


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

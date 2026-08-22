"""Keyframed Cartesian positions - the recommended way to author a path.

A keyframed path is a list of ``(timeSeconds, position)`` pairs.  It is the
most direct expression of "the source is *here* at *this* moment", it is what a
recording of real motion produces, and it is what a CSV or JSON import lands
in.  Everything else in the primitive library is a generator for one of these.

Keyframes carry their own times, so they describe geometry *and* a traversal
together.  The rest of the package keeps those apart, and this module keeps
them apart too: :class:`KeyframedPath` is geometry, parameterized by normalized
progress like every other geometry, and :meth:`KeyframedPath.traversal` returns
the time law that replays it at the authored times.  Swapping that traversal
for another one re-times the same shape rather than destroying it - which is
what makes "the same path, but accelerating" possible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spherical import SphericalPosition, spherical_to_cartesian
from .traversal import KeyframedTraversal

__all__ = [
    "Keyframe",
    "KeyframedPath",
    "keyframes_from_csv",
    "keyframes_from_json",
    "keyframes_from_arrays",
    "load_keyframes",
]

#: Interpolation kinds a keyframed path understands.
KEYFRAME_INTERPOLATIONS: tuple[str, ...] = ("hold", "linear", "cubic", "catmull_rom")


@dataclass(frozen=True)
class Keyframe:
    """One authored position and the time it is reached."""

    time_s: float
    position_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_s)):
            raise ValueError("keyframe timeSeconds must be finite")
        if len(self.position_m) != 3 or not all(
            math.isfinite(float(value)) for value in self.position_m
        ):
            raise ValueError("keyframe position must contain three finite metres")

    def describe(self) -> dict[str, Any]:
        return {
            "timeSeconds": float(self.time_s),
            "position": [float(value) for value in self.position_m],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Keyframe":
        """Accept a Cartesian ``position`` or a spherical entry, never both.

        Spherical input is converted here, at the boundary, so nothing
        downstream has to know two position formats.
        """

        time_s = float(
            data.get("timeSeconds", data.get("timeS", data.get("time", 0.0)))
        )
        position = data.get("position", data.get("positionM"))
        spherical_keys = {
            "azimuthDegrees",
            "azimuthDeg",
            "elevationDegrees",
            "elevationDeg",
            "distanceMetres",
            "distanceM",
        }
        has_spherical = bool(spherical_keys & set(data))
        if position is not None and has_spherical:
            raise ValueError(
                "a keyframe carries either a Cartesian position or spherical "
                "azimuth/elevation/distance, not both"
            )
        if position is None:
            if not has_spherical:
                raise ValueError("a keyframe needs a position")
            position = SphericalPosition.from_mapping(data).to_cartesian()
        return cls(time_s, tuple(float(value) for value in position))


@dataclass(frozen=True)
class KeyframedPath:
    """Geometry through timestamped control positions.

    ``evaluate`` takes normalized progress in ``[0, 1]`` like every other
    geometry, where ``0`` is the first keyframe's time and ``1`` the last.
    Keyframe times are honoured: a key at 9 s out of 10 sits at ``u = 0.9``,
    so uneven spacing is a slower or faster stretch of path, not a redistributed
    one.
    """

    keyframes: tuple[Keyframe, ...]
    interpolation: str = "cubic"

    def __post_init__(self) -> None:
        if len(self.keyframes) < 2:
            raise ValueError("a keyframed path needs at least two keyframes")
        if self.interpolation not in KEYFRAME_INTERPOLATIONS:
            raise ValueError(
                f"unknown keyframe interpolation {self.interpolation!r}; "
                f"expected one of {KEYFRAME_INTERPOLATIONS}"
            )
        times = self.times_s
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("keyframe times must strictly increase")

    @property
    def times_s(self) -> NDArray[np.float64]:
        return np.asarray([key.time_s for key in self.keyframes], dtype=np.float64)

    @property
    def positions_m(self) -> NDArray[np.float64]:
        return np.asarray([key.position_m for key in self.keyframes], dtype=np.float64)

    @property
    def start_time_s(self) -> float:
        return float(self.keyframes[0].time_s)

    @property
    def duration_s(self) -> float:
        span = float(self.keyframes[-1].time_s) - self.start_time_s
        # A zero span is impossible - times strictly increase - but a degenerate
        # single-instant path would divide by zero here, so keep it positive.
        return max(span, np.finfo(np.float64).eps)

    def at_time(self, time_s: ArrayLike) -> NDArray[np.float64]:
        """Positions at absolute authored times, clamped to the key range."""

        times = np.asarray(time_s, dtype=np.float64)
        progress = (times - self.start_time_s) / self.duration_s
        return self.evaluate(progress)

    def traversal(self, **overrides: Any) -> KeyframedTraversal:
        """The time law that replays these keys at the times they were authored.

        Uneven key spacing is preserved: progress advances between keys in
        proportion to the wall-clock gap, so the source is where the author put
        it at the moment they put it there.
        """

        times = self.times_s
        normalized = (times - self.start_time_s) / self.duration_s
        return KeyframedTraversal(
            times_s=tuple(float(value) for value in times),
            positions=tuple(float(value) for value in normalized),
            interpolation=str(overrides.get("interpolation", "linear")),
        )

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        progress = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        if not np.all(np.isfinite(progress)):
            raise ValueError("path parameter must be finite")
        times = self.times_s
        points = self.positions_m
        # Map progress back onto the authored time axis so that unevenly spaced
        # keys keep their spacing.
        target = self.start_time_s + progress * self.duration_s

        if self.interpolation == "hold":
            index = np.clip(
                np.searchsorted(times, target, side="right") - 1, 0, len(times) - 1
            )
            return points[index]
        if self.interpolation == "linear":
            return np.stack(
                [np.interp(target, times, points[:, axis]) for axis in range(3)],
                axis=-1,
            )
        if self.interpolation == "cubic":
            from scipy.interpolate import CubicSpline

            # ``natural`` rather than the default ``not-a-knot`` so three keys
            # do not overshoot into a curve the author never drew.
            spline = CubicSpline(times, points, axis=0, bc_type="natural")
            return np.asarray(spline(target), dtype=np.float64)
        return _catmull_rom(times, points, target)


def _catmull_rom(
    times: NDArray[np.float64], points: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Centripetal-style Catmull-Rom through the keys, clamped at the ends.

    Unlike a fitted cubic spline this passes through every control point with
    local support: moving one key changes only the two segments beside it,
    which is what makes dragging a point in the editor feel predictable.
    """

    count = len(points)
    index = np.clip(np.searchsorted(times, target, side="right") - 1, 0, count - 2)
    span = times[index + 1] - times[index]
    t = np.clip((target - times[index]) / span, 0.0, 1.0)[..., None]

    p0 = points[np.maximum(index - 1, 0)]
    p1 = points[index]
    p2 = points[index + 1]
    p3 = points[np.minimum(index + 2, count - 1)]
    # Uniform tangents scaled to the local span keep uneven key spacing from
    # producing a velocity discontinuity at every key.
    m1 = 0.5 * (p2 - p0)
    m2 = 0.5 * (p3 - p1)

    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * p1
        + (t3 - 2.0 * t2 + t) * m1
        + (-2.0 * t3 + 3.0 * t2) * p2
        + (t3 - t2) * m2
    )


# --- import ----------------------------------------------------------------


def keyframes_from_arrays(
    times_s: ArrayLike, positions_m: ArrayLike
) -> tuple[Keyframe, ...]:
    """Build keyframes from parallel arrays, as a motion recorder produces."""

    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    points = np.asarray(positions_m, dtype=np.float64)
    if points.shape != (times.size, 3):
        raise ValueError(
            f"positions must be ({times.size}, 3) to match the times, got {points.shape}"
        )
    return tuple(
        Keyframe(float(time), tuple(float(value) for value in point))
        for time, point in zip(times, points)
    )


def keyframes_from_json(payload: str | bytes | Mapping[str, Any] | Sequence[Any]):
    """Read keyframes from the documented JSON form or a bare list of keys."""

    data: Any = payload
    if isinstance(payload, (str, bytes)):
        data = json.loads(payload)
    if isinstance(data, Mapping):
        data = data.get("keyframes", data.get("points", ()))
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("JSON keyframes must be a list of keyframe objects")
    return tuple(Keyframe.from_mapping(dict(entry)) for entry in data)


#: Column names accepted for each quantity, case- and space-insensitive.
_CSV_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "time": ("time", "times", "t", "timeseconds", "time_s", "seconds"),
    "x": ("x", "xm", "x_m", "forward"),
    "y": ("y", "ym", "y_m", "left"),
    "z": ("z", "zm", "z_m", "up", "height"),
    "azimuth": ("azimuth", "azimuthdegrees", "azimuthdeg", "az"),
    "elevation": ("elevation", "elevationdegrees", "elevationdeg", "el"),
    "distance": ("distance", "distancemetres", "distancem", "radius", "r"),
}


def _normalize(name: str) -> str:
    return "".join(name.split()).replace("-", "").replace("_", "").lower()


def keyframes_from_csv(text: str, *, default_interval_s: float = 0.1):
    """Read keyframes from CSV, in either Cartesian or spherical columns.

    A header row names the columns; ``time`` is optional, and rows without one
    are spaced ``default_interval_s`` apart so a bare list of positions still
    imports.  Headerless files are read as ``time, x, y, z`` or ``x, y, z``.
    """

    rows = [row for row in csv.reader(io.StringIO(text)) if row and any(row)]
    if not rows:
        raise ValueError("CSV contains no rows")

    header = [_normalize(cell) for cell in rows[0]]
    lookup: dict[str, int] = {}
    for quantity, aliases in _CSV_COLUMNS.items():
        for position, cell in enumerate(header):
            if cell in aliases:
                lookup[quantity] = position
                break

    if {"x", "y", "z"} <= set(lookup):
        mode, body = "cartesian", rows[1:]
    elif {"azimuth", "elevation"} <= set(lookup):
        mode, body = "spherical", rows[1:]
    else:
        # No usable header: fall back to positional columns.
        width = len(rows[0])
        if width >= 4:
            lookup = {"time": 0, "x": 1, "y": 2, "z": 3}
        elif width == 3:
            lookup = {"x": 0, "y": 1, "z": 2}
        else:
            raise ValueError(
                "CSV needs x/y/z or azimuth/elevation columns, or three or "
                "four positional columns"
            )
        mode, body = "cartesian", rows

    if not body:
        raise ValueError("CSV contains a header but no data rows")

    times: list[float] = []
    points: list[tuple[float, float, float]] = []
    for number, row in enumerate(body):
        def cell(quantity: str, default: float = 0.0) -> float:
            index = lookup.get(quantity)
            if index is None or index >= len(row) or not row[index].strip():
                return default
            try:
                return float(row[index])
            except ValueError as error:
                raise ValueError(
                    f"CSV row {number + 1}: {quantity} is not a number: {row[index]!r}"
                ) from error

        if mode == "cartesian":
            point = (cell("x"), cell("y"), cell("z"))
        else:
            point = tuple(
                float(value)
                for value in spherical_to_cartesian(
                    cell("azimuth"), cell("elevation"), cell("distance", 1.0)
                )
            )
        times.append(cell("time", number * float(default_interval_s)))
        points.append(point)  # type: ignore[arg-type]

    if len(times) < 2:
        raise ValueError("a keyframed path needs at least two rows")
    return keyframes_from_arrays(times, points)


def load_keyframes(path: str | Path, **options: Any) -> tuple[Keyframe, ...]:
    """Read keyframes from a ``.csv`` or ``.json`` file on disk."""

    location = Path(path)
    text = location.read_text(encoding="utf-8")
    if location.suffix.lower() == ".json":
        return keyframes_from_json(text)
    if location.suffix.lower() in (".csv", ".tsv", ".txt"):
        return keyframes_from_csv(text, **options)
    raise ValueError(f"unsupported keyframe file type: {location.suffix!r}")

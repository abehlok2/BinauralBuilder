"""Compile the versioned trajectory dictionaries stored in voice parameters."""

from __future__ import annotations

from typing import Any, Mapping

from .geometry import (
    Arc,
    Bezier,
    Circle,
    Ellipse,
    Helix,
    Line,
    Lissajous,
    Mathematical,
    Polygon,
    Polyline,
    Spiral,
    Spline,
)
from .traversal import CanonicalTrajectory, Traversal

#: Geometry names the serialized form understands. The editor seeds most of
#: these as control points and lets them be dragged, so what comes back is
#: whatever shape is on screen rather than the primitive it started as; only a
#: payload that carries no points at all is rebuilt from the parametric class.
GEOMETRY_TYPES: tuple[str, ...] = (
    "polyline",
    "polygon",
    "spline",
    "bezier",
    "line",
    "arc",
    "circle",
    "ellipse",
    "spiral",
    "helix",
    "lissajous",
    "mathematical",
)

#: Kinds the editor seeds as smooth curves. Dragged points describe a curve, so
#: a spline through them is what reproduces the shape a user is looking at.
_SMOOTH_KINDS = frozenset({"arc", "circle", "ellipse", "spiral", "helix", "lissajous"})

#: What each parametric kind falls back to when a payload names it but carries
#: no points - a project written by hand, or by something other than the editor.
_PARAMETRIC: dict[str, Any] = {
    "line": Line,
    "arc": Arc,
    "circle": Circle,
    "ellipse": Ellipse,
    "spiral": Spiral,
    "helix": Helix,
    "lissajous": Lissajous,
}

#: Kinds whose seeded shape closes back on itself.
_CLOSED_BY_NATURE = frozenset({"circle", "ellipse", "lissajous", "polygon"})


def _from_points(kind: str, points: tuple, closed: bool):
    """Reproduce a seeded primitive from the points that represent it."""

    shut = closed or kind in _CLOSED_BY_NATURE
    if kind == "line" and len(points) == 2:
        return Line(points[0], points[1])
    if kind == "polygon" and len(points) >= 3:
        return Polygon(points)
    if kind in _SMOOTH_KINDS and len(points) >= 3:
        return Spline(points, shut)
    return Polyline(points, shut)


def _parametric(kind: str, data: Mapping[str, Any]):
    """Build a primitive from named parameters rather than from points.

    Keeps the parametric classes reachable from a saved document written by
    something other than the point editor - a generated project, or a future
    editor that keeps a circle a circle.
    """

    factory = _PARAMETRIC.get(kind)
    if factory is None:
        raise ValueError(f"geometry type {kind!r} needs control points")
    parameters = data.get("parameters")
    if not isinstance(parameters, Mapping):
        return factory()

    import inspect

    accepted = set(inspect.signature(factory).parameters)
    supplied = {
        name: value for name, value in parameters.items() if name in accepted
    }
    return factory(**supplied)


def trajectory_from_dict(payload: Mapping[str, Any]) -> CanonicalTrajectory:
    """Return a canonical trajectory from the GUI's JSON-compatible payload.

    Parsing lives in the Qt-free core so preview, export, and the path panel do
    not acquire subtly different interpretations of the same saved data.
    """

    geometry_data = payload.get("geometry")
    if not isinstance(geometry_data, Mapping):
        raise ValueError("canonicalTrajectory.geometry must be an object")
    points = tuple(
        tuple(float(axis) for axis in point)
        for point in geometry_data.get("controlPointsM", ())
    )
    kind = str(geometry_data.get("type", "polyline")).lower()
    closed = bool(geometry_data.get("closed", False))
    if kind not in GEOMETRY_TYPES:
        raise ValueError(
            f"unsupported serialized geometry type: {kind!r}; "
            f"expected one of {GEOMETRY_TYPES}"
        )

    if kind == "mathematical":
        expressions = geometry_data.get("expressions", {})
        geometry = Mathematical(
            str(expressions.get("x", "cos(2*pi*u)")),
            str(expressions.get("y", "sin(2*pi*u)")),
            str(expressions.get("z", "0")),
        )
    elif kind == "spline":
        geometry = Spline(points, closed)
    elif kind == "bezier":
        geometry = Bezier(points)
    elif kind == "polyline":
        geometry = Polyline(points, closed)
    elif kind == "polygon" and len(points) >= 3:
        geometry = Polygon(points)
    elif points:
        # The editor seeds a primitive as points and then lets them be dragged,
        # so the points are the shape - rebuilding the primitive from its
        # defaults here would silently discard every edit, which is what this
        # used to do for circles, spirals and the rest.
        geometry = _from_points(kind, points, closed)
    else:
        # No points at all: a payload that describes the primitive itself.
        geometry = _parametric(kind, geometry_data)

    traversal_data = payload.get("traversal", {})
    if not isinstance(traversal_data, Mapping):
        raise ValueError("canonicalTrajectory.traversal must be an object")
    traversal = Traversal(
        duration_s=float(traversal_data.get("durationS", 5.0)),
        mode=str(traversal_data.get("mode", "loop")),
        direction=int(traversal_data.get("direction", 1)),
        easing=str(traversal_data.get("easing", "linear")),
        steps=int(traversal_data.get("steps", 8)),
        crossfade_s=float(traversal_data.get("crossfadeS", 0.0)),
    )
    return CanonicalTrajectory(
        geometry,
        traversal,
        arc_length=bool(payload.get("arcLength", True)),
        coordinate_smoothing=bool(payload.get("coordinateSmoothing", False)),
    )

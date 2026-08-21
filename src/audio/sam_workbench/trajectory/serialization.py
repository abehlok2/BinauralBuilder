"""Compile the versioned trajectory dictionaries stored in voice parameters."""

from __future__ import annotations

from typing import Any, Mapping

from .geometry import (
    Bezier,
    Circle,
    Ellipse,
    Helix,
    Lissajous,
    Mathematical,
    Polyline,
    Spiral,
    Spline,
)
from .traversal import CanonicalTrajectory, Traversal


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
    elif kind == "circle":
        geometry = Circle()
    elif kind == "ellipse":
        geometry = Ellipse()
    elif kind == "spiral":
        geometry = Spiral()
    elif kind == "helix":
        geometry = Helix()
    elif kind == "lissajous":
        geometry = Lissajous()
    else:
        raise ValueError(f"unsupported serialized geometry type: {kind!r}")

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

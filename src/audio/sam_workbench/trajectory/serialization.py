"""Compile the versioned trajectory dictionaries stored in voice parameters."""

from __future__ import annotations

import re
from typing import Any, Mapping

import dataclasses
import inspect

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
from .keyframes import Keyframe, KeyframedPath
from .primitives import (
    DomeTraversal,
    ElevationSweep,
    FigureEight3D,
    HorizontalOrbit,
    OverheadSweep,
    Pendulum,
    RandomWalkVolume,
    RisingArc,
    SphericalOrbit,
    TiltedOrbit,
    Torus,
    VerticalOrbit,
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
    "keyframes",
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

#: The three-dimensional primitives, which are *not* seeded as draggable
#: points. A dome traversal is a dome because of its parameters; sampling it to
#: points and rebuilding a spline through them would turn every reopened
#: project into an approximation of itself, so these round-trip by name.
_SPATIAL: dict[str, Any] = {
    "horizontal_orbit": HorizontalOrbit,
    "vertical_orbit": VerticalOrbit,
    "tilted_orbit": TiltedOrbit,
    "spherical_orbit": SphericalOrbit,
    "rising_arc": RisingArc,
    "overhead_sweep": OverheadSweep,
    "elevation_sweep": ElevationSweep,
    "dome_traversal": DomeTraversal,
    "figure_eight_3d": FigureEight3D,
    "pendulum": Pendulum,
    "torus": Torus,
    "random_walk_volume": RandomWalkVolume,
}

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
    return _construct(factory, data.get("parameters"))


def _snake(name: str) -> str:
    """``startAzimuthDeg`` -> ``start_azimuth_deg``.

    The GUI writes camelCase and the dataclasses are snake_case; normalizing
    here means a project stays readable by hand without either side having to
    carry a translation table.
    """

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _construct(factory: Any, parameters: Any):
    """Build a frozen geometry from a mapping, ignoring keys it does not take.

    Unknown keys are dropped rather than rejected so a project written by a
    newer build still opens, which is the same rule the rest of the workbench
    follows for forward compatibility.
    """

    if not isinstance(parameters, Mapping):
        return factory()
    accepted = set(inspect.signature(factory).parameters)
    supplied: dict[str, Any] = {}
    for name, value in parameters.items():
        key = name if name in accepted else _snake(str(name))
        if key not in accepted:
            continue
        supplied[key] = tuple(value) if isinstance(value, list) else value
    return factory(**supplied)


def _keyframed(data: Mapping[str, Any]) -> KeyframedPath:
    """Build a keyframed path from the documented ``keyframes`` array."""

    entries = data.get("keyframes", ())
    if not isinstance(entries, (list, tuple)) or len(entries) < 2:
        raise ValueError("a keyframed path needs at least two keyframes")
    return KeyframedPath(
        tuple(Keyframe.from_mapping(dict(entry)) for entry in entries),
        str(data.get("interpolation", "cubic")),
    )


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

    if kind == "keyframes":
        geometry = _keyframed(geometry_data)
    elif kind in _SPATIAL:
        # Parametric by nature: reconstructed from its own numbers, never from
        # a sampled approximation of them.
        geometry = _construct(_SPATIAL[kind], geometry_data.get("parameters"))
    elif kind == "mathematical":
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


#: Fields that are geometry parameters rather than serialization metadata.
def geometry_to_dict(geometry: Any, *, closed: bool = False) -> dict[str, Any]:
    """Return the JSON form of a geometry object.

    Parametric geometry round-trips by name and numbers.  Point-based geometry
    round-trips as its points, because for those the points *are* the shape
    the author sees and edits.
    """

    for name, factory in _SPATIAL.items():
        if isinstance(geometry, factory):
            return {"type": name, "parameters": _fields(geometry)}
    if isinstance(geometry, KeyframedPath):
        return {
            "type": "keyframes",
            "coordinateSystem": "listener_relative_cartesian",
            "units": "metres",
            "interpolation": geometry.interpolation,
            "keyframes": [key.describe() for key in geometry.keyframes],
        }
    if isinstance(geometry, Mathematical):
        return {
            "type": "mathematical",
            "expressions": {
                "x": geometry.x_expression,
                "y": geometry.y_expression,
                "z": geometry.z_expression,
            },
        }
    for name, factory in (
        ("spline", Spline),
        ("bezier", Bezier),
        ("polygon", Polygon),
        ("polyline", Polyline),
    ):
        if isinstance(geometry, factory):
            points = getattr(geometry, "control_points_m", None) or getattr(
                geometry, "points_m", ()
            )
            return {
                "type": name,
                "controlPointsM": [list(map(float, point)) for point in points],
                "closed": bool(getattr(geometry, "closed", closed)),
            }
    if isinstance(geometry, Line):
        return {
            "type": "line",
            "controlPointsM": [list(geometry.start_m), list(geometry.end_m)],
            "closed": False,
        }
    for name, factory in _PARAMETRIC.items():
        if isinstance(geometry, factory):
            return {"type": name, "parameters": _fields(geometry)}
    raise ValueError(f"cannot serialize geometry of type {type(geometry).__name__}")


def _fields(geometry: Any) -> dict[str, Any]:
    """Dataclass fields as JSON, with tuples flattened to lists."""

    result: dict[str, Any] = {}
    for field in dataclasses.fields(geometry):
        value = getattr(geometry, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    return result


def path_model_from_dict(payload: Mapping[str, Any]):
    """Return a fully described :class:`~.path_model.PathModel` from JSON.

    This is the reader for the outer, self-describing form: the one that names
    its coordinate system, units, interpolation and playback semantics.  A
    payload that carries only ``geometry`` and ``traversal`` - the schema
    version 1 form the editor used to write - still loads, taking the defaults
    documented on :class:`~.path_model.PathModel`.
    """

    from .path_model import PathModel, SourceOrientation
    from .transforms import ListenerTransform, Transform

    trajectory = trajectory_from_dict(payload)
    geometry_data = payload.get("geometry") or {}

    transform_data = payload.get("transform")
    transform = Transform()
    if isinstance(transform_data, Mapping):
        transform = Transform(
            translation_m=tuple(transform_data.get("translationM", (0.0, 0.0, 0.0))),
            yaw_pitch_roll_deg=tuple(
                transform_data.get("yawPitchRollDegrees", (0.0, 0.0, 0.0))
            ),
            scale=tuple(transform_data.get("scale", (1.0, 1.0, 1.0))),
            shear=tuple(transform_data.get("shear", (0.0, 0.0, 0.0))),
        )

    listener_data = payload.get("listener")
    listener = ListenerTransform()
    if isinstance(listener_data, Mapping):
        listener = ListenerTransform(
            position_m=tuple(listener_data.get("positionM", (0.0, 0.0, 0.0))),
            yaw_pitch_roll_deg=tuple(
                listener_data.get("yawPitchRollDegrees", (0.0, 0.0, 0.0))
            ),
            ear_spacing_m=float(listener_data.get("earSpacingM", 0.18)),
        )

    orientation = None
    if isinstance(payload.get("sourceOrientation"), Mapping):
        orientation = SourceOrientation.from_mapping(payload["sourceOrientation"])

    # ``arcLength`` was the schema version 1 spelling of the same choice.
    speed_law = str(
        payload.get(
            "speedLaw",
            "constant_speed" if payload.get("arcLength", True) else "parameter_speed",
        )
    )
    return PathModel(
        geometry=trajectory.geometry,
        traversal=trajectory.traversal,
        transform=transform,
        coordinate_system=str(
            payload.get("coordinateSystem", "listener_relative_cartesian")
        ),
        interpolation=str(
            payload.get("interpolation", geometry_data.get("interpolation", "cubic"))
        ),
        speed_law=speed_law,
        closed=bool(geometry_data.get("closed", payload.get("closed", False))),
        listener=listener,
        orientation=orientation,
        coordinate_smoothing=bool(payload.get("coordinateSmoothing", False)),
    )

"""The path parameters a modulation route may drive, and their legal ranges.

A route names its target as a reserved parameter path - ``path.radiusM`` for a
field of the selected geometry primitive, ``transform.yawDeg`` for a component
of the whole-path transform.  This module is the one place that knows which
names exist, what units they carry, and how far each may travel from its stored
value before the motion would describe an impossible shape (a negative radius,
an elevation past the pole, a torus whose tube swallows its ring).

The catalog deliberately excludes discrete fields: ``seed``, waypoint ``steps``,
booleans and interpolation kinds are structure, not quantity, and modulating
structure either does nothing or silently rebuilds the path's identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

import numpy as np

__all__ = [
    "PATH_PREFIX",
    "TRANSFORM_PREFIX",
    "AFFECTS_SHAPE",
    "AFFECTS_TRANSFORM",
    "PathParameterSpec",
    "GEOMETRY_PARAMETER_SPECS",
    "TRANSFORM_PARAMETER_SPECS",
    "COUPLED_CONSTRAINTS",
    "route_leaf",
    "normalize_leaf",
    "split_parameter_path",
    "spec_for_field",
    "primitive_component_fields",
]


#: Route paths naming a geometry-primitive field start with this prefix.
PATH_PREFIX = "path."
#: Route paths naming a whole-path transform component start with this prefix.
TRANSFORM_PREFIX = "transform."

AFFECTS_SHAPE = "shape"
AFFECTS_TRANSFORM = "transform"

_LENGTH_MIN_M = 1e-3


@dataclass(frozen=True)
class PathParameterSpec:
    """One modulatable scalar: what it is called, and how far it may go."""

    #: Snake-case dataclass field name, or transform component key.
    field: str
    label: str
    unit: str
    minimum: float | None
    maximum: float | None
    affects: str

    def clamp(self, values: Any) -> Any:
        """Hold the motion inside the documented range."""

        if self.minimum is None and self.maximum is None:
            return values
        return np.clip(values, self.minimum, self.maximum)


def _spec(
    field: str,
    label: str,
    unit: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> PathParameterSpec:
    return PathParameterSpec(field, label, unit, minimum, maximum, AFFECTS_SHAPE)


_LENGTH = dict(minimum=_LENGTH_MIN_M)
_ELEVATION = dict(minimum=-90.0, maximum=90.0)
_POSITIVE_COUNT = dict(minimum=1e-3)
_NONNEGATIVE = dict(minimum=0.0)

GEOMETRY_PARAMETER_SPECS: dict[str, PathParameterSpec] = {
    entry.field: entry
    for entry in (
        # --- orbits ---------------------------------------------------------
        _spec("radius_m", "Radius", "m", **_LENGTH),
        _spec("start_azimuth_deg", "Start azimuth", "deg"),
        _spec("plane_azimuth_deg", "Plane azimuth", "deg"),
        _spec("start_angle_deg", "Start angle", "deg"),
        _spec("tilt_deg", "Tilt", "deg"),
        _spec("tilt_axis_azimuth_deg", "Tilt axis azimuth", "deg"),
        _spec("turns", "Turns", "turns"),
        _spec("elevation_deg", "Elevation", "deg", **_ELEVATION),
        _spec("centre_x_m", "Centre X", "m"),
        _spec("centre_y_m", "Centre Y", "m"),
        _spec("centre_z_m", "Centre Z", "m"),
        # --- spherical orbit -------------------------------------------------
        _spec("start_distance_m", "Start distance", "m", **_LENGTH),
        _spec("end_distance_m", "End distance", "m", **_LENGTH),
        _spec("cycles", "Distance cycles", "cycles", **_POSITIVE_COUNT),
        # --- sweeps ----------------------------------------------------------
        _spec("end_azimuth_deg", "End azimuth", "deg"),
        _spec("azimuth_deg", "Azimuth", "deg"),
        _spec("start_elevation_deg", "Start elevation", "deg", **_ELEVATION),
        _spec("end_elevation_deg", "End elevation", "deg", **_ELEVATION),
        _spec("distance_m", "Distance", "m", **_LENGTH),
        # --- figures ---------------------------------------------------------
        _spec("azimuth_extent_deg", "Azimuth extent", "deg"),
        _spec("elevation_extent_deg", "Elevation extent", "deg"),
        _spec("centre_azimuth_deg", "Centre azimuth", "deg"),
        _spec("centre_elevation_deg", "Centre elevation", "deg", **_ELEVATION),
        _spec("length_m", "Length", "m", **_LENGTH),
        _spec("swing_deg", "Swing", "deg"),
        _spec("swings", "Swings", "swings"),
        # --- torus -----------------------------------------------------------
        _spec("major_radius_m", "Major radius", "m", **_LENGTH),
        _spec("minor_radius_m", "Minor radius", "m", **_LENGTH),
        _spec("major_turns", "Major turns", "turns"),
        _spec("minor_turns", "Minor turns", "turns"),
        # --- random walk -------------------------------------------------------
        _spec("extent_x_m", "Extent X", "m", **_LENGTH),
        _spec("extent_y_m", "Extent Y", "m", **_LENGTH),
        _spec("extent_z_m", "Extent Z", "m", **_LENGTH),
        _spec("minimum_distance_m", "Minimum distance", "m", **_NONNEGATIVE),
    )
}

TRANSFORM_PARAMETER_SPECS: dict[str, PathParameterSpec] = {
    entry.field: entry
    for entry in (
        PathParameterSpec("yaw_deg", "Yaw", "deg", None, None, AFFECTS_TRANSFORM),
        PathParameterSpec("pitch_deg", "Pitch", "deg", None, None, AFFECTS_TRANSFORM),
        PathParameterSpec("roll_deg", "Roll", "deg", None, None, AFFECTS_TRANSFORM),
        PathParameterSpec(
            "translation_x_m", "Translation X", "m", None, None, AFFECTS_TRANSFORM
        ),
        PathParameterSpec(
            "translation_y_m", "Translation Y", "m", None, None, AFFECTS_TRANSFORM
        ),
        PathParameterSpec(
            "translation_z_m", "Translation Z", "m", None, None, AFFECTS_TRANSFORM
        ),
    )
}


def _torus_minor_inside_major(values: dict[str, Any]) -> None:
    """Keep the torus tube on its ring even while both radii move.

    A minor radius that reaches the major radius puts the path through the
    listener's head - exactly what the primitive refuses at construction time.
    The same relationship is enforced here, dynamically, on the resolved arrays.
    """

    minor = values.get("minor_radius_m")
    major = values.get("major_radius_m")
    if minor is None or major is None:
        return
    ceiling = np.asarray(major, dtype=np.float64) - _LENGTH_MIN_M
    values["minor_radius_m"] = np.minimum(np.asarray(minor), ceiling)


#: Cross-field relationships re-checked after every resolution. Each entry
#: mutates the resolved mapping in place; adding one is how a new coupled
#: constraint joins without the evaluator growing a special case per primitive.
COUPLED_CONSTRAINTS: tuple[Callable[[dict[str, Any]], None], ...] = (
    _torus_minor_inside_major,
)


def route_leaf(field: str) -> str:
    """``start_azimuth_deg`` -> ``startAzimuthDeg``: the route-path spelling."""

    first, *rest = str(field).split("_")
    return first + "".join(part.capitalize() for part in rest)


def normalize_leaf(leaf: str) -> str:
    """Accept either spelling of a leaf; ``startAzimuthDeg`` and the snake form
    both return ``start_azimuth_deg``."""

    text = str(leaf)
    if text in GEOMETRY_PARAMETER_SPECS or text in TRANSFORM_PARAMETER_SPECS:
        return text
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
    if snake in GEOMETRY_PARAMETER_SPECS or snake in TRANSFORM_PARAMETER_SPECS:
        return snake
    raise ValueError(
        f"unknown path parameter {leaf!r}; expected one of "
        f"{sorted(GEOMETRY_PARAMETER_SPECS)} or {sorted(TRANSFORM_PARAMETER_SPECS)}"
    )


def split_parameter_path(parameter_path: str) -> tuple[str, str]:
    """Split ``path.radiusM`` / ``transform.yaw_deg`` into section and field.

    Raises for anything outside the reserved prefixes, so a typo cannot bind a
    route to a parameter that will silently never be read.
    """

    text = str(parameter_path)
    for prefix, section in ((PATH_PREFIX, "geometry"), (TRANSFORM_PREFIX, "transform")):
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            if "." not in remainder:
                return section, normalize_leaf(remainder)
    raise ValueError(
        f"parameter path {parameter_path!r} must start with {PATH_PREFIX!r} "
        f"or {TRANSFORM_PREFIX!r}"
    )


def spec_for_field(field: str) -> PathParameterSpec | None:
    """The spec for a snake-case field name, or ``None`` when unmodulatable."""

    return GEOMETRY_PARAMETER_SPECS.get(field) or TRANSFORM_PARAMETER_SPECS.get(field)


def primitive_component_fields(factory: Any) -> frozenset[str]:
    """Every catalog field the named primitive actually carries.

    Vector-valued dataclass fields (``centre_m`` and friends) are offered as
    their three scalar components, which is what a route can address.
    """

    import dataclasses

    fields = {entry.name for entry in dataclasses.fields(factory)}
    components: set[str] = set()
    for name in fields:
        components.add(name)
        default = factory.__dataclass_fields__[name].default
        if isinstance(default, tuple) and len(default) == 3 and re.fullmatch(r"[a-z]+_m", name):
            stem = name[: -len("_m")]
            components.update({f"{stem}_x_m", f"{stem}_y_m", f"{stem}_z_m"})
    allowed = {
        field
        for field in GEOMETRY_PARAMETER_SPECS
        if field in components or field in fields
    }
    return frozenset(allowed)

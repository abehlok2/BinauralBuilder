"""The complete, self-describing specification of a source's motion.

:class:`CanonicalTrajectory` answers "where is the source at time ``t``".  It
deliberately does not say which frame that answer is in, what the units are, or
whether the numbers are relative to the listener - because a trajectory is
composed and re-composed during editing and carrying that metadata on every
intermediate object would be noise.

A *saved* path has to say all of it, though.  A project opened on another
machine, by another build, cannot infer whether ``[0, 1, 0]`` meant one metre
to the left of the listener or one metre along world north.  :class:`PathModel`
is that outer layer: it wraps a geometry, a transform and a traversal with the
frame, units and playback semantics needed to reproduce them exactly.

Geometry versus traversal
-------------------------
The two are kept separate throughout.  *Geometry* is where the source can be;
*traversal* is how it moves along that shape over time.  A circle stays one
circle whether it is walked at constant speed, accelerated, oscillated, or
driven from the stage timeline - so those are traversal settings, never new
geometry.  Keeping the split is what makes the primitive library and the time
laws independently extensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spherical import cartesian_array_to_spherical
from .transforms import ListenerTransform, Transform, rotation_matrix_ypr
from .traversal import CanonicalTrajectory, Traversal

__all__ = [
    "COORDINATE_SYSTEMS",
    "INTERPOLATION_KINDS",
    "SPEED_LAWS",
    "AXIS_DEFINITIONS",
    "HANDEDNESS",
    "UNITS",
    "SourceOrientation",
    "PathModel",
]

#: Whether stored coordinates follow the listener or stay fixed in the room.
COORDINATE_SYSTEMS: tuple[str, ...] = (
    "listener_relative_cartesian",
    "world_cartesian",
)

#: How positions between control points are reconstructed.
INTERPOLATION_KINDS: tuple[str, ...] = ("hold", "linear", "cubic", "catmull_rom")

#: Whether ``u`` advances by physical distance or by the curve's own parameter.
SPEED_LAWS: tuple[str, ...] = ("constant_speed", "parameter_speed")

#: The frame every core module uses; see :mod:`..conventions`.
AXIS_DEFINITIONS: Mapping[str, str] = {
    "x": "forward, positive in front of the listener",
    "y": "lateral, positive toward the listener's left",
    "z": "vertical, positive above the listener",
}
HANDEDNESS = "right"
UNITS = "metres"


@dataclass(frozen=True)
class SourceOrientation:
    """Optional aim of the source itself, independent of where it is.

    A source that faces the listener throughout its path sounds different from
    one that keeps a fixed heading while it orbits, but only a renderer with a
    directivity model can act on it.  It is stored because discarding it would
    lose authoring intent; renderers that cannot use it ignore it.
    """

    #: ``"fixed"`` holds ``yaw_pitch_roll_deg``; ``"path_tangent"`` aims along
    #: the direction of travel; ``"toward_listener"`` always faces the head.
    mode: str = "fixed"
    yaw_pitch_roll_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    MODES = ("fixed", "path_tangent", "toward_listener")

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(
                f"unknown source orientation mode {self.mode!r}; expected one of {self.MODES}"
            )
        if len(self.yaw_pitch_roll_deg) != 3 or not all(
            math.isfinite(float(value)) for value in self.yaw_pitch_roll_deg
        ):
            raise ValueError("yaw_pitch_roll_deg must contain three finite values")

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "yawPitchRollDegrees": [float(v) for v in self.yaw_pitch_roll_deg],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SourceOrientation":
        values = dict(data or {})
        angles = values.get("yawPitchRollDegrees", (0.0, 0.0, 0.0))
        return cls(
            mode=str(values.get("mode", "fixed")),
            yaw_pitch_roll_deg=tuple(float(value) for value in angles),
        )


@dataclass(frozen=True)
class PathModel:
    """A geometry, a traversal, and everything needed to interpret them."""

    geometry: object
    traversal: Traversal = field(default_factory=Traversal)
    transform: Transform = field(default_factory=Transform)
    coordinate_system: str = "listener_relative_cartesian"
    interpolation: str = "cubic"
    speed_law: str = "constant_speed"
    closed: bool = False
    #: Only meaningful when ``coordinate_system`` is ``"world_cartesian"``;
    #: also supplies the head orientation a renderer needs for world paths.
    listener: ListenerTransform = field(default_factory=ListenerTransform)
    orientation: SourceOrientation | None = None
    coordinate_smoothing: bool = False
    arclength_samples: int = 2049

    def __post_init__(self) -> None:
        if self.coordinate_system not in COORDINATE_SYSTEMS:
            raise ValueError(
                f"unknown coordinateSystem {self.coordinate_system!r}; "
                f"expected one of {COORDINATE_SYSTEMS}"
            )
        if self.interpolation not in INTERPOLATION_KINDS:
            raise ValueError(
                f"unknown interpolation {self.interpolation!r}; "
                f"expected one of {INTERPOLATION_KINDS}"
            )
        if self.speed_law not in SPEED_LAWS:
            raise ValueError(
                f"unknown speed law {self.speed_law!r}; expected one of {SPEED_LAWS}"
            )

    # --- evaluation --------------------------------------------------------

    @property
    def is_listener_relative(self) -> bool:
        return self.coordinate_system == "listener_relative_cartesian"

    def trajectory(self) -> CanonicalTrajectory:
        """The geometry/traversal pair, with the transform already applied."""

        from .geometry import TransformedGeometry

        geometry = self.geometry
        if self.transform != Transform():
            geometry = TransformedGeometry(geometry, self.transform)
        return CanonicalTrajectory(
            geometry,
            self.traversal,
            arc_length=self.speed_law == "constant_speed",
            arclength_samples=self.arclength_samples,
            coordinate_smoothing=self.coordinate_smoothing,
        )

    def positions(self, time_s: ArrayLike) -> NDArray[np.float64]:
        """Listener-relative positions in metres, whatever frame was stored.

        This is the single function every renderer should call.  A world-frame
        path is resolved against the listener pose here, so no renderer has to
        know which frame the project used.
        """

        points = np.asarray(self.trajectory().evaluate(time_s), dtype=np.float64)
        if self.is_listener_relative:
            return points
        return np.asarray(self.listener.world_to_listener(points), dtype=np.float64)

    def spherical(self, time_s: ArrayLike) -> NDArray[np.float64]:
        """The same motion as ``(..., 3)`` azimuth, elevation, distance."""

        return cartesian_array_to_spherical(self.positions(time_s))

    def orientation_matrix(self, time_s: ArrayLike) -> NDArray[np.float64] | None:
        """Per-sample source rotation matrices, or ``None`` when unspecified.

        Returned for renderers with a directivity model; the ones without one
        ignore it rather than being forced to represent it.
        """

        if self.orientation is None:
            return None
        if self.orientation.mode == "fixed":
            return rotation_matrix_ypr(*self.orientation.yaw_pitch_roll_deg)
        times = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        points = self.positions(times)
        if self.orientation.mode == "toward_listener":
            heading = -points
        else:  # path_tangent
            heading = np.gradient(points, axis=0) if len(points) > 1 else -points
        norms = np.linalg.norm(heading, axis=-1, keepdims=True)
        return np.where(norms > 1e-12, heading / np.maximum(norms, 1e-12), 0.0)

    def with_traversal(self, traversal: Traversal) -> "PathModel":
        """The same geometry moved by a different time law."""

        return replace(self, traversal=traversal)

    @property
    def duration_s(self) -> float:
        return float(getattr(self.traversal, "duration_s", 0.0) or 0.0)

    # --- serialization -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """The JSON-compatible form, including the metadata that names the frame."""

        from .serialization import geometry_to_dict

        result: dict[str, Any] = {
            "schemaVersion": 2,
            "coordinateSystem": self.coordinate_system,
            "handedness": HANDEDNESS,
            "units": UNITS,
            "axes": dict(AXIS_DEFINITIONS),
            "interpolation": self.interpolation,
            "speedLaw": self.speed_law,
            "closed": bool(self.closed),
            "geometry": geometry_to_dict(self.geometry, closed=self.closed),
            "transform": {
                "translationM": list(self.transform.translation_m),
                "yawPitchRollDegrees": list(self.transform.yaw_pitch_roll_deg),
                "scale": list(self.transform.scale),
                "shear": list(self.transform.shear),
            },
            "traversal": {
                "mode": self.traversal.mode,
                "durationS": float(self.traversal.duration_s),
                "easing": self.traversal.easing,
                "direction": int(self.traversal.direction),
                "steps": int(self.traversal.steps),
                "crossfadeS": float(self.traversal.crossfade_s),
            },
            "coordinateSmoothing": bool(self.coordinate_smoothing),
        }
        if not self.is_listener_relative or self.listener != ListenerTransform():
            result["listener"] = {
                "positionM": list(self.listener.position_m),
                "yawPitchRollDegrees": list(self.listener.yaw_pitch_roll_deg),
                "earSpacingM": float(self.listener.ear_spacing_m),
            }
        if self.orientation is not None:
            result["sourceOrientation"] = self.orientation.describe()
        return result

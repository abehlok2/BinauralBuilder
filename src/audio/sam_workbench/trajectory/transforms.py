"""Transforms: moving, turning, and scaling a path, and the legacy pixel bridge.

Three coordinate spaces meet here:

* the **canonical listener frame** - ``+x`` forward, ``+y`` left, ``+z`` up,
  metres - which everything downstream uses;
* the **world frame**, when a path is anchored to the room rather than to the
  head, related to the listener frame by :class:`ListenerFrame`;
* the **legacy GUI scene**, where the existing custom-path editor stores pixels
  with ``+x`` to the right and ``+y`` downward.

`LegacyPathTransform` is the only place the pixel convention is interpreted.
Its parameters are declared, not guessed: centre, axis convention, scene units
per metre, and an optional normalization extent. Existing profiles keep their
original pixel coordinates on disk; the translation happens on the way into the
renderer, so nothing is rewritten behind the user's back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AXIS_CONVENTIONS",
    "ListenerFrame",
    "LegacyPathTransform",
    "TransformSpec",
    "rotation_matrix",
    "transform_from_dict",
]

#: Scene axis conventions the legacy bridge understands.
AXIS_CONVENTIONS = ("x_right_y_down", "x_right_y_up")


def rotation_matrix(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> NDArray[np.float64]:
    """Rotation about ``+z`` (yaw), then ``+y`` (pitch), then ``+x`` (roll).

    Positive yaw turns toward the listener's left, matching the canonical
    azimuth convention.
    """

    yaw, pitch, roll = (math.radians(float(value)) for value in (yaw_deg, pitch_deg, roll_deg))
    about_z = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    about_y = np.array(
        [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]]
    )
    about_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]]
    )
    return about_z @ about_y @ about_x


@dataclass(frozen=True)
class TransformSpec:
    """Translation, rotation, scale, shear, and reflection of a path.

    Applied in a fixed order - reflect, shear, scale, rotate, translate - so a
    saved transform always reproduces the same path.
    """

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Shear of the forward axis by the lateral one, and of lateral by vertical.
    shear_xy: float = 0.0
    shear_yz: float = 0.0
    reflect_x: bool = False
    reflect_y: bool = False
    reflect_z: bool = False

    @property
    def is_identity(self) -> bool:
        return (
            tuple(self.translation_m) == (0.0, 0.0, 0.0)
            and (self.yaw_deg, self.pitch_deg, self.roll_deg) == (0.0, 0.0, 0.0)
            and tuple(self.scale) == (1.0, 1.0, 1.0)
            and (self.shear_xy, self.shear_yz) == (0.0, 0.0)
            and not (self.reflect_x or self.reflect_y or self.reflect_z)
        )

    def apply(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        """Transform ``(3, n)`` points in metres."""

        block = np.array(points, dtype=np.float64, copy=True)
        if block.ndim != 2 or block.shape[0] != 3:
            raise ValueError(f"expected (3, n) points, got shape {block.shape}")

        for axis, reflect in enumerate((self.reflect_x, self.reflect_y, self.reflect_z)):
            if reflect:
                block[axis] = -block[axis]

        if self.shear_xy:
            block[0] = block[0] + float(self.shear_xy) * block[1]
        if self.shear_yz:
            block[1] = block[1] + float(self.shear_yz) * block[2]

        block *= np.asarray(self.scale, dtype=np.float64)[:, None]

        if (self.yaw_deg, self.pitch_deg, self.roll_deg) != (0.0, 0.0, 0.0):
            block = rotation_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg) @ block

        return block + np.asarray(self.translation_m, dtype=np.float64)[:, None]

    def to_dict(self) -> dict:
        return {
            "translation_m": list(self.translation_m),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
            "scale": list(self.scale),
            "shear_xy": float(self.shear_xy),
            "shear_yz": float(self.shear_yz),
            "reflect_x": bool(self.reflect_x),
            "reflect_y": bool(self.reflect_y),
            "reflect_z": bool(self.reflect_z),
        }


def transform_from_dict(data) -> TransformSpec:
    fields = dict(data)
    for key in ("translation_m", "scale"):
        if key in fields:
            fields[key] = tuple(float(value) for value in fields[key])
    return TransformSpec(**fields)


@dataclass(frozen=True)
class ListenerFrame:
    """Where the listener is and which way they face, in the world frame."""

    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    #: Half the interaural distance; the ears sit at ``+/- head_radius_m`` on ``y``.
    head_radius_m: float = 0.0875

    @property
    def ear_positions_m(self) -> NDArray[np.float64]:
        """``(3, 2)`` ear positions in the listener frame: left first."""

        return np.array(
            [[0.0, 0.0], [float(self.head_radius_m), -float(self.head_radius_m)], [0.0, 0.0]],
            dtype=np.float64,
        )

    def world_to_listener(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        """Express world-frame points in the listener's own frame."""

        block = np.asarray(points, dtype=np.float64)
        translated = block - np.asarray(self.position_m, dtype=np.float64)[:, None]
        return rotation_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg).T @ translated

    def listener_to_world(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        block = np.asarray(points, dtype=np.float64)
        rotated = rotation_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg) @ block
        return rotated + np.asarray(self.position_m, dtype=np.float64)[:, None]

    def to_dict(self) -> dict:
        return {
            "position_m": list(self.position_m),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
            "head_radius_m": float(self.head_radius_m),
        }


@dataclass(frozen=True)
class LegacyPathTransform:
    """Translate legacy GUI scene coordinates into canonical metres.

    The existing custom-path editor draws in scene pixels: ``+x`` to the right,
    ``+y`` downward, viewed from above with the listener facing up the screen,
    ear markers ``A`` (screen left) and ``B`` (screen right). In the canonical
    frame that is:

    .. code-block:: text

        forward (+x) = -(scene_y - centre_y) / units_per_metre
        left    (+y) = -(scene_x - centre_x) / units_per_metre

    so marker ``A`` is the **left** ear and ``B`` is the **right** ear. The
    mapping is exactly invertible, which is what lets a legacy profile round
    trip through the canonical space without drift.
    """

    centre_scene: tuple[float, float] = (0.0, 0.0)
    axis_convention: str = "x_right_y_down"
    scene_units_per_metre: float = 742.857_142_857_142_9
    #: Optional half-extent used to normalize before scaling, for profiles
    #: stored in arbitrary units rather than pixels.
    normalization_extent: float | None = None
    elevation_m: float = 0.0

    def __post_init__(self) -> None:
        if self.axis_convention not in AXIS_CONVENTIONS:
            raise ValueError(
                f"unknown axis convention {self.axis_convention!r}; expected one of {AXIS_CONVENTIONS}"
            )
        if float(self.scene_units_per_metre) <= 0.0:
            raise ValueError("scene_units_per_metre must be positive")

    @property
    def _lateral_sign(self) -> float:
        return 1.0

    @property
    def _forward_sign(self) -> float:
        # With +y downward, screen-up is forward; with +y upward it already is.
        return -1.0 if self.axis_convention == "x_right_y_down" else 1.0

    def _scale(self) -> float:
        if self.normalization_extent:
            return float(self.normalization_extent) * float(self.scene_units_per_metre)
        return float(self.scene_units_per_metre)

    def to_canonical(self, points_scene: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convert ``(n, 2)`` or ``(2, n)`` scene points to ``(3, n)`` metres."""

        block = np.asarray(points_scene, dtype=np.float64)
        if block.ndim != 2:
            raise ValueError(f"expected 2-D scene points, got shape {block.shape}")
        if block.shape[0] != 2 or (block.shape[1] == 2 and block.shape[0] != 2):
            if block.shape[1] == 2:
                block = np.ascontiguousarray(block.T)
        if block.shape[0] != 2:
            raise ValueError(f"expected two scene coordinates per point, got shape {block.shape}")

        scale = self._scale()
        centred_x = block[0] - float(self.centre_scene[0])
        centred_y = block[1] - float(self.centre_scene[1])
        forward = self._forward_sign * centred_y / scale
        left = -centred_x / scale
        height = np.full(forward.shape, float(self.elevation_m))
        return np.vstack((forward, left, height))

    def from_canonical(self, points_m: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convert ``(3, n)`` canonical metres back to ``(n, 2)`` scene points."""

        block = np.asarray(points_m, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != 3:
            raise ValueError(f"expected (3, n) points, got shape {block.shape}")
        scale = self._scale()
        scene_x = -block[1] * scale + float(self.centre_scene[0])
        scene_y = self._forward_sign * block[0] * scale + float(self.centre_scene[1])
        return np.column_stack((scene_x, scene_y))

    def to_dict(self) -> dict:
        return {
            "centre_scene": list(self.centre_scene),
            "axis_convention": self.axis_convention,
            "scene_units_per_metre": float(self.scene_units_per_metre),
            "normalization_extent": self.normalization_extent,
            "elevation_m": float(self.elevation_m),
        }

    @classmethod
    def from_dict(cls, data) -> "LegacyPathTransform":
        fields = dict(data)
        if "centre_scene" in fields:
            fields["centre_scene"] = tuple(float(value) for value in fields["centre_scene"])
        return cls(**fields)

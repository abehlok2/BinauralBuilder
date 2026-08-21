"""Transforms for the canonical ``+x`` forward, ``+y`` left, ``+z`` up frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector3 = tuple[float, float, float]


def _vector3(value: Sequence[float], name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def rotation_matrix_ypr(
    yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> NDArray[np.float64]:
    """Return an active intrinsic yaw, pitch, roll rotation matrix.

    Positive yaw turns forward toward the listener's left. Pitch rotates about
    ``+y`` and roll rotates about ``+x``. The inverse is the transpose.
    """

    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rotate_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    rotate_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    return rotate_z @ rotate_y @ rotate_x


@dataclass(frozen=True)
class Transform:
    """A validated affine source/path transform.

    Scale is applied first, followed by XY/XZ/YZ shear, rotation, and
    translation. Negative scale components provide explicit reflections.
    """

    translation_m: Vector3 = (0.0, 0.0, 0.0)
    yaw_pitch_roll_deg: Vector3 = (0.0, 0.0, 0.0)
    scale: Vector3 = (1.0, 1.0, 1.0)
    shear: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _vector3(self.translation_m, "translation_m")
        _vector3(self.yaw_pitch_roll_deg, "yaw_pitch_roll_deg")
        scale = _vector3(self.scale, "scale")
        _vector3(self.shear, "shear")
        if np.any(scale == 0.0):
            raise ValueError("scale components must be non-zero")

    @property
    def linear_matrix(self) -> NDArray[np.float64]:
        xy, xz, yz = self.shear
        shear = np.array(((1.0, xy, xz), (0.0, 1.0, yz), (0.0, 0.0, 1.0)))
        return rotation_matrix_ypr(*self.yaw_pitch_roll_deg) @ shear @ np.diag(self.scale)

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError(f"points must end in three coordinates, got {values.shape}")
        return values @ self.linear_matrix.T + np.asarray(self.translation_m)

    def inverse(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError(f"points must end in three coordinates, got {values.shape}")
        return (values - np.asarray(self.translation_m)) @ np.linalg.inv(self.linear_matrix).T


@dataclass(frozen=True)
class ListenerTransform:
    """Listener pose and receiver geometry in world coordinates."""

    position_m: Vector3 = (0.0, 0.0, 0.0)
    yaw_pitch_roll_deg: Vector3 = (0.0, 0.0, 0.0)
    ear_spacing_m: float = 0.18

    def __post_init__(self) -> None:
        _vector3(self.position_m, "position_m")
        _vector3(self.yaw_pitch_roll_deg, "yaw_pitch_roll_deg")
        if not math.isfinite(self.ear_spacing_m) or self.ear_spacing_m <= 0.0:
            raise ValueError("ear_spacing_m must be finite and positive")

    def world_to_listener(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64) - np.asarray(self.position_m)
        return values @ rotation_matrix_ypr(*self.yaw_pitch_roll_deg)

    def listener_to_world(self, points: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        return values @ rotation_matrix_ypr(*self.yaw_pitch_roll_deg).T + self.position_m

    def ear_positions_world(self) -> NDArray[np.float64]:
        half_spacing = self.ear_spacing_m / 2.0
        return self.listener_to_world(((0.0, half_spacing, 0.0), (0.0, -half_spacing, 0.0)))

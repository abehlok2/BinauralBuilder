"""How one source's path relates to another's.

Sources rarely want to move independently. Two of them held opposite each other
across the head, or one orbiting another, or a group that repels so they never
collapse onto the same point - these are relationships between paths, not
separate paths that happen to line up.

Expressing them as a *derivation from a leader* rather than as separate
trajectories means the relationship survives editing the leader, and it means
the follower needs no path data of its own to go wrong. Every mode here is a
pure function of the leader's position and the time, so a follower can be
evaluated at any sample without having evaluated the previous one - which is
what keeps a multi-source render chunkable and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

__all__ = ["COUPLING_MODES", "CouplingSpec", "couple_positions"]

COUPLING_MODES = (
    "independent",
    "shared",
    "offset",
    "mirrored",
    "orbit",
    "repelled",
    "attracted",
    "phase_locked",
)

_EPSILON = 1e-9


@dataclass(frozen=True)
class CouplingSpec:
    """How a follower derives its position from its leader."""

    mode: str = "independent"
    leader_id: str = ""
    #: Degrees around the vertical axis, for ``offset`` and ``phase_locked``.
    angle_deg: float = 0.0
    #: Metres, for ``offset``, ``repelled``, ``attracted``, and ``orbit``.
    distance_m: float = 0.0
    #: Revolutions per second, for ``orbit``.
    rate_hz: float = 0.0
    #: How strongly ``repelled`` and ``attracted`` pull, from 0 to 1.
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in COUPLING_MODES:
            raise ValueError(f"unknown coupling mode {self.mode!r}; expected one of {COUPLING_MODES}")
        if self.mode != "independent" and not self.leader_id:
            raise ValueError(f"coupling mode {self.mode!r} needs a leader to follow")
        for name in ("angle_deg", "distance_m", "rate_hz", "strength"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must lie in [0, 1], got {self.strength!r}")

    @property
    def is_independent(self) -> bool:
        return self.mode == "independent"

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "leaderId": self.leader_id,
            "angleDeg": float(self.angle_deg),
            "distanceM": float(self.distance_m),
            "rateHz": float(self.rate_hz),
            "strength": float(self.strength),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CouplingSpec":
        values = dict(data or {})
        known: dict[str, Any] = {}
        if "mode" in values:
            known["mode"] = str(values["mode"])
        if "leaderId" in values:
            known["leader_id"] = str(values["leaderId"])
        for source, target in (
            ("angleDeg", "angle_deg"),
            ("distanceM", "distance_m"),
            ("rateHz", "rate_hz"),
            ("strength", "strength"),
        ):
            if source in values:
                known[target] = float(values[source])
        return cls(**known)


def _rotate_z(points: NDArray[np.float64], angle_rad) -> NDArray[np.float64]:
    """Rotate about the vertical axis, which is where azimuth lives."""

    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack((x * cos - y * sin, x * sin + y * cos, z), axis=-1)


def couple_positions(
    spec: CouplingSpec,
    leader: NDArray[np.floating],
    *,
    own: NDArray[np.floating] | None = None,
    times_s: NDArray[np.floating] | float = 0.0,
) -> NDArray[np.float64]:
    """Derive a follower's positions from its leader's.

    ``leader`` is ``(3,)`` or ``(frames, 3)``. ``own`` is the follower's own
    path, needed only by the modes that modify it rather than replace it.
    """

    path = np.asarray(leader, dtype=np.float64)
    single = path.ndim == 1
    block = path.reshape(1, 3) if single else path

    if spec.mode == "independent":
        if own is None:
            raise ValueError("an independent source needs its own path")
        result = np.asarray(own, dtype=np.float64)
        result = result.reshape(1, 3) if result.ndim == 1 else result
    elif spec.mode == "shared":
        result = block.copy()
    elif spec.mode == "offset":
        # Rotate around the head, then push out or pull in along the radius.
        rotated = _rotate_z(block, math.radians(spec.angle_deg))
        result = _scaled_radius(rotated, spec.distance_m)
    elif spec.mode == "mirrored":
        # Across the median plane: left becomes right, front and height stay.
        result = block * np.array([1.0, -1.0, 1.0])
    elif spec.mode == "orbit":
        times = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
        if times.size == 1 and block.shape[0] > 1:
            times = np.full(block.shape[0], float(times[0]))
        angle = 2.0 * np.pi * float(spec.rate_hz) * times + math.radians(spec.angle_deg)
        radius = float(spec.distance_m)
        circle = np.stack(
            (radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)), axis=-1
        )
        result = block + circle
    elif spec.mode in ("repelled", "attracted"):
        if own is None:
            raise ValueError(f"coupling mode {spec.mode!r} modifies the follower's own path")
        base = np.asarray(own, dtype=np.float64)
        base = base.reshape(1, 3) if base.ndim == 1 else base
        offset = base - block
        distance = np.linalg.norm(offset, axis=-1, keepdims=True)
        # Where the two coincide there is no direction to move along, so the
        # pair is separated along +y rather than by dividing by zero.
        direction = np.where(
            distance > _EPSILON, offset / np.maximum(distance, _EPSILON), np.array([0.0, 1.0, 0.0])
        )
        sign = 1.0 if spec.mode == "repelled" else -1.0
        push = sign * float(spec.strength) * float(spec.distance_m)
        if spec.mode == "attracted":
            # Never pull past the leader: an attraction that overshoots would
            # swap the two sources' sides.
            push = -np.minimum(float(spec.strength) * float(spec.distance_m), distance)
            result = base + direction * push
        else:
            result = base + direction * push
    else:  # phase_locked
        # Same radius and elevation as the leader, held at a fixed angle from
        # it: the two stay locked together as the leader moves.
        result = _rotate_z(block, math.radians(spec.angle_deg))

    return result[0] if single and result.shape[0] == 1 else result


def _scaled_radius(points: NDArray[np.float64], offset_m: float) -> NDArray[np.float64]:
    """Move points out along their own radius by ``offset_m``."""

    if abs(offset_m) <= _EPSILON:
        return points
    radius = np.linalg.norm(points, axis=-1, keepdims=True)
    direction = np.where(radius > _EPSILON, points / np.maximum(radius, _EPSILON), 0.0)
    return points + direction * float(offset_m)

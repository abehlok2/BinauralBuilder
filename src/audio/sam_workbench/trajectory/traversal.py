"""Traversal: *when* a source is where, given a geometry.

The traversal maps absolute time to a position along the path. It owns
direction, looping, easing, speed, and deliberate jumps - and nothing about the
shape of the path, so the same law can drive a circle, a spline, or a point
cloud.

Two properties matter as much as the shape:

* **Block-size invariance.** Position is a function of absolute time, so a
  chunked render lands on exactly the samples a whole render would.
* **Honest discontinuities.** A jump is reported, not hidden. The renderer
  crossfades across it; a jump that nobody declared would arrive as a click.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_JUMP_CROSSFADE_MS",
    "EASINGS",
    "constant_speed_rate_hz",
    "TRAVERSAL_MODES",
    "TraversalResult",
    "TraversalSpec",
    "traversal_from_dict",
]

#: Directions and looping laws a traversal can follow.
TRAVERSAL_MODES = ("loop", "one_shot", "ping_pong", "reverse")
#: Easing curves applied to the position within one traversal cycle.
EASINGS = ("linear", "sine", "smooth", "keyframed")
#: A jump shorter than this would still click; the renderer fades across it.
DEFAULT_JUMP_CROSSFADE_MS = 12.0


@dataclass(frozen=True)
class TraversalResult:
    """Positions along a path plus where the motion deliberately jumped."""

    #: Normalized position along the path, in ``[0, 1]``.
    position: NDArray[np.float64]
    #: True at the first sample after a declared discontinuity.
    jumps: NDArray[np.bool_]
    crossfade_ms: float = DEFAULT_JUMP_CROSSFADE_MS

    @property
    def jump_indices(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.jumps)


def _ease(values: NDArray[np.float64], easing: str) -> NDArray[np.float64]:
    if easing == "linear":
        return values
    if easing == "sine":
        # Slow at both ends, fastest in the middle.
        return 0.5 - 0.5 * np.cos(np.pi * values)
    if easing == "smooth":
        return values * values * (3.0 - 2.0 * values)
    raise ValueError(f"unknown easing {easing!r}")


@dataclass(frozen=True)
class TraversalSpec:
    """A timing law over a path.

    ``rate_hz`` is traversals per second: one full pass along an open path, or
    one lap of a closed one. ``arc_length`` asks for constant metres per second
    instead of constant parameter per second, which the geometry supports.
    """

    rate_hz: float = 0.25
    mode: str = "loop"
    easing: str = "linear"
    phase: float = 0.0
    arc_length: bool = True
    #: ``(time_s, position)`` pairs for ``easing="keyframed"``.
    keyframes: tuple[tuple[float, float], ...] = ()
    #: Absolute times at which the source jumps to a new position.
    jump_times_s: tuple[float, ...] = ()
    #: Positions to jump to, matched to ``jump_times_s`` by index. When empty
    #: the jump advances by one step of ``jump_step``.
    jump_positions: tuple[float, ...] = ()
    jump_step: float = 0.25
    crossfade_ms: float = DEFAULT_JUMP_CROSSFADE_MS

    def __post_init__(self) -> None:
        if self.mode not in TRAVERSAL_MODES:
            raise ValueError(f"unknown traversal mode {self.mode!r}; expected one of {TRAVERSAL_MODES}")
        if self.easing not in EASINGS:
            raise ValueError(f"unknown easing {self.easing!r}; expected one of {EASINGS}")
        if self.easing == "keyframed" and not self.keyframes:
            raise ValueError("keyframed traversal needs keyframes")

    # --- position -----------------------------------------------------------

    def _cycle_position(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Raw position in ``[0, 1]`` before easing, from absolute time."""

        if self.easing == "keyframed":
            knot_times = np.array([time for time, _ in self.keyframes], dtype=np.float64)
            knot_values = np.array([value for _, value in self.keyframes], dtype=np.float64)
            order = np.argsort(knot_times)
            return np.clip(np.interp(times, knot_times[order], knot_values[order]), 0.0, 1.0)

        cycles = float(self.rate_hz) * times + float(self.phase)
        if self.mode == "one_shot":
            return np.clip(cycles, 0.0, 1.0)
        if self.mode == "ping_pong":
            wrapped = np.mod(cycles, 2.0)
            return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)
        if self.mode == "reverse":
            return 1.0 - np.mod(cycles, 1.0)
        return np.mod(cycles, 1.0)

    def _apply_jumps(
        self, times: NDArray[np.float64], position: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        jumps = np.zeros(position.shape, dtype=bool)
        if not self.jump_times_s:
            return position, jumps

        offsets = np.zeros(position.shape, dtype=np.float64)
        absolute = np.full(position.shape, np.nan)
        for index, jump_time in enumerate(sorted(self.jump_times_s)):
            after = times >= float(jump_time)
            if not np.any(after):
                continue
            if index < len(self.jump_positions):
                absolute[after] = float(self.jump_positions[index])
            else:
                offsets[after] += float(self.jump_step)
            first = int(np.argmax(after))
            # Only mark a boundary that is inside this block.
            if after[first] and (first > 0 or times[0] >= float(jump_time) - 1e-12):
                jumps[first] = True

        jumped = np.where(np.isnan(absolute), np.mod(position + offsets, 1.0), absolute)
        return jumped, jumps

    def positions(self, times: NDArray[np.floating]) -> TraversalResult:
        """Positions along the path for absolute times in seconds."""

        time_array = np.atleast_1d(np.asarray(times, dtype=np.float64))
        raw = self._cycle_position(time_array)
        eased = raw if self.easing in ("keyframed", "linear") else _ease(raw, self.easing)
        position, jumps = self._apply_jumps(time_array, eased)
        return TraversalResult(position=position, jumps=jumps, crossfade_ms=float(self.crossfade_ms))

    def sample_times(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        """Absolute times for a block, so a chunked render matches a whole one."""

        index = np.arange(int(start_sample), int(start_sample) + int(frames), dtype=np.float64)
        return index / float(sample_rate)

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "rate_hz": float(self.rate_hz),
            "mode": self.mode,
            "easing": self.easing,
            "phase": float(self.phase),
            "arc_length": bool(self.arc_length),
            "keyframes": [[float(time), float(value)] for time, value in self.keyframes],
            "jump_times_s": [float(value) for value in self.jump_times_s],
            "jump_positions": [float(value) for value in self.jump_positions],
            "jump_step": float(self.jump_step),
            "crossfade_ms": float(self.crossfade_ms),
        }


def traversal_from_dict(data) -> TraversalSpec:
    """Rebuild a traversal from its serialized form."""

    fields = dict(data)
    if "keyframes" in fields:
        fields["keyframes"] = tuple((float(time), float(value)) for time, value in fields["keyframes"])
    for key in ("jump_times_s", "jump_positions"):
        if key in fields:
            fields[key] = tuple(float(value) for value in fields[key])
    return TraversalSpec(**fields)


def constant_speed_rate_hz(path_length_m: float, speed_m_s: float) -> float:
    """The traversal rate that moves a source at ``speed_m_s`` along a path."""

    if path_length_m <= 0.0:
        return 0.0
    return float(speed_m_s) / float(path_length_m)

"""Time laws and legacy-segment translation for canonical trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

TraversalMode = Literal["loop", "one_shot", "ping_pong", "discontinuous"]


def apply_easing(value: ArrayLike, easing: str = "linear") -> NDArray[np.float64]:
    progress = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    if easing in ("sine", "smooth"):
        return 0.5 - 0.5 * np.cos(np.pi * progress)
    if easing == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    if easing != "linear":
        raise ValueError(f"unsupported easing: {easing!r}")
    return progress


@dataclass(frozen=True)
class DiscontinuityBlend:
    """Path positions on either side of a jump plus equal-power weights."""

    previous_progress: NDArray[np.float64]
    next_progress: NDArray[np.float64]
    previous_gain: NDArray[np.float64]
    next_gain: NDArray[np.float64]


@dataclass(frozen=True)
class Traversal:
    duration_s: float = 1.0
    mode: TraversalMode = "loop"
    direction: int = 1
    easing: str = "linear"
    steps: int = 8
    crossfade_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if self.mode not in ("loop", "one_shot", "ping_pong", "discontinuous"):
            raise ValueError(f"unsupported traversal mode: {self.mode!r}")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.steps < 2:
            raise ValueError("steps must be at least two")
        if not math.isfinite(self.crossfade_s) or self.crossfade_s < 0.0:
            raise ValueError("crossfade_s must be finite and non-negative")
        apply_easing(0.0, self.easing)

    def _continuous_progress(self, time_s: ArrayLike) -> NDArray[np.float64]:
        normalized = np.asarray(time_s, dtype=np.float64) / self.duration_s
        if self.mode == "one_shot":
            progress = np.clip(normalized, 0.0, 1.0)
        elif self.mode == "ping_pong":
            progress = 1.0 - np.abs(2.0 * (normalized % 1.0) - 1.0)
        else:
            progress = normalized % 1.0
        if self.direction < 0:
            progress = 1.0 - progress
        return apply_easing(progress, self.easing)

    def progress(self, time_s: ArrayLike) -> NDArray[np.float64]:
        progress = self._continuous_progress(time_s)
        if self.mode == "discontinuous":
            progress = np.floor(progress * self.steps) / (self.steps - 1)
        return np.clip(progress, 0.0, 1.0)

    def discontinuity_blend(self, time_s: ArrayLike) -> DiscontinuityBlend:
        """Describe deterministic crossfades leading into each position jump.

        With no transition, the returned next gain is one. During the final
        ``crossfade_s`` before a step boundary, the two gains follow an
        equal-power curve and refer to the adjacent quantized path positions.
        """

        time = np.asarray(time_s, dtype=np.float64)
        current = self.progress(time)
        zeros = np.zeros_like(time)
        ones = np.ones_like(time)
        if self.mode != "discontinuous" or self.crossfade_s == 0.0:
            return DiscontinuityBlend(current, current, zeros, ones)

        step_period_s = self.duration_s / self.steps
        fade_s = min(self.crossfade_s, step_period_s)
        phase_s = np.mod(time, step_period_s)
        blend = np.clip((phase_s - (step_period_s - fade_s)) / fade_s, 0.0, 1.0)
        # Evaluate the next time cell rather than adding a position increment;
        # this correctly crossfades the final step back to the first on loop
        # wrap and also handles reverse traversal.
        adjacent = self.progress(time + step_period_s)
        return DiscontinuityBlend(
            previous_progress=current,
            next_progress=adjacent,
            previous_gain=np.cos(0.5 * np.pi * blend),
            next_gain=np.sin(0.5 * np.pi * blend),
        )

    def transition_weights(self, time_s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compatibility shorthand for :meth:`discontinuity_blend`."""

        blend = self.discontinuity_blend(time_s)
        return blend.previous_gain, blend.next_gain


@dataclass(frozen=True)
class KeyframedTraversal:
    """User-authored time-to-position law with hold, linear, or smooth keys."""

    times_s: tuple[float, ...]
    positions: tuple[float, ...]
    interpolation: Literal["hold", "linear", "smooth"] = "linear"

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        if len(times) < 2 or times.shape != positions.shape:
            raise ValueError("keyframed traversal requires matching time/position arrays")
        if np.any(np.diff(times) <= 0.0) or np.any((positions < 0.0) | (positions > 1.0)):
            raise ValueError("key times must increase and positions must lie in [0, 1]")
        if self.interpolation not in ("hold", "linear", "smooth"):
            raise ValueError(f"unsupported keyframe interpolation: {self.interpolation!r}")

    def progress(self, time_s: ArrayLike) -> NDArray[np.float64]:
        requested = np.asarray(time_s, dtype=np.float64)
        times = np.asarray(self.times_s)
        positions = np.asarray(self.positions)
        if self.interpolation == "hold":
            indices = np.clip(np.searchsorted(times, requested, side="right") - 1, 0, len(times) - 1)
            return positions[indices]
        if self.interpolation == "smooth":
            indices = np.clip(np.searchsorted(times, requested, side="right") - 1, 0, len(times) - 2)
            fraction = (requested - times[indices]) / (times[indices + 1] - times[indices])
            fraction = apply_easing(fraction, "smoothstep")
            result = positions[indices] + (positions[indices + 1] - positions[indices]) * fraction
            return np.where(requested <= times[0], positions[0], np.where(requested >= times[-1], positions[-1], result))
        return np.interp(requested, times, positions)


@dataclass(frozen=True)
class StochasticTraversal:
    """Seeded sample-and-hold random positions with optional smoothing."""

    interval_s: float = 1.0
    seed: int = 0
    smoothing: Literal["hold", "linear", "smooth"] = "hold"

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("interval_s must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def _value(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        # Integer hashing makes results independent of block order/size.
        values = indices.astype(np.uint64) + np.uint64(self.seed)
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
        return (values >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    def progress(self, time_s: ArrayLike) -> NDArray[np.float64]:
        normalized = np.maximum(np.asarray(time_s, dtype=np.float64), 0.0) / self.interval_s
        index = np.floor(normalized).astype(np.int64)
        current = self._value(index)
        if self.smoothing == "hold":
            return current
        fraction = normalized - index
        if self.smoothing == "smooth":
            fraction = apply_easing(fraction, "smoothstep")
        elif self.smoothing != "linear":
            raise ValueError(f"unsupported stochastic smoothing: {self.smoothing!r}")
        return current + (self._value(index + 1) - current) * fraction


@dataclass(frozen=True)
class CanonicalTrajectory:
    """Compose reusable path geometry with an independent traversal law."""

    geometry: object
    traversal: Traversal = Traversal()
    arc_length: bool = True
    arclength_samples: int = 2049

    def _positions(self, progress: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.arc_length:
            from .geometry import evaluate_arclength

            return evaluate_arclength(self.geometry, progress, self.arclength_samples)
        return self.geometry.evaluate(progress)

    def evaluate(self, time_s: ArrayLike) -> NDArray[np.float64]:
        if not isinstance(self.traversal, Traversal) or self.traversal.mode != "discontinuous" or self.traversal.crossfade_s == 0.0:
            return self._positions(self.traversal.progress(time_s))
        blend = self.traversal.discontinuity_blend(time_s)
        previous = self._positions(blend.previous_progress)
        following = self._positions(blend.next_progress)
        # Equal-power audio crossfades can exceed unity for correlated values;
        # normalize here because these are coordinates, not audio amplitudes.
        total = np.maximum(blend.previous_gain + blend.next_gain, np.finfo(float).eps)
        return (
            previous * blend.previous_gain[..., None]
            + following * blend.next_gain[..., None]
        ) / total[..., None]

    __call__ = evaluate


def segment_positions(segments: Sequence[dict], time_s: ArrayLike) -> NDArray[np.float64]:
    """Evaluate legacy trajectory-dialog segments without modifying them."""

    if not segments:
        raise ValueError("at least one segment is required")
    durations = np.asarray([float(segment.get("seconds", 0.0)) for segment in segments])
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("each legacy segment must have a positive finite duration")
    boundaries = np.concatenate(([0.0], np.cumsum(durations)))
    requested = np.asarray(time_s, dtype=np.float64)
    flat_time = np.clip(requested.reshape(-1), 0.0, boundaries[-1])
    result = np.empty((len(flat_time), 3), dtype=np.float64)

    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        mask = (flat_time >= boundaries[index]) & (
            (flat_time <= boundaries[index + 1]) if is_last else (flat_time < boundaries[index + 1])
        )
        local_time = flat_time[mask] - boundaries[index]
        normalized = apply_easing(local_time / durations[index], segment.get("easing", "linear"))
        azimuth = _segment_azimuth(segment, local_time)
        distance = _segment_distance(segment.get("distance_m", 1.0), normalized)
        angle = np.radians(azimuth)
        result[mask] = np.stack((distance * np.cos(angle), distance * np.sin(angle), np.zeros_like(angle)), axis=-1)
    return result.reshape(requested.shape + (3,))


def _segment_azimuth(segment: dict, local_time: NDArray[np.float64]) -> NDArray[np.float64]:
    mode = segment.get("mode", "rotate")
    if mode == "rotate":
        return float(segment.get("start_deg", 0.0)) + float(segment.get("speed_deg_per_s", 0.0)) * local_time
    period_s = float(segment.get("period_s", 1.0))
    if not math.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("legacy segment period_s must be finite and positive")
    oscillation = float(segment.get("extent_deg", 0.0)) * np.sin(2.0 * np.pi * local_time / period_s)
    if mode == "oscillate":
        return float(segment.get("center_deg", 0.0)) + oscillation
    if mode == "rotating_arc":
        return float(segment.get("start_deg", 0.0)) + 360.0 * float(segment.get("rotate_freq_hz", 0.0)) * local_time + oscillation
    raise ValueError(f"unsupported legacy segment mode: {mode!r}")


def _segment_distance(value: object, progress: NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = map(float, value)
        return start + (end - start) * progress
    return np.full_like(progress, float(value), dtype=np.float64)

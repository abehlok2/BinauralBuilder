"""Free-field geometric binaural rendering with fractional propagation delay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.audio.sam_workbench.conventions import AUDIO_DTYPE, DEFAULT_SPEED_OF_SOUND_M_S
from src.audio.sam_workbench.trajectory.transforms import ListenerTransform

DistanceLaw = Literal["inverse", "inverse_square", "none"]


@dataclass(frozen=True)
class GeometricSpec:
    """Configuration for a free-field geometric renderer."""

    trajectory: object
    listener: ListenerTransform = ListenerTransform()
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S
    distance_law: DistanceLaw | Callable[[NDArray[np.float64]], ArrayLike] = "inverse"
    reference_distance_m: float = 1.0
    minimum_distance_m: float = 0.05
    doppler_enabled: bool = True
    maximum_distance_m: float = 1000.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed_of_sound_m_s) or self.speed_of_sound_m_s <= 0.0:
            raise ValueError("speed_of_sound_m_s must be finite and positive")
        if not math.isfinite(self.reference_distance_m) or self.reference_distance_m <= 0.0:
            raise ValueError("reference_distance_m must be finite and positive")
        if not math.isfinite(self.minimum_distance_m) or self.minimum_distance_m <= 0.0:
            raise ValueError("minimum_distance_m must be finite and positive")
        if self.maximum_distance_m < self.minimum_distance_m:
            raise ValueError("maximum_distance_m must not be smaller than minimum_distance_m")
        if not callable(self.distance_law) and self.distance_law not in ("inverse", "inverse_square", "none"):
            raise ValueError(f"unsupported distance law: {self.distance_law!r}")


def cubic_fractional_sample(signal: ArrayLike, indices: ArrayLike) -> NDArray[np.float64]:
    """Sample a 1D signal using four-point Catmull-Rom interpolation.

    Samples outside the supplied interval are zero rather than edge-clamped,
    which makes physical propagation latency explicit and avoids inventing a
    pre-roll sample.
    """

    values = np.asarray(signal, dtype=np.float64)
    requested = np.asarray(indices, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {values.shape}")
    integer = np.floor(requested).astype(np.int64)
    fraction = requested - integer

    def sample(offset: int) -> NDArray[np.float64]:
        indices_at_offset = integer + offset
        result = np.zeros_like(requested, dtype=np.float64)
        valid = (indices_at_offset >= 0) & (indices_at_offset < len(values))
        result[valid] = values[indices_at_offset[valid]]
        return result

    p0, p1, p2, p3 = sample(-1), sample(0), sample(1), sample(2)
    return p1 + 0.5 * fraction * (
        p2 - p0
        + fraction
        * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + fraction * (3.0 * (p1 - p2) + p3 - p0))
    )


def geometric_cues(
    source_positions_m: ArrayLike,
    listener: ListenerTransform = ListenerTransform(),
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
) -> dict[str, NDArray[np.float64]]:
    """Return left/right distance, delay, common delay, and left-minus-right ITD."""

    positions = np.asarray(source_positions_m, dtype=np.float64)
    if positions.shape[-1:] != (3,):
        raise ValueError(f"source positions must end in three coordinates, got {positions.shape}")
    if speed_of_sound_m_s <= 0.0:
        raise ValueError("speed_of_sound_m_s must be positive")
    distances = np.linalg.norm(positions[..., None, :] - listener.ear_positions_world(), axis=-1)
    delays = distances / float(speed_of_sound_m_s)
    return {
        "distance_m": distances,
        "delay_s": delays,
        "common_delay_s": np.mean(delays, axis=-1),
        "itd_s": delays[..., 0] - delays[..., 1],
    }


def distance_gains(distances_m: ArrayLike, spec: GeometricSpec) -> NDArray[np.float64]:
    """Evaluate the configured distance law with one shared physical policy."""

    distance = np.clip(np.asarray(distances_m, dtype=np.float64), spec.minimum_distance_m, spec.maximum_distance_m)
    if callable(spec.distance_law):
        gains = np.asarray(spec.distance_law(distance), dtype=np.float64)
        if gains.shape != distance.shape:
            raise ValueError("custom distance law must preserve the distance array shape")
    elif spec.distance_law == "none":
        gains = np.ones_like(distance)
    elif spec.distance_law == "inverse_square":
        gains = np.square(spec.reference_distance_m / distance)
    else:
        gains = spec.reference_distance_m / distance
    if np.any(~np.isfinite(gains)) or np.any(gains < 0.0):
        raise ValueError("distance law returned a non-finite or negative gain")
    return gains


class GeometricBinauralRenderer:
    """Stateful channel-major renderer evaluated on the absolute sample clock.

    Doppler mode uses the complete time-varying propagation delay. Bypass mode
    removes the common radial delay variation while retaining differential ear
    delay (ITD), so it remains block invariant and spatially meaningful.
    """

    def __init__(self, spec: GeometricSpec):
        self.spec = spec
        self._sample_rate_hz = 44_100.0
        self._history = np.zeros(0, dtype=np.float64)
        self._next_sample = 0
        self._diagnostics: dict[str, NDArray[np.float64]] = {}

    def prepare(self, context, source=None) -> None:
        self._sample_rate_hz = float(context.sample_rate_hz)
        self.reset(0)

    def reset(self, sample_index: int = 0) -> None:
        self._history = np.zeros(0, dtype=np.float64)
        self._next_sample = int(sample_index)

    def latency_samples(self) -> int:
        """Return scheduling latency; physical propagation remains in the signal."""

        return 0

    def diagnostics(self) -> dict[str, NDArray[np.float64]]:
        return dict(self._diagnostics)

    def _positions(self, time_s: NDArray[np.float64]) -> NDArray[np.float64]:
        trajectory = self.spec.trajectory
        positions = trajectory(time_s) if callable(trajectory) else trajectory.evaluate(time_s)
        result = np.asarray(positions, dtype=np.float64)
        if result.shape != (len(time_s), 3):
            raise ValueError(f"trajectory returned {result.shape}; expected {(len(time_s), 3)}")
        return result

    def process(self, mono: ArrayLike, block) -> NDArray[np.float32]:
        source = np.asarray(mono, dtype=np.float64)
        if source.ndim != 1 or len(source) != block.frames:
            raise ValueError(f"mono input must have shape ({block.frames},), got {source.shape}")
        start_sample = int(block.start_sample)
        if start_sample != self._next_sample:
            # A seek has no source pre-roll. Reset explicitly rather than using
            # stale samples from a different absolute interval.
            self.reset(start_sample)

        sample_indices = start_sample + np.arange(block.frames, dtype=np.float64)
        time_s = sample_indices / self._sample_rate_hz
        positions = self._positions(time_s)
        cues = geometric_cues(positions, self.spec.listener, self.spec.speed_of_sound_m_s)
        gains = distance_gains(cues["distance_m"], self.spec)
        delay_samples = cues["delay_s"] * self._sample_rate_hz
        if not self.spec.doppler_enabled:
            differential = delay_samples - np.mean(delay_samples, axis=1, keepdims=True)
            # Half the ear spacing is a geometry-wide upper bound on the
            # differential propagation distance, so this fixed common delay
            # remains causal without depending on block contents.
            causal_reference_m = max(
                self.spec.reference_distance_m,
                self.spec.listener.ear_spacing_m / 2.0,
            )
            delay_samples = (
                causal_reference_m
                / self.spec.speed_of_sound_m_s
                * self._sample_rate_hz
                + differential
            )

        combined = np.concatenate((self._history, source))
        combined_origin = start_sample - len(self._history)
        output = np.empty((2, block.frames), dtype=np.float64)
        for ear in range(2):
            read_positions = sample_indices - delay_samples[:, ear] - combined_origin
            output[ear] = cubic_fractional_sample(combined, read_positions) * gains[:, ear]

        required_history = max(4, int(math.ceil(float(np.max(delay_samples, initial=0.0)))) + 3)
        self._history = combined[-required_history:]
        self._next_sample = start_sample + block.frames
        self._diagnostics = {
            **cues,
            "gain_linear": gains,
            "effective_delay_s": delay_samples / self._sample_rate_hz,
            "source_position_m": positions,
        }
        return output.astype(AUDIO_DTYPE, copy=False)

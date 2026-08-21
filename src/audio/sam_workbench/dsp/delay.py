"""Fractional delay lines.

A moving source's delay changes continuously, so reading the delay line at
whole samples would quantize the interaural time difference into audible steps
and add a buzz on fast motion. Catmull-Rom interpolation between four
neighbouring samples gives a smooth read at a fractional position.

The line keeps its history across blocks, so a chunked render is identical to a
whole one: the state is the tail of the input, not a phase that could restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["FractionalDelayLine", "cubic_interpolate", "read_fractional"]


def cubic_interpolate(
    buffer: NDArray[np.floating],
    positions: NDArray[np.floating],
    *,
    outside: str = "clamp",
) -> NDArray[np.float64]:
    """Catmull-Rom read of ``buffer`` at fractional ``positions``.

    ``outside`` decides what a read beyond the buffer means:

    * ``"clamp"`` - hold the edge sample. Right for a delay line, where a
      position before the start means "older than the history we kept".
    * ``"zero"`` - return silence. Right for shifting a finite response, where
      clamping would smear the first sample into a plateau and destroy the
      thing being shifted.
    """

    if outside not in ("clamp", "zero"):
        raise ValueError(f"unknown outside mode {outside!r}; expected 'clamp' or 'zero'")

    samples = np.asarray(buffer, dtype=np.float64)
    if samples.size == 0:
        return np.zeros(np.shape(positions), dtype=np.float64)

    requested = np.asarray(positions, dtype=np.float64)
    location = np.clip(requested, 0.0, samples.size - 1.0)
    index = np.floor(location).astype(np.int64)
    fraction = location - index

    def tap(offset: int) -> NDArray[np.float64]:
        return samples[np.clip(index + offset, 0, samples.size - 1)]

    previous, current, following, after = tap(-1), tap(0), tap(1), tap(2)
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    values = 0.5 * (
        (2.0 * current)
        + (-previous + following) * fraction
        + (2.0 * previous - 5.0 * current + 4.0 * following - after) * fraction2
        + (-previous + 3.0 * current - 3.0 * following + after) * fraction3
    )
    if outside == "zero":
        values = np.where((requested < 0.0) | (requested > samples.size - 1.0), 0.0, values)
    return values


def read_fractional(
    signal: NDArray[np.floating],
    delays_samples: NDArray[np.floating],
    *,
    outside: str = "clamp",
) -> NDArray[np.float64]:
    """Read ``signal`` delayed by a per-sample amount, without any state.

    Sample ``n`` of the result is ``signal`` at ``n - delays_samples[n]``.
    """

    samples = np.asarray(signal, dtype=np.float64)
    delays = np.asarray(delays_samples, dtype=np.float64)
    positions = np.arange(samples.size, dtype=np.float64) - delays
    return cubic_interpolate(samples, positions, outside=outside)


@dataclass
class FractionalDelayLine:
    """A delay line that survives block boundaries.

    ``max_delay_samples`` sizes the history. Three extra samples are kept for
    the interpolator's neighbours, so a read at the maximum delay still has all
    four taps available.
    """

    max_delay_samples: int = 4_096
    _history: NDArray[np.float64] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    @property
    def history_frames(self) -> int:
        return int(max(4, self.max_delay_samples + 3))

    def reset(self) -> None:
        """Clear the line. Only ever call this at a genuine transport reset."""

        object.__setattr__(self, "_history", np.zeros(self.history_frames, dtype=np.float64))

    def process(
        self, block: NDArray[np.floating], delays_samples: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Delay one block, carrying the tail of the input into the next call."""

        samples = np.asarray(block, dtype=np.float64)
        delays = np.asarray(delays_samples, dtype=np.float64)
        if delays.shape != samples.shape:
            if delays.size == 1:
                delays = np.full(samples.shape, float(delays.reshape(-1)[0]))
            else:
                raise ValueError(f"delay shape {delays.shape} does not match block {samples.shape}")

        history = self._history
        combined = np.concatenate((history, samples))
        offset = history.size
        positions = offset + np.arange(samples.size, dtype=np.float64) - np.clip(
            delays, 0.0, float(self.max_delay_samples)
        )
        output = cubic_interpolate(combined, positions)

        keep = self.history_frames
        self._history = combined[-keep:] if combined.size >= keep else np.concatenate(
            (np.zeros(keep - combined.size), combined)
        )
        return output

    def state(self) -> dict:
        """A serializable checkpoint, for the chunked BinauralBuilder path."""

        return {"history": self._history.tolist(), "max_delay_samples": int(self.max_delay_samples)}

    @classmethod
    def from_state(cls, state: dict | None) -> "FractionalDelayLine":
        if not state:
            return cls()
        line = cls(max_delay_samples=int(state.get("max_delay_samples", 4_096)))
        history = np.asarray(state.get("history", ()), dtype=np.float64)
        if history.size:
            line._history = np.resize(history, line.history_frames)
        return line

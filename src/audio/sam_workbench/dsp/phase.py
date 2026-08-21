"""Phase accumulation.

For a time-varying frequency the phase must be *integrated*:

.. code-block:: text

    phase[n + 1] = phase[n] + 2 * pi * frequency[n] / sample_rate

Computing ``sin(2 * pi * f[n] * t[n])`` instead gives the wrong phase as soon
as the frequency moves, which is audible as a pitch error and a click wherever
a render is split into blocks.

Two accumulation strategies are provided, both anchored to an absolute sample
index so a chunked render matches a whole-step render:

* :func:`phase_from_constant_frequency` - closed form, exact, no state.
* :class:`PhaseAccumulator` - checkpointable integration for arbitrary
  frequency curves, resumable at any absolute sample index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.waveforms import TWO_PI

__all__ = [
    "PhaseAccumulator",
    "phase_from_constant_frequency",
    "integrate_phase",
    "wrap_phase",
]

#: Wrap the accumulator once it exceeds this many radians, to keep float64
#: resolution high over long renders without changing the sine's value.
_WRAP_THRESHOLD_RAD = 2.0**24 * TWO_PI


def wrap_phase(phase_rad: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Wrap phase into ``[0, 2*pi)``."""

    return np.mod(np.asarray(phase_rad, dtype=np.float64), TWO_PI)


def phase_from_constant_frequency(
    frequency_hz: float,
    start_sample: int,
    frames: int,
    sample_rate: float,
    initial_phase_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Exact phase for a fixed frequency, evaluated from an absolute sample index.

    This is the preferred path for constant-frequency abstract SAM: it needs no
    state, and a chunked render is bit-identical to a whole render.
    """

    index = np.arange(int(start_sample), int(start_sample) + int(frames), dtype=np.float64)
    return TWO_PI * float(frequency_hz) * (index / float(sample_rate)) + float(initial_phase_rad)


def integrate_phase(
    frequency_hz: NDArray[np.floating],
    sample_rate: float,
    initial_phase_rad: float = 0.0,
    *,
    inclusive: bool = False,
) -> NDArray[np.float64]:
    """Integrate a frequency curve into phase.

    With ``inclusive=False`` (the default) ``phase[n]`` is the phase *at* sample
    ``n``, so ``phase[0] == initial_phase_rad``. ``inclusive=True`` reproduces
    the legacy ``numpy.cumsum`` convention, where the first sample already
    carries one frequency increment; it exists so legacy voices keep rendering
    exactly as they did.
    """

    increments = TWO_PI * np.asarray(frequency_hz, dtype=np.float64) / float(sample_rate)
    accumulated = np.cumsum(increments)
    if inclusive:
        return float(initial_phase_rad) + accumulated
    return float(initial_phase_rad) + np.concatenate((np.zeros(1, dtype=np.float64), accumulated[:-1]))


@dataclass
class PhaseAccumulator:
    """Resumable phase integrator for arbitrary frequency curves.

    The accumulator stores the absolute sample index it has reached alongside
    the phase, so a caller can assert that blocks arrive in order and that no
    chunk boundary silently restarted the integration.
    """

    phase_rad: float = 0.0
    sample_index: int = 0

    def reset(self, sample_index: int = 0, phase_rad: float = 0.0) -> None:
        self.sample_index = int(sample_index)
        self.phase_rad = float(phase_rad)

    def advance(
        self, frequency_hz: NDArray[np.floating], sample_rate: float
    ) -> NDArray[np.float64]:
        """Return the phase for one block and carry the state to the next."""

        frequency = np.asarray(frequency_hz, dtype=np.float64)
        phase = integrate_phase(frequency, sample_rate, self.phase_rad)
        if frequency.size:
            end_phase = float(phase[-1]) + TWO_PI * float(frequency[-1]) / float(sample_rate)
            if abs(end_phase) > _WRAP_THRESHOLD_RAD:
                end_phase = float(wrap_phase(end_phase))
            self.phase_rad = end_phase
            self.sample_index += int(frequency.size)
        return phase

    def state(self) -> dict[str, float]:
        """Serializable checkpoint, for the chunked BinauralBuilder path."""

        return {"phase_rad": float(self.phase_rad), "sample_index": int(self.sample_index)}

    @classmethod
    def from_state(cls, state: dict[str, float] | None) -> "PhaseAccumulator":
        if not state:
            return cls()
        return cls(phase_rad=float(state.get("phase_rad", 0.0)), sample_index=int(state.get("sample_index", 0)))

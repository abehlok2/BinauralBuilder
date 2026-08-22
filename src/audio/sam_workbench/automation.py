"""Applying compiled automation to a renderer that expects scalars.

The compiled plan describes every automated parameter as a function of the
absolute sample index. The legacy SAM2 renderer expects numbers. Bridging the
two by evaluating each parameter once per render chunk - which is what used to
happen - makes the output depend on how the caller divided the timeline, and a
preview that disagrees with an export is the disagreement a user trusts least.

Two kinds of parameter need different treatment:

* **Shape parameters** - arc width, direction offset, spatial scale - are read
  once per sample from a time axis the renderer already builds. Handing those an
  array instead of a number is enough.

* **Frequency parameters** - the modulation rate and the carrier - are not read,
  they are *integrated*. Phase at a sample is the integral of frequency from the
  source's origin to that sample, so it cannot be recovered from the frequency
  at that sample alone. :class:`AutomatedPhase` does that integral as a pure
  function of the absolute index.

A voice whose scene automates neither keeps the exact closed-form phase it
always had, so the established SAM2 acoustic behaviour is untouched for every
project that does not use these controls.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["AutomatedPhase", "TWO_PI"]

TWO_PI = 2.0 * math.pi

#: How often the integrator samples the frequency, in samples. Stage
#: transitions last seconds and modulators run at a few hertz, so 64 samples at
#: 44.1 kHz - roughly 690 Hz - resolves both with room to spare, while keeping
#: the prefix table small enough to extend cheaply.
DEFAULT_GRID = 64


class AutomatedPhase:
    """Phase in radians from integrating a time-varying frequency.

    ``frequency_at(start_sample, frames)`` returns the frequency in hertz over
    an absolute window.  :meth:`at` returns the phase accumulated from sample
    zero, which makes it a pure function of the absolute index: the same window
    produces the same phase no matter what was rendered before it, or in what
    order.

    Phase at a late sample depends on every sample before it, so the integral
    is built on a coarse grid and cached.  The cache is memoization and nothing
    else - it only ever grows, and it holds values that were already determined
    by the frequency function - so the value at a sample does not depend on
    what was rendered before it or in what order.

    Agreement between partitions is to floating-point tolerance rather than bit
    equality: summing the same increments in a different order rounds
    differently, and a window that begins mid-grid accumulates from a different
    starting point than one that spans the grid whole.  Observed disagreement
    is on the order of 1e-14 radians over tens of thousands of samples, which
    is far below anything audible and matches the documented determinism rule
    for varying block sizes.  A voice with no automated frequency does not come
    through here at all and stays bit-exact.
    """

    def __init__(
        self,
        frequency_at: Callable[[int, int], NDArray[np.float64]],
        sample_rate_hz: float,
        *,
        grid: int = DEFAULT_GRID,
        phase_offset_rad: float = 0.0,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        self._frequency_at = frequency_at
        self._sample_rate_hz = float(sample_rate_hz)
        self._grid = max(1, int(grid))
        self._phase_offset_rad = float(phase_offset_rad)
        # ``_prefix[k]`` is the phase accumulated at sample ``k * grid``.
        self._prefix: list[float] = [0.0]

    @property
    def grid(self) -> int:
        return self._grid

    def _extend_to(self, cells: int) -> None:
        """Integrate forward until the prefix table covers ``cells`` cells."""

        while len(self._prefix) <= cells:
            start = (len(self._prefix) - 1) * self._grid
            # One extra sample so the trapezoid closes on the next grid point.
            frequency = np.asarray(
                self._frequency_at(start, self._grid + 1), dtype=np.float64
            )
            step = TWO_PI * float(np.trapezoid(frequency, dx=1.0 / self._sample_rate_hz))
            self._prefix.append(self._prefix[-1] + step)

    def at(self, start_sample: int, frames: int) -> NDArray[np.float64]:
        """Phase over ``frames`` samples beginning at ``start_sample``."""

        if frames <= 0:
            return np.zeros(0, dtype=np.float64)
        start = int(start_sample)
        if start < 0:
            raise ValueError("automated phase is accumulated from sample zero")

        cell = start // self._grid
        self._extend_to(cell)
        base_sample = cell * self._grid
        # Integrate the remainder at full resolution, from the grid point below
        # the requested start through the whole window.
        span = (start - base_sample) + int(frames)
        frequency = np.asarray(self._frequency_at(base_sample, span), dtype=np.float64)
        increments = np.concatenate(
            ([0.0], np.cumsum(0.5 * (frequency[1:] + frequency[:-1])))
        ) * (TWO_PI / self._sample_rate_hz)
        phase = self._prefix[cell] + increments[start - base_sample :]
        return phase[: int(frames)] + self._phase_offset_rad

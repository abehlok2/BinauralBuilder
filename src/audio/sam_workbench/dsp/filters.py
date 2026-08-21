"""Small filters used by the geometric renderer.

Only what the specification asks for at this stage: an optional *simple*
frequency-dependent head shadow. A shadowed ear hears less high frequency than
a facing one; a one-pole low-pass whose cutoff follows the incidence angle
captures that without pretending to be a measured HRTF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

__all__ = ["OnePoleLowpass", "head_shadow_cutoff_hz"]

#: Cutoff for an ear facing the source: effectively open.
OPEN_CUTOFF_HZ = 18_000.0
#: Cutoff for an ear fully in shadow.
SHADOWED_CUTOFF_HZ = 2_000.0


def head_shadow_cutoff_hz(incidence_cosine: float | NDArray[np.floating]) -> NDArray[np.float64]:
    """Low-pass cutoff for an ear, from the cosine of its incidence angle.

    ``+1`` is an ear pointing straight at the source and ``-1`` one turned
    fully away. The interpolation is logarithmic in frequency, which matches
    how the shadow is heard far better than a linear sweep in hertz.
    """

    cosine = np.clip(np.asarray(incidence_cosine, dtype=np.float64), -1.0, 1.0)
    openness = 0.5 * (cosine + 1.0)
    log_low, log_high = math.log(SHADOWED_CUTOFF_HZ), math.log(OPEN_CUTOFF_HZ)
    return np.exp(log_low + (log_high - log_low) * openness)


@dataclass
class OnePoleLowpass:
    """A one-pole low-pass whose cutoff may change from block to block.

    The coefficient is recomputed per block rather than per sample: the head
    shadow moves at the speed of a source, not of a waveform, and a per-sample
    coefficient would cost far more than it is worth here. Filter state carries
    across blocks so there is no discontinuity at a boundary.
    """

    sample_rate_hz: float = 44_100.0
    cutoff_hz: float = OPEN_CUTOFF_HZ
    _state: NDArray[np.float64] | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._state = None

    def _coefficients(self, cutoff_hz: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = 0.5 * float(self.sample_rate_hz)
        cutoff = float(np.clip(cutoff_hz, 20.0, nyquist * 0.99))
        alpha = math.exp(-2.0 * math.pi * cutoff / float(self.sample_rate_hz))
        return np.array([1.0 - alpha]), np.array([1.0, -alpha])

    def process(self, block: NDArray[np.floating], cutoff_hz: float | None = None) -> NDArray[np.float64]:
        samples = np.asarray(block, dtype=np.float64)
        if samples.size == 0:
            return samples
        numerator, denominator = self._coefficients(
            self.cutoff_hz if cutoff_hz is None else cutoff_hz
        )
        if self._state is None:
            self._state = scipy_signal.lfilter_zi(numerator, denominator) * samples[0]
        filtered, self._state = scipy_signal.lfilter(numerator, denominator, samples, zi=self._state)
        return filtered

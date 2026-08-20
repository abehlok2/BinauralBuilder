"""Serializable controls rendered from absolute sample positions.

Absolute positioning makes every control deterministic across preview and
offline block sizes.  Composition and additional control variants can build
on the same small interface without coupling controls to a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import numpy.typing as npt


class CompiledControl(Protocol):
    def render(self, start_sample: int, frames: int, sample_rate_hz: int) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class ConstantControl:
    value: float
    type: str = "constant"

    def render(self, start_sample: int, frames: int, sample_rate_hz: int) -> npt.NDArray[np.float64]:
        _validate_block(start_sample, frames, sample_rate_hz)
        return np.full(frames, self.value, dtype=np.float64)


@dataclass(frozen=True)
class LfoControl:
    rate_hz: float
    depth: float = 1.0
    offset: float = 0.0
    phase_rad: float = 0.0
    type: str = "lfo"

    def render(self, start_sample: int, frames: int, sample_rate_hz: int) -> npt.NDArray[np.float64]:
        _validate_block(start_sample, frames, sample_rate_hz)
        if not all(math.isfinite(value) for value in (self.rate_hz, self.depth, self.offset, self.phase_rad)):
            raise ValueError("LFO parameters must be finite")
        sample = start_sample + np.arange(frames, dtype=np.float64)
        phase = self.phase_rad + 2.0 * math.pi * self.rate_hz * sample / sample_rate_hz
        return self.offset + self.depth * np.sin(phase)


def _validate_block(start_sample: int, frames: int, sample_rate_hz: int) -> None:
    if start_sample < 0 or frames < 0 or sample_rate_hz <= 0:
        raise ValueError("start_sample and frames must be non-negative and sample_rate_hz positive")

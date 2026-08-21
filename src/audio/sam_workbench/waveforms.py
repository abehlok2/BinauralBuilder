"""Dependency-free periodic waveform evaluation.

These functions map an accumulated phase in radians to a value in ``[-1, 1]``.
They sit at the bottom of the package's dependency graph so both the control
engine and the oscillators can use them without importing each other.

Naive (aliasing) shapes are used for control-rate modulators, where the shape
matters more than the spectrum. Audio-rate carriers use ``sine`` in Phase 1;
band-limited carrier waveforms arrive with the generalized signal model.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

TWO_PI: Final[float] = 2.0 * math.pi

#: Waveforms accepted wherever a modulator or oscillator shape is named.
WAVEFORMS: Final[tuple[str, ...]] = ("sine", "triangle", "square", "sawtooth")

__all__ = ["TWO_PI", "WAVEFORMS", "evaluate_waveform", "normalized_phase"]


def normalized_phase(phase_rad: NDArray[np.floating] | float) -> NDArray[np.float64]:
    """Return phase wrapped into ``[0, 1)`` cycles."""

    return np.mod(np.asarray(phase_rad, dtype=np.float64) / TWO_PI, 1.0)


def evaluate_waveform(waveform: str, phase_rad: NDArray[np.floating] | float) -> NDArray[np.float64]:
    """Evaluate a unit-amplitude periodic ``waveform`` at ``phase_rad``.

    ``sine``, ``triangle``, and ``sawtooth`` are zero at phase ``0`` and rise
    first, so a modulator waveform can be swapped without flipping the sign of
    the modulation at the start of a cycle. ``square`` is ``+1`` over the first
    half-cycle for the same reason.
    """

    name = str(waveform).lower()
    phase = np.asarray(phase_rad, dtype=np.float64)
    if name == "sine":
        return np.sin(phase)
    cycles = normalized_phase(phase)
    if name == "triangle":
        return 1.0 - 4.0 * np.abs(np.mod(cycles + 0.25, 1.0) - 0.5)
    if name == "square":
        return np.where(cycles < 0.5, 1.0, -1.0)
    if name == "sawtooth":
        return 2.0 * np.mod(cycles + 0.5, 1.0) - 1.0
    raise ValueError(f"unknown waveform {waveform!r}; expected one of {', '.join(WAVEFORMS)}")

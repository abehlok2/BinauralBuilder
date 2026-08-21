"""Splitting a signal into bands that add back up to what went in.

A multiband spatializer is only honest if the bands reconstruct. If summing
them changes the timbre, then every judgement made about a per-band setting is
confounded by a filter artefact, and the listener is comparing the crossover
rather than the thing they meant to compare.

Linkwitz-Riley is the default here because its complementary outputs sum to an
allpass: flat magnitude, with a phase shift shared by both halves. Cascading
two Butterworth sections of half the order gives the -6 dB crossing point and
matching slopes that make that work.

Splitting into three or more bands needs one extra step. Once the upper half
has been split again at a higher frequency, the low band has been through one
crossover fewer than its neighbours, so its phase no longer lines up. Passing
it through the *allpass* of that second crossover - the sum of that crossover's
own two halves - puts it back in step. Without that correction the bands still
look right individually and cancel where they overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CROSSOVER_ORDER",
    "CrossoverBank",
    "linkwitz_riley_sections",
    "split_bands",
]

#: Linkwitz-Riley fourth order: two cascaded second-order Butterworth sections.
DEFAULT_CROSSOVER_ORDER = 4


def linkwitz_riley_sections(
    cutoff_hz: float, sample_rate_hz: float, *, order: int = DEFAULT_CROSSOVER_ORDER
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Second-order-section low and high halves of one crossover.

    A Linkwitz-Riley of order ``N`` is a Butterworth of order ``N / 2`` applied
    twice, which is why the order must be even.
    """

    from scipy.signal import butter

    order = int(order)
    if order < 2 or order % 2:
        raise ValueError(f"a Linkwitz-Riley order must be even and at least 2, got {order}")

    nyquist = 0.5 * float(sample_rate_hz)
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"crossover at {cutoff_hz} Hz must lie between 0 and Nyquist ({nyquist} Hz)"
        )

    half = order // 2
    normalised = float(cutoff_hz) / nyquist
    low = butter(half, normalised, btype="low", output="sos")
    high = butter(half, normalised, btype="high", output="sos")
    return low, high


def _apply_twice(sections: NDArray[np.float64], signal: NDArray[np.floating]) -> NDArray[np.float64]:
    """Run a Butterworth section pair twice, making it Linkwitz-Riley."""

    from scipy.signal import sosfilt

    once = sosfilt(sections, np.asarray(signal, dtype=np.float64), axis=-1)
    return sosfilt(sections, once, axis=-1)


@dataclass(frozen=True)
class CrossoverBank:
    """A set of crossover frequencies and the bands they produce."""

    crossovers_hz: tuple[float, ...]
    sample_rate_hz: float
    order: int = DEFAULT_CROSSOVER_ORDER

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.crossovers_hz)
        if any(
            later <= earlier for earlier, later in zip(frequencies, frequencies[1:])
        ):
            raise ValueError(
                f"crossover frequencies must increase, got {self.crossovers_hz!r}"
            )
        nyquist = 0.5 * float(self.sample_rate_hz)
        for frequency in frequencies:
            if not 0.0 < frequency < nyquist:
                raise ValueError(
                    f"crossover at {frequency} Hz must lie between 0 and Nyquist ({nyquist} Hz)"
                )
        object.__setattr__(self, "crossovers_hz", frequencies)

    @property
    def band_count(self) -> int:
        return len(self.crossovers_hz) + 1

    def band_edges_hz(self) -> tuple[tuple[float, float], ...]:
        """Nominal low and high edge of each band, for display."""

        nyquist = 0.5 * float(self.sample_rate_hz)
        edges = (0.0, *self.crossovers_hz, nyquist)
        return tuple((edges[index], edges[index + 1]) for index in range(self.band_count))

    def split(self, signal: NDArray[np.floating]) -> NDArray[np.float64]:
        """Split into ``(bands, ...)``, ordered from lowest band upward."""

        return split_bands(
            signal, self.crossovers_hz, self.sample_rate_hz, order=self.order
        )


def split_bands(
    signal: NDArray[np.floating],
    crossovers_hz: Sequence[float],
    sample_rate_hz: float,
    *,
    order: int = DEFAULT_CROSSOVER_ORDER,
) -> NDArray[np.float64]:
    """Split a signal into bands that sum back to it.

    ``signal`` may be one dimensional or channel-major; filtering runs along the
    last axis. The result has one extra leading axis for the band.
    """

    samples = np.asarray(signal, dtype=np.float64)
    frequencies = [float(value) for value in crossovers_hz]
    if not frequencies:
        return samples[None, ...].copy()

    bands: list[NDArray[np.float64]] = []
    remainder = samples
    for position, cutoff in enumerate(frequencies):
        low_sections, high_sections = linkwitz_riley_sections(
            cutoff, sample_rate_hz, order=order
        )
        low = _apply_twice(low_sections, remainder)
        high = _apply_twice(high_sections, remainder)

        # Every band already emitted has now been through one crossover fewer
        # than this one. Running them through this crossover's allpass - the sum
        # of its own halves - keeps their phase in step with what follows.
        for index, earlier in enumerate(bands):
            bands[index] = _apply_twice(low_sections, earlier) + _apply_twice(
                high_sections, earlier
            )

        bands.append(low)
        remainder = high
    bands.append(remainder)
    return np.stack(bands, axis=0)

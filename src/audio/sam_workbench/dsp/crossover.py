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
    "CrossoverStream",
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
        """Split a *whole* signal into ``(bands, ...)``, lowest band first.

        For a signal arriving in blocks use :meth:`stream`: this call starts
        every filter from rest, which is right for one complete signal and
        wrong for a stream, where it would restart the filters at every block
        boundary.
        """

        return split_bands(
            signal, self.crossovers_hz, self.sample_rate_hz, order=self.order
        )

    def stream(self) -> "CrossoverStream":
        """A stateful splitter for audio arriving block by block."""

        return CrossoverStream(self)


@dataclass
class _Stage:
    """One Butterworth pass with the state that spans a block boundary."""

    sections: NDArray[np.float64]
    zi: NDArray[np.float64] | None = None

    def process(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        from scipy.signal import sosfilt

        if self.zi is None:
            # The state's shape follows the signal's, with the filtered axis
            # replaced by the two delays each section carries.
            self.zi = np.zeros(
                (self.sections.shape[0], *signal.shape[:-1], 2), dtype=np.float64
            )
        filtered, self.zi = sosfilt(self.sections, signal, axis=-1, zi=self.zi)
        return filtered

    def reset(self) -> None:
        self.zi = None


class CrossoverStream:
    """Split a stream into bands, carrying filter state across blocks.

    The stateless :func:`split_bands` restarts every filter at the start of
    whatever it is given. Called once per block that is a defect rather than an
    approximation: a biquad's output depends on the samples before it, and
    zeroing that history at each boundary steps the output of every band. The
    error is not small - roughly a quarter of the signal's peak at a 512-sample
    block - and it is a step, which is audible as a click rather than as a
    change of tone.

    So the filters live here instead, designed once and holding their state
    between calls. Feeding this a stream in blocks gives exactly what
    :func:`split_bands` gives for the whole stream at once.
    """

    def __init__(self, bank: "CrossoverBank") -> None:
        self.bank = bank
        # Designing the sections costs a Butterworth solve per crossover, which
        # does not belong on the audio path: it is the same filter every block.
        self._sections = [
            linkwitz_riley_sections(cutoff, bank.sample_rate_hz, order=bank.order)
            for cutoff in bank.crossovers_hz
        ]
        self._build_stages()

    def _build_stages(self) -> None:
        # Linkwitz-Riley is a Butterworth applied twice, so each half is two
        # stages; the correction path needs its own state for every earlier
        # band it has to bring back into step.
        self._main: list[tuple[list[_Stage], list[_Stage]]] = []
        self._correction: list[list[tuple[list[_Stage], list[_Stage]]]] = []
        for position, (low, high) in enumerate(self._sections):
            self._main.append(
                ([_Stage(low), _Stage(low)], [_Stage(high), _Stage(high)])
            )
            self._correction.append(
                [
                    ([_Stage(low), _Stage(low)], [_Stage(high), _Stage(high)])
                    for _ in range(position)
                ]
            )

    @property
    def band_count(self) -> int:
        return self.bank.band_count

    def reset(self) -> None:
        """Forget the history, as at the start of a new stream."""

        for low, high in self._main:
            for stage in (*low, *high):
                stage.reset()
        for group in self._correction:
            for low, high in group:
                for stage in (*low, *high):
                    stage.reset()

    @staticmethod
    def _run(stages: list[_Stage], signal: NDArray[np.float64]) -> NDArray[np.float64]:
        for stage in stages:
            signal = stage.process(signal)
        return signal

    def process(self, block: NDArray[np.floating]) -> NDArray[np.float64]:
        """Split one block into ``(bands, ...)``, lowest band first."""

        samples = np.asarray(block, dtype=np.float64)
        if not self.bank.crossovers_hz:
            return samples[None, ...].copy()

        bands: list[NDArray[np.float64]] = []
        remainder = samples
        for position, (low_stages, high_stages) in enumerate(self._main):
            low = self._run(low_stages, remainder)
            high = self._run(high_stages, remainder)
            for index, earlier in enumerate(bands):
                correction_low, correction_high = self._correction[position][index]
                bands[index] = self._run(correction_low, earlier) + self._run(
                    correction_high, earlier
                )
            bands.append(low)
            remainder = high
        bands.append(remainder)
        return np.stack(bands, axis=0)


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

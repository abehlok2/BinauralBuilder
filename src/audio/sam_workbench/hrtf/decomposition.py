"""Separating an HRIR into delay, magnitude, and phase.

Interpolating raw HRIR samples is what smears transients and weakens
interaural timing: two measurements of the same head at neighbouring angles
carry different direction-dependent delays, and averaging them comb-filters
the result rather than moving the source. Every interpolation mode above
"nearest" therefore works on a decomposition:

* a **broadband delay** per response, estimated against its own minimum-phase
  version, which is where the direction-dependent timing lives;
* a **log magnitude**, which interpolates smoothly because it is a slowly
  varying function of direction;
* a **minimum-phase** response, reconstructed from that magnitude.

The same decomposition is what the cue modifications of the HRTF Lab act on:
the common and differential split below is the ITD/ILD pair in the form the
specification describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..dsp.delay import read_fractional

__all__ = [
    "AlignedHrir",
    "align_pair",
    "broadband_delay_samples",
    "common_differential_delay",
    "common_differential_log_magnitude",
    "log_magnitude",
    "minimum_phase",
    "minimum_phase_from_log_magnitude",
    "shift_response",
    "reconstruct_from_log_magnitude",
]

_EPSILON = 1e-12


def minimum_phase_from_log_magnitude(
    log_magnitudes: NDArray[np.floating], taps: int
) -> NDArray[np.float64]:
    """The minimum-phase response whose magnitude is ``exp(log_magnitudes)``.

    The real cepstrum of a minimum-phase signal is causal, so folding the
    cepstrum of a log magnitude produces the phase that goes with it. Working
    from the log magnitude directly - rather than from an intermediate
    zero-phase impulse - avoids a second, lossy pass.
    """

    values = np.asarray(log_magnitudes, dtype=np.float64)
    length = 2 * (values.size - 1)
    if length <= 0:
        return np.zeros(int(taps), dtype=np.float64)

    cepstrum = np.fft.irfft(values, length)
    folded = np.zeros_like(cepstrum)
    half = length // 2
    folded[0] = cepstrum[0]
    folded[1:half] = 2.0 * cepstrum[1:half]
    folded[half] = cepstrum[half]

    reconstructed = np.fft.irfft(np.exp(np.fft.rfft(folded, length)), length)
    return reconstructed[: int(taps)]


def minimum_phase(hrir: NDArray[np.floating], oversample: int = 8) -> NDArray[np.float64]:
    """The minimum-phase response with the same magnitude spectrum.

    Oversampling the spectrum before folding reduces the time aliasing that a
    short cepstrum would otherwise introduce.
    """

    samples = np.asarray(hrir, dtype=np.float64)
    length = samples.size
    if length == 0:
        return samples.copy()

    padded_length = int(2 ** np.ceil(np.log2(max(8, length * max(1, oversample)))))
    spectrum = np.fft.rfft(samples, padded_length)
    log_spectrum = np.log(np.maximum(np.abs(spectrum), _EPSILON))
    return minimum_phase_from_log_magnitude(log_spectrum, length)


def broadband_delay_samples(hrir: NDArray[np.floating]) -> float:
    """Excess delay of a response over its own minimum-phase version, in samples.

    This is the robust way to ask "when does this response arrive?": a simple
    onset threshold moves with the level of the direction, while the excess
    delay is a property of the response's phase.
    """

    samples = np.asarray(hrir, dtype=np.float64)
    if samples.size < 4 or not np.any(samples):
        return 0.0

    reference = minimum_phase(samples)
    length = int(2 ** np.ceil(np.log2(2 * samples.size)))
    correlation = np.fft.irfft(
        np.fft.rfft(samples, length) * np.conj(np.fft.rfft(reference, length)), length
    )
    peak = int(np.argmax(correlation))
    if peak > length // 2:
        peak -= length

    # Parabolic refinement around the peak gives a fractional estimate.
    previous = correlation[(peak - 1) % length]
    current = correlation[peak % length]
    following = correlation[(peak + 1) % length]
    denominator = previous - 2.0 * current + following
    fraction = 0.0 if abs(denominator) < _EPSILON else 0.5 * (previous - following) / denominator
    return float(peak + np.clip(fraction, -0.5, 0.5))


def log_magnitude(hrir: NDArray[np.floating], bins: int | None = None) -> NDArray[np.float64]:
    """Natural-log magnitude spectrum, the quantity that interpolates smoothly."""

    samples = np.asarray(hrir, dtype=np.float64)
    length = bins or samples.size
    return np.log(np.maximum(np.abs(np.fft.rfft(samples, length)), _EPSILON))


def reconstruct_from_log_magnitude(
    values: NDArray[np.floating], taps: int, *, delay_samples: float = 0.0
) -> NDArray[np.float64]:
    """Rebuild a causal response from a log-magnitude spectrum and a delay.

    The magnitude fixes the timbre, the minimum-phase reconstruction supplies a
    causal phase, and the delay is re-applied fractionally on top, which is
    what preserves the interaural timing the decomposition took out.
    """

    causal = minimum_phase_from_log_magnitude(values, taps)
    if abs(delay_samples) < 1e-9:
        return causal
    return shift_response(causal, float(delay_samples), taps)


def shift_response(
    hrir: NDArray[np.floating], delay_samples: float, taps: int | None = None
) -> NDArray[np.float64]:
    """Shift a finite response by a fractional delay, padding with silence.

    Reads outside the response return zero rather than holding its edge: a
    clamped read would smear the first sample into a plateau and destroy the
    response it was meant to move.
    """

    samples = np.asarray(hrir, dtype=np.float64)
    length = int(taps or samples.size)
    headroom = int(np.ceil(abs(delay_samples))) + 4
    padded = np.concatenate((samples, np.zeros(headroom)))
    shifted = read_fractional(
        padded, np.full(padded.size, float(delay_samples)), outside="zero"
    )
    return shifted[:length]


@dataclass(frozen=True)
class AlignedHrir:
    """A response with its broadband delay removed and recorded separately."""

    aligned: NDArray[np.float64]
    delay_samples: float

    def restore(self, taps: int | None = None) -> NDArray[np.float64]:
        """Put the delay back, fractionally."""

        return shift_response(self.aligned, self.delay_samples, taps)


def align_pair(hrir_pair: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove each ear's broadband delay from a ``(2, taps)`` pair.

    Returns ``(aligned, delays)`` where ``delays`` is ``(2,)`` in samples. The
    ear responses keep their own delays rather than a shared one, because the
    difference between them *is* the interaural time difference.
    """

    pair = np.asarray(hrir_pair, dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != 2:
        raise ValueError(f"expected a (2, taps) HRIR pair, got shape {pair.shape}")

    delays = np.array([broadband_delay_samples(pair[ear]) for ear in range(2)])
    taps = pair.shape[1]
    aligned = np.vstack([shift_response(pair[ear], -delays[ear], taps) for ear in range(2)])
    return aligned, delays


def common_differential_delay(delays: NDArray[np.floating]) -> tuple[float, float]:
    """Split a ``(2,)`` delay pair into common and differential parts.

    ``D_c = (D_L + D_R) / 2`` and ``D_d = (D_L - D_R) / 2``: the common delay
    is the distance to the head and the differential delay is the ITD, which
    is the pair the cue controls scale independently.
    """

    values = np.asarray(delays, dtype=np.float64).ravel()
    return float(np.mean(values[:2])), float(0.5 * (values[0] - values[1]))


def common_differential_log_magnitude(
    log_magnitudes: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split ``(2, bins)`` log magnitudes into common and differential parts.

    ``M_c`` carries the timbre both ears share and ``M_d`` the interaural level
    difference, so ILD can be scaled without touching the shared response.
    """

    values = np.asarray(log_magnitudes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"expected (2, bins) log magnitudes, got shape {values.shape}")
    return 0.5 * (values[0] + values[1]), 0.5 * (values[0] - values[1])

"""The curves the HRTF Lab plots, computed without any Qt involvement.

Each function takes impulse responses and returns plain arrays, so the same
numbers can be asserted in a test, written to a report, or drawn on a widget
without any of those paths disagreeing about what an ITD is.

Interaural conventions follow the rest of the package: a positive ITD means the
left ear leads, and a positive ILD means the left ear is louder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT

__all__ = [
    "group_delay_samples",
    "ild_db",
    "impulse_response_curves",
    "itd_seconds",
    "magnitude_db",
    "sweep_across_azimuth",
    "unwrapped_phase_deg",
]

_EPSILON = 1e-12


def _as_pair(hrir_pair: NDArray[np.floating]) -> NDArray[np.float64]:
    pair = np.asarray(hrir_pair, dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected a ({CHANNEL_COUNT}, taps) HRIR pair, got {pair.shape}")
    return pair


def impulse_response_curves(
    hrir_pair: NDArray[np.floating], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Time in milliseconds, and each ear's response."""

    pair = _as_pair(hrir_pair)
    times_ms = np.arange(pair.shape[1]) / float(sample_rate_hz) * 1e3
    return times_ms, pair[0], pair[1]


def magnitude_db(
    hrir_pair: NDArray[np.floating], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Frequency in hertz, and each ear's magnitude in decibels."""

    pair = _as_pair(hrir_pair)
    taps = pair.shape[1]
    frequencies = np.fft.rfftfreq(taps, 1.0 / float(sample_rate_hz))
    spectra = np.fft.rfft(pair, axis=1)
    decibels = 20.0 * np.log10(np.maximum(np.abs(spectra), _EPSILON))
    return frequencies, decibels[0], decibels[1]


def unwrapped_phase_deg(
    hrir_pair: NDArray[np.floating], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Frequency, and each ear's unwrapped phase in degrees.

    Unwrapped because a wrapped phase plot is a picture of the wrapping, not of
    the response.
    """

    pair = _as_pair(hrir_pair)
    taps = pair.shape[1]
    frequencies = np.fft.rfftfreq(taps, 1.0 / float(sample_rate_hz))
    spectra = np.fft.rfft(pair, axis=1)
    phase = np.degrees(np.unwrap(np.angle(spectra), axis=1))
    return frequencies, phase[0], phase[1]


def group_delay_samples(
    hrir_pair: NDArray[np.floating], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Frequency, and each ear's group delay in samples.

    Group delay is the derivative of phase with respect to angular frequency,
    computed here from the unwrapped phase by finite difference.
    """

    frequencies, left, right = unwrapped_phase_deg(hrir_pair, sample_rate_hz)
    if frequencies.size < 2:
        zeros = np.zeros_like(frequencies)
        return frequencies, zeros, zeros

    step = frequencies[1] - frequencies[0]
    scale = -1.0 / (360.0 * step) * float(sample_rate_hz)

    def derivative(phase: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.gradient(phase) * scale

    return frequencies, derivative(left), derivative(right)


def itd_seconds(hrir_pair: NDArray[np.floating], sample_rate_hz: float) -> float:
    """Interaural time difference, positive when the left ear leads.

    Estimated by cross-correlation, which is robust to the two ears having
    different spectra - which they always do off the median plane.
    """

    pair = _as_pair(hrir_pair)
    correlation = np.correlate(pair[0], pair[1], mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (pair.shape[1] - 1)
    # A positive lag means the left channel had to be delayed to line up with
    # the right, so the left arrived first.
    return -float(lag) / float(sample_rate_hz)


def ild_db(hrir_pair: NDArray[np.floating]) -> float:
    """Interaural level difference in decibels, positive when the left is louder."""

    pair = _as_pair(hrir_pair)
    power = np.sqrt(np.mean(np.square(pair), axis=1))
    return float(20.0 * np.log10(max(power[0], _EPSILON) / max(power[1], _EPSILON)))


def sweep_across_azimuth(
    hrirs: NDArray[np.floating],
    positions_m: NDArray[np.floating],
    sample_rate_hz: float,
    *,
    elevation_deg: float = 0.0,
    tolerance_deg: float = 7.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ITD and ILD around one elevation, ordered by azimuth.

    Returns ``(azimuths_deg, itd_microseconds, ild_db)``. Measurements further
    than ``tolerance_deg`` from the requested elevation are left out rather than
    quietly folded in, so a sparse grid produces a short curve instead of a
    misleading one.
    """

    from ..hrtf.coordinates import cartesian_to_sofa_spherical

    responses = np.asarray(hrirs, dtype=np.float64)
    spherical = cartesian_to_sofa_spherical(np.asarray(positions_m, dtype=np.float64))
    near = np.abs(spherical[:, 1] - float(elevation_deg)) <= float(tolerance_deg)
    indices = np.flatnonzero(near)
    if indices.size == 0:
        empty = np.zeros(0)
        return empty, empty.copy(), empty.copy()

    azimuths = ((spherical[indices, 0] + 180.0) % 360.0) - 180.0
    order = np.argsort(azimuths)
    indices = indices[order]

    itd = np.array(
        [itd_seconds(responses[index], sample_rate_hz) * 1e6 for index in indices]
    )
    ild = np.array([ild_db(responses[index]) for index in indices])
    return azimuths[order], itd, ild

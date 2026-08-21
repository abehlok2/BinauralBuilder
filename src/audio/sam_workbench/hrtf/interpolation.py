"""Direction selection and delay-aligned log-magnitude interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _unit(points: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("HRTF directions must have non-zero radius")
    return values / norms


def nearest_indices(dataset_positions_m: ArrayLike, query_positions_m: ArrayLike) -> NDArray[np.int64]:
    reference = _unit(dataset_positions_m)
    query = _unit(query_positions_m)
    return np.argmax(query @ reference.T, axis=-1).astype(np.int64)


def nearest_weights(dataset_positions_m: ArrayLike, query_position_m: ArrayLike, count: int = 3):
    reference = _unit(dataset_positions_m)
    query = _unit(np.asarray(query_position_m, dtype=np.float64).reshape(1, 3))[0]
    angular = np.arccos(np.clip(reference @ query, -1.0, 1.0))
    count = max(1, min(int(count), len(reference)))
    indices = np.argpartition(angular, count - 1)[:count]
    distances = angular[indices]
    if distances.min(initial=np.inf) < 1e-10:
        winner = indices[np.argmin(distances)]
        return np.array([winner]), np.array([1.0])
    weights = 1.0 / np.maximum(distances, 1e-9)
    return indices, weights / np.sum(weights)


def _onset_delay(ir: NDArray[np.float64]) -> float:
    energy = np.square(ir)
    total = np.sum(energy)
    if total <= 1e-20:
        return 0.0
    cumulative = np.cumsum(energy) / total
    return float(np.searchsorted(cumulative, 0.01))


def _minimum_phase_from_magnitude(magnitude: NDArray[np.float64], fft_size: int) -> NDArray[np.float64]:
    log_mag = np.log(np.maximum(magnitude, 1e-12))
    cepstrum = np.fft.irfft(log_mag, n=fft_size)
    minimum_cepstrum = np.zeros_like(cepstrum)
    minimum_cepstrum[0] = cepstrum[0]
    half = fft_size // 2
    minimum_cepstrum[1:half] = 2.0 * cepstrum[1:half]
    if fft_size % 2 == 0:
        minimum_cepstrum[half] = cepstrum[half]
    spectrum = np.exp(np.fft.rfft(minimum_cepstrum))
    return np.fft.irfft(spectrum, n=fft_size)


def interpolate_log_magnitude_delay(dataset, query_position_m: ArrayLike, neighbours: int = 3) -> NDArray[np.float64]:
    """Interpolate aligned log magnitude and broadband delay, never raw phase."""

    indices, weights = nearest_weights(dataset.positions_m, query_position_m, neighbours)
    if len(indices) == 1 and weights[0] == 1.0:
        return np.array(dataset.ir[indices[0]], dtype=np.float64, copy=True)
    taps = dataset.taps
    fft_size = 1 << int(np.ceil(np.log2(max(8, taps * 2))))
    result = np.zeros((2, taps), dtype=np.float64)
    for ear in range(2):
        responses = dataset.ir[indices, ear]
        delays = np.array([_onset_delay(response) for response in responses]) + dataset.delay_samples[indices, ear]
        spectra = np.fft.rfft(responses, n=fft_size, axis=-1)
        log_magnitude = np.sum(weights[:, None] * np.log(np.maximum(np.abs(spectra), 1e-12)), axis=0)
        minimum_phase = _minimum_phase_from_magnitude(np.exp(log_magnitude), fft_size)
        delay = float(weights @ delays)
        frequency = np.fft.rfftfreq(fft_size)
        shifted = np.fft.irfft(np.fft.rfft(minimum_phase) * np.exp(-2j * np.pi * frequency * delay), n=fft_size)
        result[ear] = shifted[:taps]
    return result

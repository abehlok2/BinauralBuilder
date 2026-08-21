"""HRTF interpolation, in the order it was built.

1. ``nearest`` - the closest measurement, unblended. Combined with the
   renderer's crossfade this is smooth enough for moving sources and cannot
   smear a transient, because nothing is ever averaged.
2. ``three_neighbor`` - the three closest, weighted by unit-sphere distance,
   blended **after** delay alignment.
3. ``spherical_triangular`` - the spherical triangle containing the direction,
   with barycentric weights, again after alignment.
4. ``delay_magnitude`` - blend log magnitude and broadband delay separately,
   then reconstruct a minimum-phase response and re-apply the delay.
5. ``spherical_harmonic`` - fit the aligned representation over the whole
   sphere and evaluate the fit at the query direction.

Modes 2 and above all align first. Averaging raw HRIR samples across
directions comb-filters the result: the same wavefront arrives at different
times in each measurement, so their sum partially cancels. Alignment removes
the direction-dependent delay, blends what is left, and puts a blended delay
back - which moves the source instead of blurring it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT
from .dataset import PreparedHrtf
from .decomposition import (
    align_pair,
    minimum_phase_from_log_magnitude,
    shift_response,
)
from .selection import DirectionIndex, DirectionWeights

__all__ = [
    "INTERPOLATION_MODES",
    "NEAREST",
    "THREE_NEIGHBOR",
    "SPHERICAL_TRIANGULAR",
    "DELAY_MAGNITUDE",
    "SPHERICAL_HARMONIC",
    "HrirSelection",
    "HrtfInterpolator",
    "real_spherical_harmonics",
]

NEAREST = "nearest"
THREE_NEIGHBOR = "three_neighbor"
SPHERICAL_TRIANGULAR = "spherical_triangular"
DELAY_MAGNITUDE = "delay_magnitude"
SPHERICAL_HARMONIC = "spherical_harmonic"

#: In the order the specification asks for them to be implemented and adopted.
INTERPOLATION_MODES = (
    NEAREST,
    THREE_NEIGHBOR,
    SPHERICAL_TRIANGULAR,
    DELAY_MAGNITUDE,
    SPHERICAL_HARMONIC,
)

_EPSILON = 1e-12


@dataclass(frozen=True)
class HrirSelection:
    """The filter pair to use for one direction, and how it was arrived at."""

    hrirs: NDArray[np.float64]  # (2, taps)
    delays_samples: NDArray[np.float64]  # (2,), external delay when preserved
    mode: str
    #: Measurement indices that contributed, for the Inspector.
    indices: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    distance_deg: float = 0.0
    extrapolated: bool = False

    @property
    def taps(self) -> int:
        return int(self.hrirs.shape[1])


def real_spherical_harmonics(
    order: int, azimuth_deg: NDArray[np.floating], elevation_deg: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Real spherical harmonics up to ``order``, as ``(directions, (order+1)^2)``.

    Uses the canonical angles directly: azimuth measured from the front toward
    the left, elevation from the horizon upward, converted here to the
    colatitude the associated Legendre functions expect.
    """

    from scipy.special import sph_harm_y

    azimuth = np.radians(np.atleast_1d(np.asarray(azimuth_deg, dtype=np.float64)))
    colatitude = np.radians(90.0 - np.atleast_1d(np.asarray(elevation_deg, dtype=np.float64)))

    columns = []
    for degree in range(int(order) + 1):
        for index in range(-degree, degree + 1):
            harmonic = sph_harm_y(degree, abs(index), colatitude, azimuth)
            if index < 0:
                columns.append(np.sqrt(2.0) * (-1.0) ** index * harmonic.imag)
            elif index == 0:
                columns.append(harmonic.real)
            else:
                columns.append(np.sqrt(2.0) * (-1.0) ** index * harmonic.real)
    return np.column_stack(columns)


class HrtfInterpolator:
    """Chooses and blends the filter pair for a direction.

    The aligned representation is computed once, lazily, and only by the modes
    that need it: selecting the nearest measurement never pays for alignment.
    """

    def __init__(
        self,
        dataset: PreparedHrtf,
        *,
        mode: str = NEAREST,
        neighbor_count: int = 3,
        harmonic_order: int = 4,
    ) -> None:
        if mode not in INTERPOLATION_MODES:
            raise ValueError(f"unknown interpolation mode {mode!r}; expected one of {INTERPOLATION_MODES}")
        self.dataset = dataset
        self.mode = mode
        self.neighbor_count = int(neighbor_count)
        self.harmonic_order = int(harmonic_order)
        self.index = DirectionIndex(dataset.directions)

    # --- lazily prepared representations ------------------------------------

    @cached_property
    def _aligned(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Every measurement with its broadband delay removed and recorded."""

        measurements = self.dataset.measurements
        aligned = np.empty_like(self.dataset.hrirs)
        delays = np.empty((measurements, CHANNEL_COUNT), dtype=np.float64)
        for measurement in range(measurements):
            aligned[measurement], delays[measurement] = align_pair(self.dataset.hrirs[measurement])
        return aligned, delays

    @cached_property
    def _log_magnitudes(self) -> NDArray[np.float64]:
        """Log magnitude of every aligned response, ``(measurements, 2, bins)``."""

        aligned, _ = self._aligned
        spectra = np.fft.rfft(aligned, axis=2)
        return np.log(np.maximum(np.abs(spectra), _EPSILON))

    @cached_property
    def _harmonic_fit(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Least-squares spherical-harmonic fit of log magnitude and delay."""

        basis = real_spherical_harmonics(
            self.harmonic_order, self.dataset.azimuths_deg, self.dataset.elevations_deg
        )
        _, delays = self._aligned
        magnitudes = self._log_magnitudes
        measurements, ears, bins = magnitudes.shape

        magnitude_coefficients, *_ = np.linalg.lstsq(
            basis, magnitudes.reshape(measurements, ears * bins), rcond=None
        )
        delay_coefficients, *_ = np.linalg.lstsq(basis, delays, rcond=None)
        return (
            magnitude_coefficients.reshape(-1, ears, bins),
            delay_coefficients,
        )

    # --- selection ----------------------------------------------------------

    def at(self, direction: NDArray[np.floating]) -> HrirSelection:
        """The filter pair for one direction, under the configured mode."""

        if self.mode == NEAREST:
            return self._nearest(direction)
        if self.mode == THREE_NEIGHBOR:
            return self._blend(self.index.nearest_k(direction, self.neighbor_count), THREE_NEIGHBOR)
        if self.mode == SPHERICAL_TRIANGULAR:
            return self._blend(self.index.barycentric(direction), SPHERICAL_TRIANGULAR)
        if self.mode == DELAY_MAGNITUDE:
            return self._delay_magnitude(direction)
        return self._spherical_harmonic(direction)

    def _external_delay(self, weights: DirectionWeights) -> NDArray[np.float64]:
        """Blend the dataset's own external delays, when the policy kept them."""

        if not self.dataset.has_external_delay:
            return np.zeros(CHANNEL_COUNT, dtype=np.float64)
        return weights.weights @ self.dataset.delays_samples[weights.indices]

    def _nearest(self, direction: NDArray[np.floating]) -> HrirSelection:
        weights = self.index.nearest(direction)
        measurement = int(weights.indices[0])
        return HrirSelection(
            hrirs=np.array(self.dataset.hrirs[measurement], dtype=np.float64, copy=True),
            delays_samples=np.array(self.dataset.delays_samples[measurement], dtype=np.float64),
            mode=NEAREST,
            indices=weights.indices,
            weights=weights.weights,
            distance_deg=weights.distance_deg,
        )

    def _blend(self, weights: DirectionWeights, mode: str) -> HrirSelection:
        """Blend aligned responses, then put a blended delay back."""

        aligned, delays = self._aligned
        taps = self.dataset.taps
        chosen = aligned[weights.indices]  # (k, 2, taps)
        blended = np.tensordot(weights.weights, chosen, axes=(0, 0))  # (2, taps)
        blended_delays = weights.weights @ delays[weights.indices]  # (2,)

        restored = np.vstack(
            [shift_response(blended[ear], float(blended_delays[ear]), taps) for ear in range(CHANNEL_COUNT)]
        )
        return HrirSelection(
            hrirs=restored,
            delays_samples=self._external_delay(weights),
            mode=mode,
            indices=weights.indices,
            weights=weights.weights,
            distance_deg=weights.distance_deg,
            extrapolated=weights.extrapolated,
        )

    def _delay_magnitude(self, direction: NDArray[np.floating]) -> HrirSelection:
        """Blend log magnitude and delay separately, then rebuild the filter."""

        weights = (
            self.index.barycentric(direction)
            if self.index.supports_triangulation
            else self.index.nearest_k(direction, self.neighbor_count)
        )
        _, delays = self._aligned
        magnitudes = self._log_magnitudes[weights.indices]  # (k, 2, bins)
        blended_magnitude = np.tensordot(weights.weights, magnitudes, axes=(0, 0))
        blended_delays = weights.weights @ delays[weights.indices]

        taps = self.dataset.taps
        rebuilt = np.vstack(
            [
                shift_response(
                    minimum_phase_from_log_magnitude(blended_magnitude[ear], taps),
                    float(blended_delays[ear]),
                    taps,
                )
                for ear in range(CHANNEL_COUNT)
            ]
        )
        return HrirSelection(
            hrirs=rebuilt,
            delays_samples=self._external_delay(weights),
            mode=DELAY_MAGNITUDE,
            indices=weights.indices,
            weights=weights.weights,
            distance_deg=weights.distance_deg,
            extrapolated=weights.extrapolated,
        )

    def _spherical_harmonic(self, direction: NDArray[np.floating]) -> HrirSelection:
        """Evaluate a spherical-harmonic fit of the aligned representation."""

        from ..conventions import cartesian_to_spherical_deg

        azimuth, elevation, _ = cartesian_to_spherical_deg(
            np.asarray(direction, dtype=np.float64).reshape(3)
        )
        basis = real_spherical_harmonics(self.harmonic_order, [azimuth], [elevation])[0]
        magnitude_coefficients, delay_coefficients = self._harmonic_fit

        magnitude = np.tensordot(basis, magnitude_coefficients, axes=(0, 0))  # (2, bins)
        delays = basis @ delay_coefficients  # (2,)

        taps = self.dataset.taps
        rebuilt = np.vstack(
            [
                shift_response(
                    minimum_phase_from_log_magnitude(magnitude[ear], taps), float(delays[ear]), taps
                )
                for ear in range(CHANNEL_COUNT)
            ]
        )
        nearest = self.index.nearest(direction)
        return HrirSelection(
            hrirs=rebuilt,
            delays_samples=np.zeros(CHANNEL_COUNT),
            mode=SPHERICAL_HARMONIC,
            indices=nearest.indices,
            weights=np.array([1.0]),
            distance_deg=nearest.distance_deg,
            # A harmonic fit is defined everywhere, but outside the measured
            # coverage it is a projection rather than a measurement.
            extrapolated=nearest.distance_deg > 2.0 * self.index.median_spacing_deg(),
        )

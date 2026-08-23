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


# ---------------------------------------------------------------------------
# Direction-indexed interpolation modes
# ---------------------------------------------------------------------------
#
# Implemented in the order they are meant to be adopted:
#
# 1. ``nearest`` - the closest measurement, unblended. Combined with the
#    renderer's crossfade this is smooth enough for moving sources and cannot
#    smear a transient, because nothing is ever averaged.
# 2. ``three_neighbor`` - the three closest, weighted by unit-sphere distance,
#    blended **after** delay alignment.
# 3. ``spherical_triangular`` - the spherical triangle containing the
#    direction, with barycentric weights, again after alignment.
# 4. ``delay_magnitude`` - blend log magnitude and broadband delay separately,
#    then reconstruct a minimum-phase response and re-apply the delay.
# 5. ``spherical_harmonic`` - fit the aligned representation over the whole
#    sphere and evaluate the fit at the query direction.
#
# Modes 2 and above all align first. Averaging raw HRIR samples across
# directions comb-filters the result: the same wavefront arrives at different
# times in each measurement, so their sum partially cancels. Alignment removes
# the direction-dependent delay, blends what is left, and puts a blended delay
# back - which moves the source instead of blurring it.

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property

from ..conventions import CHANNEL_COUNT
from .coordinates import cartesian_to_sofa_spherical, unit_vectors
from .decomposition import align_pair, minimum_phase_from_log_magnitude, shift_response
from .selection import DirectionIndex, DirectionWeights

NEAREST = "nearest"
THREE_NEIGHBOR = "three_neighbor"
SPHERICAL_TRIANGULAR = "spherical_triangular"
DELAY_MAGNITUDE = "delay_magnitude"
SPHERICAL_HARMONIC = "spherical_harmonic"

#: In the order the specification asks for them to be implemented and adopted.
#: Strength of the ridge term on the spherical-harmonic fit, relative to the
#: mean diagonal of the normal equations. Small enough not to bias the fit,
#: large enough to keep it bounded where the measurement grid thins out.
HARMONIC_REGULARIZATION = 1e-6

INTERPOLATION_MODES = (
    NEAREST,
    THREE_NEIGHBOR,
    SPHERICAL_TRIANGULAR,
    DELAY_MAGNITUDE,
    SPHERICAL_HARMONIC,
)

_EPSILON = 1e-12

#: Interpolated filters, shared across every interpolator in the process.
#:
#: Per-instance caching turned out to buy nothing: the adaptive control
#: interval already declines to reselect until the direction has moved past its
#: tolerance, so within one render consecutive lookups are genuinely different
#: directions. What repeats is *renders* - a preview and then an export of the
#: same voice, the two halves of an A/B, several sources on one trajectory -
#: and each of those built a new interpolator and threw the work away. The
#: cache therefore lives here, keyed by what the result depends on, so it
#: survives the object that filled it.
_FILTER_CACHE: "OrderedDict[tuple, HrirSelection]" = OrderedDict()


#: Whole-process totals, alongside the per-interpolator ones. A render builds
#: an interpolator per source and discards it, so per-instance counts cannot
#: answer "did this export reuse anything?" - which is the question an export
#: report is asking.
_CACHE_HITS = 0
_CACHE_MISSES = 0


def cache_statistics() -> dict:
    """Hits, misses and size for the process, not for one interpolator."""

    total = _CACHE_HITS + _CACHE_MISSES
    return {
        "hits": int(_CACHE_HITS),
        "misses": int(_CACHE_MISSES),
        "size": len(_FILTER_CACHE),
        "hitRate": float(_CACHE_HITS / total) if total else 0.0,
    }


def _note_cache(*, hit: bool) -> None:
    global _CACHE_HITS, _CACHE_MISSES

    if hit:
        _CACHE_HITS += 1
    else:
        _CACHE_MISSES += 1


def clear_filter_cache() -> None:
    """Forget every interpolated filter. For tests and for memory pressure."""

    global _CACHE_HITS, _CACHE_MISSES

    _FILTER_CACHE.clear()
    _CACHE_HITS = _CACHE_MISSES = 0


def _dataset_fingerprint(dataset) -> tuple:
    """What distinguishes one set of measurements from another.

    The SOFA file's content hash is not enough on its own. A cue-modified
    dataset is a *view* of the file it came from and forwards that hash, so
    keying on it alone would serve the modified render the unmodified filters -
    silently, and only in hybrid mode, which is the worst place for it. The
    impulse responses themselves are therefore hashed, which is the only thing
    that cannot be wrong.

    Done once when an interpolator is built, not per lookup: a few milliseconds
    against a render, and it is what makes every other key component safe.
    """

    responses = np.ascontiguousarray(
        np.asarray(getattr(dataset, "ir", ()), dtype=np.float64)
    )
    digest = hashlib.sha256(responses.tobytes()).hexdigest()
    return (
        digest,
        responses.shape,
        float(getattr(dataset, "sample_rate_hz", 0.0)),
        str(
            getattr(
                getattr(dataset, "delay_policy", ""), "value",
                getattr(dataset, "delay_policy", ""),
            )
        ),
    )


@dataclass(frozen=True)
class HrirSelection:
    """The filter pair to use for one direction, and how it was arrived at."""

    hrirs: NDArray[np.float64]  # (2, taps)
    delays_samples: NDArray[np.float64]  # (2,), external delay when preserved
    mode: str
    #: Measurement indices that contributed, for the inspector.
    indices: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    distance_deg: float = 0.0
    extrapolated: bool = False

    @property
    def taps(self) -> int:
        return int(self.hrirs.shape[1])


def real_spherical_harmonics(
    order: int, azimuth_deg: ArrayLike, elevation_deg: ArrayLike
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

    Operates on a loaded :class:`~.sofa_io.HRTFDataset`. The aligned
    representation is computed once, lazily, and only by the modes that need
    it: selecting the nearest measurement never pays for alignment.
    """

    def __init__(
        self,
        dataset,
        *,
        mode: str = NEAREST,
        neighbor_count: int = 3,
        harmonic_order: int | None = None,
    ) -> None:
        if mode not in INTERPOLATION_MODES:
            raise ValueError(
                f"unknown interpolation mode {mode!r}; expected one of {INTERPOLATION_MODES}"
            )
        self.dataset = dataset
        self.mode = mode
        self.neighbor_count = int(neighbor_count)
        self.index = DirectionIndex(unit_vectors(dataset.positions_m))

        #: What the caller asked for, before the dataset had its say.
        self.requested_harmonic_order = (
            None if harmonic_order is None else int(harmonic_order)
        )
        self.harmonic_order = self._resolve_harmonic_order(self.requested_harmonic_order)

        # What this interpolator's results depend on, beyond the direction
        # asked for. It is the cache's identity, so two interpolators built the
        # same way over the same measurements share their work and two built
        # differently cannot see each other's.
        self._identity = (
            _dataset_fingerprint(dataset),
            self.mode,
            self.neighbor_count,
            self.harmonic_order,
        )
        self._cache_hits = 0
        self._cache_misses = 0

    # --- the spherical-harmonic order ---------------------------------------

    @property
    def max_supportable_harmonic_order(self) -> int:
        """The highest order this many measurements can actually determine.

        An order ``N`` fit has ``(N + 1)**2`` coefficients. Asking for more of
        them than there are measurements leaves the fit underdetermined: least
        squares still returns an answer, but it is the minimum-norm one, which
        is a choice nothing in the data made. So the order is capped here, and
        the cap is a property rather than a hidden clamp so the interface can
        say what it did.
        """

        return max(0, int(np.floor(np.sqrt(self.index.measurements))) - 1)

    def _resolve_harmonic_order(self, requested: int | None) -> int:
        """Pick the order to fit at, from the request and the measurements.

        Unset means "as high as the data supports", which is the useful default:
        the fixed low order this used to carry could not reproduce a measured
        response at its own direction on any real dataset - a 25-coefficient
        fit through 144 measurements was half the peak out.
        """

        supportable = self.max_supportable_harmonic_order
        if requested is None:
            return supportable
        if requested < 0:
            raise ValueError(f"harmonic_order must not be negative, got {requested}")
        return min(int(requested), supportable)

    @property
    def harmonic_order_was_reduced(self) -> bool:
        """True when the requested order was more than the data supports."""

        return (
            self.requested_harmonic_order is not None
            and self.requested_harmonic_order > self.harmonic_order
        )

    # --- lazily prepared representations ------------------------------------

    @cached_property
    def _spherical_deg(self) -> NDArray[np.float64]:
        return cartesian_to_sofa_spherical(self.dataset.positions_m)

    @cached_property
    def _aligned(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Every measurement with its broadband delay removed and recorded."""

        measurements = self.dataset.measurements
        aligned = np.empty_like(np.asarray(self.dataset.ir, dtype=np.float64))
        delays = np.empty((measurements, CHANNEL_COUNT), dtype=np.float64)
        for measurement in range(measurements):
            aligned[measurement], delays[measurement] = align_pair(self.dataset.ir[measurement])
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

        spherical = self._spherical_deg
        basis = real_spherical_harmonics(self.harmonic_order, spherical[:, 0], spherical[:, 1])
        _, delays = self._aligned
        magnitudes = self._log_magnitudes
        measurements, ears, bins = magnitudes.shape

        # Tikhonov regularization, at a strength scaled to the basis itself.
        # Without it a fit at the highest supportable order is well behaved at
        # the measurements and free to do anything between them, because the
        # directions are never distributed evenly enough to constrain every
        # coefficient equally. This costs a little accuracy at the nodes and
        # buys a fit that stays bounded where the grid is sparse.
        scale = HARMONIC_REGULARIZATION * float(np.trace(basis.T @ basis)) / basis.shape[1]
        normal = basis.T @ basis + scale * np.eye(basis.shape[1])
        magnitude_coefficients = np.linalg.solve(
            normal, basis.T @ magnitudes.reshape(measurements, ears * bins)
        )
        delay_coefficients = np.linalg.solve(normal, basis.T @ delays)
        return magnitude_coefficients.reshape(-1, ears, bins), delay_coefficients

    # --- selection ----------------------------------------------------------

    #: Direction quantization for the cache, in units of the unit vector. At
    #: 1e-3 the worst error between a direction and its cache key is well under
    #: a tenth of a degree - far below the angular tolerance a renderer works
    #: to, and far below anything audible.
    CACHE_RESOLUTION = 1e-3

    #: Bounded, so a long render with a wandering path cannot accumulate
    #: filters without limit.
    CACHE_LIMIT = 4096

    def cache_statistics(self) -> dict:
        """Hits, misses and size, for the inspector and for benchmarks."""

        total = self._cache_hits + self._cache_misses
        return {
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "size": len(_FILTER_CACHE),
            "hitRate": float(self._cache_hits / total) if total else 0.0,
        }

    def _cache_key(self, direction: ArrayLike):
        values = np.asarray(direction, dtype=np.float64).reshape(-1)
        if values.size != 3 or not np.all(np.isfinite(values)):
            return None
        norm = float(np.linalg.norm(values))
        if norm <= 0.0:
            return None
        unit = values / norm
        return tuple(np.round(unit / self.CACHE_RESOLUTION).astype(np.int64).tolist())

    def at(self, direction: ArrayLike) -> HrirSelection:
        """The filter pair for one direction, under the configured mode.

        Cached on a quantized direction. Everything else the result depends on
        - the dataset, the mode, the neighbour count, the harmonic order - is
        fixed for the life of this object, so it is part of the cache's
        identity rather than part of each key.
        """

        quantized = self._cache_key(direction)
        key = None if quantized is None else (self._identity, quantized)
        if key is not None:
            cached = _FILTER_CACHE.get(key)
            if cached is not None:
                _FILTER_CACHE.move_to_end(key)
                self._cache_hits += 1
                _note_cache(hit=True)
                return cached
            self._cache_misses += 1
            _note_cache(hit=False)

        selection = self._select(direction)
        if key is not None and self.CACHE_LIMIT > 0:
            _FILTER_CACHE[key] = selection
            while len(_FILTER_CACHE) > self.CACHE_LIMIT:
                _FILTER_CACHE.popitem(last=False)
        return selection

    def _select(self, direction: ArrayLike) -> HrirSelection:
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
        return weights.weights @ self.dataset.delay_samples[weights.indices]

    def _nearest(self, direction: ArrayLike) -> HrirSelection:
        weights = self.index.nearest(direction)
        measurement = int(weights.indices[0])
        return HrirSelection(
            hrirs=np.array(self.dataset.ir[measurement], dtype=np.float64, copy=True),
            delays_samples=np.array(self.dataset.delay_samples[measurement], dtype=np.float64),
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
            [
                shift_response(blended[ear], float(blended_delays[ear]), taps)
                for ear in range(CHANNEL_COUNT)
            ]
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

    def _delay_magnitude(self, direction: ArrayLike) -> HrirSelection:
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

    def _spherical_harmonic(self, direction: ArrayLike) -> HrirSelection:
        """Evaluate a spherical-harmonic fit of the aligned representation."""

        azimuth, elevation, _ = cartesian_to_sofa_spherical(
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

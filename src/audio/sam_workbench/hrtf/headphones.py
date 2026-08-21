"""Headphone correction, applied after binaural rendering.

The chain is fixed and the order matters:

.. code-block:: text

    mono source -> trajectory and HRTF selection -> left/right convolution
      -> headphone correction -> limiter and output gain -> stereo out

Correction belongs *after* the binaural stage because it compensates the
transducer, not the head: applying it earlier would fold the headphone's
response into the spatial cues.

Four modes, and one rule that governs all of them: a correction is only ever
applied when the user selects it. SONICOM ships an HD650 measurement, and it
is right only for an HD650. Applying it to different headphones would impose
one transducer's errors on another, so ``OFF`` is the default and the HD650
option names the model it is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from ..conventions import CHANNEL_COUNT

__all__ = [
    "CORRECTION_MODES",
    "OFF",
    "SONICOM_HD650",
    "USER_IMPULSE_RESPONSE",
    "USER_EQUALIZATION_CURVE",
    "HeadphoneCorrection",
    "apply_correction",
    "inverse_filter_from_response",
]

OFF = "off"
SONICOM_HD650 = "sonicom_hd650"
USER_IMPULSE_RESPONSE = "user_impulse_response"
USER_EQUALIZATION_CURVE = "user_equalization_curve"

CORRECTION_MODES = (OFF, SONICOM_HD650, USER_IMPULSE_RESPONSE, USER_EQUALIZATION_CURVE)

#: Regularization for inverting a measured response. Without it, a deep notch
#: in the measurement becomes an enormous peak in its inverse.
DEFAULT_REGULARIZATION_DB = -20.0
#: Correction is only meaningful inside the band the headphone reproduces.
DEFAULT_BAND_HZ = (20.0, 20_000.0)


def _inverse_magnitude(
    response: NDArray[np.floating],
    sample_rate_hz: float,
    *,
    taps: int,
    regularization_db: float,
    band_hz: tuple[float, float],
) -> NDArray[np.float64] | None:
    """The regularized, band-limited inverse magnitude, before any scaling.

    Returns ``None`` for an empty measurement, which has no inverse.
    """

    measured = np.asarray(response, dtype=np.float64).reshape(-1)
    if measured.size == 0:
        return None

    length = int(2 ** np.ceil(np.log2(max(measured.size, taps) * 2)))
    spectrum = np.fft.rfft(measured, length)
    magnitude = np.abs(spectrum)

    ceiling = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    floor = ceiling * (10.0 ** (regularization_db / 20.0))
    return 1.0 / np.maximum(magnitude, floor)


def inverse_filter_from_response(
    response: NDArray[np.floating],
    sample_rate_hz: float,
    *,
    taps: int = 512,
    regularization_db: float = DEFAULT_REGULARIZATION_DB,
    band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
    normalization: float | None = None,
) -> NDArray[np.float64]:
    """Build a correction filter that flattens a measured headphone response.

    The inverse is regularized and band-limited: a headphone measurement has
    deep, narrow notches whose exact inverse would be a huge, ringing boost of
    something the transducer cannot reproduce anyway.

    ``normalization`` divides the inverse before it is realised as a filter.
    Pass the same value for both ears of a pair - see
    :meth:`HeadphoneCorrection.from_measurement` - so the correction keeps the
    interaural level relationship the HRTF established. Left to ``None`` the
    response is scaled by its own peak, which is only correct for a single
    channel considered on its own.
    """

    inverse = _inverse_magnitude(
        response,
        sample_rate_hz,
        taps=taps,
        regularization_db=regularization_db,
        band_hz=band_hz,
    )
    if inverse is None:
        return np.array([1.0])

    if normalization is None:
        normalization = np.max(inverse)
    scale = float(normalization) if normalization > 0 else 1.0
    inverse = inverse / scale

    length = 2 * (inverse.size - 1)
    frequencies = np.fft.rfftfreq(length, d=1.0 / float(sample_rate_hz))
    low, high = band_hz
    inside = (frequencies >= low) & (frequencies <= min(high, 0.5 * sample_rate_hz))
    shaped = np.where(inside, inverse, 1.0)

    from .decomposition import minimum_phase_from_log_magnitude

    return minimum_phase_from_log_magnitude(np.log(np.maximum(shaped, 1e-12)), taps)


@dataclass(frozen=True)
class HeadphoneCorrection:
    """A correction filter pair and where it came from."""

    mode: str = OFF
    filters: NDArray[np.float64] | None = field(default=None, repr=False)
    model: str = ""
    source_path: str = ""
    sample_rate_hz: int = 44_100

    def __post_init__(self) -> None:
        if self.mode not in CORRECTION_MODES:
            raise ValueError(f"unknown correction mode {self.mode!r}; expected one of {CORRECTION_MODES}")
        if self.mode != OFF and self.filters is None:
            raise ValueError(f"correction mode {self.mode!r} needs filters")
        if self.filters is not None:
            block = np.asarray(self.filters, dtype=np.float64)
            if block.ndim != 2 or block.shape[0] != CHANNEL_COUNT:
                raise ValueError(f"expected ({CHANNEL_COUNT}, taps) filters, got shape {block.shape}")
            object.__setattr__(self, "filters", block)

    @property
    def is_active(self) -> bool:
        return self.mode != OFF and self.filters is not None

    def describe(self) -> dict[str, Any]:
        """What the manifest should record about the correction."""

        return {
            "mode": self.mode,
            "model": self.model,
            "source": self.source_path,
            "sample_rate_hz": int(self.sample_rate_hz),
            "taps": 0 if self.filters is None else int(self.filters.shape[1]),
        }

    # --- constructors -------------------------------------------------------

    @classmethod
    def off(cls) -> "HeadphoneCorrection":
        return cls(mode=OFF)

    @classmethod
    def from_measurement(
        cls,
        responses: NDArray[np.floating],
        sample_rate_hz: int,
        *,
        mode: str = USER_IMPULSE_RESPONSE,
        model: str = "",
        source_path: str = "",
        taps: int = 512,
    ) -> "HeadphoneCorrection":
        """Invert a measured ``(2, taps)`` headphone response.

        Each ear gets its own filter - headphones are not symmetric - but both
        are scaled by one shared factor. Normalizing each ear to its own peak
        would flatten the level difference between them, discarding part of
        the interaural relationship the HRTF just established.
        """

        measured = np.asarray(responses, dtype=np.float64)
        if measured.ndim != 2 or measured.shape[0] != CHANNEL_COUNT:
            raise ValueError(f"expected ({CHANNEL_COUNT}, taps) responses, got shape {measured.shape}")

        inverses = [
            _inverse_magnitude(
                measured[ear],
                sample_rate_hz,
                taps=taps,
                regularization_db=DEFAULT_REGULARIZATION_DB,
                band_hz=DEFAULT_BAND_HZ,
            )
            for ear in range(CHANNEL_COUNT)
        ]
        peaks = [float(np.max(inverse)) for inverse in inverses if inverse is not None]
        shared = max(peaks) if peaks else 1.0

        filters = np.vstack(
            [
                inverse_filter_from_response(
                    measured[ear], sample_rate_hz, taps=taps, normalization=shared
                )
                for ear in range(CHANNEL_COUNT)
            ]
        )
        return cls(
            mode=mode,
            filters=filters,
            model=model,
            source_path=source_path,
            sample_rate_hz=int(sample_rate_hz),
        )

    @classmethod
    def from_sofa(
        cls, path: str | Path, sample_rate_hz: int, *, model: str = "Sennheiser HD650", taps: int = 512
    ) -> "HeadphoneCorrection":
        """Build a correction from a headphone transfer measurement in SOFA form."""

        from .sofa_io import load_sofa

        measurement = load_sofa(path)
        responses = np.asarray(measurement.ir, dtype=np.float64)
        # A headphone measurement holds several repetitions; averaging them is
        # the point of measuring more than once.
        averaged = responses.mean(axis=0) if responses.ndim == 3 else responses
        correction = cls.from_measurement(
            averaged,
            int(round(measurement.sample_rate_hz)),
            mode=SONICOM_HD650,
            model=model,
            source_path=str(path),
            taps=taps,
        )
        if int(round(measurement.sample_rate_hz)) != int(sample_rate_hz):
            correction = correction.resampled(int(sample_rate_hz))
        return correction

    @classmethod
    def from_equalization_curve(
        cls,
        frequencies_hz: NDArray[np.floating],
        gains_db: NDArray[np.floating],
        sample_rate_hz: int,
        *,
        taps: int = 512,
        source_path: str = "",
    ) -> "HeadphoneCorrection":
        """Build a correction from a frequency/gain curve, as EQ tools export.

        The curve is interpolated in decibels across the spectrum and realized
        as a minimum-phase filter, so it adds no latency and no pre-ringing.
        """

        from .decomposition import minimum_phase_from_log_magnitude

        points = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
        gains = np.asarray(gains_db, dtype=np.float64).reshape(-1)
        if points.size != gains.size or points.size == 0:
            raise ValueError("the curve needs matching frequency and gain values")

        order = np.argsort(points)
        length = int(2 ** np.ceil(np.log2(max(taps, 64) * 2)))
        spectrum_frequencies = np.fft.rfftfreq(length, d=1.0 / float(sample_rate_hz))
        interpolated = np.interp(spectrum_frequencies, points[order], gains[order])
        log_magnitude = interpolated * (np.log(10.0) / 20.0)

        filters = np.vstack([minimum_phase_from_log_magnitude(log_magnitude, taps)] * CHANNEL_COUNT)
        return cls(
            mode=USER_EQUALIZATION_CURVE,
            filters=filters,
            model="user curve",
            source_path=source_path,
            sample_rate_hz=int(sample_rate_hz),
        )

    def resampled(self, sample_rate_hz: int) -> "HeadphoneCorrection":
        """Resample the correction filters to a different project rate."""

        if self.filters is None or int(sample_rate_hz) == int(self.sample_rate_hz):
            return self
        from math import gcd

        divisor = gcd(int(sample_rate_hz), int(self.sample_rate_hz))
        up = int(sample_rate_hz) // divisor
        down = int(self.sample_rate_hz) // divisor
        resampled = scipy_signal.resample_poly(self.filters, up, down, axis=1)
        return HeadphoneCorrection(
            mode=self.mode,
            filters=np.asarray(resampled, dtype=np.float64),
            model=self.model,
            source_path=self.source_path,
            sample_rate_hz=int(sample_rate_hz),
        )


def apply_correction(
    audio: NDArray[np.floating], correction: HeadphoneCorrection
) -> NDArray[np.float64]:
    """Apply a headphone correction to channel-major stereo audio.

    Each ear gets its own filter - headphones are not symmetric - but the two
    filters come from one measurement pair, so this is not a per-channel
    normalization: the interaural relationship the HRTF established is
    preserved.
    """

    block = np.asarray(audio, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected ({CHANNEL_COUNT}, frames) audio, got shape {block.shape}")
    if not correction.is_active or block.shape[1] == 0:
        return block

    filters = correction.filters
    return np.vstack(
        [
            scipy_signal.fftconvolve(block[ear], filters[ear])[: block.shape[1]]
            for ear in range(CHANNEL_COUNT)
        ]
    )

"""Controlled modification of measured HRTF cues.

A measured HRTF is a fixed set of interaural cues. Sometimes you want to keep
the direction it encodes but change how strongly it says it: widen the image,
soften a pinna notch that fights a particular carrier, or reduce interaural
coherence to loosen a source that sits too tightly inside the head.

Doing that by multiplying wrapped phase or by rescaling raw samples destroys
the response. The route taken here follows the decomposition the specification
sets out, per direction:

1. estimate each ear's broadband delay;
2. split it into a common delay and a differential delay;
3. remove the delay and take a minimum-phase log magnitude;
4. split that into a common magnitude and an ear-differential magnitude;
5. split the common magnitude into a smooth part and the high-frequency
   directional residual the pinna produces;
6. scale each part on its own, then rebuild a causal filter.

    D_c = (D_L + D_R)/2      D_d = (D_L - D_R)/2
    M_c = (log|H_L| + log|H_R|)/2    M_d = (log|H_L| - log|H_R|)/2

    D'_L = D_c + s_itd * D_d + e/2   D'_R = D_c - s_itd * D_d - e/2
    M'_L = M_c + s_ild * M_d         M'_R = M_c - s_ild * M_d

Every scale is neutral at 1.0 and every offset neutral at 0.0, so a transform
left at its defaults must return the dataset unchanged. Normalization, when it
is needed at all, is one gain shared by every direction and both ears:
normalizing an ear or a direction on its own would destroy the very
relationships these controls exist to adjust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT
from .decomposition import (
    broadband_delay_samples,
    common_differential_delay,
    common_differential_log_magnitude,
    minimum_phase_from_log_magnitude,
)

__all__ = [
    "BASIC_RANGES",
    "CUE_PARAMETERS",
    "EXPERT_RANGES",
    "NEUTRAL",
    "CueTransform",
    "CueWarning",
    "ModifiedHrtf",
    "transform_dataset",
    "transform_pair",
]

CUE_PARAMETERS = (
    "itd_scale",
    "ild_scale",
    "pinna_scale",
    "distance_scale",
    "coherence",
    "extra_delay_ms",
)

#: Soft ranges for the basic disclosure level.
BASIC_RANGES = {
    "itd_scale": (0.5, 1.5),
    "ild_scale": (0.5, 1.5),
    "pinna_scale": (0.5, 1.5),
    "distance_scale": (0.5, 2.0),
    "coherence": (0.7, 1.0),
    "extra_delay_ms": (-0.25, 0.25),
}

#: Wider ranges for expert override. These are interaction limits, not
#: physiological claims.
EXPERT_RANGES = {
    "itd_scale": (0.0, 2.5),
    "ild_scale": (0.0, 2.5),
    "pinna_scale": (0.0, 2.0),
    "distance_scale": (0.25, 4.0),
    "coherence": (0.0, 1.0),
    "extra_delay_ms": (-2.0, 2.0),
}

NEUTRAL = {
    "itd_scale": 1.0,
    "ild_scale": 1.0,
    "pinna_scale": 1.0,
    "distance_scale": 1.0,
    "coherence": 1.0,
    "extra_delay_ms": 0.0,
}

#: Above this the response is treated as pinna detail rather than shared
#: timbre. Elevation and front/back cues live here.
DEFAULT_PINNA_CUTOFF_HZ = 4000.0

#: Quefrency cutoff for separating smooth spectral shape from fine structure.
_SMOOTHING_CEPSTRUM_BINS = 24

_EPSILON = 1e-12


@dataclass(frozen=True)
class CueWarning:
    """A cue combination that is legal but worth questioning."""

    parameter: str
    message: str


@dataclass(frozen=True)
class CueTransform:
    """How strongly a modified HRTF should express each cue.

    Neutral at construction: every default leaves the dataset alone.
    """

    itd_scale: float = 1.0
    ild_scale: float = 1.0
    pinna_scale: float = 1.0
    distance_scale: float = 1.0
    coherence: float = 1.0
    extra_delay_ms: float = 0.0
    pinna_cutoff_hz: float = DEFAULT_PINNA_CUTOFF_HZ
    #: Fixed so a rendered result is reproducible from the project alone.
    seed: int = 0

    def __post_init__(self) -> None:
        for name in CUE_PARAMETERS:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError(f"coherence must lie in [0, 1], got {self.coherence!r}")
        if self.pinna_cutoff_hz <= 0.0:
            raise ValueError("pinna_cutoff_hz must be positive")

    @property
    def is_neutral(self) -> bool:
        """True when this transform would leave a dataset unchanged."""

        return all(
            math.isclose(float(getattr(self, name)), NEUTRAL[name], abs_tol=1e-12)
            for name in CUE_PARAMETERS
        )

    def warnings(self) -> tuple[CueWarning, ...]:
        """Cue settings that contradict the direction the HRTF encodes.

        None of these are errors. They are the combinations where what the
        listener hears stops agreeing with where the source is supposed to be.
        """

        found: list[CueWarning] = []
        if self.itd_scale == 0.0 and self.ild_scale == 0.0:
            found.append(
                CueWarning(
                    "itd_scale",
                    "both interaural cues are scaled to zero; the source will "
                    "collapse to the centre regardless of its direction",
                )
            )
        if self.itd_scale < 0.0 or self.ild_scale < 0.0:
            found.append(
                CueWarning(
                    "itd_scale" if self.itd_scale < 0.0 else "ild_scale",
                    "a negative cue scale mirrors that cue, so timing and level "
                    "will point at opposite sides",
                )
            )
        elif self.extra_delay_ms != 0.0 and self.itd_scale > 0.0:
            # An extra delay large enough to outweigh the measured one flips
            # the timing cue while the level cue keeps pointing the other way.
            if abs(self.extra_delay_ms) > 0.7:
                found.append(
                    CueWarning(
                        "extra_delay_ms",
                        "an extra differential delay beyond about 0.7 ms exceeds "
                        "a natural head width and may fight the level cue",
                    )
                )
        if self.pinna_scale == 0.0:
            found.append(
                CueWarning(
                    "pinna_scale",
                    "removing the pinna residual removes the elevation and "
                    "front/back cues with it",
                )
            )
        if self.coherence < 0.5:
            found.append(
                CueWarning(
                    "coherence",
                    "strong decorrelation widens the image but blurs its "
                    "direction",
                )
            )
        return tuple(found)

    def describe(self) -> dict[str, Any]:
        """Provenance for a derived asset."""

        payload = {name: float(getattr(self, name)) for name in CUE_PARAMETERS}
        payload["pinna_cutoff_hz"] = float(self.pinna_cutoff_hz)
        payload["seed"] = int(self.seed)
        payload["neutral"] = self.is_neutral
        return payload

    @classmethod
    def from_mapping(cls, data: Any) -> "CueTransform":
        """Build from stored options, ignoring keys this build does not know."""

        values = dict(data or {})
        known = {
            name: float(values[name])
            for name in (*CUE_PARAMETERS, "pinna_cutoff_hz")
            if name in values
        }
        if "seed" in values:
            known["seed"] = int(values["seed"])
        return cls(**known)

    def clamped(self, *, expert: bool = True) -> "CueTransform":
        """Clamp every cue into its documented range."""

        ranges = EXPERT_RANGES if expert else BASIC_RANGES
        updates = {}
        for name in CUE_PARAMETERS:
            low, high = ranges[name]
            updates[name] = float(np.clip(float(getattr(self, name)), low, high))
        return replace(self, **updates)


@dataclass(frozen=True)
class ModifiedHrtf:
    """A transformed copy of a dataset, and how it was produced."""

    hrirs: NDArray[np.float64]  # (measurements, 2, taps)
    positions_m: NDArray[np.float64]
    sample_rate_hz: float
    transform: CueTransform
    #: One gain applied to every direction and both ears, or 1.0.
    normalization_gain: float
    source_hash: str = ""

    @property
    def measurements(self) -> int:
        return int(self.hrirs.shape[0])

    @property
    def taps(self) -> int:
        return int(self.hrirs.shape[2])

    def quality_metrics(self) -> dict[str, float]:
        """Numbers worth recording alongside a derived asset."""

        return {
            "peak_amplitude": float(np.max(np.abs(self.hrirs), initial=0.0)),
            "normalization_gain": float(self.normalization_gain),
            "measurements": float(self.measurements),
            "taps": float(self.taps),
        }


def _smooth_log_magnitude(log_magnitude: NDArray[np.floating]) -> NDArray[np.float64]:
    """The low-quefrency part of a log magnitude: its broad spectral shape.

    Subtracting this leaves the fine structure - the notches and peaks the
    pinna imposes, which is what ``pinna_scale`` is meant to act on.
    """

    values = np.asarray(log_magnitude, dtype=np.float64)
    # Mirror to a full spectrum so the cepstrum is real and even.
    spectrum = np.concatenate((values, values[-2:0:-1])) if values.size > 2 else values
    cepstrum = np.fft.irfft(np.fft.rfft(spectrum), n=spectrum.size)
    lifter = np.zeros_like(cepstrum)
    cut = min(_SMOOTHING_CEPSTRUM_BINS, cepstrum.size // 2)
    lifter[:cut] = 1.0
    lifter[-cut + 1 :] = 1.0 if cut > 1 else 0.0
    smoothed = np.fft.rfft(cepstrum * lifter).real
    return smoothed[: values.size]


def _pinna_weight(bins: int, sample_rate_hz: float, cutoff_hz: float) -> NDArray[np.float64]:
    """A smooth 0-to-1 ramp above ``cutoff_hz``.

    A hard switch at the cutoff would put a step in the magnitude and ring in
    the reconstructed filter.
    """

    frequencies = np.linspace(0.0, sample_rate_hz / 2.0, bins)
    upper = min(2.0 * cutoff_hz, sample_rate_hz / 2.0)
    if upper <= cutoff_hz:
        return (frequencies >= cutoff_hz).astype(np.float64)
    ramp = (frequencies - cutoff_hz) / (upper - cutoff_hz)
    return np.clip(ramp, 0.0, 1.0)


def _delayed_response(
    response: NDArray[np.floating], delay_samples: float, taps: int
) -> NDArray[np.float64]:
    """Delay a response by a phase ramp rather than by resampling it.

    A time-domain fractional delay interpolates between samples, and that
    interpolation is not magnitude-flat: it rolls off the top of the band. Here
    the response is already being rebuilt from a magnitude we care about
    preserving, so the delay goes on as a linear phase term instead, which
    leaves every magnitude untouched by construction.

    The buffer is padded first so a delayed tail cannot wrap around onto the
    front of the response.
    """

    samples = np.asarray(response, dtype=np.float64)
    span = samples.size + int(np.ceil(abs(delay_samples))) + 8
    length = 1 << int(np.ceil(np.log2(max(span, 2))))

    padded = np.zeros(length, dtype=np.float64)
    padded[: samples.size] = samples

    spectrum = np.fft.rfft(padded)
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(length) * float(delay_samples))
    delayed = np.fft.irfft(spectrum * ramp, length)
    return delayed[:taps]


def _decorrelate(
    hrir_pair: NDArray[np.floating], coherence: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Reduce interaural coherence without changing either ear's level.

    Each ear is mixed with an all-pass version of itself: randomising phase
    while leaving magnitude alone loosens the image without touching the
    interaural level difference the ILD control owns.
    """

    if coherence >= 1.0:
        return np.asarray(hrir_pair, dtype=np.float64)

    pair = np.asarray(hrir_pair, dtype=np.float64)
    taps = pair.shape[1]
    mixed = np.empty_like(pair)
    # Opposite phase perturbations per ear decorrelate them from each other
    # faster than independent draws of the same magnitude would.
    phase = rng.uniform(-np.pi, np.pi, size=np.fft.rfftfreq(taps).size)
    phase[0] = 0.0
    if taps % 2 == 0:
        phase[-1] = 0.0

    for ear in range(CHANNEL_COUNT):
        spectrum = np.fft.rfft(pair[ear])
        sign = 1.0 if ear == 0 else -1.0
        scattered = np.fft.irfft(spectrum * np.exp(1j * sign * phase), n=taps)
        # Equal power, so reducing coherence does not also reduce level.
        mixed[ear] = math.sqrt(coherence) * pair[ear] + math.sqrt(1.0 - coherence) * scattered
    return mixed


def transform_pair(
    hrir_pair: NDArray[np.floating],
    sample_rate_hz: float,
    transform: CueTransform,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Apply the cue transform to one ``(2, taps)`` HRIR pair.

    Returns a pair of the same length. A neutral transform returns the input
    essentially unchanged.
    """

    pair = np.asarray(hrir_pair, dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected a ({CHANNEL_COUNT}, taps) HRIR pair, got {pair.shape}")
    if transform.is_neutral:
        return pair.copy()

    taps = pair.shape[1]

    # 1-2. delay, split into common and differential.
    delays = np.array([broadband_delay_samples(pair[ear]) for ear in range(CHANNEL_COUNT)])
    common_delay, differential_delay = common_differential_delay(delays)

    # 3-4. log magnitude, split the same way. Removing the delay first would
    # not change any of these numbers - a delay is a phase term and leaves
    # magnitude alone - so the responses are read as measured rather than
    # resampled onto an aligned grid first. What the alignment step exists to
    # protect is blending *across* directions, which is not happening here.
    spectra = np.fft.rfft(pair, axis=1)
    log_magnitudes = np.log(np.maximum(np.abs(spectra), _EPSILON))
    common_magnitude, differential_magnitude = common_differential_log_magnitude(log_magnitudes)

    # 5. the pinna residual is the fine structure of the shared response above
    #    the cutoff; scaling it moves elevation and front/back cues.
    if transform.pinna_scale != 1.0:
        smooth = _smooth_log_magnitude(common_magnitude)
        residual = common_magnitude - smooth
        weight = _pinna_weight(common_magnitude.size, sample_rate_hz, transform.pinna_cutoff_hz)
        scaling = 1.0 + (transform.pinna_scale - 1.0) * weight
        common_magnitude = smooth + residual * scaling

    # 6. scale each part and rebuild.
    extra_delay_samples = transform.extra_delay_ms * 1e-3 * float(sample_rate_hz)
    scaled_common_delay = common_delay * transform.distance_scale
    left_delay = (
        scaled_common_delay
        + transform.itd_scale * differential_delay
        + 0.5 * extra_delay_samples
    )
    right_delay = (
        scaled_common_delay
        - transform.itd_scale * differential_delay
        - 0.5 * extra_delay_samples
    )

    left_magnitude = common_magnitude + transform.ild_scale * differential_magnitude
    right_magnitude = common_magnitude - transform.ild_scale * differential_magnitude

    # Distance also changes level, shared by both ears so ILD is untouched.
    distance_gain = 1.0 / max(transform.distance_scale, _EPSILON)

    rebuilt = np.vstack(
        [
            _delayed_response(
                minimum_phase_from_log_magnitude(magnitude, taps), delay, taps
            )
            for magnitude, delay in (
                (left_magnitude, left_delay),
                (right_magnitude, right_delay),
            )
        ]
    ) * distance_gain

    if transform.coherence < 1.0:
        rebuilt = _decorrelate(rebuilt, transform.coherence, rng or np.random.default_rng(0))
    return rebuilt


def transform_dataset(
    dataset: Any,
    transform: CueTransform,
    *,
    peak_ceiling: float = 0.99,
) -> ModifiedHrtf:
    """Apply a cue transform across every direction of a loaded dataset.

    Normalization, if the result needs any, is one gain shared by every
    direction and both ears. Normalizing per ear would destroy the interaural
    level difference; normalizing per direction would destroy the relationship
    between directions, which is what makes a source seem to move.
    """

    hrirs = np.asarray(dataset.ir, dtype=np.float64)
    sample_rate = float(dataset.sample_rate_hz)
    measurements = hrirs.shape[0]

    modified = np.empty_like(hrirs)
    for index in range(measurements):
        # Seeded per direction so a render is reproducible, and so two
        # directions do not share the same decorrelation pattern.
        rng = np.random.default_rng((int(transform.seed), index))
        modified[index] = transform_pair(hrirs[index], sample_rate, transform, rng=rng)

    gain = 1.0
    peak = float(np.max(np.abs(modified), initial=0.0))
    if peak > peak_ceiling and peak > 0.0:
        gain = peak_ceiling / peak
        modified *= gain

    return ModifiedHrtf(
        hrirs=modified,
        positions_m=np.array(dataset.positions_m, dtype=np.float64, copy=True),
        sample_rate_hz=sample_rate,
        transform=transform,
        normalization_gain=gain,
        source_hash=str(getattr(dataset, "content_hash", "")),
    )


def summarize_warnings(warnings: Iterable[CueWarning]) -> str:
    """One line per warning, for a status area."""

    return "\n".join(f"{warning.parameter}: {warning.message}" for warning in warnings)

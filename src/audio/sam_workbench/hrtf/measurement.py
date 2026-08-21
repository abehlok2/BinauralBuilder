"""Importing a measured HRTF from swept-sine recordings.

This is Route B of the specification, and it is gated behind an Advanced flag
for a reason. Measuring your own HRTF means playing calibrated sweeps at a
known level into your own ears with microphones at the canal entrance; done
carelessly it produces both bad data and a hearing risk. Route A - choosing
among generic sets by localization test - costs nothing and should come first.

The chain implemented here is the one the specification sets out:

1. play an exponential sine sweep;
2. deconvolve the recording with the inverse sweep;
3. divide by the regularised loudspeaker/reference response;
4. window before the first strong room reflection;
5. check signal-to-noise, clipping, repeatability, and onset alignment;
6. only then write SOFA.

Step five is not advisory. :func:`import_measurement` refuses to produce an
impulse response from recordings that fail it, because a quietly-accepted bad
measurement is worse than none: it looks like data, it gets written to a file
with a hash, and it will be trusted later by someone who was not in the room.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ADVANCED_FLAG",
    "MeasurementError",
    "QualityReport",
    "QualityThresholds",
    "advanced_enabled",
    "assess_quality",
    "deconvolve",
    "divide_reference",
    "exponential_sweep",
    "import_measurement",
    "inverse_sweep",
    "window_before_reflection",
]

#: Set this to enable the measurement tools. They are off by default.
ADVANCED_FLAG = "SAM_WORKBENCH_ADVANCED"

_EPSILON = 1e-12


def advanced_enabled() -> bool:
    """Whether the advanced measurement tools have been switched on."""

    value = os.environ.get(ADVANCED_FLAG, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


class MeasurementError(RuntimeError):
    """Raised when a measurement cannot be trusted, with the reason."""


# --- sweeps -----------------------------------------------------------------


def exponential_sweep(
    start_hz: float = 20.0,
    end_hz: float = 20_000.0,
    duration_s: float = 2.0,
    sample_rate_hz: float = 48_000.0,
    *,
    fade_s: float = 0.01,
) -> NDArray[np.float64]:
    """A logarithmic sine sweep.

    Exponential rather than linear because its deconvolution separates harmonic
    distortion from the linear response: the distortion products arrive
    *before* the impulse and can be windowed away, rather than smearing into
    it. That property is the reason this is the standard measurement signal.
    """

    if not 0.0 < start_hz < end_hz:
        raise ValueError(f"sweep must rise from a positive frequency, got {start_hz}..{end_hz}")
    nyquist = 0.5 * float(sample_rate_hz)
    if end_hz > nyquist:
        raise ValueError(f"sweep end {end_hz} Hz is above Nyquist ({nyquist} Hz)")
    frames = int(round(float(duration_s) * float(sample_rate_hz)))
    if frames < 2:
        raise ValueError("sweep duration is too short")

    times = np.arange(frames) / float(sample_rate_hz)
    duration = frames / float(sample_rate_hz)
    ratio = math.log(end_hz / start_hz)
    phase = 2.0 * np.pi * start_hz * duration / ratio * (np.exp(times * ratio / duration) - 1.0)
    sweep = np.sin(phase)

    fade = int(round(max(0.0, float(fade_s)) * float(sample_rate_hz)))
    if fade > 0:
        ramp = np.hanning(2 * fade)
        sweep[:fade] *= ramp[:fade]
        sweep[-fade:] *= ramp[fade:]
    return sweep


def inverse_sweep(
    start_hz: float = 20.0,
    end_hz: float = 20_000.0,
    duration_s: float = 2.0,
    sample_rate_hz: float = 48_000.0,
) -> NDArray[np.float64]:
    """The inverse filter for :func:`exponential_sweep`.

    Time-reversed and amplitude-corrected: an exponential sweep spends longer
    at low frequencies, so reversing alone would leave a pink tilt. The
    correction removes it, making the deconvolved result flat.
    """

    sweep = exponential_sweep(start_hz, end_hz, duration_s, sample_rate_hz, fade_s=0.0)
    frames = sweep.size
    ratio = math.log(end_hz / start_hz)
    # An exponential sweep spends longer at low frequencies, so its energy is
    # pink and the time-reversed matched filter alone would leave that tilt.
    # The envelope removes it by attenuating the low end. In the *reversed*
    # sweep the low frequencies sit at the end - index zero is the sweep's last
    # moment, its highest frequency - so the envelope decays with index,
    # reaching f0 / f1 where the reversal is at its lowest.
    envelope = np.exp(-np.arange(frames) * ratio / frames)
    return sweep[::-1] * envelope


def deconvolve(
    recording: NDArray[np.floating], inverse: NDArray[np.floating], *, normalize: bool = True
) -> NDArray[np.float64]:
    """Recover an impulse response by convolving a recording with the inverse sweep."""

    from scipy.signal import fftconvolve

    signal = np.asarray(recording, dtype=np.float64)
    filter_taps = np.asarray(inverse, dtype=np.float64)
    result = fftconvolve(signal, filter_taps, mode="full")
    if normalize:
        peak = float(np.max(np.abs(result), initial=0.0))
        if peak > _EPSILON:
            result = result / peak
    return result


def divide_reference(
    response: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    sample_rate_hz: float = 48_000.0,
    regularization_db: float = -20.0,
    band_hz: tuple[float, float] = (40.0, 18_000.0),
) -> NDArray[np.float64]:
    """Remove the loudspeaker and reference chain from a measurement.

    Regularised and band-limited: outside the band the loudspeaker has little
    output, and dividing by nearly nothing would turn its noise floor into an
    enormous boost that the measurement then carries forever.
    """

    measured = np.asarray(response, dtype=np.float64)
    chain = np.asarray(reference, dtype=np.float64)
    length = 1 << int(np.ceil(np.log2(max(measured.size + chain.size, 4))))

    numerator = np.fft.rfft(measured, length)
    denominator = np.fft.rfft(chain, length)

    magnitude = np.abs(denominator)
    ceiling = float(np.max(magnitude)) if magnitude.size else 1.0
    floor = ceiling * (10.0 ** (float(regularization_db) / 20.0))
    safe = np.where(magnitude > floor, denominator, floor * np.exp(1j * np.angle(denominator)))

    corrected = numerator / safe
    frequencies = np.fft.rfftfreq(length, 1.0 / float(sample_rate_hz))
    low, high = band_hz
    inside = (frequencies >= low) & (frequencies <= min(high, 0.5 * float(sample_rate_hz)))
    corrected = np.where(inside, corrected, 0.0)
    return np.fft.irfft(corrected, length)[: measured.size]


def window_before_reflection(
    response: NDArray[np.floating],
    sample_rate_hz: float = 48_000.0,
    *,
    max_length_ms: float = 5.0,
    reflection_threshold_db: float = -20.0,
    fade_ms: float = 0.5,
) -> tuple[NDArray[np.float64], int]:
    """Keep the direct arrival and cut before the first strong reflection.

    Returns the windowed response and the sample where the cut was made. A
    room reflection is part of the room, not of the head, and leaving it in
    bakes the measurement space into every direction rendered with the result.
    """

    signal = np.asarray(response, dtype=np.float64)
    if signal.size == 0:
        return signal.copy(), 0

    peak_index = int(np.argmax(np.abs(signal)))
    peak = float(abs(signal[peak_index]))
    if peak <= _EPSILON:
        return signal.copy(), signal.size

    limit = peak_index + max(1, int(round(float(max_length_ms) * 1e-3 * float(sample_rate_hz))))
    limit = min(limit, signal.size)

    # Look for a later arrival that rises back above the threshold, which is
    # what a discrete reflection looks like after the direct sound has decayed.
    threshold = peak * (10.0 ** (float(reflection_threshold_db) / 20.0))
    search_start = peak_index + max(1, int(round(0.0005 * float(sample_rate_hz))))
    cut = limit
    for index in range(search_start, limit):
        if abs(signal[index]) > threshold and abs(signal[index]) > abs(signal[index - 1]):
            cut = index
            break

    windowed = np.zeros_like(signal)
    windowed[:cut] = signal[:cut]
    fade = min(int(round(float(fade_ms) * 1e-3 * float(sample_rate_hz))), cut)
    if fade > 1:
        windowed[cut - fade : cut] *= np.hanning(2 * fade)[fade:]
    return windowed, int(cut)


# --- quality ----------------------------------------------------------------


@dataclass(frozen=True)
class QualityThresholds:
    """What a measurement has to clear to be accepted."""

    min_snr_db: float = 30.0
    max_sample_peak: float = 0.99
    #: Repeats must agree this well, as a correlation between them.
    min_repeatability: float = 0.95
    #: And their onsets must line up this closely, in samples.
    max_onset_spread_samples: float = 4.0


@dataclass(frozen=True)
class QualityReport:
    """Whether a measurement can be trusted, and why not if it cannot."""

    snr_db: float
    peak: float
    clipped: bool
    repeatability: float
    onset_spread_samples: float
    onset_index: int
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passes(self) -> bool:
        return not self.issues

    def describe(self) -> dict[str, Any]:
        return {
            "snrDb": float(self.snr_db),
            "peak": float(self.peak),
            "clipped": bool(self.clipped),
            "repeatability": float(self.repeatability),
            "onsetSpreadSamples": float(self.onset_spread_samples),
            "onsetIndex": int(self.onset_index),
            "passes": bool(self.passes),
            "issues": list(self.issues),
        }


def _onset(signal: NDArray[np.floating], fraction: float = 0.2) -> int:
    """First sample rising above a fraction of the peak: the direct arrival."""

    values = np.abs(np.asarray(signal, dtype=np.float64))
    peak = float(np.max(values, initial=0.0))
    if peak <= _EPSILON:
        return 0
    above = np.flatnonzero(values >= fraction * peak)
    return int(above[0]) if above.size else 0


def assess_quality(
    recordings: Sequence[NDArray[np.floating]],
    response: NDArray[np.floating],
    *,
    sample_rate_hz: float = 48_000.0,
    noise_floor: NDArray[np.floating] | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Check signal-to-noise, clipping, repeatability, and onset alignment."""

    thresholds = thresholds or QualityThresholds()
    takes = [np.asarray(take, dtype=np.float64) for take in recordings]
    if not takes:
        raise ValueError("quality assessment needs at least one recording")

    issues: list[str] = []

    peak = max(float(np.max(np.abs(take), initial=0.0)) for take in takes)
    clipped = peak >= thresholds.max_sample_peak
    if clipped:
        issues.append(
            f"a recording peaks at {peak:.3f}, at or above {thresholds.max_sample_peak}; "
            "reduce the level and measure again"
        )

    if noise_floor is not None and np.asarray(noise_floor).size:
        noise = float(np.sqrt(np.mean(np.square(np.asarray(noise_floor, dtype=np.float64)))))
    else:
        # Without a separate noise recording, use the tail of the response,
        # which is past the direct sound and any early reflection.
        tail = np.asarray(response, dtype=np.float64)
        noise = float(np.sqrt(np.mean(np.square(tail[-max(16, tail.size // 8) :]))))
    signal_rms = float(np.sqrt(np.mean(np.square(np.asarray(response, dtype=np.float64)))))
    snr_db = 20.0 * math.log10(max(signal_rms, _EPSILON) / max(noise, _EPSILON))
    if snr_db < thresholds.min_snr_db:
        issues.append(
            f"signal-to-noise is {snr_db:.1f} dB, below the {thresholds.min_snr_db:.0f} dB minimum"
        )

    repeatability = 1.0
    onset_spread = 0.0
    onsets = [float(_onset(take)) for take in takes]
    if len(takes) > 1:
        length = min(take.size for take in takes)
        correlations = []
        for first in range(len(takes)):
            for second in range(first + 1, len(takes)):
                a = takes[first][:length] - np.mean(takes[first][:length])
                b = takes[second][:length] - np.mean(takes[second][:length])
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                correlations.append(float(np.dot(a, b) / denominator) if denominator > _EPSILON else 0.0)
        repeatability = float(np.min(correlations)) if correlations else 1.0
        onset_spread = float(np.max(onsets) - np.min(onsets))

        if repeatability < thresholds.min_repeatability:
            issues.append(
                f"repeats correlate at {repeatability:.3f}, below {thresholds.min_repeatability}; "
                "the subject or the rig moved between takes"
            )
        if onset_spread > thresholds.max_onset_spread_samples:
            issues.append(
                f"onsets differ by {onset_spread:.0f} samples, more than "
                f"{thresholds.max_onset_spread_samples:.0f}; the geometry was not held constant"
            )

    return QualityReport(
        snr_db=snr_db,
        peak=peak,
        clipped=clipped,
        repeatability=repeatability,
        onset_spread_samples=onset_spread,
        onset_index=int(onsets[0]) if onsets else 0,
        issues=tuple(issues),
    )


def import_measurement(
    recordings: Sequence[NDArray[np.floating]],
    *,
    sample_rate_hz: float = 48_000.0,
    sweep_range_hz: tuple[float, float] = (20.0, 20_000.0),
    sweep_duration_s: float = 2.0,
    reference: NDArray[np.floating] | None = None,
    noise_floor: NDArray[np.floating] | None = None,
    thresholds: QualityThresholds | None = None,
    require_advanced: bool = True,
    max_length_ms: float = 5.0,
) -> tuple[NDArray[np.float64], QualityReport]:
    """Turn swept-sine recordings into one windowed impulse response.

    Raises :class:`MeasurementError` when the advanced flag is unset, or when
    the recordings fail the quality checks. A measurement that fails is not
    returned with a warning attached: it is refused, because a file written
    from it would outlive the warning.
    """

    if require_advanced and not advanced_enabled():
        raise MeasurementError(
            "the measurement tools are an advanced feature and are off by default; "
            f"set {ADVANCED_FLAG}=1 to enable them. Choosing among generic HRTFs by "
            "localization test costs nothing and should be tried first."
        )
    if not recordings:
        raise MeasurementError("no recordings were supplied")

    start_hz, end_hz = sweep_range_hz
    inverse = inverse_sweep(start_hz, end_hz, sweep_duration_s, sample_rate_hz)

    responses = [deconvolve(take, inverse) for take in recordings]
    averaged = np.mean(np.stack(responses, axis=0), axis=0)

    if reference is not None:
        reference_response = deconvolve(np.asarray(reference, dtype=np.float64), inverse)
        averaged = divide_reference(
            averaged, reference_response, sample_rate_hz=sample_rate_hz
        )

    windowed, _cut = window_before_reflection(
        averaged, sample_rate_hz, max_length_ms=max_length_ms
    )

    report = assess_quality(
        recordings,
        windowed,
        sample_rate_hz=sample_rate_hz,
        noise_floor=noise_floor,
        thresholds=thresholds,
    )
    if not report.passes:
        raise MeasurementError(
            "the measurement did not pass its quality checks:\n  - "
            + "\n  - ".join(report.issues)
        )
    return windowed, report

"""A broadband anchor to give a low-frequency carrier something to localise by.

A 100-300 Hz sine has almost no energy where the pinna does its work, so the
spectral notches that tell a listener "above" from "in front" have nothing to
act on. The carrier can be placed by interaural time and level differences, and
those alone tend to be heard inside the head.

The fix is not to change the carrier but to add a quiet broadband companion on
the same path. The pinna colours the anchor, the listener localises the anchor,
and the carrier is pulled outward along with it.

It is never enabled silently. The default level is -30 dB relative to the
primary source and the default state is off, because an anchor that a listener
did not ask for is an unexplained noise underneath their session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..conventions import db_to_linear

__all__ = [
    "ANCHOR_PATH_MODES",
    "ANCHOR_SOURCE_TYPES",
    "DEFAULT_ANCHOR_LEVEL_DB",
    "AnchorSpec",
    "anchor_directions",
    "make_anchor_signal",
]

ANCHOR_SOURCE_TYPES = (
    "pink_noise",
    "shaped_noise",
    "sparse_grains",
    "harmonic_bank",
    "imported",
)

#: Where the anchor sits relative to the primary source.
ANCHOR_PATH_MODES = ("same", "offset", "mirrored")

#: Quiet enough to support the carrier rather than become the source.
DEFAULT_ANCHOR_LEVEL_DB = -30.0


@dataclass(frozen=True)
class AnchorSpec:
    """The optional broadband companion and how it follows the source."""

    #: Off unless the user turns it on. See the module docstring.
    enabled: bool = False
    source_type: str = "pink_noise"
    level_db: float = DEFAULT_ANCHOR_LEVEL_DB
    band_hz: tuple[float, float] = (200.0, 16_000.0)
    #: 1.0 keeps both ears fed by one signal; lower widens and diffuses it.
    coherence: float = 1.0
    path_mode: str = "same"
    #: Degrees, for ``path_mode="offset"``.
    path_offset_deg: float = 0.0
    fade_ms: float = 25.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.source_type not in ANCHOR_SOURCE_TYPES:
            raise ValueError(
                f"unknown anchor source {self.source_type!r}; expected one of {ANCHOR_SOURCE_TYPES}"
            )
        if self.path_mode not in ANCHOR_PATH_MODES:
            raise ValueError(
                f"unknown anchor path mode {self.path_mode!r}; expected one of {ANCHOR_PATH_MODES}"
            )
        low, high = self.band_hz
        if not (0.0 <= low < high):
            raise ValueError(f"band_hz must be an increasing, non-negative pair, got {self.band_hz!r}")
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError(f"coherence must lie in [0, 1], got {self.coherence!r}")
        if not math.isfinite(self.level_db):
            raise ValueError("level_db must be finite")

    @property
    def gain(self) -> float:
        """Linear gain relative to the primary source."""

        return float(db_to_linear(self.level_db)) if self.enabled else 0.0

    @classmethod
    def from_mapping(cls, data: object) -> "AnchorSpec":
        """Rebuild from stored options, ignoring keys this build does not know."""

        values = dict(data or {})  # type: ignore[arg-type]
        known: dict[str, object] = {}
        if "enabled" in values:
            known["enabled"] = bool(values["enabled"])
        if "sourceType" in values:
            known["source_type"] = str(values["sourceType"])
        if "levelDb" in values:
            known["level_db"] = float(values["levelDb"])
        if "bandHz" in values:
            band = values["bandHz"]
            known["band_hz"] = (float(band[0]), float(band[1]))
        if "coherence" in values:
            known["coherence"] = float(values["coherence"])
        if "pathMode" in values:
            known["path_mode"] = str(values["pathMode"])
        if "pathOffsetDeg" in values:
            known["path_offset_deg"] = float(values["pathOffsetDeg"])
        if "fadeMs" in values:
            known["fade_ms"] = float(values["fadeMs"])
        if "seed" in values:
            known["seed"] = int(values["seed"])
        return cls(**known)  # type: ignore[arg-type]

    def describe(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "sourceType": self.source_type,
            "levelDb": float(self.level_db),
            "bandHz": [float(self.band_hz[0]), float(self.band_hz[1])],
            "coherence": float(self.coherence),
            "pathMode": self.path_mode,
            "pathOffsetDeg": float(self.path_offset_deg),
            "fadeMs": float(self.fade_ms),
            "seed": int(self.seed),
        }


def _band_limit(
    signal: NDArray[np.floating], sample_rate_hz: float, band_hz: tuple[float, float]
) -> NDArray[np.float64]:
    """Keep only the band the anchor is meant to occupy.

    Done on the spectrum rather than with a filter cascade: the anchor is
    generated offline in one block, so there is no state to carry and no reason
    to accept a filter's transition band.
    """

    samples = np.asarray(signal, dtype=np.float64)
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / float(sample_rate_hz))
    low, high = band_hz
    spectrum[(frequencies < low) | (frequencies > high)] = 0.0
    return np.fft.irfft(spectrum, samples.size)


def _pink(rng: np.random.Generator, frames: int) -> NDArray[np.float64]:
    """Noise with a 1/f magnitude, shaped in the frequency domain."""

    white = rng.normal(0.0, 1.0, frames)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(frames)
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    scale[0] = 0.0  # no DC in an anchor
    return np.fft.irfft(spectrum * scale, frames)


def _sparse_grains(
    rng: np.random.Generator, frames: int, sample_rate_hz: float
) -> NDArray[np.float64]:
    """Short broadband bursts: transients the ear can localise individually."""

    signal = np.zeros(frames)
    grain_length = max(8, int(0.004 * sample_rate_hz))
    window = np.hanning(grain_length)
    spacing = max(grain_length * 2, int(0.08 * sample_rate_hz))
    for start in range(0, max(1, frames - grain_length), spacing):
        jitter = int(rng.integers(0, max(1, spacing // 2)))
        offset = min(start + jitter, max(0, frames - grain_length))
        signal[offset : offset + grain_length] += window * rng.normal(0.0, 1.0, grain_length)
    return signal


def _harmonic_bank(frames: int, sample_rate_hz: float, band_hz: tuple[float, float]) -> NDArray[np.float64]:
    """A harmonic series across the band, tonal rather than noisy."""

    low, high = band_hz
    fundamental = max(low, 50.0)
    times = np.arange(frames) / float(sample_rate_hz)
    signal = np.zeros(frames)
    harmonic = 1
    while fundamental * harmonic <= min(high, sample_rate_hz / 2.0 - 1.0):
        signal += np.sin(2.0 * np.pi * fundamental * harmonic * times) / harmonic
        harmonic += 1
        if harmonic > 200:  # pragma: no cover - guards a pathological band
            break
    return signal


def _apply_fade(signal: NDArray[np.floating], frames_fade: int) -> NDArray[np.float64]:
    """Fade the anchor in and out so it never announces itself with a click."""

    samples = np.asarray(signal, dtype=np.float64).copy()
    span = min(int(frames_fade), samples.size // 2)
    if span <= 0:
        return samples
    ramp = np.linspace(0.0, 1.0, span)
    samples[:span] *= ramp
    samples[-span:] *= ramp[::-1]
    return samples


def make_anchor_signal(
    spec: AnchorSpec,
    frames: int,
    sample_rate_hz: float,
    *,
    material: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Generate the anchor's mono signal, peak-normalised before its level.

    Returns silence when the anchor is disabled, so a caller can mix
    unconditionally without having to branch.
    """

    frames = int(frames)
    if frames <= 0 or not spec.enabled:
        return np.zeros(max(frames, 0), dtype=np.float64)

    rng = np.random.default_rng(spec.seed)
    if spec.source_type == "pink_noise":
        signal = _pink(rng, frames)
    elif spec.source_type == "shaped_noise":
        signal = rng.normal(0.0, 1.0, frames)
    elif spec.source_type == "sparse_grains":
        signal = _sparse_grains(rng, frames, sample_rate_hz)
    elif spec.source_type == "harmonic_bank":
        signal = _harmonic_bank(frames, sample_rate_hz, spec.band_hz)
    else:  # imported
        if material is None:
            raise ValueError("an imported anchor needs the audio to import")
        source = np.asarray(material, dtype=np.float64).reshape(-1)
        if source.size == 0:
            raise ValueError("imported anchor material is empty")
        repeats = int(np.ceil(frames / source.size))
        signal = np.tile(source, repeats)[:frames]

    signal = _band_limit(signal, sample_rate_hz, spec.band_hz)
    peak = float(np.max(np.abs(signal), initial=0.0))
    if peak > 0.0:
        signal = signal / peak

    fade_frames = int(round(spec.fade_ms * 1e-3 * float(sample_rate_hz)))
    return _apply_fade(signal, fade_frames) * spec.gain


def anchor_directions(
    directions: NDArray[np.floating], spec: AnchorSpec
) -> NDArray[np.float64]:
    """Where the anchor sits: on the source, beside it, or opposite it.

    ``directions`` is ``(3,)`` or ``(frames, 3)`` and the result matches.
    """

    path = np.asarray(directions, dtype=np.float64)
    single = path.ndim == 1
    block = path.reshape(1, 3) if single else path

    if spec.path_mode == "same" or (spec.path_mode == "offset" and spec.path_offset_deg == 0.0):
        result = block
    elif spec.path_mode == "mirrored":
        # Mirror across the median plane: the left/right component flips and
        # the front/back and up/down components stay.
        result = block * np.array([1.0, -1.0, 1.0])
    else:
        angle = math.radians(spec.path_offset_deg)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = block @ rotation.T

    return result[0] if single else result

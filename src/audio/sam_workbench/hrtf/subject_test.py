"""Choosing a personal HRTF subject by listening, and recording the result.

There is no universally best subject. Compatibility depends on the
relationship between a measured listener's pinnae, head, and torso and your
own, so the only honest selection method is a structured audition: play the
same directions and the same material through each candidate, rate what you
hear, and keep the scores.

This module is the Qt-free half of that: the audition grid, the test signals,
the rating criteria, the store, and the ranking. The GUI drives it.

Two things it deliberately does *not* do. It does not declare a winner from
acoustics - the scores come from a listener - and it does not assume one
subject suits every use: a set that externalizes broadband noise beautifully
may be wrong for a low-frequency SAM carrier, so ratings are filed under a
named profile and ranked within it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..conventions import DEFAULT_SAMPLE_RATE_HZ, spherical_to_cartesian_m
from .storage import ratings_path, read_json, write_json

__all__ = [
    "AUDITION_AZIMUTHS_DEG",
    "AUDITION_ELEVATIONS_DEG",
    "RATING_CRITERIA",
    "SIGNAL_TYPES",
    "DEFAULT_PROFILE",
    "AuditionDirection",
    "SubjectRating",
    "RatingStore",
    "audition_directions",
    "make_test_signal",
    "rank_subjects",
]

#: The horizontal directions worth checking: front, the two sides, the two
#: rears, and directly behind. Front/back and left/right errors show up here.
AUDITION_AZIMUTHS_DEG: tuple[float, ...] = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0)
#: Elevations spanning below, level, and well above the horizon.
AUDITION_ELEVATIONS_DEG: tuple[float, ...] = (-45.0, -30.0, 0.0, 30.0, 45.0, 60.0)

#: What to audition. Each stresses a different part of the response: clicks
#: and percussion expose timing and smearing, noise exposes spectral cues,
#: speech is what a listener knows best, and motion exposes interpolation.
SIGNAL_TYPES: tuple[str, ...] = (
    "click",
    "pink_noise_burst",
    "speech",
    "percussive",
    "moving_broadband",
    "session_material",
)

#: What to rate. Deliberately perceptual: these are what a listener can judge.
RATING_CRITERIA: tuple[str, ...] = (
    "externalization",
    "front_back_accuracy",
    "elevation_clarity",
    "left_right_accuracy",
    "motion_smoothness",
    "timbre_naturalness",
    "perceived_distance",
    "listening_comfort",
)

#: Ratings are filed per profile, because "best" depends on the material.
DEFAULT_PROFILE = "default"

RATING_MINIMUM = 1
RATING_MAXIMUM = 5


@dataclass(frozen=True)
class AuditionDirection:
    """One direction in the audition grid."""

    azimuth_deg: float
    elevation_deg: float

    @property
    def label(self) -> str:
        return f"{self.azimuth_deg:+.0f}° az, {self.elevation_deg:+.0f}° el"

    def unit_vector(self) -> NDArray[np.float64]:
        return spherical_to_cartesian_m(self.azimuth_deg, self.elevation_deg, 1.0)


def audition_directions(
    azimuths_deg: Sequence[float] = AUDITION_AZIMUTHS_DEG,
    elevations_deg: Sequence[float] = AUDITION_ELEVATIONS_DEG,
) -> tuple[AuditionDirection, ...]:
    """The full audition grid, horizontal ring first."""

    ordered = sorted(elevations_deg, key=lambda value: abs(value))
    return tuple(
        AuditionDirection(azimuth_deg=float(azimuth), elevation_deg=float(elevation))
        for elevation in ordered
        for azimuth in azimuths_deg
    )


# --- test signals -----------------------------------------------------------


def _pink_noise(frames: int, generator: np.random.Generator) -> NDArray[np.float64]:
    """Pink noise by spectral shaping, which is exact rather than approximate."""

    spectrum = np.fft.rfft(generator.normal(size=frames))
    frequencies = np.fft.rfftfreq(frames, d=1.0)
    shaping = np.ones_like(frequencies)
    shaping[1:] = 1.0 / np.sqrt(frequencies[1:])
    shaped = np.fft.irfft(spectrum * shaping, frames)
    peak = float(np.max(np.abs(shaped)))
    return shaped / peak if peak > 0 else shaped


def make_test_signal(
    signal_type: str,
    *,
    duration_s: float = 1.5,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    seed: int = 0,
    material: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Generate one audition signal, mono and peak-normalized.

    ``session_material`` returns the caller's own audio - the carrier or noise
    a session actually uses - because a subject that suits test signals may
    still be wrong for the material a listener will spend an hour with.
    """

    if signal_type not in SIGNAL_TYPES:
        raise ValueError(f"unknown signal type {signal_type!r}; expected one of {SIGNAL_TYPES}")

    frames = max(1, int(float(duration_s) * int(sample_rate_hz)))
    generator = np.random.default_rng(seed)
    times = np.arange(frames) / float(sample_rate_hz)

    if signal_type == "session_material":
        if material is None:
            raise ValueError("session_material needs the session's own audio")
        supplied = np.asarray(material, dtype=np.float64).reshape(-1)
        if supplied.size >= frames:
            return supplied[:frames]
        return np.pad(supplied, (0, frames - supplied.size))

    if signal_type == "click":
        # Sparse, full-bandwidth impulses: nothing exposes smearing faster.
        signal = np.zeros(frames)
        for index in range(max(1, int(duration_s * 3))):
            position = int(index * sample_rate_hz / 3.0)
            if position < frames:
                signal[position] = 1.0
        return signal

    if signal_type == "pink_noise_burst":
        noise = _pink_noise(frames, generator)
        envelope = np.zeros(frames)
        burst = max(1, int(0.25 * sample_rate_hz))
        gap = max(1, int(0.15 * sample_rate_hz))
        position = 0
        while position < frames:
            stop = min(frames, position + burst)
            window = np.hanning(max(2, stop - position))
            envelope[position:stop] = window[: stop - position]
            position = stop + gap
        return noise * envelope

    if signal_type == "speech":
        # A speech-like signal rather than a recording: a voiced fundamental
        # with formants and a syllabic envelope, so the fixture stays synthetic
        # and no third-party audio is bundled.
        fundamental = 120.0 + 20.0 * np.sin(2 * math.pi * 1.5 * times)
        phase = 2 * math.pi * np.cumsum(fundamental) / float(sample_rate_hz)
        voiced = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 12))
        formant = 1.0 + 0.6 * np.sin(2 * math.pi * 3.0 * times)
        syllables = np.clip(np.sin(2 * math.pi * 3.5 * times) ** 2, 0.05, 1.0)
        signal = voiced * formant * syllables
        peak = float(np.max(np.abs(signal)))
        return signal / peak if peak > 0 else signal

    if signal_type == "percussive":
        signal = np.zeros(frames)
        for index in range(max(1, int(duration_s * 4))):
            position = int(index * sample_rate_hz / 4.0)
            if position >= frames:
                break
            length = min(frames - position, int(0.12 * sample_rate_hz))
            decay = np.exp(-np.arange(length) / (0.02 * sample_rate_hz))
            signal[position : position + length] += _pink_noise(length, generator) * decay
        peak = float(np.max(np.abs(signal)))
        return signal / peak if peak > 0 else signal

    # moving_broadband: continuous, full-bandwidth, for judging interpolation.
    return _pink_noise(frames, generator)


# --- ratings ----------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRating:
    """One listener's scores for one subject, under one profile."""

    subject: str
    scores: dict[str, int] = field(default_factory=dict)
    profile: str = DEFAULT_PROFILE
    signal_type: str = "pink_noise_burst"
    asset_path: str = ""
    notes: str = ""
    recorded_utc: str = ""

    def __post_init__(self) -> None:
        cleaned: dict[str, int] = {}
        for criterion, value in self.scores.items():
            if criterion not in RATING_CRITERIA:
                continue
            cleaned[criterion] = int(np.clip(int(value), RATING_MINIMUM, RATING_MAXIMUM))
        object.__setattr__(self, "scores", cleaned)
        if not self.recorded_utc:
            object.__setattr__(
                self, "recorded_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    @property
    def mean_score(self) -> float:
        return float(np.mean(list(self.scores.values()))) if self.scores else 0.0

    def weighted_score(self, weights: Mapping[str, float] | None = None) -> float:
        """Mean score, optionally weighting the criteria that matter to a use."""

        if not self.scores:
            return 0.0
        if not weights:
            return self.mean_score
        total = sum(float(weights.get(name, 0.0)) for name in self.scores)
        if total <= 0.0:
            return self.mean_score
        return float(
            sum(value * float(weights.get(name, 0.0)) for name, value in self.scores.items()) / total
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubjectRating":
        return cls(
            subject=str(data.get("subject", "")),
            scores={str(key): int(value) for key, value in dict(data.get("scores", {})).items()},
            profile=str(data.get("profile", DEFAULT_PROFILE)),
            signal_type=str(data.get("signal_type", "pink_noise_burst")),
            asset_path=str(data.get("asset_path", "")),
            notes=str(data.get("notes", "")),
            recorded_utc=str(data.get("recorded_utc", "")),
        )


@dataclass
class RatingStore:
    """Subject ratings, kept in the application data directory."""

    path: Path | None = None
    ratings: list[SubjectRating] = field(default_factory=list)

    @property
    def location(self) -> Path:
        return self.path or ratings_path()

    def load(self) -> "RatingStore":
        payload = read_json(self.location, default={"ratings": []}) or {}
        self.ratings = [SubjectRating.from_dict(entry) for entry in payload.get("ratings", [])]
        return self

    def save(self) -> Path:
        return write_json(
            self.location,
            {"version": 1, "ratings": [rating.to_dict() for rating in self.ratings]},
        )

    def add(self, rating: SubjectRating) -> None:
        self.ratings.append(rating)

    def profiles(self) -> tuple[str, ...]:
        return tuple(sorted({rating.profile for rating in self.ratings}))

    def subjects(self, profile: str | None = None) -> tuple[str, ...]:
        return tuple(
            sorted({rating.subject for rating in self.ratings if profile in (None, rating.profile)})
        )

    def for_profile(self, profile: str) -> list[SubjectRating]:
        return [rating for rating in self.ratings if rating.profile == profile]


@dataclass(frozen=True)
class SubjectScore:
    """A subject's standing under one profile."""

    subject: str
    score: float
    ratings: int
    criteria: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "score": self.score,
            "ratings": self.ratings,
            "criteria": dict(self.criteria),
        }


def rank_subjects(
    ratings: Iterable[SubjectRating],
    *,
    profile: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[SubjectScore]:
    """Rank subjects best first, from the scores a listener gave them.

    Averaging repeated ratings of the same subject is the point: one audition
    is a mood, several are an opinion.
    """

    selected = [
        rating for rating in ratings if profile is None or rating.profile == profile
    ]
    grouped: dict[str, list[SubjectRating]] = {}
    for rating in selected:
        grouped.setdefault(rating.subject, []).append(rating)

    scores: list[SubjectScore] = []
    for subject, entries in grouped.items():
        per_criterion: dict[str, float] = {}
        for criterion in RATING_CRITERIA:
            values = [entry.scores[criterion] for entry in entries if criterion in entry.scores]
            if values:
                per_criterion[criterion] = float(np.mean(values))
        overall = float(np.mean([entry.weighted_score(weights) for entry in entries]))
        scores.append(
            SubjectScore(
                subject=subject, score=overall, ratings=len(entries), criteria=per_criterion
            )
        )

    scores.sort(key=lambda entry: (-entry.score, entry.subject))
    return scores


__all__.extend(["SubjectScore", "RATING_MAXIMUM", "RATING_MINIMUM"])

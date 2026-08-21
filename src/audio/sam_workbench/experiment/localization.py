"""A blinded localization test for choosing between candidate HRTFs.

Preference ratings answer "which did you like"; this answers "where did you
hear it", which has a right answer. That difference matters, because the HRTF
someone prefers on first listen is not reliably the one they localise best
with, and only the second question can be scored against ground truth.

The design follows Route A of the specification:

* short, level-matched stimuli, so loudness cannot stand in for localisation;
* trials spread across front/back, elevation, lateral and distance, because a
  set can be excellent laterally and useless above;
* hidden labels, so the listener cannot score a subject they already trust;
* repeated trials, so consistency is measurable rather than assumed;
* confidence per response, so a lucky guess and a certain answer are not
  weighted the same;
* a ranking that minimises weighted angular error *and* front/back reversals.

Front/back reversals are counted separately rather than folded into angular
error. A reversal is a categorical failure - the listener heard the source
behind them when it was in front - and averaging it into a degrees-of-error
figure hides exactly the failure that most distinguishes a bad generic HRTF
from a good one.

Everything is seeded. The same session definition produces the same trial
order on any machine, which is what makes a comparison between two listeners,
or the same listener twice, mean anything.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_WEIGHTS",
    "LocalizationScore",
    "LocalizationSession",
    "LocalizationTrial",
    "TrialResponse",
    "angular_error_deg",
    "build_session",
    "is_front_back_reversal",
    "rank_candidates",
    "score_responses",
]

#: Trial directions, chosen to cover the failures that separate HRTFs:
#: lateral placement, front/back confusion, and elevation.
FRONT_BACK_AZIMUTHS_DEG = (0.0, 30.0, -30.0, 150.0, -150.0, 180.0)
LATERAL_AZIMUTHS_DEG = (60.0, -60.0, 90.0, -90.0, 120.0, -120.0)
ELEVATIONS_DEG = (-30.0, 0.0, 30.0, 60.0)
DISTANCES_M = (0.5, 1.0, 2.0)

#: How much each failure counts toward the ranking score. Reversals dominate
#: because they are the failure a listener actually notices.
DEFAULT_WEIGHTS = {
    "azimuth_error": 1.0,
    "elevation_error": 0.75,
    "front_back_reversal": 45.0,
    "distance_error": 5.0,
    "inconsistency": 0.5,
}

_EPSILON = 1e-12


def _wrap_deg(value):
    """Wrap an angle to (-180, 180], so 350 and -10 are ten degrees apart."""

    return ((np.asarray(value, dtype=np.float64) + 180.0) % 360.0) - 180.0


def angular_error_deg(
    target_azimuth: float,
    target_elevation: float,
    reported_azimuth: float,
    reported_elevation: float,
) -> float:
    """Great-circle angle between where a source was and where it was heard.

    Measured on the sphere rather than by differencing azimuth and elevation
    separately: near the poles a large azimuth difference is a small angle, and
    scoring it as large would penalise a listener for being right.
    """

    def unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
        return np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )

    dot = float(
        np.clip(
            np.dot(unit(target_azimuth, target_elevation), unit(reported_azimuth, reported_elevation)),
            -1.0,
            1.0,
        )
    )
    return float(math.degrees(math.acos(dot)))


def is_front_back_reversal(target_azimuth_deg: float, reported_azimuth_deg: float) -> bool:
    """Whether a response put a front source behind, or the reverse.

    Sources near the lateral axis are excluded: at 90 degrees the front/back
    distinction is not what is being tested, and a few degrees of error would
    otherwise register as a categorical failure.
    """

    target = float(_wrap_deg(target_azimuth_deg))
    reported = float(_wrap_deg(reported_azimuth_deg))
    if abs(abs(target) - 90.0) < 20.0:
        return False
    return (abs(target) < 90.0) != (abs(reported) < 90.0)


@dataclass(frozen=True)
class LocalizationTrial:
    """One presentation: a hidden candidate at a known direction."""

    id: str
    #: Opaque while the session runs. The candidate it stands for lives in the
    #: session's reveal map, which the scoring step uses and the listener does not.
    label: str
    candidate: str
    azimuth_deg: float
    elevation_deg: float
    distance_m: float = 1.0
    signal_type: str = "pink_noise_burst"
    repeat_index: int = 0
    kind: str = "front_back"

    def describe(self, *, reveal: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "azimuthDeg": float(self.azimuth_deg),
            "elevationDeg": float(self.elevation_deg),
            "distanceM": float(self.distance_m),
            "signalType": self.signal_type,
            "repeatIndex": int(self.repeat_index),
            "kind": self.kind,
        }
        if reveal:
            payload["candidate"] = self.candidate
        return payload


@dataclass(frozen=True)
class TrialResponse:
    """Where the listener said it was, and how sure they were."""

    trial_id: str
    azimuth_deg: float
    elevation_deg: float = 0.0
    distance_m: float | None = None
    #: 1 to 5. A guess and a certainty should not weigh the same.
    confidence: int = 3

    def __post_init__(self) -> None:
        if not 1 <= int(self.confidence) <= 5:
            raise ValueError(f"confidence runs from 1 to 5, got {self.confidence!r}")
        for name in ("azimuth_deg", "elevation_deg"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def describe(self) -> dict[str, Any]:
        return {
            "trialId": self.trial_id,
            "azimuthDeg": float(self.azimuth_deg),
            "elevationDeg": float(self.elevation_deg),
            "distanceM": None if self.distance_m is None else float(self.distance_m),
            "confidence": int(self.confidence),
        }


@dataclass(frozen=True)
class LocalizationSession:
    """A built, ordered set of trials, and what the labels stand for."""

    trials: tuple[LocalizationTrial, ...]
    seed: int
    #: label -> candidate. Held apart from the trials so a session can be
    #: presented without ever handing the listener the answer.
    reveal: Mapping[str, str] = field(default_factory=dict)
    signal_types: tuple[str, ...] = ("pink_noise_burst",)

    @property
    def candidates(self) -> tuple[str, ...]:
        return tuple(sorted({trial.candidate for trial in self.trials}))

    def trial(self, trial_id: str) -> LocalizationTrial | None:
        for trial in self.trials:
            if trial.id == trial_id:
                return trial
        return None

    def blinded(self) -> tuple[dict[str, Any], ...]:
        """What the listener's side of the test is allowed to see."""

        return tuple(trial.describe(reveal=False) for trial in self.trials)

    def describe(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "signalTypes": list(self.signal_types),
            "trials": [trial.describe(reveal=True) for trial in self.trials],
            "reveal": dict(self.reveal),
        }


def _label_for(candidate: str, seed: int) -> str:
    """A stable opaque label. Same session, same label; different seed, different."""

    digest = hashlib.sha256(f"{seed}\x1f{candidate}".encode("utf-8")).hexdigest()
    return f"set-{digest[:6]}"


def build_session(
    candidates: Sequence[str],
    *,
    seed: int = 0,
    repeats: int = 2,
    signal_types: Sequence[str] = ("pink_noise_burst", "click"),
    include_distance: bool = True,
) -> LocalizationSession:
    """Build a blinded, randomised, reproducible localization session.

    Every candidate meets the same directions the same number of times, so a
    ranking cannot be an artefact of one set having been given easier trials.
    Only the order is randomised, and it is randomised from ``seed``.
    """

    if not candidates:
        raise ValueError("a localization session needs at least one candidate")
    repeats = max(1, int(repeats))

    directions: list[tuple[float, float, str]] = []
    directions += [(azimuth, 0.0, "front_back") for azimuth in FRONT_BACK_AZIMUTHS_DEG]
    directions += [(azimuth, 0.0, "lateral") for azimuth in LATERAL_AZIMUTHS_DEG]
    directions += [(0.0, elevation, "elevation") for elevation in ELEVATIONS_DEG if elevation]
    directions += [(45.0, elevation, "elevation") for elevation in ELEVATIONS_DEG if elevation]

    trials: list[LocalizationTrial] = []
    reveal: dict[str, str] = {}
    for candidate in candidates:
        label = _label_for(candidate, seed)
        reveal[label] = candidate
        for repeat in range(repeats):
            for index, (azimuth, elevation, kind) in enumerate(directions):
                signal = signal_types[index % len(signal_types)]
                trials.append(
                    LocalizationTrial(
                        id=f"{label}-{repeat}-{len(trials)}",
                        label=label,
                        candidate=candidate,
                        azimuth_deg=azimuth,
                        elevation_deg=elevation,
                        distance_m=1.0,
                        signal_type=signal,
                        repeat_index=repeat,
                        kind=kind,
                    )
                )
            if include_distance:
                for distance in DISTANCES_M:
                    trials.append(
                        LocalizationTrial(
                            id=f"{label}-{repeat}-d{len(trials)}",
                            label=label,
                            candidate=candidate,
                            azimuth_deg=30.0,
                            elevation_deg=0.0,
                            distance_m=distance,
                            signal_type=signal_types[0],
                            repeat_index=repeat,
                            kind="distance",
                        )
                    )

    order = np.random.default_rng(seed).permutation(len(trials))
    shuffled = tuple(trials[int(index)] for index in order)
    return LocalizationSession(
        trials=shuffled, seed=int(seed), reveal=reveal, signal_types=tuple(signal_types)
    )


@dataclass(frozen=True)
class LocalizationScore:
    """How well one candidate was localised."""

    candidate: str
    trials: int
    responses: int
    azimuth_error_deg: float
    elevation_error_deg: float
    angular_error_deg: float
    front_back_reversals: int
    front_back_rate: float
    distance_error_m: float
    inconsistency_deg: float
    mean_confidence: float
    weighted_error: float

    def describe(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "trials": int(self.trials),
            "responses": int(self.responses),
            "azimuthErrorDeg": float(self.azimuth_error_deg),
            "elevationErrorDeg": float(self.elevation_error_deg),
            "angularErrorDeg": float(self.angular_error_deg),
            "frontBackReversals": int(self.front_back_reversals),
            "frontBackRate": float(self.front_back_rate),
            "distanceErrorM": float(self.distance_error_m),
            "inconsistencyDeg": float(self.inconsistency_deg),
            "meanConfidence": float(self.mean_confidence),
            "weightedError": float(self.weighted_error),
        }


def score_responses(
    session: LocalizationSession,
    responses: Iterable[TrialResponse],
    *,
    weights: Mapping[str, float] | None = None,
    use_confidence: bool = True,
) -> tuple[LocalizationScore, ...]:
    """Score every candidate, best first.

    Unanswered trials are simply not scored rather than counted as perfect or
    as failures; the response count is reported so a partly finished session is
    visibly partial.
    """

    weight = {**DEFAULT_WEIGHTS, **dict(weights or {})}
    by_trial = {response.trial_id: response for response in responses}

    grouped: dict[str, list[tuple[LocalizationTrial, TrialResponse]]] = {}
    counts: dict[str, int] = {}
    for trial in session.trials:
        counts[trial.candidate] = counts.get(trial.candidate, 0) + 1
        response = by_trial.get(trial.id)
        if response is not None:
            grouped.setdefault(trial.candidate, []).append((trial, response))

    scores: list[LocalizationScore] = []
    for candidate in sorted(counts):
        pairs = grouped.get(candidate, [])
        if not pairs:
            scores.append(
                LocalizationScore(
                    candidate=candidate,
                    trials=counts[candidate],
                    responses=0,
                    azimuth_error_deg=float("nan"),
                    elevation_error_deg=float("nan"),
                    angular_error_deg=float("nan"),
                    front_back_reversals=0,
                    front_back_rate=float("nan"),
                    distance_error_m=float("nan"),
                    inconsistency_deg=float("nan"),
                    mean_confidence=float("nan"),
                    weighted_error=float("inf"),
                )
            )
            continue

        confidences = np.array([response.confidence for _, response in pairs], dtype=np.float64)
        # A confident wrong answer says more about the HRTF than a hesitant one.
        factors = confidences / 3.0 if use_confidence else np.ones_like(confidences)

        azimuth_errors = np.array(
            [abs(float(_wrap_deg(response.azimuth_deg - trial.azimuth_deg))) for trial, response in pairs]
        )
        elevation_errors = np.array(
            [abs(response.elevation_deg - trial.elevation_deg) for trial, response in pairs]
        )
        angular_errors = np.array(
            [
                angular_error_deg(
                    trial.azimuth_deg, trial.elevation_deg, response.azimuth_deg, response.elevation_deg
                )
                for trial, response in pairs
            ]
        )
        reversals = np.array(
            [
                is_front_back_reversal(trial.azimuth_deg, response.azimuth_deg)
                for trial, response in pairs
            ]
        )
        distance_pairs = [
            (trial.distance_m, response.distance_m)
            for trial, response in pairs
            if response.distance_m is not None
        ]
        distance_error = (
            float(np.mean([abs(reported - target) for target, reported in distance_pairs]))
            if distance_pairs
            else 0.0
        )

        # Consistency: how far apart the repeats of one direction landed. A set
        # that gives a different answer each time is unreliable even when its
        # average is good.
        repeats: dict[tuple[float, float, float], list[float]] = {}
        for trial, response in pairs:
            key = (trial.azimuth_deg, trial.elevation_deg, trial.distance_m)
            repeats.setdefault(key, []).append(float(_wrap_deg(response.azimuth_deg)))
        spreads = [
            float(np.std(values)) for values in repeats.values() if len(values) > 1
        ]
        inconsistency = float(np.mean(spreads)) if spreads else 0.0

        weighted = (
            weight["azimuth_error"] * float(np.mean(azimuth_errors * factors))
            + weight["elevation_error"] * float(np.mean(elevation_errors * factors))
            + weight["front_back_reversal"] * float(np.mean(reversals * factors))
            + weight["distance_error"] * distance_error
            + weight["inconsistency"] * inconsistency
        )

        scores.append(
            LocalizationScore(
                candidate=candidate,
                trials=counts[candidate],
                responses=len(pairs),
                azimuth_error_deg=float(np.mean(azimuth_errors)),
                elevation_error_deg=float(np.mean(elevation_errors)),
                angular_error_deg=float(np.mean(angular_errors)),
                front_back_reversals=int(np.sum(reversals)),
                front_back_rate=float(np.mean(reversals)),
                distance_error_m=distance_error,
                inconsistency_deg=inconsistency,
                mean_confidence=float(np.mean(confidences)),
                weighted_error=weighted,
            )
        )

    # Best first; ties broken by name so the order is reproducible.
    scores.sort(key=lambda score: (score.weighted_error, score.candidate))
    return tuple(scores)


def rank_candidates(
    session: LocalizationSession,
    responses: Iterable[TrialResponse],
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Just the candidate names, best localised first."""

    return tuple(score.candidate for score in score_responses(session, responses, weights=weights))

"""Building, running, and exporting a listening experiment.

A result is only worth keeping if someone else can say what was actually
played. So a condition here is not a label - it is the full acoustic
description of one rendering path: which asset, verified by hash; how its
delay was handled; which interpolation ran; what the cue transform did; what
correction was applied afterwards; at what rate and what gain.

The export carries all of it alongside the responses, plus the seed that
produced the order. That is what "complete acoustic provenance" has to mean in
practice: enough to rebuild the exact thing the listener heard, not enough to
recognise its name.

Randomisation is seeded, and blinding is applied at presentation rather than
at construction, so the same experiment can be exported with labels revealed
for analysis and without them for running.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..version import PACKAGE_VERSION

__all__ = [
    "Condition",
    "ExperimentDesign",
    "ExperimentResult",
    "ConditionRun",
    "Rating",
    "build_runs",
    "export_result",
]

#: Bumped when the export layout changes in a way a reader must notice.
EXPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Condition:
    """One complete rendering path, described well enough to reproduce."""

    id: str
    name: str = ""
    renderer: str = "hrtf"
    asset_path: str = ""
    asset_sha256: str = ""
    subject: str = ""
    dataset: str = ""
    processing: str = ""
    interpolation: str = "nearest"
    delay_policy: str = "bake_delay_into_ir"
    sample_rate_hz: float = 48_000.0
    crossfade_ms: float = 12.0
    #: The Phase 5 cue transform, as ``CueTransform.describe()``.
    cue: Mapping[str, Any] = field(default_factory=dict)
    #: The Phase 5 anchor, as ``AnchorSpec.describe()``.
    anchor: Mapping[str, Any] = field(default_factory=dict)
    headphone_correction: str = "off"
    headphone_asset: str = ""
    output_gain_db: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a condition needs a stable identifier")

    @property
    def is_reproducible(self) -> bool:
        """Whether this condition names an asset and pins it by hash.

        A condition without a hash can still be run; it just cannot be proven
        later to have used the file it claims.
        """

        if self.renderer in ("hrtf", "hybrid"):
            return bool(self.asset_path) and bool(self.asset_sha256)
        return True

    def provenance(self) -> dict[str, Any]:
        """Everything needed to rebuild what the listener heard."""

        return {
            "id": self.id,
            "name": self.name,
            "renderer": self.renderer,
            "assetPath": self.asset_path,
            "assetSha256": self.asset_sha256,
            "subject": self.subject,
            "dataset": self.dataset,
            "processing": self.processing,
            "interpolation": self.interpolation,
            "delayPolicy": self.delay_policy,
            "sampleRateHz": float(self.sample_rate_hz),
            "crossfadeMs": float(self.crossfade_ms),
            "cue": dict(self.cue),
            "anchor": dict(self.anchor),
            "headphoneCorrection": self.headphone_correction,
            "headphoneAsset": self.headphone_asset,
            "outputGainDb": float(self.output_gain_db),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Condition":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            renderer=str(data.get("renderer", "hrtf")),
            asset_path=str(data.get("assetPath", "")),
            asset_sha256=str(data.get("assetSha256", "")),
            subject=str(data.get("subject", "")),
            dataset=str(data.get("dataset", "")),
            processing=str(data.get("processing", "")),
            interpolation=str(data.get("interpolation", "nearest")),
            delay_policy=str(data.get("delayPolicy", "bake_delay_into_ir")),
            sample_rate_hz=float(data.get("sampleRateHz", 48_000.0)),
            crossfade_ms=float(data.get("crossfadeMs", 12.0)),
            cue=dict(data.get("cue", {}) or {}),
            anchor=dict(data.get("anchor", {}) or {}),
            headphone_correction=str(data.get("headphoneCorrection", "off")),
            headphone_asset=str(data.get("headphoneAsset", "")),
            output_gain_db=float(data.get("outputGainDb", 0.0)),
            notes=str(data.get("notes", "")),
        )

    def blinded_label(self, seed: int) -> str:
        """An opaque name for this condition, stable within one experiment."""

        digest = hashlib.sha256(f"{seed}\x1f{self.id}".encode("utf-8")).hexdigest()
        return f"condition-{digest[:6]}"


@dataclass(frozen=True)
class ConditionRun:
    """One scheduled presentation of one condition."""

    index: int
    condition_id: str
    label: str
    repeat: int = 0

    def describe(self, *, reveal: bool = False) -> dict[str, Any]:
        payload = {"index": int(self.index), "label": self.label, "repeat": int(self.repeat)}
        if reveal:
            payload["conditionId"] = self.condition_id
        return payload


@dataclass(frozen=True)
class Rating:
    """One captured response, on whatever scale the design declared."""

    run_index: int
    scores: Mapping[str, float] = field(default_factory=dict)
    comment: str = ""
    recorded_utc: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "runIndex": int(self.run_index),
            "scores": {key: float(value) for key, value in self.scores.items()},
            "comment": self.comment,
            "recordedUtc": self.recorded_utc or _now(),
        }


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExperimentDesign:
    """Conditions, how often each is presented, and in what order."""

    conditions: tuple[Condition, ...]
    repeats: int = 1
    seed: int = 0
    blinded: bool = True
    #: What the listener is asked, per presentation.
    criteria: tuple[str, ...] = ("externalization", "front_back_accuracy", "preference")
    name: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("an experiment needs at least one condition")
        identifiers = [condition.id for condition in self.conditions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("condition identifiers must be unique")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be at least 1")

    def condition(self, condition_id: str) -> Condition | None:
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        return None

    @property
    def reveal(self) -> dict[str, str]:
        """label -> condition id, for analysis rather than for the listener."""

        return {
            condition.blinded_label(self.seed): condition.id for condition in self.conditions
        }

    def unreproducible(self) -> tuple[str, ...]:
        """Conditions that could not later be proven to have used their asset."""

        return tuple(
            condition.id for condition in self.conditions if not condition.is_reproducible
        )

    def build_runs(self) -> tuple[ConditionRun, ...]:
        """The presentation order: every condition, ``repeats`` times, shuffled.

        Balanced by construction and shuffled from the seed, so no condition
        can be advantaged by position and the order reproduces exactly.
        """

        return build_runs(self.conditions, repeats=self.repeats, seed=self.seed)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": int(self.seed),
            "repeats": int(self.repeats),
            "blinded": bool(self.blinded),
            "criteria": list(self.criteria),
            "notes": self.notes,
            "conditions": [condition.provenance() for condition in self.conditions],
        }


def build_runs(
    conditions: Sequence[Condition], *, repeats: int = 1, seed: int = 0
) -> tuple[ConditionRun, ...]:
    """Balanced, seeded presentation order."""

    scheduled: list[tuple[str, str, int]] = []
    for repeat in range(max(1, int(repeats))):
        for condition in conditions:
            scheduled.append((condition.id, condition.blinded_label(seed), repeat))

    order = np.random.default_rng(seed).permutation(len(scheduled))
    runs: list[ConditionRun] = []
    for position, source in enumerate(order):
        condition_id, label, repeat = scheduled[int(source)]
        runs.append(ConditionRun(index=position, condition_id=condition_id, label=label, repeat=repeat))
    return tuple(runs)


@dataclass(frozen=True)
class ExperimentResult:
    """A design, the order it ran in, and what the listener said."""

    design: ExperimentDesign
    runs: tuple[ConditionRun, ...]
    ratings: tuple[Rating, ...] = ()
    listener: str = ""
    started_utc: str = ""
    finished_utc: str = ""

    def rating_for(self, run_index: int) -> Rating | None:
        for rating in self.ratings:
            if rating.run_index == run_index:
                return rating
        return None

    @property
    def completion(self) -> float:
        """Fraction of scheduled presentations that were answered."""

        if not self.runs:
            return 0.0
        answered = {rating.run_index for rating in self.ratings}
        return len(answered & {run.index for run in self.runs}) / len(self.runs)

    def aggregate(self) -> dict[str, dict[str, float]]:
        """Mean score per criterion per condition, for a first look.

        Deliberately plain: the export carries every individual response, so
        any analysis worth trusting can be done on those rather than on an
        average computed here.
        """

        by_run = {run.index: run.condition_id for run in self.runs}
        collected: dict[str, dict[str, list[float]]] = {}
        for rating in self.ratings:
            condition_id = by_run.get(rating.run_index)
            if condition_id is None:
                continue
            for criterion, value in rating.scores.items():
                collected.setdefault(condition_id, {}).setdefault(criterion, []).append(float(value))
        return {
            condition_id: {
                criterion: float(np.mean(values)) for criterion, values in criteria.items()
            }
            for condition_id, criteria in collected.items()
        }

    def describe(self, *, reveal: bool = True) -> dict[str, Any]:
        """The full export. ``reveal=False`` withholds which label was which."""

        payload: dict[str, Any] = {
            "schemaVersion": EXPORT_SCHEMA_VERSION,
            "application": "BinauralBuilder SAM workbench",
            "applicationVersion": PACKAGE_VERSION,
            "listener": self.listener,
            "startedUtc": self.started_utc,
            "finishedUtc": self.finished_utc or _now(),
            "completion": float(self.completion),
            "design": self.design.describe(),
            "runs": [run.describe(reveal=reveal) for run in self.runs],
            "ratings": [rating.describe() for rating in self.ratings],
            "unreproducibleConditions": list(self.design.unreproducible()),
        }
        if reveal:
            payload["reveal"] = self.design.reveal
            payload["aggregate"] = self.aggregate()
        return payload


def export_result(
    result: ExperimentResult, path: str | Path, *, reveal: bool = True
) -> Path:
    """Write the export as JSON, atomically.

    Written to a temporary sibling and renamed, so an interrupted write cannot
    leave a half-written result that looks complete.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.describe(reveal=reveal)

    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_result(path: str | Path) -> dict[str, Any]:
    """Read an exported result back."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def conditions_from_assets(
    assets: Iterable[Mapping[str, Any]], *, renderer: str = "hrtf"
) -> tuple[Condition, ...]:
    """Build one condition per asset, carrying whatever provenance is known.

    Convenience for the common case: comparing several subjects with every
    other setting held constant.
    """

    conditions: list[Condition] = []
    for entry in assets:
        identifier = str(entry.get("id") or entry.get("subject") or entry.get("path") or "")
        if not identifier:
            raise ValueError("each asset needs an id, a subject, or a path")
        conditions.append(
            Condition(
                id=identifier,
                name=str(entry.get("name", entry.get("subject", identifier))),
                renderer=renderer,
                asset_path=str(entry.get("path", "")),
                asset_sha256=str(entry.get("sha256", "")),
                subject=str(entry.get("subject", "")),
                dataset=str(entry.get("dataset", "")),
                processing=str(entry.get("processing", "")),
                sample_rate_hz=float(entry.get("sampleRateHz", 48_000.0)),
            )
        )
    return tuple(conditions)

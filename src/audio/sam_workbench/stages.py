"""Named stages over the existing step timeline.

BinauralBuilder already has a timeline: steps with durations, crossfades, and
transition voices. The specification is explicit that this should be the macro
timeline rather than being replaced, and that a competing lane system added too
early would fight the step model it duplicates. So a stage here is a *label and
an envelope over existing steps*, not a second scheduler.

A stage names a span - preparation, disruption, return, whatever the user calls
them - and may fade parameters in and out across its edges. Overlapping stages
blend by weight, and everything is a pure function of time, so a render at
second 400 does not depend on having rendered second 399 first. That is what
makes a staged render deterministic and chunkable, which the existing renderer
requires.

Bindings address a stable object identifier and a schema-defined parameter
path. Never a display name, and never a list index: renaming a source or
reordering a list must not silently retarget automation onto something else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .validation import ValidationIssue

__all__ = [
    "CURVES",
    "ParameterBinding",
    "StageConfig",
    "Timeline",
    "apply_curve",
    "stages_from_steps",
]

#: Interpolation shapes a binding may use across a stage edge.
CURVES = ("hold", "linear", "smooth", "exponential")

_EPSILON = 1e-12


def apply_curve(progress: NDArray[np.floating] | float, curve: str = "linear"):
    """Shape a 0-to-1 progress value.

    ``hold`` stays at zero until the very end, which is what makes a stage
    switch abruptly rather than sweep.
    """

    values = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    if curve == "linear":
        shaped = values
    elif curve == "smooth":
        shaped = values * values * (3.0 - 2.0 * values)
    elif curve == "exponential":
        # Perceptually closer to a level fade than a straight line is.
        shaped = np.expm1(3.0 * values) / math.expm1(3.0)
    elif curve == "hold":
        shaped = (values >= 1.0).astype(np.float64)
    else:
        raise ValueError(f"unknown curve {curve!r}; expected one of {CURVES}")
    return float(shaped) if np.isscalar(progress) or np.ndim(progress) == 0 else shaped


@dataclass(frozen=True)
class ParameterBinding:
    """One parameter a stage drives, addressed by identity rather than by name."""

    target_id: str
    parameter_path: str
    value: float
    curve: str = "linear"

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("a binding needs a stable target identifier")
        if not self.parameter_path:
            raise ValueError("a binding needs a schema-defined parameter path")
        if self.curve not in CURVES:
            raise ValueError(f"unknown curve {self.curve!r}; expected one of {CURVES}")
        if not math.isfinite(float(self.value)):
            raise ValueError("a binding value must be finite")

    @property
    def key(self) -> tuple[str, str]:
        return (self.target_id, self.parameter_path)

    def describe(self) -> dict[str, Any]:
        return {
            "targetId": self.target_id,
            "parameterPath": self.parameter_path,
            "value": float(self.value),
            "curve": self.curve,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ParameterBinding":
        return cls(
            target_id=str(data["targetId"]),
            parameter_path=str(data["parameterPath"]),
            value=float(data["value"]),
            curve=str(data.get("curve", "linear")),
        )


@dataclass(frozen=True)
class StageConfig:
    """A named span of the timeline, with optional edges that fade."""

    id: str
    name: str = ""
    start_s: float = 0.0
    duration_s: float = 0.0
    transition_in_s: float = 0.0
    transition_out_s: float = 0.0
    parameter_overrides: tuple[ParameterBinding, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a stage needs a stable identifier")
        for name in ("start_s", "duration_s", "transition_in_s", "transition_out_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")

    @property
    def end_s(self) -> float:
        return float(self.start_s) + float(self.duration_s)

    def weight_at(self, seconds: NDArray[np.floating] | float):
        """How strongly this stage applies, from 0 to 1.

        The edges fade over the transition times and the middle sits at one. A
        stage shorter than its own transitions is scaled down rather than
        allowed to overshoot, so the two edges cannot sum past unity.
        """

        times = np.asarray(seconds, dtype=np.float64)
        start, end = float(self.start_s), self.end_s
        duration = max(end - start, 0.0)

        fade_in = float(self.transition_in_s)
        fade_out = float(self.transition_out_s)
        total_fade = fade_in + fade_out
        if duration > 0.0 and total_fade > duration:
            # Squeeze both edges proportionally rather than letting them cross.
            scale = duration / total_fade
            fade_in *= scale
            fade_out *= scale

        weight = np.zeros_like(times, dtype=np.float64)
        inside = (times >= start) & (times < end) if duration > 0.0 else np.zeros_like(times, bool)
        weight[inside] = 1.0

        if fade_in > _EPSILON:
            rising = inside & (times < start + fade_in)
            weight[rising] = apply_curve((times[rising] - start) / fade_in, "smooth")
        if fade_out > _EPSILON:
            falling = inside & (times >= end - fade_out)
            weight[falling] = np.minimum(
                weight[falling], apply_curve((end - times[falling]) / fade_out, "smooth")
            )

        return float(weight) if np.ndim(seconds) == 0 else weight

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "startS": float(self.start_s),
            "durationS": float(self.duration_s),
            "transitionInS": float(self.transition_in_s),
            "transitionOutS": float(self.transition_out_s),
            "parameterOverrides": [binding.describe() for binding in self.parameter_overrides],
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StageConfig":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            start_s=float(data.get("startS", 0.0)),
            duration_s=float(data.get("durationS", 0.0)),
            transition_in_s=float(data.get("transitionInS", 0.0)),
            transition_out_s=float(data.get("transitionOutS", 0.0)),
            parameter_overrides=tuple(
                ParameterBinding.from_mapping(entry)
                for entry in data.get("parameterOverrides", ())
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Timeline:
    """An ordered set of stages over one session."""

    stages: tuple[StageConfig, ...] = ()
    schema_version: int = 1

    @property
    def duration_s(self) -> float:
        return max((stage.end_s for stage in self.stages), default=0.0)

    def stage(self, stage_id: str) -> StageConfig | None:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        return None

    def validate(self) -> tuple[ValidationIssue, ...]:
        """Structural problems, and overlaps worth mentioning.

        Overlap is a warning rather than an error: crossfading one stage into
        the next is a normal thing to want, and the blend below handles it.
        """

        issues: list[ValidationIssue] = []
        seen: set[str] = set()
        for position, stage in enumerate(self.stages):
            path = f"stages[{position}]"
            if stage.id in seen:
                issues.append(ValidationIssue(f"{path}.id", f"duplicate stage id {stage.id!r}"))
            seen.add(stage.id)
            if stage.duration_s <= 0.0:
                issues.append(
                    ValidationIssue(f"{path}.durationS", "a stage with no duration never applies", "warning")
                )
            if stage.transition_in_s + stage.transition_out_s > stage.duration_s > 0.0:
                issues.append(
                    ValidationIssue(
                        f"{path}.transitionInS",
                        "the transitions are longer than the stage; they will be scaled to fit",
                        "warning",
                    )
                )

        ordered = sorted(self.stages, key=lambda stage: stage.start_s)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.start_s < earlier.end_s:
                issues.append(
                    ValidationIssue(
                        "stages",
                        f"{earlier.id!r} and {later.id!r} overlap; their bindings will blend",
                        "warning",
                    )
                )
        return tuple(issues)

    def weights_at(self, seconds: float) -> dict[str, float]:
        """Each stage's weight at one instant."""

        return {stage.id: float(stage.weight_at(float(seconds))) for stage in self.stages}

    def resolve(
        self, seconds: float, *, defaults: Mapping[tuple[str, str], float] | None = None
    ) -> dict[tuple[str, str], float]:
        """The parameter values in force at one instant.

        Where stages overlap, their bindings blend by weight. A parameter with a
        default blends from that default, so a stage fading in moves a value
        away from where it was rather than jumping to its own.
        """

        defaults = dict(defaults or {})
        totals: dict[tuple[str, str], float] = {}
        weights: dict[tuple[str, str], float] = {}

        for stage in self.stages:
            weight = float(stage.weight_at(float(seconds)))
            if weight <= _EPSILON:
                continue
            for binding in stage.parameter_overrides:
                shaped = weight if binding.curve == "linear" else apply_curve(weight, binding.curve)
                totals[binding.key] = totals.get(binding.key, 0.0) + shaped * binding.value
                weights[binding.key] = weights.get(binding.key, 0.0) + shaped

        resolved: dict[tuple[str, str], float] = {}
        for key, total in totals.items():
            weight = weights[key]
            if weight <= _EPSILON:
                continue
            if weight >= 1.0:
                resolved[key] = total / weight
            else:
                # Short of full weight the remainder comes from the default,
                # which is what makes a fade-in start from where things were.
                base = float(defaults.get(key, 0.0))
                resolved[key] = total + (1.0 - weight) * base
        return resolved

    def describe(self) -> dict[str, Any]:
        return {
            "schemaVersion": int(self.schema_version),
            "stages": [stage.describe() for stage in self.stages],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Timeline":
        values = dict(data or {})
        return cls(
            stages=tuple(StageConfig.from_mapping(entry) for entry in values.get("stages", ())),
            schema_version=int(values.get("schemaVersion", 1)),
        )

    def with_stage(self, stage: StageConfig) -> "Timeline":
        return replace(self, stages=self.stages + (stage,))


def stages_from_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    groups: Iterable[Sequence[int]] | None = None,
    names: Sequence[str] | None = None,
    transition_s: float = 0.0,
) -> Timeline:
    """Build stages from the existing step list.

    This is the mapping the specification asks for: a stage is a step, or a
    contiguous group of steps, given a name. Timing comes from the steps
    themselves, so the stage view and the step view cannot disagree about when
    anything happens.

    ``groups`` names contiguous index runs; without it every step becomes its
    own stage.
    """

    starts: list[float] = []
    clock = 0.0
    for step in steps:
        starts.append(clock)
        clock += float(step.get("duration", 0.0) or 0.0)

    if groups is None:
        spans = [(index,) for index in range(len(steps))]
    else:
        spans = [tuple(int(index) for index in group) for group in groups]

    stages: list[StageConfig] = []
    for position, span in enumerate(spans):
        if not span:
            continue
        indices = sorted(span)
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"a stage must cover contiguous steps, got {span!r}")
        if indices[-1] >= len(steps):
            raise ValueError(f"step index {indices[-1]} is beyond the {len(steps)} steps given")

        start = starts[indices[0]]
        duration = sum(float(steps[index].get("duration", 0.0) or 0.0) for index in indices)
        default_name = str(steps[indices[0]].get("description", "") or f"Stage {position + 1}")
        name = names[position] if names is not None and position < len(names) else default_name
        stages.append(
            StageConfig(
                id=f"stage-{position + 1}",
                name=name,
                start_s=start,
                duration_s=duration,
                transition_in_s=transition_s if position else 0.0,
                transition_out_s=transition_s if position < len(spans) - 1 else 0.0,
            )
        )
    return Timeline(stages=tuple(stages))

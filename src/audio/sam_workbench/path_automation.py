"""Reserved path-parameter automation, shared by both front doors.

A modulation route or stage override may target a path's own numbers -
``path.radiusM`` for a geometry field, ``transform.yawDeg`` for the whole-path
transform.  Two doors lead from a saved document to a rendered source, and
both must interpret such a target identically: the compiled scene plan, and
the per-voice compatibility adapter.  This module is the one interpretation.

The contract matches every other automatable parameter.  A control compiled
here yields the parameter's effective value over an absolute window - stages
blend from the stored base, modulation routes add to it - and
:func:`~.trajectory.dynamic.bind_path_parameters` turns those values into
positions behind the same callable renderers already consume.

The clock mapping is the one :func:`.compat._scene_automation` documents: the
renderer's trajectory clock counts seconds since its source began sounding,
and ``origin_sample`` places sample zero of that clock on the scene's absolute
timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .trajectory.dynamic import (
    PathBinding,
    bind_path_parameters,
    path_parameter_bases,
)
from .trajectory.parameter_catalog import PATH_PREFIX, TRANSFORM_PREFIX, split_parameter_path
from .validation import ValidationIssue

__all__ = [
    "RUNTIME_BOUND_POSITIONS",
    "is_reserved_path",
    "reserved_targets",
    "BoundTrajectory",
    "compile_bound_trajectory",
]


#: Voice-parameter key carrying the pre-bound positions callable from
#: :func:`compile_bound_trajectory` into a per-voice renderer. It exists only
#: inside a render call's parameter copy; it is never serialized and never
#: reaches a manifest, which reads the original document.
RUNTIME_BOUND_POSITIONS = "_boundPositions"


def is_reserved_path(parameter_path: str) -> bool:
    text = str(parameter_path)
    return text.startswith(PATH_PREFIX) or text.startswith(TRANSFORM_PREFIX)


def reserved_targets(scene: Mapping[str, Any] | None, source_id: str) -> tuple[str, ...]:
    """Which reserved paths this scene automates for one source."""

    if not scene:
        return ()
    from .scene_state import automated_paths

    return tuple(
        path for path in automated_paths(scene, str(source_id)) if is_reserved_path(path)
    )


class _SceneSeriesControl:
    """One reserved parameter's series over absolute windows."""

    __slots__ = ("_scene", "_source_id", "_rate", "_origin", "_base", "_path")

    def __init__(
        self,
        scene: Mapping[str, Any],
        source_id: str,
        sample_rate_hz: float,
        origin_sample: int,
        base: Mapping[str, Any],
        parameter_path: str,
    ) -> None:
        self._scene = scene
        self._source_id = str(source_id)
        self._rate = float(sample_rate_hz)
        self._origin = int(origin_sample)
        self._base = dict(base)
        self._path = str(parameter_path)

    def at(self, start_sample: int, frames: int) -> np.ndarray:
        from .scene_state import scene_parameter_series

        series = scene_parameter_series(
            self._scene,
            self._source_id,
            int(self._origin) + int(start_sample),
            int(frames),
            self._rate,
            self._base,
        )
        if self._path in series:
            return np.asarray(series[self._path], dtype=np.float64)
        # Outside every stage and route the parameter is simply its stored
        # base, which the compiler seeded into ``_base`` for exactly this.
        return np.full(int(frames), float(self._base.get(self._path, 0.0)))


@dataclass(frozen=True)
class BoundTrajectory:
    """What a document's path means once its automation is attached."""

    #: ``PathModel`` when unbound, ``ModulatedPath`` when bound, ``None`` when
    #: the voice carries no readable path at all.
    model: Any = None
    #: Reserved paths that were bound successfully.
    matched: frozenset[str] = frozenset()
    #: Findings for readiness and the compile report.
    issues: tuple[ValidationIssue, ...] = ()
    #: Bound parameters whose resolved motion touches a documented limit.
    clamped: tuple[str, ...] = ()
    #: True when shape motion forced constant-speed traversal to give way.
    speed_law_fallback: bool = False


def compile_bound_trajectory(
    payload: Any,
    scene: Mapping[str, Any] | None,
    source_id: str,
    *,
    sample_rate_hz: float,
    origin_sample: int,
    params: Mapping[str, Any] | None = None,
) -> BoundTrajectory:
    """Attach this scene's path automation to a saved trajectory.

    The returned model is what renderers consume; ``matched`` names the
    reserved paths taken over here, so the generic control compiler can leave
    them alone.
    """

    model = None
    issues: list[ValidationIssue] = []
    if isinstance(payload, Mapping) and payload.get("geometry"):
        from .trajectory import path_model_from_dict

        try:
            model = path_model_from_dict(payload)
        except (ValueError, TypeError, KeyError) as error:
            return BoundTrajectory(
                issues=(
                    ValidationIssue(
                        "canonicalTrajectory", f"unreadable path: {error}"
                    ),
                )
            )

    targets = reserved_targets(scene, source_id)
    if not targets or model is None:
        return BoundTrajectory(model=model)

    bases = path_parameter_bases(model)
    bindings: list[PathBinding] = []
    matched: set[str] = set()
    unknown: list[str] = []
    for parameter_path in sorted(targets):
        try:
            section, field_name = split_parameter_path(parameter_path)
        except ValueError as error:
            issues.append(ValidationIssue(f"automation.{parameter_path}", str(error)))
            continue
        if f"{PATH_PREFIX if section == 'geometry' else TRANSFORM_PREFIX}{field_name}" not in bases:
            unknown.append(parameter_path)
            continue
        base = dict(params or {})
        for key, value in bases.items():
            base.setdefault(key, value)
        control = _SceneSeriesControl(
            scene, source_id, sample_rate_hz, origin_sample, base, parameter_path
        )
        canonical = (
            f"{PATH_PREFIX if section == 'geometry' else TRANSFORM_PREFIX}{field_name}"
        )
        try:
            binding = PathBinding(canonical, section, field_name, control)
            bound = bind_path_parameters(
                model,
                [binding],
                sample_rate_hz=sample_rate_hz,
                origin_sample=int(origin_sample),
            )
        except ValueError as error:
            issues.append(
                ValidationIssue(f"automation.{parameter_path}", str(error))
            )
            continue
        model = bound
        matched.add(canonical)

    for parameter_path in unknown:
        issues.append(
            ValidationIssue(
                f"automation.{parameter_path}",
                f"{parameter_path!r} is not a parameter of this path's "
                "geometry, so its automation will not be applied",
            )
        )

    clamped: list[str] = []
    speed_law_fallback = False
    if matched and hasattr(model, "sample_parameter"):
        duration_s = max(float(model.duration_s), 1e-6)
        times = np.linspace(0.0, duration_s, 2048)
        for parameter_path in sorted(matched):
            spec = _spec_for_path(parameter_path)
            values = model.sample_parameter(parameter_path, times)
            if values is None or values.size == 0 or spec is None:
                continue
            minimum, maximum = spec.minimum, spec.maximum
            eps = 1e-9
            touched = (
                (minimum is not None and bool(np.any(values <= minimum + eps)))
                or (maximum is not None and bool(np.any(values >= maximum - eps)))
            )
            if touched:
                clamped.append(parameter_path)
        speed_law_fallback = not model.uses_constant_speed_law

    return BoundTrajectory(
        model=model,
        matched=frozenset(matched),
        issues=tuple(issues),
        clamped=tuple(clamped),
        speed_law_fallback=speed_law_fallback,
    )


def _spec_for_path(parameter_path: str):
    from .trajectory.parameter_catalog import spec_for_field

    _section, field_name = split_parameter_path(parameter_path)
    return spec_for_field(field_name)

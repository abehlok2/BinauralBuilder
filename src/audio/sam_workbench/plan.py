"""The canonical compiled scene: one immutable, sample-accurate description.

Preview, export, the HRTF lab, analysis, experiments and benchmarking each used
to work out for themselves what a project meant - where a source starts, which
renderer it uses, whether its configuration is valid, which asset it needs,
what its automation is worth at a given moment. Six answers to one question is
five chances to disagree, and a disagreement between preview and export is the
one a user notices last and trusts least.

This module compiles a project once into a :class:`CompiledScenePlan`: an
immutable, Qt-free value describing exactly what is to be rendered, in absolute
samples. Nothing edits a plan and nothing saves one. It is derived, so throwing
it away and compiling again is always correct.

Where projects come from
------------------------
Two front doors lead here, and the rule between them is stated in
:mod:`.scene_state`: the BinauralBuilder **track dictionary** is the persisted
document, and :class:`~.model.Project` is the standalone SAM API. Neither is
"the real one". :func:`plan_from_track` and :func:`plan_from_project` are the
two adapters, and they produce the same type, so everything downstream consumes
one shape regardless of which door a project came through.

What is not in here
-------------------
A trajectory is stored as its :class:`~.trajectory.PathModel`, never as a
materialized ``(frames, 3)`` array. A path is a function of time; sampling it
at the audio rate would cost megabytes per minute per source to answer a
question every renderer prefers to ask at its own control rate. The plan
carries the model and lets each renderer sample it as it needs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .conventions import DEFAULT_SAMPLE_RATE_HZ, intersect_window, seconds_to_samples
from .parameters import sam2_parameter_defaults
from .path_automation import compile_bound_trajectory, is_reserved_path
from .render.registry import REGISTRY
from .scene_state import (
    AMPLITUDE_PATHS,
    SOURCE_ID_KEY,
    assign_source_ids,
    automated_paths,
    iter_sam_voices,
    normalize_sam_scene,
    scene_gain_envelope,
    scene_parameter_series,
    validate_scene,
)
from .stages import Timeline
from .modulation import ModulationMatrix
from .trajectory import PathModel
from .trajectory.transforms import ListenerTransform
from .validation import ValidationIssue

__all__ = [
    "AssetReference",
    "CompiledControl",
    "CompiledSourcePlan",
    "CompiledScenePlan",
    "intersect_window",
    "plan_from_track",
    "plan_from_project",
]


@dataclass(frozen=True)
class AssetReference:
    """An external file a render depends on, and what it hashed to."""

    key: str
    kind: str
    path: str
    sha256: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class CompiledControl:
    """One parameter's value as a function of the absolute sample index.

    A control is compiled, not sampled: it holds the rule, and produces values
    for whatever window is asked for. That is what makes a render independent
    of how it was divided into blocks - which the previous scene evaluation was
    not, because it resolved every non-gain parameter once at the start of each
    chunk and held it for the chunk's length.
    """

    path: str
    #: Set when the value never varies, so a caller can skip the array.
    constant: float | None = None
    automatable: bool = True
    _evaluate: Callable[[int, int], NDArray[np.float64]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_constant(self) -> bool:
        return self._evaluate is None

    def at(self, start_sample: int, frames: int) -> NDArray[np.float64]:
        """This control's value over ``frames`` samples from ``start_sample``."""

        if frames <= 0:
            return np.zeros(0, dtype=np.float64)
        if self._evaluate is None:
            return np.full(int(frames), float(self.constant or 0.0), dtype=np.float64)
        values = np.asarray(self._evaluate(int(start_sample), int(frames)), dtype=np.float64)
        if values.shape != (int(frames),):
            raise ValueError(
                f"control {self.path!r} produced {values.shape}, expected ({frames},)"
            )
        return values

    def value_at(self, sample: int) -> float:
        """The single value at one absolute sample."""

        return float(self.at(int(sample), 1)[0])

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "constant": self.constant,
            "automated": not self.is_constant,
            "automatable": self.automatable,
        }


@dataclass(frozen=True)
class CompiledSourcePlan:
    """One source, fully resolved against the absolute timeline."""

    source_id: str
    name: str
    enabled: bool
    start_sample: int
    #: ``None`` for a source with no declared duration, which runs as long as
    #: the render asks for.
    end_sample: int | None
    #: The generator's own parameters, after scene overrides that do not vary.
    generator: Mapping[str, Any]
    renderer_id: str
    renderer_config: Mapping[str, Any]
    #: A ``PathModel``, or its automation-bound form when the scene drives the
    #: path's own numbers; both answer ``positions(times_s)`` identically.
    trajectory: Any | None = None
    listener: ListenerTransform = field(default_factory=ListenerTransform)
    assets: tuple[AssetReference, ...] = ()
    #: Automated non-gain parameters, by parameter path.
    controls: Mapping[str, CompiledControl] = field(default_factory=dict)
    #: The scene's audible level for this source, always present.
    gain: CompiledControl = field(default_factory=lambda: CompiledControl("gain", 1.0))
    bus_id: str = "master"
    seed: int = 0
    latency_samples: int = 0
    tail_samples: int = 0

    @property
    def frames(self) -> int | None:
        """The source's own length, or ``None`` when it has no declared end."""

        return None if self.end_sample is None else self.end_sample - self.start_sample

    def window(self, start_sample: int, frames: int) -> tuple[int, int, int] | None:
        """This source's overlap with a render window; see :func:`intersect_window`."""

        return intersect_window(self.start_sample, self.end_sample, start_sample, frames)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "name": self.name,
            "enabled": self.enabled,
            "startSample": self.start_sample,
            "endSample": self.end_sample,
            "renderer": self.renderer_id,
            "rendererConfig": dict(self.renderer_config),
            "hasTrajectory": self.trajectory is not None,
            "assets": [asset.describe() for asset in self.assets],
            "controls": [control.describe() for control in self.controls.values()],
            "busId": self.bus_id,
            "seed": self.seed,
            "latencySamples": self.latency_samples,
            "tailSamples": self.tail_samples,
        }


@dataclass(frozen=True)
class CompiledScenePlan:
    """Everything a render needs, in absolute samples, and nothing it does not."""

    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    start_sample: int = 0
    frames: int = 0
    sources: tuple[CompiledSourcePlan, ...] = ()
    stages: Timeline = field(default_factory=Timeline)
    modulators: tuple[Mapping[str, Any], ...] = ()
    modulation: ModulationMatrix = field(default_factory=ModulationMatrix)
    buses: tuple[Mapping[str, Any], ...] = ()
    #: Per-source routing entries, keyed by source identifier.
    routing: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Multiband routing configuration, carried for Phase 2's signal path.
    band_routing: Mapping[str, Any] = field(default_factory=dict)
    master_gain_db: float = 0.0
    limiter_enabled: bool = True
    limiter_ceiling_dbfs: float = -1.0
    #: The project's seed. Per-source seeds are derived from it and the source
    #: identifier, so adding a source cannot change another source's noise.
    seed: int = 0
    environment: Mapping[str, Any] = field(default_factory=dict)
    experiment: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[ValidationIssue, ...] = ()

    # --- derived ------------------------------------------------------------

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.frames

    @property
    def latency_samples(self) -> int:
        """The largest latency any source introduces; the mix waits for it."""

        return max((source.latency_samples for source in self.sources), default=0)

    @property
    def tail_samples(self) -> int:
        """How far past the last source a render must continue to avoid a cut."""

        return max((source.tail_samples for source in self.sources), default=0)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.warnings if issue.severity == "error")

    @property
    def is_renderable(self) -> bool:
        return not self.errors

    def source(self, source_id: str) -> CompiledSourcePlan | None:
        for entry in self.sources:
            if entry.source_id == source_id:
                return entry
        return None

    def active_sources(self, start_sample: int, frames: int) -> tuple[CompiledSourcePlan, ...]:
        """The sources that actually overlap a window, in plan order."""

        return tuple(
            source
            for source in self.sources
            if source.enabled and source.window(start_sample, frames) is not None
        )

    def for_window(self, start_sample: int, frames: int) -> "CompiledScenePlan":
        """The same plan restricted to a different absolute window.

        The sources keep their own absolute positions; only the render range
        moves. A preview seeking is asking for this, not for a new compile.
        """

        return replace(self, start_sample=int(start_sample), frames=int(frames))

    def renderer_ids(self) -> tuple[str, ...]:
        return tuple(sorted({source.renderer_id for source in self.sources}))

    def assets(self) -> tuple[AssetReference, ...]:
        """Every distinct asset the plan depends on, for a render manifest."""

        seen: dict[tuple[str, str], AssetReference] = {}
        for source in self.sources:
            for asset in source.assets:
                seen.setdefault((asset.kind, asset.path), asset)
        return tuple(seen.values())

    def describe(self) -> dict[str, Any]:
        """A JSON-compatible summary, for a manifest or a diagnostics view."""

        return {
            "sampleRateHz": int(self.sample_rate_hz),
            "startSample": int(self.start_sample),
            "frames": int(self.frames),
            "latencySamples": self.latency_samples,
            "tailSamples": self.tail_samples,
            "seed": int(self.seed),
            "masterGainDb": float(self.master_gain_db),
            "limiterEnabled": bool(self.limiter_enabled),
            "limiterCeilingDbfs": float(self.limiter_ceiling_dbfs),
            "renderers": list(self.renderer_ids()),
            "sources": [source.describe() for source in self.sources],
            "buses": [dict(bus) for bus in self.buses],
            "bandRouting": dict(self.band_routing),
            "environment": dict(self.environment),
            "experiment": dict(self.experiment),
            "assets": [asset.describe() for asset in self.assets()],
            "diagnostics": dict(self.diagnostics),
            "warnings": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.warnings
            ],
        }


# --- compilation ------------------------------------------------------------


def _derive_seed(project_seed: int, source_id: str) -> int:
    """A per-source seed that does not move when another source is added.

    Deriving it from the identifier rather than from a counter is what keeps a
    noise source reproducible while the scene around it is edited.
    """

    digest = hashlib.sha256(f"{int(project_seed)}:{source_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _trajectory_from(
    params: Mapping[str, Any],
    scene: Mapping[str, Any] | None,
    source_id: str,
    *,
    sample_rate_hz: float,
    origin_sample: int,
    path: str,
    warnings: list[ValidationIssue],
):
    """Compile the stored path plus its scene automation into one model.

    A voice's ``canonicalTrajectory`` is read through the same
    :func:`path_model_from_dict` the editor and the renderers use, so a saved
    path means one thing everywhere.  Routes or stages targeting the path's own
    numbers are attached here rather than interpreted a second time elsewhere;
    a malformed path is reported and dropped rather than silently replaced by
    a default, which would move the source.
    """

    bound = compile_bound_trajectory(
        params.get("canonicalTrajectory"),
        scene,
        source_id,
        sample_rate_hz=sample_rate_hz,
        origin_sample=origin_sample,
        params=params,
    )
    for issue in bound.issues:
        warnings.append(ValidationIssue(f"{path}.{issue.path}", issue.message, issue.severity))
    return bound.model


def _compile_controls(
    scene: Mapping[str, Any] | None,
    source_id: str,
    base: Mapping[str, Any],
    sample_rate_hz: float,
    path: str,
    warnings: list[ValidationIssue],
) -> dict[str, CompiledControl]:
    """Turn a scene's automation into per-parameter functions of absolute time.

    Only paths that actually vary become controls; the rest stay in the
    generator parameters as the constants they are.
    """

    if not scene:
        return {}
    from .parameters import field_for

    controls: dict[str, CompiledControl] = {}
    for parameter_path in automated_paths(scene, source_id):
        if is_reserved_path(parameter_path):
            # Path parameters belong to the bound-trajectory compiler; one
            # that failed to bind there was already reported there.
            continue
        registered = field_for(parameter_path)
        automatable = True if registered is None else bool(registered.automatable)
        if not automatable:
            warnings.append(
                ValidationIssue(
                    f"{path}.automation.{parameter_path}",
                    f"{parameter_path!r} is not an automatable parameter; "
                    "its automation will not be applied",
                    "warning",
                )
            )
            continue

        def evaluate(
            start_sample: int, frames: int, _path: str = parameter_path
        ) -> NDArray[np.float64]:
            series = scene_parameter_series(
                scene, source_id, start_sample, frames, sample_rate_hz, base
            )
            if _path in series:
                return np.asarray(series[_path], dtype=np.float64)
            # Outside every stage and route the parameter is simply itself.
            return np.full(int(frames), float(base.get(_path, 0.0) or 0.0), dtype=np.float64)

        controls[parameter_path] = CompiledControl(
            path=parameter_path, automatable=True, _evaluate=evaluate
        )
    return controls


def _compile_gain(
    scene: Mapping[str, Any] | None, source_id: str, sample_rate_hz: float
) -> CompiledControl:
    """The source's own audible level, without its routing.

    Routing must be applied exactly once. This gain is what stages and
    modulation do to a source; its bus gain, mute and solo belong to the mixer,
    which is where there is an actual bus to meter and to process. The plan
    carries the routing separately for exactly that reason, so folding it in
    here as well would attenuate every routed source twice - a -6 dB bus
    arriving as -12 dB.
    """

    if not scene:
        return CompiledControl("gain", constant=1.0)

    def evaluate(start_sample: int, frames: int) -> NDArray[np.float64]:
        return scene_gain_envelope(
            scene, source_id, start_sample, frames, sample_rate_hz,
            include_routing=False,
        )

    return CompiledControl("gain", automatable=True, _evaluate=evaluate)


def _compile_source(
    *,
    source_id: str,
    name: str,
    enabled: bool,
    params: Mapping[str, Any],
    start_sample: int,
    end_sample: int | None,
    scene: Mapping[str, Any] | None,
    sample_rate_hz: int,
    listener: ListenerTransform,
    project_seed: int,
    routing: Mapping[str, Any],
    path: str,
    warnings: list[ValidationIssue],
) -> CompiledSourcePlan:
    renderer_id = str(params.get("rendererMode", "abstract_pm")).lower()
    if renderer_id not in REGISTRY:
        warnings.append(
            ValidationIssue(
                f"{path}.rendererMode",
                f"{renderer_id!r} is not a renderer in this build; "
                f"expected one of {', '.join(REGISTRY.identifiers)}",
            )
        )
        definition = REGISTRY.get("abstract_pm")
        renderer_id = "abstract_pm"
    else:
        definition = REGISTRY.get(renderer_id)

    for issue in definition.validate(params):
        warnings.append(ValidationIssue(f"{path}.{issue.path}", issue.message, issue.severity))
    config = definition.config_from(params)

    base = dict(params)
    controls = _compile_controls(scene, source_id, base, sample_rate_hz, path, warnings)
    generator = {
        key: value for key, value in base.items() if key not in controls
    }

    return CompiledSourcePlan(
        source_id=source_id,
        name=name,
        enabled=bool(enabled),
        start_sample=int(start_sample),
        end_sample=None if end_sample is None else int(end_sample),
        generator=generator,
        renderer_id=renderer_id,
        renderer_config=config,
        trajectory=_trajectory_from(
            params,
            scene,
            source_id,
            sample_rate_hz=float(sample_rate_hz),
            origin_sample=int(start_sample),
            path=path,
            warnings=warnings,
        ),
        listener=listener,
        assets=tuple(
            AssetReference(**entry) for entry in definition.required_assets(params)
        ),
        controls=controls,
        gain=_compile_gain(scene, source_id, sample_rate_hz),
        bus_id=str(routing.get("busId", "master") or "master"),
        seed=_derive_seed(project_seed, source_id),
        latency_samples=definition.latency_samples(params, sample_rate_hz),
        tail_samples=definition.tail_samples(params, sample_rate_hz),
    )


def _scene_sections(scene: Mapping[str, Any] | None):
    """The shared parts of a scene, compiled once for the whole plan."""

    if not scene:
        return Timeline(), (), ModulationMatrix(), (), {}, {}, {}, {}
    routing = dict(scene.get("routing") or {})
    return (
        Timeline.from_mapping(scene.get("stages")),
        tuple(dict(item) for item in scene.get("modulators", ())),
        ModulationMatrix.from_mapping(scene.get("modulation")),
        tuple(dict(bus) for bus in routing.get("buses", ())),
        {
            str(entry.get("sourceId")): dict(entry)
            for entry in routing.get("sources", ())
            if entry.get("sourceId")
        },
        dict(routing.get("bands") or {}),
        dict(scene.get("environment") or {}),
        dict(scene.get("experiment") or {}),
    )


def plan_from_track(
    track_data: Mapping[str, Any],
    *,
    start_sample: int = 0,
    frames: int | None = None,
    sample_rate_hz: int | None = None,
    listener: ListenerTransform | None = None,
) -> CompiledScenePlan:
    """Compile a BinauralBuilder track and its ``sam_scene`` into one plan.

    ``track_data`` is read, not edited, apart from the stable source
    identifiers :func:`assign_source_ids` maintains - those belong in the track
    and exist precisely so that a compile does not have to invent them.
    """

    warnings: list[ValidationIssue] = []
    mutable = dict(track_data)
    settings = dict(mutable.get("global_settings") or {})
    rate = int(sample_rate_hz or settings.get("sample_rate") or DEFAULT_SAMPLE_RATE_HZ)

    scene_present = mutable.get("sam_scene") is not None
    scene = assign_source_ids(mutable, persist_scene=scene_present) if scene_present else None
    if scene is not None:
        scene = normalize_sam_scene(scene)
        warnings.extend(validate_scene(scene, mutable.get("steps")))
    else:
        assign_source_ids(mutable, persist_scene=False)

    stages, modulators, matrix, buses, routing, bands, environment, experiment = _scene_sections(scene)
    pose = listener or ListenerTransform()
    project_seed = int(settings.get("random_seed", 0) or 0)

    sources: list[CompiledSourcePlan] = []
    running_start = 0.0
    for position, step in enumerate(mutable.get("steps") or ()):
        step_start = float(step.get("start", step.get("start_time", running_start)) or 0.0)
        step_duration = float(step.get("duration", 0.0) or 0.0)
        for index, voice in enumerate(step.get("voices", ()) or ()):
            if voice not in [entry for _, entry in iter_sam_voices([step])]:
                continue
            params = dict(voice.get("params") or {})
            source_id = str(voice.get(SOURCE_ID_KEY, "")).strip() or f"source.{len(sources) + 1}"
            # The step governs a voice's length. A ``duration`` in the voice's
            # own parameters is the legacy transition-span field, which
            # generate_voice_audio strips before calling the synth precisely so
            # that the step's duration is the one that applies.
            duration = step_duration
            start = seconds_to_samples(step_start, rate)
            sources.append(
                _compile_source(
                    source_id=source_id,
                    name=str(voice.get("description", "") or source_id),
                    enabled=not bool(voice.get("muted", False)),
                    params=params,
                    start_sample=start,
                    end_sample=start + seconds_to_samples(duration, rate) if duration > 0.0 else None,
                    scene=scene,
                    sample_rate_hz=rate,
                    listener=pose,
                    project_seed=project_seed,
                    routing=routing.get(source_id, {}),
                    path=f"steps[{position}].voices[{index}]",
                    warnings=warnings,
                )
            )
        running_start = step_start + step_duration

    span = frames
    if span is None:
        latest = max(
            (source.end_sample or source.start_sample for source in sources), default=0
        )
        span = max(0, latest - int(start_sample))

    return CompiledScenePlan(
        sample_rate_hz=rate,
        start_sample=int(start_sample),
        frames=int(span),
        sources=tuple(sources),
        stages=stages,
        modulators=modulators,
        modulation=matrix,
        buses=buses,
        routing=routing,
        band_routing=bands,
        master_gain_db=float(settings.get("master_gain_db", 0.0) or 0.0),
        limiter_enabled=bool(settings.get("limiter_enabled", True)),
        limiter_ceiling_dbfs=float(settings.get("limiter_ceiling_dbfs", -1.0)),
        seed=project_seed,
        environment=environment,
        experiment=experiment,
        diagnostics={"origin": "track", "sceneSchema": (scene or {}).get("schemaVersion")},
        warnings=tuple(warnings),
    )


def plan_from_project(
    project,
    *,
    start_sample: int = 0,
    frames: int | None = None,
) -> CompiledScenePlan:
    """Compile the standalone :class:`~.model.Project` document into one plan.

    The same type comes out as from a track, which is the point: nothing
    downstream should need to know which front door a project came through.
    """

    warnings: list[ValidationIssue] = []
    rate = int(project.audio.sample_rate_hz)
    pose = ListenerTransform(
        position_m=tuple(project.listener.position_m),
        yaw_pitch_roll_deg=(
            float(project.listener.yaw_deg),
            float(project.listener.pitch_deg),
            float(project.listener.roll_deg),
        ),
    )

    sources: list[CompiledSourcePlan] = []
    for position, source in enumerate(project.sources):
        params = dict(source.extras)
        params.setdefault("rendererMode", "abstract_pm")
        params.setdefault("amp", float(source.amplitude_linear))
        params.setdefault("carrierFreq", float(source.signal.carrier_frequency_hz))
        start = seconds_to_samples(source.start_s, rate)
        end = (
            None
            if source.duration_s is None
            else start + seconds_to_samples(source.duration_s, rate)
        )
        sources.append(
            _compile_source(
                source_id=str(source.id),
                name=str(source.name),
                enabled=bool(source.enabled),
                params=params,
                start_sample=start,
                end_sample=end,
                scene=None,
                sample_rate_hz=rate,
                listener=pose,
                project_seed=int(project.audio.random_seed),
                routing={},
                path=f"sources[{position}]",
                warnings=warnings,
            )
        )

    span = frames
    if span is None:
        latest = max((entry.end_sample or entry.start_sample for entry in sources), default=0)
        span = max(0, latest - int(start_sample))

    return CompiledScenePlan(
        sample_rate_hz=rate,
        start_sample=int(start_sample),
        frames=int(span),
        sources=tuple(sources),
        master_gain_db=float(project.output.master_gain_db),
        limiter_enabled=bool(project.output.limiter_enabled),
        limiter_ceiling_dbfs=float(project.output.limiter_ceiling_dbfs),
        seed=int(project.audio.random_seed),
        diagnostics={"origin": "project", "schemaVersion": project.schema_version},
        warnings=tuple(warnings),
    )

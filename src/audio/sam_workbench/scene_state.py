"""Versioned track-level scene state shared by every SAM source.

Scene controls used to be stored under three keys in an individual voice.  A
voice renderer could consequently neither see the other sources nor know which
copy was authoritative.  This module defines the compatibility envelope used at
the track boundary and keeps migration explicit.

Ownership
---------
There are three things that could each be mistaken for "the scene", so the rule
between them is written down here rather than left to be inferred:

* the **track dictionary** (``track_data["sam_scene"]``) is the *persisted*
  scene.  It is what a user's file contains and the only thing that survives a
  save; everything else is derived from it.
* :mod:`.model`'s ``Project`` is the *standalone* SAM API - the typed document
  the command line and the tests build directly, without a BinauralBuilder
  track.  It is not a second scene format: it is a different front door, and
  :mod:`.plan` adapts either one into the same compiled form.
* :class:`~.plan.CompiledScenePlan` is the *compiled* scene.  It is immutable,
  sample-accurate and derived; nothing edits it and nothing saves it.

So: edit the track dictionary, compile the plan, render the plan.  Neither of
the other two is allowed to become a third authority.

Identity
--------
A source's identifier has to outlive editing.  It is stored on the voice, in
the track, as ``sam_source_id`` - not derived from list position, because
reordering steps or voices would then silently reassign every automation route
and mute to a different source.  :func:`assign_source_ids` writes those
identifiers into the real track; :func:`ensure_source_ids` remains for callers
that hold only a render copy.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .modulation import ModulationMatrix
from .stages import Timeline
from .validation import ValidationIssue

#: Version 2 adds the assets, environment, experiment and extensions sections.
#: Version 1 documents remain readable and are migrated on load.
SAM_SCENE_SCHEMA_VERSION = 2
LEGACY_SCENE_KEYS = ("samStages", "samModulation", "samRouting")

#: The voice key that carries a source's stable identifier in the track.
SOURCE_ID_KEY = "sam_source_id"

#: Voices this module considers scene sources.
SAM_SYNTH_MARKER = "spatial_angle_modulation_sam2"

#: Parameter paths that mean "the source's own level". Gain is special because
#: it is the one target the compatibility renderer can already apply per sample.
AMPLITUDE_PATHS = frozenset({"amp", "amplitude", "amplitude_linear", "signal.amplitude"})

#: Target identifiers that mean "whichever source is asking".
WILDCARD_TARGETS = frozenset({"voice", "*"})

__all__ = [
    "SAM_SCENE_SCHEMA_VERSION",
    "LEGACY_SCENE_KEYS",
    "SOURCE_ID_KEY",
    "AMPLITUDE_PATHS",
    "WILDCARD_TARGETS",
    "empty_sam_scene",
    "normalize_sam_scene",
    "migrate_voice_scene",
    "migrate_scene",
    "ensure_source_ids",
    "assign_source_ids",
    "iter_sam_voices",
    "validate_scene",
    "scene_gain_envelope",
    "scene_parameter_overrides",
    "scene_parameter_series",
    "modulator_series",
    "automated_paths",
]


# --- shape ------------------------------------------------------------------


def empty_sam_scene() -> dict[str, Any]:
    """A complete scene with every section present and nothing configured."""

    return {
        "schemaVersion": SAM_SCENE_SCHEMA_VERSION,
        "sources": [],
        "stages": Timeline().describe(),
        "modulators": [],
        "modulation": ModulationMatrix().describe(),
        "buses": [],
        "routing": {"schemaVersion": 1, "buses": [], "sources": [], "bands": {}},
        # --- sections added in schema version 2 ------------------------------
        #
        # Only the ones whose ownership is unambiguous. Each is a container the
        # scene owns outright, so adding it here cannot contend with data that
        # already lives on a voice or in the track.
        #: Declared external files: SOFA assets, headphone corrections, imported
        #: paths. Content hashes belong here so a render manifest can name what
        #: it actually used.
        "assets": [],
        #: Acoustic environment shared by every geometric/HRTF source: speed of
        #: sound, distance law, air absorption. Per-source overrides stay on the
        #: source.
        "environment": {},
        #: Labels, conditions and blinding for experiment mode. Deliberately
        #: separate from the acoustic parameters it describes.
        "experiment": {},
        #: Anything a newer build or an external tool wants to attach. Never
        #: interpreted here, only preserved.
        "extensions": {},
    }


#: Sections that must exist and be of the named container type after loading.
_SECTION_DEFAULTS: dict[str, type] = {
    "sources": list,
    "modulators": list,
    "buses": list,
    "assets": list,
    "environment": dict,
    "experiment": dict,
    "extensions": dict,
}


# --- migration --------------------------------------------------------------


def _migrate_1_to_2(scene: dict[str, Any]) -> dict[str, Any]:
    """Add the version 2 sections, leaving every existing value untouched."""

    blank = empty_sam_scene()
    for section in ("assets", "environment", "experiment", "extensions"):
        scene.setdefault(section, copy.deepcopy(blank[section]))
    return scene


#: One function per version step, keyed by the version it upgrades *from*.
#: Adding version 3 means adding ``2: _migrate_2_to_3`` and nothing else - the
#: loop below finds its way from any stored version to the current one.
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_1_to_2,
}


def migrate_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a scene document to the current schema version.

    Raises for a version from the future, because a newer document may mean
    things this build would silently misread - which is worse than refusing it.
    """

    result = copy.deepcopy(dict(scene))
    version = int(result.get("schemaVersion", 1))
    if version > SAM_SCENE_SCHEMA_VERSION:
        raise ValueError(
            f"sam_scene schemaVersion {version} was written by a newer build; "
            f"this one understands up to {SAM_SCENE_SCHEMA_VERSION}"
        )
    while version < SAM_SCENE_SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"no migration from sam_scene schemaVersion {version}")
        result = migration(result)
        version += 1
    result["schemaVersion"] = SAM_SCENE_SCHEMA_VERSION
    return result


def normalize_sam_scene(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a defensive, migrated scene copy with all top-level sections.

    Unknown top-level keys are carried through untouched: a document written by
    a newer build, or by an external tool, must survive a load and a save
    without this build quietly deleting the parts it does not recognise.
    """

    scene = empty_sam_scene()
    if value:
        scene.update(migrate_scene(value))
    scene["schemaVersion"] = SAM_SCENE_SCHEMA_VERSION

    # Accept the concise ``buses`` section while routing remains the complete
    # serial form consumed by the existing routing panel.
    routing = dict(scene.get("routing") or {})
    routing.setdefault("schemaVersion", 1)
    routing.setdefault("buses", copy.deepcopy(scene.get("buses") or []))
    routing.setdefault("sources", [])
    routing.setdefault("bands", {})
    scene["routing"] = routing
    scene["buses"] = copy.deepcopy(routing["buses"])

    for section, kind in _SECTION_DEFAULTS.items():
        if not isinstance(scene.get(section), kind):
            scene[section] = kind()
    return scene


def migrate_voice_scene(params: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Remove legacy scene keys from a voice and return their track-level form."""

    if not any(key in params for key in LEGACY_SCENE_KEYS):
        return None
    scene = empty_sam_scene()
    scene["stages"] = copy.deepcopy(params.pop("samStages", scene["stages"]))
    scene["modulation"] = copy.deepcopy(params.pop("samModulation", scene["modulation"]))
    scene["routing"] = copy.deepcopy(params.pop("samRouting", scene["routing"]))
    return normalize_sam_scene(scene)


# --- identity ---------------------------------------------------------------


def iter_sam_voices(steps: Iterable[Mapping[str, Any]]):
    """Yield ``(step, voice)`` for every SAM2 voice, in track order."""

    for step in steps or ():
        for voice in step.get("voices", ()) or ():
            if SAM_SYNTH_MARKER in str(voice.get("synth_function_name", "")):
                yield step, voice


def _voice_display_name(step: Mapping[str, Any], voice: Mapping[str, Any], position: int) -> str:
    """A name a person can recognise in the scene panels.

    Purely cosmetic: it is recomputed from whatever the voice is called now, and
    is never used to find a source. That is what the identifier is for, and
    keeping the two apart is what lets a voice be renamed without detaching its
    automation.
    """

    for key in ("description", "name", "label"):
        text = str(voice.get(key, "")).strip()
        if text:
            return text
    step_name = str(step.get("name", "") or step.get("description", "")).strip()
    return f"{step_name} voice {position}" if step_name else f"Source {position}"


def _next_source_id(known: set[str], serial: int) -> tuple[str, int]:
    while True:
        candidate = f"source.{serial}"
        serial += 1
        if candidate not in known:
            return candidate, serial


def assign_source_ids(
    track_data: MutableMapping[str, Any],
    scene: MutableMapping[str, Any] | None = None,
    *,
    persist_scene: bool | None = None,
) -> dict[str, Any]:
    """Give every SAM voice in the *real* track a stable identifier.

    This mutates ``track_data`` and returns the scene it maintained.  That is
    the whole point: identifiers assigned to a render-time deepcopy are thrown
    away with it, so every render invents new ones, and any automation route or
    mute that referred to a source by identifier attaches to a different source
    the moment a voice is reordered.

    An identifier already on a voice is always kept.  Names are refreshed from
    the voice each time, since a name is for reading and an identifier is for
    referring.

    ``persist_scene`` decides whether the maintained scene is written back into
    the track.  It defaults to writing one only when the track already had a
    scene or a scene was supplied, so a project that has never used scene
    features does not silently acquire a ``sam_scene`` section on its next
    render.  Identifiers are assigned to the voices either way, because those
    are what everything downstream uses to name a source.
    """

    steps = track_data.get("steps") or []
    if persist_scene is None:
        persist_scene = scene is not None or track_data.get("sam_scene") is not None
    if scene is None:
        scene = normalize_sam_scene(track_data.get("sam_scene"))

    records = {
        str(item.get("id")): dict(item)
        for item in scene.get("sources", ())
        if item.get("id")
    }
    known = set(records)
    serial = 1
    ordered: list[dict[str, Any]] = []

    for position, (step, voice) in enumerate(iter_sam_voices(steps), start=1):
        source_id = str(voice.get(SOURCE_ID_KEY, "")).strip()
        if not source_id:
            source_id, serial = _next_source_id(known, serial)
        voice[SOURCE_ID_KEY] = source_id
        known.add(source_id)
        # Keep any unknown keys a newer build put on the record.
        record = records.pop(source_id, {"id": source_id})
        record["id"] = source_id
        record["name"] = _voice_display_name(step, voice, position)
        ordered.append(record)

    # Records with no voice left are kept rather than dropped: a source removed
    # from the track may be restored by an undo, and losing its routing and
    # automation in the meantime would be a worse surprise than a stale entry.
    for orphan in records.values():
        orphan.setdefault("orphaned", True)
        ordered.append(orphan)

    scene["sources"] = ordered
    if persist_scene:
        track_data["sam_scene"] = scene
    return scene


def ensure_source_ids(
    steps: Sequence[MutableMapping[str, Any]], scene: MutableMapping[str, Any]
) -> None:
    """Assign stable identifiers over a bare step list.

    Kept for callers that hold steps without the surrounding track.  Prefer
    :func:`assign_source_ids`, which persists what it assigns.
    """

    assign_source_ids({"steps": steps, "sam_scene": scene}, scene, persist_scene=False)


# --- validation -------------------------------------------------------------


def validate_scene(
    scene_data: Mapping[str, Any] | None, steps: Sequence[Mapping[str, Any]] | None = None
) -> tuple[ValidationIssue, ...]:
    """Structural problems in a scene, and references that lead nowhere.

    A dangling identifier is an error rather than a warning: an automation
    route or a mute pointed at a source that does not exist does nothing at
    all, and doing nothing silently is exactly the failure the scene's stable
    identifiers were introduced to prevent.
    """

    if scene_data is None:
        return ()
    scene = normalize_sam_scene(scene_data)
    issues: list[ValidationIssue] = []

    declared: list[str] = []
    for position, record in enumerate(scene.get("sources", ())):
        identifier = str(record.get("id", "")).strip()
        if not identifier:
            issues.append(ValidationIssue(f"sam_scene.sources[{position}].id", "a scene source needs an identifier"))
            continue
        if identifier in declared:
            issues.append(
                ValidationIssue(
                    f"sam_scene.sources[{position}].id",
                    f"duplicate source identifier {identifier!r}; "
                    "automation and routing cannot tell the two apart",
                )
            )
        declared.append(identifier)
    known = set(declared)

    if steps is not None:
        track_ids = {
            str(voice.get(SOURCE_ID_KEY, "")).strip()
            for _, voice in iter_sam_voices(steps)
            if str(voice.get(SOURCE_ID_KEY, "")).strip()
        }
        for identifier in sorted(track_ids - known):
            issues.append(
                ValidationIssue(
                    "sam_scene.sources",
                    f"the track has a voice with source identifier {identifier!r} "
                    "that the scene does not declare",
                    "warning",
                )
            )
        known |= track_ids

    def check_reference(path: str, identifier: str) -> None:
        text = str(identifier).strip()
        if not text or text in WILDCARD_TARGETS or text in known:
            return
        issues.append(ValidationIssue(path, f"{text!r} is not a source in this scene"))

    timeline = Timeline.from_mapping(scene.get("stages"))
    issues.extend(timeline.validate())
    for position, stage in enumerate(timeline.stages):
        for index, binding in enumerate(stage.parameter_overrides):
            check_reference(
                f"sam_scene.stages[{position}].parameterOverrides[{index}].targetId",
                binding.target_id,
            )

    matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
    issues.extend(matrix.validate())
    defined = {str(item.get("id", "")).strip() for item in scene.get("modulators", ())}
    for position, route in enumerate(matrix.routes):
        check_reference(f"sam_scene.modulation.routes[{position}].targetId", route.target_id)
        if route.modulator_id not in defined:
            issues.append(
                ValidationIssue(
                    f"sam_scene.modulation.routes[{position}].modulatorId",
                    f"{route.modulator_id!r} has no modulator definition; it falls back to a "
                    "documented 1 Hz sine, which is almost certainly not what was intended",
                    "warning",
                )
            )

    routing = dict(scene.get("routing") or {})
    bus_ids = {str(bus.get("id", "")).strip() for bus in routing.get("buses", ())} | {"master"}
    for position, route in enumerate(routing.get("sources", ())):
        check_reference(f"sam_scene.routing.sources[{position}].sourceId", route.get("sourceId", ""))
        bus = str(route.get("busId", "master")).strip() or "master"
        if bus not in bus_ids:
            issues.append(
                ValidationIssue(
                    f"sam_scene.routing.sources[{position}].busId",
                    f"{bus!r} is not a bus in this scene",
                )
            )
    return tuple(issues)


# --- modulators -------------------------------------------------------------


def modulator_series(
    scene: Mapping[str, Any], times: np.ndarray, required: Sequence[str] = ()
) -> dict[str, np.ndarray]:
    """Every modulator's value over ``times``, in 0 to 1.

    A pure function of absolute time, which is what makes a render independent
    of how it was divided into blocks.
    """

    values: dict[str, np.ndarray] = {}
    for item in scene.get("modulators", ()):
        identifier = str(item.get("id", ""))
        if not identifier:
            continue
        rate = float(item.get("rateHz", 1.0))
        phase = math.radians(float(item.get("phaseDeg", 0.0)))
        wave = str(item.get("waveform", "sine"))
        seed = int(item.get("seed", 0))
        values[identifier] = _modulator_waveform(wave, times, rate, phase, seed)
    # The first matrix UI shipped before explicit modulator definitions. Its
    # identifiers compile to a documented 1 Hz sine rather than remaining an
    # attractive but inaudible control; validate_scene warns about each one, so
    # the fallback is visible rather than indefinite.
    for identifier in required:
        values.setdefault(identifier, (np.sin(2.0 * np.pi * times) + 1.0) * 0.5)
    return values


#: Waveforms a modulator can take. ``random`` is the reason ``seed`` exists:
#: without one an export would not match the preview it was approved from.
MODULATOR_WAVEFORMS = ("sine", "triangle", "square", "random")


def _modulator_waveform(
    wave: str, times: np.ndarray, rate: float, phase: float, seed: int
) -> np.ndarray:
    """One modulator's value over ``times``, in 0 to 1.

    A pure function of absolute time, including the random one: its value comes
    from interpolating a seeded sequence indexed by the cycle number, not from
    drawing as the render advances. Drawing would make the result depend on how
    the timeline was cut into blocks, and on whether anything was rendered
    before it.
    """

    cycle = rate * times + phase / (2.0 * np.pi)
    if wave == "square":
        return (np.sign(np.sin(2.0 * np.pi * cycle)) + 1.0) * 0.5
    if wave == "triangle":
        # Rises over the first half of the cycle and falls over the second.
        fraction = np.mod(cycle, 1.0)
        return 1.0 - 2.0 * np.abs(fraction - 0.5)
    if wave == "random":
        step = np.floor(cycle).astype(np.int64)
        fraction = cycle - step
        low, high = _random_steps(seed, step), _random_steps(seed, step + 1)
        # Smoothstep between held values, so a walk wanders rather than clicks.
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        return low + (high - low) * blend
    return (np.sin(2.0 * np.pi * cycle) + 1.0) * 0.5


def _random_steps(seed: int, index: np.ndarray) -> np.ndarray:
    """A stable value in 0 to 1 for every step index, for one seed.

    Hashed per index rather than generated in sequence, so asking for step
    1,000 costs the same as step 1 and gives the same answer however the render
    reached it.
    """

    # Mixed in uint64, where wraparound is defined rather than an overflow
    # warning on every call.
    mixed = np.asarray(index, dtype=np.int64).astype(np.uint64)
    mixed = mixed * np.uint64(6364136223846793005)
    mixed = mixed + (np.uint64(seed & 0xFFFFFFFFFFFFFFFF) * np.uint64(1442695040888963407))
    mixed ^= mixed >> np.uint64(33)
    mixed = mixed * np.uint64(0xFF51AFD7ED558CCD)
    mixed ^= mixed >> np.uint64(33)
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


#: Retained under its former private name for callers outside this module.
_modulator_values = modulator_series


# --- evaluation -------------------------------------------------------------


def _stage_series(
    timeline: Timeline,
    times: np.ndarray,
    targets: frozenset[str] | set[str],
    defaults: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Resolve stage bindings across ``times`` rather than at one instant.

    The same blend :meth:`Timeline.resolve` performs, evaluated as arrays: where
    stages overlap their bindings blend by weight, and a parameter short of full
    weight blends from its default so a fade-in moves a value away from where it
    was instead of jumping to the stage's own.
    """

    from .stages import apply_curve

    totals: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for stage in timeline.stages:
        weight = np.asarray(stage.weight_at(times), dtype=np.float64)
        if not np.any(weight > 0.0):
            continue
        for binding in stage.parameter_overrides:
            if binding.target_id not in targets:
                continue
            shaped = weight if binding.curve == "linear" else np.asarray(
                apply_curve(weight, binding.curve), dtype=np.float64
            )
            path = binding.parameter_path
            totals[path] = totals.get(path, 0.0) + shaped * float(binding.value)
            weights[path] = weights.get(path, 0.0) + shaped

    resolved: dict[str, np.ndarray] = {}
    for path, total in totals.items():
        weight = weights[path]
        base = float(defaults.get(path, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            saturated = np.divide(total, weight, out=np.full_like(total, base), where=weight >= 1.0)
        partial = total + (1.0 - weight) * base
        resolved[path] = np.where(weight >= 1.0, saturated, np.where(weight > 0.0, partial, base))
    return resolved


def scene_parameter_series(
    scene_data: Mapping[str, Any] | None,
    source_id: str,
    start_sample: int,
    frames: int,
    sample_rate: float,
    base: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Every automated non-gain parameter, sample by sample over a window.

    This is the block-invariant form.  Each value comes from the absolute
    sample index, so a window's contents never depend on where the caller chose
    to cut the timeline.  Gain is excluded because :func:`scene_gain_envelope`
    already applies it to the rendered audio.
    """

    if not scene_data or frames <= 0:
        return {}
    scene = normalize_sam_scene(scene_data)
    times = (np.arange(int(frames), dtype=np.float64) + int(start_sample)) / float(sample_rate)
    targets = {str(source_id)} | set(WILDCARD_TARGETS)

    defaults: dict[str, float] = {}
    for name, value in (base or {}).items():
        try:
            defaults[name] = float(value)
        except (TypeError, ValueError):
            continue

    series = _stage_series(Timeline.from_mapping(scene.get("stages")), times, targets, defaults)

    matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
    values = modulator_series(scene, times, matrix.modulators)
    for route in matrix.routes:
        if not route.is_active or route.target_id not in targets:
            continue
        modulator = values.get(route.modulator_id)
        if modulator is None:
            continue
        path = route.parameter_path
        current = series.get(path)
        if current is None:
            current = np.full(int(frames), defaults.get(path, 0.0), dtype=np.float64)
        series[path] = np.asarray(route.apply(modulator, current), dtype=np.float64)

    return {path: value for path, value in series.items() if path not in AMPLITUDE_PATHS}


def automated_paths(
    scene_data: Mapping[str, Any] | None, source_id: str, *, include_gain: bool = False
) -> tuple[str, ...]:
    """Which parameter paths this scene automates for a source.

    Determined from the scene's structure rather than by sampling it, because a
    stage that has not begun yet, or one whose weight happens to be zero at the
    instant sampled, contributes nothing at that instant and everything later.
    A caller building compiled controls needs to know the parameter is
    automated at all, not whether it is moving right now.
    """

    if not scene_data:
        return ()
    scene = normalize_sam_scene(scene_data)
    targets = {str(source_id)} | set(WILDCARD_TARGETS)
    paths: set[str] = set()
    for stage in Timeline.from_mapping(scene.get("stages")).stages:
        for binding in stage.parameter_overrides:
            if binding.target_id in targets:
                paths.add(binding.parameter_path)
    for route in ModulationMatrix.from_mapping(scene.get("modulation")).routes:
        if route.is_active and route.target_id in targets:
            paths.add(route.parameter_path)
    if not include_gain:
        paths -= set(AMPLITUDE_PATHS)
    return tuple(sorted(paths))


def scene_gain_envelope(
    scene_data: Mapping[str, Any] | None,
    source_id: str,
    start_sample: int,
    frames: int,
    sample_rate: float,
    *,
    include_routing: bool = True,
) -> np.ndarray:
    """Evaluate audible stage/modulation/routing gain for one source.

    ``include_routing`` exists because routing must be applied exactly once.
    Folding a source's bus gain and its mute/solo state into its own envelope
    gets the level right when nothing downstream sums buses, and is wrong the
    moment something does. A caller that mixes through
    :class:`~.render.scene_mix.SceneMixer` passes ``False`` and lets the mixer
    apply routing, where there is an actual bus to meter and to process.
    """

    if not scene_data or frames <= 0:
        return np.ones(max(0, frames), dtype=np.float64)
    scene = normalize_sam_scene(scene_data)
    times = (np.arange(frames, dtype=np.float64) + int(start_sample)) / float(sample_rate)
    gain = np.ones(frames, dtype=np.float64)
    targets = {str(source_id)} | set(WILDCARD_TARGETS)

    timeline = Timeline.from_mapping(scene.get("stages"))
    for stage in timeline.stages:
        weight = np.asarray(stage.weight_at(times), dtype=np.float64)
        for binding in stage.parameter_overrides:
            if binding.target_id in targets and binding.parameter_path in AMPLITUDE_PATHS:
                gain *= 1.0 + weight * (float(binding.value) - 1.0)

    matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
    modulators = modulator_series(scene, times, matrix.modulators)
    for route in matrix.routes:
        if route.is_active and route.target_id in targets and route.parameter_path in AMPLITUDE_PATHS:
            value = modulators.get(route.modulator_id)
            if value is not None:
                gain += float(route.depth * route.polarity) * value

    if not include_routing:
        return gain

    routing = dict(scene.get("routing") or {})
    buses = {str(bus.get("id", "master")): bus for bus in routing.get("buses", ())}
    source_routes = tuple(routing.get("sources", ()))
    solo_sources = {
        str(item.get("sourceId")) for item in source_routes
        if item.get("soloed") and not item.get("muted")
    }
    solo_buses = {
        identifier for identifier, bus in buses.items()
        if bus.get("soloed") and not bus.get("muted")
    }
    for route in source_routes:
        if str(route.get("sourceId")) != source_id:
            continue
        bus = buses.get(str(route.get("busId", "master")), {})
        if (route.get("muted") or bus.get("muted")
                or (solo_sources and source_id not in solo_sources)
                or (solo_buses and str(route.get("busId", "master")) not in solo_buses)):
            return np.zeros(frames, dtype=np.float64)
        gain *= 10.0 ** ((float(route.get("gainDb", 0.0)) + float(bus.get("gainDb", 0.0))) / 20.0)
        break
    return gain


def scene_parameter_overrides(
    scene_data: Mapping[str, Any] | None,
    source_id: str,
    seconds: float,
    base: Mapping[str, Any],
) -> dict[str, float]:
    """Resolve non-gain voice parameters at one absolute instant.

    Retained for the legacy compatibility renderer, whose SAM2 parameters are
    scalars.  Prefer :func:`scene_parameter_series`, which is what the compiled
    plan carries; this is the value that series takes at ``seconds``, so the two
    agree at every instant they are both asked about.
    """

    if not scene_data:
        return {}
    sample_rate = 1000.0
    series = scene_parameter_series(
        scene_data,
        source_id,
        int(round(float(seconds) * sample_rate)),
        1,
        sample_rate,
        base,
    )
    return {path: float(value[0]) for path, value in series.items()}

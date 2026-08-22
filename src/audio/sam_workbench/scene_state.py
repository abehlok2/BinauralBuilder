"""Versioned track-level state shared by every SAM source.

Scene controls used to be stored under three keys in an individual voice.  A
voice renderer could consequently neither see the other sources nor know which
copy was authoritative.  This module defines the small compatibility envelope
used at the track boundary and keeps migration explicit.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from .modulation import ModulationMatrix
from .stages import Timeline

SAM_SCENE_SCHEMA_VERSION = 1
LEGACY_SCENE_KEYS = ("samStages", "samModulation", "samRouting")


def empty_sam_scene() -> dict[str, Any]:
    return {
        "schemaVersion": SAM_SCENE_SCHEMA_VERSION,
        "sources": [],
        "stages": Timeline().describe(),
        "modulators": [],
        "modulation": ModulationMatrix().describe(),
        "buses": [],
        "routing": {"schemaVersion": 1, "buses": [], "sources": [], "bands": {}},
    }


def normalize_sam_scene(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a defensive, validated scene copy with all top-level sections."""

    scene = empty_sam_scene()
    if value:
        scene.update(copy.deepcopy(dict(value)))
    version = int(scene.get("schemaVersion", 1))
    if version != SAM_SCENE_SCHEMA_VERSION:
        raise ValueError(f"unsupported sam_scene schemaVersion {version}")
    scene["schemaVersion"] = version
    # Accept the concise ``buses`` section while routing remains the complete
    # serial form consumed by the existing routing panel.
    routing = dict(scene.get("routing") or {})
    routing.setdefault("schemaVersion", 1)
    routing.setdefault("buses", copy.deepcopy(scene.get("buses") or []))
    routing.setdefault("sources", [])
    routing.setdefault("bands", {})
    scene["routing"] = routing
    scene["buses"] = copy.deepcopy(routing["buses"])
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


def ensure_source_ids(steps: Sequence[MutableMapping[str, Any]], scene: MutableMapping[str, Any]) -> None:
    """Assign stable IDs once; list positions are used only for initial creation."""

    known = {str(item.get("id")) for item in scene.get("sources", ()) if item.get("id")}
    records = list(scene.get("sources", ()))
    serial = 1
    for step in steps:
        for voice in step.get("voices", ()):
            name = str(voice.get("synth_function_name", ""))
            if "spatial_angle_modulation_sam2" not in name:
                continue
            source_id = str(voice.get("sam_source_id", "")).strip()
            while not source_id:
                candidate = f"source.{serial}"
                serial += 1
                if candidate not in known:
                    source_id = candidate
            voice["sam_source_id"] = source_id
            if source_id not in known:
                known.add(source_id)
                records.append({"id": source_id, "name": source_id})
    scene["sources"] = records


def _modulator_values(scene: Mapping[str, Any], times: np.ndarray,
                      required: Sequence[str] = ()) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for item in scene.get("modulators", ()):
        identifier = str(item.get("id", ""))
        if not identifier:
            continue
        rate = float(item.get("rateHz", 1.0))
        phase = math.radians(float(item.get("phaseDeg", 0.0)))
        wave = str(item.get("waveform", "sine"))
        cycle = 2.0 * np.pi * rate * times + phase
        values[identifier] = ((np.sign(np.sin(cycle)) + 1.0) * .5
                              if wave == "square" else (np.sin(cycle) + 1.0) * .5)
    # The first matrix UI shipped before explicit modulator definitions.  Its
    # identifiers therefore compile to a documented 1 Hz sine rather than
    # remaining an attractive but inaudible control.
    for identifier in required:
        values.setdefault(identifier, (np.sin(2.0 * np.pi * times) + 1.0) * .5)
    return values


def scene_gain_envelope(scene_data: Mapping[str, Any] | None, source_id: str,
                        start_sample: int, frames: int, sample_rate: float) -> np.ndarray:
    """Evaluate audible stage/modulation/routing gain for one source."""

    if not scene_data or frames <= 0:
        return np.ones(max(0, frames), dtype=np.float64)
    scene = normalize_sam_scene(scene_data)
    times = (np.arange(frames, dtype=np.float64) + int(start_sample)) / float(sample_rate)
    gain = np.ones(frames, dtype=np.float64)

    timeline = Timeline.from_mapping(scene.get("stages"))
    amplitude_paths = {"amp", "amplitude", "amplitude_linear", "signal.amplitude"}
    for stage in timeline.stages:
        weight = np.asarray(stage.weight_at(times), dtype=np.float64)
        for binding in stage.parameter_overrides:
            if binding.target_id in (source_id, "voice", "*") and binding.parameter_path in amplitude_paths:
                gain *= 1.0 + weight * (float(binding.value) - 1.0)

    matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
    modulators = _modulator_values(scene, times, matrix.modulators)
    for route in matrix.routes:
        if route.is_active and route.target_id in (source_id, "voice", "*") and route.parameter_path in amplitude_paths:
            value = modulators.get(route.modulator_id)
            if value is not None:
                gain += float(route.depth * route.polarity) * value

    routing = dict(scene.get("routing") or {})
    buses = {str(bus.get("id", "master")): bus for bus in routing.get("buses", ())}
    source_routes = tuple(routing.get("sources", ()))
    solo_sources = {str(item.get("sourceId")) for item in source_routes
                    if item.get("soloed") and not item.get("muted")}
    solo_buses = {identifier for identifier, bus in buses.items()
                  if bus.get("soloed") and not bus.get("muted")}
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


def scene_parameter_overrides(scene_data: Mapping[str, Any] | None, source_id: str,
                              seconds: float, base: Mapping[str, Any]) -> dict[str, float]:
    """Resolve non-gain voice parameters at a render window's absolute start.

    The compatibility renderer has scalar legacy parameters.  Resolving them
    at each requested preview/export chunk makes the scene authoritative while
    preserving the exact block-invariant core for voices without a scene.
    """

    if not scene_data:
        return {}
    scene = normalize_sam_scene(scene_data)
    targets = {source_id, "voice", "*"}
    defaults: dict[tuple[str, str], float] = {}
    for target in targets:
        for name, value in base.items():
            try:
                defaults[(target, name)] = float(value)
            except (TypeError, ValueError):
                pass
    resolved = Timeline.from_mapping(scene.get("stages")).resolve(seconds, defaults=defaults)
    matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
    for route in matrix.routes:
        key = (route.target_id, route.parameter_path)
        if route.target_id in targets and key not in resolved:
            try:
                resolved[key] = float(base.get(route.parameter_path, 0.0))
            except (TypeError, ValueError):
                resolved[key] = 0.0
    values = _modulator_values(scene, np.asarray([seconds]), matrix.modulators)
    scalar_values = {key: float(value[0]) for key, value in values.items()}
    resolved = matrix.evaluate(scalar_values, resolved)
    output: dict[str, float] = {}
    for (target, path), value in resolved.items():
        if target in targets and path not in {"amp", "amplitude", "amplitude_linear", "signal.amplitude"}:
            output[path] = float(value)
    return output

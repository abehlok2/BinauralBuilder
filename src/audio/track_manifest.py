"""A reconstruction manifest for an ordinary track export.

The typed-project exporter has written manifests since Phase 1. Normal
BinauralBuilder export - the one almost everybody uses - wrote a WAV and
nothing else, so an exported file carried no record of the acoustic condition
that produced it: which dataset, which interpolation, which path, which seeds.

The claim this module has to earn is "sufficient to reconstruct". That is not
provable by listing fields, so :func:`reconstruct_track` exists alongside
:func:`build_track_manifest` and rebuilds a renderable track from the manifest
alone. The tests render both and compare the audio; if a field were missing,
the two would differ.

Qt-free, and outside :mod:`src.audio.sam_workbench` because it reads the legacy
track format that package is not allowed to depend on.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MANIFEST_VERSION",
    "build_track_manifest",
    "manifest_path_for",
    "reconstruct_track",
    "write_track_manifest",
]

MANIFEST_VERSION = "1.0"

#: Bumped when the *track* dictionary's own shape changes, independently of the
#: scene schema, so a reader can tell which half it does not understand.
TRACK_SCHEMA_VERSION = 1


def manifest_path_for(audio_path: str | Path) -> Path:
    """Where the manifest for an audio file goes: beside it, same stem."""

    destination = Path(audio_path)
    return destination.with_suffix(destination.suffix + ".manifest.json")


def _sofa_reference(asset: str) -> dict[str, Any]:
    """The dataset a source names, with its hash when it can be read.

    The hash is what makes the reference verifiable rather than a filename that
    may since have been replaced. A dataset that cannot be read is reported as
    such instead of silently omitted: "I could not hash this" and "there was
    nothing to hash" are different facts.
    """

    if not asset:
        return {}
    record: dict[str, Any] = {"path": str(asset)}
    try:
        from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

        dataset = load_sofa(str(asset))
        record["sha256"] = dataset.content_hash
        record["measurements"] = int(len(dataset.positions_m))
        record["sampleRateHz"] = float(dataset.sample_rate_hz)
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        record["unreadable"] = str(error)
    return record


def _coverage_warnings(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Coverage advice for this source's path against its own dataset."""

    asset = str(params.get("hrtfAsset", "") or "")
    trajectory = params.get("canonicalTrajectory")
    if not asset or not isinstance(trajectory, Mapping):
        return []
    try:
        import numpy as np

        from src.audio.sam_workbench.hrtf.coverage import assess_path_coverage
        from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
        from src.audio.sam_workbench.trajectory import path_model_from_dict

        dataset = load_sofa(asset)
        model = path_model_from_dict(dict(trajectory))
        # Sampled across the traversal rather than at an arbitrary resolution,
        # so the direction steps checked are the ones the render will take.
        samples = model.positions(np.linspace(0.0, float(model.duration_s), 256))
        report = assess_path_coverage(dataset.positions_m, samples)
        return [
            {"path": issue.path, "message": issue.message, "severity": issue.severity}
            for issue in report.issues
        ]
    except Exception:  # noqa: BLE001 - a manifest must not fail over advice
        return []


def _source_records(track_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every voice, with the parameters that decide how it sounds.

    Parameters are stored whole rather than filtered to a known list. A
    filtered copy would drop exactly the keys a newer build added, which are
    the ones a reader would most need.
    """

    records: list[dict[str, Any]] = []
    for position, step in enumerate(track_data.get("steps") or ()):
        for index, voice in enumerate(step.get("voices") or ()):
            params = dict(voice.get("params") or {})
            records.append(
                {
                    "stepIndex": position,
                    "voiceIndex": index,
                    "id": str(voice.get("sam_source_id", "") or ""),
                    "name": str(voice.get("description", "") or ""),
                    "synthFunction": str(voice.get("synth_function_name", "") or ""),
                    "isTransition": bool(voice.get("is_transition", False)),
                    "renderer": str(params.get("rendererMode", "abstract_pm")),
                    "parameters": copy.deepcopy(params),
                    "trajectory": copy.deepcopy(params.get("canonicalTrajectory"))
                    if isinstance(params.get("canonicalTrajectory"), Mapping)
                    else None,
                    "listener": copy.deepcopy(params.get("listener"))
                    if isinstance(params.get("listener"), Mapping)
                    else None,
                    "sofa": _sofa_reference(str(params.get("hrtfAsset", "") or "")),
                    "coverageWarnings": _coverage_warnings(params),
                }
            )
    return records


def _renderer_versions(records) -> dict[str, Any]:
    from src.audio.sam_workbench.render.registry import REGISTRY
    from src.audio.sam_workbench.version import PACKAGE_VERSION

    used = sorted({record["renderer"] for record in records})
    described: dict[str, Any] = {"engineVersion": PACKAGE_VERSION, "modes": {}}
    for identifier in used:
        if identifier in REGISTRY:
            definition = REGISTRY.get(identifier)
            described["modes"][identifier] = {
                "label": definition.capabilities.label,
                "physicalAzimuth": bool(definition.capabilities.physical_azimuth),
                "physicalElevation": bool(definition.capabilities.physical_elevation),
                "physicalDistance": bool(definition.capabilities.physical_distance),
                "supportsCueModification": bool(
                    definition.capabilities.supports_cue_modification
                ),
                "honestyNote": definition.capabilities.honesty_note,
            }
        else:
            described["modes"][identifier] = {"unknownToThisBuild": True}
    return described


def build_track_manifest(
    track_data: Mapping[str, Any],
    *,
    audio_path: str | Path,
    target_level: float = 0.25,
    metrics: Mapping[str, Any] | None = None,
    quality_profile: str = "offline",
) -> dict[str, Any]:
    """Describe an export completely enough to reproduce its sound."""

    from src.audio.sam_workbench.scene_state import (
        SAM_SCENE_SCHEMA_VERSION,
        normalize_sam_scene,
    )

    settings = dict(track_data.get("global_settings") or {})
    scene = normalize_sam_scene(track_data.get("sam_scene"))
    records = _source_records(track_data)
    destination = Path(audio_path)

    steps = [
        {
            "index": position,
            "description": str(step.get("description", "") or ""),
            "durationS": float(step.get("duration", 0.0) or 0.0),
        }
        for position, step in enumerate(track_data.get("steps") or ())
    ]

    manifest: dict[str, Any] = {
        "manifestVersion": MANIFEST_VERSION,
        "application": "binauralbuilder",
        "trackSchemaVersion": TRACK_SCHEMA_VERSION,
        "sceneSchemaVersion": SAM_SCENE_SCHEMA_VERSION,
        "audioFile": destination.name,
        "renderer": _renderer_versions(records),
        "audio": {
            "sampleRateHz": int(settings.get("sample_rate", 44100) or 44100),
            "crossfadeDurationS": float(settings.get("crossfade_duration", 0.0) or 0.0),
            "crossfadeCurve": str(settings.get("crossfade_curve", "linear") or "linear"),
            "channels": 2,
            "channelOrder": ["left", "right"],
        },
        "steps": steps,
        "sources": records,
        "scene": {
            "sources": copy.deepcopy(scene.get("sources", [])),
            "stages": copy.deepcopy(scene.get("stages", {})),
            "modulation": copy.deepcopy(scene.get("modulation", {})),
            "modulators": copy.deepcopy(scene.get("modulators", [])),
            "routing": copy.deepcopy(scene.get("routing", {})),
            "buses": copy.deepcopy(scene.get("buses", [])),
        },
        "seeds": {
            "trackRandomSeed": settings.get("random_seed"),
            "perSourceSeedsDerivedFrom": "project seed and source identifier",
        },
        "output": {
            "targetLevel": float(target_level),
            "normalization": "peak to target level, applied once to both ears",
            "perChannelNormalization": False,
            "limiterEnabled": bool(settings.get("limiter_enabled", False)),
            "dither": str(settings.get("dither", "none") or "none"),
            "qualityProfile": quality_profile,
        },
        "determinism": {
            "blockSizeIndependent": True,
            "sequentialChunkingEnabled": _chunking_enabled(),
        },
        "diagnostics": dict(metrics or {}),
    }
    manifest["trackSha256"] = _track_sha256(track_data)
    return manifest


def _chunking_enabled() -> bool:
    try:
        from src.synth_functions.sound_creator import ENABLE_SEQUENTIAL_CHUNKING

        return bool(ENABLE_SEQUENTIAL_CHUNKING)
    except Exception:  # pragma: no cover - defensive
        return False


def _track_sha256(track_data: Mapping[str, Any]) -> str:
    payload = json.dumps(track_data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_track_manifest(manifest: Mapping[str, Any], audio_path: str | Path) -> Path:
    """Write the manifest beside its audio file and return where it went."""

    destination = manifest_path_for(audio_path)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return destination


def reconstruct_track(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a renderable track from a manifest.

    This is what makes "sufficient to reconstruct" checkable rather than
    asserted. Anything the manifest failed to record would show up here as a
    render that does not match the one the manifest describes.
    """

    audio = dict(manifest.get("audio") or {})
    settings: dict[str, Any] = {
        "sample_rate": int(audio.get("sampleRateHz", 44100)),
        "crossfade_duration": float(audio.get("crossfadeDurationS", 0.0)),
        "crossfade_curve": str(audio.get("crossfadeCurve", "linear")),
    }
    seeds = dict(manifest.get("seeds") or {})
    if seeds.get("trackRandomSeed") is not None:
        settings["random_seed"] = seeds["trackRandomSeed"]
    output = dict(manifest.get("output") or {})
    if output.get("limiterEnabled") is not None:
        settings["limiter_enabled"] = bool(output["limiterEnabled"])
    if output.get("dither"):
        settings["dither"] = output["dither"]

    steps: list[dict[str, Any]] = []
    for entry in manifest.get("steps") or ():
        steps.append(
            {
                "duration": float(entry.get("durationS", 0.0)),
                "description": str(entry.get("description", "") or ""),
                "voices": [],
            }
        )

    for record in manifest.get("sources") or ():
        index = int(record.get("stepIndex", 0))
        while len(steps) <= index:
            steps.append({"duration": 0.0, "description": "", "voices": []})
        voice: dict[str, Any] = {
            "synth_function_name": record.get("synthFunction", ""),
            "params": copy.deepcopy(record.get("parameters") or {}),
            "is_transition": bool(record.get("isTransition", False)),
        }
        if record.get("id"):
            voice["sam_source_id"] = record["id"]
        if record.get("name"):
            voice["description"] = record["name"]
        steps[index]["voices"].append(voice)

    track: dict[str, Any] = {"global_settings": settings, "steps": steps}
    scene = manifest.get("scene")
    if isinstance(scene, Mapping) and any(scene.values()):
        track["sam_scene"] = copy.deepcopy(dict(scene))
    return track

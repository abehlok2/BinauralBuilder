"""Session planning data structures and conversion helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .utils.voice_file import load_voice_preset
from .utils.noise_file import load_noise_params
import json
import tempfile

try:  # Optional dependency: audio_engine requires PortAudio via slab
    from .synth_functions.audio_engine import (  # type: ignore
        BrainwaveState as _BrainwaveState,
        get_preset_nodes_for_state as _get_preset_nodes_for_state,
    )
except Exception:  # pragma: no cover - allow operation without slab
    _BrainwaveState = None  # type: ignore
    _get_preset_nodes_for_state = None  # type: ignore


class _FallbackBrainwaveState(Enum):
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


_FALLBACK_PRESETS = {
    "delta": {"base_freq": 100.0, "beat_freq": 2.0, "volume": 0.8},
    "theta": {"base_freq": 200.0, "beat_freq": 5.0, "volume": 0.8},
    "alpha": {"base_freq": 440.0, "beat_freq": 10.0, "volume": 0.8},
    "beta": {"base_freq": 528.0, "beat_freq": 20.0, "volume": 0.7},
    "gamma": {"base_freq": 528.0, "beat_freq": 40.0, "volume": 0.7},
}


@dataclass
class SessionPresetChoice:
    """Descriptor for a selectable preset in the session builder UI."""

    id: str
    label: str
    kind: str
    description: str = ""
    source_path: Optional[Path] = None
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class SessionStep:
    """One step of a session timeline."""

    binaural_preset_id: str
    duration: float
    start: Optional[float] = None
    noise_preset_id: Optional[str] = None
    warmup_clip_path: Optional[str] = None
    crossfade_duration: Optional[float] = None
    crossfade_curve: Optional[str] = None
    description: str = ""
    noise_volume: float = 1.0
    binaural_volume: float = 1.0


@dataclass
class Session:
    """High level session definition that can be converted into ``track_data``."""

    steps: List[SessionStep] = field(default_factory=list)
    sample_rate: int = 44100
    crossfade_duration: float = 0.0
    crossfade_curve: str = "linear"
    output_filename: str = "session_output.flac"
    background_noise_preset_id: Optional[str] = None
    background_noise_gain: float = 1.0
    background_noise_start_time: float = 0.0
    background_noise_fade_in: float = 0.0
    background_noise_fade_out: float = 0.0
    background_noise_fade_in: float = 0.0
    background_noise_fade_out: float = 0.0
    background_noise_amp_envelope: Optional[List[List[float]]] = None
    normalization_level: float = 0.95


def _collect_files(directories: Iterable[Path], extension: str) -> List[Path]:
    files: List[Path] = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        if dir_path.is_file() and dir_path.suffix == extension:
            files.append(dir_path)
            continue
        if not dir_path.is_dir():
            continue
        files.extend(sorted(dir_path.glob(f"*.{extension.lstrip('.')}")))
    return files


def _iter_brainwave_states():
    if _BrainwaveState is not None:
        return list(_BrainwaveState)
    return list(_FallbackBrainwaveState)


def _node_attr(node, name: str, default):
    if hasattr(node, name):
        return getattr(node, name)
    if isinstance(node, dict):
        return node.get(name, default)
    return default


def _node_to_dict(node, default_duration: float) -> Dict[str, float]:
    if hasattr(node, "to_dict"):
        try:
            data = node.to_dict()  # type: ignore[attr-defined]
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            pass
    if isinstance(node, dict):
        return dict(node)
    return {
        "duration": _node_attr(node, "duration", default_duration),
        "base_freq": _node_attr(node, "base_freq", 0.0),
        "beat_freq": _node_attr(node, "beat_freq", 0.0),
        "volume_left": _node_attr(node, "volume_left", 0.0),
        "volume_right": _node_attr(node, "volume_right", 0.0),
    }


def build_binaural_preset_catalog(
    duration: float = 300.0,
    preset_dirs: Optional[Iterable[Path]] = None,
) -> Dict[str, SessionPresetChoice]:
    """Return all builtin and on-disk binaural voice presets."""

    catalog: Dict[str, SessionPresetChoice] = {}

    # Built-in presets (Alpha-Gamma) removed by user request.
    # for state in _iter_brainwave_states():
    #     if _get_preset_nodes_for_state is not None:
    #         nodes = _get_preset_nodes_for_state(state, duration)
    #     else:
    #         preset = _FALLBACK_PRESETS[state.value]
    #         nodes = [
    #             {
    #                 "duration": duration,
    #                 "base_freq": preset["base_freq"],
    #                 "beat_freq": preset["beat_freq"],
    #                 "volume_left": preset["volume"],
    #                 "volume_right": preset["volume"],
    #             }
    #         ]
    #     if not nodes:
    #         continue
    #     first_node = nodes[0]
    #     node_dict = _node_to_dict(first_node, duration)
    #     preset_id = f"builtin:{state.value}"
    #     voice_payload = {
    #         "synth_function_name": "binaural_beat",
    #         "params": {
    #             "baseFreq": _node_attr(first_node, "base_freq", node_dict.get("base_freq", 0.0)),
    #             "beatFreq": _node_attr(first_node, "beat_freq", node_dict.get("beat_freq", 0.0)),
    #             "ampL": _node_attr(first_node, "volume_left", node_dict.get("volume_left", 0.0)),
    #             "ampR": _node_attr(first_node, "volume_right", node_dict.get("volume_right", 0.0)),
    #         },
    #         "is_transition": False,
    #         "description": f"Built-in {state.name.title()} preset",
    #     }
    #     catalog[preset_id] = SessionPresetChoice(
    #         id=preset_id,
    #         label=f"{state.name.title()} (Built-in)",
    #         kind="binaural",
    #         description="Generated from audio_engine defaults.",
    #         payload={
    #             "voice_data": voice_payload,
    #             "nodes": [_node_to_dict(n, duration) for n in nodes],
    #             "brainwave_state": state.value,
    #         },
    #     )

    preset_dirs = preset_dirs or []
    # Load .voice files
    voice_files = _collect_files(preset_dirs, ".voice")
    for path in voice_files:
        try:
            preset = load_voice_preset(str(path))
        except Exception:
            continue
        preset_id = f"voice:{path.stem}"
        voice_payload = {
            "synth_function_name": preset.synth_function_name,
            "params": dict(preset.params or {}),
            "is_transition": bool(preset.is_transition),
        }
        if preset.volume_envelope is not None:
            voice_payload["volume_envelope"] = preset.volume_envelope
        if preset.description:
            voice_payload["description"] = preset.description
        label = preset.description or path.stem.replace("_", " ").title()
        catalog[preset_id] = SessionPresetChoice(
            id=preset_id,
            label=label,
            kind="binaural",
            description=preset.description or "Voice preset loaded from file.",
            source_path=path,
            payload={"voice_data": voice_payload},
        )

    # Load .json files (new format)
    json_files = _collect_files(preset_dirs, ".json")
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Expecting structure like F10 Alt.json:
            # { "progression": [ { "voices": [...] }, ... ] }
            progression = data.get("progression", [])
            if not progression or not isinstance(progression, list):
                continue
            
            first_step = progression[0]
            voices = first_step.get("voices", [])
            if not voices:
                continue

            preset_id = f"json:{path.stem}"
            label = path.stem.replace("_", " ").title()
            description = first_step.get("description", "JSON preset loaded from file.")
            
            catalog[preset_id] = SessionPresetChoice(
                id=preset_id,
                label=label,
                kind="binaural",
                description=description,
                source_path=path,
                payload={"voices": voices},
            )
        except Exception:
            continue

    # Load from src.presets (generated file)
    try:
        from src.presets import BINAURAL_PRESETS
        for name, data in BINAURAL_PRESETS.items():
            try:
                progression = data.get("progression", [])
                if not progression or not isinstance(progression, list):
                    continue
                
                first_step = progression[0]
                voices = first_step.get("voices", [])
                if not voices:
                    continue

                preset_id = f"preset:{name}"
                label = name.replace("_", " ").title()
                description = first_step.get("description", "Preset loaded from presets.py.")
                
                catalog[preset_id] = SessionPresetChoice(
                    id=preset_id,
                    label=label,
                    kind="binaural",
                    description=description,
                    payload={"voices": voices},
                )
            except Exception:
                continue
    except ImportError:
        pass

    return catalog


def build_noise_preset_catalog(
    preset_dirs: Optional[Iterable[Path]] = None,
) -> Dict[str, SessionPresetChoice]:
    """Return catalog entries for background noise presets discovered on disk."""

    catalog: Dict[str, SessionPresetChoice] = {}
    preset_dirs = preset_dirs or []
    noise_files = _collect_files(preset_dirs, ".noise")
    for path in noise_files:
        try:
            params = load_noise_params(str(path))
        except Exception:
            continue
        preset_id = f"noise:{path.stem}"
        catalog[preset_id] = SessionPresetChoice(
            id=preset_id,
            label=path.stem.replace("_", " ").title(),
            kind="noise",
            description="Noise preset loaded from file.",
            source_path=path,
            payload={
                "params": asdict(params),
                "params_path": str(path),
            },
        )

    # Load from src.presets (generated file)
    try:
        from src.presets import NOISE_PRESETS
        for name, data in NOISE_PRESETS.items():
            try:
                # Noise presets in presets.py are just the params dict
                preset_id = f"noise_preset:{name}"
                catalog[preset_id] = SessionPresetChoice(
                    id=preset_id,
                    label=name.replace("_", " ").title(),
                    kind="noise",
                    description="Noise preset loaded from presets.py.",
                    payload={
                        "params": data,
                        "params_path": None, # No file path for these
                    },
                )
            except Exception:
                continue
    except ImportError:
        pass

    return catalog


def session_to_track_data(
    session: Session,
    binaural_catalog: Mapping[str, SessionPresetChoice],
    noise_catalog: Mapping[str, SessionPresetChoice],
) -> Dict[str, object]:
    """Convert a :class:`Session` into the ``track_data`` dictionary."""

    track_data: Dict[str, object] = {
        "global_settings": {
            "sample_rate": session.sample_rate,
            "crossfade_duration": session.crossfade_duration,
            "crossfade_curve": session.crossfade_curve,
            "crossfade_duration": session.crossfade_duration,
            "crossfade_curve": session.crossfade_curve,
            "output_filename": session.output_filename,
            "normalization_level": session.normalization_level,
        },
        "background_noise": {},
        "clips": [],
        "steps": [],
    }

    if session.background_noise_preset_id:
        choice = noise_catalog.get(session.background_noise_preset_id)
        if choice is None:
            raise KeyError(
                f"Unknown background noise preset: {session.background_noise_preset_id}"
            )
        params_path = choice.payload.get("params_path") or (
            str(choice.source_path) if choice.source_path else None
        )
        
        # Handle in-memory noise params (from presets.py)
        if params_path is None and "params" in choice.payload:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".noise", mode="w", encoding="utf-8") as tmp:
                    json.dump(choice.payload["params"], tmp, indent=2)
                    params_path = tmp.name
            except Exception:
                pass

        if params_path is None:
            raise ValueError(
                f"Noise preset '{choice.id}' is missing an associated parameter file."
            )
        track_data["background_noise"] = {
            "noise_file": params_path,
            "gain": session.background_noise_gain,
            "start_time": session.background_noise_start_time,
            "fade_in": session.background_noise_fade_in,
            "fade_out": session.background_noise_fade_out,
        }
        if session.background_noise_amp_envelope:
            track_data["background_noise"]["amp_envelope"] = session.background_noise_amp_envelope
    else:
        track_data["background_noise"] = {
            "file_path": "",
            "amp": 0.0,
            "pan": 0.0,
            "start_time": 0.0,
            "fade_in": 0.0,
            "fade_out": 0.0,
            "amp_envelope": [],
        }

    current_time = 0.0
    for step in session.steps:
        choice = binaural_catalog.get(step.binaural_preset_id)
        if choice is None:
            raise KeyError(f"Unknown binaural preset: {step.binaural_preset_id}")
        voice_payload = choice.payload.get("voice_data")
        voices_payload = choice.payload.get("voices")
        
        if voices_payload:
             # New JSON format with multiple voices
             voice_entries = [copy.deepcopy(v) for v in voices_payload]
        elif isinstance(voice_payload, dict):
            # Old .voice format with single voice
            voice_entries = [copy.deepcopy(voice_payload)]
        else:
            raise ValueError(f"Binaural preset '{choice.id}' is missing voice data.")

        # Tag binaural voices
        for voice in voice_entries:
            voice["voice_type"] = "binaural"

        step_start = step.start if step.start is not None else current_time
        step_entry: Dict[str, object] = {
            "duration": step.duration,
            "start": step_start,
            "voices": voice_entries,
            "binaural_volume": step.binaural_volume,
            "noise_volume": step.noise_volume,
            "normalization_level": session.normalization_level,
        }
        if step.description:
            step_entry["description"] = step.description
        if step.crossfade_duration is not None:
            step_entry["crossfade_duration"] = step.crossfade_duration
        if step.crossfade_curve:
            step_entry["crossfade_curve"] = step.crossfade_curve
        if step.noise_preset_id:
            step_entry["noise_preset_id"] = step.noise_preset_id

        track_data["steps"].append(step_entry)

        if step.warmup_clip_path:
            track_data["clips"].append(
                {
                    "file_path": step.warmup_clip_path,
                    "path": step.warmup_clip_path,
                    "start": step_start,
                    "duration": 0.0,
                    "amp": 1.0,
                    "pan": 0.0,
                    "fade_in": 0.0,
                    "fade_out": 0.0,
                    "description": step.description or choice.label,
                }
            )

        if step.noise_preset_id:
            noise_choice = noise_catalog.get(step.noise_preset_id)
            if noise_choice:
                noise_params = noise_choice.payload.get("params", {})
                
                # Determine if it's a transition or static noise
                is_transition = getattr(noise_params, "transition", False) or noise_params.get("transition", False)
                
                # Create a voice entry for the noise
                noise_voice = {
                    "synth_function_name": "noise_swept_notch_transition" if is_transition else "noise_swept_notch",
                    "params": dict(noise_params),
                    "is_transition": bool(is_transition),
                    "voice_type": "noise",
                }
                
                # Add the noise voice to the step's voices
                step_entry["voices"].append(noise_voice)

        step_crossfade = (
            step.crossfade_duration
            if step.crossfade_duration is not None
            else session.crossfade_duration
        )
        effective_advance = max(0.0, step.duration - max(0.0, step_crossfade))
        if step.start is None:
            current_time += effective_advance
        else:
            current_time = max(current_time, step_start + step.duration)

    return track_data


__all__ = [
    "Session",
    "SessionStep",
    "SessionPresetChoice",
    "build_binaural_preset_catalog",
    "build_noise_preset_catalog",
    "session_to_track_data",
]

"""Helper for saving and loading noise generator parameters."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any

from .colored_noise import DEFAULT_COLOR_PRESETS, load_custom_color_presets

# Default file extension for noise parameter files
NOISE_FILE_EXTENSION = ".noise"

@dataclass
class NoiseParams:
    """Representation of parameters used for noise generation."""
    duration_seconds: float = 60.0
    sample_rate: int = 44100
    noise_type: str = "pink"
    lfo_waveform: str = "sine"
    transition: bool = False
    # Non-transition mode uses ``lfo_freq`` and ``sweeps``
    lfo_freq: float = 1.0 / 12.0
    # Transition mode
    start_lfo_freq: float = 1.0 / 12.0
    end_lfo_freq: float = 1.0 / 12.0
    sweeps: List[Dict[str, Any]] = field(default_factory=list)
    noise_parameters: Dict[str, Any] = field(default_factory=dict)
    start_lfo_phase_offset_deg: int = 0
    end_lfo_phase_offset_deg: int = 0
    start_intra_phase_offset_deg: int = 0
    end_intra_phase_offset_deg: int = 0
    initial_offset: float = 0.0
    duration: float = 0.0
    input_audio_path: str = ""
    start_time: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    amp_envelope: List[Dict[str, Any]] = field(default_factory=list)
    static_notches: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def color_params(self) -> Dict[str, Any]:
        """Backwards-compatible alias for noise colour parameters."""

        return self.noise_parameters

    @color_params.setter
    def color_params(self, value: Dict[str, Any]) -> None:
        self.noise_parameters = value or {}


def _color_parameters_for_type(noise_type: str) -> Dict[str, Any]:
    """Return the full colour parameter set for ``noise_type`` if known."""

    key = (noise_type or "").strip().lower()
    if not key:
        return {}

    presets: Dict[str, Dict[str, Any]] = {
        name.lower(): params for name, params in DEFAULT_COLOR_PRESETS.items()
    }
    for name, preset in load_custom_color_presets().items():
        presets[name.lower()] = preset

    params = presets.get(key, {}).copy()
    if params:
        params.setdefault("name", noise_type)
    return params


def _normalized_noise_parameters(params: NoiseParams) -> Dict[str, Any]:
    """Ensure the noise parameters contain all colour fields and a name."""

    merged = dict(params.noise_parameters or {})
    if not merged:
        merged = _color_parameters_for_type(params.noise_type)

    if params.noise_type and not merged.get("name"):
        merged["name"] = params.noise_type
    return merged


def save_noise_params(params: NoiseParams, filepath: str) -> None:
    """Save ``params`` to ``filepath`` using JSON inside a ``.noise`` file."""
    path = Path(filepath)
    if path.suffix != NOISE_FILE_EXTENSION:
        path = path.with_suffix(NOISE_FILE_EXTENSION)
    data = asdict(params)
    data["noise_parameters"] = _normalized_noise_parameters(params)
    data.pop("noise_type", None)
    data.pop("color_params", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_noise_params(filepath: str) -> NoiseParams:
    """Load noise parameters from ``filepath`` and return a :class:`NoiseParams`."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Noise parameter file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = NoiseParams()
    noise_params = data.get("noise_parameters") or data.get("color_params") or {}
    noise_type = data.get("noise_type", params.noise_type)
    for k, v in data.items():
        target = "duration" if k == "post_offset" else k
        if target in {"noise_parameters", "color_params", "noise_type"}:
            continue
        if hasattr(params, target):
            setattr(params, target, v)

    params.noise_parameters = noise_params or _color_parameters_for_type(noise_type)
    if noise_type and not params.noise_parameters.get("name"):
        params.noise_parameters["name"] = noise_type
    params.noise_type = params.noise_parameters.get("name", noise_type)
    return params

__all__ = ["NoiseParams", "save_noise_params", "load_noise_params", "NOISE_FILE_EXTENSION"]


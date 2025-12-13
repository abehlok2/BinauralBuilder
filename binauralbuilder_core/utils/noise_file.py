"""Helper for saving and loading noise generator parameters."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any

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
    color_params: Dict[str, Any] = field(default_factory=dict)
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


def save_noise_params(params: NoiseParams, filepath: str) -> None:
    """Save ``params`` to ``filepath`` using JSON inside a ``.noise`` file."""
    path = Path(filepath)
    if path.suffix != NOISE_FILE_EXTENSION:
        path = path.with_suffix(NOISE_FILE_EXTENSION)
    data = asdict(params)
    data["color_params"] = _normalized_color_params(params.noise_type, params.color_params)
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
    for k, v in data.items():
        target = "duration" if k == "post_offset" else k
        if hasattr(params, target):
            setattr(params, target, v)

    params.color_params = _normalized_color_params(params.noise_type, params.color_params)
    if params.noise_type and not params.color_params.get("name"):
        params.color_params["name"] = params.noise_type
    return params

__all__ = ["NoiseParams", "save_noise_params", "load_noise_params", "NOISE_FILE_EXTENSION"]

# Default colour parameter fallbacks to ensure .noise files are explicit
COLOR_PARAM_DEFAULTS: Dict[str, Any] = {
    "exponent": 1.0,
    "high_exponent": None,
    "distribution_curve": 1.0,
    "lowcut": None,
    "highcut": None,
    "amplitude": 1.0,
    "seed": 1,
}


def _normalized_color_params(noise_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**COLOR_PARAM_DEFAULTS, **(params or {})}
    exponent = merged.get("exponent", COLOR_PARAM_DEFAULTS["exponent"])
    merged.setdefault("high_exponent", exponent)
    merged.setdefault("distribution_curve", COLOR_PARAM_DEFAULTS["distribution_curve"])
    merged.setdefault("lowcut", COLOR_PARAM_DEFAULTS["lowcut"])
    merged.setdefault("highcut", COLOR_PARAM_DEFAULTS["highcut"])
    merged.setdefault("amplitude", COLOR_PARAM_DEFAULTS["amplitude"])
    merged.setdefault("seed", COLOR_PARAM_DEFAULTS["seed"])
    if noise_type and not merged.get("name"):
        merged["name"] = noise_type
    return merged

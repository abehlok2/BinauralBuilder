"""Helper for saving and loading noise generator parameters."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any

# Default file extension for noise parameter files
NOISE_FILE_EXTENSION = ".noise"

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
    """Return ``params`` with all required colour fields populated."""

    merged = {**COLOR_PARAM_DEFAULTS, **(params or {})}
    exponent = merged.get("exponent", COLOR_PARAM_DEFAULTS["exponent"])
    if merged.get("high_exponent") is None:
        merged["high_exponent"] = exponent
    if merged.get("distribution_curve") is None:
        merged["distribution_curve"] = COLOR_PARAM_DEFAULTS["distribution_curve"]
    if merged.get("lowcut") is None:
        merged["lowcut"] = COLOR_PARAM_DEFAULTS["lowcut"]
    if merged.get("highcut") is None:
        merged["highcut"] = COLOR_PARAM_DEFAULTS["highcut"]
    if merged.get("amplitude") is None:
        merged["amplitude"] = COLOR_PARAM_DEFAULTS["amplitude"]
    if merged.get("seed") is None:
        merged["seed"] = COLOR_PARAM_DEFAULTS["seed"]
    if noise_type and not merged.get("name"):
        merged["name"] = noise_type
    return merged


@dataclass
class NoiseParams:
    """Representation of parameters used for noise generation."""
    duration_seconds: float = 60.0
    sample_rate: int = 44100
    lfo_waveform: str = "sine"
    transition: bool = False
    # Non-transition mode uses ``lfo_freq`` and ``sweeps``
    lfo_freq: float = 1.0 / 12.0
    # Transition mode
    start_lfo_freq: float = 1.0 / 12.0
    end_lfo_freq: float = 1.0 / 12.0
    sweeps: List[Dict[str, Any]] = field(default_factory=list)
    noise_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"name": "pink"}
    )
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

    @property
    def noise_type(self) -> str:
        """Alias returning the selected noise colour name."""

        return (self.noise_parameters or {}).get("name", "")

    @noise_type.setter
    def noise_type(self, value: str) -> None:
        params = dict(self.noise_parameters or {})
        if value:
            params.setdefault("name", value)
        self.noise_parameters = params


def _normalized_noise_parameters(params: NoiseParams) -> Dict[str, Any]:
    """Ensure the noise parameters contain all colour fields and a name."""

    merged = dict(params.noise_parameters or {})
    if not merged:
        merged = _normalized_color_params("pink", {})

    noise_name = merged.get("name", "pink")
    return _normalized_color_params(noise_name, merged)


def save_noise_params(params: NoiseParams, filepath: str) -> None:
    """Save ``params`` to ``filepath`` using JSON inside a ``.noise`` file."""
    path = Path(filepath)
    if path.suffix != NOISE_FILE_EXTENSION:
        path = path.with_suffix(NOISE_FILE_EXTENSION)
    data = asdict(params)
    data["noise_parameters"] = _normalized_noise_parameters(params)
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
    noise_type = data.get("noise_type", "")
    for k, v in data.items():
        target = "duration" if k == "post_offset" else k
        if target in {"noise_parameters", "color_params", "noise_type"}:
            continue
        if hasattr(params, target):
            setattr(params, target, v)

    if noise_type and not noise_params.get("name"):
        noise_params["name"] = noise_type

    noise_name = noise_params.get("name", "pink")
    params.noise_parameters = _normalized_color_params(
        noise_name, noise_params or _normalized_color_params(noise_name, {})
    )
    return params


__all__ = [
    "NoiseParams",
    "save_noise_params",
    "load_noise_params",
    "NOISE_FILE_EXTENSION",
]

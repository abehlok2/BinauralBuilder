"""Level-matched renderer and opt-in legacy migration comparisons."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
from numpy.typing import NDArray

from ..compat import render_sam2_voice

@dataclass(frozen=True)
class ComparisonResult:
    mode_a: str
    mode_b: str
    audio_a: NDArray[np.float32]
    audio_b: NDArray[np.float32]
    gain_a: float
    gain_b: float

def _rms(audio):
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0

def render_ab_comparison(params: Mapping, duration_s: float, sample_rate_hz: int,
                         mode_a: str = "abstract_pm", mode_b: str = "hrtf") -> ComparisonResult:
    """Render two conditions and level-match B using one gain shared by both ears."""
    conditions = []
    for mode in (mode_a, mode_b):
        condition = copy.deepcopy(dict(params)); condition["rendererMode"] = mode
        conditions.append(render_sam2_voice(duration_s, sample_rate_hz, params=condition))
    rms_a, rms_b = _rms(conditions[0]), _rms(conditions[1])
    gain_b = rms_a / rms_b if rms_b > 1e-15 else 1.0
    matched_b = np.asarray(conditions[1] * gain_b, dtype=np.float32)
    return ComparisonResult(mode_a, mode_b, conditions[0], matched_b, 1.0, gain_b)

def render_legacy_migration_preview(
    legacy_renderer: Callable[..., NDArray], params: Mapping, duration_s: float,
    sample_rate_hz: int,
) -> ComparisonResult:
    """Compare frozen legacy output with explicit SOFA without rewriting params."""
    original = copy.deepcopy(dict(params))
    legacy = np.asarray(legacy_renderer(duration_s, sample_rate_hz, **original), dtype=np.float32)
    explicit_params = copy.deepcopy(original); explicit_params["rendererMode"] = "hrtf"
    explicit = render_sam2_voice(duration_s, sample_rate_hz, params=explicit_params)
    rms_legacy, rms_explicit = _rms(legacy), _rms(explicit)
    gain = rms_legacy / rms_explicit if rms_explicit > 1e-15 else 1.0
    if dict(params) != original:
        raise RuntimeError("migration preview mutated its input")
    return ComparisonResult("legacy_slab", "hrtf", legacy,
                            np.asarray(explicit * gain, dtype=np.float32), 1.0, gain)

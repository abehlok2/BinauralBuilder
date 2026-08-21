"""Capture the current SAM2 behaviour of both Python synthesis trees.

Phase 0 records what ``src.synth_functions.spatial_angle_modulation`` and
``binauralbuilder_core.synth_functions.spatial_angle_modulation`` produce today,
*before* either tree is changed, so the later delegation work can prove exactly
what it preserves and what it deliberately changes.  The two trees currently
disagree, which is why both are captured separately.

Regenerate with::

    python tests/sam_workbench/fixtures/capture_legacy_sam2.py

The capture is intentionally tiny: 10 ms per case at 44.1 kHz.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
ARRAY_PATH = FIXTURE_DIR / "legacy_sam2_reference.npz"
METADATA_PATH = FIXTURE_DIR / "legacy_sam2_reference.json"

SAMPLE_RATE_HZ = 44_100
DURATION_S = 0.01

#: ``case name -> (function name, keyword parameters)``.
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "static_open_sinusoidal": (
        "spatial_angle_modulation_sam2",
        {
            "amp": 0.6,
            "carrierFreq": 220.0,
            "modFreq": 3.0,
            "arcWidthDeg": 90.0,
            "directionOffsetDeg": 10.0,
            "spatialScale": 1.0,
            "pathType": "open",
            "pathShape": "sinusoidal",
        },
    ),
    "static_closed_circular": (
        "spatial_angle_modulation_sam2",
        {
            "amp": 0.5,
            "carrierFreq": 180.0,
            "modFreq": 5.0,
            "arcWidthDeg": 120.0,
            "directionOffsetDeg": -20.0,
            "spatialScale": 0.8,
            "pathType": "closed",
            "rotationDirection": "cw",
        },
    ),
    "static_discontinuous": (
        "spatial_angle_modulation_sam2",
        {
            "amp": 0.4,
            "carrierFreq": 300.0,
            "modFreq": 2.0,
            "arcWidthDeg": 60.0,
            "pathType": "discontinuous",
            "discontinuousSteps": 6,
        },
    ),
    "static_constant_angle_polarity": (
        # arcWidthDeg = 0 removes the path shape, leaving a constant interaural
        # phase, so the recorded ear polarity is unambiguous.
        "spatial_angle_modulation_sam2",
        {
            "amp": 0.5,
            "carrierFreq": 220.0,
            "modFreq": 4.0,
            "arcWidthDeg": 0.0,
            "directionOffsetDeg": 90.0,
            "spatialScale": 1.0,
            "pathType": "open",
            "pathShape": "sinusoidal",
        },
    ),
    "transition_carrier_and_mod": (
        "spatial_angle_modulation_sam2_transition",
        {
            "amp": 0.6,
            "startCarrierFreq": 200.0,
            "endCarrierFreq": 260.0,
            "startModFreq": 3.0,
            "endModFreq": 6.0,
            "startArcWidthDeg": 90.0,
            "endArcWidthDeg": 45.0,
            "pathType": "open",
            "pathShape": "sinusoidal",
        },
    ),
}

TREES: dict[str, str] = {
    "src": "src.synth_functions.spatial_angle_modulation",
    "binauralbuilder_core": "binauralbuilder_core.synth_functions.spatial_angle_modulation",
}

#: Both trees currently render the legacy polarity ``left = carrier - ipd`` and
#: ``right = carrier + ipd``.  The canonical exact mode uses the opposite
#: (left-plus/right-minus) convention, so this is recorded, never assumed.
LEGACY_POLARITY = "left_minus_right_plus"


REPOSITORY_ROOT = FIXTURE_DIR.parents[2]


def _resolve(module_name: str, function_name: str) -> Callable[..., np.ndarray]:
    import importlib
    import sys

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    return getattr(importlib.import_module(module_name), function_name)


def capture() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_s": DURATION_S,
        "legacy_polarity": LEGACY_POLARITY,
        "array_layout": "frame_major_(frames, 2)",
        "cases": {},
    }

    for tree, module_name in TREES.items():
        for case, (function_name, parameters) in CASES.items():
            key = f"{tree}.{case}"
            function = _resolve(module_name, function_name)
            audio = np.asarray(function(DURATION_S, SAMPLE_RATE_HZ, **parameters), dtype=np.float32)
            arrays[key] = audio
            metadata["cases"][key] = {
                "tree": tree,
                "module": module_name,
                "function": function_name,
                "params": parameters,
                "shape": list(audio.shape),
                "dtype": str(audio.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest(),
            }
    return arrays, metadata


def main() -> int:
    arrays, metadata = capture()
    np.savez_compressed(ARRAY_PATH, **arrays)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {ARRAY_PATH} ({len(arrays)} cases)")
    print(f"wrote {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

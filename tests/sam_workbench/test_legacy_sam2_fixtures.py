"""Guard the captured legacy SAM2 behaviour of both Python synthesis trees.

These fixtures were captured in Phase 0, before any delegation work, so a later
change to either tree is either intentional (regenerate the fixture) or a
regression (this test fails).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
ARRAY_PATH = FIXTURE_DIR / "legacy_sam2_reference.npz"
METADATA_PATH = FIXTURE_DIR / "legacy_sam2_reference.json"


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arrays():
    with np.load(ARRAY_PATH) as data:
        yield {key: np.array(data[key]) for key in data.files}


def _case_names() -> list[str]:
    return sorted(json.loads(METADATA_PATH.read_text(encoding="utf-8"))["cases"])


def test_fixture_files_exist():
    assert ARRAY_PATH.exists()
    assert METADATA_PATH.exists()


def test_capture_covers_both_python_trees(metadata):
    trees = {case["tree"] for case in metadata["cases"].values()}
    assert trees == {"src", "binauralbuilder_core"}


def test_stored_arrays_match_their_recorded_hashes(metadata, arrays):
    for key, case in metadata["cases"].items():
        audio = arrays[key]
        assert list(audio.shape) == case["shape"]
        assert str(audio.dtype) == case["dtype"]
        digest = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
        assert digest == case["sha256"], f"{key} fixture no longer matches its hash"


@pytest.mark.parametrize("case_name", _case_names())
def test_legacy_functions_still_reproduce_the_capture(case_name, metadata, arrays):
    pytest.importorskip("scipy")
    import importlib

    case = metadata["cases"][case_name]
    module = importlib.import_module(case["module"])
    function = getattr(module, case["function"])
    rendered = np.asarray(
        function(metadata["duration_s"], metadata["sample_rate_hz"], **case["params"]),
        dtype=np.float32,
    )
    np.testing.assert_allclose(rendered, arrays[case_name], rtol=0, atol=1e-6)


def test_the_two_trees_still_disagree_so_delegation_is_required(arrays):
    """Documented gap: the duplicate SAM2 implementations are not equivalent today."""

    src_audio = arrays["src.static_open_sinusoidal"]
    core_audio = arrays["binauralbuilder_core.static_open_sinusoidal"]
    assert src_audio.shape == core_audio.shape
    assert not np.allclose(src_audio, core_audio, atol=1e-4)


def test_captured_legacy_polarity_is_left_minus_right_plus(metadata, arrays):
    """The canonical exact mode is left-plus/right-minus; legacy SAM2 is the opposite."""

    assert metadata["legacy_polarity"] == "left_minus_right_plus"
    case = metadata["cases"]["src.static_constant_angle_polarity"]
    audio = arrays["src.static_constant_angle_polarity"]
    sample_rate = metadata["sample_rate_hz"]
    time = np.arange(audio.shape[0], dtype=np.float64) / sample_rate

    # With a zero arc width the interaural phase is constant, so the ear
    # polarity is the only thing this comparison can be sensitive to.
    amplitude = case["params"]["amp"]
    carrier_phase = 2.0 * np.pi * case["params"]["carrierFreq"] * time
    interaural_phase = case["params"]["spatialScale"] * np.sin(
        np.radians(case["params"]["directionOffsetDeg"])
    )

    np.testing.assert_allclose(
        audio[:, 0], amplitude * np.sin(carrier_phase - interaural_phase), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        audio[:, 1], amplitude * np.sin(carrier_phase + interaural_phase), rtol=0, atol=1e-6
    )

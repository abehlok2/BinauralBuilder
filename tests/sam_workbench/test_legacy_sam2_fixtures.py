"""Guard the captured legacy SAM2 behaviour of both Python synthesis trees.

The fixtures were captured in Phase 0, before any delegation work. Phase 1
pointed both trees at the canonical renderer, so this module now records which
parts of that capture had to stay identical and which changed on purpose:

* **static ``src`` voices** must still render bit-for-bit as captured;
* **transition voices** changed deliberately - phase is integrated from the
  transition's own origin and ``initial_offset`` is time rather than a phase
  angle, which is what makes a chunked render match a whole-step render;
* **``binauralbuilder_core`` voices** changed deliberately - that tree carried
  a second, different SAM2 under the same public name and now resolves to the
  one canonical implementation.

The capture file itself is left untouched: it is the historical record the
delegation work is measured against.
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


STATIC_SRC_CASES = [
    name for name in _case_names() if name.startswith("src.") and "transition" not in name
]


@pytest.mark.parametrize("case_name", STATIC_SRC_CASES)
def test_static_src_voices_still_reproduce_the_capture(case_name, metadata, arrays):
    """Delegation must not change what an existing static preset sounds like."""

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


def test_transition_voices_changed_deliberately_and_stayed_close(metadata, arrays):
    """The transition renderer was replaced; the sound is close, the defect is gone."""

    pytest.importorskip("scipy")
    import importlib

    case_name = "src.transition_carrier_and_mod"
    case = metadata["cases"][case_name]
    module = importlib.import_module(case["module"])
    function = getattr(module, case["function"])
    rendered = np.asarray(
        function(
            metadata["duration_s"],
            metadata["sample_rate_hz"],
            initial_offset=0.0,
            transition_duration=metadata["duration_s"],
            **case["params"],
        ),
        dtype=np.float32,
    )
    captured = arrays[case_name]
    assert rendered.shape == captured.shape
    # Same signal, one sample of phase convention apart: highly correlated but
    # not identical. A regression would destroy the correlation entirely.
    for channel in (0, 1):
        correlation = float(np.corrcoef(rendered[:, channel], captured[:, channel])[0, 1])
        assert correlation > 0.99
    assert float(np.max(np.abs(rendered - captured))) > 0.0


def test_core_tree_voices_now_match_the_canonical_renderer(metadata, arrays):
    """The duplicate tree's independent SAM2 was replaced by delegation."""

    pytest.importorskip("scipy")
    import importlib

    for case_name in (name for name in _case_names() if name.startswith("binauralbuilder_core.")):
        case = metadata["cases"][case_name]
        if "transition" in case_name:
            continue
        module = importlib.import_module(case["module"])
        rendered = np.asarray(
            getattr(module, case["function"])(
                metadata["duration_s"], metadata["sample_rate_hz"], **case["params"]
            ),
            dtype=np.float32,
        )
        src_equivalent = arrays[case_name.replace("binauralbuilder_core.", "src.")]
        np.testing.assert_allclose(rendered, src_equivalent, rtol=0, atol=1e-6)


def test_the_capture_shows_why_delegation_was_needed(arrays):
    """As captured, the two trees disagreed; that is the gap Phase 1 closed."""

    src_audio = arrays["src.static_open_sinusoidal"]
    core_audio = arrays["binauralbuilder_core.static_open_sinusoidal"]
    assert src_audio.shape == core_audio.shape
    assert not np.allclose(src_audio, core_audio, atol=1e-4)


def test_both_trees_now_resolve_to_one_implementation():
    """After delegation the same parameters give the same samples in both trees."""

    src_tree = pytest.importorskip("src.synth_functions.spatial_angle_modulation")
    core_tree = pytest.importorskip("binauralbuilder_core.synth_functions.spatial_angle_modulation")
    params = {"amp": 0.6, "carrierFreq": 220.0, "modFreq": 3.0, "arcWidthDeg": 90.0}
    np.testing.assert_array_equal(
        src_tree.spatial_angle_modulation_sam2(0.01, 44_100, **params),
        core_tree.spatial_angle_modulation_sam2(0.01, 44_100, **params),
    )


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

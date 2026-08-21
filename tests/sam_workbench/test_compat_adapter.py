"""The BinauralBuilder SAM compatibility adapter.

Covers the naming, shape, polarity, and offset conversions the adapter owns,
and the consolidation of the two public synthesis trees onto one implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.compat import (
    SAM2_PARAMETER_DEFAULTS,
    SAM2_TRANSITION_PARAMETER_DEFAULTS,
    Sam2Spec,
    compiled_source_from_sam2,
    render_sam2,
    render_sam2_voice,
    sam2_spec_from_params,
)
from src.audio.sam_workbench.conventions import to_channel_major, to_frame_major
from src.audio.sam_workbench.dsp import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
)

SAMPLE_RATE = 44_100
BASE_PARAMS = {
    "amp": 0.6,
    "carrierFreq": 220.0,
    "modFreq": 3.0,
    "arcWidthDeg": 90.0,
    "directionOffsetDeg": 10.0,
    "spatialScale": 1.0,
    "pathType": "open",
    "pathShape": "sinusoidal",
}


# --- parameter translation --------------------------------------------------


def test_camel_case_parameters_become_typed_snake_case_fields():
    spec = sam2_spec_from_params(BASE_PARAMS)
    assert spec.amplitude == pytest.approx(0.6)
    assert spec.carrier_start_hz == pytest.approx(220.0)
    assert spec.modulation_start_hz == pytest.approx(3.0)
    assert spec.arc_width_start_deg == pytest.approx(90.0)
    assert spec.direction_offset_start_deg == pytest.approx(10.0)
    assert spec.path_type == "open"
    assert spec.path_shape == "sinusoidal"


def test_missing_parameters_fall_back_to_the_authoritative_defaults():
    spec = sam2_spec_from_params({})
    assert spec.amplitude == pytest.approx(SAM2_PARAMETER_DEFAULTS["amp"])
    assert spec.carrier_start_hz == pytest.approx(SAM2_PARAMETER_DEFAULTS["carrierFreq"])
    assert spec.modulation_start_hz == pytest.approx(SAM2_PARAMETER_DEFAULTS["modFreq"])
    assert spec.discontinuous_steps == SAM2_PARAMETER_DEFAULTS["discontinuousSteps"]


@pytest.mark.parametrize(
    ("alias", "canonical", "value"),
    [("beatFreq", "modulation_start_hz", 7.0), ("arcWidth", "arc_width_start_deg", 33.0), ("directionOffset", "direction_offset_start_deg", -12.0)],
)
def test_older_parameter_aliases_are_still_accepted(alias, canonical, value):
    spec = sam2_spec_from_params({alias: value})
    assert getattr(spec, canonical) == pytest.approx(value)


def test_path_shape_defaults_follow_the_path_type():
    assert sam2_spec_from_params({"pathType": "closed"}).path_shape == "ramp"
    assert sam2_spec_from_params({"pathType": "discontinuous"}).path_shape == "square"
    assert sam2_spec_from_params({"pathType": "open"}).path_shape == "sinusoidal"


def test_unknown_parameters_are_preserved_rather_than_erased():
    spec = sam2_spec_from_params({**BASE_PARAMS, "futureCue": 3, "vendorExtension": {"a": 1}})
    assert spec.unknown_params == {"futureCue": 3, "vendorExtension": {"a": 1}}


def test_transition_pairs_are_read_only_for_transition_voices():
    params = {**BASE_PARAMS, "startCarrierFreq": 100.0, "endCarrierFreq": 400.0}
    static = sam2_spec_from_params(params)
    assert static.carrier_start_hz == static.carrier_end_hz == pytest.approx(220.0)

    transition = sam2_spec_from_params(params, is_transition=True, transition_duration=1.0)
    assert transition.carrier_start_hz == pytest.approx(100.0)
    assert transition.carrier_end_hz == pytest.approx(400.0)
    assert transition.is_ramped is True


def test_transition_defaults_hold_the_static_value_when_only_one_end_is_given():
    spec = sam2_spec_from_params(
        {**BASE_PARAMS, "startCarrierFreq": 150.0}, is_transition=True, transition_duration=1.0
    )
    assert spec.carrier_start_hz == pytest.approx(150.0)
    assert spec.carrier_end_hz == pytest.approx(150.0)
    assert set(SAM2_TRANSITION_PARAMETER_DEFAULTS) >= {"startCarrierFreq", "endCarrierFreq"}


def test_core_tree_parameter_names_are_translated_not_dropped():
    spec = sam2_spec_from_params(
        {"peakPhaseDev": 0.55, "phaseOffsetL": 0.1, "phaseOffsetR": math.pi / 2}
    )
    assert spec.spatial_scale_start == pytest.approx(0.55)
    assert spec.left_phase_rad == pytest.approx(0.1)
    assert spec.right_phase_rad == pytest.approx(math.pi / 2)
    assert "peakPhaseDev" not in spec.unknown_params


# --- polarity ---------------------------------------------------------------


def test_unversioned_voices_keep_the_legacy_ear_polarity():
    assert sam2_spec_from_params(BASE_PARAMS).ear_polarity == EAR_POLARITY_LEGACY


def test_versioned_voices_default_to_the_canonical_exact_polarity():
    spec = sam2_spec_from_params({**BASE_PARAMS, "samSchemaVersion": 1})
    assert spec.ear_polarity == EAR_POLARITY_CANONICAL


def test_polarity_can_be_named_explicitly():
    for polarity in (EAR_POLARITY_CANONICAL, EAR_POLARITY_LEGACY, EAR_POLARITY_SAME):
        spec = sam2_spec_from_params({**BASE_PARAMS, "earPolarity": polarity})
        assert spec.ear_polarity == polarity


def test_canonical_and_legacy_polarity_swap_the_ears():
    legacy = render_sam2_voice(0.05, SAMPLE_RATE, params=BASE_PARAMS)
    canonical = render_sam2_voice(
        0.05, SAMPLE_RATE, params={**BASE_PARAMS, "samSchemaVersion": 1}
    )
    np.testing.assert_allclose(legacy[:, 0], canonical[:, 1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(legacy[:, 1], canonical[:, 0], rtol=0, atol=1e-6)


def test_legacy_polarity_matches_the_documented_equations():
    params = {**BASE_PARAMS, "arcWidthDeg": 0.0, "directionOffsetDeg": 90.0}
    audio = render_sam2_voice(0.02, SAMPLE_RATE, params=params)
    times = np.arange(audio.shape[0]) / SAMPLE_RATE
    carrier = 2 * math.pi * 220.0 * times
    interaural = 1.0 * math.sin(math.radians(90.0))
    np.testing.assert_allclose(audio[:, 0], 0.6 * np.sin(carrier - interaural), atol=1e-6)
    np.testing.assert_allclose(audio[:, 1], 0.6 * np.sin(carrier + interaural), atol=1e-6)


# --- array layout -----------------------------------------------------------


def test_adapter_returns_frame_major_float32_and_the_core_stays_channel_major():
    frames = 512
    voice = render_sam2_voice(frames / SAMPLE_RATE + 1e-6, SAMPLE_RATE, params=BASE_PARAMS)
    assert voice.shape == (frames, 2)
    assert voice.dtype == np.float32

    spec = sam2_spec_from_params(BASE_PARAMS)
    core = render_sam2(spec, frames, SAMPLE_RATE)
    assert core.shape == (2, frames)
    np.testing.assert_allclose(to_frame_major(core), voice, rtol=0, atol=1e-6)
    np.testing.assert_allclose(to_channel_major(voice), core, rtol=0, atol=1e-6)


def test_zero_length_renders_return_an_empty_stereo_buffer():
    empty = render_sam2_voice(0.0, SAMPLE_RATE, params=BASE_PARAMS)
    assert empty.shape == (0, 2)
    assert empty.dtype == np.float32


def test_rendered_audio_is_finite_and_within_the_amplitude():
    audio = render_sam2_voice(0.2, SAMPLE_RATE, params={**BASE_PARAMS, "pathType": "custom",
                                                        "customPathProfile": {"points": [[0, 0], [10, 40], [60, 90], [100, 20]],
                                                                              "closedLoop": True, "kind": "spline"}})
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 0.6 + 1e-6


def test_block_size_does_not_change_the_samples():
    spec = sam2_spec_from_params(BASE_PARAMS)
    reference = render_sam2(spec, 4096, SAMPLE_RATE)
    for block_size in (1, 100, 4096):
        np.testing.assert_array_equal(render_sam2(spec, 4096, SAMPLE_RATE, block_size=block_size), reference)


def test_compiled_source_carries_the_legacy_path_as_a_modulation_provider():
    source = compiled_source_from_sam2(sam2_spec_from_params(BASE_PARAMS))
    assert source.modulation_provider is not None
    assert source.carrier_phase_provider is not None
    assert source.ear_polarity == EAR_POLARITY_LEGACY


def test_spec_is_a_plain_dataclass_with_documented_units():
    spec = Sam2Spec()
    assert spec.arc_width_start_deg == pytest.approx(90.0)
    assert spec.left_phase_rad == pytest.approx(0.0)
    assert spec.transition_duration_s is None


# --- both public trees resolve to one implementation ------------------------


@pytest.fixture(scope="module")
def trees():
    src_tree = pytest.importorskip("src.synth_functions.spatial_angle_modulation")
    core_tree = pytest.importorskip("binauralbuilder_core.synth_functions.spatial_angle_modulation")
    return src_tree, core_tree


@pytest.mark.parametrize(
    "params",
    [
        BASE_PARAMS,
        {**BASE_PARAMS, "pathType": "closed", "rotationDirection": "ccw"},
        {**BASE_PARAMS, "pathType": "discontinuous", "discontinuousSteps": 6},
    ],
    ids=["open", "closed", "discontinuous"],
)
def test_both_public_trees_produce_identical_static_audio(trees, params):
    src_tree, core_tree = trees
    from_src = src_tree.spatial_angle_modulation_sam2(0.05, SAMPLE_RATE, **params)
    from_core = core_tree.spatial_angle_modulation_sam2(0.05, SAMPLE_RATE, **params)
    np.testing.assert_array_equal(from_src, from_core)
    np.testing.assert_allclose(
        from_src, render_sam2_voice(0.05, SAMPLE_RATE, params=params), rtol=0, atol=1e-7
    )


def test_both_public_trees_produce_identical_transition_audio(trees):
    src_tree, core_tree = trees
    params = {
        "amp": 0.5,
        "startCarrierFreq": 200.0,
        "endCarrierFreq": 260.0,
        "startModFreq": 3.0,
        "endModFreq": 6.0,
    }
    from_src = src_tree.spatial_angle_modulation_sam2_transition(
        0.05, SAMPLE_RATE, initial_offset=0.0, transition_duration=0.05, **params
    )
    from_core = core_tree.spatial_angle_modulation_sam2_transition(
        0.05, SAMPLE_RATE, initial_offset=0.0, transition_duration=0.05, **params
    )
    np.testing.assert_array_equal(from_src, from_core)


def test_the_public_static_function_accepts_the_chunk_offset_keyword(trees):
    src_tree, _ = trees
    at_zero = src_tree.spatial_angle_modulation_sam2(0.05, SAMPLE_RATE, initial_offset=0.0, **BASE_PARAMS)
    later = src_tree.spatial_angle_modulation_sam2(0.05, SAMPLE_RATE, initial_offset=0.5, **BASE_PARAMS)
    assert at_zero.shape == later.shape
    assert not np.allclose(at_zero, later, atol=1e-3)


def test_legacy_private_path_helpers_still_resolve(trees):
    src_tree, _ = trees
    from src.audio.sam_workbench.trajectory import legacy_paths

    assert src_tree._resolve_sam2_shape is legacy_paths.resolve_sam2_shape
    assert src_tree._progress_open is legacy_paths.progress_open
    assert src_tree.SAM2_DEFAULT_SHAPES_BY_TYPE is legacy_paths.SAM2_DEFAULT_SHAPES_BY_TYPE


def test_sam2_functions_are_still_discoverable_through_sound_creator():
    sound_creator = pytest.importorskip("src.synth_functions.sound_creator")
    assert "spatial_angle_modulation_sam2" in sound_creator.SYNTH_FUNCTIONS
    assert "spatial_angle_modulation_sam2_transition" in sound_creator.SYNTH_FUNCTIONS


def _apply_sound_creator_edge_fade(audio: np.ndarray) -> np.ndarray:
    """`generate_voice_audio` adds a 10 ms fade at each end when a voice has no envelope."""

    faded = audio.copy()
    fade_frames = min(faded.shape[0], int(0.01 * SAMPLE_RATE))
    if fade_frames > 1:
        faded[:fade_frames] *= np.linspace(0, 1, fade_frames)[:, None]
        faded[-fade_frames:] *= np.linspace(1, 0, fade_frames)[:, None]
    return faded


def test_generate_voice_audio_renders_a_sam2_voice_through_the_adapter():
    sound_creator = pytest.importorskip("src.synth_functions.sound_creator")
    voice = {
        "synth_function_name": "spatial_angle_modulation_sam2",
        "is_transition": False,
        "voice_type": "binaural",
        "params": dict(BASE_PARAMS),
    }
    audio = sound_creator.generate_voice_audio(voice, 0.05, SAMPLE_RATE, global_start_time=0.0)
    expected = _apply_sound_creator_edge_fade(render_sam2_voice(0.05, SAMPLE_RATE, params=BASE_PARAMS))
    assert audio.shape == (int(0.05 * SAMPLE_RATE), 2)
    np.testing.assert_allclose(audio, expected, rtol=0, atol=1e-6)


def test_generate_voice_audio_is_chunk_invariant_for_sam2():
    """Away from `sound_creator`'s own per-chunk edge fades, the samples line up.

    The 10 ms fade at each chunk edge is existing `sound_creator` behaviour, not
    the SAM adapter's; what Phase 1 fixes is the waveform underneath it, which
    used to restart at every chunk.
    """

    sound_creator = pytest.importorskip("src.synth_functions.sound_creator")
    voice = {
        "synth_function_name": "spatial_angle_modulation_sam2",
        "is_transition": False,
        "voice_type": "binaural",
        "params": dict(BASE_PARAMS),
    }
    whole = sound_creator.generate_voice_audio(voice, 0.5, SAMPLE_RATE, global_start_time=0.0)
    chunks = [
        sound_creator.generate_voice_audio(
            voice, 0.25, SAMPLE_RATE, global_start_time=0.0, chunk_start_time=0.25 * index
        )
        for index in range(2)
    ]
    joined = np.concatenate(chunks)
    assert joined.shape == whole.shape

    fade_frames = int(0.01 * SAMPLE_RATE)
    interior = np.ones(whole.shape[0], dtype=bool)
    for boundary in (0, whole.shape[0] // 2, whole.shape[0]):
        interior[max(0, boundary - fade_frames) : boundary + fade_frames] = False
    np.testing.assert_allclose(joined[interior], whole[interior], rtol=0, atol=1e-6)

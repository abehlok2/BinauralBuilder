from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.synth_functions.sound_creator as sound_creator


def test_generate_audio_preserves_voice_state_across_long_step_chunks(monkeypatch):
    sample_rate = 100
    track_data = {
        "global_settings": {"sample_rate": sample_rate},
        "steps": [{"duration": 1.2, "voices": [{"synth_function_name": "dummy"}]}],
    }

    monkeypatch.setattr(sound_creator, "ENABLE_SEQUENTIAL_CHUNKING", True)
    monkeypatch.setattr(sound_creator, "SEQUENTIAL_CHUNK_THRESHOLD_SECONDS", 0.5)
    monkeypatch.setattr(sound_creator, "SEQUENTIAL_CHUNK_DURATION_SECONDS", 0.4)

    calls = []

    def fake_generate_single_step_audio_segment(
        step_data,
        global_settings,
        target_duration_seconds,
        duration_override=None,
        chunk_start_time=0.0,
        continuity_context=None,
        voice_states=None,
        return_state=False,
    ):
        calls.append(
            {
                "chunk_start_time": chunk_start_time,
                "voice_states": voice_states,
                "return_state": return_state,
            }
        )
        n = int(target_duration_seconds * sample_rate)
        audio = np.zeros((n, 2), dtype=np.float32)
        next_state = [{"chunk_index": len(calls)}]
        if return_state:
            return audio, next_state
        return audio

    monkeypatch.setattr(sound_creator, "generate_single_step_audio_segment", fake_generate_single_step_audio_segment)
    monkeypatch.setattr(sound_creator, "_write_audio_file", lambda *_args, **_kwargs: True)

    assert sound_creator.generate_audio(track_data, output_filename="dummy.wav") is True
    assert len(calls) == 3

    assert calls[0]["voice_states"] == [None]
    assert calls[1]["voice_states"] == [{"chunk_index": 1}]
    assert calls[2]["voice_states"] == [{"chunk_index": 2}]
    assert all(call["return_state"] for call in calls)


def test_voice_logical_key_uses_trimmed_description_for_cross_step_matching():
    voice_with_spaces = {
        "description": "  7  ",
        "synth_function_name": "binaural_beat",
        "voice_type": "binaural",
    }
    voice_trimmed = {
        "description": "7",
        "synth_function_name": "binaural_beat",
        "voice_type": "binaural",
    }

    assert sound_creator._voice_logical_key(voice_with_spaces, 0) == sound_creator._voice_logical_key(voice_trimmed, 1)


def test_assemble_track_crossfade_applies_fade_in_and_fade_out(monkeypatch):
    sample_rate = 100
    track_data = {
        "global_settings": {
            "sample_rate": sample_rate,
            "crossfade_duration": 0.2,
            "crossfade_curve": "linear",
            "crossfade_overlap": 1.0,
        },
        "steps": [
            {"duration": 1.0, "voices": []},
            {"duration": 1.0, "voices": []},
        ],
    }

    call_idx = {"count": 0}

    def fake_generate_single_step_audio_segment(step_data, _global_settings, target_duration_seconds, **kwargs):
        n = int(float(target_duration_seconds) * sample_rate)
        level = 1.0 if call_idx["count"] == 0 else 0.0
        call_idx["count"] += 1
        audio = np.full((n, 2), level, dtype=np.float32)
        if kwargs.get("return_state"):
            return audio, []
        return audio

    monkeypatch.setattr(sound_creator, "generate_single_step_audio_segment", fake_generate_single_step_audio_segment)

    audio = sound_creator.assemble_track_from_data(
        track_data,
        sample_rate,
        crossfade_duration=0.2,
        crossfade_curve="linear",
        crossfade_overlap=1.0,
    )

    # Transition starts at sample 80 and spans 20 samples.
    assert audio[80, 0] == pytest.approx(1.0, abs=1e-6)
    assert audio[90, 0] == pytest.approx(0.47368422, abs=1e-5)
    assert audio[99, 0] == pytest.approx(0.0, abs=1e-6)


def test_crossfade_overlap_factor_controls_step_overlap(monkeypatch):
    sample_rate = 100
    base_track_data = {
        "global_settings": {
            "sample_rate": sample_rate,
            "crossfade_duration": 0.2,
            "crossfade_curve": "linear",
        },
        "steps": [
            {"duration": 1.0, "voices": []},
            {"duration": 1.0, "voices": []},
        ],
    }

    call_idx = {"count": 0}

    def fake_generate_single_step_audio_segment(_step_data, _global_settings, target_duration_seconds, **kwargs):
        n = int(float(target_duration_seconds) * sample_rate)
        level = 1.0 if call_idx["count"] == 0 else 0.5
        call_idx["count"] += 1
        audio = np.full((n, 2), level, dtype=np.float32)
        if kwargs.get("return_state"):
            return audio, []
        return audio

    # Full overlap case
    track_data_full = {
        "global_settings": {**base_track_data["global_settings"], "crossfade_overlap": 1.0},
        "steps": [dict(step) for step in base_track_data["steps"]],
    }
    monkeypatch.setattr(sound_creator, "generate_single_step_audio_segment", fake_generate_single_step_audio_segment)
    full = sound_creator.assemble_track_from_data(
        track_data_full,
        sample_rate,
        crossfade_duration=0.2,
        crossfade_curve="linear",
        crossfade_overlap=1.0,
    )

    # Zero overlap case
    track_data_zero = {
        "global_settings": {**base_track_data["global_settings"], "crossfade_overlap": 0.0},
        "steps": [dict(step) for step in base_track_data["steps"]],
    }
    zero = sound_creator.assemble_track_from_data(
        track_data_zero,
        sample_rate,
        crossfade_duration=0.2,
        crossfade_curve="linear",
        crossfade_overlap=0.0,
    )

    # With full overlap, total length is reduced by 20 samples. Without overlap, full 200 samples.
    assert full.shape[0] == 180
    assert zero.shape[0] == 200


def test_spatial_angle_modulation_generates_stereo_with_expected_shape():
    from src.synth_functions.spatial_angle_modulation import spatial_angle_modulation_sam2

    audio = spatial_angle_modulation_sam2(
        duration=1.0,
        sample_rate=1000,
        amp=0.5,
        carrierFreq=220.0,
        modFreq=8.0,
        peakPhaseDev=0.7,
        phaseOffsetL=0.0,
        phaseOffsetR=np.pi / 2.0,
        pathType='open',
    )

    assert audio.shape == (1000, 2)
    assert np.max(np.abs(audio)) <= 0.501
    assert not np.allclose(audio[:, 0], audio[:, 1])


def test_spatial_angle_modulation_transition_interpolates_parameters():
    from src.synth_functions.spatial_angle_modulation import spatial_angle_modulation_sam2_transition

    audio = spatial_angle_modulation_sam2_transition(
        duration=1.0,
        sample_rate=1000,
        amp=0.6,
        startCarrierFreq=220.0,
        endCarrierFreq=440.0,
        startModFreq=4.0,
        endModFreq=12.0,
        startPeakPhaseDev=0.3,
        endPeakPhaseDev=1.0,
        startPhaseOffsetL=0.0,
        endPhaseOffsetL=np.pi / 4.0,
        startPhaseOffsetR=np.pi / 2.0,
        endPhaseOffsetR=np.pi / 3.0,
        pathType='closed',
    )

    assert audio.shape == (1000, 2)
    assert np.max(np.abs(audio)) <= 0.6 + 1e-3
    assert np.std(audio[:, 0]) > 0.05


def test_spatial_angle_modulation_original_engine_signature_still_available():
    from src.synth_functions.spatial_angle_modulation import spatial_angle_modulation

    audio = spatial_angle_modulation(
        duration=0.25,
        sample_rate=800,
        amp=0.6,
        carrierFreq=300.0,
        beatFreq=6.0,
        pathShape='circle',
        pathRadius=1.0,
        arcStartDeg=0.0,
        arcEndDeg=180.0,
    )

    assert audio.shape == (200, 2)

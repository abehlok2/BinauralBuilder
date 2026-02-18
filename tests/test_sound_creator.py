from pathlib import Path
import sys

import numpy as np

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

    assert calls[0]["voice_states"] is None
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

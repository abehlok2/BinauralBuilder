"""Assembling a track without ever holding all of it.

The bounded encoder still needed a finished track to read from, so the largest
allocation in an export - one array for the whole render, about 1.3 GB for an
hour of stereo float32 - was untouched by it. Nothing needs that array: a
sample is final once the step after it has been placed, because no later step
reaches back past its own start.

The streaming assembler is a second placement implementation, and that is a
cost worth naming. The alternative was refactoring the function every export
depends on - including background noise, trimming and explicit step placement -
which is a larger risk than holding two implementations equal by test. So these
tests assert equality directly, sample for sample, rather than trusting it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import src.synth_functions.sound_creator as sound_creator

RATE = 8000


def _track(steps=3, duration=0.5, **step_extra):
    return {
        "global_settings": {"sample_rate": RATE},
        "steps": [
            {
                "duration": duration,
                "description": f"step {index}",
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {
                            "amp_left": 0.25, "amp_right": 0.25,
                            "baseFreq": 200.0 + 20 * index, "beatFreq": 4.0,
                        },
                    }
                ],
                **step_extra,
            }
            for index in range(steps)
        ],
    }


def _streamed(track, crossfade=0.0, **kwargs):
    blocks = list(
        sound_creator.iter_assembled_track_blocks(track, RATE, crossfade, **kwargs)
    )
    if not blocks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(blocks, axis=0)


# --- the equality that makes this usable at all -----------------------------


@pytest.mark.parametrize(
    "steps, duration, crossfade",
    [
        (1, 0.7, 0.0),
        (3, 0.5, 0.0),
        (3, 0.5, 0.1),
        (5, 0.3, 0.05),
        (2, 0.25, 0.2),
    ],
)
def test_streaming_assembly_is_the_same_samples(steps, duration, crossfade):
    track = _track(steps, duration)
    whole = sound_creator.assemble_track_from_data(track, RATE, crossfade)
    streamed = _streamed(track, crossfade)

    assert streamed.shape == whole.shape
    assert np.array_equal(streamed, whole)


@pytest.mark.parametrize("curve", ["linear", "equal_power"])
def test_every_crossfade_curve_agrees(curve):
    track = _track(3, 0.4)
    whole = sound_creator.assemble_track_from_data(track, RATE, 0.1, curve)
    streamed = _streamed(track, 0.1, crossfade_curve=curve)
    assert np.array_equal(streamed, whole)


def test_a_per_step_crossfade_override_agrees():
    track = _track(3, 0.4)
    track["steps"][1]["crossfade_duration"] = 0.15
    whole = sound_creator.assemble_track_from_data(track, RATE, 0.05)
    streamed = _streamed(track, 0.05)
    assert np.array_equal(streamed, whole)


# --- what it holds ----------------------------------------------------------


def test_the_blocks_are_stereo_float32_and_add_up():
    track = _track(4, 0.3)
    blocks = list(sound_creator.iter_assembled_track_blocks(track, RATE, 0.0))
    assert blocks
    for block in blocks:
        assert block.ndim == 2 and block.shape[1] == 2
        assert block.dtype == np.float32
    total = sum(block.shape[0] for block in blocks)
    assert total == sound_creator.assemble_track_from_data(track, RATE, 0.0).shape[0]


def test_peak_memory_does_not_follow_the_track_length():
    """The property the whole exercise is for."""

    import tracemalloc

    def peak_for(steps):
        track = {
            "global_settings": {"sample_rate": 44100},
            "steps": [
                {
                    "duration": 30.0,
                    "voices": [
                        {
                            "synth_function_name": "binaural_beat",
                            "params": {"amp_left": 0.3, "amp_right": 0.3,
                                       "baseFreq": 200.0, "beatFreq": 4.0},
                        }
                    ],
                }
                for _ in range(steps)
            ],
        }
        tracemalloc.start()
        for _block in sound_creator.iter_assembled_track_blocks(track, 44100, 0.0):
            pass  # consumed and dropped, as the encoder does
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    short, long = peak_for(4), peak_for(20)  # 2 minutes against 10
    # Five times the audio must not be five times the memory. Measured flat at
    # about 127 MB for both; the allowance is for allocator noise, not growth.
    assert long < short * 1.25


def test_the_whole_track_path_still_grows_with_length():
    """Stated so the comparison above is not asserted against nothing."""

    import tracemalloc

    def peak_for(steps):
        track = {
            "global_settings": {"sample_rate": 44100},
            "steps": [
                {
                    "duration": 30.0,
                    "voices": [
                        {
                            "synth_function_name": "binaural_beat",
                            "params": {"amp_left": 0.3, "amp_right": 0.3,
                                       "baseFreq": 200.0, "beatFreq": 4.0},
                        }
                    ],
                }
                for _ in range(steps)
            ],
        }
        tracemalloc.start()
        sound_creator.assemble_track_from_data(track, 44100, 0.0)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    assert peak_for(20) > peak_for(4) * 1.4


# --- when it must not be used ----------------------------------------------


def test_a_track_of_plain_steps_is_streamable():
    assert sound_creator.track_is_streamable(_track()) is True


def test_explicit_step_placement_is_not_streamable():
    """Explicit starts can overlap three steps at once, or run backwards."""

    track = _track()
    track["steps"][1]["start"] = 0.1
    assert sound_creator.track_is_streamable(track) is False


def test_a_background_noise_layer_is_not_streamable():
    """It is generated for the finished duration and mixed over all of it."""

    track = _track()
    track["background_noise"] = {"file": "somewhere.json"}
    assert sound_creator.track_is_streamable(track) is False


def test_an_empty_background_noise_section_does_not_block_streaming():
    track = _track()
    track["background_noise"] = {}
    assert sound_creator.track_is_streamable(track) is True


# --- through the real export ------------------------------------------------


def test_a_streamed_export_and_a_whole_track_export_are_the_same_file(tmp_path, monkeypatch):
    """Same bytes, so switching paths cannot change what a user gets."""

    rate = 44100
    track = {
        "global_settings": {"sample_rate": rate},
        "steps": [
            {
                "duration": 20.0,
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {"amp_left": 0.3, "amp_right": 0.3,
                                   "baseFreq": 200.0, "beatFreq": 4.0},
                    }
                ],
            }
            for _ in range(4)
        ],
    }

    streamed = tmp_path / "streamed.wav"
    assert sound_creator.generate_audio(track, output_filename=str(streamed)) is True

    whole = tmp_path / "whole.wav"
    monkeypatch.setattr(
        sound_creator, "_stream_assemble_export", lambda *_a, **_k: None
    )
    assert sound_creator.generate_audio(track, output_filename=str(whole)) is True

    assert hashlib.sha256(streamed.read_bytes()).hexdigest() == (
        hashlib.sha256(whole.read_bytes()).hexdigest()
    )


def test_a_short_export_does_not_bother_streaming(tmp_path):
    """A round trip through the filesystem would cost more than it saves."""

    assert (
        sound_creator._stream_assemble_export(
            _track(2, 0.5), RATE, 0.0, "linear", 1.0,
            str(tmp_path / "short.wav"), 0.25, 0.0, None,
        )
        is None
    )


def test_a_format_the_encoder_cannot_stream_falls_back(tmp_path):
    assert (
        sound_creator._stream_assemble_export(
            _track(60, 30.0), 44100, 0.0, "linear", 1.0,
            str(tmp_path / "track.mp3"), 0.25, 0.0, None,
        )
        is None
    )


def test_cancellation_travels_out_of_a_streamed_assembly():
    """A cancel must not be swallowed by the block generator."""

    calls = {"n": 0}

    def cancel(_fraction):
        calls["n"] += 1
        if calls["n"] > 1:
            raise sound_creator.RenderCancelled()

    with pytest.raises(sound_creator.RenderCancelled):
        list(
            sound_creator.iter_assembled_track_blocks(
                _track(6, 0.3), RATE, 0.0, progress_callback=cancel
            )
        )

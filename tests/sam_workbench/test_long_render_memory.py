"""What bounds a long render's memory, and what still does not.

The export is bounded: `streaming.py` measures the peak while spooling to
temporary storage, then encodes from it in blocks. The *render* is not - the
track is still materialized before the encoder sees it.

There is a flag that looks like it would fix that, and it would not. These
tests record what was measured, so the next person to read
``ENABLE_SEQUENTIAL_CHUNKING`` and think "why is this off?" has the answer
without having to rediscover it.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.synth_functions.sound_creator as sound_creator

RATE = 44100


def _track(duration_s):
    return {
        "global_settings": {"sample_rate": RATE},
        "steps": [
            {
                "duration": duration_s,
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {
                            "amp_left": 0.3, "amp_right": 0.3,
                            "baseFreq": 200.0, "beatFreq": 4.0,
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def restore_flags():
    saved = (
        sound_creator.ENABLE_SEQUENTIAL_CHUNKING,
        sound_creator.SEQUENTIAL_CHUNK_THRESHOLD_SECONDS,
        sound_creator.SEQUENTIAL_CHUNK_DURATION_SECONDS,
    )
    yield
    (
        sound_creator.ENABLE_SEQUENTIAL_CHUNKING,
        sound_creator.SEQUENTIAL_CHUNK_THRESHOLD_SECONDS,
        sound_creator.SEQUENTIAL_CHUNK_DURATION_SECONDS,
    ) = saved


def test_sequential_chunking_is_not_silently_enabled():
    """It is off for a reason, and the reason is the next test."""

    assert sound_creator.ENABLE_SEQUENTIAL_CHUNKING is False


def test_chunked_generation_does_not_reproduce_the_unchunked_render(restore_flags):
    """Why the flag is off, pinned as a measurement rather than a claim.

    Each chunk fades in from zero instead of continuing the previous one,
    because the synth functions carry no oscillator phase across a chunk
    boundary. Enabling this would put an audible fade into every long render at
    every boundary. When that is fixed, this test should start failing - and
    the fix is to invert it, not to delete it.
    """

    duration = 90.0
    boundary = 30.0

    sound_creator.ENABLE_SEQUENTIAL_CHUNKING = False
    whole = sound_creator.assemble_track_from_data(_track(duration), RATE, 0.0)

    sound_creator.ENABLE_SEQUENTIAL_CHUNKING = True
    sound_creator.SEQUENTIAL_CHUNK_THRESHOLD_SECONDS = 60.0
    sound_creator.SEQUENTIAL_CHUNK_DURATION_SECONDS = boundary
    chunked = sound_creator.assemble_track_from_data(_track(duration), RATE, 0.0)

    assert whole.shape == chunked.shape
    assert not np.allclose(whole, chunked, atol=1e-3)

    # The divergence is at the boundary, and it is a fade rather than noise.
    at_boundary = int(boundary * RATE)
    assert abs(float(chunked[at_boundary, 0])) < abs(float(whole[at_boundary, 0]))


def test_the_export_is_bounded_even_though_the_render_is_not(tmp_path):
    """The half that is done, asserted where it actually holds."""

    import tracemalloc

    from src.audio.sam_workbench.streaming import stream_normalized_export

    def blocks(frames, chunk=1 << 16):
        generator = np.random.default_rng(0)
        produced = 0
        while produced < frames:
            span = min(chunk, frames - produced)
            yield (generator.normal(size=(span, 2)) * 0.3).astype(np.float32)
            produced += span

    tracemalloc.start()
    stream_normalized_export(blocks(RATE * 120), tmp_path / "long.wav", RATE)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Two minutes of stereo float32 is about 42 MB; the encoder uses a fraction.
    assert peak < 16 * 1024 * 1024


def test_the_renderer_still_materializes_the_whole_track():
    """Recorded so the remaining work is visible rather than assumed done.

    assemble_track_from_data returns an array, so its caller necessarily holds
    the whole render. Making it yield blocks is the remaining half of bounded
    memory, and it is blocked on the same per-voice state continuity that the
    chunking flag is blocked on.
    """

    audio = sound_creator.assemble_track_from_data(_track(0.5), RATE, 0.0)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (int(0.5 * RATE), 2)

"""What bounds a long render's memory.

The export is bounded: `streaming.py` measures the peak while spooling to
temporary storage, then encodes from it in blocks.

The render is bounded too, for steps whose voices can be continued across a
chunk boundary. That was not true before: `ENABLE_SEQUENTIAL_CHUNKING` was off
because a chunked render did not reproduce a whole one. The recorded reason -
that the synth functions carry no oscillator phase - was wrong. The cause was a
10 ms declick fade applied to every generated buffer, which is right at the
edges of a voice and wrong at every internal chunk boundary. These tests pin
the corrected behaviour so it cannot regress back into a fade per boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.synth_functions.sound_creator as sound_creator

RATE = 44100

#: The synth functions that are actually in use. Others exist and several are
#: dormant; the continuity gate handles those at runtime rather than here.
LIVE_SYNTHS = {
    "binaural_beat": {
        "amp_left": 0.3, "amp_right": 0.3, "baseFreq": 200.0, "beatFreq": 4.0,
    },
    "isochronic_tone": {
        "amp": 0.3, "baseFreq": 200.0, "beatFreq": 4.0,
    },
    "spatial_angle_modulation_sam2": {
        "amp": 0.3, "carrierFreq": 200.0, "beatFreq": 4.0,
    },
}


def _track(duration_s, synth="binaural_beat", params=None):
    return {
        "global_settings": {"sample_rate": RATE},
        "steps": [
            {
                "duration": duration_s,
                "voices": [
                    {
                        "synth_function_name": synth,
                        "params": dict(params if params is not None else LIVE_SYNTHS[synth]),
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


def _render(track, *, chunked, boundary=30.0):
    sound_creator.ENABLE_SEQUENTIAL_CHUNKING = chunked
    sound_creator.SEQUENTIAL_CHUNK_THRESHOLD_SECONDS = 60.0
    sound_creator.SEQUENTIAL_CHUNK_DURATION_SECONDS = boundary
    return sound_creator.assemble_track_from_data(track, RATE, 0.0)


def test_sequential_chunking_is_on():
    """The flag is on, and the next test is why that is safe."""

    assert sound_creator.ENABLE_SEQUENTIAL_CHUNKING is True


@pytest.mark.parametrize("synth", sorted(LIVE_SYNTHS))
def test_chunked_generation_reproduces_the_unchunked_render(synth, restore_flags):
    """A chunked render is the whole render, sample for sample.

    Not "close enough": any difference at all is a discontinuity partway
    through a step, and the boundary is exactly where one would appear.
    """

    duration, boundary = 90.0, 30.0
    track = _track(duration, synth)

    whole = _render(track, chunked=False)
    chunked = _render(track, chunked=True, boundary=boundary)

    assert whole.shape == chunked.shape
    assert np.array_equal(whole, chunked), (
        f"{synth}: max difference {np.max(np.abs(whole - chunked)):.3e}"
    )


def test_the_declick_fade_stays_at_the_voice_edges(restore_flags):
    """The bug that kept the flag off, pinned where it happened.

    The fade belongs at the start and end of a voice. Applying it per generated
    buffer put a 10 ms dip at every chunk boundary - the audio either side was
    correct and bit-identical, and only the seam was wrong.
    """

    boundary = 30.0
    chunked = _render(_track(90.0), chunked=True, boundary=boundary)

    at_boundary = int(boundary * RATE)
    fade = int(0.01 * RATE)
    seam = np.abs(chunked[at_boundary - fade:at_boundary + fade, 0])
    settled = np.abs(chunked[at_boundary + 5 * fade:at_boundary + 15 * fade, 0])

    # A dip would show as the seam's peak falling well below the surrounding
    # signal's. Without one they are the same waveform.
    assert seam.max() > 0.9 * settled.max()

    # The voice's own edges are still faded.
    assert abs(float(chunked[0, 0])) < 1e-6
    assert abs(float(chunked[-1, 0])) < 1e-6


def test_a_voice_that_cannot_continue_is_rendered_whole(restore_flags, monkeypatch):
    """The gate, exercised through a synth that reports no state.

    Rather than an allowlist of safe functions - one that would rot as
    functions are added - the harness asks the states that actually came back.
    A synth returning none falls back to whole-step rendering.
    """

    calls = []

    def stateless(duration, sample_rate=44100, **params):
        calls.append(duration)
        frames = int(duration * sample_rate)
        t = np.arange(frames) / float(sample_rate)
        tone = (0.2 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
        return np.stack([tone, tone], axis=1)

    monkeypatch.setitem(sound_creator.SYNTH_FUNCTIONS, "stateless_probe", stateless)
    track = _track(90.0, "stateless_probe", params={})

    audio = _render(track, chunked=True, boundary=30.0)

    assert audio.shape == (int(90.0 * RATE), 2)
    # The first chunk is attempted, found wanting, and the step is then
    # rendered whole - so the last call covers the entire step.
    assert calls[-1] == pytest.approx(90.0)


def test_the_export_is_bounded(tmp_path):
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


def test_a_chunked_step_does_not_allocate_the_whole_step_per_chunk(restore_flags):
    """What chunking actually buys.

    The step buffer is still allocated once - assemble_track_from_data returns
    an array, so its caller holds the whole render either way. What chunking
    bounds is the *transient* on top of it: the synth's working arrays are one
    chunk wide rather than one step wide.
    """

    import tracemalloc

    duration, boundary = 120.0, 15.0
    step_bytes = int(duration * RATE) * 2 * 4

    tracemalloc.start()
    _render(_track(duration), chunked=True, boundary=boundary)
    _, chunked_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _render(_track(duration), chunked=False)
    _, whole_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert chunked_peak < whole_peak
    # The saving is the working set, so it scales with the step, not with a
    # fixed overhead being shaved off.
    assert whole_peak - chunked_peak > step_bytes

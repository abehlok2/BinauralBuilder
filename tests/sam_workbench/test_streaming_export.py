"""Exporting a long track without holding it in memory.

Exact normalization needs the peak, and the peak is not known until the last
sample exists. The export path answered that by keeping three full-length
arrays alive at once - the rendered track, a float copy scaled by the gain, and
an int16 copy to encode - which for an hour of stereo audio is about four
gigabytes to write a file.

The streaming path measures the peak while spooling float32 to temporary
storage, then reads it back in bounded blocks. These tests hold two things
together: that the memory really is bounded, and that choosing the bounded path
does not change a single sample of the output.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import soundfile as sf

from src.audio.sam_workbench.streaming import (
    DEFAULT_STREAM_CHUNK_FRAMES,
    PeakMeter,
    TrackStream,
    iter_track_blocks,
    stream_normalized_export,
)

RATE = 44100
TARGET = 0.25


def _track(frames, seed=0):
    generator = np.random.default_rng(seed)
    return (generator.normal(size=(frames, 2)) * 0.3).astype(np.float32)


def _in_memory_export(track, path):
    """Exactly what the export path did before, for comparison."""

    normalized = track * (TARGET / np.abs(track).max())
    sf.write(str(path), np.int16(np.clip(normalized * 32767, -32768, 32767)), RATE, subtype="PCM_16")


# --- the output is unchanged ------------------------------------------------


@pytest.mark.parametrize("frames", [1000, 200_000, DEFAULT_STREAM_CHUNK_FRAMES * 3 + 17])
def test_the_streamed_export_is_byte_identical_to_the_in_memory_one(tmp_path, frames):
    """Choosing the bounded path is a memory decision, not an audio one."""

    track = _track(frames)
    _in_memory_export(track, tmp_path / "memory.wav")
    stream_normalized_export(
        iter_track_blocks(track), tmp_path / "streamed.wav", RATE, target_level=TARGET
    )
    memory, _ = sf.read(str(tmp_path / "memory.wav"), dtype="int16")
    streamed, _ = sf.read(str(tmp_path / "streamed.wav"), dtype="int16")
    assert np.array_equal(memory, streamed)


def test_the_export_hits_the_target_level_exactly(tmp_path):
    stream_normalized_export(
        iter_track_blocks(_track(50_000)), tmp_path / "out.wav", RATE, target_level=TARGET
    )
    audio, _ = sf.read(str(tmp_path / "out.wav"), dtype="float32")
    assert np.abs(audio).max() == pytest.approx(TARGET, abs=1e-4)


def test_the_reported_gain_is_the_one_that_was_applied(tmp_path):
    track = _track(50_000)
    report = stream_normalized_export(
        iter_track_blocks(track), tmp_path / "out.wav", RATE, target_level=TARGET
    )
    assert report["peak"] == pytest.approx(float(np.abs(track).max()), rel=1e-6)
    assert report["gain"] == pytest.approx(TARGET / report["peak"], rel=1e-6)
    assert report["frames"] == len(track)


def test_the_block_size_does_not_change_the_output(tmp_path):
    track = _track(120_000)
    outputs = []
    for chunk in (4096, 65536, 200_000):
        path = tmp_path / f"chunk{chunk}.wav"
        stream_normalized_export(
            iter_track_blocks(track, chunk), path, RATE, target_level=TARGET,
            chunk_frames=chunk,
        )
        outputs.append(sf.read(str(path), dtype="int16")[0])
    assert all(np.array_equal(outputs[0], other) for other in outputs[1:])


# --- the memory really is bounded -------------------------------------------


def test_streaming_allocates_far_less_than_the_track_it_writes(tmp_path):
    """The whole point: peak allocation must not scale with track length."""

    track = _track(RATE * 60)  # one minute, about 21 MB as float32

    tracemalloc.start()
    _in_memory_export(track, tmp_path / "memory.wav")
    _, in_memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    stream_normalized_export(
        iter_track_blocks(track), tmp_path / "streamed.wav", RATE, target_level=TARGET
    )
    _, streamed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert streamed_peak < in_memory_peak / 4
    assert streamed_peak < 32 * 1024 * 1024


def _synthetic_blocks(frames, chunk=DEFAULT_STREAM_CHUNK_FRAMES):
    """Blocks generated on demand, so nothing holds the whole track.

    This is what a renderer producing its own blocks looks like from the
    encoder's side. ``iter_track_blocks`` bridges from a track that is already
    in memory, so measuring against it would measure that array rather than
    this path.
    """

    generator = np.random.default_rng(0)
    produced = 0
    while produced < frames:
        span = min(chunk, frames - produced)
        yield (generator.normal(size=(span, 2)) * 0.3).astype(np.float32)
        produced += span


def test_a_longer_track_does_not_cost_more_memory(tmp_path):
    """Bounded means bounded, not merely smaller."""

    peaks = []
    for seconds in (10, 40):
        tracemalloc.start()
        stream_normalized_export(
            _synthetic_blocks(RATE * seconds),
            tmp_path / f"{seconds}.wav", RATE, target_level=TARGET,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    # Four times the audio, no more memory: the encoder never accumulates.
    assert peaks[1] < peaks[0] * 1.5


def test_the_encoder_never_holds_more_than_a_few_blocks(tmp_path):
    tracemalloc.start()
    stream_normalized_export(
        _synthetic_blocks(RATE * 120), tmp_path / "long.wav", RATE, target_level=TARGET
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Two minutes of stereo float32 is about 42 MB; the encoder uses a fraction.
    assert peak < 16 * 1024 * 1024


def test_the_temporary_file_is_removed(tmp_path):
    stream = TrackStream(channels=2, directory=tmp_path)
    stream.write(_track(1000))
    path = stream.path
    assert path.exists()
    stream.close()
    assert not path.exists()


# --- robustness -------------------------------------------------------------


def test_non_finite_samples_are_counted_and_silenced(tmp_path):
    """One NaN would otherwise poison the peak and silence the whole render."""

    track = _track(10_000)
    track[5, 0] = np.nan
    track[7, 1] = np.inf
    report = stream_normalized_export(
        iter_track_blocks(track), tmp_path / "out.wav", RATE, target_level=TARGET
    )
    assert report["nonFiniteSamples"] == 2
    audio, _ = sf.read(str(tmp_path / "out.wav"), dtype="float32")
    assert np.all(np.isfinite(audio))
    assert np.abs(audio).max() == pytest.approx(TARGET, abs=1e-4)


def test_a_silent_track_is_not_amplified(tmp_path):
    report = stream_normalized_export(
        iter_track_blocks(np.zeros((10_000, 2), dtype=np.float32)),
        tmp_path / "silent.wav", RATE, target_level=TARGET,
    )
    assert report["gain"] == 1.0
    audio, _ = sf.read(str(tmp_path / "silent.wav"), dtype="float32")
    assert np.abs(audio).max() == 0.0


def test_a_stream_refuses_audio_of_the_wrong_shape(tmp_path):
    with TrackStream(channels=2, directory=tmp_path) as stream:
        with pytest.raises(ValueError, match="expected"):
            stream.write(np.zeros((10, 3), dtype=np.float32))


def test_the_peak_meter_measures_every_sample():
    """An estimate from a subset would give a different, wrong gain."""

    meter = PeakMeter()
    meter.update(np.array([[0.1, 0.2]], dtype=np.float32))
    meter.update(np.array([[0.9, -0.3]], dtype=np.float32))
    assert meter.peak == pytest.approx(0.9)
    assert meter.frames == 2


def test_the_export_path_chooses_streaming_only_for_long_tracks():
    from src.synth_functions.sound_creator import _should_stream_export
    from src.audio.sam_workbench.streaming import STREAMING_THRESHOLD_FRAMES

    short = np.zeros((1000, 2), dtype=np.float32)
    long = np.zeros((STREAMING_THRESHOLD_FRAMES, 2), dtype=np.float32)
    assert not _should_stream_export(short, "out.wav")
    assert _should_stream_export(long, "out.wav")
    # MP3 goes through pydub, which wants the whole array anyway.
    assert not _should_stream_export(long, "out.mp3")

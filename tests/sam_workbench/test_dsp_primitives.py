"""Envelopes, block iteration, mixing, and the output-safety stage.

The rule these primitives exist to protect: one gain for both ears, always. A
per-channel fade, mix, or limiter would alter the interaural cues the workbench
is built to control.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.conventions import db_to_linear
from src.audio.sam_workbench.dsp import (
    DEFAULT_CROSSFADE_MS,
    RenderBlock,
    RenderContext,
    apply_fade_in,
    apply_fade_out,
    apply_shared_gain,
    crossfade_blocks,
    equal_power_crossfade,
    iter_blocks,
    limit_lookahead,
    linear_fade,
    milliseconds_to_frames,
    mix_stems,
    peak_level,
    safety_clamp,
    shared_normalization_gain,
    true_peak_estimate_db,
)

SAMPLE_RATE = 44_100


# --- blocks -----------------------------------------------------------------


@pytest.mark.parametrize(("frames", "block_size"), [(0, 64), (1, 64), (100, 64), (256, 256), (256, 1)])
def test_blocks_tile_the_range_exactly_once(frames, block_size):
    blocks = list(iter_blocks(frames, block_size))
    assert sum(block.frames for block in blocks) == frames
    position = 0
    for block in blocks:
        assert block.start_sample == position
        position = block.end_sample


def test_blocks_keep_the_absolute_start_sample():
    blocks = list(iter_blocks(10, 4, start_sample=1000))
    assert [block.start_sample for block in blocks] == [1000, 1004, 1008]
    assert blocks[-1].frames == 2
    assert RenderBlock(1000, 4).end_sample == 1004


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_block_sizes_are_rejected(block_size):
    with pytest.raises(ValueError):
        list(iter_blocks(10, block_size))
    with pytest.raises(ValueError):
        RenderContext(sample_rate_hz=SAMPLE_RATE, block_size=block_size)


# --- envelopes --------------------------------------------------------------


def test_linear_fade_spans_zero_to_one():
    rising = linear_fade(64)
    assert rising[0] == 0.0
    assert rising[-1] == 1.0
    np.testing.assert_allclose(linear_fade(64, rising=False), rising[::-1])
    assert linear_fade(0).shape == (0,)
    assert linear_fade(1)[0] == 1.0


def test_equal_power_crossfade_conserves_power():
    fade_out, fade_in = equal_power_crossfade(128)
    np.testing.assert_allclose(fade_out**2 + fade_in**2, np.ones(128), atol=1e-12)
    assert fade_out[0] == pytest.approx(1.0)
    assert fade_in[-1] == pytest.approx(1.0)


def test_crossfade_between_blocks_stays_bounded():
    frames = 256
    outgoing = np.ones((2, frames))
    incoming = -np.ones((2, frames))
    faded = crossfade_blocks(outgoing, incoming)
    assert faded.shape == (2, frames)
    assert np.max(np.abs(faded)) <= 1.0 + 1e-12
    assert faded[0, 0] == pytest.approx(1.0)
    assert faded[0, -1] == pytest.approx(-1.0)


def test_crossfade_rejects_mismatched_or_mono_blocks():
    with pytest.raises(ValueError):
        crossfade_blocks(np.ones((2, 10)), np.ones((2, 11)))
    with pytest.raises(ValueError):
        crossfade_blocks(np.ones((1, 10)), np.ones((1, 10)))


def test_fades_apply_to_both_channels_identically():
    audio = np.ones((2, 100))
    audio[1] *= 0.5
    apply_fade_in(audio, 10)
    apply_fade_out(audio, 10)
    np.testing.assert_allclose(audio[0] * 0.5, audio[1], atol=1e-15)
    assert audio[0, 0] == 0.0
    assert audio[0, -1] == 0.0
    assert audio[0, 50] == 1.0


def test_millisecond_conversions_round_to_whole_frames():
    assert milliseconds_to_frames(DEFAULT_CROSSFADE_MS, SAMPLE_RATE) == 353
    assert milliseconds_to_frames(0.0, SAMPLE_RATE) == 0
    assert milliseconds_to_frames(-5.0, SAMPLE_RATE) == 0


# --- mixing -----------------------------------------------------------------


def test_mixing_sums_stems_into_one_buffer():
    first = np.ones((2, 10))
    second = np.full((2, 4), 0.5)
    mixed = mix_stems((first, second), 10)
    assert mixed.shape == (2, 10)
    np.testing.assert_allclose(mixed[:, :4], 1.5)
    np.testing.assert_allclose(mixed[:, 4:], 1.0)


def test_mixing_rejects_frame_major_stems():
    with pytest.raises(ValueError):
        mix_stems((np.ones((10, 2)),), 10)


def test_shared_gain_scales_both_channels_by_the_same_factor():
    audio = np.vstack((np.ones(8), np.full(8, 0.25)))
    scaled = apply_shared_gain(audio, -6.0)
    np.testing.assert_allclose(scaled[0], db_to_linear(-6.0))
    np.testing.assert_allclose(scaled[1] / scaled[0], 0.25)


def test_normalization_gain_is_returned_not_applied():
    audio = np.vstack((np.full(8, 0.5), np.full(8, 0.1)))
    gain = shared_normalization_gain(audio, 1.0)
    assert gain == pytest.approx(2.0)
    assert peak_level(audio) == pytest.approx(0.5)
    assert shared_normalization_gain(np.zeros((2, 4)), 1.0) == 1.0


# --- output safety ----------------------------------------------------------


def test_limiter_holds_the_ceiling_without_touching_the_ild():
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 220.0 * times)
    audio = np.vstack((tone, 0.5 * tone)) * 1.5

    limited, report = limit_lookahead(audio, SAMPLE_RATE, ceiling_dbfs=-1.0)
    assert np.max(np.abs(limited)) <= db_to_linear(-1.0) + 1e-9
    assert report.max_gain_reduction_db > 0.0
    assert report.peak_dbfs_after == pytest.approx(-1.0, abs=0.01)
    assert report.clipped_samples == 0
    # The channel ratio is untouched: one gain envelope, both ears.
    np.testing.assert_allclose(limited[1], limited[0] * 0.5, atol=1e-12)


def test_limiter_leaves_quiet_audio_alone():
    audio = np.full((2, 128), 0.1)
    limited, report = limit_lookahead(audio, SAMPLE_RATE)
    np.testing.assert_array_equal(limited, audio)
    assert report.max_gain_reduction_db == 0.0


def test_limiter_reacts_before_a_transient_arrives():
    audio = np.zeros((2, 4096))
    audio[:, 2048:2148] = 4.0
    limited, _ = limit_lookahead(audio, SAMPLE_RATE, ceiling_dbfs=-1.0, lookahead_ms=5.0)
    assert np.max(np.abs(limited[:, 2048:2148])) <= db_to_linear(-1.0) + 1e-9


def test_safety_clamp_is_immediate_and_shared():
    audio = np.vstack((np.full(16, 2.0), np.full(16, 1.0)))
    clamped, report = safety_clamp(audio, ceiling_dbfs=-6.0)
    assert np.max(np.abs(clamped)) <= db_to_linear(-6.0) + 1e-12
    np.testing.assert_allclose(clamped[1], clamped[0] * 0.5, atol=1e-12)
    assert report.lookahead_ms == 0.0


def test_true_peak_estimate_is_at_least_the_sample_peak():
    times = np.arange(512) / SAMPLE_RATE
    audio = np.vstack((np.sin(2 * np.pi * 10_000.0 * times),) * 2) * 0.5
    sample_peak_db = 20 * np.log10(np.max(np.abs(audio)))
    assert true_peak_estimate_db(audio) >= sample_peak_db - 1e-9
    assert np.isfinite(true_peak_estimate_db(audio))

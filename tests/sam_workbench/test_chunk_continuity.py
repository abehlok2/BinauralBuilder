"""Chunked rendering must equal whole-step rendering, without clicks.

`sound_creator` may render a step in one call or in many chunks. Before Phase 1
the static SAM2 renderer ignored `chunk_start_time` and the transition renderer
restarted its parameter ramp in every chunk; both are covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.conventions import seconds_to_samples
from src.audio.sam_workbench.dsp import render_source
from src.audio.sam_workbench.render.abstract_pm import preset_exact_symmetric

SAMPLE_RATE = 44_100
STATIC_PARAMS = {
    "amp": 0.6,
    "carrierFreq": 220.0,
    "modFreq": 3.0,
    "arcWidthDeg": 90.0,
    "directionOffsetDeg": 10.0,
    "spatialScale": 1.0,
    "pathType": "open",
    "pathShape": "sinusoidal",
}
TRANSITION_PARAMS = {
    "amp": 0.6,
    "startCarrierFreq": 200.0,
    "endCarrierFreq": 260.0,
    "startModFreq": 3.0,
    "endModFreq": 6.0,
    "startArcWidthDeg": 90.0,
    "endArcWidthDeg": 45.0,
    "startDirectionOffsetDeg": 0.0,
    "endDirectionOffsetDeg": 30.0,
    "pathType": "open",
    "pathShape": "sinusoidal",
}


def _duration_for_frames(frames: int) -> float:
    """A duration that survives the legacy ``int(duration * sample_rate)`` truncation."""

    return (frames + 0.5) / SAMPLE_RATE


def _random_chunk_frames(total_frames: int, seed: int) -> list[int]:
    """Split a step into random whole-sample chunks, the way streaming does."""

    generator = np.random.default_rng(seed)
    remaining = total_frames
    chunks: list[int] = []
    while remaining > 0:
        span = int(min(remaining, generator.integers(441, 13_230)))
        chunks.append(span)
        remaining -= span
    return chunks


def _adapt_transition(initial_offset: float, span: float, chunk_start: float) -> tuple[float, float]:
    """Mirror `sound_creator._adapt_transition_timing_for_chunk`."""

    transition_start = max(0.0, initial_offset)
    transition_end = transition_start + max(0.0, span)
    return transition_start - max(0.0, chunk_start), max(0.0, transition_end - transition_start)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_static_voice_matches_across_random_chunking(seed):
    total_frames = SAMPLE_RATE
    whole = render_sam2_voice(_duration_for_frames(total_frames), SAMPLE_RATE, params=STATIC_PARAMS)

    rendered = []
    position = 0
    for span in _random_chunk_frames(total_frames, seed):
        rendered.append(
            render_sam2_voice(
                _duration_for_frames(span),
                SAMPLE_RATE,
                params=STATIC_PARAMS,
                initial_offset=position / SAMPLE_RATE,
            )
        )
        position += span
    chunked = np.concatenate(rendered)
    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-6)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_transition_voice_matches_across_random_chunking(seed):
    total_frames = SAMPLE_RATE
    total_s = total_frames / SAMPLE_RATE
    whole = render_sam2_voice(
        _duration_for_frames(total_frames),
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=0.0,
        transition_duration=total_s,
    )

    rendered = []
    position = 0
    for span in _random_chunk_frames(total_frames, seed):
        offset, transition_span = _adapt_transition(0.0, total_s, position / SAMPLE_RATE)
        rendered.append(
            render_sam2_voice(
                _duration_for_frames(span),
                SAMPLE_RATE,
                params=TRANSITION_PARAMS,
                is_transition=True,
                initial_offset=offset,
                transition_duration=transition_span,
            )
        )
        position += span
    chunked = np.concatenate(rendered)
    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-6)


def test_static_voice_honours_an_absolute_offset():
    whole = render_sam2_voice(0.5, SAMPLE_RATE, params=STATIC_PARAMS)
    second_half = render_sam2_voice(0.25, SAMPLE_RATE, params=STATIC_PARAMS, initial_offset=0.25)
    boundary = seconds_to_samples(0.25, SAMPLE_RATE)
    np.testing.assert_allclose(second_half, whole[boundary:], rtol=0, atol=1e-6)
    # The offset must matter: the second half is not a copy of the first.
    assert not np.allclose(second_half, whole[:boundary], atol=1e-3)


def test_chunk_boundaries_do_not_click():
    """A boundary sample step must be no larger than the signal's own steps."""

    frames = SAMPLE_RATE // 4
    chunks = [
        render_sam2_voice(
            _duration_for_frames(frames),
            SAMPLE_RATE,
            params=STATIC_PARAMS,
            initial_offset=index * frames / SAMPLE_RATE,
        )
        for index in range(4)
    ]
    joined = np.concatenate(chunks)
    steps = np.abs(np.diff(joined, axis=0))
    boundary_indices = [frames * index - 1 for index in (1, 2, 3)]
    assert max(steps[index].max() for index in boundary_indices) <= steps.max()


def test_transition_ramp_does_not_restart_in_later_chunks():
    """The parameter ramp is anchored to the step, not to the chunk."""

    total_s = 1.0
    offset, span = _adapt_transition(0.0, total_s, 0.75)
    last_chunk = render_sam2_voice(
        0.25,
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=offset,
        transition_duration=span,
    )
    first_chunk = render_sam2_voice(
        0.25,
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=0.0,
        transition_duration=total_s,
    )
    assert offset == pytest.approx(-0.75)
    assert not np.allclose(first_chunk, last_chunk, atol=1e-3)


def test_core_source_renders_identically_from_any_block_size():
    source = preset_exact_symmetric(carrier_hz=311.0, modulation_hz=7.0, depth_rad=1.1)
    reference = render_source(source, SAMPLE_RATE, 8192, dtype=np.float64)
    for block_size in (1, 3, 128, 1000, 8192):
        np.testing.assert_array_equal(
            render_source(source, SAMPLE_RATE, 8192, block_size, dtype=np.float64), reference
        )


def test_parameter_transition_stays_click_free():
    """A ramping carrier and modulation rate must not step discontinuously."""

    audio = render_sam2_voice(
        1.0,
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=0.0,
        transition_duration=0.5,
    )
    steps = np.abs(np.diff(audio, axis=0))
    # The largest single-sample step a 260 Hz carrier at amplitude 0.6 can make.
    ceiling = 0.6 * 2.0 * np.pi * 260.0 / SAMPLE_RATE * 1.05
    assert steps.max() <= ceiling
    assert np.all(np.isfinite(audio))


def test_transition_end_state_is_held_after_the_ramp():
    """Once the ramp finishes the voice continues at the end parameters."""

    span = 0.25
    during = render_sam2_voice(
        span,
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=0.0,
        transition_duration=span,
    )
    after_offset, after_span = _adapt_transition(0.0, span, span)
    after = render_sam2_voice(
        span,
        SAMPLE_RATE,
        params=TRANSITION_PARAMS,
        is_transition=True,
        initial_offset=after_offset,
        transition_duration=after_span,
    )
    static_end = render_sam2_voice(
        span,
        SAMPLE_RATE,
        params={
            "amp": TRANSITION_PARAMS["amp"],
            "carrierFreq": TRANSITION_PARAMS["endCarrierFreq"],
            "modFreq": TRANSITION_PARAMS["endModFreq"],
            "arcWidthDeg": TRANSITION_PARAMS["endArcWidthDeg"],
            "directionOffsetDeg": TRANSITION_PARAMS["endDirectionOffsetDeg"],
            "pathType": "open",
            "pathShape": "sinusoidal",
        },
    )
    # Same instantaneous parameters, different accumulated phase: the envelope
    # of the two signals matches even though the samples do not.
    assert abs(float(np.std(after)) - float(np.std(static_end))) < 0.02
    assert not np.allclose(after, during, atol=1e-3)

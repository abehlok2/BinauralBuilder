"""A fast path must not buzz.

A 21-turn torus moves about 1.1 degrees per 128-sample control interval, which
is past the renderer's 1 degree tolerance, so it reselects the HRTF at the
fastest rate the control grid allows - roughly 345 times a second. The
transition between two filters was 441 samples long (10 ms) and the interval
between requests was 128, so no transition could ever finish. Each new request
replaced the running one, and the audible signal jumped from a partly-completed
mixture straight to the incoming filter in a single sample.

That is a discontinuity per control interval: a buzz at the control rate, a
harmonic ladder above it, sidebands around the carrier, and broadband energy
from the steps themselves. These tests pin the properties that make it not
happen: transitions are never abandoned, and each one ends exactly where the
next begins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.dsp.binaural_convolution import (
    FADE_EQUAL_POWER,
    FADE_LINEAR,
    BinauralConvolver,
    BinauralFilterPair,
)
from src.audio.sam_workbench.dsp.blocks import iter_blocks
from src.audio.sam_workbench.render.hrtf import HRTFRenderer, HRTFRendererSpec
from src.audio.sam_workbench.trajectory.primitives import Torus

SOFA = str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa")
RATE = 44100
CARRIER_HZ = 200.0
DURATION_S = 4.0
#: The parameters from the report: 21 complete orbits of the listener.
TORUS = Torus(major_radius_m=1.5, minor_radius_m=0.4, major_turns=21.0, minor_turns=1.0)


def _trajectory(t):
    return TORUS.evaluate(np.asarray(t, dtype=np.float64) / DURATION_S)


def _spec(**overrides):
    settings = dict(
        sofa_path=SOFA,
        trajectory=_trajectory,
        crossfade_ms=10.0,
        control_interval_samples=128,
        min_control_interval_samples=128,
        max_angular_error_deg=1.0,
        interpolation="delay_magnitude",
    )
    settings.update(overrides)
    return HRTFRendererSpec(**settings)


def _carrier(frames):
    return 0.5 * np.sin(2.0 * np.pi * CARRIER_HZ * np.arange(frames) / RATE)


def _render(block_size=1024, frames=None, **overrides):
    from src.audio.sam_workbench.dsp.blocks import RenderContext

    frames = int(RATE * DURATION_S) if frames is None else frames
    renderer = HRTFRenderer(_spec(**overrides))
    renderer.prepare(RenderContext(sample_rate_hz=RATE, block_size=block_size))
    mono = _carrier(frames)
    pieces = [
        renderer.process(mono[block.start_sample : block.end_sample], block)
        for block in iter_blocks(frames, block_size)
    ]
    return np.concatenate(pieces, axis=1), renderer.diagnostics()


# --- the renderer under the demanding path ----------------------------------


def test_the_torus_really_does_request_a_filter_every_control_interval():
    """Without this the rest of the file would be testing an easy case."""

    _, diagnostics = _render()
    expected = int(RATE * DURATION_S) // 128
    assert diagnostics["filter_requests"] >= 0.95 * expected


def test_no_transition_is_ever_abandoned():
    """The defect itself: a fade replaced part-way is a discontinuity."""

    _, diagnostics = _render()
    assert diagnostics["fade_restarts"] == 0
    assert diagnostics["mid_fade_requests"] == 0
    assert diagnostics["queued_filter_updates"] == 0


def test_every_transition_finishes_inside_its_own_control_interval():
    """Which is what makes each interval end where the next one starts."""

    _, diagnostics = _render()
    assert diagnostics["maximum_filter_age_samples"] <= 128


def test_the_output_has_no_step_larger_than_the_signal_itself_makes():
    """A restarted fade showed up here as a jump many times the input's slew."""

    audio, _ = _render()
    mono = _carrier(audio.shape[1])
    rendered_step = float(np.abs(np.diff(audio.astype(np.float64), axis=1)).max())
    source_step = float(np.abs(np.diff(mono)).max())

    # The HRIRs have gain, so the output's slew is not bounded by the input's
    # outright; it is bounded by a small multiple of it. The defect produced
    # roughly 35x, against the 3x allowed here.
    assert rendered_step < 3.0 * source_step


def test_energy_stays_near_the_carrier():
    """The buzz was broadband, and it was loud enough to measure as such."""

    audio, _ = _render()
    spectrum = np.abs(np.fft.rfft(audio[0].astype(np.float64)))
    freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / RATE)
    # The path modulates the carrier, so allow generous room around it and
    # count everything well above as spill.
    near = (freqs > 20.0) & (freqs < 2.0 * CARRIER_HZ)
    far = freqs >= 2.0 * CARRIER_HZ
    ratio = float(np.sum(spectrum[far] ** 2) / np.sum(spectrum[near] ** 2))

    # Measured at 4.8e-3 with transitions being restarted, and 1.2e-6 without.
    assert ratio < 1e-4


@pytest.mark.parametrize("block_size", [128, 256, 512, 1024, 4096])
def test_the_fix_does_not_depend_on_the_caller_s_block_size(block_size):
    """Transitions are placed on the absolute grid, not on block boundaries."""

    reference, _ = _render(block_size=1024)
    rendered, diagnostics = _render(block_size=block_size)
    assert diagnostics["fade_restarts"] == 0
    assert np.abs(rendered - reference).max() < 2e-6


# --- the convolver's transition contract ------------------------------------


def _pair(seed, taps=32):
    return np.random.default_rng(seed).normal(size=(2, taps))


def test_a_request_arriving_mid_fade_waits_rather_than_restarting():
    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=10.0)
    convolver.set_filters(_pair(0))
    convolver.set_filters(_pair(1))
    assert convolver.is_fading

    convolver.process(np.zeros(64))
    convolver.set_filters(_pair(2))

    assert convolver.counters["fade_restarts"] == 0
    assert convolver.counters["mid_fade_requests"] == 1
    assert convolver.has_queued_filters is True


def test_the_newest_waiting_request_is_the_one_that_happens():
    """An older waiting request is out of date before it is ever heard."""

    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=10.0)
    convolver.set_filters(_pair(0))
    convolver.set_filters(_pair(1))
    convolver.process(np.zeros(32))

    convolver.set_filters(_pair(2))
    newest = _pair(3)
    convolver.set_filters(newest)

    assert convolver.counters["dropped_filter_updates"] == 1
    # Run past the end of the fade so the queued pair is promoted.
    convolver.process(np.zeros(1024))
    assert np.array_equal(convolver._current.taps, newest)
    assert convolver.counters["fade_restarts"] == 0


def test_a_queued_transition_begins_exactly_where_the_previous_one_ended():
    """Not at the next block boundary, which would depend on the block size.

    Both runs request the third filter at the same absolute sample, mid-fade.
    Only how the stream is cut into blocks differs, and that must not reach the
    audio: the queued transition starts where the running one finished, which
    is a position on the sample timeline rather than in the caller's loop.
    """

    request_at = 300  # inside the 441-sample fade started below
    signal = np.sin(2.0 * np.pi * 300.0 * np.arange(4096) / RATE)

    def run(block):
        convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=10.0)
        convolver.set_filters(_pair(0))
        convolver.set_filters(_pair(1))
        pieces, position = [], 0
        while position < signal.size:
            span = min(block, signal.size - position)
            if position < request_at < position + span:
                span = request_at - position
            pieces.append(convolver.process(signal[position : position + span]))
            position += span
            if position == request_at:
                convolver.set_filters(_pair(2))
        return np.concatenate(pieces, axis=1)

    assert np.abs(run(128) - run(1000)).max() < 1e-9
    assert np.abs(run(4096) - run(1000)).max() < 1e-9


def test_a_linear_fade_does_not_bulge_where_an_equal_power_one_does():
    """Two filtered copies of one carrier are correlated, so they add.

    Equal power holds the sum of squares constant, which is right for unrelated
    signals and wrong here: at the midpoint both weights are 0.707, so nearly
    identical filters sum to 1.414 - about 3 dB of gain, once per transition.
    """

    taps = np.zeros((2, 8))
    taps[:, 0] = 1.0
    nudged = taps.copy()
    nudged[:, 0] = 0.999

    def midpoint_gain(curve):
        convolver = BinauralConvolver(
            sample_rate_hz=RATE, crossfade_ms=10.0, fade_curve=curve
        )
        convolver.set_filters(taps)
        convolver.set_filters(nudged)
        steady = np.ones(220)  # half of a 441-sample fade
        convolver.process(steady)
        return float(convolver.process(np.ones(1))[0, 0])

    assert midpoint_gain(FADE_EQUAL_POWER) > 1.35
    assert abs(midpoint_gain(FADE_LINEAR) - 1.0) < 0.01


def test_the_fade_restart_counter_can_actually_fire():
    """The tripwire has to be able to trip, or asserting zero proves nothing.

    Every other assertion in this module says ``fade_restarts`` is zero. That
    was true for the wrong reason for a while: nothing incremented it, so it
    would have read zero however badly a later change behaved. This reaches
    past the public interface on purpose - no caller can displace a running
    fade any more - to show the counter notices when one is displaced.
    """

    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=10.0)
    convolver.set_filters(_pair(0))
    convolver.set_filters(_pair(1))
    convolver.process(np.zeros(32))
    assert convolver.is_fading
    assert convolver.counters["fade_restarts"] == 0

    convolver._begin(BinauralFilterPair(_pair(2)), 480, "linear")

    assert convolver.counters["fade_restarts"] == 1


def test_a_fade_that_runs_to_completion_is_not_counted_as_a_restart():
    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=10.0)
    convolver.set_filters(_pair(0))
    convolver.set_filters(_pair(1), fade_frames=64)
    convolver.process(np.zeros(128))

    convolver._begin(BinauralFilterPair(_pair(2)), 64, "linear")

    assert convolver.counters["fade_restarts"] == 0


def test_the_angular_error_diagnostic_reports_what_the_bound_reached():
    """Under motion the minimum interval can no longer keep up, the bound stops
    holding, and the diagnostic is what says by how much."""

    def diagnostics_for(turns):
        path = Torus(
            major_radius_m=1.5, minor_radius_m=0.4, major_turns=turns, minor_turns=1.0
        )
        _, report = _render(
            frames=RATE, trajectory=lambda t: path.evaluate(np.asarray(t, dtype=np.float64))
        )
        return report

    slow = diagnostics_for(1.0)
    fast = diagnostics_for(40.0)

    assert slow["maximum_angular_error_during_fade"] > 0.0
    assert fast["maximum_angular_error_during_fade"] > slow["maximum_angular_error_during_fade"]
    # Degrading, not breaking: no transition is abandoned at any speed.
    assert slow["fade_restarts"] == 0
    assert fast["fade_restarts"] == 0

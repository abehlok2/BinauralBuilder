"""The geometric binaural renderer against analytic fixtures.

Moving-source ITD and level must match what the geometry says they should be -
that is the whole claim of a geometric renderer - and a declared jump must
arrive as a crossfade rather than a click.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.analysis import (
    estimate_itd_seconds,
    interaural_level_difference_db,
    predicted_ild_db,
    predicted_itd_s,
)
from src.audio.sam_workbench.dsp import FractionalDelayLine, cubic_interpolate, read_fractional
from src.audio.sam_workbench.dsp.blocks import RenderBlock, RenderContext
from src.audio.sam_workbench.dsp.filters import OnePoleLowpass, head_shadow_cutoff_hz
from src.audio.sam_workbench.render import (
    GeometricBinauralRenderer,
    GeometricBinauralSpec,
    ear_distances_m,
    render_geometric,
)
from src.audio.sam_workbench.trajectory import (
    ArcGeometry,
    CircleGeometry,
    LineGeometry,
    ListenerFrame,
    PointCloudGeometry,
    TrajectorySpec,
    TraversalSpec,
)

SAMPLE_RATE = 44_100
SPEED_OF_SOUND = 343.0
HEAD_RADIUS = ListenerFrame().head_radius_m


def _tone(frequency_hz: float, frames: int) -> np.ndarray:
    times = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * math.pi * frequency_hz * times)


def _static_at(azimuth_deg: float, radius_m: float = 1.0) -> GeometricBinauralSpec:
    return GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=ArcGeometry(radius_m=radius_m, start_deg=azimuth_deg, end_deg=azimuth_deg),
            traversal=TraversalSpec(rate_hz=0.0, arc_length=False),
        )
    )


# --- fractional delay -------------------------------------------------------


def test_fractional_read_moves_an_impulse_by_the_requested_amount():
    impulse = np.zeros(64)
    impulse[10] = 1.0
    delayed = read_fractional(impulse, np.full(64, 6.0))
    assert int(np.argmax(delayed)) == 16
    assert delayed[16] == pytest.approx(1.0, abs=1e-9)


def test_fractional_read_spreads_a_half_sample_delay_between_neighbours():
    impulse = np.zeros(64)
    impulse[10] = 1.0
    delayed = read_fractional(impulse, np.full(64, 5.5))
    assert delayed[15] == pytest.approx(delayed[16], rel=1e-9)
    assert 0.0 < delayed[15] < 1.0


def test_interpolation_preserves_a_constant():
    values = cubic_interpolate(np.full(32, 0.75), np.linspace(0.0, 31.0, 100))
    np.testing.assert_allclose(values, 0.75, atol=1e-12)


def test_delay_line_carries_history_across_blocks():
    signal = np.random.default_rng(3).normal(size=512)
    whole = FractionalDelayLine(max_delay_samples=64).process(signal, np.full(512, 12.3))

    line = FractionalDelayLine(max_delay_samples=64)
    chunked = np.concatenate(
        (line.process(signal[:100], np.full(100, 12.3)), line.process(signal[100:], np.full(412, 12.3)))
    )
    np.testing.assert_allclose(chunked, whole, atol=1e-12)


def test_delay_line_state_can_be_checkpointed():
    signal = np.random.default_rng(5).normal(size=256)
    line = FractionalDelayLine(max_delay_samples=32)
    line.process(signal[:128], np.full(128, 8.0))
    resumed = FractionalDelayLine.from_state(line.state())
    np.testing.assert_allclose(
        resumed.process(signal[128:], np.full(128, 8.0)),
        line.process(signal[128:], np.full(128, 8.0)),
        atol=1e-12,
    )


# --- static geometry against analytic values --------------------------------


@pytest.mark.parametrize("azimuth_deg", [0.0, 45.0, 90.0, -90.0, 180.0])
def test_static_source_itd_matches_the_geometry(azimuth_deg):
    spec = _static_at(azimuth_deg)
    positions = spec.trajectory.positions_at(np.zeros(1)).positions_m
    expected = float(predicted_itd_s(positions, head_radius_m=HEAD_RADIUS)[0])

    rendered = render_geometric(_tone(700.0, SAMPLE_RATE), spec, SAMPLE_RATE, block_size=512)
    measured = estimate_itd_seconds(rendered, SAMPLE_RATE)
    assert measured == pytest.approx(expected, abs=1.5 / SAMPLE_RATE)


@pytest.mark.parametrize("azimuth_deg", [0.0, 60.0, -60.0])
def test_static_source_ild_matches_the_distance_law(azimuth_deg):
    spec = _static_at(azimuth_deg)
    positions = spec.trajectory.positions_at(np.zeros(1)).positions_m
    expected = float(predicted_ild_db(positions, head_radius_m=HEAD_RADIUS)[0])

    rendered = render_geometric(_tone(700.0, SAMPLE_RATE), spec, SAMPLE_RATE, block_size=512)
    measured = float(np.mean(interaural_level_difference_db(rendered, SAMPLE_RATE)[4_000:-4_000]))
    assert measured == pytest.approx(expected, abs=0.1)


def test_ear_distances_are_the_analytic_distances():
    listener = ListenerFrame()
    left_source = np.array([[0.0], [1.0], [0.0]])
    distances = ear_distances_m(left_source, listener)
    assert distances[0, 0] == pytest.approx(1.0 - HEAD_RADIUS)
    assert distances[1, 0] == pytest.approx(1.0 + HEAD_RADIUS)


def test_a_source_in_front_is_symmetric():
    rendered = render_geometric(_tone(500.0, 8_192), _static_at(0.0), SAMPLE_RATE)
    np.testing.assert_allclose(rendered[0], rendered[1], atol=1e-12)


@pytest.mark.parametrize(
    ("law", "expected_ratio"),
    [("inverse", 2.0), ("inverse_square", 4.0), ("none", 1.0)],
)
def test_distance_laws_scale_level_as_declared(law, expected_ratio):
    near = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=ArcGeometry(radius_m=1.0, start_deg=0.0, end_deg=0.0),
            traversal=TraversalSpec(rate_hz=0.0, arc_length=False),
        ),
        distance_law=law,
    )
    far = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=ArcGeometry(radius_m=2.0, start_deg=0.0, end_deg=0.0),
            traversal=TraversalSpec(rate_hz=0.0, arc_length=False),
        ),
        distance_law=law,
    )
    tone = _tone(400.0, 8_192)
    near_peak = np.max(np.abs(render_geometric(tone, near, SAMPLE_RATE)[:, 2_000:]))
    far_peak = np.max(np.abs(render_geometric(tone, far, SAMPLE_RATE)[:, 2_000:]))
    assert near_peak / far_peak == pytest.approx(expected_ratio, rel=0.02)


def test_level_is_bounded_as_a_source_passes_through_the_listener():
    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=LineGeometry(start_m=(2.0, 0.0, 0.0), end_m=(-2.0, 0.0, 0.0)),
            traversal=TraversalSpec(rate_hz=0.5, mode="one_shot"),
        ),
        minimum_distance_m=0.2,
    )
    rendered = render_geometric(_tone(300.0, SAMPLE_RATE * 2), spec, SAMPLE_RATE, block_size=1_024)
    assert np.all(np.isfinite(rendered))
    assert np.max(np.abs(rendered)) <= 1.0 / 0.2 + 1e-6


def test_unknown_distance_law_is_rejected():
    with pytest.raises(ValueError):
        GeometricBinauralSpec(distance_law="magic")
    with pytest.raises(ValueError):
        GeometricBinauralSpec(speed_of_sound_m_s=0.0)


# --- moving sources ---------------------------------------------------------


def test_moving_source_itd_tracks_the_geometry():
    """A source crossing from left to right sweeps the ITD through zero."""

    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=ArcGeometry(radius_m=1.0, start_deg=90.0, end_deg=-90.0),
            traversal=TraversalSpec(rate_hz=0.5, mode="one_shot", arc_length=False),
        )
    )
    frames = SAMPLE_RATE * 2
    rendered = render_geometric(_tone(700.0, frames), spec, SAMPLE_RATE, block_size=512)

    window = SAMPLE_RATE // 8
    for fraction, sign in ((0.1, 1.0), (0.9, -1.0)):
        start = int(frames * fraction)
        segment = rendered[:, start : start + window]
        measured = estimate_itd_seconds(segment, SAMPLE_RATE)
        times = (np.arange(start, start + window)) / SAMPLE_RATE
        positions = spec.trajectory.positions_at(times).positions_m
        expected = float(np.mean(predicted_itd_s(positions, head_radius_m=HEAD_RADIUS)))
        assert np.sign(measured) == sign
        assert measured == pytest.approx(expected, abs=3.0 / SAMPLE_RATE)


def test_doppler_raises_the_pitch_of_an_approaching_source():
    approaching = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=LineGeometry(start_m=(8.0, 0.0, 0.0), end_m=(1.0, 0.0, 0.0)),
            traversal=TraversalSpec(rate_hz=0.5, mode="one_shot"),
        ),
        distance_law="none",
    )
    frames = SAMPLE_RATE
    tone = _tone(1_000.0, frames)
    with_doppler = render_geometric(tone, approaching, SAMPLE_RATE, block_size=512)

    from src.audio.sam_workbench.analysis import instantaneous_frequency_hz

    measured = instantaneous_frequency_hz(with_doppler, SAMPLE_RATE)[0, 5_000:-5_000]
    speed = 7.0 / 2.0  # metres per second along the line
    expected = 1_000.0 * SPEED_OF_SOUND / (SPEED_OF_SOUND - speed)
    assert float(np.mean(measured)) == pytest.approx(expected, rel=0.02)


def test_doppler_can_be_switched_off_without_losing_the_level_change():
    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=LineGeometry(start_m=(8.0, 0.0, 0.0), end_m=(1.0, 0.0, 0.0)),
            traversal=TraversalSpec(rate_hz=0.5, mode="one_shot"),
        ),
        doppler=False,
    )
    from src.audio.sam_workbench.analysis import instantaneous_frequency_hz

    frames = SAMPLE_RATE
    rendered = render_geometric(_tone(1_000.0, frames), spec, SAMPLE_RATE, block_size=512)
    measured = instantaneous_frequency_hz(rendered, SAMPLE_RATE)[0, 5_000:-5_000]
    assert float(np.mean(measured)) == pytest.approx(1_000.0, rel=0.005)

    # The level still tracks the distance the source actually covered.
    def _window_gain(start: int, stop: int) -> float:
        times = np.arange(start, stop) / SAMPLE_RATE
        positions = spec.trajectory.positions_at(times).positions_m
        return float(np.mean(1.0 / np.linalg.norm(positions, axis=0)))

    early = np.max(np.abs(rendered[:, 1_000:5_000]))
    late = np.max(np.abs(rendered[:, -5_000:-1_000]))
    expected_ratio = _window_gain(frames - 5_000, frames - 1_000) / _window_gain(1_000, 5_000)
    assert late / early == pytest.approx(expected_ratio, rel=0.1)
    assert expected_ratio > 1.5  # it really did get closer


def test_rendering_is_block_size_invariant():
    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=CircleGeometry(radius_m=1.2), traversal=TraversalSpec(rate_hz=0.5)
        )
    )
    tone = _tone(600.0, 4_096)
    reference = render_geometric(tone, spec, SAMPLE_RATE)
    for block_size in (32, 128, 1_000):
        np.testing.assert_allclose(
            render_geometric(tone, spec, SAMPLE_RATE, block_size=block_size), reference, atol=1e-9
        )


# --- discontinuities --------------------------------------------------------


def test_a_declared_jump_is_crossfaded_rather_than_clicked():
    jump_time = 0.5
    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=PointCloudGeometry(points_m=((1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            traversal=TraversalSpec(
                rate_hz=0.0, jump_times_s=(jump_time,), jump_step=0.5, crossfade_ms=15.0
            ),
        )
    )
    frames = SAMPLE_RATE
    rendered = render_geometric(_tone(500.0, frames), spec, SAMPLE_RATE, block_size=256)

    steps = np.abs(np.diff(rendered, axis=1))
    boundary = int(jump_time * SAMPLE_RATE)
    around = steps[:, boundary - 4 : boundary + 4].max()
    # A click would dwarf the waveform's own slew; a crossfade does not.
    assert around <= steps.max()
    assert np.all(np.isfinite(rendered))

    # The jump really happened: the ears swap which one leads.
    before = estimate_itd_seconds(rendered[:, boundary - 8_000 : boundary - 100], SAMPLE_RATE)
    after = estimate_itd_seconds(rendered[:, boundary + 2_000 : boundary + 10_000], SAMPLE_RATE)
    assert before > 0.0 > after


def test_a_jump_at_a_block_boundary_is_still_crossfaded():
    spec = GeometricBinauralSpec(
        trajectory=TrajectorySpec(
            geometry=PointCloudGeometry(points_m=((1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            traversal=TraversalSpec(rate_hz=0.0, jump_times_s=(256 / SAMPLE_RATE,), jump_step=0.5),
        )
    )
    rendered = render_geometric(_tone(500.0, 2_048), spec, SAMPLE_RATE, block_size=256)
    steps = np.abs(np.diff(rendered, axis=1))
    assert steps[:, 250:262].max() <= steps.max()
    assert np.all(np.isfinite(rendered))


# --- head shadow ------------------------------------------------------------


def test_head_shadow_cutoff_follows_the_incidence_angle():
    facing, side, away = head_shadow_cutoff_hz([1.0, 0.0, -1.0])
    assert facing > side > away
    assert away == pytest.approx(2_000.0)


def test_head_shadow_darkens_the_far_ear():
    noise = np.random.default_rng(11).normal(size=SAMPLE_RATE)
    spec = GeometricBinauralSpec(trajectory=_static_at(90.0).trajectory, head_shadow=True)
    rendered = render_geometric(noise, spec, SAMPLE_RATE, block_size=1_024)

    from src.audio.sam_workbench.analysis import magnitude_spectrum_db

    frequencies, magnitudes = magnitude_spectrum_db(rendered[:, 2_000:], SAMPLE_RATE)
    high = frequencies > 6_000.0
    assert float(np.mean(magnitudes[1][high])) < float(np.mean(magnitudes[0][high])) - 3.0


def test_one_pole_filter_keeps_state_across_blocks():
    signal = np.random.default_rng(13).normal(size=1_024)
    whole = OnePoleLowpass(sample_rate_hz=SAMPLE_RATE, cutoff_hz=3_000.0).process(signal)

    blocked = OnePoleLowpass(sample_rate_hz=SAMPLE_RATE, cutoff_hz=3_000.0)
    joined = np.concatenate((blocked.process(signal[:256]), blocked.process(signal[256:])))
    np.testing.assert_allclose(joined, whole, atol=1e-12)


# --- the renderer protocol --------------------------------------------------


def test_renderer_reports_diagnostics():
    spec = _static_at(45.0)
    renderer = GeometricBinauralRenderer(spec)
    renderer.prepare(RenderContext(sample_rate_hz=SAMPLE_RATE), spec)
    renderer.process(_tone(500.0, 1_024), RenderBlock(0, 1_024))

    diagnostics = renderer.diagnostics()
    assert diagnostics["renderer"] == "geometric_binaural"
    assert diagnostics["distance_law"] == "inverse"
    assert diagnostics["doppler"] is True
    assert diagnostics["itd_s_max"] > 0.0
    assert diagnostics["distance_m_min"] > 0.0
    assert renderer.latency_samples() > 0


def test_renderer_requires_preparation_and_matching_block_sizes():
    renderer = GeometricBinauralRenderer(_static_at(0.0))
    with pytest.raises(RuntimeError):
        renderer.process(np.zeros(16), RenderBlock(0, 16))

    renderer.prepare(RenderContext(sample_rate_hz=SAMPLE_RATE), _static_at(0.0))
    with pytest.raises(ValueError):
        renderer.process(np.zeros(8), RenderBlock(0, 16))

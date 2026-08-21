"""The abstract phase-modulation renderer against its mathematical reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.controls import KeyframeControl
from src.audio.sam_workbench.dsp import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    CompiledSource,
    ModulatorSpec,
    PhaseAccumulator,
    RenderBlock,
    RenderContext,
    integrate_phase,
    phase_from_constant_frequency,
    render_source,
)
from src.audio.sam_workbench.dsp.source import (
    instantaneous_frequencies,
    interaural_phase_difference,
)
from src.audio.sam_workbench.render.abstract_pm import (
    PRESET_BUILDERS,
    AbstractPMRenderer,
    peak_frequency_deviation_hz,
    preset_binaural_beat,
    preset_diotic,
    preset_exact_symmetric,
)
from src.audio.sam_workbench.model import SignalSpec, Source

SAMPLE_RATE = 44_100


def _exact_reference(
    amplitude: float,
    carrier_hz: float,
    modulation_hz: float,
    depth_rad: float,
    frames: int,
    *,
    carrier_phase: float = 0.0,
    modulation_phase: float = 0.0,
) -> np.ndarray:
    times = np.arange(frames) / SAMPLE_RATE
    carrier = 2 * math.pi * carrier_hz * times + carrier_phase
    modulation = depth_rad * np.sin(2 * math.pi * modulation_hz * times + modulation_phase)
    return np.vstack(
        (amplitude * np.sin(carrier + modulation), amplitude * np.sin(carrier - modulation))
    )


def test_exact_symmetric_matches_the_reference_equations():
    source = preset_exact_symmetric(carrier_hz=440.0, modulation_hz=4.0, depth_rad=1.2, amplitude=0.4)
    rendered = render_source(source, SAMPLE_RATE, 4096, dtype=np.float64)
    expected = _exact_reference(0.4, 440.0, 4.0, 1.2, 4096)
    np.testing.assert_allclose(rendered, expected, rtol=0, atol=1e-12)


def test_project_signal_waveform_reaches_the_canonical_renderer():
    source = Source(signal=SignalSpec(carrier_frequency_hz=220.0, waveform="square"))
    rendered = render_source(source, SAMPLE_RATE, 2048, dtype=np.float64)
    assert set(np.unique(rendered)) <= {-0.5, 0.5}


@pytest.mark.parametrize("block_size", [1, 7, 64, 127, 512, 4096, 10_000])
def test_rendering_is_block_size_invariant(block_size):
    source = preset_exact_symmetric(carrier_hz=317.0, modulation_hz=5.5, depth_rad=0.9)
    whole = render_source(source, SAMPLE_RATE, 4096, dtype=np.float64)
    blocked = render_source(source, SAMPLE_RATE, 4096, block_size, dtype=np.float64)
    np.testing.assert_array_equal(whole, blocked)


def test_rendering_from_an_absolute_offset_continues_the_waveform():
    source = preset_exact_symmetric(carrier_hz=220.0, modulation_hz=3.0, depth_rad=1.0)
    whole = render_source(source, SAMPLE_RATE, 2048, dtype=np.float64)
    tail = render_source(source, SAMPLE_RATE, 1024, start_sample=1024, dtype=np.float64)
    np.testing.assert_array_equal(whole[:, 1024:], tail)


def test_instantaneous_frequencies_are_opposed_around_the_carrier():
    carrier, modulation, depth = 220.0, 4.0, 1.0
    source = preset_exact_symmetric(carrier_hz=carrier, modulation_hz=modulation, depth_rad=depth)
    frequencies = instantaneous_frequencies(source, 0, SAMPLE_RATE, SAMPLE_RATE)
    deviation = depth * modulation
    assert frequencies[0].max() == pytest.approx(carrier + deviation, abs=0.05)
    assert frequencies[0].min() == pytest.approx(carrier - deviation, abs=0.05)
    # The two ears deviate in opposite directions at every instant.
    assert np.allclose(frequencies[0] + frequencies[1], 2 * carrier, atol=1e-6)


def test_interaural_phase_difference_is_twice_the_modulation():
    source = preset_exact_symmetric(carrier_hz=220.0, modulation_hz=4.0, depth_rad=0.8)
    ipd = interaural_phase_difference(source, 0, SAMPLE_RATE, SAMPLE_RATE)
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    np.testing.assert_allclose(ipd, 2 * 0.8 * np.sin(2 * math.pi * 4.0 * times), atol=1e-12)


def test_peak_frequency_deviation_uses_the_closed_form():
    source = preset_exact_symmetric(modulation_hz=6.0, depth_rad=2.0)
    assert peak_frequency_deviation_hz(source) == pytest.approx(12.0)


def test_legacy_polarity_is_the_mirror_of_the_canonical_mode():
    canonical = preset_exact_symmetric(carrier_hz=200.0, modulation_hz=4.0, depth_rad=1.0)
    legacy = canonical.with_polarity(EAR_POLARITY_LEGACY)
    assert canonical.ear_polarity == EAR_POLARITY_CANONICAL
    rendered_canonical = render_source(canonical, SAMPLE_RATE, 1024, dtype=np.float64)
    rendered_legacy = render_source(legacy, SAMPLE_RATE, 1024, dtype=np.float64)
    np.testing.assert_allclose(rendered_canonical[0], rendered_legacy[1], atol=1e-15)
    np.testing.assert_allclose(rendered_canonical[1], rendered_legacy[0], atol=1e-15)


def test_unknown_polarity_is_rejected():
    with pytest.raises(ValueError):
        CompiledSource(ear_polarity="left_only")


def test_diotic_preset_has_no_interaural_difference():
    rendered = render_source(preset_diotic(), SAMPLE_RATE, 1024, dtype=np.float64)
    np.testing.assert_array_equal(rendered[0], rendered[1])


def test_binaural_beat_preset_separates_the_ear_frequencies():
    source = preset_binaural_beat(carrier_hz=220.0, beat_hz=4.0)
    frequencies = instantaneous_frequencies(source, 0, SAMPLE_RATE, SAMPLE_RATE)
    assert np.mean(frequencies[0]) == pytest.approx(222.0, abs=0.01)
    assert np.mean(frequencies[1]) == pytest.approx(218.0, abs=0.01)


def test_asymmetric_depth_gives_each_ear_its_own_excursion():
    source = CompiledSource(
        amplitude=0.5,
        carrier_frequency_hz=220.0,
        modulators=(ModulatorSpec(rate_hz=4.0, depth_rad=1.0),),
        left_depth_scale=1.0,
        right_depth_scale=0.25,
    )
    frequencies = instantaneous_frequencies(source, 0, SAMPLE_RATE, SAMPLE_RATE)
    left_excursion = frequencies[0].max() - frequencies[0].min()
    right_excursion = frequencies[1].max() - frequencies[1].min()
    assert left_excursion == pytest.approx(4 * right_excursion, rel=0.02)


def test_multi_modulator_sums_every_modulator():
    single = CompiledSource(modulators=(ModulatorSpec(rate_hz=4.0, depth_rad=0.8),))
    both = CompiledSource(
        modulators=(
            ModulatorSpec(rate_hz=4.0, depth_rad=0.8),
            ModulatorSpec(rate_hz=11.0, depth_rad=0.3),
        )
    )
    times = np.arange(2048) / SAMPLE_RATE
    expected_extra = 0.3 * np.sin(2 * math.pi * 11.0 * times)
    difference = interaural_phase_difference(both, 0, 2048, SAMPLE_RATE) - interaural_phase_difference(
        single, 0, 2048, SAMPLE_RATE
    )
    np.testing.assert_allclose(difference, 2 * expected_extra, atol=1e-12)


@pytest.mark.parametrize("name", sorted(PRESET_BUILDERS))
def test_every_preset_renders_finite_bounded_stereo(name):
    source = PRESET_BUILDERS[name]()
    rendered = render_source(source, SAMPLE_RATE, 4096, 512, dtype=np.float64)
    assert rendered.shape == (2, 4096)
    assert np.all(np.isfinite(rendered))
    assert np.max(np.abs(rendered)) <= 0.5 + 1e-12


# --- phase accumulation -----------------------------------------------------


def test_time_varying_frequency_integrates_instead_of_multiplying():
    """`sin(2*pi*f[n]*t[n])` is wrong for a moving frequency; the integral is right."""

    frames = SAMPLE_RATE
    sweep = KeyframeControl(keyframes=((0.0, 100.0), (1.0, 200.0)))
    frequency = sweep.render(0, frames, SAMPLE_RATE)
    phase = integrate_phase(frequency, SAMPLE_RATE)

    times = np.arange(frames) / SAMPLE_RATE
    analytic = 2 * math.pi * (100.0 * times + 0.5 * 100.0 * times**2)
    np.testing.assert_allclose(phase, analytic, atol=0.02)

    naive = 2 * math.pi * frequency * times
    assert np.max(np.abs(naive - analytic)) > 100.0


def test_phase_accumulator_resumes_across_blocks():
    frames = 4096
    sweep = KeyframeControl(keyframes=((0.0, 100.0), (0.1, 400.0)))
    frequency = sweep.render(0, frames, SAMPLE_RATE)
    whole = integrate_phase(frequency, SAMPLE_RATE)

    accumulator = PhaseAccumulator()
    blocks = [accumulator.advance(frequency[:1000], SAMPLE_RATE), accumulator.advance(frequency[1000:], SAMPLE_RATE)]
    np.testing.assert_allclose(np.concatenate(blocks), whole, atol=1e-9)
    assert accumulator.sample_index == frames
    assert PhaseAccumulator.from_state(accumulator.state()).phase_rad == accumulator.phase_rad


def test_constant_frequency_phase_is_exact_from_any_offset():
    whole = phase_from_constant_frequency(440.0, 0, 1024, SAMPLE_RATE)
    tail = phase_from_constant_frequency(440.0, 512, 512, SAMPLE_RATE)
    np.testing.assert_array_equal(whole[512:], tail)


def test_inclusive_integration_matches_the_legacy_cumsum_convention():
    frequency = np.full(16, 100.0)
    legacy = np.cumsum(2 * np.pi * frequency / SAMPLE_RATE)
    np.testing.assert_allclose(
        integrate_phase(frequency, SAMPLE_RATE, inclusive=True), legacy, atol=1e-15
    )


# --- renderer protocol ------------------------------------------------------


def test_renderer_reports_diagnostics_and_zero_latency():
    renderer = AbstractPMRenderer()
    renderer.prepare(RenderContext(sample_rate_hz=SAMPLE_RATE), preset_exact_symmetric())
    audio = renderer.process(None, RenderBlock(start_sample=0, frames=2048))
    assert audio.shape == (2, 2048)
    assert renderer.latency_samples() == 0
    diagnostics = renderer.diagnostics()
    assert diagnostics["renderer"] == "exact_symmetric_sam"
    assert diagnostics["ear_polarity"] == EAR_POLARITY_CANONICAL
    assert diagnostics["instantaneous_frequency_hz_max"] > diagnostics["instantaneous_frequency_hz_min"]


def test_renderer_requires_preparation():
    with pytest.raises(RuntimeError):
        AbstractPMRenderer().process(None, RenderBlock(0, 16))


def test_renderer_blocks_join_without_a_discontinuity():
    renderer = AbstractPMRenderer()
    renderer.prepare(RenderContext(sample_rate_hz=SAMPLE_RATE), preset_exact_symmetric(carrier_hz=440.0))
    first = renderer.process(None, RenderBlock(0, 512))
    second = renderer.process(None, RenderBlock(512, 512))
    joined = np.concatenate((first, second), axis=1)
    steps = np.abs(np.diff(joined, axis=1))
    assert steps[:, 511].max() <= steps.max()

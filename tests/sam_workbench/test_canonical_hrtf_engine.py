"""One HRTF engine, every advertised mode, and no per-sample Python convolution.

The workbench had two HRTF engines. The production one convolved sample by
sample in Python and offered two of the five interpolation modes the rest of
the workbench advertises; a second one served the HRTF Lab with block
convolution and all five. So an audition and an export of identical settings
could differ, a mode could validate and then be rendered as something else, and
production rendered slower than real time.

These tests hold the unification in place: one convolution core, every mode
reaching the audio, and configuration that cannot validate unless rendering
will honour it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.dsp.binaural_convolution import BinauralConvolver
from src.audio.sam_workbench.hrtf.interpolation import INTERPOLATION_MODES
from src.audio.sam_workbench.render.hrtf import (
    INTERPOLATION_ALIASES,
    HRTFRendererSpec,
    canonical_interpolation,
    render_hrtf,
)
from src.audio.sam_workbench.render.registry import renderer
from src.audio.sam_workbench.trajectory import spherical_to_cartesian

SOFA = str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa")
RATE = 44100
TRAJECTORY = {
    "geometry": {"type": "dome_traversal", "parameters": {"turns": 2}},
    "traversal": {"durationS": 1.0},
}
ALL_MODES = tuple(INTERPOLATION_MODES) + tuple(INTERPOLATION_ALIASES)


def _moving(t):
    times = np.asarray(t)
    return spherical_to_cartesian(60.0 * np.sin(2.0 * np.pi * 0.5 * times), 20.0, 1.5)


def _spec(**overrides):
    settings = dict(sofa_path=SOFA, trajectory=_moving)
    settings.update(overrides)
    return HRTFRendererSpec(**settings)


def _voice(**options):
    params = {
        "amp": 0.5,
        "carrierFreq": 300.0,
        "rendererMode": "hrtf",
        "hrtfAsset": SOFA,
        "canonicalTrajectory": TRAJECTORY,
        "hrtfOptions": dict(options),
    }
    return render_sam2_voice(1.0, RATE, params=params)


# --- one engine -------------------------------------------------------------


def test_both_front_doors_share_one_convolution_core():
    """Two engines convolving differently is how audition and export diverged."""

    from src.audio.sam_workbench.render.hrtf import HRTFRenderer, SpatialHrtfRenderer
    from src.audio.sam_workbench.dsp.blocks import RenderContext
    from src.audio.sam_workbench.hrtf import default_hrtf_cache

    streaming = HRTFRenderer(_spec())
    streaming.prepare(RenderContext(RATE, 512))
    dataset = default_hrtf_cache.get(SOFA, RATE, "bake_delay_into_ir", None)
    lab = SpatialHrtfRenderer(dataset, RATE)

    assert isinstance(streaming._convolver, BinauralConvolver)
    assert isinstance(lab._convolver, BinauralConvolver)


def test_production_no_longer_convolves_sample_by_sample():
    """The engine's inner loop is a block transform, not a Python for-loop."""

    import inspect

    from src.audio.sam_workbench.render.hrtf import HRTFRenderer

    source = inspect.getsource(HRTFRenderer.process)
    assert "for local, value in enumerate" not in source
    assert "_convolver.process" in inspect.getsource(HRTFRenderer._render_segment)


# --- every advertised mode --------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_advertised_mode_renders_through_production(mode):
    audio = _voice(interpolation=mode)
    assert audio.shape == (RATE, 2)
    assert np.all(np.isfinite(audio))
    assert np.abs(audio).max() > 0.0


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_advertised_mode_is_accepted_by_the_registry(mode):
    """Validation must not accept an option production rejects, or vice versa."""

    issues = renderer("hrtf").validate(
        {"rendererMode": "hrtf", "hrtfAsset": SOFA, "hrtfOptions": {"interpolation": mode}}
    )
    assert issues == ()


def test_the_registry_offers_exactly_the_modes_that_render():
    offered = set(renderer("hrtf").field("interpolation").choices)
    assert offered == set(ALL_MODES)


def test_an_unknown_mode_is_refused_by_both_the_registry_and_the_renderer():
    assert renderer("hrtf").validate(
        {"rendererMode": "hrtf", "hrtfAsset": SOFA, "hrtfOptions": {"interpolation": "bogus"}}
    )
    with pytest.raises(ValueError, match="unsupported interpolation"):
        _spec(interpolation="bogus")


def test_the_legacy_alias_renders_identically_to_its_canonical_name():
    assert canonical_interpolation("logmag_delay") == "delay_magnitude"
    assert np.allclose(_voice(interpolation="logmag_delay"), _voice(interpolation="delay_magnitude"))


def test_the_interpolation_modes_are_audibly_different():
    """A mode that renders identically to another is not really implemented."""

    rendered = {mode: _voice(interpolation=mode) for mode in INTERPOLATION_MODES}
    assert not np.allclose(rendered["nearest"], rendered["three_neighbor"], atol=1e-6)
    assert not np.allclose(rendered["nearest"], rendered["spherical_harmonic"], atol=1e-6)


# --- the whole configuration reaches the audio ------------------------------


@pytest.mark.parametrize(
    "options, label",
    [
        ({"interpolation": "three_neighbor", "neighborCount": 7}, "neighbour count"),
        ({"interpolation": "spherical_harmonic", "harmonicOrder": 0}, "harmonic order"),
        ({"listener": {"positionM": [0.5, 0.0, 0.0]}}, "listener position"),
        ({"listener": {"yawPitchRollDegrees": [90.0, 0.0, 0.0]}}, "listener orientation"),
        ({"maxAngularErrorDeg": 20.0}, "angular error bound"),
        ({"distanceLaw": "inverse_square"}, "distance law"),
        ({"crossfadeMs": 1.0}, "crossfade"),
    ],
)
def test_a_configured_setting_changes_the_rendered_audio(options, label):
    """A setting that changes nothing is a setting that is being dropped."""

    baseline = _voice(interpolation=options.get("interpolation", "nearest"))
    assert not np.allclose(baseline, _voice(**options), atol=1e-6), label


def test_the_asset_hash_is_enforced():
    params = {
        "amp": 0.5, "rendererMode": "hrtf", "hrtfAsset": SOFA,
        "hrtfAssetHash": "0" * 64, "canonicalTrajectory": TRAJECTORY,
    }
    with pytest.raises(ValueError, match="hash"):
        render_sam2_voice(0.1, RATE, params=params)


# --- block invariance and continuity ----------------------------------------


@pytest.mark.parametrize("mode", INTERPOLATION_MODES)
def test_the_render_does_not_depend_on_the_block_size(mode):
    """Filter changes happen at absolute samples, never at block boundaries."""

    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(6000) / RATE) * 0.1
    spec = _spec(interpolation=mode, crossfade_ms=8.0)
    reference = render_hrtf(mono, spec, RATE, block_size=len(mono))
    for block_size in (73, 512, 1024):
        assert np.abs(render_hrtf(mono, spec, RATE, block_size=block_size) - reference).max() < 2e-6


def test_a_filter_change_does_not_click():
    """The crossfade is what makes a direction change inaudible."""

    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(20000) / RATE) * 0.3
    faded = render_hrtf(mono, _spec(crossfade_ms=12.0), RATE, block_size=512)
    abrupt = render_hrtf(mono, _spec(crossfade_ms=0.0), RATE, block_size=512)

    def worst_step(audio):
        steps = np.abs(np.diff(audio.astype(np.float64), axis=1))
        return float(steps.max() / max(np.median(steps), 1e-12))

    assert worst_step(faded) < worst_step(abrupt)


# --- the shared convolution core --------------------------------------------


@pytest.mark.parametrize("taps, block", [(64, 512), (256, 512), (1024, 512), (4096, 512), (200, 128)])
def test_the_binaural_convolver_matches_a_direct_convolution(taps, block):
    """Both the plain and the partitioned path are exact, not approximate."""

    from scipy.signal import fftconvolve

    generator = np.random.default_rng(0)
    filters = generator.normal(size=(2, taps))
    signal = generator.normal(size=8192)

    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=0.0)
    convolver.set_filters(filters)
    rendered = np.concatenate(
        [convolver.process(signal[start : start + block]) for start in range(0, len(signal), block)],
        axis=1,
    )
    expected = np.stack([fftconvolve(signal, filters[ear])[: len(signal)] for ear in range(2)])
    assert np.abs(rendered - expected).max() < 1e-11


@pytest.mark.parametrize("blocks", [[512], [256], [1024], [128, 256, 512, 777]])
def test_the_binaural_convolver_is_block_invariant(blocks):
    generator = np.random.default_rng(1)
    filters = generator.normal(size=(2, 256)) * 0.1
    signal = np.sin(2.0 * np.pi * 220.0 * np.arange(20000) / RATE)

    def run(pattern):
        convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=0.0)
        convolver.set_filters(filters)
        pieces, position, index = [], 0, 0
        while position < len(signal):
            span = pattern[index % len(pattern)]
            pieces.append(convolver.process(signal[position : position + span]))
            position += span
            index += 1
        return np.concatenate(pieces, axis=1)

    assert np.abs(run(blocks) - run([512])).max() < 1e-12


def test_the_convolver_installs_its_first_filter_outright():
    """Fading into the first filter would open a render with a ramp."""

    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=12.0)
    assert convolver.set_filters(np.ones((2, 8))) is False
    assert convolver.is_fading is False


def test_replacing_a_filter_with_the_same_one_does_not_start_a_fade():
    """Reinstalling an unchanged filter is work, and a needless crossfade."""

    filters = np.ones((2, 8))
    convolver = BinauralConvolver(sample_rate_hz=RATE, crossfade_ms=12.0)
    convolver.set_filters(filters)
    assert convolver.set_filters(filters.copy()) is False


def test_the_convolver_refuses_to_process_before_a_filter_is_installed():
    with pytest.raises(RuntimeError, match="set_filters"):
        BinauralConvolver(sample_rate_hz=RATE).process(np.zeros(16))


# --- adaptive spatial update ------------------------------------------------


def _adaptive_counts(max_angular_error_deg, turns):
    """``(selections, filter_changes)`` over one second of a rotating path.

    Selections are what the adaptive policy governs. Filter *changes* are
    bounded by how many distinct directions the dataset measured, so on a
    coarse grid two different error bounds can install the same number of
    filters while asking very different numbers of questions.
    """

    from src.audio.sam_workbench.dsp.blocks import RenderContext, iter_blocks
    from src.audio.sam_workbench.render.hrtf import HRTFRenderer

    def path(t):
        times = np.asarray(t)
        return spherical_to_cartesian(360.0 * turns * times, 0.0, 1.5)

    renderer_ = HRTFRenderer(_spec(trajectory=path, max_angular_error_deg=max_angular_error_deg))
    renderer_.prepare(RenderContext(RATE, 512))
    mono = np.zeros(RATE)
    position = 0
    for block in iter_blocks(len(mono), 512, 0):
        renderer_.process(mono[position : position + block.frames], block)
        position += block.frames
    diagnostics = renderer_.diagnostics()
    return diagnostics["selections"], diagnostics["filter_changes"]


def test_a_slow_path_is_reselected_less_often_than_a_fast_one():
    """What matters is how far the source turned, not how much time passed."""

    slow, _ = _adaptive_counts(1.0, turns=0.25)
    fast, _ = _adaptive_counts(1.0, turns=8.0)
    assert slow < fast


def test_a_looser_error_bound_reselects_less_often():
    assert _adaptive_counts(15.0, turns=4.0)[0] < _adaptive_counts(1.0, turns=4.0)[0]


def test_a_much_looser_bound_also_installs_fewer_filters():
    """Past the dataset's own resolution, fewer questions mean fewer answers."""

    assert _adaptive_counts(60.0, turns=4.0)[1] < _adaptive_counts(1.0, turns=4.0)[1]


def test_a_fixed_interval_is_still_available():
    """Setting no error bound falls back to the interval the caller names."""

    from src.audio.sam_workbench.dsp.blocks import RenderContext, iter_blocks
    from src.audio.sam_workbench.render.hrtf import HRTFRenderer

    def path(t):
        times = np.asarray(t)
        return spherical_to_cartesian(360.0 * 4.0 * times, 0.0, 1.5)

    engine = HRTFRenderer(
        _spec(trajectory=path, max_angular_error_deg=None, control_interval_samples=1024,
              min_control_interval_samples=1024)
    )
    engine.prepare(RenderContext(RATE, 512))
    mono = np.zeros(RATE)
    position = 0
    for block in iter_blocks(len(mono), 512, 0):
        engine.process(mono[position : position + block.frames], block)
        position += block.frames
    # One second at 44100 with a 1024-sample grid is about 43 opportunities.
    assert 40 <= engine.diagnostics()["selections"] <= 45

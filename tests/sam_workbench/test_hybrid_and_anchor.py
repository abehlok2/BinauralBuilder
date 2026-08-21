"""The hybrid renderer's three stages, and the broadband spatial anchor.

The rules this file protects:

* physical, creative, and output stay separate, so an A/B compares one thing;
* a hybrid render with nothing creative set is exactly the physical render;
* the anchor is never enabled silently, and its level is controlled;
* the anchor follows the source's path, beside it, or opposite it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.hrtf.modification import CueTransform
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
from src.audio.sam_workbench.render.anchor import (
    ANCHOR_SOURCE_TYPES,
    DEFAULT_ANCHOR_LEVEL_DB,
    AnchorSpec,
    anchor_directions,
    make_anchor_signal,
)
from src.audio.sam_workbench.render.hrtf import render_spatial_hrtf
from src.audio.sam_workbench.render.hybrid import HybridSpec, render_hybrid

FIXTURE = "tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa"


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(FIXTURE)


@pytest.fixture(scope="module")
def mono():
    return np.random.default_rng(0).normal(0.0, 0.1, 8192)


def direction(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def azimuth_of(vector) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def cues(stereo) -> tuple[int, float]:
    correlation = np.correlate(stereo[0], stereo[1], mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (stereo.shape[1] - 1)
    power = np.sqrt(np.mean(np.square(stereo), axis=1))
    return lag, 20.0 * np.log10(power[0] / max(power[1], 1e-12))


# --- stage separation -------------------------------------------------------


def test_a_hybrid_render_with_nothing_creative_is_the_physical_render(dataset, mono):
    """Otherwise "physical vs hybrid" would compare two unrelated things."""

    rate = int(dataset.sample_rate_hz)
    plain = render_spatial_hrtf(mono, direction(30.0), dataset, rate)
    hybrid = render_hybrid(mono, direction(30.0), dataset, rate, HybridSpec())

    assert hybrid.physical
    assert hybrid.audio == pytest.approx(plain, abs=1e-12)
    assert np.all(hybrid.anchor_stem == 0.0)


def test_the_creative_stage_can_be_removed_for_an_honest_comparison(dataset, mono):
    rate = int(dataset.sample_rate_hz)
    spec = HybridSpec(
        cue=CueTransform(itd_scale=1.5, ild_scale=1.2),
        anchor=AnchorSpec(enabled=True),
    )

    creative = render_hybrid(mono, direction(30.0), dataset, rate, spec)
    physical = render_hybrid(mono, direction(30.0), dataset, rate, spec.as_physical())
    reference = render_spatial_hrtf(mono, direction(30.0), dataset, rate)

    assert not creative.physical
    assert physical.physical
    assert physical.audio == pytest.approx(reference, abs=1e-12)


def test_the_creative_stage_actually_moves_the_cues(dataset, mono):
    rate = int(dataset.sample_rate_hz)
    physical = render_hybrid(mono, direction(30.0), dataset, rate, HybridSpec())
    widened = render_hybrid(
        mono,
        direction(30.0),
        dataset,
        rate,
        HybridSpec(cue=CueTransform(itd_scale=1.5, ild_scale=1.3)),
    )

    physical_lag, physical_ild = cues(physical.audio)
    widened_lag, widened_ild = cues(widened.audio)

    assert abs(widened_lag) > abs(physical_lag)
    assert abs(widened_ild) > abs(physical_ild)


def test_stems_come_back_separately_so_a_comparison_can_be_level_matched(dataset, mono):
    rate = int(dataset.sample_rate_hz)
    result = render_hybrid(
        mono,
        direction(30.0),
        dataset,
        rate,
        HybridSpec(anchor=AnchorSpec(enabled=True, level_db=-20.0)),
    )

    assert result.source_stem.shape == result.audio.shape
    assert result.anchor_stem.shape == result.audio.shape
    assert result.audio == pytest.approx(result.source_stem + result.anchor_stem)

    levels = result.levels()
    assert levels["source"] > 0.0
    assert levels["anchor"] > 0.0
    assert levels["mix"] > 0.0


def test_output_gain_is_the_last_stage_and_scales_the_mix(dataset, mono):
    rate = int(dataset.sample_rate_hz)
    unity = render_hybrid(mono, direction(30.0), dataset, rate, HybridSpec())
    quiet = render_hybrid(
        mono, direction(30.0), dataset, rate, HybridSpec(output_gain_db=-6.0)
    )

    assert quiet.audio == pytest.approx(unity.audio * 10 ** (-6.0 / 20.0), rel=1e-9)
    # The stems are pre-gain, so a level comparison is unaffected by it.
    assert quiet.source_stem == pytest.approx(unity.source_stem)


# --- the anchor -------------------------------------------------------------


def test_the_anchor_is_off_by_default_and_silent_when_off():
    spec = AnchorSpec()

    assert spec.enabled is False
    assert spec.level_db == DEFAULT_ANCHOR_LEVEL_DB == -30.0
    assert spec.gain == 0.0
    assert np.all(make_anchor_signal(spec, 1024, 44_100) == 0.0)


@pytest.mark.parametrize("source_type", [s for s in ANCHOR_SOURCE_TYPES if s != "imported"])
def test_every_generated_anchor_is_finite_and_band_limited(source_type):
    rate = 44_100
    # Measured without the fade: any finite fade is a time-domain window, and
    # windowing necessarily spreads a little energy past the band edges. The
    # fade has its own test below.
    spec = AnchorSpec(
        enabled=True, source_type=source_type, band_hz=(500.0, 8000.0), fade_ms=0.0
    )
    signal = make_anchor_signal(spec, 8192, rate)

    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0

    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / rate)
    inside = spectrum[(frequencies >= 500.0) & (frequencies <= 8000.0)].sum()
    outside = spectrum[(frequencies < 400.0) | (frequencies > 9000.0)].sum()
    assert inside > 0.0
    assert outside < inside * 1e-9


def test_an_imported_anchor_needs_material():
    spec = AnchorSpec(enabled=True, source_type="imported")
    with pytest.raises(ValueError):
        make_anchor_signal(spec, 1024, 44_100)

    material = np.sin(np.linspace(0.0, 300.0, 2048))
    signal = make_anchor_signal(spec, 4096, 44_100, material=material)
    assert signal.size == 4096
    assert np.all(np.isfinite(signal))


def test_the_anchor_level_is_controlled(dataset, mono):
    """Exit criterion: anchor on/off must be a level-controlled comparison."""

    rate = int(dataset.sample_rate_hz)
    ratios = []
    for level_db in (-40.0, -30.0, -20.0):
        result = render_hybrid(
            mono,
            direction(30.0),
            dataset,
            rate,
            HybridSpec(anchor=AnchorSpec(enabled=True, level_db=level_db)),
        )
        levels = result.levels()
        ratios.append(20.0 * np.log10(levels["anchor"] / max(levels["source"], 1e-12)))

    # Ten decibels asked for is ten decibels delivered.
    assert ratios[1] - ratios[0] == pytest.approx(10.0, abs=0.1)
    assert ratios[2] - ratios[1] == pytest.approx(10.0, abs=0.1)
    # And the anchor stays well below the source it is supporting.
    assert ratios[1] < -10.0


def test_turning_the_anchor_on_leaves_the_source_stem_alone(dataset, mono):
    """So an on/off comparison changes exactly one thing."""

    rate = int(dataset.sample_rate_hz)
    off = render_hybrid(mono, direction(30.0), dataset, rate, HybridSpec())
    on = render_hybrid(
        mono,
        direction(30.0),
        dataset,
        rate,
        HybridSpec(anchor=AnchorSpec(enabled=True)),
    )

    assert on.source_stem == pytest.approx(off.source_stem)
    assert np.any(on.anchor_stem != 0.0)


def test_the_anchor_carries_the_high_frequency_energy_a_low_carrier_lacks(dataset):
    """The whole reason the anchor exists."""

    rate = int(dataset.sample_rate_hz)
    frames = 8192
    times = np.arange(frames) / rate
    carrier = 0.3 * np.sin(2.0 * np.pi * 180.0 * times)  # a low SAM-like carrier

    result = render_hybrid(
        carrier,
        direction(30.0),
        dataset,
        rate,
        HybridSpec(anchor=AnchorSpec(enabled=True, level_db=-20.0, band_hz=(2000.0, 16000.0))),
    )

    frequencies = np.fft.rfftfreq(frames, 1.0 / rate)
    high = frequencies > 4000.0
    # A 180 Hz tone does not fit a whole number of periods into this window, so
    # a rectangular FFT would smear it across the whole spectrum and credit the
    # carrier with high-frequency energy it does not have.
    window = np.hanning(frames)

    def high_energy(stereo):
        return float(np.sum(np.abs(np.fft.rfft(stereo[0] * window, n=frames))[high]))

    assert high_energy(result.anchor_stem) > 20.0 * high_energy(result.source_stem)


@pytest.mark.parametrize(
    ("mode", "offset", "expected"),
    [("same", 0.0, 30.0), ("offset", 45.0, 75.0), ("mirrored", 0.0, -30.0)],
)
def test_the_anchor_follows_the_requested_path(mode, offset, expected):
    spec = AnchorSpec(enabled=True, path_mode=mode, path_offset_deg=offset)
    placed = anchor_directions(direction(30.0), spec)

    assert azimuth_of(placed) == pytest.approx(expected, abs=1e-6)


def test_a_moving_path_is_offset_frame_by_frame():
    frames = 256
    azimuths = np.linspace(0.0, 90.0, frames)
    path = np.stack(
        [np.cos(np.radians(azimuths)), np.sin(np.radians(azimuths)), np.zeros(frames)], axis=1
    )
    placed = anchor_directions(path, AnchorSpec(enabled=True, path_mode="mirrored"))

    assert placed.shape == path.shape
    assert azimuth_of(placed[0]) == pytest.approx(0.0, abs=1e-6)
    assert azimuth_of(placed[-1]) == pytest.approx(-90.0, abs=1e-6)


def test_an_unknown_anchor_setting_is_rejected():
    with pytest.raises(ValueError):
        AnchorSpec(source_type="trombone")
    with pytest.raises(ValueError):
        AnchorSpec(path_mode="sideways")
    with pytest.raises(ValueError):
        AnchorSpec(band_hz=(9000.0, 500.0))
    with pytest.raises(ValueError):
        AnchorSpec(coherence=2.0)


def test_the_anchor_fades_so_it_never_clicks():
    spec = AnchorSpec(enabled=True, fade_ms=10.0, level_db=0.0)
    signal = make_anchor_signal(spec, 8192, 44_100)

    assert abs(signal[0]) < 1e-9
    assert abs(signal[-1]) < 1e-9
    assert np.max(np.abs(signal)) > 0.1


# --- description ------------------------------------------------------------


def test_the_spec_describes_itself_for_the_project_file(dataset):
    spec = HybridSpec(
        cue=CueTransform(itd_scale=1.2),
        anchor=AnchorSpec(enabled=True, level_db=-25.0, source_type="sparse_grains"),
        output_gain_db=-6.0,
    )
    described = spec.describe()

    assert described["physical"] is False
    assert described["cue"]["itd_scale"] == pytest.approx(1.2)
    assert described["anchor"]["enabled"] is True
    assert described["anchor"]["sourceType"] == "sparse_grains"
    assert described["outputGainDb"] == pytest.approx(-6.0)
    assert HybridSpec().describe()["physical"] is True

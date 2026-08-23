"""The three-dimensional path as the renderers actually receive it.

The gap this file exists to keep closed: a path could be authored in three
dimensions and then flattened on its way to the renderer, so that preview and
export reduced it to a moving azimuth at a fixed elevation and distance. The
tests here assert on the positions and audio that come out the far end, not on
what was stored going in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.compat import _hrtf_trajectory, hrtf_coverage_report
from src.audio.sam_workbench.hrtf.coverage import assess_path_coverage
from src.audio.sam_workbench.render.creative import (
    CREATIVE_MAPPING_NOTICE,
    CreativeMappingSpec,
    TrajectoryControl,
)
from src.audio.sam_workbench.render.hrtf import HRTFRendererSpec, render_hrtf
from src.audio.sam_workbench.render.hybrid import SIGNAL_CHAIN
from src.audio.sam_workbench.trajectory import (
    DomeTraversal,
    cartesian_array_to_spherical,
    path_model_from_dict,
    spherical_to_cartesian,
)

SOFA = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"
SAMPLE_RATE = 44100


def _dome_voice(**trajectory):
    payload = {
        "geometry": {"type": "dome_traversal", "parameters": {"distanceM": 1.5, "turns": 3, **trajectory}},
        "traversal": {"durationS": 4.0},
    }
    return {
        "rendererMode": "hrtf",
        "carrierFreq": 440.0,
        "amp": 0.5,
        "canonicalTrajectory": payload,
    }


# --- the trajectory reaches the renderer ------------------------------------


def test_a_three_dimensional_path_is_not_flattened_on_its_way_to_the_renderer():
    """Elevation and distance must both survive, not just azimuth."""

    voice = _dome_voice(endElevationDeg=80.0)
    voice["canonicalTrajectory"]["geometry"]["parameters"]["distanceM"] = 1.5
    positions = _hrtf_trajectory(voice, {})(np.linspace(0.0, 4.0, 256))
    spherical = cartesian_array_to_spherical(positions)
    assert np.ptp(spherical[:, 1]) > 45.0, "elevation was held constant"
    assert np.ptp(spherical[:, 0]) > 90.0, "azimuth barely moved"


def test_a_varying_distance_reaches_the_renderer():
    voice = {
        "rendererMode": "hrtf",
        "canonicalTrajectory": {
            "geometry": {
                "type": "spherical_orbit",
                "parameters": {"startDistanceM": 0.5, "endDistanceM": 3.0},
            },
            "traversal": {"durationS": 4.0},
        },
    }
    distances = np.linalg.norm(
        _hrtf_trajectory(voice, {})(np.linspace(0.0, 4.0, 256)), axis=1
    )
    assert distances.min() == pytest.approx(0.5, abs=0.01)
    assert distances.max() == pytest.approx(3.0, abs=0.01)


def test_a_voice_with_no_trajectory_still_gets_the_legacy_sinusoid():
    """The legacy path is preserved exactly; only its role is now the fallback."""

    params = {"modFreq": 4.0, "arcWidthDeg": 90.0, "directionOffsetDeg": 0.0}
    options = {"distanceM": 1.2, "elevationDeg": 15.0}
    times = np.linspace(0.0, 1.0, 64)

    azimuth = 0.5 * 90.0 * np.sin(2.0 * np.pi * 4.0 * times)
    expected = spherical_to_cartesian(azimuth, 15.0, 1.2)
    assert _hrtf_trajectory(params, options)(times) == pytest.approx(np.asarray(expected))


def test_the_renderer_accepts_a_single_sample_query():
    """The streaming renderer asks for one position at a time."""

    result = _hrtf_trajectory(_dome_voice(), {})(np.array([0.5]))
    assert np.asarray(result).shape == (1, 3)


# --- distance is audible ----------------------------------------------------


def _render_at(distance_m, *, law="inverse", delay=False, signal=None):
    frames = 4096
    source = np.sin(2.0 * np.pi * 440.0 * np.arange(frames) / SAMPLE_RATE) if signal is None else signal
    spec = HRTFRendererSpec(
        sofa_path=SOFA,
        trajectory=lambda t: spherical_to_cartesian(np.zeros_like(np.asarray(t)), 0.0, distance_m),
        distance_law=law,
        propagation_delay=delay,
        reference_distance_m=1.0,
        interpolation="nearest",
    )
    return render_hrtf(source, spec, SAMPLE_RATE, block_size=1024)


def _rms(audio):
    return float(np.sqrt(np.mean(np.square(audio))))


@pytest.mark.parametrize("law, expected", [("inverse", 0.5), ("inverse_square", 0.25)])
def test_distance_attenuates_under_the_configured_law(law, expected):
    near = _rms(_render_at(1.0, law=law))
    far = _rms(_render_at(2.0, law=law))
    assert far / near == pytest.approx(expected, rel=0.02)


def test_distance_is_ignored_when_no_law_is_configured():
    """The neutral default keeps a direction-only render exactly as it was."""

    assert _render_at(1.0, law="none") == pytest.approx(_render_at(9.0, law="none"))


def test_propagation_delay_moves_the_onset_by_the_travel_time():
    impulse = np.zeros(4096)
    impulse[100] = 1.0
    near = _render_at(1.0, law="none", delay=True, signal=impulse)
    far = _render_at(4.0, law="none", delay=True, signal=impulse)
    travel = (4.0 - 1.0) / 343.0 * SAMPLE_RATE
    shift = int(np.argmax(np.abs(far[0]))) - int(np.argmax(np.abs(near[0])))
    assert shift == pytest.approx(travel, abs=2.0)


def test_a_moving_source_does_not_click_when_distance_changes():
    """Gain and delay slew across the control interval rather than stepping."""

    frames = 8192
    source = np.sin(2.0 * np.pi * 220.0 * np.arange(frames) / SAMPLE_RATE)
    spec = HRTFRendererSpec(
        sofa_path=SOFA,
        trajectory=lambda t: spherical_to_cartesian(
            np.zeros_like(np.asarray(t)), 0.0, 0.5 + 2.0 * np.asarray(t)
        ),
        distance_law="inverse",
        propagation_delay=True,
        interpolation="nearest",
        control_interval_samples=128,
    )
    rendered = render_hrtf(source, spec, SAMPLE_RATE, block_size=1024)
    # A step at a control boundary shows up as a sample-to-sample jump far
    # larger than the waveform's own slope.
    jumps = np.abs(np.diff(rendered[0]))
    assert np.max(jumps) < 12.0 * np.median(jumps[jumps > 0.0])


# --- coverage warnings ------------------------------------------------------


def _horizontal_only_dataset(step_deg=5.0):
    azimuth = np.arange(0.0, 360.0, step_deg)
    return spherical_to_cartesian(azimuth, np.zeros_like(azimuth), 1.0)


def _full_sphere_dataset(step_deg=10.0):
    return np.array(
        [
            spherical_to_cartesian(azimuth, elevation, 1.0)
            for elevation in np.arange(-40.0, 91.0, step_deg)
            for azimuth in np.arange(0.0, 360.0, step_deg)
        ]
    )


def _messages(report):
    return report.summary()


def test_a_dome_over_a_horizontal_only_dataset_warns_about_every_relevant_thing():
    report = assess_path_coverage(
        _horizontal_only_dataset(),
        DomeTraversal().evaluate(np.linspace(0.0, 1.0, 200)),
        interpolation="logmag_delay",
    )
    text = _messages(report)
    assert not report.ok
    assert "above measured coverage" in text
    assert "only 0 measurement" in text
    assert "sparse regions" in text


def test_a_matched_path_and_dataset_produce_no_warnings():
    from src.audio.sam_workbench.trajectory import HorizontalOrbit

    report = assess_path_coverage(
        _horizontal_only_dataset(),
        HorizontalOrbit().evaluate(np.linspace(0.0, 1.0, 200)),
        interpolation="logmag_delay",
    )
    assert report.ok, report.summary()


def test_nearest_neighbour_fallback_is_reported():
    from src.audio.sam_workbench.trajectory import HorizontalOrbit

    report = assess_path_coverage(
        _horizontal_only_dataset(),
        HorizontalOrbit().evaluate(np.linspace(0.0, 1.0, 200)),
        interpolation="nearest",
    )
    assert "nearest-neighbour" in report.summary()


def test_a_path_that_outruns_the_control_interval_is_reported():
    report = assess_path_coverage(
        _full_sphere_dataset(),
        DomeTraversal(turns=40).evaluate(np.linspace(0.0, 1.0, 200)),
        interpolation="logmag_delay",
        control_interval_samples=128,
        crossfade_ms=10.0,
    )
    text = report.summary()
    assert "between filter updates" in text
    # The blend span reported is the transition the renderer actually performs,
    # which is capped at the control interval, rather than the raw crossfade
    # setting it would otherwise have used.
    assert "will smear rather than track the motion" in text
    assert f"{128 / 44100 * 1000.0:.1f} ms transition" in text


def test_a_below_head_path_reports_the_missing_lower_hemisphere():
    report = assess_path_coverage(
        _full_sphere_dataset(),
        DomeTraversal(start_elevation_deg=-80.0, end_elevation_deg=-60.0).evaluate(
            np.linspace(0.0, 1.0, 200)
        ),
        interpolation="logmag_delay",
    )
    assert "below measured coverage" in report.summary()


def test_the_voice_level_report_only_answers_for_hrtf_voices():
    assert hrtf_coverage_report({"rendererMode": "abstract_pm"}, _full_sphere_dataset()) is None
    report = hrtf_coverage_report(_dome_voice(endElevationDeg=85.0), _horizontal_only_dataset())
    assert report is not None and not report.ok


def test_a_coverage_report_serializes_for_an_export_record():
    described = assess_path_coverage(
        _horizontal_only_dataset(),
        DomeTraversal().evaluate(np.linspace(0.0, 1.0, 64)),
    ).describe()
    assert described["warnings"]
    assert set(described) >= {"measuredElevationDeg", "requestedElevationDeg", "sparseFraction"}


# --- creative mappings ------------------------------------------------------


def test_creative_mappings_are_labelled_and_never_claim_to_be_physical():
    spec = CreativeMappingSpec.documented_default()
    described = spec.describe()
    assert described["physical"] is False
    assert "not spatialization" in described["notice"]
    # The label travels with the control into the saved document.
    control = spec.controls(path_model_from_dict(_dome_voice()["canonicalTrajectory"]))
    assert CREATIVE_MAPPING_NOTICE in control["carrier_hz"].to_dict()["note"]


def test_a_trajectory_control_renders_identically_in_any_block_order():
    """Preview seeks and sequential exports have to agree sample for sample."""

    model = path_model_from_dict(_dome_voice()["canonicalTrajectory"])
    control = TrajectoryControl(trajectory=model, quantity="elevation", output_low=0.0, output_high=1.0)
    whole = control.render(0, 512, SAMPLE_RATE)
    pieces = np.concatenate(
        [control.render(start, 128, SAMPLE_RATE) for start in (0, 128, 256, 384)]
    )
    assert whole == pytest.approx(pieces)


def test_a_creative_mapping_spec_round_trips():
    spec = CreativeMappingSpec.documented_default()
    assert CreativeMappingSpec.from_mapping(spec.describe()).describe() == spec.describe()


def test_no_creative_mapping_is_applied_by_default():
    assert CreativeMappingSpec().is_neutral


def test_the_hybrid_stage_order_is_stated_explicitly():
    assert SIGNAL_CHAIN.index("SAM") < SIGNAL_CHAIN.index("HRTF interpolation")
    assert SIGNAL_CHAIN.index("HRTF interpolation") < SIGNAL_CHAIN.index("Cue modification")

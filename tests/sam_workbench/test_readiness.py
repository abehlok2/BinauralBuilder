"""Readiness: what validity cannot tell you.

A project can be entirely legal and still not produce what its author expects.
The dataset moved. The path goes where the dataset never measured. A control is
set that this renderer ignores. Every check here has to name the thing and say
what to do about it, because a warning that says only "something is wrong"
costs more attention than it saves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.sam_workbench.readiness import assess_readiness

SOFA = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")


def _params(**overrides):
    params = {"rendererMode": "abstract_pm"}
    params.update(overrides)
    return params


# --- the shape of the advice ------------------------------------------------


def test_every_issue_names_a_path_and_says_something():
    report = assess_readiness(_params(rendererMode="hrtf"))
    assert report.issues
    for issue in report.issues:
        assert issue.path
        assert len(issue.message) > 20


def test_a_complete_configuration_is_ready():
    report = assess_readiness(_params(rendererMode="hrtf", hrtfAsset=SOFA,
                                      headphoneAsset="/phones.wav"))
    assert report.ready is True
    assert report.errors == ()


# --- assets -----------------------------------------------------------------


def test_a_missing_sofa_asset_blocks_an_hrtf_render():
    report = assess_readiness(_params(rendererMode="hrtf"))
    assert report.ready is False
    assert any(issue.path == "hrtfAsset" for issue in report.errors)


def test_a_sofa_path_that_no_longer_exists_is_an_error():
    report = assess_readiness(
        _params(rendererMode="hrtf", hrtfAsset="/gone/dataset.sofa")
    )
    assert any("no longer at that path" in issue.message for issue in report.errors)


def test_a_changed_dataset_is_reported_rather_than_rendered_quietly(tmp_path):
    """The render would not match what the author approved."""

    report = assess_readiness(
        _params(
            rendererMode="hrtf",
            hrtfAsset=SOFA,
            hrtfAssetHash="0" * 64,
        )
    )
    assert report.ready is False
    assert any("not the one this project was authored against" in issue.message
               for issue in report.errors)


def test_a_matching_hash_raises_nothing():
    from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

    report = assess_readiness(
        _params(
            rendererMode="hrtf",
            hrtfAsset=SOFA,
            hrtfAssetHash=load_sofa(SOFA).content_hash,
        )
    )
    assert report.ready is True


def test_a_missing_headphone_profile_is_advice_not_an_error():
    report = assess_readiness(_params(rendererMode="hrtf", hrtfAsset=SOFA))
    headphone = [i for i in report.issues if i.path == "headphoneAsset"]
    assert headphone and headphone[0].severity == "info"
    assert report.ready is True


# --- coverage ---------------------------------------------------------------


def test_a_path_the_dataset_cannot_support_is_reported():
    report = assess_readiness(
        _params(
            rendererMode="hrtf",
            hrtfAsset=SOFA,
            canonicalTrajectory={
                "geometry": {"type": "dome_traversal", "parameters": {"turns": 40}},
                "traversal": {"durationS": 1.0},
            },
        )
    )
    assert any("filter updates" in issue.message for issue in report.issues)


def test_coverage_is_not_consulted_for_a_renderer_that_does_not_convolve():
    report = assess_readiness(
        _params(
            rendererMode="abstract_pm",
            hrtfAsset=SOFA,
            canonicalTrajectory={
                "geometry": {"type": "dome_traversal", "parameters": {"turns": 40}},
                "traversal": {"durationS": 1.0},
            },
        )
    )
    assert not any("filter updates" in issue.message for issue in report.issues)


# --- inactive controls ------------------------------------------------------


def test_a_setting_this_renderer_ignores_is_named():
    """From the value alone, ignored and disabled look the same."""

    report = assess_readiness(
        _params(hrtfAsset=SOFA, hrtfOptions={"interpolation": "delay_magnitude"})
    )
    paths = {issue.path for issue in report.warnings}
    assert "hrtfAsset" in paths
    assert "hrtfOptions.interpolation" in paths
    for issue in report.warnings:
        assert "does not read this" in issue.message


def test_cue_modification_under_plain_hrtf_is_named():
    report = assess_readiness(
        _params(
            rendererMode="hrtf",
            hrtfAsset=SOFA,
            hrtfOptions={"cue": {"neutral": False, "itdScale": 1.4}},
        )
    )
    assert any(issue.path == "hrtfOptions.cue" for issue in report.warnings)


def test_the_same_setting_under_hybrid_is_not_named():
    report = assess_readiness(
        _params(
            rendererMode="hybrid",
            hrtfAsset=SOFA,
            hrtfOptions={"cue": {"neutral": False, "itdScale": 1.4}},
        )
    )
    assert not any(issue.path == "hrtfOptions.cue" for issue in report.warnings)


# --- output -----------------------------------------------------------------


def test_a_render_that_will_clip_is_an_error():
    report = assess_readiness(_params(), peak=1.02)
    assert report.ready is False
    assert any("clip" in issue.message for issue in report.errors)


def test_a_render_close_to_full_scale_is_a_warning():
    report = assess_readiness(_params(), peak=0.995)
    assert report.ready is True
    assert any(issue.path == "output.peak" for issue in report.warnings)


def test_a_comfortable_peak_says_nothing():
    report = assess_readiness(_params(), peak=0.4)
    assert not any(issue.path == "output.peak" for issue in report.issues)


# --- the scene --------------------------------------------------------------


def test_a_route_naming_a_missing_source_is_reported():
    scene = {
        "sources": [{"id": "sam.1", "name": "Lead"}],
        "routing": {
            "buses": [{"id": "master", "name": "Master"}],
            "sources": [{"sourceId": "ghost", "busId": "master"}],
        },
    }
    report = assess_readiness(_params(), scene=scene)
    assert any("ghost" in f"{i.path} {i.message}" for i in report.issues)


def test_a_route_to_a_bus_that_does_not_exist_is_reported():
    scene = {
        "sources": [{"id": "sam.1"}],
        "routing": {
            "buses": [{"id": "master"}],
            "sources": [{"sourceId": "sam.1", "busId": "reverb"}],
        },
    }
    report = assess_readiness(_params(), scene=scene)
    assert any("reverb" in i.message for i in report.issues)


def test_the_report_is_describable_and_summarizable():
    report = assess_readiness(_params(rendererMode="hrtf"))
    described = report.describe()
    assert described and set(described[0]) == {"path", "message", "severity"}
    assert report.summary()
    assert assess_readiness(
        _params(rendererMode="hrtf", hrtfAsset=SOFA, headphoneAsset="/p.wav")
    ).summary() == "Ready to render."


# --- the dialog shows them --------------------------------------------------


def test_the_dialog_reports_readiness_alongside_validation(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "params": {"rendererMode": "hrtf", "hrtfAsset": "/gone.sofa"},
        }
    )
    qtbot.addWidget(dialog)

    reported = " ".join(f"{i.path} {i.message}" for i in dialog.issues())
    assert "no longer at that path" in reported

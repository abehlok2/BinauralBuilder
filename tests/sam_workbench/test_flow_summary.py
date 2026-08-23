"""What happens when I press render, answered from the production parameters.

The configuration is spread across tabs. Each part is legible on its own and
the whole is not, so it is possible to set an interpolation mode a non-HRTF
renderer never reads, or leave a path enabled that nothing consumes, and see
nothing wrong. These tests hold the summary to describing the render that will
actually happen rather than the controls that happen to hold values.
"""

from __future__ import annotations

import pytest

from src.audio.sam_workbench.flow import SIGNAL_CHAIN_LABELS, summarize_flow
from src.audio.sam_workbench.validation import ValidationIssue


def _params(**overrides):
    params = {"rendererMode": "abstract_pm"}
    params.update(overrides)
    return params


def test_the_chain_names_every_stage_in_the_specified_order():
    summary = summarize_flow(_params())
    assert tuple(stage.name for stage in summary.stages) == SIGNAL_CHAIN_LABELS


def test_an_inactive_stage_is_shown_rather_than_omitted():
    """A chain that changes length as options toggle is harder to read."""

    text = summarize_flow(_params()).chain_text()
    for label in SIGNAL_CHAIN_LABELS:
        assert label in text
    assert "(Path)" in text


# --- settings the selected renderer will never read -------------------------


def test_a_sofa_asset_on_a_non_hrtf_renderer_is_reported_as_unused():
    summary = summarize_flow(
        _params(hrtfAsset="/x.sofa", hrtfOptions={"interpolation": "nearest"})
    )
    assert "hrtfAsset" in summary.inactive
    assert "hrtfOptions.interpolation" in summary.inactive


def test_the_same_asset_on_an_hrtf_renderer_is_not_reported():
    summary = summarize_flow(
        _params(
            rendererMode="hrtf",
            hrtfAsset="/x.sofa",
            hrtfOptions={"interpolation": "nearest"},
        )
    )
    assert summary.inactive == ()
    assert summary.asset == "/x.sofa"
    assert summary.interpolation == "nearest"


def test_cue_modification_is_reported_unused_on_a_renderer_that_lacks_it():
    """Only hybrid declares support; hrtf holds the value and ignores it."""

    summary = summarize_flow(
        _params(
            rendererMode="hrtf",
            hrtfAsset="/x.sofa",
            hrtfOptions={"cue": {"neutral": False, "itdScale": 1.4}},
        )
    )
    assert "hrtfOptions.cue" in summary.inactive

    hybrid = summarize_flow(
        _params(
            rendererMode="hybrid",
            hrtfAsset="/x.sofa",
            hrtfOptions={"cue": {"neutral": False, "itdScale": 1.4}},
        )
    )
    assert "hrtfOptions.cue" not in hybrid.inactive
    assert "Cue transform" in hybrid.active_stages


def test_a_neutral_cue_does_not_light_the_stage():
    summary = summarize_flow(
        _params(rendererMode="hybrid", hrtfAsset="/x.sofa",
                hrtfOptions={"cue": {"neutral": True}})
    )
    assert "Cue transform" not in summary.active_stages


# --- what it reports --------------------------------------------------------


def test_the_path_is_described_by_its_geometry_and_frame():
    summary = summarize_flow(
        _params(
            rendererMode="hrtf",
            hrtfAsset="/x.sofa",
            canonicalTrajectory={
                "geometry": {"type": "torus"},
                "traversal": {"frame": "listener"},
            },
        )
    )
    assert "torus" in summary.path_status
    assert "listener" in summary.path_status
    assert "Path" in summary.active_stages


def test_the_scene_is_summarized_by_its_roster_not_invented():
    scene = {
        "sources": [{"id": "a"}, {"id": "b"}],
        "routing": {"buses": [{"id": "master"}], "sources": [{"sourceId": "a", "solo": True}]},
    }
    summary = summarize_flow(_params(), scene=scene)
    assert "2 source(s)" in summary.scene_status
    assert "1 bus(es)" in summary.scene_status
    assert "1 soloed" in summary.scene_status


def test_warnings_are_the_ones_passed_in_not_a_second_opinion():
    issue = ValidationIssue("hrtfAsset", "missing", "warning")
    summary = summarize_flow(_params(), issues=(issue,))
    assert summary.warnings == (issue,)


def test_an_hrtf_render_carries_a_cost_estimate():
    summary = summarize_flow(
        _params(rendererMode="hrtf", hrtfAsset="/x.sofa",
                hrtfOptions={"interpolation": "delay_magnitude"})
    )
    assert summary.cost
    assert "%" in summary.cost


def test_the_summary_is_json_describable():
    described = summarize_flow(_params()).describe()
    import json

    assert json.loads(json.dumps(described))["renderer"] == "abstract_pm"


# --- the dialog shows it ----------------------------------------------------


def test_the_dialog_keeps_the_summary_visible(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)

    assert dialog.flow_summary is not None
    text = dialog.flow_label.text()
    assert "Renderer:" in text
    assert "→" in text


def test_the_dialog_summary_follows_the_renderer(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)

    dialog.renderer_combo.setCurrentIndex(dialog.renderer_combo.findData("hybrid"))
    dialog._revalidate()

    assert dialog.flow_summary.renderer_id == "hybrid"
    assert "Hybrid" in dialog.flow_label.text()

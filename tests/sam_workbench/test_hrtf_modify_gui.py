"""The Modify tab: cue controls, overlays, blinded A/B/X, and derived export.

The rules this file protects:

* every control starts neutral, and Reset returns to exactly that;
* the plots always show the measurement alongside the modification;
* the comparison is blinded and level-matched;
* a derived file is only reported as saved once it verifies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from src.audio.sam_workbench.hrtf import storage  # noqa: E402
from src.audio.sam_workbench.hrtf.derived import read_derived_provenance  # noqa: E402
from src.audio.sam_workbench.hrtf.modification import (  # noqa: E402
    BASIC_RANGES,
    CUE_PARAMETERS,
    EXPERT_RANGES,
    NEUTRAL,
)
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa  # noqa: E402
from src.ui.hrtf_cue_plot import Curve, HrtfCurvePlot, PlotThrottle  # noqa: E402
from src.ui.sam_hrtf_modify_panel import AbxTrial, SamHrtfModifyPanel  # noqa: E402

FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_sonicom_hrir.sofa")


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(FIXTURE)


@pytest.fixture
def panel(qtbot, tmp_path, dataset):
    storage.set_data_root(tmp_path)
    widget = SamHrtfModifyPanel()
    qtbot.addWidget(widget)
    widget.set_dataset(dataset)
    yield widget
    storage.set_data_root(None)


# --- neutrality -------------------------------------------------------------


def test_every_control_starts_neutral(panel):
    assert panel.is_neutral
    transform = panel.cue_transform()
    for name in CUE_PARAMETERS:
        assert getattr(transform, name) == pytest.approx(NEUTRAL[name])
    assert panel.params()["hrtfOptions"]["cue"]["neutral"] is True


def test_the_anchor_starts_off(panel):
    assert panel.anchor_spec().enabled is False
    assert panel.params()["hrtfOptions"]["anchor"]["enabled"] is False


def test_reset_returns_to_the_measured_asset(panel):
    panel.expert_check.setChecked(True)
    panel._sliders["itd_scale"].setValue(220)
    panel._sliders["coherence"].setValue(30)
    assert not panel.is_neutral

    panel.reset_cues()

    assert panel.is_neutral
    assert panel.warning_label.text() == ""


def test_a_neutral_panel_leaves_the_responses_untouched(panel, dataset):
    assert panel.modified_hrirs() == pytest.approx(np.asarray(dataset.ir), abs=0.0)


# --- ranges -----------------------------------------------------------------


def test_the_basic_ranges_are_narrower_than_the_expert_ones(panel):
    for name in CUE_PARAMETERS:
        basic_low, basic_high = BASIC_RANGES[name]
        expert_low, expert_high = EXPERT_RANGES[name]
        assert expert_low <= basic_low
        assert expert_high >= basic_high


def test_the_basic_range_clamps_and_expert_releases(panel):
    panel._sliders["pinna_scale"].setValue(0)
    assert panel.cue_transform().pinna_scale == pytest.approx(BASIC_RANGES["pinna_scale"][0])

    panel.expert_check.setChecked(True)
    panel._sliders["pinna_scale"].setValue(0)
    assert panel.cue_transform().pinna_scale == pytest.approx(0.0)


def test_switching_ranges_keeps_the_value_already_set(panel):
    panel.expert_check.setChecked(True)
    panel._sliders["itd_scale"].setValue(140)
    panel.expert_check.setChecked(False)

    assert panel.cue_transform().itd_scale == pytest.approx(1.4)


def test_a_contradictory_setting_is_explained_not_blocked(panel):
    panel.expert_check.setChecked(True)
    panel._sliders["pinna_scale"].setValue(0)

    assert "elevation" in panel.warning_label.text()
    # Still allowed: the warning is guidance, not a veto.
    assert panel.cue_transform().pinna_scale == pytest.approx(0.0)


# --- plots ------------------------------------------------------------------


@pytest.mark.parametrize(
    "view", ["impulse", "magnitude", "group_delay", "phase", "itd", "ild"]
)
def test_every_view_plots_the_measurement_against_the_modification(panel, view):
    index = panel.plot_selector.findData(view)
    assert index >= 0
    panel.plot_selector.setCurrentIndex(index)
    panel.refresh_plots()

    assert panel.plot.curve_count == 2


def test_a_panel_with_no_asset_says_so_rather_than_plotting_nothing(qtbot):
    widget = SamHrtfModifyPanel()
    qtbot.addWidget(widget)
    widget.set_dataset(None)

    assert widget.plot.curve_count == 0
    assert not widget.derive_button.isEnabled()


def test_the_modified_curve_moves_when_a_cue_moves(panel, dataset):
    from src.audio.sam_workbench.analysis.hrtf_curves import sweep_across_azimuth

    rate = dataset.sample_rate_hz
    _, original_itd, _ = sweep_across_azimuth(dataset.ir, dataset.positions_m, rate)

    panel._sliders["itd_scale"].setValue(150)
    _, modified_itd, _ = sweep_across_azimuth(
        panel.modified_hrirs(), dataset.positions_m, rate
    )

    assert np.mean(np.abs(modified_itd)) > np.mean(np.abs(original_itd)) * 1.2


# --- the throttle -----------------------------------------------------------


def test_rapid_updates_collapse_into_one_redraw(qtbot):
    throttle = PlotThrottle(interval_ms=30)
    fired = []
    throttle.triggered.connect(lambda: fired.append(1))

    for _ in range(20):
        throttle.request()
    assert throttle.pending
    assert fired == []  # nothing yet: a drag must not redraw twenty times

    qtbot.wait(80)
    assert sum(fired) == 1
    assert not throttle.pending


def test_the_throttle_can_be_flushed(qtbot):
    throttle = PlotThrottle(interval_ms=5000)
    fired = []
    throttle.triggered.connect(lambda: fired.append(1))

    throttle.request()
    throttle.flush()
    assert sum(fired) == 1

    # Nothing pending means nothing to emit.
    throttle.flush()
    assert sum(fired) == 1


# --- blinded comparison -----------------------------------------------------


def test_x_is_one_of_the_two_candidates_and_is_not_announced():
    seen = {AbxTrial(seed=index).x_is_b for index in range(40)}
    # Over many trials both assignments occur, so X is genuinely random.
    assert seen == {True, False}

    trial = AbxTrial(seed=1)
    assert trial.answered is False
    assert trial.correct is None


def test_answering_records_a_running_score():
    trial = AbxTrial(seed=3)
    truth = "B" if trial.x_is_b else "A"

    assert trial.answer(truth) is True
    assert trial.answered
    assert trial.score == "1 of 1 correct"

    trial.reset()
    wrong = "A" if trial.x_is_b else "B"
    assert trial.answer(wrong) is False
    assert trial.score == "1 of 2 correct"


def test_an_invalid_answer_is_rejected():
    with pytest.raises(ValueError):
        AbxTrial(seed=0).answer("maybe")


def test_a_trial_needs_something_to_compare(panel):
    panel.prepare_abx()

    assert "Nothing to compare" in panel.status_label.text()
    assert panel._abx_renders == {}


def test_a_prepared_trial_is_level_matched(panel):
    panel._sliders["ild_scale"].setValue(150)
    panel.prepare_abx()

    assert set(panel._abx_renders) == {"A", "B", "X"}
    left, _ = panel._abx_renders["A"]
    right, _ = panel._abx_renders["B"]

    def rms(block):
        return float(np.sqrt(np.mean(np.square(block))))

    # Otherwise the comparison measures whichever came out louder.
    assert rms(left) == pytest.approx(rms(right), rel=1e-9)


def test_x_really_is_one_of_the_two_renders(panel):
    panel._sliders["itd_scale"].setValue(180)
    panel.prepare_abx()

    candidate_a = panel._abx_renders["A"][0]
    candidate_b = panel._abx_renders["B"][0]
    x = panel._abx_renders["X"][0]

    if panel.abx.x_is_b:
        assert x == pytest.approx(candidate_b)
    else:
        assert x == pytest.approx(candidate_a)
    # And the two candidates are genuinely different, or the test is vacuous.
    assert not np.allclose(candidate_a, candidate_b)


def test_changing_a_cue_invalidates_a_prepared_trial(panel):
    panel._sliders["itd_scale"].setValue(180)
    panel.prepare_abx()
    assert panel.play_a_button.isEnabled()

    panel._sliders["ild_scale"].setValue(120)

    assert panel._abx_renders == {}
    assert not panel.play_a_button.isEnabled()


def test_playing_a_candidate_emits_it_for_the_host(panel, qtbot):
    panel._sliders["itd_scale"].setValue(180)
    panel.prepare_abx()

    with qtbot.waitSignal(panel.auditionRendered, timeout=1000) as caught:
        panel.play("X")

    audio, rate = caught.args[0]
    assert audio.shape[0] == 2
    assert rate > 0


# --- derived export ---------------------------------------------------------


def test_a_neutral_panel_refuses_to_write_a_derived_file(panel, monkeypatch):
    called = []
    monkeypatch.setattr(
        "src.ui.sam_hrtf_modify_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: called.append(1) or ("", ""),
    )

    panel.derive_and_save()

    assert "neutral" in panel.status_label.text()
    assert called == []  # never even asked where to put it


def test_a_derived_file_is_written_verified_and_carries_its_provenance(
    panel, tmp_path, monkeypatch
):
    destination = tmp_path / "derived.sofa"
    monkeypatch.setattr(
        "src.ui.sam_hrtf_modify_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(destination), ""),
    )
    panel._sliders["itd_scale"].setValue(130)

    panel.derive_and_save()

    assert destination.exists()
    assert "verified" in panel.status_label.text()

    provenance = read_derived_provenance(destination)
    assert provenance is not None
    assert provenance.derived is True
    assert provenance.transform["itd_scale"] == pytest.approx(1.3)
    assert provenance.source_sha256


def test_cancelling_the_save_dialog_writes_nothing(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.ui.sam_hrtf_modify_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )
    panel._sliders["itd_scale"].setValue(130)

    panel.derive_and_save()

    assert list(tmp_path.glob("*.sofa")) == []


# --- parameter round trip ---------------------------------------------------


def test_params_round_trip_through_the_project(panel):
    panel.expert_check.setChecked(True)
    panel._sliders["itd_scale"].setValue(125)
    panel._sliders["coherence"].setValue(80)
    panel.anchor_check.setChecked(True)
    panel.anchor_level.setValue(-24.0)

    stored = panel.params()
    restored = SamHrtfModifyPanel()
    restored.set_params(stored)

    assert restored.cue_transform().itd_scale == pytest.approx(1.25)
    assert restored.cue_transform().coherence == pytest.approx(0.8)
    assert restored.anchor_spec().enabled is True
    assert restored.anchor_spec().level_db == pytest.approx(-24.0)


def test_a_stored_value_beyond_the_basic_range_widens_the_controls(qtbot):
    """A project that used an expert value must not be silently clamped."""

    widget = SamHrtfModifyPanel()
    qtbot.addWidget(widget)
    widget.set_params({"hrtfOptions": {"cue": {"itd_scale": 2.2}}})

    assert widget.expert_check.isChecked()
    assert widget.cue_transform().itd_scale == pytest.approx(2.2)


def test_unknown_option_keys_survive(panel):
    panel.set_params({"hrtfOptions": {"somethingNewer": 5, "cue": {"itd_scale": 1.1}}})

    assert panel.params()["hrtfOptions"]["somethingNewer"] == 5


# --- the plot widget --------------------------------------------------------


def test_the_plot_widget_rejects_mismatched_series(qtbot):
    plot = HrtfCurvePlot()
    qtbot.addWidget(plot)

    with pytest.raises(ValueError):
        plot.set_curves(np.arange(4), [Curve("bad", np.arange(3))])


def test_the_plot_widget_reports_what_it_holds(qtbot):
    plot = HrtfCurvePlot("title")
    qtbot.addWidget(plot)
    assert plot.title == "title"
    assert plot.curve_count == 0

    plot.set_curves(np.arange(8), [Curve("a", np.arange(8)), Curve("b", np.arange(8))])
    assert plot.curve_count == 2

    plot.clear()
    assert plot.curve_count == 0


def test_an_axis_label_never_loses_its_minus_sign():
    """Regression: a narrow label box clipped the sign off negative values.

    On an ITD plot the sign is which ear leads, so a clipped minus turns a
    left-leading source into a right-leading one at a glance.
    """

    from src.ui.hrtf_cue_plot import _axis_label

    assert _axis_label(-971.94).startswith("-")
    assert _axis_label(-1094.6).startswith("-")
    assert _axis_label(-5.5) == "-5.5"
    assert _axis_label(0.0) == "0"
    # Four significant figures would render 1094.6 as "1095", not "1.095e+03".
    assert "e" not in _axis_label(1094.6)


@pytest.mark.parametrize(
    "labels",
    [["-1094.6", "1085"], ["-0.00123", "0.004"], ["1", "2"], ["-12345", "12345"]],
)
def test_the_axis_margin_grows_to_fit_whatever_it_must_show(qtbot, labels):
    """The margin is measured from the labels, so none of them is clipped."""

    from PyQt5.QtGui import QPainter

    plot = HrtfCurvePlot()
    qtbot.addWidget(plot)
    plot.resize(400, 160)

    painter = QPainter(plot)
    font = painter.font()
    font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
    painter.setFont(font)
    margin = plot._left_margin_for(painter, labels)
    widest = max(painter.fontMetrics().horizontalAdvance(text) for text in labels)
    painter.end()

    assert widest <= margin - 5.0
    # ...but never so wide that the curve has nowhere left to go.
    assert margin <= plot.width() * 0.4 + 1.0

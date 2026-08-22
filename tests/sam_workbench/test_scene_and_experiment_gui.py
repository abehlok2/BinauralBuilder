"""The Scene views and the Phase 7 dialogs.

The rules this file protects:

* a modulation route that would close a loop is refused before it is stored,
  and the loop is named rather than hinted at;
* stages come from steps, so the timeline and the step list cannot disagree;
* the cost readout follows the scene it is describing;
* a listening test never shows which dataset is which until it is scored;
* the measurement tools stay off until the advanced flag is set;
* a table rebuilt with new rows leaves no orphaned cell widget drawing over
  the first cell;
* the direction the cue plots show is the measured one nearest the head view's
  marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtWidgets import QDoubleSpinBox, QTableWidget  # noqa: E402

from src.audio.sam_workbench.hrtf.measurement import ADVANCED_FLAG  # noqa: E402
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa  # noqa: E402
from src.audio.sam_workbench.parameters import (  # noqa: E402
    WORKBENCH_OWNED_KEYS,
    validate_sam2_params,
)
from src.audio.sam_workbench.render.routing import BusSpec, SourceRouting  # noqa: E402
from src.ui.head_view import HeadDirectionSelector, HeadDirectionView  # noqa: E402
from src.ui.sam_experiment_dialog import SamExperimentDialog  # noqa: E402
from src.ui.sam_hrtf_modify_panel import SamHrtfModifyPanel  # noqa: E402
from src.ui.sam_localization_dialog import SamLocalizationDialog  # noqa: E402
from src.ui.sam_measurement_dialog import SamMeasurementDialog  # noqa: E402
from src.ui.sam_modulation_panel import SamModulationPanel  # noqa: E402
from src.ui.sam_routing_panel import SamRoutingPanel  # noqa: E402
from src.ui.sam_stage_panel import SamStagePanel  # noqa: E402
from src.ui.table_widgets import clear_cell_widgets, place, reset_rows  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

STEPS = [
    {"duration": 120.0, "description": "induction"},
    {"duration": 300.0, "description": "deepening"},
    {"duration": 420.0, "description": "work"},
    {"duration": 180.0, "description": "return"},
]


# --- the head view ----------------------------------------------------------


def test_head_view_puts_the_listeners_left_on_the_left(qtbot):
    """A source drawn to the left of the widget must be heard on the left."""

    view = HeadDirectionView("top")
    qtbot.addWidget(view)
    view.resize(200, 200)

    for azimuth in (0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0):
        view.set_direction(azimuth, 0.0)
        point = view._to_point(azimuth, 0.0)
        recovered, _elevation = view._from_point(point)
        assert recovered == pytest.approx(azimuth, abs=1e-6)

    centre, radius = view._centre_radius()
    # Straight up the widget is in front; the listener's left is screen left.
    front, _ = view._from_point(type(centre)(centre.x(), centre.y() - radius))
    left, _ = view._from_point(type(centre)(centre.x() - radius, centre.y()))
    right, _ = view._from_point(type(centre)(centre.x() + radius, centre.y()))
    assert front == pytest.approx(0.0, abs=1e-6)
    assert left == pytest.approx(90.0, abs=1e-6)
    assert right == pytest.approx(-90.0, abs=1e-6)


def test_head_selector_keeps_both_views_in_step(qtbot):
    selector = HeadDirectionSelector()
    qtbot.addWidget(selector)
    selector.set_direction(30.0, 20.0)
    assert selector.top.azimuth_deg == pytest.approx(30.0)
    assert selector.side.elevation_deg == pytest.approx(20.0)

    with qtbot.waitSignal(selector.directionChanged):
        selector.top.set_direction(-60.0, -10.0, emit=True)
    assert selector.side.azimuth_deg == pytest.approx(-60.0)
    assert selector.elevation_deg == pytest.approx(-10.0)


def test_modify_plots_follow_the_head_view(qtbot):
    """Choosing a direction snaps the plots to the nearest measured one."""

    panel = SamHrtfModifyPanel()
    qtbot.addWidget(panel)
    dataset = load_sofa(str(FIXTURES / "synthetic_hrir.sofa"))
    panel.set_dataset(dataset)

    from src.audio.sam_workbench.hrtf.coordinates import cartesian_to_sofa_spherical

    spherical = cartesian_to_sofa_spherical(dataset.positions_m)
    for azimuth in (0.0, 90.0, -90.0, 180.0):
        panel.direction_selector.set_direction(azimuth, 0.0, emit=True)
        chosen = spherical[panel._plot_index()]
        assert float(chosen[0]) == pytest.approx(azimuth, abs=1e-6)

    # A direction between two measurements snaps to the nearer one rather than
    # interpolating: these plots show what the file contains.
    panel.direction_selector.set_direction(37.0, 0.0, emit=True)
    assert float(spherical[panel._plot_index()][0]) == pytest.approx(45.0, abs=1e-6)
    assert "45" in panel.direction_label.text()


# --- the modulation matrix --------------------------------------------------


def test_modulation_cell_round_trips_through_the_matrix(qtbot):
    panel = SamModulationPanel()
    qtbot.addWidget(panel)
    panel.add_target("voice", "beatFreq")

    panel.table.setCurrentCell(0, 0)
    panel.depth_spin.setValue(0.4)
    panel.polarity_combo.setCurrentIndex(1)
    panel.curve_combo.setCurrentText("smooth")
    assert panel.apply_cell() is True

    route = panel.matrix.route(panel._modulators[0], "voice", "beatFreq")
    assert route is not None
    assert route.depth == pytest.approx(0.4)
    assert route.polarity == -1
    assert route.curve == "smooth"

    restored = SamModulationPanel()
    qtbot.addWidget(restored)
    restored.set_params(panel.params())
    assert restored.matrix.describe() == panel.matrix.describe()


def test_modulation_refuses_a_cycle_and_names_it(qtbot):
    """A loop is invisible cell by cell, so refusing it must say what it is."""

    panel = SamModulationPanel()
    qtbot.addWidget(panel)
    panel.add_modulator("voice")
    panel.add_target("voice", "beatFreq")
    panel.add_target("lfo.slow", "rate")

    panel.table.setCurrentCell(panel._modulators.index("lfo.slow"), 0)
    panel.depth_spin.setValue(0.5)
    assert panel.apply_cell() is True

    panel.table.setCurrentCell(
        panel._modulators.index("voice"), panel._columns.index(("lfo.slow", "rate"))
    )
    panel.depth_spin.setValue(0.2)
    assert panel.apply_cell() is False
    assert "loop" in panel.status_label.text()
    assert "lfo.slow" in panel.status_label.text() and "voice" in panel.status_label.text()
    # Refused before it was stored, not stored and complained about.
    assert panel.matrix.route("voice", "lfo.slow", "rate") is None
    assert panel.matrix.find_cycle() == ()


def test_removing_a_modulator_removes_its_routes(qtbot):
    panel = SamModulationPanel()
    qtbot.addWidget(panel)
    panel.add_target("voice", "beatFreq")
    panel.table.setCurrentCell(0, 0)
    panel.depth_spin.setValue(0.3)
    panel.apply_cell()
    assert len(panel.matrix.routes) == 1

    panel.modulator_list.setCurrentRow(0)
    panel._remove_modulator()
    assert panel.matrix.routes == ()


# --- stages -----------------------------------------------------------------


def test_stages_are_built_from_steps_and_match_their_timing(qtbot):
    panel = SamStagePanel()
    qtbot.addWidget(panel)
    panel.set_steps(STEPS)
    panel.transition_spin.setValue(20.0)
    timeline = panel.build_from_steps()

    assert len(timeline.stages) == len(STEPS)
    assert timeline.duration_s == pytest.approx(sum(step["duration"] for step in STEPS))
    assert [stage.name for stage in timeline.stages] == [step["description"] for step in STEPS]

    clock = 0.0
    for stage, step in zip(timeline.stages, STEPS):
        assert stage.start_s == pytest.approx(clock)
        assert stage.duration_s == pytest.approx(step["duration"])
        clock += step["duration"]

    # The first stage has nothing to fade in from, and the last nothing to fade
    # out to; the joins between them carry the crossfade.
    assert timeline.stages[0].transition_in_s == pytest.approx(0.0)
    assert timeline.stages[-1].transition_out_s == pytest.approx(0.0)
    assert timeline.stages[1].transition_in_s == pytest.approx(20.0)


def test_stages_group_steps(qtbot):
    panel = SamStagePanel()
    qtbot.addWidget(panel)
    panel.set_steps(STEPS)
    panel.group_spin.setValue(2)
    timeline = panel.build_from_steps()

    assert len(timeline.stages) == 2
    assert timeline.stages[0].duration_s == pytest.approx(420.0)
    assert timeline.stages[1].duration_s == pytest.approx(600.0)


def test_stage_overrides_round_trip(qtbot):
    panel = SamStagePanel()
    qtbot.addWidget(panel)
    panel.set_steps(STEPS)
    panel.build_from_steps()
    panel.add_override("beatFreq", 4.5)

    restored = SamStagePanel()
    qtbot.addWidget(restored)
    restored.set_params(panel.params())
    assert restored.timeline.describe() == panel.timeline.describe()
    assert restored.timeline.stages[0].parameter_overrides[0].parameter_path == "beatFreq"


def test_timeline_view_maps_clicks_to_stages(qtbot):
    panel = SamStagePanel()
    qtbot.addWidget(panel)
    panel.set_steps(STEPS)
    panel.build_from_steps()
    view = panel.timeline_view
    view.resize(400, 120)

    for index, stage in enumerate(panel.timeline.stages):
        middle = 0.5 * (stage.start_s + stage.end_s)
        assert view._index_at(view._x_for(middle)) == index


# --- routing and cost -------------------------------------------------------


def test_routing_round_trips_and_tracks_cost(qtbot):
    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    before = panel.cost_inputs()

    panel._buses.append(BusSpec(id="reverb", name="Reverb send", gain_db=-3.0))
    panel._sources.append(SourceRouting(source_id="source.2", bus_id="reverb"))
    panel._refresh_buses()
    panel._refresh_sources()
    panel.set_crossovers((180.0, 1200.0, 6000.0))

    after = panel.cost_inputs()
    assert before.bands == 1 and after.bands == 4
    assert after.sources == 2
    assert after.convolutions > before.convolutions
    assert "4 band" in panel.cost_label.text()

    restored = SamRoutingPanel()
    qtbot.addWidget(restored)
    restored.set_params(panel.params())
    assert restored.describe() == panel.describe()


def test_master_bus_cannot_be_removed_and_orphans_come_home(qtbot):
    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    panel._buses.append(BusSpec(id="reverb", name="Reverb send"))
    panel._sources.append(SourceRouting(source_id="source.2", bus_id="reverb"))
    panel._refresh_buses()
    panel._refresh_sources()

    panel.bus_table.setCurrentCell(0, 0)
    panel._remove_bus()
    assert [bus.id for bus in panel.buses] == ["master", "reverb"]
    assert "master" in panel.status_label.text().lower()

    # Removing a bus must not leave a source pointing at nothing, which would
    # silently drop it from the mix.
    panel.bus_table.setCurrentCell(1, 0)
    panel._remove_bus()
    assert [bus.id for bus in panel.buses] == ["master"]
    assert all(source.bus_id == "master" for source in panel.sources)


def test_band_strip_positions_are_logarithmic(qtbot):
    from src.ui.sam_routing_panel import BandStripView

    strip = BandStripView()
    qtbot.addWidget(strip)
    assert strip._position(20.0) == pytest.approx(0.0)
    assert strip._position(20_000.0) == pytest.approx(1.0)
    # A decade is a decade wherever it sits, which is the point of a log axis.
    assert strip._position(200.0) - strip._position(20.0) == pytest.approx(
        strip._position(2000.0) - strip._position(200.0)
    )


def test_rebuilt_tables_leave_no_orphan_widgets(qtbot):
    """A replaced cell widget lingers until the event loop collects it."""

    table = QTableWidget(0, 2)
    qtbot.addWidget(table)

    def registered():
        return {
            table.cellWidget(row, column)
            for row in range(table.rowCount())
            for column in range(table.columnCount())
        }

    for rows in (1, 2, 3, 1):
        reset_rows(table, rows)
        for row in range(rows):
            place(table, row, 1, QDoubleSpinBox())
        orphans = [w for w in table.viewport().findChildren(QDoubleSpinBox) if w not in registered()]
        assert orphans == []

    clear_cell_widgets(table)
    assert table.viewport().findChildren(QDoubleSpinBox) == []


# --- the workbench dialog ---------------------------------------------------


def test_scene_panels_reach_the_voice_parameters(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)
    dialog.set_steps(STEPS)
    dialog.stage_panel.build_from_steps()
    dialog.modulation_panel.add_target("voice", "beatFreq")
    dialog.modulation_panel.table.setCurrentCell(0, 0)
    dialog.modulation_panel.depth_spin.setValue(0.25)
    dialog.modulation_panel.apply_cell()
    dialog.routing_panel.set_crossovers((300.0,))

    params = dialog.collect_params()
    for key in ("samStages", "samModulation", "samRouting"):
        assert key in params
        # The workbench owns these keys, so validation must not report them as
        # unknown and the compatibility view must not call them untouched.
        assert key in WORKBENCH_OWNED_KEYS
    assert validate_sam2_params(params) == ()
    assert "(none)" in dialog.compatibility_view.toPlainText()

    reopened = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": dict(params)}
    )
    qtbot.addWidget(reopened)
    restored = reopened.collect_params()
    for key in ("samStages", "samModulation", "samRouting"):
        assert restored[key] == params[key]


# --- the localization test --------------------------------------------------


def test_localization_test_stays_blind_until_it_is_scored(qtbot):
    dialog = SamLocalizationDialog(["alpha", "beta", "gamma"])
    qtbot.addWidget(dialog)
    dialog.repeats_spin.setValue(1)
    assert dialog.start() is True

    names = {"alpha", "beta", "gamma"}
    while dialog.current_trial() is not None:
        trial = dialog.current_trial()
        # Nothing on screen may name the dataset being presented.
        assert trial.label not in names
        assert not (names & set(dialog.trial_label.text().split()))
        dialog.selector.set_direction(trial.azimuth_deg, trial.elevation_deg)
        dialog.record_current()

    assert dialog.results_table.rowCount() == 3
    revealed = {
        dialog.results_table.item(row, 0).text()
        for row in range(dialog.results_table.rowCount())
    }
    assert revealed == names


def test_localization_scores_rank_the_listener_who_pointed_correctly(qtbot):
    dialog = SamLocalizationDialog(["good", "bad"])
    qtbot.addWidget(dialog)
    dialog.repeats_spin.setValue(1)
    dialog.start()

    while dialog.current_trial() is not None:
        trial = dialog.current_trial()
        if trial.candidate == "good":
            dialog.selector.set_direction(trial.azimuth_deg, trial.elevation_deg)
        else:
            # A front/back reversal, the failure that separates HRTFs.
            dialog.selector.set_direction(180.0 - trial.azimuth_deg, trial.elevation_deg)
        dialog.record_current()

    assert dialog.results_table.item(0, 0).text() == "good"
    assert "good" in dialog.status_label.text()


def test_localization_needs_two_candidates(qtbot):
    dialog = SamLocalizationDialog(["only-one"])
    qtbot.addWidget(dialog)
    assert dialog.start() is False
    assert "two candidates" in dialog.status_label.text()


def test_localization_export_carries_the_responses(qtbot, tmp_path):
    dialog = SamLocalizationDialog(["alpha", "beta"])
    qtbot.addWidget(dialog)
    dialog.repeats_spin.setValue(1)
    dialog.start()
    total = len(dialog.session.trials)
    while dialog.current_trial() is not None:
        dialog.record_current()

    written = dialog.export(str(tmp_path / "localization.json"))
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert len(payload["responses"]) == total
    assert len(payload["scores"]) == 2


# --- the listening experiment -----------------------------------------------


def test_experiment_runs_blinded_and_exports_the_reveal(qtbot, tmp_path):
    dialog = SamExperimentDialog(
        [
            {"id": "a", "path": "/library/a.sofa", "sha256": "a" * 64, "subject": "a"},
            {"id": "b", "path": "/library/b.sofa", "sha256": "b" * 64, "subject": "b"},
        ]
    )
    qtbot.addWidget(dialog)
    dialog.repeats_spin.setValue(2)
    assert dialog.start() is True

    while dialog.current_run() is not None:
        run = dialog.current_run()
        assert run.label not in ("a", "b")
        dialog.sliders["preference"].setValue(80 if run.condition_id == "a" else 20)
        dialog.record_current()

    aggregate = dialog.result.aggregate()
    assert aggregate["a"]["preference"] == pytest.approx(80.0)
    assert aggregate["b"]["preference"] == pytest.approx(20.0)
    assert dialog.result.completion == pytest.approx(1.0)

    written = dialog.export(str(tmp_path / "experiment.json"))
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert set(payload["reveal"].values()) == {"a", "b"}
    assert len(payload["ratings"]) == 4


def test_experiment_flags_conditions_it_could_not_rebuild(qtbot):
    dialog = SamExperimentDialog()
    qtbot.addWidget(dialog)
    dialog._add_named_condition = lambda: None  # the dialog's own prompt is modal
    from src.audio.sam_workbench.experiment.session import Condition

    dialog.add_condition(Condition(id="reproducible", asset_path="/a.sofa", asset_sha256="a" * 64))
    dialog.add_condition(Condition(id="vague", renderer="hrtf"))
    assert dialog.condition_table.item(0, 4).text() == "yes"
    assert dialog.condition_table.item(1, 4).text() != "yes"

    dialog.start()
    assert "vague" in dialog.status_label.text()


def test_experiment_needs_two_conditions(qtbot):
    dialog = SamExperimentDialog()
    qtbot.addWidget(dialog)
    assert dialog.start() is False
    assert not dialog.start_button.isEnabled()


# --- measurement and Mesh2HRTF ---------------------------------------------


def test_measurement_is_off_until_the_advanced_flag_is_set(qtbot, monkeypatch):
    monkeypatch.delenv(ADVANCED_FLAG, raising=False)
    dialog = SamMeasurementDialog()
    qtbot.addWidget(dialog)
    assert dialog.measure_body.isEnabled() is False
    # The refusal must say what to do about it and what to try first.
    assert ADVANCED_FLAG in dialog.gate_label.text()
    assert "localization test" in dialog.gate_label.text()


def test_measurement_enabled_by_the_flag_recovers_the_response(qtbot, monkeypatch, tmp_path):
    soundfile = pytest.importorskip("soundfile")
    monkeypatch.setenv(ADVANCED_FLAG, "1")

    dialog = SamMeasurementDialog()
    qtbot.addWidget(dialog)
    assert dialog.measure_body.isEnabled() is True
    dialog.duration_s.setValue(1.0)

    from src.audio.sam_workbench.hrtf.measurement import exponential_sweep

    rate = 48_000
    sweep = exponential_sweep(20.0, 20_000.0, 1.0, float(rate))
    truth = np.zeros(64)
    truth[10] = 1.0
    truth[25] = -0.3
    rng = np.random.default_rng(0)
    paths = []
    for take in range(3):
        # Well below full scale: a clipped take is refused, as it should be.
        recorded = 0.5 * np.convolve(sweep, truth)[: sweep.size]
        recorded = recorded + rng.normal(0.0, 1e-6, sweep.size)
        path = tmp_path / f"take{take}.wav"
        soundfile.write(str(path), recorded, rate)
        paths.append(str(path))

    dialog.add_recordings(paths)
    assert dialog.run_import() is True

    response = dialog._response
    peak = int(np.argmax(np.abs(response)))
    # The reflection is recovered at its own delay and amplitude.
    assert response[peak + 15] / response[peak] == pytest.approx(-0.3, abs=0.01)
    assert "signal-to-noise" in dialog.status_label.text()


def test_a_clipped_measurement_is_refused_rather_than_warned_about(qtbot, monkeypatch, tmp_path):
    soundfile = pytest.importorskip("soundfile")
    monkeypatch.setenv(ADVANCED_FLAG, "1")

    dialog = SamMeasurementDialog()
    qtbot.addWidget(dialog)
    dialog.duration_s.setValue(1.0)

    from src.audio.sam_workbench.hrtf.measurement import exponential_sweep

    sweep = exponential_sweep(20.0, 20_000.0, 1.0, 48_000.0)
    path = tmp_path / "clipped.wav"
    soundfile.write(str(path), np.clip(sweep * 4.0, -1.0, 1.0), 48_000)

    dialog.add_recordings([str(path)])
    assert dialog.run_import() is False
    assert dialog._response is None
    assert dialog.save_response_button.isEnabled() is False
    assert "did not pass" in dialog.quality_view.toPlainText()


def test_mesh_import_checks_plausibility_against_a_baseline(qtbot):
    dialog = SamMeasurementDialog()
    qtbot.addWidget(dialog)

    assert dialog.load_simulated(str(FIXTURES / "synthetic_hrir.sofa")) is True
    text = dialog.comparison_view.toPlainText()
    assert "interaural delay" in text
    # A horizon-only evaluation grid cannot carry elevation, and says so.
    messages = [
        dialog.issue_table.item(row, 2).text() for row in range(dialog.issue_table.rowCount())
    ]
    assert any("elevation" in message for message in messages)

    assert dialog.load_baseline(str(FIXTURES / "synthetic_sonicom_hrir.sofa")) is True
    compared = dialog.comparison_view.toPlainText()
    assert "correlation" in compared.lower()
    # Two sets of the same head model must agree about which way the delay runs.
    assert "1.00" in compared or "0.9" in compared


def test_mesh_guidance_is_shown_rather_than_automated(qtbot):
    from src.audio.sam_workbench.hrtf.mesh2hrtf import IMPORT_STEPS

    dialog = SamMeasurementDialog()
    qtbot.addWidget(dialog)
    labels = [child.text() for child in dialog.findChildren(type(dialog.gate_label))]
    guidance = "\n".join(labels)
    assert all(step in guidance for step in IMPORT_STEPS)

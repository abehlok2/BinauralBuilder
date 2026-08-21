"""GUI integration: the existing editor, the workbench dialog, and their panels.

The rules this file protects:

* the existing `VoiceEditorDialog` stays the entry point - no second main window;
* SAM parameter tables are generated from the one registry;
* a voice keeps every parameter it arrived with, including ones this build has
  never heard of;
* the workbench edits a copy, so Cancel cannot touch the live project;
* preview and export go through the same canonical renderer.
"""

from __future__ import annotations

import ast
import collections
import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtWidgets import QDialog, QMainWindow  # noqa: E402

from src.audio.sam_workbench.compat import render_sam2_voice  # noqa: E402
from src.audio.sam_workbench.parameters import (  # noqa: E402
    ADVANCED,
    BASIC,
    EXPERT,
    sam2_parameter_defaults,
)
from src.audio.sam_workbench.preview import render_voice_preview  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 44_100

SAM_VOICE = {
    "synth_function_name": "spatial_angle_modulation_sam2",
    "is_transition": False,
    "params": {
        "amp": 0.6,
        "carrierFreq": 220.0,
        "modFreq": 4.0,
        "arcWidthDeg": 90.0,
        "directionOffsetDeg": 10.0,
        "spatialScale": 1.0,
        "pathType": "open",
        "pathShape": "sinusoidal",
        "legacyMystery": "keep me",
    },
    "volume_envelope": None,
    "description": "",
}


class FakeApp:
    """The slice of `TrackEditorApp` that `VoiceEditorDialog` actually uses."""

    def __init__(self, voice: dict | None = None) -> None:
        self.track_data = {
            "global_settings": {"sample_rate": SAMPLE_RATE},
            "steps": [
                {"duration": 10.0, "voices": [copy.deepcopy(voice or SAM_VOICE)]}
            ],
        }
        self.prefs = None

    def get_selected_step_index(self) -> int:
        return 0

    def get_selected_voice_index(self) -> int:
        return 0


@pytest.fixture
def fake_app() -> FakeApp:
    return FakeApp()


@pytest.fixture
def voice_editor(qtbot, fake_app):
    from src.ui.voice_editor_dialog import VoiceEditorDialog

    dialog = VoiceEditorDialog(
        parent=None, app_ref=fake_app, step_index=0, voice_index=0
    )
    qtbot.addWidget(dialog)
    return dialog


# --- the registry reaches the existing editor -------------------------------


def test_sam_parameter_tables_are_declared_once():
    """The editor's parameter dictionary must not shadow a key with a later copy."""

    source = (REPOSITORY_ROOT / "src/ui/voice_editor_dialog.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    tables = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "param_definitions"
    ]
    assert tables, "param_definitions not found"
    for table in tables:
        keys = [key.value for key in table.value.keys]
        duplicates = [
            name for name, count in collections.Counter(keys).items() if count > 1
        ]
        assert duplicates == [], f"duplicate parameter tables: {duplicates}"


def test_editor_defaults_come_from_the_registry():
    from src.ui.voice_editor_dialog import get_default_params_for_function

    for is_transition in (False, True):
        editor_defaults = get_default_params_for_function(
            "spatial_angle_modulation_sam2", is_transition
        )
        registry_defaults = dict(sam2_parameter_defaults(is_transition))
        for name, default in registry_defaults.items():
            assert editor_defaults[name] == default


def test_editor_tooltips_come_from_the_registry():
    from src.audio.sam_workbench.parameters import field_for
    from src.ui.voice_editor_dialog import PARAM_TOOLTIPS

    tooltips = PARAM_TOOLTIPS["spatial_angle_modulation_sam2"]
    assert tooltips["carrierFreq"] == field_for("carrierFreq").tooltip
    assert tooltips["startCarrierFreq"] == field_for("carrierFreq").tooltip


# --- unknown parameters survive ---------------------------------------------


def test_saving_a_voice_keeps_parameters_the_editor_never_showed(
    voice_editor, fake_app
):
    assert voice_editor._preserved_params == {"legacyMystery": "keep me"}
    voice_editor.save_voice()
    saved = fake_app.track_data["steps"][0]["voices"][0]["params"]
    assert saved["legacyMystery"] == "keep me"
    assert saved["carrierFreq"] == pytest.approx(220.0)


def test_preset_round_trip_keeps_unknown_parameters(voice_editor, tmp_path):
    from src.utils.voice_file import VoicePreset, load_voice_preset, save_voice_preset

    collected = voice_editor._collect_data_from_ui()
    assert collected["params"]["legacyMystery"] == "keep me"

    path = tmp_path / "sam.voice"
    save_voice_preset(
        VoicePreset(
            synth_function_name=collected["synth_function_name"],
            is_transition=collected["is_transition"],
            params=collected["params"],
            volume_envelope=collected["volume_envelope"],
            description="",
        ),
        path,
    )
    restored = load_voice_preset(path)
    assert restored.params["legacyMystery"] == "keep me"

    voice_editor.apply_voice_preset(restored)
    assert voice_editor._collect_data_from_ui()["params"]["legacyMystery"] == "keep me"


# --- the workbench entry point ----------------------------------------------


def test_editor_offers_the_workbench_for_sam_voices(voice_editor):
    assert voice_editor.workbench_button.isEnabled()
    assert "Workbench" in voice_editor.workbench_button.text()


def test_workbench_button_is_disabled_for_other_voices(qtbot):
    from src.ui.voice_editor_dialog import VoiceEditorDialog

    other = copy.deepcopy(SAM_VOICE)
    other["synth_function_name"] = "binaural_beat"
    other["params"] = {"ampL": 0.5, "ampR": 0.5, "baseFreq": 200.0, "beatFreq": 4.0}
    dialog = VoiceEditorDialog(
        parent=None, app_ref=FakeApp(other), step_index=0, voice_index=0
    )
    qtbot.addWidget(dialog)
    assert not dialog.workbench_button.isEnabled()


def test_no_second_main_window_is_introduced():
    from src.ui import sam_workbench_dialog

    source = (REPOSITORY_ROOT / "src/ui/sam_workbench_dialog.py").read_text(
        encoding="utf-8"
    )
    assert "QMainWindow" not in source
    assert issubclass(sam_workbench_dialog.SamWorkbenchDialog, QDialog)
    assert not issubclass(sam_workbench_dialog.SamWorkbenchDialog, QMainWindow)


# --- the workbench dialog ---------------------------------------------------


@pytest.fixture
def workbench(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    voice = copy.deepcopy(SAM_VOICE)
    dialog = SamWorkbenchDialog(voice, sample_rate=SAMPLE_RATE)
    qtbot.addWidget(dialog)
    return dialog, voice


def test_workbench_loads_the_voice_into_its_controls(workbench):
    dialog, _ = workbench
    assert dialog.basic_panel.control_for("carrierFreq").value() == pytest.approx(220.0)
    assert dialog.basic_panel.control_for("pathType").value() == "open"


def test_workbench_edits_a_copy_so_cancel_changes_nothing(workbench):
    dialog, voice = workbench
    dialog.basic_panel.control_for("carrierFreq").set_value(333.0)
    dialog.basic_panel.flush()

    assert dialog.voice_data()["params"]["carrierFreq"] == pytest.approx(333.0)
    assert voice["params"]["carrierFreq"] == pytest.approx(220.0)

    dialog.reject()
    assert voice["params"]["carrierFreq"] == pytest.approx(220.0)


def test_apply_publishes_the_edited_voice(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    applied: list[dict] = []
    dialog = SamWorkbenchDialog(
        copy.deepcopy(SAM_VOICE), sample_rate=SAMPLE_RATE, on_apply=applied.append
    )
    qtbot.addWidget(dialog)
    dialog.basic_panel.control_for("modFreq").set_value(7.5)
    with qtbot.waitSignal(dialog.voiceApplied, timeout=1_000):
        dialog.apply_changes()
    assert applied[-1]["params"]["modFreq"] == pytest.approx(7.5)
    assert applied[-1]["params"]["legacyMystery"] == "keep me"


def test_workbench_preserves_parameters_it_does_not_edit(workbench):
    dialog, _ = workbench
    params = dialog.voice_data()["params"]
    assert params["legacyMystery"] == "keep me"
    assert (
        "Parameters preserved but not edited here"
        in dialog.compatibility_view.toPlainText()
    )
    assert "legacyMystery" in dialog.compatibility_view.toPlainText()


def test_validation_messages_are_bound_to_their_fields(qtbot):
    """A carrier above Nyquist badges its own field and blocks Apply."""

    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    voice = copy.deepcopy(SAM_VOICE)
    voice["params"]["carrierFreq"] = 30_000.0
    dialog = SamWorkbenchDialog(voice, sample_rate=SAMPLE_RATE)
    qtbot.addWidget(dialog)

    control = dialog.basic_panel.control_for("carrierFreq")
    assert "Nyquist" in control.badge.toolTip()
    assert control.badge.text() == "!"
    assert not dialog.buttons.button(dialog.buttons.Ok).isEnabled()
    assert not dialog.buttons.button(dialog.buttons.Apply).isEnabled()
    assert "Cannot apply" in dialog.status_label.text()

    control.set_value(220.0)
    dialog.basic_panel.flush()
    assert control.badge.toolTip() == ""
    assert dialog.buttons.button(dialog.buttons.Ok).isEnabled()


def test_disclosure_mode_reveals_the_advanced_tab(workbench):
    dialog, _ = workbench
    dialog.mode_combo.setCurrentText(BASIC)
    assert not dialog.tabs.isTabEnabled(1)
    dialog.mode_combo.setCurrentText(EXPERT)
    assert dialog.tabs.isTabEnabled(1)
    assert "earPolarity" in dialog.advanced_panel.parameter_names


def test_phase_three_path_tools_are_part_of_the_standard_workbench(workbench):
    dialog, _ = workbench
    labels = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
    assert "Path & Geometry" in labels
    assert dialog.path_panel.designer_button.isEnabled()
    assert "visual path designer" in dialog.path_panel.designer_button.text().lower()


def test_path_panel_preserves_legacy_profile_before_designer_is_accepted(qtbot):
    from src.ui.sam_path_panel import SamPathPanel

    profile = {"kind": "linear", "points": [[0.0, -100.0], [100.0, 0.0]]}
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    panel.set_params({"customPathProfile": profile})
    assert panel.params()["customPathProfile"] == profile
    assert "unchanged until you accept" in panel.metadata_label.text()


def test_visual_path_designer_writes_canonical_and_compatibility_models(qtbot):
    from src.ui.sam_path_editor_dialog import SamPathEditorDialog

    dialog = SamPathEditorDialog(
        profile={"kind": "linear", "points": [[0, -100], [-100, -100]]}
    )
    qtbot.addWidget(dialog)
    dialog.add_point((2.0, 0.5, 0.25))
    spec = dialog.trajectory_spec()
    profile = dialog.compatibility_profile()
    assert spec["coordinateFrame"] == "listener"
    assert spec["geometry"]["controlPointsM"][-1] == [2.0, 0.5, 0.25]
    assert spec["traversal"]["mode"] == "loop"
    assert profile["schemaVersion"] == 2
    assert profile["coordinateSpace"] == "normalized_listener_2d"
    assert profile["sceneUnitsPerMetre"] == pytest.approx(100.0)


def test_visual_designer_does_not_import_legacy_editors():
    source = (REPOSITORY_ROOT / "src/ui/sam_path_panel.py").read_text(encoding="utf-8")
    assert "custom_path_creator_dialog" not in source
    assert "spatial_trajectory_dialog" not in source


def test_advanced_values_are_only_written_in_advanced_modes(workbench):
    dialog, _ = workbench
    dialog.mode_combo.setCurrentText(BASIC)
    assert "rotationDirection" not in dialog.collect_params()

    dialog.mode_combo.setCurrentText(ADVANCED)
    dialog.advanced_panel.control_for("rotationDirection").set_value("ccw")
    assert dialog.collect_params()["rotationDirection"] == "ccw"


# --- preview ----------------------------------------------------------------


def test_preview_renders_in_a_worker_and_feeds_the_analysis_views(qtbot, workbench):
    dialog, _ = workbench
    dialog.preview_seconds.setValue(0.5)
    dialog.start_preview()
    qtbot.waitUntil(lambda: dialog.last_preview is not None, timeout=15_000)

    result = dialog.last_preview
    assert result.frames == pytest.approx(SAMPLE_RATE // 2, abs=2)
    assert "Preview ready" in dialog.status_label.text()
    assert dialog.analysis_panel.waveform_plot.series
    assert dialog.analysis_panel.ipd_plot.series
    assert "IPD" in dialog.analysis_panel.summary_label.text()

    qtbot.waitUntil(lambda: dialog.preview_button.isEnabled(), timeout=5_000)


def test_preview_audio_matches_the_export_renderer(workbench):
    dialog, _ = workbench
    params = dialog.voice_data()["params"]
    preview = render_voice_preview(
        dialog.voice_data(), sample_rate_hz=SAMPLE_RATE, duration_s=0.25, fade_ms=0.0
    )
    exported = render_sam2_voice(0.25, SAMPLE_RATE, params=params)
    np.testing.assert_allclose(preview.audio, exported, rtol=0, atol=1e-6)


def test_worker_reports_failure_without_crashing(qtbot, monkeypatch):
    """A render error becomes a message, not a traceback in a dead thread."""

    from src.ui import sam_workbench_dialog

    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(sam_workbench_dialog, "render_voice_preview", explode)
    worker = sam_workbench_dialog.PreviewWorker(SAM_VOICE, SAMPLE_RATE, 0.1)
    with qtbot.waitSignal(worker.failed, timeout=5_000) as blocker:
        worker.run()
    assert "synthetic render failure" in blocker.args[0]


def test_dialog_surfaces_a_failed_preview(qtbot, workbench):
    dialog, _ = workbench
    dialog._on_preview_failed("no renderer available")
    assert "Preview failed" in dialog.status_label.text()


# --- panels -----------------------------------------------------------------


def test_basic_panel_only_owns_its_own_fields(qtbot):
    from src.ui.sam_basic_panel import SamBasicPanel

    panel = SamBasicPanel(mode=BASIC)
    qtbot.addWidget(panel)
    merged = panel.apply_to({"customPathProfile": {"points": [[1, 2]]}, "vendorKey": 1})
    assert merged["customPathProfile"] == {"points": [[1, 2]]}
    assert merged["vendorKey"] == 1
    assert "carrierFreq" in merged


def test_basic_panel_coalesces_rapid_edits(qtbot):
    from src.ui.sam_basic_panel import SamBasicPanel

    panel = SamBasicPanel(mode=BASIC)
    qtbot.addWidget(panel)
    emitted: list[dict] = []
    panel.paramsChanged.connect(emitted.append)

    control = panel.control_for("carrierFreq")
    for value in (300.0, 310.0, 320.0, 330.0):
        control.set_value(value)
        control.valueEdited.emit("carrierFreq", value)

    assert emitted == []
    qtbot.waitUntil(lambda: len(emitted) == 1, timeout=2_000)
    assert emitted[-1]["carrierFreq"] == pytest.approx(330.0)


def test_reset_restores_the_registry_default(qtbot):
    from src.ui.sam_basic_panel import SamBasicPanel

    panel = SamBasicPanel(mode=BASIC)
    qtbot.addWidget(panel)
    control = panel.control_for("modFreq")
    control.set_value(11.0)
    control.reset_to_default()
    assert control.value() == pytest.approx(4.0)


def test_transition_panel_shows_start_and_end_rows(qtbot):
    from src.ui.sam_basic_panel import SamBasicPanel

    panel = SamBasicPanel(mode=BASIC, is_transition=True)
    qtbot.addWidget(panel)
    names = set(panel.parameter_names)
    assert {"startCarrierFreq", "endCarrierFreq"} <= names
    assert "carrierFreq" not in names
    assert "amp" in names  # shared across a transition


def test_analysis_panel_clears_for_unusable_audio(qtbot):
    from src.ui.sam_analysis_panel import SamAnalysisPanel

    panel = SamAnalysisPanel()
    qtbot.addWidget(panel)
    panel.set_audio(np.zeros((2, 2)), SAMPLE_RATE)
    assert panel.waveform_plot.series == []
    assert "Render a preview" in panel.summary_label.text()


def test_analysis_panel_plots_are_paintable(qtbot):
    from src.ui.sam_analysis_panel import SamAnalysisPanel

    panel = SamAnalysisPanel()
    qtbot.addWidget(panel)
    result = render_voice_preview(SAM_VOICE, sample_rate_hz=SAMPLE_RATE, duration_s=0.5)
    panel.set_audio(result.audio, SAMPLE_RATE)
    panel.resize(800, 600)
    panel.show()
    qtbot.waitExposed(panel)
    panel.repaint()
    assert len(panel.waveform_plot.series) == 2
    assert len(panel.frequency_plot.series) == 2
    x_range, y_range = panel.ipd_plot.data_ranges()
    assert x_range[1] > x_range[0]
    assert y_range[0] < 0.0 < y_range[1]

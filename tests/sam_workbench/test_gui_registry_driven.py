"""The GUI offers what the engine implements, and nothing else.

Every list of renderers, interpolation modes or delay policies repeated in a
widget is a list that can fall behind the engine. When it does, the failure is
not a crash: it is a control the user can set that rendering ignores, or a mode
the engine gained that nobody can reach. These tests hold the lists together.
"""

from __future__ import annotations

import pytest

from src.audio.sam_workbench.hrtf.interpolation import INTERPOLATION_MODES
from src.audio.sam_workbench.hrtf.sofa_io import DelayPolicy
from src.audio.sam_workbench.parameters import validate_sam2_params
from src.audio.sam_workbench.render.registry import REGISTRY


def test_the_lab_offers_exactly_the_engine_s_interpolation_modes():
    from src.ui.sam_hrtf_lab import INTERPOLATION_CHOICES

    assert tuple(value for _, value in INTERPOLATION_CHOICES) == tuple(INTERPOLATION_MODES)


def test_every_offered_interpolation_mode_is_labelled():
    """An unlabelled mode still appears; it must not appear as a bare key."""

    from src.ui.sam_hrtf_lab import INTERPOLATION_CHOICES

    for label, value in INTERPOLATION_CHOICES:
        assert label and label != value


def test_the_routing_panel_offers_exactly_the_engine_s_modes():
    from src.ui.sam_routing_panel import _INTERPOLATION_MODES

    assert tuple(_INTERPOLATION_MODES) == tuple(INTERPOLATION_MODES)


def test_the_delay_choices_are_the_delay_policies():
    from src.ui.sam_hrtf_lab import DELAY_CHOICES

    assert tuple(value for _, value in DELAY_CHOICES) == tuple(
        policy.value for policy in DelayPolicy
    )


# --- one validation, not two ------------------------------------------------


def test_the_gui_validator_runs_the_renderer_s_own_validation():
    """A config the GUI accepts and the plan rejects is a control that lies.

    The renderer definition validates its own configuration when the scene plan
    compiles. The parameter validator the dialog displays must run the same
    check, or the dialog will accept settings that production refuses.
    """

    params = {
        "rendererMode": "hrtf",
        "hrtfAsset": "/nonexistent.sofa",
        "hrtfOptions": {"maxAngularErrorDeg": -5.0},
    }
    from_gui = validate_sam2_params(params)
    from_registry = REGISTRY.get("hrtf").validate(params)

    assert from_registry, "fixture should provoke at least one renderer issue"
    gui_paths = {issue.path for issue in from_gui}
    for issue in from_registry:
        assert issue.path in gui_paths, f"{issue.path} is enforced but never shown"


def test_a_renderer_missing_its_asset_is_an_error_the_gui_shows():
    issues = validate_sam2_params({"rendererMode": "hrtf"})
    assert any("hrtfAsset" in issue.path for issue in issues)


# --- relevance --------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier, wants_hrtf",
    [("abstract_pm", False), ("geometric", False), ("hrtf", True), ("hybrid", True)],
)
def test_only_renderers_that_use_a_sofa_asset_enable_the_hrtf_tab(
    qtbot, identifier, wants_hrtf
):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)

    dialog.renderer_combo.setCurrentIndex(dialog.renderer_combo.findData(identifier))
    dialog._revalidate()

    assert dialog.tabs.isTabEnabled(dialog._hrtf_tab) is wants_hrtf
    # Whatever the mode, the reason is stated rather than left to be guessed.
    assert dialog.tabs.tabToolTip(dialog._hrtf_tab)


def test_all_four_renderers_are_reachable_from_the_dialog(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)

    offered = {
        dialog.renderer_combo.itemData(index)
        for index in range(dialog.renderer_combo.count())
    }
    assert {"abstract_pm", "geometric", "hrtf", "hybrid"} <= offered
    assert offered == {entry.identifier for entry in REGISTRY.voice_renderable}

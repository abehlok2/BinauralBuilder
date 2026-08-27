"""Every field a user can change should say what it does.

A number with a label and no explanation is a control someone has to guess
at, and guessing at "jump crossfade" is not a reasonable thing to ask. This
walks the constructed dialogs rather than the source, so a tooltip attached
through a shared helper counts just as much as one written inline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSlider,
)

from src.ui.sam_path3d_dialog import SamPath3DDialog

#: The kinds of widget that carry a value someone chooses.
VALUE_WIDGETS = (QAbstractSpinBox, QComboBox, QCheckBox, QLineEdit, QSlider)


def _orbit_spec():
    return {
        "schemaVersion": 2,
        "coordinateSystem": "listener_relative_cartesian",
        "geometry": {
            "type": "horizontal_orbit",
            "parameters": {
                "radius_m": 1.5,
                "elevation_deg": 0.0,
                "start_azimuth_deg": 0.0,
                "turns": 1.0,
            },
        },
        "transform": {},
        "traversal": {"mode": "loop", "durationS": 4.0},
    }


def _untipped(widget):
    found = []
    for child in widget.findChildren(VALUE_WIDGETS):
        # A spin box owns an internal line edit that is not separately
        # addressable; the box itself is what carries the explanation.
        if isinstance(child, QLineEdit) and isinstance(child.parent(), QAbstractSpinBox):
            continue
        if not child.toolTip().strip():
            found.append(f"{type(child).__name__} {child.objectName() or '<unnamed>'}")
    return found


def test_every_field_in_the_path_designer_explains_itself(qtbot):
    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)
    assert _untipped(widget) == []


def test_the_path_editor_explains_itself(qtbot):
    from src.ui.sam_path_editor_dialog import SamPathEditorDialog

    widget = SamPathEditorDialog()
    qtbot.addWidget(widget)
    assert _untipped(widget) == []


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("src.ui.sam_routing_panel", "SamRoutingPanel"),
        ("src.ui.sam_stage_panel", "SamStagePanel"),
        ("src.ui.sam_modulation_panel", "SamModulationPanel"),
    ],
)
def test_the_scene_panels_explain_themselves(qtbot, module_name, class_name):
    import importlib

    widget = getattr(importlib.import_module(module_name), class_name)()
    qtbot.addWidget(widget)
    assert _untipped(widget) == []


def test_each_axis_box_says_which_axis_it_is(qtbot):
    """Three identical spinners side by side do not say which is which, and
    getting it wrong puts a source behind the listener instead of in front."""

    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)

    tips = [spin.toolTip() for spin in widget.cartesian_row.spins]
    assert "X:" in tips[0] and "forward" in tips[0]
    assert "Y:" in tips[1] and "left" in tips[1]
    assert "Z:" in tips[2] and "above" in tips[2]


def test_describing_an_axis_row_keeps_the_per_axis_note(qtbot):
    from src.ui.sam_path3d_dialog import _AxisRow

    row = _AxisRow(" m")
    qtbot.addWidget(row)
    row.setToolTip("Where the whole path sits.")
    for spin in row.spins:
        assert "Where the whole path sits." in spin.toolTip()
    assert "X:" in row.spins[0].toolTip()


# --- the two the report singled out ------------------------------------------


def test_the_jump_controls_say_what_they_do_and_when_they_apply(qtbot):
    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)

    for spin in (widget.steps_spin, widget.crossfade_spin):
        tip = spin.toolTip()
        assert "discontinuous" in tip, "it must say which mode uses it"
        assert len(tip) > 80, "it must say what it does, not just name itself"


def test_the_jump_controls_are_disabled_outside_the_mode_that_uses_them(qtbot):
    """An enabled field the render ignores is a promise the output does not
    keep."""

    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)

    widget.mode_combo.setCurrentText("loop")
    assert widget.steps_spin.isEnabled() is False
    assert widget.crossfade_spin.isEnabled() is False

    widget.mode_combo.setCurrentText("discontinuous")
    assert widget.steps_spin.isEnabled() is True
    assert widget.crossfade_spin.isEnabled() is True


def test_a_disabled_jump_control_says_why_it_is_disabled(qtbot):
    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)
    widget.mode_combo.setCurrentText("loop")
    tip = widget.steps_spin.toolTip()
    assert "Inactive:" in tip
    assert "ignored by preview and export" in tip


def test_every_hrtf_rating_criterion_is_explained(qtbot):
    """The criteria come from the core, so adding one there must not leave an
    unexplained slider behind here."""

    from src.audio.sam_workbench.hrtf.subject_test import RATING_CRITERIA
    from src.ui.sam_hrtf_lab import _CRITERION_TOOLTIPS

    assert set(RATING_CRITERIA) <= set(_CRITERION_TOOLTIPS)

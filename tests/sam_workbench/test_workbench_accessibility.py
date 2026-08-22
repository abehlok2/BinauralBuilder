"""Reaching the workbench without a mouse, and without seeing it.

Two audiences, one set of requirements. A screen reader announces a widget's
accessible name; a widget without one is announced as its class, so a row of
controls becomes "combo box, combo box, spin box" and the dialog is unusable.
And a keyboard user needs every action to have a key, in an order that follows
the dialog rather than the order the widgets were constructed in.

The rules this file protects:

* every control a user acts on has an accessible name and a description;
* preview, play and stop have shortcuts, and the shortcuts are written down
  where someone might find them;
* every tab is reachable by key;
* the orientation note can be dismissed, and stays dismissed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QKeySequence  # noqa: E402

from src.ui.sam_workbench_dialog import SamWorkbenchDialog  # noqa: E402

VOICE = {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}


class _MemorySettings:
    """QSettings' interface, without touching the developer's real settings."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key, default=None, type=None):  # noqa: A002 - QSettings' signature
        return self.values.get(key, default)

    def setValue(self, key, value) -> None:  # noqa: N802 - QSettings' signature
        self.values[key] = value


@pytest.fixture
def workbench(qtbot, monkeypatch):
    settings = _MemorySettings()
    monkeypatch.setattr(SamWorkbenchDialog, "_onboarding_settings", staticmethod(lambda: settings))
    dialog = SamWorkbenchDialog(dict(VOICE))
    qtbot.addWidget(dialog)
    return dialog, settings


def test_every_control_a_user_acts_on_is_named(workbench):
    dialog, _settings = workbench
    controls = (
        dialog.renderer_combo,
        dialog.mode_combo,
        dialog.tabs,
        dialog.preview_seconds,
        dialog.preview_button,
        dialog.play_button,
        dialog.stop_button,
        dialog.status_label,
        dialog.compatibility_view,
    )
    for control in controls:
        assert control.accessibleName(), control
        # A name says what it is; the description says what it does. A control
        # with neither is announced as its class.
        assert control.accessibleDescription(), control.accessibleName()


def test_the_transport_has_keys_and_says_what_they_are(workbench):
    dialog, _settings = workbench
    assert dialog.preview_button.shortcut() == QKeySequence("Ctrl+R")
    assert dialog.play_button.shortcut() == QKeySequence("Ctrl+P")
    assert dialog.stop_button.shortcut() == QKeySequence("Ctrl+.")

    for button in (dialog.preview_button, dialog.play_button, dialog.stop_button):
        assert button.shortcut().toString() in button.toolTip()


def test_every_tab_is_reachable_from_the_keyboard(workbench):
    dialog, _settings = workbench
    assert len(dialog._tab_shortcuts) == dialog.tabs.count()
    for index, shortcut in enumerate(dialog._tab_shortcuts):
        assert shortcut.key() == QKeySequence(f"Ctrl+{index + 1}")
        shortcut.activated.emit()
        assert dialog.tabs.currentIndex() == index


def test_tab_order_follows_the_dialog_rather_than_construction(workbench):
    dialog, _settings = workbench
    order = [
        dialog.renderer_combo,
        dialog.mode_combo,
        dialog.tabs,
        dialog.preview_seconds,
        dialog.preview_button,
        dialog.play_button,
        dialog.stop_button,
    ]
    for widget in order:
        assert widget.focusPolicy() != Qt.NoFocus, widget.accessibleName()


def test_the_orientation_note_can_be_dismissed_for_good(workbench):
    dialog, settings = workbench
    assert dialog.onboarding.isVisibleTo(dialog) is True

    dialog.dismiss_onboarding()
    assert dialog.onboarding.isVisibleTo(dialog) is False
    assert settings.values["onboardingDismissed"] is True


def test_a_dismissed_orientation_note_does_not_come_back(qtbot, monkeypatch):
    settings = _MemorySettings()
    settings.values["onboardingDismissed"] = True
    monkeypatch.setattr(SamWorkbenchDialog, "_onboarding_settings", staticmethod(lambda: settings))

    dialog = SamWorkbenchDialog(dict(VOICE))
    qtbot.addWidget(dialog)
    assert dialog.onboarding.isVisibleTo(dialog) is False


def test_unreadable_settings_do_not_stop_the_dialog_opening(qtbot, monkeypatch):
    """A preference store is a convenience; it must not be a dependency."""

    def explode():
        raise OSError("settings are not writable here")

    monkeypatch.setattr(SamWorkbenchDialog, "_onboarding_settings", staticmethod(explode))
    dialog = SamWorkbenchDialog(dict(VOICE))
    qtbot.addWidget(dialog)
    assert dialog.onboarding.isVisibleTo(dialog) is True
    dialog.dismiss_onboarding()
    assert dialog.onboarding.isVisibleTo(dialog) is False


def test_a_failed_preview_is_reported_in_the_status_line_not_a_dialog(workbench, monkeypatch):
    """An error the user can read and keep working around beats a modal."""

    dialog, _settings = workbench
    dialog._on_preview_failed("the renderer refused: no HRTF asset selected")
    assert "no HRTF asset selected" in dialog.status_label.text()
    assert dialog.preview_button.isEnabled()

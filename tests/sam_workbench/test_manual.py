"""The in-app manual stays complete and reachable from the workbench."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from src.audio.sam_workbench.parameters import SAM2_FIELDS
from src.ui.sam_workbench_dialog import SamWorkbenchDialog
from src.ui.sam_workbench_manual import SamWorkbenchManualDialog, manual_html


VOICE = {
    "synth_function_name": "spatial_angle_modulation_sam2",
    "is_transition": False,
    "params": {},
}


def test_manual_documents_every_registered_parameter():
    document = manual_html()
    for field in SAM2_FIELDS:
        assert field.label in document
        assert field.name in document
        assert field.tooltip in document


def test_manual_is_searchable(qtbot):
    manual = SamWorkbenchManualDialog()
    qtbot.addWidget(manual)
    manual.search_edit.setText("interaural delay")
    manual.find_next()
    assert manual.search_status.text() == 'Found "interaural delay".'


def test_workbench_opens_and_reuses_manual(qtbot):
    workbench = SamWorkbenchDialog(VOICE)
    qtbot.addWidget(workbench)
    assert workbench.manual_button.shortcut().toString() == "F1"

    workbench.open_manual()
    first = workbench._manual_dialog
    qtbot.addWidget(first)
    assert first.isVisible()

    workbench.open_manual()
    assert workbench._manual_dialog is first

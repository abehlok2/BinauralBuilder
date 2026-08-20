import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.audio.sam_workbench.model import Project, Source
from src.audio.sam_workbench.preview import PreviewBoundary, ProjectPreviewRenderer


def test_preview_matches_renderer_and_swaps_at_block_boundary():
    first = Project(sources=(Source(amplitude_linear=0.1),))
    second = replace(first, sources=(replace(first.sources[0], amplitude_linear=0.3),))
    boundary = PreviewBoundary(ProjectPreviewRenderer(first))
    initial = np.empty((2, 64), dtype=np.float32)
    boundary.process_into(initial)
    boundary.submit(ProjectPreviewRenderer(second))
    changed = np.empty_like(initial)
    boundary.process_into(changed)
    np.testing.assert_array_equal(changed, ProjectPreviewRenderer(second).process(64))
    assert boundary.telemetry.renderer_swaps == 1
    assert boundary.telemetry.rendered_blocks == 2


def test_pending_preview_queue_is_bounded_and_latest_wins():
    project = Project()
    boundary = PreviewBoundary()
    for gain in (0.1, 0.2, 0.4):
        changed = replace(project, sources=(replace(project.sources[0], amplitude_linear=gain),))
        boundary.submit(ProjectPreviewRenderer(changed))
    output = np.empty((2, 16), dtype=np.float32)
    boundary.process_into(output)
    expected = replace(project, sources=(replace(project.sources[0], amplitude_linear=0.4),))
    np.testing.assert_array_equal(output, ProjectPreviewRenderer(expected).process(16))


def test_gui_edits_are_undoable_and_update_snapshot(tmp_path):
    pytest.importorskip("PyQt5")
    try:
        from PyQt5.QtCore import QSettings
        from PyQt5.QtWidgets import QApplication
        from src.audio.sam_workbench.gui import WorkbenchWindow
    except ImportError as error:
        pytest.skip(f"Qt runtime unavailable: {error}")

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    window = WorkbenchWindow(Project(name="Before"), settings)
    window.rename_project("After")
    assert window.project.name == "After"
    window.undo_stack.undo()
    assert window.project.name == "Before"
    window.undo_stack.redo()
    assert window.project.name == "After"
    window.close()
    app.processEvents()

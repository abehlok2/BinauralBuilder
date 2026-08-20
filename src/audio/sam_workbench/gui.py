"""Phase-two PyQt5 shell for editing immutable SAM project snapshots."""
from __future__ import annotations
from dataclasses import replace
from typing import Callable
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (QApplication, QDockWidget, QLabel, QListWidget,
    QMainWindow, QTabWidget, QTextEdit, QToolBar, QUndoCommand, QUndoStack, QWidget)
from .model import Project
from .preview import PreviewBoundary, ProjectPreviewRenderer

class ReplaceProjectCommand(QUndoCommand):
    def __init__(self, window: "WorkbenchWindow", before: Project, after: Project, text: str):
        super().__init__(text); self.window = window; self.before = before; self.after = after
    def redo(self): self.window._install_project(self.after)
    def undo(self): self.window._install_project(self.before)

class WorkbenchWindow(QMainWindow):
    """Persistent dock shell; widgets never own DSP or canonical geometry."""
    def __init__(self, project: Project | None = None, settings: QSettings | None = None):
        super().__init__(); self.project = project or Project()
        self.settings = settings or QSettings("BinauralBuilder", "SAMWorkbench")
        self.undo_stack = QUndoStack(self)
        self.preview = PreviewBoundary(ProjectPreviewRenderer(self.project))
        self.setObjectName("samWorkbenchWindow"); self._build_ui(); self._restore_layout()
    def _build_ui(self):
        toolbar = QToolBar("Transport", self); toolbar.setObjectName("transportToolbar")
        toolbar.addWidget(QLabel("Stopped")); self.addToolBar(toolbar)
        tabs = QTabWidget(self)
        for title in ("Path", "Timeline", "Mixer", "HRTF Lab", "Experiments"):
            page = QWidget(tabs); page.setObjectName(title.lower().replace(" ", "_")); tabs.addTab(page, title)
        self.setCentralWidget(tabs)
        self.scene_tree = QListWidget(self); self._add_dock("Scene", self.scene_tree, Qt.LeftDockWidgetArea, "sceneDock")
        self.inspector = QTextEdit(self); self.inspector.setReadOnly(True)
        self._add_dock("Inspector", self.inspector, Qt.RightDockWidgetArea, "inspectorDock")
        self.issues = QTextEdit(self); self.issues.setReadOnly(True)
        self._add_dock("Analysis / Render / Issues / Log", self.issues, Qt.BottomDockWidgetArea, "issuesDock")
        menu = self.menuBar().addMenu("&Edit"); menu.addAction(self.undo_stack.createUndoAction(self, "Undo")); menu.addAction(self.undo_stack.createRedoAction(self, "Redo"))
        self._refresh_views()
    def _add_dock(self, title, widget, area, name):
        dock = QDockWidget(title, self); dock.setObjectName(name); dock.setWidget(widget); self.addDockWidget(area, dock)
    def edit_project(self, transform: Callable[[Project], Project], description: str):
        updated = transform(self.project)
        if not isinstance(updated, Project): raise TypeError("project transform must return Project")
        self.undo_stack.push(ReplaceProjectCommand(self, self.project, updated, description))
    def rename_project(self, name: str): self.edit_project(lambda p: replace(p, name=name), "Rename project")
    def _install_project(self, project):
        renderer = ProjectPreviewRenderer(project); self.project = project; self.preview.submit(renderer); self._refresh_views()
    def _refresh_views(self):
        self.setWindowTitle(self.project.name); self.scene_tree.clear(); self.scene_tree.addItems(s.name for s in self.project.sources)
        self.inspector.setPlainText(f"Project: {self.project.name}\nSample rate: {self.project.audio.sample_rate_hz} Hz\nRenderer: abstract SAM\nSources: {len(self.project.sources)}")
        self.statusBar().showMessage(f"{self.project.audio.sample_rate_hz} Hz | abstract | 0 frames latency | modified")
    def _restore_layout(self):
        geometry = self.settings.value("main/geometry"); state = self.settings.value("main/state")
        if geometry is not None: self.restoreGeometry(geometry)
        if state is not None: self.restoreState(state)
    def closeEvent(self, event):
        self.settings.setValue("main/geometry", self.saveGeometry()); self.settings.setValue("main/state", self.saveState()); super().closeEvent(event)

def run_gui(project: Project | None = None) -> int:
    app = QApplication.instance() or QApplication([]); window = WorkbenchWindow(project); window.show(); return app.exec_()

"""Phase-3 path editing and canonical trajectory/cue preview."""

from __future__ import annotations

import copy
import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.analysis.trajectory_metrics import trajectory_metrics
from src.audio.sam_workbench.trajectory.legacy_paths import (
    legacy_profile_geometry,
)
from src.audio.sam_workbench.trajectory import (
    CanonicalTrajectory,
    Polyline,
    Spline,
    Traversal,
    segment_positions,
    trajectory_from_dict,
)
from .sam_analysis_panel import PlotSeries, PlotWidget
from .sam_path3d_dialog import SamPath3DDialog
from .sam_path_editor_dialog import SamPathEditorDialog


class TrajectoryPreview(QWidget):
    """Small top-down canonical view (+x forward, +y left)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = np.empty((0, 3))
        self.setMinimumHeight(220)

    def set_points(self, points):
        self._points = np.asarray(points, dtype=float)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawText(8, 18, "Top view — forward +x, left +y (metres)")
        rect = self.rect().adjusted(28, 28, -20, -22)
        painter.setPen(QPen(Qt.gray, 1, Qt.DashLine))
        painter.drawLine(
            rect.center().x(), rect.top(), rect.center().x(), rect.bottom()
        )
        painter.drawLine(
            rect.left(), rect.center().y(), rect.right(), rect.center().y()
        )
        if len(self._points) < 2:
            return
        xy = self._points[:, :2]
        extent = max(float(np.max(np.abs(xy))), 0.1)
        path = QPainterPath()
        for index, (forward, left) in enumerate(xy):
            point = QPointF(
                rect.center().x() - left / extent * rect.width() * 0.45,
                rect.center().y() - forward / extent * rect.height() * 0.45,
            )
            path.moveTo(point) if index == 0 else path.lineTo(point)
        painter.setPen(QPen(Qt.cyan, 2))
        painter.drawPath(path)


class SamPathPanel(QWidget):
    """Launch and preview the metre-based interactive path designer."""

    paramsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params = {}
        self._profile = {}
        self._segments = []
        self._trajectory_spec = {}
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.designer_button = QPushButton("Open visual path designer…")
        self.designer_button.setToolTip(
            "Edit x/y from a top-down metre grid and enter elevation (z) numerically. This is not a full 3D viewer."
        )
        self.designer_button.clicked.connect(self.open_designer)
        buttons.addWidget(self.designer_button)
        self.designer_3d_button = QPushButton("Open 3D path designer…")
        self.designer_3d_button.setToolTip(
            "Edit the full three-dimensional path: perspective, top, front and "
            "side views, numeric x/y/z and azimuth/elevation/distance entry, "
            "keyframes, and the 3-D primitives. Geometry and traversal are "
            "edited separately."
        )
        self.designer_3d_button.clicked.connect(self.open_3d_designer)
        buttons.addWidget(self.designer_3d_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.metadata_label = QLabel()
        self.metadata_label.setWordWrap(True)
        layout.addWidget(self.metadata_label)
        self.preview = TrajectoryPreview()
        layout.addWidget(self.preview)
        plots = QGridLayout()
        self.itd_plot = PlotWidget("Geometric ITD", x_label="time (s)", y_label="s")
        self.ild_plot = PlotWidget("Geometric ILD", x_label="time (s)", y_label="dB")
        self.distance_plot = PlotWidget(
            "Listener distance", x_label="time (s)", y_label="m"
        )
        plots.addWidget(self.itd_plot, 0, 0)
        plots.addWidget(self.ild_plot, 0, 1)
        plots.addWidget(self.distance_plot, 1, 0, 1, 2)
        layout.addLayout(plots)

    def set_params(self, params):
        self._params = copy.deepcopy(dict(params))
        self._profile = copy.deepcopy(self._params.get("customPathProfile") or {})
        self._segments = copy.deepcopy(self._params.get("spatialTrajectory") or [])
        self._trajectory_spec = copy.deepcopy(
            self._params.get("canonicalTrajectory") or {}
        )
        self.refresh_preview()

    def params(self):
        result = {}
        if self._profile:
            result["customPathProfile"] = copy.deepcopy(self._profile)
        if self._segments or "spatialTrajectory" in self._params:
            result["spatialTrajectory"] = copy.deepcopy(self._segments)
        if self._trajectory_spec:
            result["canonicalTrajectory"] = copy.deepcopy(self._trajectory_spec)
        return result

    def open_designer(self):
        dialog = SamPathEditorDialog(self._profile, self._trajectory_spec, self)
        if dialog.exec_() == QDialog.Accepted:
            self._profile = dialog.compatibility_profile()
            self._trajectory_spec = dialog.trajectory_spec()
            self.refresh_preview()
            self.paramsChanged.emit(self.params())

    def open_3d_designer(self):
        """Edit the canonical trajectory in the multi-view three-dimensional editor.

        The legacy compatibility profile is left alone: this editor writes the
        canonical path, and a two-dimensional projection of a path with height
        in it would be a worse record of the author's intent than no update.
        """

        dialog = SamPath3DDialog(self._trajectory_spec, self)
        if dialog.exec_() == QDialog.Accepted:
            self._trajectory_spec = dialog.trajectory_spec()
            self.refresh_preview()
            self.paramsChanged.emit(self.params())

    def refresh_preview(self):
        if self._profile:
            version = self._profile.get("schemaVersion", "legacy (unversioned)")
            scale = self._profile.get("sceneUnitsPerMetre", 100.0)
            self.metadata_label.setText(
                f"Custom path: {version}; {scale:g} scene units/m. Legacy data is unchanged until you accept the visual designer."
            )
        elif self._segments:
            self.metadata_label.setText(
                f"Legacy trajectory: {len(self._segments)} segment(s); saved form is preserved."
            )
        else:
            self.metadata_label.setText(
                "No custom path or trajectory segments. Use an editor above to create one."
            )
        try:
            if self._trajectory_spec:
                traversal_data = self._trajectory_spec.get("traversal", {})
                duration = float(traversal_data.get("durationS", 5.0))
                times = np.linspace(0.0, duration, 512)
                trajectory = trajectory_from_dict(self._trajectory_spec)
                points = trajectory.evaluate(times)
            elif self._segments:
                times = np.linspace(
                    0.0,
                    max(sum(float(s.get("seconds", 0)) for s in self._segments), 1.0),
                    512,
                )
                points = segment_positions(self._segments, times)
            elif len(self._profile.get("points", [])) >= 2:
                times = np.linspace(0.0, 1.0, 512)
                points = legacy_profile_geometry(self._profile).evaluate(
                    np.linspace(0, 1, 512)
                )
            else:
                raise ValueError
            self.preview.set_points(points)
            metrics = trajectory_metrics(points, 1.0 / (times[1] - times[0]))
            self.itd_plot.set_series([PlotSeries(times, metrics["itd_s"], name="ITD")])
            self.ild_plot.set_series([PlotSeries(times, metrics["ild_db"], name="ILD")])
            distance = np.linalg.norm(points, axis=1)
            self.distance_plot.set_series(
                [PlotSeries(times, distance, name="distance")]
            )
        except (ValueError, TypeError, KeyError):
            self.preview.set_points([])
            for plot in (self.itd_plot, self.ild_plot, self.distance_plot):
                plot.clear("Create a valid path to preview cues")

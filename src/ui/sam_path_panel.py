"""Trajectory views for the SAM workbench.

A top-down path plot in the canonical listener frame - forward up the screen,
left to the left, ears marked where they actually are - plus the binaural cues
the geometry implies over time. Everything is computed by
:mod:`src.audio.sam_workbench.trajectory` and
:mod:`src.audio.sam_workbench.analysis`, so the picture and the render come
from the same numbers.

The panel also owns the legacy bridge in the GUI: it shows what a saved
custom-path profile means in metres, and offers the explicit migration that
writes coordinate metadata into the profile. Nothing is rewritten unless the
user asks for it.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.analysis import (
    predicted_ild_db,
    predicted_itd_s,
    trajectory_summary,
)
from src.audio.sam_workbench.trajectory import (
    LegacyPathTransform,
    ListenerFrame,
    PROFILE_VERSION,
    TrajectorySpec,
    profile_points,
    profile_round_trip,
    profile_schema_version,
    profile_to_geometry,
    profile_transform,
    upgrade_profile,
)

from .sam_analysis_panel import LEFT_COLOR, RIGHT_COLOR, PlotSeries, PlotWidget

__all__ = ["PathPlotWidget", "SamPathPanel"]

PATH_COLOR = QColor(120, 190, 120)
HEAD_COLOR = QColor(150, 150, 150)
SOURCE_COLOR = QColor(235, 200, 90)


class PathPlotWidget(QWidget):
    """Top-down view of a path in the canonical listener frame.

    Forward (``+x``) is up the screen and left (``+y``) is to the left, so the
    plot reads the way a listener would describe it. The ear markers are drawn
    from the listener's own head radius rather than from a decorative constant,
    which is what makes the picture comparable with the rendered cues.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path_m: np.ndarray | None = None
        self._source_m: np.ndarray | None = None
        self._head_radius_m = 0.0875
        self._message = "No path to show"
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_path(
        self,
        path_m: np.ndarray | None,
        *,
        source_m: np.ndarray | None = None,
        head_radius_m: float = 0.0875,
    ) -> None:
        self._path_m = None if path_m is None else np.asarray(path_m, dtype=np.float64)
        self._source_m = None if source_m is None else np.asarray(source_m, dtype=np.float64)
        self._head_radius_m = float(head_radius_m)
        self._message = "" if self._path_m is not None else "No path to show"
        self.update()

    def clear(self, message: str = "No path to show") -> None:
        self._path_m = None
        self._source_m = None
        self._message = message
        self.update()

    @property
    def path_m(self) -> np.ndarray | None:
        return self._path_m

    def extent_m(self) -> float:
        """Half-width of the view in metres, always showing the whole head."""

        if self._path_m is None or self._path_m.size == 0:
            return 1.0
        reach = float(np.max(np.abs(self._path_m[:2]))) if self._path_m.shape[1] else 1.0
        return max(0.5, reach * 1.15)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        area = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
        side = min(area.width(), area.height())
        square = QRectF(
            area.center().x() - side / 2.0, area.center().y() - side / 2.0, side, side
        )

        painter.setPen(QPen(QColor(128, 128, 128, 160), 1.0))
        painter.drawRect(square)
        if self._path_m is None:
            painter.drawText(square, Qt.AlignCenter, self._message)
            painter.end()
            return

        extent = self.extent_m()
        scale = (side / 2.0) / extent

        def to_point(forward: float, left: float) -> QPointF:
            # Forward is up the screen; left is to the left.
            return QPointF(square.center().x() - left * scale, square.center().y() - forward * scale)

        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)

        painter.setPen(QPen(QColor(128, 128, 128, 70), 1.0, Qt.DotLine))
        painter.drawLine(to_point(-extent, 0.0), to_point(extent, 0.0))
        painter.drawLine(to_point(0.0, -extent), to_point(0.0, extent))
        painter.setPen(QPen(QColor(128, 128, 128, 200), 1.0))
        painter.drawText(QRectF(square.center().x() - 40, square.top() + 2, 80, 14), Qt.AlignCenter, "front")
        painter.drawText(QRectF(square.left() + 2, square.center().y() - 8, 40, 14), Qt.AlignLeft, "left")
        painter.drawText(
            QRectF(square.right() - 42, square.center().y() - 8, 40, 14), Qt.AlignRight, "right"
        )
        painter.drawText(
            QRectF(square.left() + 2, square.bottom() - 16, 90, 14), Qt.AlignLeft, f"{extent:.2f} m"
        )

        head = self._head_radius_m * scale
        painter.setPen(QPen(HEAD_COLOR, 1.2))
        painter.drawEllipse(square.center(), head * 1.6, head * 1.6)
        for label, sign, colour in (("L", 1.0, LEFT_COLOR), ("R", -1.0, RIGHT_COLOR)):
            marker = to_point(0.0, sign * self._head_radius_m)
            painter.setPen(QPen(QColor(colour), 1.5))
            painter.drawEllipse(marker, 3.5, 3.5)
            painter.drawText(QRectF(marker.x() - 12, marker.y() - 20, 24, 14), Qt.AlignCenter, label)

        painter.setPen(QPen(PATH_COLOR, 1.8))
        path = QPainterPath()
        path.moveTo(to_point(float(self._path_m[0, 0]), float(self._path_m[1, 0])))
        for index in range(1, self._path_m.shape[1]):
            path.lineTo(to_point(float(self._path_m[0, index]), float(self._path_m[1, index])))
        painter.drawPath(path)

        if self._source_m is not None and self._source_m.size >= 3:
            marker = to_point(float(self._source_m[0]), float(self._source_m[1]))
            painter.setPen(QPen(SOURCE_COLOR, 2.0))
            painter.drawEllipse(marker, 5.0, 5.0)
        painter.end()


class SamPathPanel(QWidget):
    """Path view, predicted cues, and the legacy profile bridge."""

    PLOT_POINTS = 400
    profileMigrated = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path_plot = PathPlotWidget()
        self.itd_plot = PlotWidget("Predicted ITD", x_label="s", y_label="ms")
        self.ild_plot = PlotWidget("Predicted ILD", x_label="s", y_label="dB")
        self.summary_label = QLabel("No trajectory loaded.")
        self.summary_label.setWordWrap(True)

        self.legacy_label = QLabel("No custom path profile in this voice.")
        self.legacy_label.setWordWrap(True)
        self.scene_units_spin = QDoubleSpinBox()
        self.scene_units_spin.setRange(1.0, 100_000.0)
        self.scene_units_spin.setDecimals(2)
        self.scene_units_spin.setValue(LegacyPathTransform().scene_units_per_metre)
        self.scene_units_spin.setToolTip(
            "How many scene units of the legacy editor make one metre. The default "
            "puts the editor's ear markers at the listener's head radius."
        )
        self.scene_units_spin.valueChanged.connect(lambda _: self._refresh_legacy())
        self.migrate_button = QPushButton("Add coordinate metadata to profile…")
        self.migrate_button.setToolTip(
            "Write the declared coordinate space into the saved profile. Legacy "
            "profiles keep working untouched until you commit this migration."
        )
        self.migrate_button.setEnabled(False)
        self.migrate_button.clicked.connect(self.migrate_profile)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.addWidget(self.path_plot, 0, 0, 2, 1)
        grid.addWidget(self.itd_plot, 0, 1)
        grid.addWidget(self.ild_plot, 1, 1)
        layout.addLayout(grid, 1)
        layout.addWidget(self.summary_label)

        legacy_box = QGroupBox("Legacy custom path")
        legacy_form = QFormLayout(legacy_box)
        legacy_form.addRow(self.legacy_label)
        legacy_form.addRow("Scene units per metre:", self.scene_units_spin)
        legacy_form.addRow(self.migrate_button)
        layout.addWidget(legacy_box)

        self._trajectory: TrajectorySpec | None = None
        self._listener = ListenerFrame()
        self._profile: dict[str, Any] | None = None

    # --- trajectory ---------------------------------------------------------

    @property
    def trajectory(self) -> TrajectorySpec | None:
        return self._trajectory

    def clear(self, message: str = "No trajectory loaded.") -> None:
        self.path_plot.clear()
        self.itd_plot.clear(message)
        self.ild_plot.clear(message)
        self.summary_label.setText(message)
        self._trajectory = None

    def set_trajectory(
        self,
        trajectory: TrajectorySpec | None,
        *,
        listener: ListenerFrame | None = None,
        duration_s: float = 4.0,
    ) -> None:
        """Show a canonical trajectory and the cues it implies."""

        self._listener = listener or self._listener
        self._trajectory = trajectory
        if trajectory is None:
            self.clear()
            return

        self.path_plot.set_path(
            trajectory.sample_path(self.PLOT_POINTS), head_radius_m=self._listener.head_radius_m
        )

        times = np.linspace(0.0, max(0.01, float(duration_s)), self.PLOT_POINTS)
        positions = trajectory.positions_at(times, listener=self._listener).positions_m
        itd_ms = 1_000.0 * predicted_itd_s(positions, head_radius_m=self._listener.head_radius_m)
        ild = predicted_ild_db(positions, head_radius_m=self._listener.head_radius_m)

        self.itd_plot.set_series([PlotSeries(times, itd_ms, QColor(LEFT_COLOR), "left leads")])
        self.ild_plot.set_series([PlotSeries(times, ild, QColor(RIGHT_COLOR), "left minus right")])

        sample_rate = self.PLOT_POINTS / max(0.01, float(duration_s))
        metrics = trajectory_summary(
            positions, sample_rate, head_radius_m=self._listener.head_radius_m
        )
        self.summary_label.setText(
            "Path {path_length_m:.2f} m  ·  speed {speed_m_s_mean:.2f} m/s (peak "
            "{speed_m_s_max:.2f})  ·  distance {distance_m_min:.2f}-{distance_m_max:.2f} m  ·  "
            "azimuth {azimuth_deg_min:+.0f} to {azimuth_deg_max:+.0f}°  ·  ITD "
            "{itd_s_min:+.4f} to {itd_s_max:+.4f} s  ·  ILD {ild_db_min:+.1f} to "
            "{ild_db_max:+.1f} dB".format(**metrics)
        )

    def show_source_at(self, time_s: float) -> None:
        """Move the source marker to where the trajectory is at ``time_s``."""

        if self._trajectory is None:
            return
        positions = self._trajectory.positions_at(np.array([float(time_s)]), listener=self._listener)
        self.path_plot.set_path(
            self.path_plot.path_m,
            source_m=positions.positions_m[:, 0],
            head_radius_m=self._listener.head_radius_m,
        )

    # --- legacy profiles ----------------------------------------------------

    @property
    def profile(self) -> dict[str, Any] | None:
        return None if self._profile is None else copy.deepcopy(self._profile)

    def legacy_transform(self) -> LegacyPathTransform:
        """The bridge currently declared in the panel."""

        base = profile_transform(self._profile or {})
        return LegacyPathTransform(
            centre_scene=base.centre_scene,
            axis_convention=base.axis_convention,
            scene_units_per_metre=float(self.scene_units_spin.value()),
            normalization_extent=base.normalization_extent,
            elevation_m=base.elevation_m,
        )

    def set_profile(self, profile: Mapping[str, Any] | None) -> None:
        """Load a legacy custom-path profile and show what it means in metres."""

        self._profile = dict(profile) if isinstance(profile, Mapping) else None
        if self._profile is not None:
            declared = profile_transform(self._profile)
            blocked = self.scene_units_spin.blockSignals(True)
            try:
                self.scene_units_spin.setValue(declared.scene_units_per_metre)
            finally:
                self.scene_units_spin.blockSignals(blocked)
        self._refresh_legacy()

    def _refresh_legacy(self) -> None:
        profile = self._profile
        if not profile or profile_points(profile).shape[0] < 2:
            self.legacy_label.setText("No custom path profile in this voice.")
            self.migrate_button.setEnabled(False)
            return

        version = profile_schema_version(profile)
        transform = self.legacy_transform()
        geometry = profile_to_geometry(profile, transform)
        round_trip = profile_round_trip(profile, transform)
        drift = float(np.max(np.abs(round_trip - profile_points(profile)))) if round_trip.size else 0.0

        if geometry is not None:
            self.path_plot.set_path(
                geometry.evaluate(np.linspace(0.0, 1.0, self.PLOT_POINTS)),
                head_radius_m=self._listener.head_radius_m,
            )

        self.migrate_button.setEnabled(version < PROFILE_VERSION)
        self.legacy_label.setText(
            f"Profile version {version}"
            f"{' (legacy, no coordinate metadata)' if version < PROFILE_VERSION else ''}  ·  "
            f"{profile_points(profile).shape[0]} points  ·  "
            f"kind {profile.get('kind', 'linear')}  ·  "
            f"round trip error {drift:.3g} scene units"
        )

    def migrate_profile(self) -> dict[str, Any] | None:
        """Write coordinate metadata into a copy of the profile and publish it."""

        if not self._profile:
            return None
        upgraded = upgrade_profile(self._profile, self.legacy_transform())
        self._profile = upgraded
        self._refresh_legacy()
        self.profileMigrated.emit(copy.deepcopy(upgraded))
        return upgraded

"""Modern visual editor for canonical SAM geometry and traversal.

Unlike the application's legacy pixel-path and segment editors, this dialog
works directly in metres in the canonical listener frame.  It keeps geometry
and traversal independent and writes a versioned specification while also
producing a compatibility profile for existing SAM2 voices.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.trajectory.legacy_paths import canonical_profile_points

_METRES_TO_SCENE = 100.0


class PathNode(QGraphicsEllipseItem):
    """Draggable canonical control point with a stable table row."""

    def __init__(
        self, row: int, position_m: Sequence[float], owner: "CanonicalPathCanvas"
    ):
        super().__init__(-6, -6, 12, 12)
        self.row = row
        self.z_m = float(position_m[2])
        self.owner = owner
        self.setBrush(QBrush(Qt.cyan))
        self.setPen(QPen(Qt.darkCyan, 2))
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPos(owner.metres_to_scene(position_m))
        self.setToolTip(
            f"Control point {row + 1}; drag to move in the horizontal plane"
        )

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene() is not None:
            self.owner.node_moved.emit(
                self.row, self.owner.scene_to_metres(self.pos(), self.z_m)
            )
            self.owner.redraw()
        return result


class CanonicalPathCanvas(QGraphicsView):
    """Interactive top-down path canvas in the canonical listener frame."""

    node_moved = pyqtSignal(int, object)
    point_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        scene = QGraphicsScene(parent)
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumSize(560, 390)
        self.setSceneRect(-280, -190, 560, 380)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._nodes: list[PathNode] = []
        self._closed = False
        self._smooth = False
        self._path = QGraphicsPathItem()
        self._path.setPen(QPen(QBrush(Qt.blue), 3))
        scene.addItem(self._path)
        self._draw_reference()

    @staticmethod
    def metres_to_scene(point_m: Sequence[float]) -> QPointF:
        forward, left = float(point_m[0]), float(point_m[1])
        return QPointF(-left * _METRES_TO_SCENE, -forward * _METRES_TO_SCENE)

    @staticmethod
    def scene_to_metres(point: QPointF, z_m: float = 0.0) -> tuple[float, float, float]:
        return (
            -point.y() / _METRES_TO_SCENE,
            -point.x() / _METRES_TO_SCENE,
            float(z_m),
        )

    def _draw_reference(self):
        scene = self.scene()
        grid_pen = QPen(Qt.lightGray, 1, Qt.DotLine)
        for value in range(-200, 201, 50):
            scene.addLine(value, -175, value, 175, grid_pen).setZValue(-10)
            scene.addLine(-265, value, 265, value, grid_pen).setZValue(-10)
        scene.addLine(0, -180, 0, 180, QPen(Qt.darkGray, 2)).setZValue(-9)
        scene.addLine(-270, 0, 270, 0, QPen(Qt.darkGray, 2)).setZValue(-9)
        listener = scene.addEllipse(
            -16, -12, 32, 24, QPen(Qt.black, 2), QBrush(Qt.lightGray)
        )
        listener.setZValue(-5)
        for text, position in (
            ("FRONT +x", QPointF(8, -180)),
            ("LEFT +y", QPointF(-265, 5)),
        ):
            label = scene.addText(text)
            label.setPos(position)
            label.setDefaultTextColor(Qt.darkGreen)

    def set_path(self, points_m, *, closed=False, smooth=False):
        for node in self._nodes:
            self.scene().removeItem(node)
        self._closed, self._smooth = bool(closed), bool(smooth)
        self._nodes = [PathNode(row, point, self) for row, point in enumerate(points_m)]
        for node in self._nodes:
            self.scene().addItem(node)
        self.redraw()

    def set_shape(self, *, closed: bool, smooth: bool):
        self._closed, self._smooth = closed, smooth
        self.redraw()

    def redraw(self):
        if not self._nodes:
            self._path.setPath(QPainterPath())
            return
        points = [node.pos() for node in self._nodes]
        path = QPainterPath(points[0])
        if self._smooth and len(points) >= 3:
            for index in range(len(points) - 1):
                before = points[max(index - 1, 0)]
                start, end = points[index], points[index + 1]
                after = points[min(index + 2, len(points) - 1)]
                c1 = start + (end - before) / 6.0
                c2 = end - (after - start) / 6.0
                path.cubicTo(c1, c2, end)
        else:
            for point in points[1:]:
                path.lineTo(point)
        if self._closed and len(points) >= 3:
            path.closeSubpath()
        self._path.setPath(path)

    def mouseDoubleClickEvent(self, event):
        self.point_requested.emit(self.scene_to_metres(self.mapToScene(event.pos())))
        event.accept()


class SamPathEditorDialog(QDialog):
    """Visual metre-based geometry and traversal designer."""

    def __init__(self, profile=None, trajectory_spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM Visual Path Designer")
        self.resize(1040, 720)
        self._profile = copy.deepcopy(profile or {})
        self._spec = copy.deepcopy(trajectory_spec or {})
        self._updating = False
        self._points = self._load_points()
        self._build_ui()
        self._populate()

    def _load_points(self):
        canonical = self._spec.get("geometry", {}).get("controlPointsM")
        if isinstance(canonical, list) and len(canonical) >= 2:
            return [list(map(float, point[:3])) for point in canonical]
        if len(self._profile.get("points", [])) >= 2:
            return canonical_profile_points(self._profile).tolist()
        return [[1.0, 1.0, 0.0], [1.5, 0.0, 0.0], [1.0, -1.0, 0.0]]

    def _build_ui(self):
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.primitive_combo = QComboBox()
        self.primitive_combo.addItems(
            [
                "polyline",
                "polygon",
                "spline",
                "bezier",
                "line",
                "arc",
                "circle",
                "ellipse",
                "spiral",
                "helix",
                "lissajous",
                "mathematical",
            ]
        )
        self.primitive_combo.setToolTip(
            "Seeds the path with a shape. Every seed becomes control points you "
            "can drag, so the saved path is whatever is on the canvas rather "
            "than the primitive it started from."
        )
        self.primitive_combo.currentTextChanged.connect(self.seed_primitive)
        self.closed_check = QCheckBox("Closed path")
        self.closed_check.toggled.connect(self._shape_changed)
        self.arc_length_check = QCheckBox("Constant physical speed")
        self.arc_length_check.setChecked(True)
        header.addWidget(QLabel("Geometry:"))
        header.addWidget(self.primitive_combo)
        header.addWidget(self.closed_check)
        header.addWidget(self.arc_length_check)
        header.addStretch(1)
        layout.addLayout(header)

        content = QHBoxLayout()
        self.canvas = CanonicalPathCanvas(self)
        self.canvas.node_moved.connect(self._node_moved)
        self.canvas.point_requested.connect(self.add_point)
        content.addWidget(self.canvas, 2)

        side = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["X forward", "Y left", "Z up"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._table_changed)
        side.addWidget(self.table, 1)
        point_buttons = QHBoxLayout()
        add = QPushButton("Add point")
        add.clicked.connect(lambda: self.add_point((1.0, 0.0, 0.0)))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        point_buttons.addWidget(add)
        point_buttons.addWidget(remove)
        side.addLayout(point_buttons)

        self.expression_widget = QWidget()
        expression_form = QFormLayout(self.expression_widget)
        self.expression_edits = {}
        defaults = {"x": "cos(2*pi*u)", "y": "sin(2*pi*u)", "z": "0"}
        for axis in ("x", "y", "z"):
            editor = QLineEdit(defaults[axis])
            editor.setToolTip(
                "Expression in normalized progress u (0..1). Allowed: +, -, *, /, %, **, "
                "pi, e, sin, cos, tan, sqrt, abs, exp, log, sinh, cosh, tanh."
            )
            editor.editingFinished.connect(self._mathematical_changed)
            expression_form.addRow(f"{axis}(u)", editor)
            self.expression_edits[axis] = editor
        side.addWidget(self.expression_widget)

        traversal = QWidget()
        form = QFormLayout(traversal)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["loop", "ping_pong", "one_shot", "discontinuous"])
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.01, 36000)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" s")
        self.easing_combo = QComboBox()
        self.easing_combo.addItems(["linear", "sine", "smoothstep"])
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Forward", 1)
        self.direction_combo.addItem("Reverse", -1)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 128)
        self.steps_spin.setValue(8)
        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(0, 10)
        self.crossfade_spin.setDecimals(3)
        self.crossfade_spin.setSuffix(" s")
        form.addRow("Traversal", self.mode_combo)
        form.addRow("Cycle duration", self.duration_spin)
        form.addRow("Easing", self.easing_combo)
        form.addRow("Direction", self.direction_combo)
        form.addRow("Jump positions", self.steps_spin)
        form.addRow("Jump crossfade", self.crossfade_spin)
        side.addWidget(traversal)
        content.addLayout(side, 1)
        layout.addLayout(content, 1)

        help_text = QLabel(
            "Drag nodes to reshape the path, double-click the canvas to add a point, or edit exact x/y/z metre values. Geometry and traversal are saved independently."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self):
        geometry = self._spec.get("geometry", {})
        primitive = geometry.get(
            "type", "spline" if self._profile.get("kind") == "spline" else "polyline"
        )
        self.primitive_combo.blockSignals(True)
        self.primitive_combo.setCurrentText(primitive)
        self.primitive_combo.blockSignals(False)
        expressions = geometry.get("expressions", {})
        for axis, editor in self.expression_edits.items():
            if axis in expressions:
                editor.setText(str(expressions[axis]))
        self.closed_check.setChecked(
            bool(geometry.get("closed", self._profile.get("closedLoop", False)))
        )
        self.arc_length_check.setChecked(bool(self._spec.get("arcLength", True)))
        traversal = self._spec.get("traversal", {})
        self.mode_combo.setCurrentText(traversal.get("mode", "loop"))
        self.duration_spin.setValue(float(traversal.get("durationS", 5.0)))
        self.easing_combo.setCurrentText(traversal.get("easing", "linear"))
        self.direction_combo.setCurrentIndex(
            0 if int(traversal.get("direction", 1)) == 1 else 1
        )
        self.steps_spin.setValue(int(traversal.get("steps", 8)))
        self.crossfade_spin.setValue(float(traversal.get("crossfadeS", 0.0)))
        self._sync_views()

    def _sync_views(self):
        self.expression_widget.setVisible(
            self.primitive_combo.currentText() == "mathematical"
        )
        if self.primitive_combo.currentText() == "mathematical":
            self._sample_mathematical_path()
        self._updating = True
        self.table.setRowCount(len(self._points))
        for row, point in enumerate(self._points):
            for column, value in enumerate(point):
                self.table.setItem(row, column, QTableWidgetItem(f"{value:.4f}"))
        self._updating = False
        self.canvas.set_path(
            self._points,
            closed=self.closed_check.isChecked(),
            smooth=self.primitive_combo.currentText() in ("spline", "bezier"),
        )

    def _shape_changed(self):
        self.canvas.set_shape(
            closed=self.closed_check.isChecked(),
            smooth=self.primitive_combo.currentText() in ("spline", "bezier"),
        )

    def _node_moved(self, row, point):
        if row >= len(self._points):
            return
        self._points[row][:2] = point[:2]
        self._updating = True
        for column in (0, 1):
            self.table.item(row, column).setText(f"{point[column]:.4f}")
        self._updating = False

    def _table_changed(self, item):
        if self._updating:
            return
        try:
            self._points[item.row()][item.column()] = float(item.text())
        except ValueError:
            return
        self._sync_views()

    def add_point(self, point):
        self._points.append(list(map(float, point)))
        self._sync_views()
        self.table.selectRow(len(self._points) - 1)

    def remove_selected(self):
        rows = sorted(
            {index.row() for index in self.table.selectedIndexes()}, reverse=True
        )
        if len(self._points) - len(rows) < 2:
            return
        for row in rows:
            self._points.pop(row)
        self._sync_views()

    def seed_primitive(self, primitive):
        if primitive == "line":
            self._points = [[1, 1, 0], [1, -1, 0]]
        elif primitive == "arc":
            self._points = [
                [1.25 * math.cos(t), 1.25 * math.sin(t), 0]
                for t in [math.radians(-45.0 + 90.0 * i / 8) for i in range(9)]
            ]
            self.closed_check.setChecked(False)
        elif primitive == "polygon":
            self._points = [
                [1.2 * math.cos(t), 1.2 * math.sin(t), 0]
                for t in [2 * math.pi * i / 6 for i in range(6)]
            ]
            self.closed_check.setChecked(True)
        elif primitive in ("circle", "ellipse"):
            ry = 1.0 if primitive == "circle" else 0.6
            self._points = [
                [1.25 * math.cos(t), ry * math.sin(t), 0]
                for t in [2 * math.pi * i / 12 for i in range(12)]
            ]
            self.closed_check.setChecked(True)
        elif primitive == "spiral":
            self._points = [
                [(0.15 + 0.07 * i) * math.cos(i), (0.15 + 0.07 * i) * math.sin(i), 0]
                for i in range(24)
            ]
            self.closed_check.setChecked(False)
        elif primitive == "helix":
            self._points = [
                [math.cos(t), math.sin(t), t / (2 * math.pi) - 0.5]
                for t in [4 * math.pi * i / 31 for i in range(32)]
            ]
            self.closed_check.setChecked(False)
        elif primitive == "lissajous":
            self._points = [
                [math.cos(3 * t), math.sin(2 * t), 0.25 * math.sin(t)]
                for t in [2 * math.pi * i / 47 for i in range(48)]
            ]
            self.closed_check.setChecked(True)
        elif primitive == "mathematical":
            self.closed_check.setChecked(True)
            self._sample_mathematical_path()
        elif primitive == "spline":
            self._points = [[1, 1, 0], [1.7, 0.4, 0.2], [0.8, -0.5, 0.1], [1.2, -1, 0]]
        else:
            self._points = [[1, 1, 0], [1.5, 0, 0], [1, -1, 0]]
        self._sync_views()

    def _mathematical_changed(self):
        if self.primitive_combo.currentText() == "mathematical":
            self._sample_mathematical_path()
            self._sync_views()

    def _sample_mathematical_path(self):
        """Sample expressions through the Qt-free, restricted core evaluator."""
        from src.audio.sam_workbench.trajectory import Mathematical

        geometry = Mathematical(
            *(self.expression_edits[a].text() for a in ("x", "y", "z"))
        )
        try:
            self._points = geometry.evaluate(np.linspace(0.0, 1.0, 65)).tolist()
            for editor in self.expression_edits.values():
                editor.setStyleSheet("")
        except ValueError as error:
            for editor in self.expression_edits.values():
                editor.setStyleSheet("border: 1px solid #c33;")
                editor.setToolTip(str(error))

    def trajectory_spec(self):
        geometry = {
            "type": self.primitive_combo.currentText(),
            "controlPointsM": copy.deepcopy(self._points),
            "closed": self.closed_check.isChecked(),
        }
        if self.primitive_combo.currentText() == "mathematical":
            geometry["expressions"] = {
                axis: editor.text() for axis, editor in self.expression_edits.items()
            }
        return {
            "schemaVersion": 1,
            "coordinateFrame": "listener",
            "geometry": geometry,
            "traversal": {
                "mode": self.mode_combo.currentText(),
                "durationS": self.duration_spin.value(),
                "easing": self.easing_combo.currentText(),
                "direction": self.direction_combo.currentData(),
                "steps": self.steps_spin.value(),
                "crossfadeS": self.crossfade_spin.value(),
            },
            "arcLength": self.arc_length_check.isChecked(),
        }

    def compatibility_profile(self):
        profile = copy.deepcopy(self._profile)
        scene_points = [
            [-point[1] * _METRES_TO_SCENE, -point[0] * _METRES_TO_SCENE]
            for point in self._points
        ]
        profile.update(
            {
                "schemaVersion": 2,
                "coordinateSpace": "normalized_listener_2d",
                "sceneUnitsPerMetre": _METRES_TO_SCENE,
                "axisConvention": "x_right_y_down",
                "sceneCentre": [0.0, 0.0],
                "kind": (
                    "spline"
                    if self.primitive_combo.currentText() in ("spline", "bezier")
                    else "linear"
                ),
                "closedLoop": self.closed_check.isChecked(),
                "points": scene_points,
            }
        )
        return profile

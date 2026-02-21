import math
from typing import List, Tuple

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class _DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, x: float, y: float, radius: float = 5.0):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setBrush(Qt.darkCyan)
        self.setPen(QPen(Qt.black, 1.0))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setPos(x, y)


class _PathView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setMinimumSize(440, 320)
        self.setFocusPolicy(Qt.StrongFocus)
        self.scene().setSceneRect(-210, -140, 420, 280)

        self.ear_a = QPointF(-65.0, 0.0)
        self.ear_b = QPointF(65.0, 0.0)
        self._points: List[_DraggablePoint] = []
        self._path_kind = "linear"
        self._closed_loop = False
        self._close_snap_px = 12.0
        self._path_item = QGraphicsPathItem()
        self._path_item.setPen(QPen(Qt.blue, 2.0))
        self.scene().addItem(self._path_item)
        self._x_axis_item = self.scene().addText("")
        self._y_axis_item = self.scene().addText("")
        self._x_axis_item.setDefaultTextColor(Qt.darkGreen)
        self._y_axis_item.setDefaultTextColor(Qt.darkGreen)

        self._draw_head_guides()

    def _draw_head_guides(self) -> None:
        head = self.scene().addEllipse(-90, -60, 180, 120, QPen(Qt.darkGray, 1.0))
        head.setZValue(-5)
        x_axis = self.scene().addLine(-200, 0, 200, 0, QPen(Qt.lightGray, 1.0, Qt.DashLine))
        y_axis = self.scene().addLine(0, -130, 0, 130, QPen(Qt.lightGray, 1.0, Qt.DashLine))
        x_axis.setZValue(-6)
        y_axis.setZValue(-6)

        for label, pt in (("A", self.ear_a), ("B", self.ear_b)):
            marker = self.scene().addEllipse(pt.x() - 8, pt.y() - 8, 16, 16, QPen(Qt.black, 1.0))
            marker.setBrush(Qt.lightGray)
            marker.setZValue(5)
            text = self.scene().addText(label)
            text.setPos(pt.x() - 5, pt.y() - 14)
            text.setZValue(6)

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        for p in self._points:
            self.scene().removeItem(p)
        self._points = []
        for x, y in points:
            point_item = _DraggablePoint(x, y)
            self.scene().addItem(point_item)
            self._points.append(point_item)
        self._redraw_path()

    def set_path_kind(self, kind: str) -> None:
        self._path_kind = str(kind or "linear").lower()
        self._redraw_path()

    def set_closed_loop(self, closed: bool) -> None:
        self._closed_loop = bool(closed)
        self._redraw_path()

    def is_closed_loop(self) -> bool:
        return self._closed_loop

    def get_points(self) -> List[Tuple[float, float]]:
        return [(float(p.scenePos().x()), float(p.scenePos().y())) for p in self._points]

    def mouseDoubleClickEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        pt = _DraggablePoint(scene_pos.x(), scene_pos.y())
        self.scene().addItem(pt)
        self._points.append(pt)
        self._redraw_path()
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        self._snap_endpoints_if_needed()
        self._redraw_path()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = [p for p in self._points if p.isSelected()]
            for p in selected:
                self.scene().removeItem(p)
                self._points.remove(p)
            if selected:
                self._closed_loop = False
                self._redraw_path()
                event.accept()
                return
        super().keyPressEvent(event)

    def _snap_endpoints_if_needed(self) -> None:
        if len(self._points) < 3:
            self._closed_loop = False
            return
        first = self._points[0]
        last = self._points[-1]
        if math.hypot(first.scenePos().x() - last.scenePos().x(), first.scenePos().y() - last.scenePos().y()) <= self._close_snap_px:
            last.setPos(first.scenePos())
            self._closed_loop = True
        else:
            self._closed_loop = False

    def _redraw_path(self) -> None:
        if not self._points:
            self._path_item.setPath(QPainterPath())
            self._x_axis_item.setPlainText("")
            self._y_axis_item.setPlainText("")
            return
        ordered = self.get_points()
        path = QPainterPath(QPointF(ordered[0][0], ordered[0][1]))
        if self._path_kind == "spline" and len(ordered) >= 3:
            for i in range(len(ordered) - 1):
                p0 = ordered[i - 1] if i > 0 else ordered[0]
                p1 = ordered[i]
                p2 = ordered[i + 1]
                p3 = ordered[i + 2] if (i + 2) < len(ordered) else ordered[-1]
                c1x = p1[0] + (p2[0] - p0[0]) / 6.0
                c1y = p1[1] + (p2[1] - p0[1]) / 6.0
                c2x = p2[0] - (p3[0] - p1[0]) / 6.0
                c2y = p2[1] - (p3[1] - p1[1]) / 6.0
                path.cubicTo(c1x, c1y, c2x, c2y, p2[0], p2[1])
        else:
            for x, y in ordered[1:]:
                path.lineTo(x, y)
        if self._closed_loop and len(ordered) >= 3:
            path.closeSubpath()
        self._path_item.setPath(path)
        xs = [pt[0] for pt in ordered]
        ys = [pt[1] for pt in ordered]
        dx = max(xs) - min(xs) if len(xs) > 1 else 0.0
        dy = max(ys) - min(ys) if len(ys) > 1 else 0.0
        self._x_axis_item.setPlainText(f"ΔX distance: {dx:.1f}")
        self._y_axis_item.setPlainText(f"ΔY distance: {dy:.1f}")
        self._x_axis_item.setPos(-205, 118)
        self._y_axis_item.setPos(-205, 98)


class CustomPathCreatorDialog(QDialog):
    def __init__(self, parent=None, profile=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Path Creator")
        self.profile = profile if isinstance(profile, dict) else {}

        main_layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Profile Type:"))
        self.path_kind_combo = QComboBox()
        self.path_kind_combo.addItems(["linear", "circular", "ovoid", "spline"])
        top_row.addWidget(self.path_kind_combo)

        self.seed_btn = QPushButton("Seed Shape")
        self.seed_btn.clicked.connect(self._seed_shape)
        top_row.addWidget(self.seed_btn)
        top_row.addStretch(1)
        main_layout.addLayout(top_row)

        self.view = _PathView(self)
        self.path_kind_combo.currentTextChanged.connect(self.view.set_path_kind)
        main_layout.addWidget(self.view)

        help_label = QLabel("Double-click to add points. Drag points to refine. Press Delete to remove selected points. Drag one endpoint onto the other to close a loop.")
        help_label.setWordWrap(True)
        main_layout.addWidget(help_label)

        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        main_layout.addLayout(buttons)

        self._load_profile()

    def _load_profile(self) -> None:
        kind = str(self.profile.get("kind", "linear"))
        idx = self.path_kind_combo.findText(kind)
        self.path_kind_combo.setCurrentIndex(idx if idx >= 0 else 0)
        points = self.profile.get("points")
        self.view.set_path_kind(kind)
        self.view.set_closed_loop(bool(self.profile.get("closedLoop", False)))
        if isinstance(points, list) and points:
            clean = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    clean.append((float(p[0]), float(p[1])))
            if clean:
                self.view.set_points(clean)
                return
        self._seed_shape()

    def _seed_shape(self) -> None:
        kind = self.path_kind_combo.currentText()
        self.view.set_path_kind(kind)
        self.view.set_closed_loop(kind in ("circular", "ovoid"))
        if kind == "linear":
            pts = [(-120.0, -20.0), (0.0, 0.0), (120.0, 20.0)]
        elif kind == "circular":
            pts = [(95.0 * math.cos(t), 95.0 * math.sin(t)) for t in [i * (math.pi / 6.0) for i in range(12)]]
        elif kind == "ovoid":
            pts = [(130.0 * math.cos(t), 75.0 * math.sin(t)) for t in [i * (math.pi / 6.0) for i in range(12)]]
        else:
            pts = [(-130.0, 20.0), (-70.0, -65.0), (-10.0, 40.0), (40.0, -35.0), (120.0, 50.0)]
        self.view.set_points(pts)

    def get_profile(self) -> dict:
        return {
            "kind": self.path_kind_combo.currentText(),
            "closedLoop": self.view.is_closed_loop(),
            "points": [[x, y] for x, y in self.view.get_points()],
        }

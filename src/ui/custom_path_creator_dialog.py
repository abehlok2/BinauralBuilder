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
        self.scene().setSceneRect(-210, -140, 420, 280)

        self.ear_a = QPointF(-65.0, 0.0)
        self.ear_b = QPointF(65.0, 0.0)
        self._points: List[_DraggablePoint] = []
        self._path_item = QGraphicsPathItem()
        self._path_item.setPen(QPen(Qt.blue, 2.0))
        self.scene().addItem(self._path_item)

        self._draw_head_guides()

    def _draw_head_guides(self) -> None:
        head = self.scene().addEllipse(-90, -60, 180, 120, QPen(Qt.darkGray, 1.0))
        head.setZValue(-5)

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
        self._redraw_path()
        super().mouseReleaseEvent(event)

    def _redraw_path(self) -> None:
        if not self._points:
            self._path_item.setPath(QPainterPath())
            return
        ordered = self.get_points()
        path = QPainterPath(QPointF(ordered[0][0], ordered[0][1]))
        for x, y in ordered[1:]:
            path.lineTo(x, y)
        self._path_item.setPath(path)


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
        main_layout.addWidget(self.view)

        help_label = QLabel("Double-click to add points. Drag points to refine. Use 'Seed Shape' for presets.")
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
            "points": [[x, y] for x, y in self.view.get_points()],
        }

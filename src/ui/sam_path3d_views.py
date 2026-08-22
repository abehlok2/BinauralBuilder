"""Linked views for three-dimensional path editing.

A single perspective view cannot be edited precisely, because depth is
ambiguous: a point dragged across the screen has moved along a ray, and which
point on that ray it landed on is a guess.  The fix is not a better perspective
view but more views - three orthographic ones, each of which fixes exactly the
two axes it shows.

* top    - forward/left, the plan the listener is standing in;
* front  - left/up, what the listener sees;
* side   - forward/up, the median plane.

Between them every axis is directly editable in at least two places, and a
point selected in any view is selected in all of them.  The perspective view
stays, because it is the only one that shows what the path actually looks like;
it just is not where precise work happens.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

__all__ = ["PLANES", "OrthographicPathView", "PerspectivePathView"]

#: Which two canonical axes each orthographic view edits, and how they map to
#: screen. ``(horizontal_axis, sign, vertical_axis, sign, title, labels)``.
#:
#: The signs put the listener's own frame on screen the way a person expects to
#: read it: in the top view the listener's left is on the *left* of the screen,
#: which means +y runs leftward, and forward runs up the screen.
PLANES: dict[str, dict] = {
    "top": {
        "horizontal": (1, -1.0),
        "vertical": (0, +1.0),
        "title": "Top — forward / left",
        "right_label": "right −y",
        "up_label": "front +x",
        "locked": 2,
    },
    "front": {
        "horizontal": (1, -1.0),
        "vertical": (2, +1.0),
        "title": "Front — left / height",
        "right_label": "right −y",
        "up_label": "up +z",
        "locked": 0,
    },
    "side": {
        "horizontal": (0, +1.0),
        "vertical": (2, +1.0),
        "title": "Side — forward / height",
        "right_label": "front +x",
        "up_label": "up +z",
        "locked": 1,
    },
}

_GRID = QColor(70, 74, 82)
_AXIS = QColor(110, 116, 126)
_PATH = QColor(80, 190, 230)
_NODE = QColor(120, 210, 250)
_SELECTED = QColor(255, 190, 60)
_LISTENER = QColor(190, 195, 205)
_NOSE = QColor(120, 220, 140)
_SHELL = QColor(140, 120, 200)
_MARKER = QColor(255, 120, 120)
_TEXT = QColor(190, 195, 205)


class _PathViewBase(QWidget):
    """Shared state, scaling, and listener drawing for every view."""

    #: A point was clicked. Carries its index, or -1 for "nothing".
    selectionChanged = pyqtSignal(int)
    #: A point was dragged. Carries its index and the new ``(x, y, z)`` metres.
    pointMoved = pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(230, 190)
        self.setMouseTracking(True)
        self._points = np.zeros((0, 3))
        self._curve = np.zeros((0, 3))
        self._selected = -1
        self._marker = None
        self._extent_m = 2.0
        self._show_shell = False
        self._shell_radius_m = 1.5
        self._editable = True

    # --- state ------------------------------------------------------------

    def set_path(self, control_points, curve=None):
        """Set the editable control points and the curve they produce."""

        self._points = np.asarray(control_points, dtype=float).reshape(-1, 3)
        self._curve = (
            np.asarray(curve, dtype=float).reshape(-1, 3)
            if curve is not None and len(curve)
            else self._points
        )
        self._rescale()
        self.update()

    def set_selected(self, index):
        self._selected = int(index)
        self.update()

    def set_marker(self, position):
        """The live preview position, or ``None`` to hide it."""

        self._marker = None if position is None else np.asarray(position, dtype=float)
        self.update()

    def set_coverage_shell(self, visible, radius_m=1.5):
        self._show_shell = bool(visible)
        self._shell_radius_m = max(float(radius_m), 0.05)
        self.update()

    def set_editable(self, editable):
        self._editable = bool(editable)

    def _rescale(self):
        source = self._curve if len(self._curve) else self._points
        extent = 1.0
        if len(source):
            extent = float(np.max(np.abs(source)))
        if self._show_shell:
            extent = max(extent, self._shell_radius_m)
        self._extent_m = max(extent * 1.15, 0.5)

    def _radius(self):
        return max(min(self.width(), self.height()) * 0.5 - 22.0, 10.0)

    def _scale(self):
        return self._radius() / self._extent_m

    def _centre(self):
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    # --- painting helpers ---------------------------------------------------

    def _draw_frame(self, painter, title):
        painter.fillRect(self.rect(), QColor(28, 30, 34))
        painter.setPen(QPen(_TEXT))
        painter.drawText(8, 15, title)

    def _draw_grid(self, painter):
        """A one-metre grid, so distances can be read off directly."""

        centre = self._centre()
        scale = self._scale()
        radius = self._radius()
        painter.setPen(QPen(_GRID, 1, Qt.DotLine))
        step = 1.0 if self._extent_m <= 6.0 else 5.0
        distance = step
        while distance * scale <= radius:
            offset = distance * scale
            for sign in (-1.0, 1.0):
                painter.drawLine(
                    QPointF(centre.x() + sign * offset, centre.y() - radius),
                    QPointF(centre.x() + sign * offset, centre.y() + radius),
                )
                painter.drawLine(
                    QPointF(centre.x() - radius, centre.y() + sign * offset),
                    QPointF(centre.x() + radius, centre.y() + sign * offset),
                )
            distance += step
        painter.setPen(QPen(_AXIS, 1))
        painter.drawLine(
            QPointF(centre.x() - radius, centre.y()),
            QPointF(centre.x() + radius, centre.y()),
        )
        painter.drawLine(
            QPointF(centre.x(), centre.y() - radius),
            QPointF(centre.x(), centre.y() + radius),
        )

    def _draw_polyline(self, painter, projected, colour, width=2):
        if len(projected) < 2:
            return
        path = QPainterPath(projected[0])
        for point in projected[1:]:
            path.lineTo(point)
        painter.setPen(QPen(colour, width))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _draw_nodes(self, painter, projected):
        for index, point in enumerate(projected):
            chosen = index == self._selected
            painter.setBrush(QBrush(_SELECTED if chosen else _NODE))
            painter.setPen(QPen(Qt.black, 1))
            size = 6.0 if chosen else 4.0
            painter.drawEllipse(point, size, size)

    def _draw_marker(self, painter, point):
        painter.setBrush(QBrush(_MARKER))
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(point, 5.0, 5.0)

    # --- interaction --------------------------------------------------------

    def _hit(self, position):
        projected = self._project(self._points)
        for index, point in enumerate(projected):
            if (point - position).manhattanLength() <= 12:
                return index
        return -1

    def mousePressEvent(self, event):
        index = self._hit(QPointF(event.pos()))
        self._selected = index
        self.selectionChanged.emit(index)
        self.update()
        event.accept()

    def _project(self, points) -> list[QPointF]:  # pragma: no cover - overridden
        raise NotImplementedError


class OrthographicPathView(_PathViewBase):
    """One editable plane of the canonical frame."""

    def __init__(self, plane: str, parent=None):
        super().__init__(parent)
        if plane not in PLANES:
            raise ValueError(f"unknown plane {plane!r}; expected one of {tuple(PLANES)}")
        self.plane = plane
        self._config = PLANES[plane]
        self._dragging = -1

    # --- projection ---------------------------------------------------------

    def _project(self, points) -> list[QPointF]:
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        h_axis, h_sign = self._config["horizontal"]
        v_axis, v_sign = self._config["vertical"]
        centre, scale = self._centre(), self._scale()
        return [
            QPointF(
                centre.x() + h_sign * float(point[h_axis]) * scale,
                centre.y() - v_sign * float(point[v_axis]) * scale,
            )
            for point in values
        ]

    def _unproject(self, position: QPointF, original) -> np.ndarray:
        """Screen position back to metres, leaving the third axis untouched.

        Leaving the locked axis alone is the whole reason for having three
        views: a drag here says nothing about the axis this view does not
        show, so it must not silently reset it.
        """

        h_axis, h_sign = self._config["horizontal"]
        v_axis, v_sign = self._config["vertical"]
        centre, scale = self._centre(), self._scale()
        result = np.array(original, dtype=float)
        result[h_axis] = h_sign * (position.x() - centre.x()) / scale
        result[v_axis] = v_sign * (centre.y() - position.y()) / scale
        return result

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_frame(painter, self._config["title"])
        self._draw_grid(painter)
        self._draw_shell(painter)
        self._draw_listener(painter)
        self._draw_polyline(painter, self._project(self._curve), _PATH)
        self._draw_nodes(painter, self._project(self._points))
        if self._marker is not None:
            self._draw_marker(painter, self._project([self._marker])[0])
        self._draw_labels(painter)

    def _draw_labels(self, painter):
        painter.setPen(QPen(_TEXT.darker(130)))
        painter.drawText(self.width() - 62, self.height() // 2 - 4, self._config["right_label"])
        painter.drawText(self.width() // 2 + 6, 30, self._config["up_label"])
        painter.drawText(8, self.height() - 8, f"{self._extent_m:.1f} m")

    def _draw_shell(self, painter):
        """The HRTF measurement sphere, as a circle in every plane."""

        if not self._show_shell:
            return
        painter.setPen(QPen(_SHELL, 1, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        offset = self._shell_radius_m * self._scale()
        painter.drawEllipse(self._centre(), offset, offset)

    def _draw_listener(self, painter):
        """A head that can be told apart from a dot: nose, ears, axes.

        Which features are visible depends on the plane, because that is what
        tells the three views apart at a glance.
        """

        centre, scale = self._centre(), self._scale()
        head = max(0.09 * scale, 7.0)
        painter.setPen(QPen(_LISTENER, 1))
        painter.setBrush(QBrush(QColor(58, 62, 70)))

        if self.plane == "top":
            painter.drawEllipse(centre, head, head)
            # Nose forward, so front and back are never in doubt.
            painter.setBrush(QBrush(_NOSE))
            painter.setPen(QPen(_NOSE, 1))
            painter.drawPolygon(
                QPolygonF([
                    QPointF(centre.x(), centre.y() - head - 6.0),
                    QPointF(centre.x() - 4.0, centre.y() - head + 2.0),
                    QPointF(centre.x() + 4.0, centre.y() - head + 2.0),
                ])
            )
            self._draw_ears(painter, centre, head, horizontal=True)
        elif self.plane == "front":
            painter.drawEllipse(centre, head, head * 1.15)
            self._draw_ears(painter, centre, head, horizontal=True)
            # Head-height plane: everything on this line is at ear level.
            painter.setPen(QPen(_LISTENER.darker(160), 1, Qt.DashLine))
            painter.drawLine(
                QPointF(centre.x() - self._radius(), centre.y()),
                QPointF(centre.x() + self._radius(), centre.y()),
            )
        else:  # side
            painter.drawEllipse(centre, head, head * 1.15)
            painter.setBrush(QBrush(_NOSE))
            painter.setPen(QPen(_NOSE, 1))
            painter.drawPolygon(
                QPolygonF([
                    QPointF(centre.x() + head + 6.0, centre.y()),
                    QPointF(centre.x() + head - 2.0, centre.y() - 4.0),
                    QPointF(centre.x() + head - 2.0, centre.y() + 4.0),
                ])
            )
            # The vertical axis, which is what this view exists to show.
            painter.setPen(QPen(_LISTENER.darker(160), 1, Qt.DashLine))
            painter.drawLine(
                QPointF(centre.x(), centre.y() - self._radius()),
                QPointF(centre.x(), centre.y() + self._radius()),
            )

    def _draw_ears(self, painter, centre, head, horizontal):
        painter.setPen(QPen(_LISTENER, 2))
        painter.setBrush(Qt.NoBrush)
        for sign in (-1.0, 1.0):
            if horizontal:
                painter.drawLine(
                    QPointF(centre.x() + sign * head, centre.y() - 2.0),
                    QPointF(centre.x() + sign * (head + 4.0), centre.y() + 2.0),
                )

    # --- interaction --------------------------------------------------------

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self._editable and self._selected >= 0 and event.button() == Qt.LeftButton:
            self._dragging = self._selected

    def mouseMoveEvent(self, event):
        if self._dragging < 0 or self._dragging >= len(self._points):
            return
        moved = self._unproject(QPointF(event.pos()), self._points[self._dragging])
        self._points[self._dragging] = moved
        self.pointMoved.emit(self._dragging, tuple(float(value) for value in moved))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = -1
        event.accept()


class PerspectivePathView(_PathViewBase):
    """An orbiting three-dimensional view, for reading the path rather than editing it.

    Points are selectable here so a shape spotted in perspective can be worked
    on in the orthographic views, but they are not draggable: a drag would be a
    guess about depth, and guessing is exactly what the other three views exist
    to avoid.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._yaw_deg = 35.0
        self._pitch_deg = 22.0
        self._last_drag = None
        self.set_editable(False)
        self.setToolTip("Drag to orbit. Edit positions in the top, front and side views.")

    def set_orbit(self, yaw_deg, pitch_deg):
        self._yaw_deg = float(yaw_deg) % 360.0
        self._pitch_deg = max(-89.0, min(89.0, float(pitch_deg)))
        self.update()

    def _project(self, points) -> list[QPointF]:
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(values):
            return []
        yaw = math.radians(self._yaw_deg)
        pitch = math.radians(self._pitch_deg)
        # Camera basis: right and up vectors of an orbiting eye. An orthographic
        # projection onto them keeps parallel lines parallel, so the grid stays
        # readable at every orbit angle.
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        forward = np.array(
            [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)]
        )
        up = np.cross(right, forward)
        centre, scale = self._centre(), self._scale() * 0.9
        horizontal = values @ right
        vertical = values @ up
        return [
            QPointF(centre.x() + float(h) * scale, centre.y() - float(v) * scale)
            for h, v in zip(horizontal, vertical)
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_frame(painter, "Perspective — drag to orbit")
        self._draw_ground(painter)
        self._draw_shell(painter)
        self._draw_listener(painter)
        self._draw_polyline(painter, self._project(self._curve), _PATH)
        self._draw_nodes(painter, self._project(self._points))
        if self._marker is not None:
            self._draw_marker(painter, self._project([self._marker])[0])
        painter.setPen(QPen(_TEXT.darker(130)))
        painter.drawText(8, self.height() - 8, f"yaw {self._yaw_deg:.0f}°  pitch {self._pitch_deg:.0f}°")

    def _draw_ground(self, painter):
        """The head-height plane, drawn as a grid so height reads as height."""

        painter.setPen(QPen(_GRID, 1))
        span = max(1, int(math.ceil(self._extent_m)))
        for offset in range(-span, span + 1):
            for line in (
                [(offset, -span, 0.0), (offset, span, 0.0)],
                [(-span, offset, 0.0), (span, offset, 0.0)],
            ):
                projected = self._project(line)
                painter.drawLine(projected[0], projected[1])

    def _draw_shell(self, painter):
        if not self._show_shell:
            return
        painter.setPen(QPen(_SHELL, 1, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        angle = np.linspace(0.0, 2.0 * np.pi, 48)
        radius = self._shell_radius_m
        for ring in (
            np.stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)), axis=-1),
            np.stack((radius * np.cos(angle), np.zeros_like(angle), radius * np.sin(angle)), axis=-1),
            np.stack((np.zeros_like(angle), radius * np.cos(angle), radius * np.sin(angle)), axis=-1),
        ):
            self._draw_polyline(painter, self._project(ring), _SHELL, 1)

    def _draw_listener(self, painter):
        centre = self._project([(0.0, 0.0, 0.0)])[0]
        head = max(0.09 * self._scale(), 6.0)
        painter.setPen(QPen(_LISTENER, 1))
        painter.setBrush(QBrush(QColor(58, 62, 70)))
        painter.drawEllipse(centre, head, head)
        # Nose, ear axis and vertical axis, drawn in the projection so they
        # turn with the orbit and keep saying which way the listener faces.
        for direction, colour in (
            ((0.45, 0.0, 0.0), _NOSE),
            ((0.0, 0.22, 0.0), _LISTENER),
            ((0.0, -0.22, 0.0), _LISTENER),
            ((0.0, 0.0, 0.45), _AXIS),
        ):
            end = self._project([direction])[0]
            painter.setPen(QPen(colour, 2))
            painter.drawLine(centre, end)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._last_drag = event.pos()

    def mouseMoveEvent(self, event):
        if self._last_drag is None or not (event.buttons() & Qt.LeftButton):
            return
        delta = event.pos() - self._last_drag
        self._last_drag = event.pos()
        self.set_orbit(self._yaw_deg + delta.x() * 0.5, self._pitch_deg + delta.y() * 0.5)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._last_drag = None
        event.accept()

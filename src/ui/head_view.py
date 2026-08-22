"""A head seen from above and behind, with a direction you can click.

Typing an azimuth into a spin box is precise and tells you nothing. When the
question is "which way is this source", a picture of a head with a marker on it
answers instantly, and clicking the picture is how people actually want to
choose a direction.

Two views, because one is not enough: from above, azimuth is an angle around
the head and reads directly; elevation does not appear at all. So the top view
carries azimuth and a side view carries elevation, and the two together cover
the sphere without pretending a flat picture is a globe.

The canonical frame is +x forward, +y left, +z up, and the drawing follows it:
straight up the widget is in front of the listener, and the left of the widget
is the listener's left. That means a source drawn on the left of the screen is
heard on the left, which is the only arrangement that does not require constant
mental translation.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

__all__ = ["HeadDirectionView", "HeadDirectionSelector"]

_MARGIN = 14.0
_HEAD_FRACTION = 0.30


class HeadDirectionView(QWidget):
    """One projection of the head, with a draggable direction marker.

    ``plane="top"`` shows azimuth around the head; ``plane="side"`` shows
    elevation. Both report through :attr:`directionChanged`.
    """

    directionChanged = pyqtSignal(float, float)  # azimuth, elevation

    def __init__(self, plane: str = "top", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if plane not in ("top", "side"):
            raise ValueError(f"plane must be 'top' or 'side', got {plane!r}")
        self._plane = plane
        self._azimuth = 0.0
        self._elevation = 0.0
        self._enabled_marks: list[tuple[float, float]] = []
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(
            "Click or drag to place the source. Straight up is in front of the "
            "listener; the left of the view is the listener's left ear."
            if plane == "top"
            else "Click or drag to set elevation. Up is above the listener."
        )

    # --- state --------------------------------------------------------------

    @property
    def plane(self) -> str:
        return self._plane

    @property
    def azimuth_deg(self) -> float:
        return self._azimuth

    @property
    def elevation_deg(self) -> float:
        return self._elevation

    def set_direction(self, azimuth_deg: float, elevation_deg: float, *, emit: bool = False) -> None:
        self._azimuth = ((float(azimuth_deg) + 180.0) % 360.0) - 180.0
        self._elevation = max(-90.0, min(90.0, float(elevation_deg)))
        self.update()
        if emit:
            self.directionChanged.emit(self._azimuth, self._elevation)

    def set_marks(self, directions) -> None:
        """Faint markers for the directions a dataset actually measured."""

        self._enabled_marks = [(float(a), float(e)) for a, e in directions]
        self.update()

    # --- geometry -----------------------------------------------------------

    def _centre_radius(self) -> tuple[QPointF, float]:
        rect = QRectF(
            _MARGIN, _MARGIN, max(1.0, self.width() - 2 * _MARGIN), max(1.0, self.height() - 2 * _MARGIN)
        )
        radius = 0.5 * min(rect.width(), rect.height())
        return QPointF(rect.center()), radius

    def _to_point(self, azimuth_deg: float, elevation_deg: float) -> QPointF:
        centre, radius = self._centre_radius()
        if self._plane == "top":
            # Azimuth zero points up the widget; positive azimuth is the
            # listener's left, which is the left of the screen.
            angle = math.radians(azimuth_deg)
            # Elevated sources sit nearer the centre, as they would if you were
            # looking down on them.
            shrink = math.cos(math.radians(elevation_deg))
            return QPointF(
                centre.x() - radius * shrink * math.sin(angle),
                centre.y() - radius * shrink * math.cos(angle),
            )
        # Side view: forward to the right, up is up.
        angle = math.radians(elevation_deg)
        facing = 1.0 if abs(azimuth_deg) <= 90.0 else -1.0
        return QPointF(
            centre.x() + radius * math.cos(angle) * facing,
            centre.y() - radius * math.sin(angle),
        )

    def _from_point(self, point: QPointF) -> tuple[float, float]:
        centre, radius = self._centre_radius()
        dx = point.x() - centre.x()
        dy = point.y() - centre.y()
        if self._plane == "top":
            azimuth = math.degrees(math.atan2(-dx, -dy))
            return azimuth, self._elevation
        elevation = math.degrees(math.atan2(-dy, abs(dx) if abs(dx) > 1e-6 else 1e-6))
        elevation = max(-90.0, min(90.0, elevation))
        azimuth = self._azimuth
        if dx < 0 and abs(self._azimuth) <= 90.0:
            azimuth = 180.0 - self._azimuth
        elif dx >= 0 and abs(self._azimuth) > 90.0:
            azimuth = 180.0 - self._azimuth
        return azimuth, elevation

    # --- interaction --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._apply(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.buttons() & Qt.LeftButton:
            self._apply(event)

    def _apply(self, event) -> None:
        azimuth, elevation = self._from_point(QPointF(event.pos()))
        self.set_direction(azimuth, elevation, emit=True)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()
        foreground = palette.color(palette.WindowText)
        centre, radius = self._centre_radius()

        ring = QColor(foreground)
        ring.setAlpha(55)
        painter.setPen(QPen(ring, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        painter.drawLine(QPointF(centre.x() - radius, centre.y()), QPointF(centre.x() + radius, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - radius), QPointF(centre.x(), centre.y() + radius))

        # Measured directions, so a listener can see where the data actually is.
        if self._enabled_marks:
            faint = QColor(foreground)
            faint.setAlpha(45)
            painter.setPen(Qt.NoPen)
            painter.setBrush(faint)
            for azimuth, elevation in self._enabled_marks:
                painter.drawEllipse(self._to_point(azimuth, elevation), 1.6, 1.6)

        self._draw_head(painter, centre, radius * _HEAD_FRACTION, foreground)

        label = QColor(foreground)
        label.setAlpha(165)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        painter.setPen(QPen(label, 1))
        # Labels sit just inside the ring: at full radius the text box runs past
        # the widget edge and the word is clipped.
        inset = 0.82
        if self._plane == "top":
            pairs = (("front", 0.0), ("left", 90.0), ("back", 180.0), ("right", -90.0))
            for text, azimuth in pairs:
                point = self._to_point(azimuth, 0.0)
                x = centre.x() + (point.x() - centre.x()) * inset
                y = centre.y() + (point.y() - centre.y()) * inset
                painter.drawText(QRectF(x - 26.0, y - 8.0, 52.0, 16.0), Qt.AlignCenter, text)
        else:
            for text, elevation in (("+90", 90.0), ("0", 0.0), ("-90", -90.0)):
                point = self._to_point(0.0, elevation)
                y = centre.y() + (point.y() - centre.y()) * inset
                painter.drawText(
                    QRectF(centre.x() - radius, y - 8.0, radius * 0.9, 16.0),
                    Qt.AlignRight | Qt.AlignVCenter,
                    text,
                )

        marker = self._to_point(self._azimuth, self._elevation)
        accent = QColor(230, 120, 40)
        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(marker, 6.0, 6.0)
        painter.drawLine(QPointF(centre.x(), centre.y()), marker)

        painter.setPen(QPen(label, 1))
        readout = (
            f"az {self._azimuth:+.0f}°  el {self._elevation:+.0f}°"
            if self._plane == "top"
            else f"el {self._elevation:+.0f}°"
        )
        painter.drawText(QRectF(0.0, self.height() - 16.0, self.width(), 16.0), Qt.AlignCenter, readout)
        painter.end()

    def _draw_head(self, painter: QPainter, centre: QPointF, size: float, foreground: QColor) -> None:
        """A head, so which way is forward is never in doubt."""

        outline = QColor(foreground)
        outline.setAlpha(120)
        painter.setPen(QPen(outline, 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, size, size)

        if self._plane == "top":
            # A nose at the front and two ears, which orient the whole picture.
            nose = QPolygonF(
                [
                    QPointF(centre.x() - size * 0.22, centre.y() - size * 0.94),
                    QPointF(centre.x() + size * 0.22, centre.y() - size * 0.94),
                    QPointF(centre.x(), centre.y() - size * 1.34),
                ]
            )
            painter.drawPolygon(nose)
            for side in (-1.0, 1.0):
                painter.drawEllipse(
                    QPointF(centre.x() + side * size * 1.02, centre.y()), size * 0.16, size * 0.30
                )
        else:
            nose = QPolygonF(
                [
                    QPointF(centre.x() + size * 0.94, centre.y() - size * 0.18),
                    QPointF(centre.x() + size * 0.94, centre.y() + size * 0.18),
                    QPointF(centre.x() + size * 1.34, centre.y()),
                ]
            )
            painter.drawPolygon(nose)
            painter.drawEllipse(QPointF(centre.x() - size * 0.1, centre.y()), size * 0.18, size * 0.32)


class HeadDirectionSelector(QWidget):
    """Top and side views side by side, kept in step with one another."""

    directionChanged = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.top = HeadDirectionView("top")
        self.side = HeadDirectionView("side")
        layout.addWidget(self.top)
        layout.addWidget(self.side)

        for view in (self.top, self.side):
            view.directionChanged.connect(self._on_view_changed)

    def _on_view_changed(self, azimuth: float, elevation: float) -> None:
        for view in (self.top, self.side):
            view.set_direction(azimuth, elevation)
        self.directionChanged.emit(azimuth, elevation)

    @property
    def azimuth_deg(self) -> float:
        return self.top.azimuth_deg

    @property
    def elevation_deg(self) -> float:
        return self.top.elevation_deg

    def set_direction(self, azimuth_deg: float, elevation_deg: float, *, emit: bool = False) -> None:
        for view in (self.top, self.side):
            view.set_direction(azimuth_deg, elevation_deg)
        if emit:
            self.directionChanged.emit(self.azimuth_deg, self.elevation_deg)

    def set_marks(self, directions) -> None:
        for view in (self.top, self.side):
            view.set_marks(directions)

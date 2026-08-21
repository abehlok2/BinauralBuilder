"""Where a SOFA asset actually has measurements, drawn with QPainter.

Sparse or uneven coverage is the usual reason an HRTF sounds wrong in one
region and fine everywhere else, and a table of numbers does not make that
obvious. Plotting azimuth against elevation does: a band of dots along the
horizon with nothing above it explains a subject that cannot place height long
before any listening test would.

Drawn with QPainter rather than a plotting library, so the workbench keeps its
existing dependency set.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

__all__ = ["CoveragePlotWidget"]

_MARGIN = 34.0


class CoveragePlotWidget(QWidget):
    """Azimuth against elevation for every measurement in a dataset."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._azimuths_deg = np.zeros(0)
        self._elevations_deg = np.zeros(0)
        self._highlight: tuple[float, float] | None = None
        self._message = "No asset selected."
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setToolTip(
            "Each dot is one measured direction. Azimuth runs left (+90) to "
            "right (-90) through the front at 0; elevation runs bottom to top."
        )

    # --- data ---------------------------------------------------------------

    def set_directions(self, azimuths_deg, elevations_deg) -> None:
        """Show one dot per measured direction."""

        azimuths = np.asarray(azimuths_deg, dtype=np.float64).reshape(-1)
        elevations = np.asarray(elevations_deg, dtype=np.float64).reshape(-1)
        if azimuths.size != elevations.size:
            raise ValueError("azimuths and elevations must have the same length")
        # Wrap to a signed range so the front of the head sits in the middle
        # rather than split across both edges.
        self._azimuths_deg = ((azimuths + 180.0) % 360.0) - 180.0
        self._elevations_deg = elevations
        self._message = "" if azimuths.size else "This asset reports no directions."
        self.update()

    def set_highlight(self, azimuth_deg: float | None, elevation_deg: float | None = None) -> None:
        """Mark the direction currently being auditioned."""

        if azimuth_deg is None or elevation_deg is None:
            self._highlight = None
        else:
            wrapped = ((float(azimuth_deg) + 180.0) % 360.0) - 180.0
            self._highlight = (wrapped, float(elevation_deg))
        self.update()

    def clear(self, message: str = "No asset selected.") -> None:
        self._azimuths_deg = np.zeros(0)
        self._elevations_deg = np.zeros(0)
        self._highlight = None
        self._message = message
        self.update()

    @property
    def measurement_count(self) -> int:
        return int(self._azimuths_deg.size)

    # --- painting -----------------------------------------------------------

    def _plot_rect(self) -> QRectF:
        return QRectF(
            _MARGIN,
            10.0,
            max(1.0, self.width() - _MARGIN - 14.0),
            max(1.0, self.height() - 10.0 - 24.0),
        )

    def _to_pixel(self, azimuth_deg: float, elevation_deg: float, rect: QRectF) -> QPointF:
        # Azimuth increases toward the left ear, so +90 is drawn on the left.
        x = rect.left() + rect.width() * (1.0 - (azimuth_deg + 180.0) / 360.0)
        y = rect.bottom() - rect.height() * ((elevation_deg + 90.0) / 180.0)
        return QPointF(x, y)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()
        foreground = palette.color(palette.WindowText)
        rect = self._plot_rect()

        grid = QColor(foreground)
        grid.setAlpha(60)
        painter.setPen(QPen(grid, 1))
        painter.drawRect(rect)

        # Reference lines at the horizon and straight ahead.
        for elevation in (-45.0, 0.0, 45.0):
            point = self._to_pixel(0.0, elevation, rect)
            painter.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
        for azimuth in (-90.0, 0.0, 90.0):
            point = self._to_pixel(azimuth, 0.0, rect)
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))

        label = QColor(foreground)
        label.setAlpha(170)
        painter.setPen(QPen(label, 1))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        for azimuth, text in ((90.0, "L +90"), (0.0, "front 0"), (-90.0, "R -90")):
            point = self._to_pixel(azimuth, 0.0, rect)
            painter.drawText(
                QRectF(point.x() - 34.0, rect.bottom() + 3.0, 68.0, 18.0),
                Qt.AlignHCenter | Qt.AlignTop,
                text,
            )
        for elevation in (-90.0, 0.0, 90.0):
            point = self._to_pixel(0.0, elevation, rect)
            painter.drawText(
                QRectF(0.0, point.y() - 9.0, _MARGIN - 5.0, 18.0),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{elevation:+.0f}",
            )

        if self._message:
            painter.setPen(QPen(label, 1))
            painter.drawText(rect, Qt.AlignCenter, self._message)
            painter.end()
            return

        dot = QColor(palette.color(palette.Highlight))
        dot.setAlpha(190)
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot)
        radius = 2.0 if self._azimuths_deg.size > 400 else 2.8
        for azimuth, elevation in zip(self._azimuths_deg, self._elevations_deg):
            point = self._to_pixel(float(azimuth), float(elevation), rect)
            painter.drawEllipse(point, radius, radius)

        if self._highlight is not None:
            point = self._to_pixel(self._highlight[0], self._highlight[1], rect)
            painter.setBrush(Qt.NoBrush)
            marker = QColor(230, 120, 40)
            painter.setPen(QPen(marker, 2))
            painter.drawEllipse(point, 6.5, 6.5)
            painter.drawLine(QPointF(point.x() - 10.0, point.y()), QPointF(point.x() - 3.0, point.y()))
            painter.drawLine(QPointF(point.x() + 3.0, point.y()), QPointF(point.x() + 10.0, point.y()))

        painter.end()

"""Original-against-modified curves, drawn with QPainter.

The HRTF Lab needs to show the same quantity twice - as measured and as
modified - so the effect of a cue control is visible rather than inferred. One
widget does that for every quantity the panel plots: impulse response,
magnitude, group delay, ITD and ILD across azimuth.

Curves are drawn, not plotted by a library, so the workbench keeps its existing
dependency set.

A throttle sits in front of the redraw. Dragging a cue slider produces events
far faster than a recomputed HRIR can follow, and the audio preview is on the
same thread as the paint; coalescing the updates keeps a drag from starving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

__all__ = ["DEFAULT_THROTTLE_MS", "Curve", "HrtfCurvePlot", "PlotThrottle"]

#: Fast enough to feel continuous, slow enough not to crowd out audio.
DEFAULT_THROTTLE_MS = 80

# The left margin is measured from the labels it has to hold, not fixed: a box
# too narrow clips the minus sign off the low label, and on an ITD plot the
# sign is which ear leads. This is the floor, used before any data arrives.
_LEFT_MARGIN = 44.0

#: Never let the axis labels eat more than this share of the widget.
_MAX_LEFT_MARGIN_FRACTION = 0.4
_BOTTOM_MARGIN = 22.0
_TOP_MARGIN = 16.0
_RIGHT_MARGIN = 10.0


class PlotThrottle(QObject):
    """Coalesces rapid update requests into one redraw per interval.

    One throttle is shared by every plot in the panel, so moving a control
    refreshes the whole view once rather than each plot separately.
    """

    triggered = pyqtSignal()

    def __init__(self, interval_ms: int = DEFAULT_THROTTLE_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._fire)
        self._pending = False

    @property
    def interval_ms(self) -> int:
        return int(self._timer.interval())

    @property
    def pending(self) -> bool:
        return self._pending

    def request(self) -> None:
        """Ask for a redraw soon. Repeated calls inside the window collapse."""

        self._pending = True
        if not self._timer.isActive():
            self._timer.start()

    def flush(self) -> None:
        """Emit now, if anything is waiting. Used by tests and by Save."""

        self._timer.stop()
        self._fire()

    def _fire(self) -> None:
        if self._pending:
            self._pending = False
            self.triggered.emit()


@dataclass
class Curve:
    """One named series, with the colour and style it is drawn in."""

    label: str
    values: np.ndarray
    colour: QColor = field(default_factory=lambda: QColor(70, 130, 200))
    dashed: bool = False
    width: float = 1.6


class HrtfCurvePlot(QWidget):
    """A small line plot with a shared x axis and any number of curves."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._x: np.ndarray = np.zeros(0)
        self._curves: list[Curve] = []
        self._x_label = ""
        self._y_label = ""
        self._log_x = False
        self._message = "No asset selected."
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    # --- data ---------------------------------------------------------------

    def set_curves(
        self,
        x: np.ndarray,
        curves: list[Curve],
        *,
        x_label: str = "",
        y_label: str = "",
        log_x: bool = False,
    ) -> None:
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        for curve in curves:
            series = np.asarray(curve.values, dtype=np.float64).reshape(-1)
            if series.size != values.size:
                raise ValueError(
                    f"curve {curve.label!r} has {series.size} points but x has {values.size}"
                )
            curve.values = series
        self._x = values
        self._curves = list(curves)
        self._x_label = x_label
        self._y_label = y_label
        self._log_x = bool(log_x)
        self._message = "" if values.size else "Nothing to plot."
        self.update()

    def clear(self, message: str = "No asset selected.") -> None:
        self._x = np.zeros(0)
        self._curves = []
        self._message = message
        self.update()

    @property
    def curve_count(self) -> int:
        return len(self._curves)

    @property
    def title(self) -> str:
        return self._title

    # --- painting -----------------------------------------------------------

    def _plot_rect(self, left_margin: float = _LEFT_MARGIN) -> QRectF:
        return QRectF(
            left_margin,
            _TOP_MARGIN,
            max(1.0, self.width() - left_margin - _RIGHT_MARGIN),
            max(1.0, self.height() - _TOP_MARGIN - _BOTTOM_MARGIN),
        )

    def _left_margin_for(self, painter: QPainter, labels: list[str]) -> float:
        """Enough room for the widest label, within reason."""

        metrics = painter.fontMetrics()
        needed = max((metrics.horizontalAdvance(text) for text in labels), default=0.0)
        ceiling = max(_LEFT_MARGIN, self.width() * _MAX_LEFT_MARGIN_FRACTION)
        return float(min(max(_LEFT_MARGIN, needed + 9.0), ceiling))

    def _bounds(self) -> tuple[float, float, float, float]:
        stacked = np.concatenate([curve.values for curve in self._curves])
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            return 0.0, 1.0, -1.0, 1.0
        low, high = float(np.min(finite)), float(np.max(finite))
        if high - low < 1e-12:
            low, high = low - 0.5, high + 0.5
        pad = 0.06 * (high - low)
        x_axis = self._x_positions()
        return float(x_axis[0]), float(x_axis[-1]), low - pad, high + pad

    def _x_positions(self) -> np.ndarray:
        if not self._log_x:
            return self._x
        # A log axis cannot show DC; start at the first positive bin.
        safe = np.where(self._x > 0.0, self._x, np.nan)
        first = np.nanmin(safe) if np.any(np.isfinite(safe)) else 1.0
        return np.log10(np.where(self._x > 0.0, self._x, first))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()
        foreground = palette.color(palette.WindowText)
        rect = self._plot_rect()

        frame = QColor(foreground)
        frame.setAlpha(60)
        painter.setPen(QPen(frame, 1))
        painter.drawRect(rect)

        label_colour = QColor(foreground)
        label_colour.setAlpha(190)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)

        if self._title:
            painter.setPen(QPen(label_colour, 1))
            painter.drawText(
                QRectF(rect.left(), 0.0, rect.width(), _TOP_MARGIN),
                Qt.AlignLeft | Qt.AlignVCenter,
                self._title,
            )

        if self._message or not self._curves:
            painter.setPen(QPen(label_colour, 1))
            painter.drawText(rect, Qt.AlignCenter, self._message or "Nothing to plot.")
            painter.end()
            return

        x_low, x_high, y_low, y_high = self._bounds()
        # Re-lay the plot now that the labels are known, so neither is clipped.
        labels = [_axis_label(y_high), _axis_label(y_low)]
        rect = self._plot_rect(self._left_margin_for(painter, labels))
        painter.setPen(QPen(frame, 1))
        painter.drawRect(rect)

        x_axis = self._x_positions()
        x_span = max(x_high - x_low, 1e-12)
        y_span = max(y_high - y_low, 1e-12)

        def to_point(index: int, value: float) -> QPointF:
            return QPointF(
                rect.left() + rect.width() * (float(x_axis[index]) - x_low) / x_span,
                rect.bottom() - rect.height() * (value - y_low) / y_span,
            )

        # A zero line, where zero is on screen: ITD and ILD are read against it.
        if y_low < 0.0 < y_high:
            zero = rect.bottom() - rect.height() * (0.0 - y_low) / y_span
            zero_pen = QColor(foreground)
            zero_pen.setAlpha(70)
            painter.setPen(QPen(zero_pen, 1, Qt.DashLine))
            painter.drawLine(QPointF(rect.left(), zero), QPointF(rect.right(), zero))

        for curve in self._curves:
            pen = QPen(curve.colour, curve.width)
            if curve.dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            points = [to_point(index, float(value)) for index, value in enumerate(curve.values)]
            finite = [
                point for point, value in zip(points, curve.values) if np.isfinite(value)
            ]
            if len(finite) > 1:
                painter.drawPolyline(*finite)

        painter.setPen(QPen(label_colour, 1))
        for text, top in zip(labels, (rect.top() - 6.0, rect.bottom() - 8.0)):
            painter.drawText(
                QRectF(0.0, top, rect.left() - 5.0, 14.0),
                Qt.AlignRight | Qt.AlignVCenter,
                text,
            )
        if self._x_label:
            painter.drawText(
                QRectF(rect.left(), rect.bottom(), rect.width(), _BOTTOM_MARGIN),
                Qt.AlignHCenter | Qt.AlignVCenter,
                self._x_label,
            )
        if self._y_label:
            painter.drawText(
                QRectF(rect.left() + 4.0, rect.top(), rect.width(), 14.0),
                Qt.AlignLeft | Qt.AlignVCenter,
                self._y_label,
            )

        # Legend, only when there is more than one thing to tell apart.
        if len(self._curves) > 1:
            width = max(
                painter.fontMetrics().horizontalAdvance(curve.label)
                for curve in self._curves
            )
            swatch = 16.0
            left = rect.right() - (width + swatch + 12.0)
            offset = 0.0
            for curve in self._curves:
                painter.setPen(QPen(curve.colour, 2))
                y = rect.top() + 6.0 + offset
                painter.drawLine(QPointF(left, y), QPointF(left + swatch, y))
                painter.setPen(QPen(label_colour, 1))
                painter.drawText(
                    QRectF(left + swatch + 4.0, y - 7.0, width + 4.0, 14.0),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    curve.label,
                )
                offset += 13.0

        painter.end()


def _axis_label(value: float) -> str:
    """A short axis label that never loses its sign to rounding."""

    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    return f"{value:.4g}"

"""Analysis views for the SAM workbench.

Waveform, spectrum, instantaneous frequency, and IPD, drawn with QPainter so
the panel adds no plotting dependency and stays cheap enough to redraw while a
preview is playing. The numbers come from
:mod:`src.audio.sam_workbench.analysis`, which is Qt-free and shared with the
export report, so a plot and a report can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.audio.sam_workbench.analysis import (
    instantaneous_frequency_hz,
    interaural_phase_difference_rad,
    magnitude_spectrum_db,
    peak_envelope,
    summarize_cues,
)

LEFT_COLOR = QColor(80, 170, 245)
RIGHT_COLOR = QColor(245, 140, 80)
TICK_GRID_COLOR = QColor(150, 150, 155, 95)
AXIS_COLOR = QColor(170, 172, 178)
TEXT_COLOR = QColor(205, 207, 212)
DIM_TEXT_COLOR = QColor(150, 152, 158)

__all__ = [
    "PlotWidget",
    "PlotSeries",
    "SamAnalysisPanel",
    "nice_ticks",
    "format_tick_value",
]


def _nice_step(rough: float) -> float:
    """A round step at or above ``rough``: 1/2/2.5/5 times a power of ten."""

    if not math.isfinite(rough) or rough <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(rough))
    fraction = rough / (10.0**exponent)
    for mantissa in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= mantissa + 1e-12:
            return mantissa * (10.0**exponent)
    return 10.0 ** (exponent + 1)


def nice_ticks(low: float, high: float, target: int = 5) -> np.ndarray:
    """Round tick values covering ``[low, high]`` without exceeding it much.

    Ticks land on round numbers (so a reader can interpolate by eye) and stay
    inside the plotted range, which is what makes the axis edge meaningful.
    """

    low = float(low)
    high = float(high)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return np.array([low, high], dtype=np.float64)
    step = _nice_step((high - low) / max(target - 1, 1))
    start = math.ceil(low / step - 1e-9) * step
    count = int(math.floor((high - start) / step + 1e-9)) + 1
    ticks = start + np.arange(max(count, 0), dtype=np.float64) * step
    tolerance = step * 1e-6
    return ticks[(ticks >= low - tolerance) & (ticks <= high + tolerance)]


def _decimals_for_step(step: float) -> int:
    decimals = 0
    scaled = abs(float(step))
    while decimals < 4 and abs(scaled - round(scaled)) > 1e-9:
        scaled *= 10.0
        decimals += 1
    return decimals


def format_tick_value(value: float, step: float) -> str:
    """A tick label whose precision matches the step it sits on."""

    value = float(value)
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude >= 10_000.0 or magnitude < 0.001):
        return f"{value:.3g}"
    return f"{value:.{_decimals_for_step(step)}f}"


@dataclass
class PlotSeries:
    """One curve, in data coordinates."""

    x: np.ndarray
    y: np.ndarray
    color: QColor = field(default_factory=lambda: QColor(LEFT_COLOR))
    name: str = ""
    #: Optional lower bound for a filled band (used by the waveform envelope).
    y_low: np.ndarray | None = None


class PlotWidget(QWidget):
    """A small, dependency-free line plot with readable axes.

    Deliberately not a plotting library: axis ranges, round-number ticks with
    labels, a grid aligned to those ticks, a compact legend, and polylines.
    Margins are computed from the actual label text, so nothing is squeezed
    and nothing is guessed.
    """

    def __init__(
        self,
        title: str = "",
        *,
        x_label: str = "",
        y_label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._series: list[PlotSeries] = []
        self._x_range: tuple[float, float] | None = None
        self._y_range: tuple[float, float] | None = None
        self._message = "No preview rendered yet"
        self.setMinimumHeight(170)
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #: The data-area rectangle from the last paint, for tests and for
        #: anything that wants to translate a click into data coordinates.
        self.last_plot_rect: QRectF | None = None

    # --- data ---------------------------------------------------------------

    def set_series(
        self,
        series: Sequence[PlotSeries],
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        self._series = list(series)
        self._x_range = x_range
        self._y_range = y_range
        self._message = "" if self._series else "No data"
        self.update()

    def clear(self, message: str = "No preview rendered yet") -> None:
        self._series = []
        self._message = message
        self.update()

    @property
    def series(self) -> list[PlotSeries]:
        return list(self._series)

    def _legend_items(self) -> list[tuple[str, QColor]]:
        """Distinct named series, in draw order, for the in-plot legend."""

        items: list[tuple[str, QColor]] = []
        seen: set[str] = set()
        for entry in self._series:
            if entry.name and entry.name not in seen:
                seen.add(entry.name)
                items.append((entry.name, QColor(entry.color)))
        return items

    def data_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Effective ``(x_range, y_range)`` after defaults and padding."""

        if self._x_range is not None:
            x_range = self._x_range
        elif self._series:
            x_range = (
                float(min(float(np.min(entry.x)) for entry in self._series if entry.x.size)),
                float(max(float(np.max(entry.x)) for entry in self._series if entry.x.size)),
            )
        else:
            x_range = (0.0, 1.0)

        if self._y_range is not None:
            y_range = self._y_range
        elif self._series:
            lows = [float(np.min(entry.y)) for entry in self._series if entry.y.size]
            highs = [float(np.max(entry.y)) for entry in self._series if entry.y.size]
            for entry in self._series:
                if entry.y_low is not None and entry.y_low.size:
                    lows.append(float(np.min(entry.y_low)))
            y_range = (min(lows), max(highs)) if lows and highs else (-1.0, 1.0)
            # A little headroom so a curve at full scale does not sit on the frame.
            span = y_range[1] - y_range[0]
            pad = max(span * 0.04, 1e-9)
            y_range = (y_range[0] - pad, y_range[1] + pad)
        else:
            y_range = (-1.0, 1.0)

        if x_range[1] - x_range[0] <= 0:
            x_range = (x_range[0], x_range[0] + 1.0)
        if y_range[1] - y_range[0] <= 0:
            padding = max(abs(y_range[0]) * 0.1, 0.5)
            y_range = (y_range[0] - padding, y_range[1] + padding)
        return x_range, y_range

    # --- painting -----------------------------------------------------------

    def _tick_font(self) -> QFont:
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        return font

    def _layout_margins(self, metrics: QFontMetricsF) -> QRectF:
        """The data rectangle, inset by whatever its own labels need."""

        (x_low, x_high), (y_low, y_high) = self.data_ranges()
        y_ticks = nice_ticks(y_low, y_high)
        x_ticks = nice_ticks(x_low, x_high)
        step_for_labels = getattr(self, "_y_step", abs(y_high - y_low) or 1.0)
        texts = [format_tick_value(float(v), step_for_labels) for v in y_ticks]
        widest = 0.0
        for text in texts:
            widest = max(widest, metrics.horizontalAdvance(text))

        tick_length = 4.0
        left = widest + tick_length + 6.0
        bottom = metrics.height() + tick_length + 2.0
        top = metrics.height() * 0.9 + 4.0 if self._title else 6.0
        right = 10.0
        if self._x_label:
            bottom += metrics.height()
        if self._y_label:
            left += metrics.height() * 0.9 + 4.0
        return QRectF(self.rect()).adjusted(left, top, -right, -bottom)

    def _prepare_axes(self):
        """Ranges, ticks, per-axis steps, and the laid-out data rectangle."""

        (x_low, x_high), (y_low, y_high) = self.data_ranges()
        x_tick_values = nice_ticks(x_low, x_high)
        y_tick_values = nice_ticks(y_low, y_high)
        x_step = (
            float(np.mean(np.diff(x_tick_values)))
            if len(x_tick_values) > 1
            else max(abs(x_high - x_low), 1e-9)
        )
        y_step = (
            float(np.mean(np.diff(y_tick_values)))
            if len(y_tick_values) > 1
            else max(abs(y_high - y_low), 1e-9)
        )
        self._x_step = x_step
        self._y_step = y_step
        font = self._tick_font()
        metrics = QFontMetricsF(font)
        rectangle = self._layout_margins(metrics)
        return (
            (x_low, x_high),
            (y_low, y_high),
            x_tick_values,
            y_tick_values,
            x_step,
            y_step,
            font,
            metrics,
            rectangle,
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Everything numeric is resolved before the painter exists: layout
        # must never depend on an active paint device.
        (
            (x_low, x_high),
            (y_low, y_high),
            x_ticks,
            y_ticks,
            _x_step,
            _y_step,
            font,
            metrics,
            rectangle,
        ) = self._prepare_axes()
        self.last_plot_rect = QRectF(rectangle)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(font)

        if not self._series:
            painter.setPen(QPen(AXIS_COLOR, 1.0))
            painter.drawRect(rectangle)
            if self._title:
                painter.drawText(
                    QRectF(0, 2.0, self.width(), metrics.height() * 1.3),
                    Qt.AlignHCenter | Qt.AlignTop,
                    self._title,
                )
            painter.drawText(rectangle, Qt.AlignCenter, self._message)
            painter.end()
            return

        def to_point(x_value: float, y_value: float) -> QPointF:
            x_fraction = (x_value - x_low) / (x_high - x_low)
            y_fraction = (y_value - y_low) / (y_high - y_low)
            return QPointF(
                rectangle.left() + x_fraction * rectangle.width(),
                rectangle.bottom() - y_fraction * rectangle.height(),
            )

        tick_length = 4.0

        # Grid first, aligned to the ticks it explains.
        grid_pen = QPen(TICK_GRID_COLOR, 1.0)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)
        for value in x_ticks:
            point = to_point(float(value), y_low)
            painter.drawLine(
                QPointF(point.x(), rectangle.top()),
                QPointF(point.x(), rectangle.bottom()),
            )
        for value in y_ticks:
            point = to_point(x_low, float(value))
            painter.drawLine(
                QPointF(rectangle.left(), point.y()),
                QPointF(rectangle.right(), point.y()),
            )

        frame_pen = QPen(AXIS_COLOR, 1.0)
        frame_pen.setCosmetic(True)
        painter.setPen(frame_pen)
        painter.drawRect(rectangle)
        for value in x_ticks:
            point = to_point(float(value), y_low)
            painter.drawLine(
                QPointF(point.x(), rectangle.bottom()),
                QPointF(point.x(), rectangle.bottom() + tick_length),
            )
        for value in y_ticks:
            point = to_point(x_low, float(value))
            painter.drawLine(
                QPointF(rectangle.left() - tick_length, point.y()),
                QPointF(rectangle.left(), point.y()),
            )

        painter.setFont(font)
        for value in x_ticks:
            point = to_point(float(value), y_low)
            painter.save()
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(point.x() - 60.0, rectangle.bottom() + tick_length + 1.0, 120.0,
                       metrics.height()),
                Qt.AlignHCenter | Qt.AlignTop,
                format_tick_value(float(value), self._x_step),
            )
            painter.restore()
        for value in y_ticks:
            point = to_point(x_low, float(value))
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(
                    rectangle.left() - tick_length - 6.0 - 90.0,
                    point.y() - metrics.height() / 2.0,
                    90.0,
                    metrics.height(),
                ),
                Qt.AlignRight | Qt.AlignVCenter,
                format_tick_value(float(value), self._y_step),
            )

        if self._x_label:
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(
                    rectangle.left(),
                    rectangle.bottom() + tick_length + 1.0 + metrics.height(),
                    rectangle.width(),
                    metrics.height(),
                ),
                Qt.AlignHCenter | Qt.AlignTop,
                self._x_label,
            )
        if self._y_label:
            painter.save()
            painter.translate(4.0, rectangle.center().y())
            painter.rotate(-90.0)
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(-rectangle.height() / 2.0, 0.0, rectangle.height(), metrics.height()),
                Qt.AlignHCenter | Qt.AlignVCenter,
                self._y_label,
            )
            painter.restore()

        if self._title:
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(0, 2.0, self.width(), metrics.height() * 1.3),
                Qt.AlignHCenter | Qt.AlignTop,
                self._title,
            )

        self._draw_legend(painter, rectangle, metrics)
        self._draw_series(painter, to_point)
        painter.end()

    def _draw_legend(self, painter: QPainter, rectangle: QRectF, metrics: QFontMetricsF) -> None:
        items = self._legend_items()
        if len(items) < 2:
            return
        swatch = 14.0
        gap = 5.0
        row_height = metrics.height() + 2.0
        width = max(
            swatch + gap + metrics.horizontalAdvance(name) for name, _color in items
        ) + 10.0
        height = row_height * len(items) + 6.0
        origin = QPointF(rectangle.right() - width - 6.0, rectangle.top() + 6.0)
        background = QColor(24, 26, 30, 190)
        box = QRectF(origin.x(), origin.y(), width, height)
        painter.fillRect(box, background)
        painter.setPen(QPen(AXIS_COLOR, 1.0))
        painter.drawRect(box)
        for index, (name, color) in enumerate(items):
            y = origin.y() + 4.0 + index * row_height + row_height / 2.0
            pen = QPen(color, 1.8)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(origin.x() + 5.0, y), QPointF(origin.x() + 5.0 + swatch, y)
            )
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF(origin.x() + 5.0 + swatch + gap, y - row_height / 2.0,
                       width, row_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                name,
            )

    def _draw_series(self, painter: QPainter, to_point) -> None:
        for entry in self._series:
            if entry.x.size == 0:
                continue
            if entry.y_low is not None:
                path = QPainterPath()
                path.moveTo(to_point(float(entry.x[0]), float(entry.y[0])))
                for index in range(1, entry.x.size):
                    path.lineTo(to_point(float(entry.x[index]), float(entry.y[index])))
                for index in range(entry.x.size - 1, -1, -1):
                    path.lineTo(to_point(float(entry.x[index]), float(entry.y_low[index])))
                path.closeSubpath()
                fill = QColor(entry.color)
                fill.setAlpha(80)
                painter.fillPath(path, fill)
                continue
            pen = QPen(entry.color, 1.6)
            pen.setCosmetic(True)
            painter.setPen(pen)
            path = QPainterPath()
            path.moveTo(to_point(float(entry.x[0]), float(entry.y[0])))
            for index in range(1, entry.x.size):
                path.lineTo(to_point(float(entry.x[index]), float(entry.y[index])))
            painter.drawPath(path)


class SamAnalysisPanel(QWidget):
    """Waveform, spectrum, instantaneous frequency, and IPD for a preview buffer."""

    #: Analysis is decimated to this many points per curve before drawing.
    #: Enough that a fast SAM sweep stays smooth at the sizes this panel gets.
    PLOT_POINTS = 900

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.waveform_plot = PlotWidget("Waveform", x_label="time (s)", y_label="amplitude")
        self.spectrum_plot = PlotWidget("Spectrum", x_label="frequency (Hz)", y_label="dBFS")
        self.frequency_plot = PlotWidget(
            "Instantaneous frequency", x_label="time (s)", y_label="Hz"
        )
        self.ipd_plot = PlotWidget(
            "Interaural phase difference", x_label="time (s)", y_label="rad"
        )
        self.summary_label = QLabel("Render a preview to analyse it.")
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self.waveform_plot, 0, 0)
        grid.addWidget(self.spectrum_plot, 0, 1)
        grid.addWidget(self.frequency_plot, 1, 0)
        grid.addWidget(self.ipd_plot, 1, 1)
        layout.addLayout(grid)
        layout.addWidget(self.summary_label)

        self._audio: np.ndarray | None = None
        self._sample_rate = 44_100

    # --- data ---------------------------------------------------------------

    def clear(self, message: str = "No preview rendered yet") -> None:
        for plot in (self.waveform_plot, self.spectrum_plot, self.frequency_plot, self.ipd_plot):
            plot.clear(message)
        self.summary_label.setText("Render a preview to analyse it.")
        self._audio = None

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Analyse a frame-major ``(frames, 2)`` preview buffer and redraw."""

        block = np.asarray(audio, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != 2 or block.shape[0] < 4:
            self.clear("Preview too short to analyse")
            return
        self._audio = block
        self._sample_rate = int(sample_rate)

        self._draw_waveform(block, sample_rate)
        self._draw_spectrum(block, sample_rate)
        self._draw_frequency(block, sample_rate)
        self._draw_ipd(block, sample_rate)

        cues = summarize_cues(block, sample_rate, edge_trim_frames=self._edge_trim(block))
        self.summary_label.setText(
            "Peak L/R {peak_left_dbfs:.1f} / {peak_right_dbfs:.1f} dBFS  ·  "
            "instantaneous frequency {frequency_hz_min:.1f}-{frequency_hz_max:.1f} Hz  ·  "
            "IPD {ipd_rad_min:+.2f} to {ipd_rad_max:+.2f} rad  ·  "
            "ILD {ild_db_min:+.1f} to {ild_db_max:+.1f} dB".format(**cues)
        )

    def _edge_trim(self, block: np.ndarray) -> int:
        """Samples to ignore at each end.

        Wide enough to clear both the analytic signal's edge ringing and the
        preview's own fade in and out, which would otherwise dominate the
        measured frequency range; never more than a quarter of the buffer.
        """

        frames = int(block.shape[0])
        return max(1, min(frames // 4, max(64, frames // 20)))

    # --- individual views ---------------------------------------------------

    def _draw_waveform(self, block: np.ndarray, sample_rate: int) -> None:
        minimum, maximum = peak_envelope(block, self.PLOT_POINTS)
        duration = block.shape[0] / float(sample_rate)
        times = np.linspace(0.0, duration, minimum.shape[1])
        self.waveform_plot.set_series(
            [
                PlotSeries(times, maximum[0], QColor(LEFT_COLOR), "left", y_low=minimum[0]),
                PlotSeries(times, maximum[1], QColor(RIGHT_COLOR), "right", y_low=minimum[1]),
            ],
            x_range=(0.0, duration),
        )

    def _draw_spectrum(self, block: np.ndarray, sample_rate: int) -> None:
        frequencies, magnitudes = magnitude_spectrum_db(block, sample_rate)
        if frequencies.size == 0:
            self.spectrum_plot.clear("Preview too short for a spectrum")
            return
        # Show the band that matters for a SAM carrier rather than everything.
        peak_bin = int(np.argmax(magnitudes[0]))
        upper = float(min(frequencies[-1], max(2_000.0, frequencies[peak_bin] * 8.0)))
        inside = frequencies <= upper
        self.spectrum_plot.set_series(
            [
                PlotSeries(frequencies[inside], magnitudes[0][inside], QColor(LEFT_COLOR), "left"),
                PlotSeries(frequencies[inside], magnitudes[1][inside], QColor(RIGHT_COLOR), "right"),
            ],
            x_range=(0.0, upper),
            y_range=(-96.0, 6.0),
        )

    def _decimate(self, values: np.ndarray) -> np.ndarray:
        if values.size <= self.PLOT_POINTS:
            return values
        step = int(np.ceil(values.size / self.PLOT_POINTS))
        return values[::step]

    def _draw_frequency(self, block: np.ndarray, sample_rate: int) -> None:
        frequency = instantaneous_frequency_hz(block, sample_rate)
        # Trim the analytic-signal edges, where the Hilbert transform rings.
        edge = self._edge_trim(block)
        trimmed = frequency[:, edge:-edge] if frequency.shape[1] > 2 * edge else frequency
        left = self._decimate(trimmed[0])
        right = self._decimate(trimmed[1])
        times = np.linspace(0.0, trimmed.shape[1] / float(sample_rate), left.size)
        self.frequency_plot.set_series(
            [
                PlotSeries(times, left, QColor(LEFT_COLOR), "left"),
                PlotSeries(times, right, QColor(RIGHT_COLOR), "right"),
            ]
        )

    def _draw_ipd(self, block: np.ndarray, sample_rate: int) -> None:
        ipd = interaural_phase_difference_rad(block)
        edge = self._edge_trim(block)
        trimmed = ipd[edge:-edge] if ipd.size > 2 * edge else ipd
        values = self._decimate(trimmed)
        times = np.linspace(0.0, trimmed.size / float(sample_rate), values.size)
        self.ipd_plot.set_series(
            [PlotSeries(times, values, QColor(LEFT_COLOR), "left minus right")],
            y_range=(-np.pi, np.pi),
        )

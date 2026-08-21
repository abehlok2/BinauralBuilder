"""Analysis views for the SAM workbench.

Waveform, spectrum, instantaneous frequency, and IPD, drawn with QPainter so
the panel adds no plotting dependency and stays cheap enough to redraw while a
preview is playing. The numbers come from
:mod:`src.audio.sam_workbench.analysis`, which is Qt-free and shared with the
export report, so a plot and a report can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
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
GRID_COLOR = QColor(128, 128, 128, 70)
AXIS_COLOR = QColor(128, 128, 128, 160)

__all__ = ["PlotWidget", "PlotSeries", "SamAnalysisPanel"]


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
    """A minimal, dependency-free line plot.

    Deliberately small: axis ranges, a grid, one label, and polylines. Anything
    richer belongs in a dedicated analysis window, not in a parameter dialog.
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
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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
        else:
            y_range = (-1.0, 1.0)

        if x_range[1] - x_range[0] <= 0:
            x_range = (x_range[0], x_range[0] + 1.0)
        if y_range[1] - y_range[0] <= 0:
            padding = max(abs(y_range[0]) * 0.1, 0.5)
            y_range = (y_range[0] - padding, y_range[1] + padding)
        return x_range, y_range

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rectangle = QRectF(self.rect()).adjusted(48.0, 18.0, -8.0, -22.0)

        painter.setPen(QPen(AXIS_COLOR, 1.0))
        painter.drawRect(rectangle)

        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        if self._title:
            painter.drawText(QRectF(0, 0, self.width(), 18.0), Qt.AlignCenter, self._title)

        if not self._series:
            painter.drawText(rectangle, Qt.AlignCenter, self._message)
            painter.end()
            return

        (x_low, x_high), (y_low, y_high) = self.data_ranges()

        def to_point(x_value: float, y_value: float) -> QPointF:
            x_fraction = (x_value - x_low) / (x_high - x_low)
            y_fraction = (y_value - y_low) / (y_high - y_low)
            return QPointF(
                rectangle.left() + x_fraction * rectangle.width(),
                rectangle.bottom() - y_fraction * rectangle.height(),
            )

        painter.setPen(QPen(GRID_COLOR, 1.0, Qt.DotLine))
        for step in range(1, 4):
            fraction = step / 4.0
            y = rectangle.bottom() - fraction * rectangle.height()
            painter.drawLine(QPointF(rectangle.left(), y), QPointF(rectangle.right(), y))

        painter.setPen(QPen(AXIS_COLOR, 1.0))
        painter.drawText(
            QRectF(0, rectangle.top() - 6.0, 44.0, 14.0), Qt.AlignRight | Qt.AlignVCenter, f"{y_high:.3g}"
        )
        painter.drawText(
            QRectF(0, rectangle.bottom() - 8.0, 44.0, 14.0), Qt.AlignRight | Qt.AlignVCenter, f"{y_low:.3g}"
        )
        if self._x_label:
            painter.drawText(
                QRectF(rectangle.left(), rectangle.bottom() + 4.0, rectangle.width(), 16.0),
                Qt.AlignRight,
                self._x_label,
            )
        if self._y_label:
            painter.drawText(
                QRectF(4.0, rectangle.top() - 16.0, 120.0, 14.0), Qt.AlignLeft, self._y_label
            )

        for entry in self._series:
            if entry.x.size == 0:
                continue
            painter.setPen(QPen(entry.color, 1.2))
            if entry.y_low is not None:
                path = QPainterPath()
                path.moveTo(to_point(float(entry.x[0]), float(entry.y[0])))
                for index in range(1, entry.x.size):
                    path.lineTo(to_point(float(entry.x[index]), float(entry.y[index])))
                for index in range(entry.x.size - 1, -1, -1):
                    path.lineTo(to_point(float(entry.x[index]), float(entry.y_low[index])))
                path.closeSubpath()
                fill = QColor(entry.color)
                fill.setAlpha(110)
                painter.fillPath(path, fill)
                continue
            path = QPainterPath()
            path.moveTo(to_point(float(entry.x[0]), float(entry.y[0])))
            for index in range(1, entry.x.size):
                path.lineTo(to_point(float(entry.x[index]), float(entry.y[index])))
            painter.drawPath(path)
        painter.end()


class SamAnalysisPanel(QWidget):
    """Waveform, spectrum, instantaneous frequency, and IPD for a preview buffer."""

    #: Analysis is decimated to this many points per curve before drawing.
    PLOT_POINTS = 600

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.waveform_plot = PlotWidget("Waveform", x_label="s", y_label="amplitude")
        self.spectrum_plot = PlotWidget("Spectrum", x_label="Hz", y_label="dBFS")
        self.frequency_plot = PlotWidget("Instantaneous frequency", x_label="s", y_label="Hz")
        self.ipd_plot = PlotWidget("Interaural phase difference", x_label="s", y_label="rad")
        self.summary_label = QLabel("Render a preview to analyse it.")
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
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

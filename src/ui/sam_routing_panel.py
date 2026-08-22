"""Buses, per-source routing, multiband splits, and what the scene will cost.

Four things that belong on one page because they are one decision. How many
sources, how many bands, and which interpolation mode all feed the cost
estimate, and the cost estimate is the thing that says whether the scene can
run in real time. Splitting them across tabs would mean editing in one place
and discovering the consequence in another.

The band strip is drawn rather than tabulated: crossover frequencies are a
layout along a log axis, and a picture of that layout answers "is the low band
too wide" immediately, where a column of numbers does not.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.cost import CostInputs, estimate_cost, measure_throughput
from src.audio.sam_workbench.render.routing import BandRouting, BusSpec, MASTER_BUS, SourceRouting

from .table_widgets import place, reset_rows

__all__ = ["SamRoutingPanel", "BandStripView"]

_MIN_HZ = 20.0
_MAX_HZ = 20_000.0
_INTERPOLATION_MODES = (
    "nearest",
    "three_neighbor",
    "spherical_triangular",
    "delay_magnitude",
    "spherical_harmonic",
)


class BandStripView(QWidget):
    """The crossover layout on a log frequency axis, band by band."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._routing = BandRouting()
        self.setMinimumHeight(74)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(
            "Each block is one band. Width follows the log frequency axis, so a "
            "band that looks wide really is wide."
        )

    def set_routing(self, routing: BandRouting) -> None:
        self._routing = routing
        self.update()

    @staticmethod
    def _position(frequency: float) -> float:
        frequency = max(_MIN_HZ, min(_MAX_HZ, float(frequency)))
        return (math.log10(frequency) - math.log10(_MIN_HZ)) / (
            math.log10(_MAX_HZ) - math.log10(_MIN_HZ)
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        foreground = self.palette().color(self.palette().WindowText)

        left, top = 4.0, 4.0
        width = max(1.0, self.width() - 8.0)
        height = max(1.0, self.height() - 26.0)

        edges = [_MIN_HZ, *sorted(self._routing.crossovers_hz), _MAX_HZ]
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)

        for index in range(len(edges) - 1):
            start = left + width * self._position(edges[index])
            end = left + width * self._position(edges[index + 1])
            block = QRectF(start, top, max(1.0, end - start), height)

            enabled = self._routing.enabled_for(index)
            gain_db = (
                self._routing.band_gains_db[index]
                if index < len(self._routing.band_gains_db)
                else 0.0
            )
            hue = QColor.fromHsv(int(28 + 190 * index / max(1, len(edges) - 2)) % 360, 150, 200)
            hue.setAlpha(120 if enabled else 35)
            painter.setBrush(hue)
            painter.setPen(QPen(QColor(foreground.red(), foreground.green(), foreground.blue(), 90), 1))
            painter.drawRect(block)

            label = QColor(foreground)
            label.setAlpha(200 if enabled else 90)
            painter.setPen(QPen(label, 1))
            text = f"band {index + 1}\n{gain_db:+.1f} dB" + ("" if enabled else "\nmuted")
            painter.drawText(block, Qt.AlignCenter, text)

        axis = QColor(foreground)
        axis.setAlpha(150)
        painter.setPen(QPen(axis, 1))
        for frequency in (20.0, 100.0, 500.0, 2000.0, 8000.0, 20000.0):
            x = left + width * self._position(frequency)
            text = f"{frequency / 1000:g}k" if frequency >= 1000 else f"{frequency:g}"
            # The first and last labels sit on the widget edge, where a centred
            # text box would be half outside it and the label clipped.
            box = QRectF(x - 22.0, top + height + 2.0, 44.0, 16.0)
            box.moveLeft(min(max(box.left(), 0.0), self.width() - box.width()))
            painter.drawText(box, Qt.AlignCenter, text)
        painter.end()


class SamRoutingPanel(QWidget):
    """Edit buses, source routing, and multiband splits, with a cost readout."""

    routingChanged = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buses: list[BusSpec] = [BusSpec(id=MASTER_BUS, name="Master")]
        self._sources: list[SourceRouting] = [SourceRouting(source_id="source.1")]
        self._band = BandRouting()
        self._throughput: float | None = None
        self._loading = False

        self._build_ui()
        self._refresh_all()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(self._build_buses(), 1)
        top.addWidget(self._build_sources(), 1)
        layout.addLayout(top, 1)

        layout.addWidget(self._build_bands())
        layout.addWidget(self._build_cost())

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_buses(self) -> QWidget:
        box = QGroupBox("Buses")
        outer = QVBoxLayout(box)
        self.bus_table = QTableWidget(0, 4)
        self.bus_table.setHorizontalHeaderLabels(["Bus", "Gain dB", "Mute", "Solo"])
        self.bus_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bus_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bus_table.verticalHeader().setVisible(False)
        self.bus_table.setMinimumHeight(150)
        outer.addWidget(self.bus_table, 1)

        row = QHBoxLayout()
        add = QPushButton("Add bus…")
        add.clicked.connect(self._add_bus)
        row.addWidget(add)
        remove = QPushButton("Remove")
        remove.setToolTip("The master bus cannot be removed; everything mixes into it.")
        remove.clicked.connect(self._remove_bus)
        row.addWidget(remove)
        outer.addLayout(row)
        return box

    def _build_sources(self) -> QWidget:
        box = QGroupBox("Sources")
        outer = QVBoxLayout(box)
        self.source_table = QTableWidget(0, 5)
        self.source_table.setHorizontalHeaderLabels(["Source", "Bus", "Gain dB", "Mute", "Solo"])
        self.source_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.source_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.source_table.verticalHeader().setVisible(False)
        self.source_table.setMinimumHeight(150)
        outer.addWidget(self.source_table, 1)

        row = QHBoxLayout()
        add = QPushButton("Add source…")
        add.clicked.connect(self._add_source)
        row.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_source)
        row.addWidget(remove)
        outer.addLayout(row)
        return box

    def _build_bands(self) -> QWidget:
        box = QGroupBox("Multiband split")
        outer = QVBoxLayout(box)

        self.band_strip = BandStripView()
        outer.addWidget(self.band_strip)

        row = QHBoxLayout()
        row.addWidget(QLabel("Crossovers:"))
        self.crossover_label = QLabel("none — wideband")
        self.crossover_label.setToolTip(
            "Linkwitz-Riley crossovers. Adjacent bands sum to an allpass, so "
            "splitting and re-summing does not change the response."
        )
        row.addWidget(self.crossover_label, 1)

        add = QPushButton("Add crossover…")
        add.clicked.connect(self._add_crossover)
        row.addWidget(add)
        remove = QPushButton("Remove highest")
        remove.clicked.connect(self._remove_crossover)
        row.addWidget(remove)

        row.addWidget(QLabel("Order:"))
        self.order_combo = QComboBox()
        for order in (2, 4, 8):
            self.order_combo.addItem(f"LR{order}", order)
        self.order_combo.setCurrentIndex(1)
        self.order_combo.currentIndexChanged.connect(self._on_band_changed)
        row.addWidget(self.order_combo)
        outer.addLayout(row)

        self.band_table = QTableWidget(0, 3)
        self.band_table.setHorizontalHeaderLabels(["Band", "Gain dB", "Enabled"])
        self.band_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.band_table.verticalHeader().setVisible(False)
        self.band_table.setMinimumHeight(150)
        outer.addWidget(self.band_table)
        return box

    def _build_cost(self) -> QWidget:
        box = QGroupBox("Cost")
        outer = QHBoxLayout(box)

        form = QFormLayout()
        self.taps_spin = QSpinBox()
        self.taps_spin.setRange(16, 8192)
        self.taps_spin.setValue(256)
        self.taps_spin.setToolTip("HRIR length. Overlap-save cost grows with its logarithm.")
        self.taps_spin.valueChanged.connect(self._refresh_cost)
        form.addRow("HRIR taps:", self.taps_spin)

        self.block_spin = QSpinBox()
        self.block_spin.setRange(32, 8192)
        self.block_spin.setValue(512)
        self.block_spin.setToolTip("Audio callback block size, which sets the real-time budget.")
        self.block_spin.valueChanged.connect(self._refresh_cost)
        form.addRow("Block size:", self.block_spin)
        outer.addLayout(form)

        second = QFormLayout()
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(list(_INTERPOLATION_MODES))
        self.interpolation_combo.currentIndexChanged.connect(self._refresh_cost)
        second.addRow("Interpolation:", self.interpolation_combo)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(8000.0, 192_000.0)
        self.rate_spin.setDecimals(0)
        self.rate_spin.setValue(48_000.0)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.valueChanged.connect(self._refresh_cost)
        second.addRow("Sample rate:", self.rate_spin)
        outer.addLayout(second)

        readout = QVBoxLayout()
        self.cost_label = QLabel("")
        self.cost_label.setWordWrap(True)
        self.cost_label.setMinimumWidth(300)
        readout.addWidget(self.cost_label, 1)
        self.measure_button = QPushButton("Measure this machine")
        self.measure_button.setToolTip(
            "Time an actual overlap-save convolution here, so the estimate is "
            "about this machine rather than a conservative guess."
        )
        self.measure_button.clicked.connect(self.measure)
        readout.addWidget(self.measure_button)
        outer.addLayout(readout, 1)
        return box

    # --- state --------------------------------------------------------------

    @property
    def buses(self) -> tuple[BusSpec, ...]:
        return tuple(self._buses)

    @property
    def sources(self) -> tuple[SourceRouting, ...]:
        return tuple(self._sources)

    @property
    def band_routing(self) -> BandRouting:
        return self._band

    def describe(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "buses": [bus.describe() for bus in self._buses],
            "sources": [source.describe() for source in self._sources],
            "bands": self._band.describe(),
        }

    def params(self) -> dict[str, Any]:
        return {"samRouting": self.describe()}

    def set_params(self, params: Mapping[str, Any]) -> None:
        data = params.get("samRouting")
        if not isinstance(data, Mapping):
            return
        self._loading = True
        try:
            buses = [
                BusSpec(
                    id=str(entry.get("id", MASTER_BUS)),
                    name=str(entry.get("name", entry.get("id", "Bus"))),
                    gain_db=float(entry.get("gainDb", 0.0)),
                    muted=bool(entry.get("muted", False)),
                    soloed=bool(entry.get("soloed", False)),
                )
                for entry in data.get("buses", ())
            ]
            self._buses = buses or [BusSpec(id=MASTER_BUS, name="Master")]
            self._sources = [
                SourceRouting(
                    source_id=str(entry.get("sourceId", "source")),
                    bus_id=str(entry.get("busId", MASTER_BUS)),
                    gain_db=float(entry.get("gainDb", 0.0)),
                    muted=bool(entry.get("muted", False)),
                    soloed=bool(entry.get("soloed", False)),
                )
                for entry in data.get("sources", ())
            ] or [SourceRouting(source_id="source.1")]
            bands = dict(data.get("bands", {}) or {})
            self._band = BandRouting(
                crossovers_hz=tuple(float(value) for value in bands.get("crossoversHz", ())),
                band_gains_db=tuple(float(value) for value in bands.get("bandGainsDb", ())),
                band_enabled=tuple(bool(value) for value in bands.get("bandEnabled", ())),
                order=int(bands.get("order", 4)),
            )
            index = self.order_combo.findData(self._band.order)
            if index >= 0:
                self.order_combo.setCurrentIndex(index)
            self._refresh_all()
        finally:
            self._loading = False

    # --- buses --------------------------------------------------------------

    def _add_bus(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add bus", "Bus name:")
        name = str(name).strip()
        if not accepted or not name:
            return
        identifier = name.lower().replace(" ", ".")
        if any(bus.id == identifier for bus in self._buses):
            self.status_label.setText(f"A bus called {identifier!r} already exists.")
            return
        self._buses.append(BusSpec(id=identifier, name=name))
        self._refresh_buses()
        self._refresh_sources()
        self._emit()

    def _remove_bus(self) -> None:
        row = self.bus_table.currentRow()
        if not 0 <= row < len(self._buses):
            return
        if self._buses[row].id == MASTER_BUS:
            self.status_label.setText("The master bus is where everything mixes; it stays.")
            return
        removed = self._buses.pop(row)
        # A source pointing at a bus that no longer exists would silently
        # vanish from the mix, so move those back to master instead.
        self._sources = [
            source if source.bus_id != removed.id else SourceRouting(
                source_id=source.source_id,
                bus_id=MASTER_BUS,
                gain_db=source.gain_db,
                muted=source.muted,
                soloed=source.soloed,
            )
            for source in self._sources
        ]
        self._refresh_buses()
        self._refresh_sources()
        self._emit()

    def _refresh_buses(self) -> None:
        table = self.bus_table
        table.blockSignals(True)
        reset_rows(table, len(self._buses))
        for row, bus in enumerate(self._buses):
            item = QTableWidgetItem(bus.name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip(bus.id)
            table.setItem(row, 0, item)
            place(table, row, 1, self._gain_spin(bus.gain_db, lambda value, r=row: self._set_bus(r, gain_db=value)))
            place(table, row, 2, self._check(bus.muted, lambda state, r=row: self._set_bus(r, muted=bool(state))))
            place(table, row, 3, self._check(bus.soloed, lambda state, r=row: self._set_bus(r, soloed=bool(state))))
        table.blockSignals(False)

    def _set_bus(self, row: int, **changes: Any) -> None:
        if not 0 <= row < len(self._buses):
            return
        bus = self._buses[row]
        self._buses[row] = BusSpec(
            id=bus.id,
            name=bus.name,
            gain_db=float(changes.get("gain_db", bus.gain_db)),
            muted=bool(changes.get("muted", bus.muted)),
            soloed=bool(changes.get("soloed", bus.soloed)),
        )
        self._emit()

    # --- sources ------------------------------------------------------------

    def _add_source(self) -> None:
        default = f"source.{len(self._sources) + 1}"
        name, accepted = QInputDialog.getText(self, "Add source", "Source identifier:", text=default)
        name = str(name).strip()
        if not accepted or not name:
            return
        if any(source.source_id == name for source in self._sources):
            self.status_label.setText(f"A source called {name!r} already exists.")
            return
        self._sources.append(SourceRouting(source_id=name))
        self._refresh_sources()
        self._refresh_cost()
        self._emit()

    def _remove_source(self) -> None:
        row = self.source_table.currentRow()
        if not 0 <= row < len(self._sources) or len(self._sources) <= 1:
            return
        self._sources.pop(row)
        self._refresh_sources()
        self._refresh_cost()
        self._emit()

    def _refresh_sources(self) -> None:
        table = self.source_table
        table.blockSignals(True)
        reset_rows(table, len(self._sources))
        for row, source in enumerate(self._sources):
            item = QTableWidgetItem(source.source_id)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, item)

            combo = QComboBox()
            for bus in self._buses:
                combo.addItem(bus.name, bus.id)
            index = combo.findData(source.bus_id)
            combo.setCurrentIndex(max(0, index))
            combo.currentIndexChanged.connect(
                lambda _index, r=row, c=combo: self._set_source(r, bus_id=c.currentData())
            )
            place(table, row, 1, combo)
            place(table, row, 2, self._gain_spin(source.gain_db, lambda value, r=row: self._set_source(r, gain_db=value)))
            place(table, row, 3, self._check(source.muted, lambda state, r=row: self._set_source(r, muted=bool(state))))
            place(table, row, 4, self._check(source.soloed, lambda state, r=row: self._set_source(r, soloed=bool(state))))
        table.blockSignals(False)

    def _set_source(self, row: int, **changes: Any) -> None:
        if not 0 <= row < len(self._sources):
            return
        source = self._sources[row]
        self._sources[row] = SourceRouting(
            source_id=source.source_id,
            bus_id=str(changes.get("bus_id", source.bus_id)),
            gain_db=float(changes.get("gain_db", source.gain_db)),
            muted=bool(changes.get("muted", source.muted)),
            soloed=bool(changes.get("soloed", source.soloed)),
        )
        soloed = [s.source_id for s in self._sources if s.soloed]
        if soloed:
            self.status_label.setText("Solo active: " + ", ".join(soloed))
        self._emit()

    # --- bands --------------------------------------------------------------

    def _add_crossover(self) -> None:
        frequency, accepted = QInputDialog.getDouble(
            self, "Add crossover", "Crossover frequency (Hz):", 500.0, 20.0, 20_000.0, 1
        )
        if not accepted:
            return
        self.set_crossovers((*self._band.crossovers_hz, float(frequency)))

    def _remove_crossover(self) -> None:
        if not self._band.crossovers_hz:
            return
        self.set_crossovers(tuple(sorted(self._band.crossovers_hz))[:-1])

    def set_crossovers(self, frequencies) -> None:
        ordered = tuple(sorted(float(value) for value in frequencies))
        count = len(ordered) + 1
        gains = list(self._band.band_gains_db)[:count]
        enabled = list(self._band.band_enabled)[:count]
        gains += [0.0] * (count - len(gains))
        enabled += [True] * (count - len(enabled))
        self._band = BandRouting(
            crossovers_hz=ordered,
            band_gains_db=tuple(gains),
            band_enabled=tuple(enabled),
            order=int(self.order_combo.currentData() or 4),
        )
        self._refresh_bands()
        self._refresh_cost()
        self._emit()

    def _on_band_changed(self) -> None:
        self.set_crossovers(self._band.crossovers_hz)

    def _set_band(self, band: int, **changes: Any) -> None:
        gains = list(self._band.band_gains_db)
        enabled = list(self._band.band_enabled)
        while len(gains) <= band:
            gains.append(0.0)
        while len(enabled) <= band:
            enabled.append(True)
        if "gain_db" in changes:
            gains[band] = float(changes["gain_db"])
        if "enabled" in changes:
            enabled[band] = bool(changes["enabled"])
        self._band = BandRouting(
            crossovers_hz=self._band.crossovers_hz,
            band_gains_db=tuple(gains),
            band_enabled=tuple(enabled),
            order=self._band.order,
        )
        self.band_strip.set_routing(self._band)
        self._emit()

    def _refresh_bands(self) -> None:
        self.crossover_label.setText(
            "none — wideband"
            if self._band.is_wideband
            else ", ".join(f"{value:g} Hz" for value in self._band.crossovers_hz)
        )
        self.band_strip.set_routing(self._band)

        table = self.band_table
        table.blockSignals(True)
        reset_rows(table, self._band.band_count)
        edges = [_MIN_HZ, *self._band.crossovers_hz, _MAX_HZ]
        for band in range(self._band.band_count):
            item = QTableWidgetItem(f"{edges[band]:g} – {edges[band + 1]:g} Hz")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(band, 0, item)
            gain = (
                self._band.band_gains_db[band]
                if band < len(self._band.band_gains_db)
                else 0.0
            )
            place(table, band, 1, self._gain_spin(gain, lambda value, b=band: self._set_band(b, gain_db=value)))
            place(
                table,
                band,
                2,
                self._check(self._band.enabled_for(band), lambda state, b=band: self._set_band(b, enabled=bool(state))),
            )
        table.blockSignals(False)

    # --- cost ---------------------------------------------------------------

    def cost_inputs(self) -> CostInputs:
        return CostInputs(
            sources=max(1, len(self._sources)),
            bands=max(1, self._band.band_count),
            hrir_taps=int(self.taps_spin.value()),
            interpolation=self.interpolation_combo.currentText(),
            sample_rate_hz=float(self.rate_spin.value()),
            block_size=int(self.block_spin.value()),
            renderer="hrtf",
            crossover_order=int(self.order_combo.currentData() or 4),
        )

    def _refresh_cost(self) -> None:
        estimate = estimate_cost(self.cost_inputs(), throughput_macs=self._throughput)
        verdict = (
            "fits real time" if estimate.feasible_realtime else "offline rendering only"
        )
        self.cost_label.setText(
            f"{estimate.summary()}\n{verdict} — "
            f"{estimate.inputs.convolutions} convolutions, "
            f"block budget {estimate.block_budget_s * 1000:.2f} ms\n"
            + (
                "throughput measured on this machine"
                if estimate.measured
                else "throughput not measured; a conservative fallback is in use"
            )
        )

    def measure(self) -> float:
        """Time this machine, so the estimate stops being a guess."""

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._throughput = float(measure_throughput())
        finally:
            QApplication.restoreOverrideCursor()
        self.status_label.setText(
            f"Measured {self._throughput / 1e9:.2f} GMAC/s of sustained convolution."
        )
        self._refresh_cost()
        return self._throughput

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _gain_spin(value: float, on_change) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-60.0, 12.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setValue(float(value))
        spin.setSuffix(" dB")
        spin.valueChanged.connect(on_change)
        return spin

    @staticmethod
    def _check(value: bool, on_change) -> QCheckBox:
        box = QCheckBox()
        box.setChecked(bool(value))
        box.stateChanged.connect(on_change)
        return box

    def _refresh_all(self) -> None:
        self._refresh_buses()
        self._refresh_sources()
        self._refresh_bands()
        self._refresh_cost()

    def _emit(self) -> None:
        if not self._loading:
            self.routingChanged.emit(self.describe())

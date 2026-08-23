"""The modulation matrix, drawn as a matrix.

Rows are modulators, columns are driven parameters, and a cell is a route. That
is the shape :mod:`src.audio.sam_workbench.modulation` stores, and it is the
shape the specification asks the expert view to draw, so the widget is a grid
rather than a list of routes: a list makes you read every entry to answer "what
drives this parameter", while the grid answers it by looking at a column.

Cycles are the one thing the matrix must never accept. A route is checked
*before* it is committed, and one that would close a loop is refused with the
loop spelled out - the offending cell flashes and the status line names the
chain - rather than accepted and left to fail at render time.

Targets are found by search rather than by scrolling: the parameter registry is
too large for a combo box to be a way of finding anything.
"""

from __future__ import annotations

from typing import Any, Mapping

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.modulation import (
    ModulationCycleError,
    ModulationMatrix,
    ModulationRoute,
    search_parameters,
)
from src.audio.sam_workbench.scene_state import MODULATOR_WAVEFORMS
from src.audio.sam_workbench.stages import CURVES

__all__ = ["SamModulationPanel"]

#: Modulators a scene starts with. Named rather than numbered, because a matrix
#: of "lfo1..lfo4" tells you nothing about what any row does.
DEFAULT_MODULATORS = ("lfo.slow", "lfo.fast", "envelope", "random.walk")

#: What each default modulator actually is. Before these existed every row
#: rendered as the same 1 Hz sine whatever it was called, so "random.walk" was
#: a name with no randomness behind it.
#: What a modulator with no stored definition renders as. Documented rather
#: than implicit, because it is what the engine already falls back to.
_FALLBACK_SPEC: dict[str, Any] = {
    "waveform": "sine",
    "rateHz": 1.0,
    "phaseDeg": 0.0,
    "seed": 0,
}

DEFAULT_MODULATOR_SPECS: dict[str, dict[str, Any]] = {
    "lfo.slow": {"waveform": "sine", "rateHz": 0.05, "phaseDeg": 0.0, "seed": 0},
    "lfo.fast": {"waveform": "sine", "rateHz": 1.0, "phaseDeg": 0.0, "seed": 0},
    "envelope": {"waveform": "triangle", "rateHz": 0.02, "phaseDeg": 0.0, "seed": 0},
    "random.walk": {"waveform": "random", "rateHz": 0.2, "phaseDeg": 0.0, "seed": 1},
}


def _tint(depth: float, polarity: int) -> QColor:
    """Warm for a positive route, cool for a negative one, by strength."""

    strength = min(1.0, abs(float(depth)))
    alpha = int(30 + 110 * strength)
    colour = QColor(230, 120, 40) if polarity >= 0 else QColor(60, 130, 210)
    colour.setAlpha(alpha)
    return colour


class SamModulationPanel(QWidget):
    """Edit a :class:`ModulationMatrix` as a grid of cells."""

    matrixChanged = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix = ModulationMatrix()
        self._modulators: list[str] = list(DEFAULT_MODULATORS)
        # Identifier -> the definition the engine renders. A row used to be a
        # name and nothing else, so every route compiled to the documented 1 Hz
        # sine fallback: "random.walk" modulated exactly like "lfo.slow". The
        # panel now owns the definition the render actually uses.
        self._definitions: dict[str, dict[str, Any]] = {
            name: dict(spec) for name, spec in DEFAULT_MODULATOR_SPECS.items()
        }
        #: Identifier -> display name for the track's sources, as modulation
        #: targets. Empty until a track supplies one.
        self._roster: dict[str, str] = {}
        #: Columns are (target_id, parameter_path) pairs.
        self._columns: list[tuple[str, str]] = []
        self._selected: tuple[int, int] | None = None
        self._loading = False
        self._build_ui()
        self._rebuild_grid()
        if self._modulators:
            self.modulator_list.setCurrentRow(0)
        self._show_definition()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Rows modulate columns. Pick a cell, set its depth and sign below, "
            "then Apply. A route that would make a modulator depend on itself "
            "is refused, with the loop named."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QHBoxLayout()
        body.addWidget(self._build_sources(), 0)
        body.addWidget(self._build_grid(), 1)
        layout.addLayout(body, 1)

        layout.addWidget(self._build_editor())

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_sources(self) -> QWidget:
        box = QGroupBox("Modulators and targets")
        outer = QVBoxLayout(box)

        outer.addWidget(QLabel("Modulators (rows)"))
        self.modulator_list = QListWidget()
        self.modulator_list.setMaximumWidth(230)
        self.modulator_list.currentRowChanged.connect(lambda _row: self._show_definition())
        outer.addWidget(self.modulator_list, 1)

        # What the selected row *is*, not just what it is called.
        definition = QFormLayout()
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(list(MODULATOR_WAVEFORMS))
        self.waveform_combo.setToolTip(
            "Shape this modulator traces. Random holds a seeded value and "
            "smooths between held values, so it wanders rather than clicks."
        )
        self.waveform_combo.currentTextChanged.connect(
            lambda value: self._set_definition(waveform=value)
        )
        definition.addRow("Waveform", self.waveform_combo)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.001, 100.0)
        self.rate_spin.setDecimals(3)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setToolTip("Cycles per second.")
        self.rate_spin.valueChanged.connect(
            lambda value: self._set_definition(rateHz=float(value))
        )
        definition.addRow("Rate", self.rate_spin)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-360.0, 360.0)
        self.phase_spin.setSuffix("°")
        self.phase_spin.setToolTip("Where in the cycle the render starts.")
        self.phase_spin.valueChanged.connect(
            lambda value: self._set_definition(phaseDeg=float(value))
        )
        definition.addRow("Phase", self.phase_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setToolTip(
            "Only the random waveform reads this. It is what makes an export "
            "match the preview it was approved from."
        )
        self.seed_spin.valueChanged.connect(
            lambda value: self._set_definition(seed=int(value))
        )
        definition.addRow("Seed", self.seed_spin)
        outer.addLayout(definition)

        row = QHBoxLayout()
        add_modulator = QPushButton("Add…")
        add_modulator.setToolTip("Add a modulation source row.")
        add_modulator.clicked.connect(self._add_modulator)
        row.addWidget(add_modulator)
        remove_modulator = QPushButton("Remove")
        remove_modulator.setToolTip("Remove the selected row and every route on it.")
        remove_modulator.clicked.connect(self._remove_modulator)
        row.addWidget(remove_modulator)
        outer.addLayout(row)

        outer.addWidget(QLabel("Find a target parameter"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("beat, carrier, azimuth…")
        self.search_box.setToolTip(
            "Search the SAM parameter registry by name, label, alias, or tooltip."
        )
        self.search_box.textChanged.connect(self._on_search)
        outer.addWidget(self.search_box)

        self.search_results = QListWidget()
        self.search_results.setMaximumWidth(230)
        self.search_results.setToolTip("Double-click a parameter to add it as a column.")
        self.search_results.itemDoubleClicked.connect(self._add_searched_target)
        outer.addWidget(self.search_results, 1)

        target_row = QHBoxLayout()
        add_target = QPushButton("Add column")
        add_target.clicked.connect(self._add_searched_target)
        target_row.addWidget(add_target)
        remove_target = QPushButton("Remove column")
        remove_target.clicked.connect(self._remove_column)
        target_row.addWidget(remove_target)
        outer.addLayout(target_row)
        return box

    def _build_grid(self) -> QWidget:
        box = QGroupBox("Matrix")
        outer = QVBoxLayout(box)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        # Columns size to their parameter names rather than stretching: a
        # clipped name ("patialBeatFreq") names a different parameter, and the
        # whole point of a column is knowing what it drives.
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setMinimumSectionSize(90)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_cell_selected)
        outer.addWidget(self.table, 1)
        return box

    def _build_editor(self) -> QWidget:
        box = QGroupBox("Selected route")
        outer = QHBoxLayout(box)

        form = QFormLayout()
        self.cell_label = QLabel("no cell selected")
        self.cell_label.setWordWrap(True)
        form.addRow("Cell:", self.cell_label)

        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(-10.0, 10.0)
        self.depth_spin.setDecimals(3)
        self.depth_spin.setSingleStep(0.05)
        self.depth_spin.setToolTip(
            "How far this modulator moves the target, in the target's own units, "
            "for a modulator value of one."
        )
        form.addRow("Depth:", self.depth_spin)
        outer.addLayout(form)

        second = QFormLayout()
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItem("positive (+)", 1)
        self.polarity_combo.addItem("inverted (−)", -1)
        self.polarity_combo.setToolTip(
            "Sign, kept separate from depth so the matrix shows direction and "
            "amount as two readable things."
        )
        second.addRow("Polarity:", self.polarity_combo)

        self.curve_combo = QComboBox()
        self.curve_combo.addItems(list(CURVES))
        self.curve_combo.setCurrentText("linear")
        self.curve_combo.setToolTip("How the modulator's 0-to-1 value is shaped before scaling.")
        second.addRow("Curve:", self.curve_combo)

        self.enabled_check = QCheckBox("enabled")
        self.enabled_check.setChecked(True)
        self.enabled_check.setToolTip(
            "A disabled route is kept but carries nothing, so it can be parked "
            "without deleting it."
        )
        second.addRow("", self.enabled_check)
        outer.addLayout(second)

        buttons = QVBoxLayout()
        self.apply_button = QPushButton("Apply to cell")
        self.apply_button.clicked.connect(self.apply_cell)
        buttons.addWidget(self.apply_button)
        self.clear_button = QPushButton("Clear cell")
        self.clear_button.clicked.connect(self.clear_cell)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        return box

    # --- state --------------------------------------------------------------

    @property
    def matrix(self) -> ModulationMatrix:
        return self._matrix

    def set_matrix(self, matrix: ModulationMatrix | Mapping[str, Any] | None) -> None:
        if isinstance(matrix, ModulationMatrix):
            self._matrix = matrix
        else:
            self._matrix = ModulationMatrix.from_mapping(matrix)
        for route in self._matrix.routes:
            if route.modulator_id not in self._modulators:
                self._modulators.append(route.modulator_id)
            column = (route.target_id, route.parameter_path)
            if column not in self._columns:
                self._columns.append(column)
        self._loading = True
        try:
            self._rebuild_grid()
        finally:
            self._loading = False

    def params(self) -> dict[str, Any]:
        """The serialized matrix, under the workbench's own key."""

        return {"samModulation": self._matrix.describe()}

    def set_params(self, params: Mapping[str, Any]) -> None:
        self.set_matrix(params.get("samModulation"))

    # --- modulator definitions ----------------------------------------------

    def modulators(self) -> tuple[dict[str, Any], ...]:
        """The definitions the engine renders, in row order.

        This is what belongs in ``scene["modulators"]``. Returning only the
        names is what left every row compiling to the same fallback sine.
        """

        return tuple(
            {"id": name, **self._definitions.get(name, dict(_FALLBACK_SPEC))}
            for name in self._modulators
        )

    def set_modulators(self, definitions) -> None:
        """Adopt stored modulator definitions, keeping any row order they imply."""

        names: list[str] = []
        specs: dict[str, dict[str, Any]] = {}
        for entry in definitions or ():
            if not isinstance(entry, Mapping):
                continue
            identifier = str(entry.get("id", "")).strip()
            if not identifier:
                continue
            spec = {key: entry[key] for key in ("waveform", "rateHz", "phaseDeg", "seed") if key in entry}
            specs[identifier] = {**_FALLBACK_SPEC, **spec}
            names.append(identifier)
        if not names:
            return
        for name in self._modulators:
            if name not in specs:
                names.append(name)
                specs[name] = self._definitions.get(name, dict(_FALLBACK_SPEC))
        self._modulators = names
        self._definitions = specs
        self._rebuild_grid()
        self._show_definition()

    def _current_modulator(self) -> str:
        row = self.modulator_list.currentRow()
        if 0 <= row < len(self._modulators):
            return self._modulators[row]
        return ""

    def _show_definition(self) -> None:
        """Put the selected row's definition into the controls."""

        name = self._current_modulator()
        spec = self._definitions.get(name, dict(_FALLBACK_SPEC))
        enabled = bool(name)
        for widget in (self.waveform_combo, self.rate_spin, self.phase_spin, self.seed_spin):
            widget.setEnabled(enabled)
            widget.blockSignals(True)
        try:
            self.waveform_combo.setCurrentText(str(spec.get("waveform", "sine")))
            self.rate_spin.setValue(float(spec.get("rateHz", 1.0)))
            self.phase_spin.setValue(float(spec.get("phaseDeg", 0.0)))
            self.seed_spin.setValue(int(spec.get("seed", 0)))
            # Only the random waveform reads a seed; showing it live for the
            # others would suggest it does something there.
            self.seed_spin.setEnabled(enabled and spec.get("waveform") == "random")
        finally:
            for widget in (self.waveform_combo, self.rate_spin, self.phase_spin, self.seed_spin):
                widget.blockSignals(False)

    def _set_definition(self, **changes: Any) -> None:
        name = self._current_modulator()
        if not name or self._loading:
            return
        spec = dict(self._definitions.get(name, dict(_FALLBACK_SPEC)))
        spec.update(changes)
        self._definitions[name] = spec
        if "waveform" in changes:
            self.seed_spin.setEnabled(changes["waveform"] == "random")
        self._emit()

    # --- targets from the track ---------------------------------------------

    def set_roster(self, sources) -> None:
        """Adopt the track's sources as available modulation targets."""

        roster: dict[str, str] = {}
        for entry in sources or ():
            if isinstance(entry, Mapping):
                identifier = str(entry.get("id", "")).strip()
                name = str(entry.get("name", "") or identifier)
            else:
                identifier = str(entry).strip()
                name = identifier
            if identifier:
                roster.setdefault(identifier, name)
        self._roster = roster

    def roster(self) -> dict[str, str]:
        return dict(self._roster)

    def unresolved_targets(self) -> tuple[str, ...]:
        """Routed target identifiers the track does not have.

        ``voice`` is the voice being edited rather than a scene source, so it
        is never counted as dangling.
        """

        if not self._roster:
            return ()
        return tuple(
            sorted(
                {
                    target
                    for target, _path in self._columns
                    if target != "voice"
                    and not target.startswith("lfo.")
                    and target not in self._roster
                    and target not in self._modulators
                }
            )
        )

    # --- rows and columns ---------------------------------------------------

    def add_modulator(self, name: str) -> None:
        name = str(name).strip()
        if name and name not in self._modulators:
            self._modulators.append(name)
            self._definitions.setdefault(name, dict(_FALLBACK_SPEC))
            self._rebuild_grid()
            self._show_definition()

    def _add_modulator(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "Add modulator", "Modulator identifier (for example lfo.slow):"
        )
        if accepted:
            self.add_modulator(name)

    def _remove_modulator(self) -> None:
        row = self.modulator_list.currentRow()
        if not 0 <= row < len(self._modulators):
            return
        name = self._modulators.pop(row)
        self._definitions.pop(name, None)
        self._matrix = ModulationMatrix(
            routes=tuple(r for r in self._matrix.routes if r.modulator_id != name),
            schema_version=self._matrix.schema_version,
        )
        self._rebuild_grid()
        self._emit()

    def add_target(self, target_id: str, parameter_path: str) -> None:
        column = (str(target_id), str(parameter_path))
        if column[0] and column[1] and column not in self._columns:
            self._columns.append(column)
            self._rebuild_grid()

    def _on_search(self, text: str) -> None:
        self.search_results.clear()
        for entry in search_parameters(text, limit=25):
            label = getattr(entry, "label", "") or entry.name
            item = QListWidgetItem(f"{label}  ·  {entry.name}")
            item.setData(Qt.UserRole, entry.name)
            item.setToolTip(str(getattr(entry, "tooltip", "")) or entry.name)
            self.search_results.addItem(item)

    def _add_searched_target(self, *_args: Any) -> None:
        item = self.search_results.currentItem()
        if item is None:
            self.status_label.setText("Search for a parameter, then choose one to add as a column.")
            return
        self.add_target("voice", str(item.data(Qt.UserRole)))

    def _remove_column(self) -> None:
        column = self.table.currentColumn()
        if not 0 <= column < len(self._columns):
            return
        target_id, path = self._columns.pop(column)
        self._matrix = ModulationMatrix(
            routes=tuple(
                r
                for r in self._matrix.routes
                if not (r.target_id == target_id and r.parameter_path == path)
            ),
            schema_version=self._matrix.schema_version,
        )
        self._rebuild_grid()
        self._emit()

    # --- grid ---------------------------------------------------------------

    def _rebuild_grid(self) -> None:
        self.modulator_list.clear()
        self.modulator_list.addItems(self._modulators)

        self.table.blockSignals(True)
        self.table.clear()
        self.table.setRowCount(len(self._modulators))
        self.table.setColumnCount(len(self._columns))
        self.table.setVerticalHeaderLabels(self._modulators)
        self.table.setHorizontalHeaderLabels(
            [f"{target}\n{path}" for target, path in self._columns]
        )
        for row in range(len(self._modulators)):
            for column in range(len(self._columns)):
                self.table.setItem(row, column, self._cell_item(row, column))
        self.table.blockSignals(False)
        self._refresh_editor()

    def _cell_item(self, row: int, column: int) -> QTableWidgetItem:
        route = self._route_at(row, column)
        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)
        if route is None:
            item.setText("·")
            item.setToolTip("No route. Set a depth below and Apply to create one.")
            return item
        sign = "+" if route.polarity >= 0 else "−"
        item.setText(f"{sign}{abs(route.depth):.2f}\n{route.curve}")
        if route.enabled:
            item.setBackground(_tint(route.depth, route.polarity))
        else:
            item.setText(item.text() + "\n(off)")
        item.setToolTip(
            f"{route.modulator_id} → {route.target_id}.{route.parameter_path}\n"
            f"depth {route.depth:g}, polarity {route.polarity:+d}, curve {route.curve}, "
            f"{'enabled' if route.enabled else 'disabled'}"
        )
        return item

    def _route_at(self, row: int, column: int) -> ModulationRoute | None:
        if not (0 <= row < len(self._modulators) and 0 <= column < len(self._columns)):
            return None
        target_id, path = self._columns[column]
        return self._matrix.route(self._modulators[row], target_id, path)

    def _on_cell_selected(self) -> None:
        row = self.table.currentRow()
        column = self.table.currentColumn()
        self._selected = (row, column) if row >= 0 and column >= 0 else None
        self._refresh_editor()

    def _refresh_editor(self) -> None:
        if self._selected is None:
            self.cell_label.setText("no cell selected")
            self.apply_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            return
        row, column = self._selected
        if not (0 <= row < len(self._modulators) and 0 <= column < len(self._columns)):
            self._selected = None
            self._refresh_editor()
            return
        target_id, path = self._columns[column]
        self.cell_label.setText(f"{self._modulators[row]} → {target_id}.{path}")
        self.apply_button.setEnabled(True)
        route = self._route_at(row, column)
        self.clear_button.setEnabled(route is not None)
        if route is not None:
            self.depth_spin.setValue(float(route.depth))
            self.polarity_combo.setCurrentIndex(0 if route.polarity >= 0 else 1)
            self.curve_combo.setCurrentText(route.curve)
            self.enabled_check.setChecked(route.enabled)

    # --- commit -------------------------------------------------------------

    def apply_cell(self) -> bool:
        """Commit the editor onto the selected cell. False when refused."""

        if self._selected is None:
            return False
        row, column = self._selected
        target_id, path = self._columns[column]
        route = ModulationRoute(
            modulator_id=self._modulators[row],
            target_id=target_id,
            parameter_path=path,
            depth=float(self.depth_spin.value()),
            polarity=int(self.polarity_combo.currentData()),
            curve=self.curve_combo.currentText(),
            enabled=bool(self.enabled_check.isChecked()),
        )
        try:
            self._matrix = self._matrix.with_route(route)
        except ModulationCycleError as error:
            # Refused before it is stored, and the loop is named: a cycle is
            # invisible cell by cell, so saying "invalid" would not help.
            self.status_label.setText(
                "Refused: this route would close a modulation loop "
                + " → ".join(error.cycle)
            )
            item = self.table.item(row, column)
            if item is not None:
                item.setBackground(QColor(200, 60, 60, 120))
            return False

        signed = route.depth * route.polarity
        self.status_label.setText(
            f"{route.modulator_id} → {route.target_id}.{route.parameter_path} "
            f"at {signed:+g} ({route.curve})"
            f"{'' if route.enabled else ', disabled'}"
        )
        self.table.setItem(row, column, self._cell_item(row, column))
        self._emit()
        return True

    def clear_cell(self) -> None:
        if self._selected is None:
            return
        row, column = self._selected
        target_id, path = self._columns[column]
        self._matrix = self._matrix.without_route(self._modulators[row], target_id, path)
        self.table.setItem(row, column, self._cell_item(row, column))
        self.status_label.setText("Route removed.")
        self._refresh_editor()
        self._emit()

    def _emit(self) -> None:
        if not self._loading:
            self.matrixChanged.emit(self._matrix.describe())

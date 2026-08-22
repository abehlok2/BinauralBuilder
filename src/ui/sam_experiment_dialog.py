"""A listening experiment: conditions in, blinded presentations, an export out.

Where the localization test asks "where was it", this asks "how was it" - the
subjective comparison that decides between two renderings that are both
plausible. The same discipline applies: the listener sees a scrambled label,
never the condition name, and the reveal lives in the export rather than in the
runner.

A condition records enough to reproduce the rendering it stands for. That is
what :meth:`Condition.is_reproducible` checks, and a design containing
conditions that fail it is allowed but flagged: an experiment whose winning
condition cannot be rebuilt afterwards has not settled anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.experiment.session import (
    Condition,
    ConditionRun,
    ExperimentDesign,
    ExperimentResult,
    Rating,
    conditions_from_assets,
    export_result,
)

__all__ = ["SamExperimentDialog"]

DEFAULT_CRITERIA = ("externalization", "front_back_accuracy", "preference")


class SamExperimentDialog(QDialog):
    """Build a design, run it blinded, and export the result."""

    resultExported = pyqtSignal(str)

    def __init__(
        self,
        assets: Sequence[Mapping[str, Any]] = (),
        *,
        play_condition=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Listening experiment")
        self.resize(820, 660)
        if parent is not None:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        self._play_condition = play_condition
        self._conditions: list[Condition] = list(conditions_from_assets(assets)) if assets else []
        self._design: ExperimentDesign | None = None
        self._runs: tuple[ConditionRun, ...] = ()
        self._ratings: list[Rating] = []
        self._index = 0
        self._result: ExperimentResult | None = None

        self._build_ui()
        self._refresh_conditions()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_design())
        self.pages.addWidget(self._build_runner())
        self.pages.addWidget(self._build_results())
        layout.addWidget(self.pages, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _build_design(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("Conditions")
        box_layout = QVBoxLayout(box)
        self.condition_table = QTableWidget(0, 5)
        self.condition_table.setHorizontalHeaderLabels(
            ["Condition", "Renderer", "Asset", "Interpolation", "Reproducible"]
        )
        self.condition_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.condition_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.condition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        box_layout.addWidget(self.condition_table, 1)

        row = QHBoxLayout()
        add_file = QPushButton("Add SOFA condition…")
        add_file.clicked.connect(self._add_condition_from_file)
        row.addWidget(add_file)
        add_named = QPushButton("Add named condition…")
        add_named.setToolTip("For a rendering not backed by a SOFA file, such as the abstract renderer.")
        add_named.clicked.connect(self._add_named_condition)
        row.addWidget(add_named)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_condition)
        row.addWidget(remove)
        row.addStretch(1)
        box_layout.addLayout(row)
        outer.addWidget(box, 1)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("what this experiment is asking")
        form.addRow("Name:", self.name_edit)

        self.listener_edit = QLineEdit()
        self.listener_edit.setPlaceholderText("listener identifier")
        form.addRow("Listener:", self.listener_edit)

        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 10)
        self.repeats_spin.setValue(2)
        self.repeats_spin.setToolTip("How many times each condition is presented.")
        form.addRow("Repeats:", self.repeats_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setToolTip("Fixes the presentation order and the blinded labels.")
        form.addRow("Seed:", self.seed_spin)

        self.blinded_check = QCheckBox("hide which condition is which until export")
        self.blinded_check.setChecked(True)
        form.addRow("", self.blinded_check)
        outer.addLayout(form)

        criteria = QGroupBox("Criteria")
        criteria_layout = QVBoxLayout(criteria)
        self.criteria_list = QListWidget()
        self.criteria_list.addItems(list(DEFAULT_CRITERIA))
        self.criteria_list.setMaximumHeight(110)
        criteria_layout.addWidget(self.criteria_list)
        criteria_row = QHBoxLayout()
        add_criterion = QPushButton("Add…")
        add_criterion.clicked.connect(self._add_criterion)
        criteria_row.addWidget(add_criterion)
        remove_criterion = QPushButton("Remove")
        remove_criterion.clicked.connect(self._remove_criterion)
        criteria_row.addWidget(remove_criterion)
        criteria_row.addStretch(1)
        criteria_layout.addLayout(criteria_row)
        outer.addWidget(criteria)

        start_row = QHBoxLayout()
        start_row.addStretch(1)
        self.start_button = QPushButton("Start experiment")
        self.start_button.clicked.connect(self.start)
        start_row.addWidget(self.start_button)
        outer.addLayout(start_row)
        return page

    def _build_runner(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        header = QHBoxLayout()
        self.run_label = QLabel("")
        header.addWidget(self.run_label, 1)
        self.progress = QProgressBar()
        header.addWidget(self.progress, 1)
        outer.addLayout(header)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_current)
        outer.addWidget(self.play_button)

        self.criteria_box = QGroupBox("Rate this presentation")
        self.criteria_form = QFormLayout(self.criteria_box)
        outer.addWidget(self.criteria_box)

        outer.addWidget(QLabel("Comment:"))
        self.comment_edit = QTextEdit()
        self.comment_edit.setMaximumHeight(80)
        outer.addWidget(self.comment_edit)

        row = QHBoxLayout()
        row.addStretch(1)
        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_current)
        row.addWidget(self.skip_button)
        self.next_button = QPushButton("Record and continue")
        self.next_button.clicked.connect(self.record_current)
        row.addWidget(self.next_button)
        outer.addLayout(row)
        outer.addStretch(1)
        return page

    def _build_results(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("Mean score per criterion. The export carries every individual response."))

        self.results_table = QTableWidget(0, 0)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        outer.addWidget(self.results_table, 1)

        row = QHBoxLayout()
        self.export_button = QPushButton("Export result…")
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        again = QPushButton("Back to design")
        again.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        row.addWidget(again)
        row.addStretch(1)
        outer.addLayout(row)
        return page

    # --- conditions ---------------------------------------------------------

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return tuple(self._conditions)

    def add_condition(self, condition: Condition) -> None:
        self._conditions.append(condition)
        self._refresh_conditions()

    def _add_condition_from_file(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "Add SOFA conditions", "", "SOFA files (*.sofa);;All files (*)"
        )
        for path in paths:
            name = Path(path).stem
            self.add_condition(
                Condition(id=name, name=name, renderer="hrtf", asset_path=str(path), subject=name)
            )

    def _add_named_condition(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add condition", "Condition name:")
        name = str(name).strip()
        if accepted and name:
            self.add_condition(Condition(id=name, name=name, renderer="abstract_pm"))

    def _remove_condition(self) -> None:
        row = self.condition_table.currentRow()
        if 0 <= row < len(self._conditions):
            self._conditions.pop(row)
            self._refresh_conditions()

    def _refresh_conditions(self) -> None:
        table = self.condition_table
        table.setRowCount(len(self._conditions))
        for row, condition in enumerate(self._conditions):
            values = (
                condition.name or condition.id,
                condition.renderer,
                Path(condition.asset_path).name if condition.asset_path else "—",
                condition.interpolation,
                "yes" if condition.is_reproducible else "not fully",
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                if column == 4 and text != "yes":
                    item.setToolTip(
                        "This condition does not record enough to rebuild the "
                        "rendering later. The experiment can still run; the "
                        "export will say so."
                    )
                table.setItem(row, column, item)
        self.start_button.setEnabled(len(self._conditions) >= 2)

    def _add_criterion(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add criterion", "Criterion:")
        name = str(name).strip()
        if accepted and name:
            self.criteria_list.addItem(name)

    def _remove_criterion(self) -> None:
        row = self.criteria_list.currentRow()
        if row >= 0 and self.criteria_list.count() > 1:
            self.criteria_list.takeItem(row)

    def criteria(self) -> tuple[str, ...]:
        return tuple(
            self.criteria_list.item(row).text() for row in range(self.criteria_list.count())
        )

    # --- running ------------------------------------------------------------

    def start(self) -> bool:
        if len(self._conditions) < 2:
            self.status_label.setText("An experiment needs at least two conditions.")
            return False
        self._design = ExperimentDesign(
            conditions=tuple(self._conditions),
            repeats=int(self.repeats_spin.value()),
            seed=int(self.seed_spin.value()),
            blinded=bool(self.blinded_check.isChecked()),
            criteria=self.criteria(),
            name=self.name_edit.text(),
        )
        self._runs = self._design.build_runs()
        self._ratings = []
        self._index = 0
        self.progress.setRange(0, len(self._runs))
        self._build_criteria_sliders()
        self.pages.setCurrentIndex(1)
        self._load_run()

        unreproducible = self._design.unreproducible()
        self.status_label.setText(
            f"{len(self._runs)} presentations."
            + (
                " Not fully reproducible: " + ", ".join(unreproducible)
                if unreproducible
                else ""
            )
        )
        return True

    def _build_criteria_sliders(self) -> None:
        while self.criteria_form.rowCount():
            self.criteria_form.removeRow(0)
        self.sliders: dict[str, QSlider] = {}
        for criterion in self.criteria():
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(25)
            self.sliders[criterion] = slider
            self.criteria_form.addRow(criterion.replace("_", " ") + ":", slider)

    def current_run(self) -> ConditionRun | None:
        if self._index >= len(self._runs):
            return None
        return self._runs[self._index]

    def _load_run(self) -> None:
        run = self.current_run()
        if run is None:
            self.finish()
            return
        # Blinded runs show the scrambled label; an open run may name the
        # condition, which is the only difference the flag makes to the runner.
        identity = run.label
        if self._design is not None and not self._design.blinded:
            condition = self._design.condition(run.condition_id)
            identity = condition.name or condition.id if condition else run.condition_id
        self.run_label.setText(f"Presentation {self._index + 1} of {len(self._runs)} — {identity}")
        self.progress.setValue(self._index)
        self.comment_edit.clear()
        for slider in getattr(self, "sliders", {}).values():
            slider.setValue(50)
        self.play_button.setEnabled(self._play_condition is not None)
        if self._play_condition is None:
            self.play_button.setText("Play (no renderer connected)")
        self.play_current()

    def play_current(self) -> None:
        run = self.current_run()
        if run is None or self._play_condition is None or self._design is None:
            return
        condition = self._design.condition(run.condition_id)
        if condition is None:
            return
        try:
            self._play_condition(condition)
        except Exception as error:  # pragma: no cover - surfaced, not swallowed
            self.status_label.setText(f"Could not present this condition: {error}")

    def record_current(self) -> None:
        run = self.current_run()
        if run is None:
            return
        self._ratings.append(
            Rating(
                run_index=run.index,
                scores={name: float(slider.value()) for name, slider in self.sliders.items()},
                comment=self.comment_edit.toPlainText(),
            )
        )
        self._advance()

    def skip_current(self) -> None:
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self._runs):
            self.finish()
        else:
            self._load_run()

    # --- results ------------------------------------------------------------

    def finish(self) -> ExperimentResult | None:
        if self._design is None:
            return None
        self._result = ExperimentResult(
            design=self._design,
            runs=self._runs,
            ratings=tuple(self._ratings),
            listener=self.listener_edit.text(),
        )
        aggregate = self._result.aggregate()
        criteria = self.criteria()
        self.results_table.setColumnCount(1 + len(criteria))
        self.results_table.setHorizontalHeaderLabels(
            ["Condition", *[name.replace("_", " ") for name in criteria]]
        )
        self.results_table.setRowCount(len(aggregate))
        for row, (condition_id, scores) in enumerate(sorted(aggregate.items())):
            self.results_table.setItem(row, 0, QTableWidgetItem(condition_id))
            for column, criterion in enumerate(criteria, start=1):
                value = scores.get(criterion)
                self.results_table.setItem(
                    row, column, QTableWidgetItem("—" if value is None else f"{value:.1f}")
                )
        self.progress.setValue(len(self._runs))
        self.pages.setCurrentIndex(2)
        self.status_label.setText(
            f"{self._result.completion * 100:.0f}% of presentations answered."
        )
        return self._result

    @property
    def result(self) -> ExperimentResult | None:
        return self._result

    def export(self, path: str | None = None, *, reveal: bool = True) -> Path | None:
        if self._result is None:
            self.status_label.setText("Nothing to export yet.")
            return None
        if path is None:
            path, _filter = QFileDialog.getSaveFileName(
                self, "Export experiment result", "experiment.json", "JSON (*.json)"
            )
        if not path:
            return None
        destination = export_result(self._result, path, reveal=reveal)
        self.status_label.setText(f"Written to {destination}")
        self.resultExported.emit(str(destination))
        return destination

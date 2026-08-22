"""Stages drawn along a time axis, over the steps that already exist.

The specification is careful about this: a stage is a step, or a contiguous run
of steps, given a name. It is a *view* of the session, not a second lane editor
competing with the step list, so this panel builds its stages from the steps
and never invents timing of its own. Change a step's duration and the stage
view follows; it cannot disagree with the step list about when anything
happens.

The timeline is painted rather than tabulated because the questions people ask
of it are spatial: does this stage overlap the next one, is the crossfade long
enough to hide the change, where is the gap. A table answers none of those at a
glance, and the trapezoid edges show the transitions as what they are.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.stages import (
    CURVES,
    ParameterBinding,
    StageConfig,
    Timeline,
    stages_from_steps,
)

from .table_widgets import place, reset_rows

__all__ = ["SamStagePanel", "StageTimelineView"]


class StageTimelineView(QWidget):
    """Stage blocks along a time axis, with their transitions drawn."""

    stageClicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._timeline = Timeline()
        self._selected = -1
        self.setMinimumHeight(120)
        # Spare vertical space is worth more to the timeline than to an empty
        # group box below it, up to the point where taller stops helping.
        self.setMaximumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click a stage to edit it. Sloped edges are its transitions.")

    def set_timeline(self, timeline: Timeline) -> None:
        self._timeline = timeline
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = int(index)
        self.update()

    # --- geometry -----------------------------------------------------------

    def _span(self) -> tuple[float, float]:
        duration = max(1e-6, float(self._timeline.duration_s))
        return 0.0, duration

    def _x_for(self, seconds: float) -> float:
        start, end = self._span()
        left, width = 6.0, max(1.0, self.width() - 12.0)
        return left + width * (float(seconds) - start) / (end - start)

    def _index_at(self, x: float) -> int:
        for index, stage in enumerate(self._timeline.stages):
            if self._x_for(stage.start_s) <= x <= self._x_for(stage.end_s):
                return index
        return -1

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        index = self._index_at(float(event.pos().x()))
        if index >= 0:
            self.set_selected(index)
            self.stageClicked.emit(index)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        foreground = self.palette().color(self.palette().WindowText)

        top, height = 8.0, max(10.0, self.height() - 34.0)
        base = top + height

        axis = QColor(foreground)
        axis.setAlpha(90)
        painter.setPen(QPen(axis, 1))
        painter.drawLine(QPointF(6.0, base), QPointF(self.width() - 6.0, base))

        if not self._timeline.stages:
            painter.setPen(QPen(QColor(foreground.red(), foreground.green(), foreground.blue(), 150), 1))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No stages yet — build them from the session's steps below.",
            )
            painter.end()
            return

        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)

        for index, stage in enumerate(self._timeline.stages):
            left = self._x_for(stage.start_s)
            right = self._x_for(stage.end_s)
            fade_in = self._x_for(stage.start_s + stage.transition_in_s) - left
            fade_out = right - self._x_for(stage.end_s - stage.transition_out_s)

            # A trapezoid: the sloped edges *are* the transitions, at the same
            # scale as the stage, so a crossfade that is too short looks it.
            shape = QPolygonF(
                [
                    QPointF(left, base),
                    QPointF(left + max(0.0, fade_in), top),
                    QPointF(right - max(0.0, fade_out), top),
                    QPointF(right, base),
                ]
            )
            selected = index == self._selected
            colour = QColor.fromHsv((28 + 47 * index) % 360, 140, 210)
            colour.setAlpha(170 if selected else 95)
            painter.setBrush(colour)
            painter.setPen(QPen(QColor(230, 120, 40) if selected else axis, 2 if selected else 1))
            painter.drawPolygon(shape)

            label = QColor(foreground)
            label.setAlpha(220)
            painter.setPen(QPen(label, 1))
            painter.drawText(
                QRectF(left, top, max(1.0, right - left), height),
                Qt.AlignCenter | Qt.TextWordWrap,
                stage.name or stage.id,
            )

        painter.setPen(QPen(axis, 1))
        _, duration = self._span()
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            seconds = duration * fraction
            x = self._x_for(seconds)
            # The first and last labels sit on the widget edge, where a centred
            # text box would be half outside it and the number clipped.
            box = QRectF(x - 30.0, base + 2.0, 60.0, 16.0)
            box.moveLeft(min(max(box.left(), 0.0), self.width() - box.width()))
            painter.drawText(box, Qt.AlignCenter, f"{seconds:.1f} s")
        painter.end()


class SamStagePanel(QWidget):
    """Build and edit a stage timeline over an existing step list."""

    timelineChanged = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._timeline = Timeline()
        self._steps: list[Mapping[str, Any]] = []
        self._selected = -1
        self._loading = False
        self._build_ui()
        self._refresh()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "A stage is one step, or a run of consecutive steps, given a name. "
            "Timing comes from the steps themselves, so this view and the step "
            "list cannot disagree."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.timeline_view = StageTimelineView()
        self.timeline_view.stageClicked.connect(self._select_stage)
        layout.addWidget(self.timeline_view, 1)

        build_row = QHBoxLayout()
        build_row.addWidget(QLabel("Steps per stage:"))
        self.group_spin = QSpinBox()
        self.group_spin.setRange(1, 64)
        self.group_spin.setValue(1)
        self.group_spin.setToolTip("Group this many consecutive steps into each stage.")
        build_row.addWidget(self.group_spin)

        build_row.addWidget(QLabel("Crossfade:"))
        self.transition_spin = QDoubleSpinBox()
        self.transition_spin.setRange(0.0, 120.0)
        self.transition_spin.setDecimals(2)
        self.transition_spin.setSuffix(" s")
        self.transition_spin.setValue(2.0)
        build_row.addWidget(self.transition_spin)

        self.build_button = QPushButton("Build stages from steps")
        self.build_button.clicked.connect(self.build_from_steps)
        build_row.addWidget(self.build_button)
        self.step_label = QLabel("no steps loaded")
        build_row.addWidget(self.step_label, 1)
        layout.addLayout(build_row)

        # The editor keeps its natural height; the spare space belongs to the
        # timeline above it rather than to an acre of empty group box.
        layout.addWidget(self._build_editor())
        layout.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_editor(self) -> QWidget:
        box = QGroupBox("Selected stage")
        outer = QHBoxLayout(box)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._apply_editor)
        form.addRow("Name:", self.name_edit)

        self.start_spin = self._seconds_spin("When this stage begins.")
        self.start_spin.editingFinished.connect(self._apply_editor)
        form.addRow("Start:", self.start_spin)

        self.duration_spin = self._seconds_spin("How long it lasts.")
        self.duration_spin.editingFinished.connect(self._apply_editor)
        form.addRow("Duration:", self.duration_spin)
        outer.addLayout(form)

        second = QFormLayout()
        self.fade_in_spin = self._seconds_spin("Crossfade at the start of the stage.")
        self.fade_in_spin.editingFinished.connect(self._apply_editor)
        second.addRow("Transition in:", self.fade_in_spin)

        self.fade_out_spin = self._seconds_spin("Crossfade at the end of the stage.")
        self.fade_out_spin.editingFinished.connect(self._apply_editor)
        second.addRow("Transition out:", self.fade_out_spin)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("what this stage is for")
        self.notes_edit.editingFinished.connect(self._apply_editor)
        second.addRow("Notes:", self.notes_edit)
        outer.addLayout(second)

        overrides = QVBoxLayout()
        overrides.addWidget(QLabel("Parameter overrides"))
        self.override_table = QTableWidget(0, 3)
        self.override_table.setHorizontalHeaderLabels(["Parameter", "Value", "Curve"])
        self.override_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.override_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.override_table.setMinimumHeight(120)
        overrides.addWidget(self.override_table, 1)

        row = QHBoxLayout()
        add = QPushButton("Add override…")
        add.clicked.connect(self._add_override)
        row.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_override)
        row.addWidget(remove)
        overrides.addLayout(row)
        outer.addLayout(overrides, 1)
        return box

    @staticmethod
    def _seconds_spin(tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 86_400.0)
        spin.setDecimals(2)
        spin.setSuffix(" s")
        spin.setToolTip(tooltip)
        return spin

    # --- state --------------------------------------------------------------

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    def set_steps(self, steps: Sequence[Mapping[str, Any]]) -> None:
        """Give the panel the session's steps; stages are built from these."""

        self._steps = [dict(step) for step in steps]
        total = sum(float(step.get("duration", 0.0) or 0.0) for step in self._steps)
        self.step_label.setText(
            f"{len(self._steps)} step(s), {total:.1f} s total"
            if self._steps
            else "no steps loaded"
        )
        self.build_button.setEnabled(bool(self._steps))

    def set_timeline(self, timeline: Timeline | Mapping[str, Any] | None) -> None:
        self._timeline = (
            timeline if isinstance(timeline, Timeline) else Timeline.from_mapping(timeline)
        )
        self._selected = 0 if self._timeline.stages else -1
        self._loading = True
        try:
            self._refresh()
        finally:
            self._loading = False

    def params(self) -> dict[str, Any]:
        return {"samStages": self._timeline.describe()}

    def set_params(self, params: Mapping[str, Any]) -> None:
        self.set_timeline(params.get("samStages"))

    # --- building -----------------------------------------------------------

    def build_from_steps(self) -> Timeline:
        """Group the steps into stages. The steps stay the source of truth."""

        if not self._steps:
            self.status_label.setText("No steps to build from.")
            return self._timeline
        size = max(1, int(self.group_spin.value()))
        groups = [
            tuple(range(start, min(start + size, len(self._steps))))
            for start in range(0, len(self._steps), size)
        ]
        self._timeline = stages_from_steps(
            self._steps, groups=groups, transition_s=float(self.transition_spin.value())
        )
        self._selected = 0 if self._timeline.stages else -1
        self.status_label.setText(
            f"Built {len(self._timeline.stages)} stage(s) covering "
            f"{self._timeline.duration_s:.1f} s."
        )
        self._refresh()
        self._emit()
        return self._timeline

    # --- editing ------------------------------------------------------------

    def _select_stage(self, index: int) -> None:
        self._selected = int(index)
        self._load_editor()

    def _current(self) -> StageConfig | None:
        if 0 <= self._selected < len(self._timeline.stages):
            return self._timeline.stages[self._selected]
        return None

    def _replace_current(self, stage: StageConfig) -> None:
        stages = list(self._timeline.stages)
        stages[self._selected] = stage
        self._timeline = Timeline(
            stages=tuple(stages), schema_version=self._timeline.schema_version
        )
        self._refresh()
        self._emit()

    def _apply_editor(self) -> None:
        stage = self._current()
        if stage is None or self._loading:
            return
        updated = StageConfig(
            id=stage.id,
            name=self.name_edit.text(),
            start_s=float(self.start_spin.value()),
            duration_s=float(self.duration_spin.value()),
            transition_in_s=float(self.fade_in_spin.value()),
            transition_out_s=float(self.fade_out_spin.value()),
            parameter_overrides=stage.parameter_overrides,
            notes=self.notes_edit.text(),
        )
        if updated.describe() == stage.describe():
            return
        self._replace_current(updated)

    def _load_editor(self) -> None:
        stage = self._current()
        self._loading = True
        try:
            enabled = stage is not None
            for widget in (
                self.name_edit,
                self.start_spin,
                self.duration_spin,
                self.fade_in_spin,
                self.fade_out_spin,
                self.notes_edit,
                self.override_table,
            ):
                widget.setEnabled(enabled)
            if stage is None:
                self.override_table.setRowCount(0)
                return
            self.name_edit.setText(stage.name)
            self.start_spin.setValue(stage.start_s)
            self.duration_spin.setValue(stage.duration_s)
            self.fade_in_spin.setValue(stage.transition_in_s)
            self.fade_out_spin.setValue(stage.transition_out_s)
            self.notes_edit.setText(stage.notes)
            self._load_overrides(stage)
        finally:
            self._loading = False
        self.timeline_view.set_selected(self._selected)

    def _load_overrides(self, stage: StageConfig) -> None:
        table = self.override_table
        table.blockSignals(True)
        reset_rows(table, len(stage.parameter_overrides))
        for row, binding in enumerate(stage.parameter_overrides):
            name = QTableWidgetItem(f"{binding.target_id}.{binding.parameter_path}")
            name.setFlags(name.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name)

            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            spin.setValue(float(binding.value))
            spin.valueChanged.connect(lambda value, r=row: self._set_override(r, value=value))
            place(table, row, 1, spin)

            combo = QComboBox()
            combo.addItems(list(CURVES))
            combo.setCurrentText(binding.curve)
            combo.currentTextChanged.connect(lambda text, r=row: self._set_override(r, curve=text))
            place(table, row, 2, combo)
        table.blockSignals(False)

    def _set_override(self, row: int, **changes: Any) -> None:
        stage = self._current()
        if stage is None or not 0 <= row < len(stage.parameter_overrides):
            return
        bindings = list(stage.parameter_overrides)
        existing = bindings[row]
        bindings[row] = ParameterBinding(
            target_id=existing.target_id,
            parameter_path=existing.parameter_path,
            value=float(changes.get("value", existing.value)),
            curve=str(changes.get("curve", existing.curve)),
        )
        self._replace_current(
            StageConfig(
                id=stage.id,
                name=stage.name,
                start_s=stage.start_s,
                duration_s=stage.duration_s,
                transition_in_s=stage.transition_in_s,
                transition_out_s=stage.transition_out_s,
                parameter_overrides=tuple(bindings),
                notes=stage.notes,
            )
        )

    def add_override(self, parameter_path: str, value: float = 0.0, target_id: str = "voice") -> None:
        stage = self._current()
        if stage is None:
            return
        binding = ParameterBinding(
            target_id=target_id, parameter_path=str(parameter_path), value=float(value)
        )
        self._replace_current(
            StageConfig(
                id=stage.id,
                name=stage.name,
                start_s=stage.start_s,
                duration_s=stage.duration_s,
                transition_in_s=stage.transition_in_s,
                transition_out_s=stage.transition_out_s,
                parameter_overrides=(*stage.parameter_overrides, binding),
                notes=stage.notes,
            )
        )

    def _add_override(self) -> None:
        if self._current() is None:
            self.status_label.setText("Select a stage first.")
            return
        name, accepted = QInputDialog.getText(
            self, "Add override", "Parameter name (for example beatFreq):"
        )
        if accepted and str(name).strip():
            self.add_override(str(name).strip())

    def _remove_override(self) -> None:
        stage = self._current()
        row = self.override_table.currentRow()
        if stage is None or not 0 <= row < len(stage.parameter_overrides):
            return
        bindings = list(stage.parameter_overrides)
        bindings.pop(row)
        self._replace_current(
            StageConfig(
                id=stage.id,
                name=stage.name,
                start_s=stage.start_s,
                duration_s=stage.duration_s,
                transition_in_s=stage.transition_in_s,
                transition_out_s=stage.transition_out_s,
                parameter_overrides=tuple(bindings),
                notes=stage.notes,
            )
        )

    # --- refresh ------------------------------------------------------------

    def _refresh(self) -> None:
        self.timeline_view.set_timeline(self._timeline)
        self.timeline_view.set_selected(self._selected)
        self._load_editor()
        issues = self._timeline.validate()
        if issues:
            self.status_label.setText(
                "; ".join(f"[{issue.severity}] {issue.path}: {issue.message}" for issue in issues)
            )

    def _emit(self) -> None:
        if not self._loading:
            self.timelineChanged.emit(self._timeline.describe())

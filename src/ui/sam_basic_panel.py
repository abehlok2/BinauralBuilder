"""Registry-driven SAM parameter controls.

Every row here is generated from
:data:`src.audio.sam_workbench.parameters.SAM2_FIELDS`, so a field is declared
once and the label, unit, bounds, decimals, default, and tooltip follow it into
the GUI. Adding a parameter means adding it to the registry, not to a table
here and another one in the generic editor.

Edits are coalesced: dragging a spin box emits one `paramsChanged` after a
short quiet period rather than one per pixel, so a live preview is not
restarted forty times a second.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.parameters import (
    ADVANCED,
    BASIC,
    EXPERT,
    ParameterField,
    fields_for_mode,
)

__all__ = ["NumericControl", "ChoiceControl", "SamBasicPanel"]

#: Quiet period before an edit is published, in milliseconds.
COALESCE_MS = 120

_ERROR_STYLE = "border: 1px solid #d04040;"


class _ControlBase(QWidget):
    """Shared chrome: label, unit, reset-to-default, and a warning badge."""

    valueEdited = pyqtSignal(str, object)

    def __init__(self, entry: ParameterField, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.name = name

        self.badge = QLabel("")
        self.badge.setVisible(False)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip(f"Restore the default ({entry.default!r})")
        self.reset_button.setFixedWidth(56)
        self.reset_button.clicked.connect(self.reset_to_default)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def _finish_layout(self, editor: QWidget) -> None:
        self._layout.addWidget(editor, 1)
        if self.entry.unit:
            unit_label = QLabel(self.entry.unit)
            unit_label.setToolTip(f"Unit of {self.entry.label}")
            self._layout.addWidget(unit_label)
        self._layout.addWidget(self.reset_button)
        self._layout.addWidget(self.badge)
        editor.setToolTip(self.entry.tooltip)
        self.setToolTip(self.entry.tooltip)

    # --- issues -------------------------------------------------------------

    def show_issue(self, message: str | None, severity: str = "error") -> None:
        """Attach a validation message to this field, or clear it."""

        editor = self.editor()
        if not message:
            self.badge.setVisible(False)
            self.badge.setText("")
            self.badge.setToolTip("")
            editor.setStyleSheet("")
            editor.setToolTip(self.entry.tooltip)
            return
        self.badge.setText("!" if severity == "error" else "i")
        self.badge.setToolTip(message)
        self.badge.setVisible(True)
        editor.setToolTip(f"{self.entry.tooltip}\n\n{severity.title()}: {message}")
        editor.setStyleSheet(_ERROR_STYLE if severity == "error" else "")

    # --- interface ----------------------------------------------------------

    def editor(self) -> QWidget:
        raise NotImplementedError

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def reset_to_default(self) -> None:
        self.set_value(self.entry.default)
        self.valueEdited.emit(self.name, self.value())


class NumericControl(_ControlBase):
    """A spin box bounded and formatted by its registry field."""

    def __init__(self, entry: ParameterField, name: str, parent: QWidget | None = None) -> None:
        super().__init__(entry, name, parent)
        if entry.kind == "int":
            self._spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self._spin.setRange(
                int(entry.minimum if entry.minimum is not None else -1_000_000),
                int(entry.maximum if entry.maximum is not None else 1_000_000),
            )
        else:
            self._spin = QDoubleSpinBox()
            self._spin.setDecimals(entry.decimals)
            self._spin.setRange(
                float(entry.minimum if entry.minimum is not None else -1e9),
                float(entry.maximum if entry.maximum is not None else 1e9),
            )
            self._spin.setSingleStep(10.0 ** (-entry.decimals) * 10.0)
        self._spin.valueChanged.connect(lambda _: self.valueEdited.emit(self.name, self.value()))
        self.set_value(entry.default)
        self._finish_layout(self._spin)

    def editor(self) -> QWidget:
        return self._spin

    def value(self) -> Any:
        return self._spin.value()

    def set_value(self, value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(self.entry.default)
        blocked = self._spin.blockSignals(True)
        try:
            self._spin.setValue(int(round(number)) if self.entry.kind == "int" else number)
        finally:
            self._spin.blockSignals(blocked)


class ChoiceControl(_ControlBase):
    """A combo box over a registry field's declared choices."""

    def __init__(self, entry: ParameterField, name: str, parent: QWidget | None = None) -> None:
        super().__init__(entry, name, parent)
        self._combo = QComboBox()
        self._combo.addItems(list(entry.choices))
        self._combo.currentIndexChanged.connect(
            lambda _: self.valueEdited.emit(self.name, self.value())
        )
        self.set_value(entry.default)
        self._finish_layout(self._combo)

    def editor(self) -> QWidget:
        return self._combo

    def value(self) -> Any:
        return self._combo.currentText()

    def set_value(self, value: Any) -> None:
        text = str(value).lower()
        index = self._combo.findText(text)
        blocked = self._combo.blockSignals(True)
        try:
            if index >= 0:
                self._combo.setCurrentIndex(index)
            elif self._combo.count():
                self._combo.setCurrentIndex(0)
        finally:
            self._combo.blockSignals(blocked)


class SamBasicPanel(QWidget):
    """The Basic (and optionally Advanced/Expert) SAM controls.

    The panel owns only the fields the registry says belong to the selected
    disclosure mode. Parameters it does not own - custom path profiles, unknown
    legacy keys, anything an extension added - are never touched, so editing a
    carrier frequency cannot quietly drop them.
    """

    paramsChanged = pyqtSignal(dict)

    def __init__(
        self,
        *,
        mode: str = BASIC,
        is_transition: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._is_transition = bool(is_transition)
        self._controls: dict[str, _ControlBase] = {}

        self._coalesce = QTimer(self)
        self._coalesce.setSingleShot(True)
        self._coalesce.setInterval(COALESCE_MS)
        self._coalesce.timeout.connect(self._emit_params)

        layout = QVBoxLayout(self)
        for group_mode, title in ((BASIC, "Basic"), (ADVANCED, "Advanced"), (EXPERT, "Expert")):
            entries = [entry for entry in fields_for_mode(mode) if entry.mode == group_mode]
            if not entries:
                continue
            box = QGroupBox(title)
            form = QFormLayout(box)
            for entry in entries:
                for name in entry.names_for(self._is_transition):
                    control = self._build_control(entry, name)
                    if control is None:
                        continue
                    self._controls[name] = control
                    form.addRow(self._row_label(entry, name), control)
            layout.addWidget(box)
        layout.addStretch(1)

    # --- construction -------------------------------------------------------

    def _build_control(self, entry: ParameterField, name: str) -> _ControlBase | None:
        if entry.kind in ("float", "int"):
            control = NumericControl(entry, name, self)
        elif entry.kind == "choice":
            control = ChoiceControl(entry, name, self)
        else:
            # JSON profiles and other structured values keep their existing
            # specialized editors in the generic voice dialog.
            return None
        control.valueEdited.connect(lambda *_: self._coalesce.start())
        return control

    @staticmethod
    def _row_label(entry: ParameterField, name: str) -> str:
        if name == entry.start_name:
            return f"{entry.label} (start)"
        if name == entry.end_name:
            return f"{entry.label} (end)"
        return entry.label

    # --- values -------------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._controls)

    def control_for(self, name: str) -> _ControlBase | None:
        return self._controls.get(name)

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Load values for the fields this panel owns, leaving the rest alone."""

        for name, control in self._controls.items():
            if name in params and params[name] is not None:
                control.set_value(params[name])

    def params(self) -> dict[str, Any]:
        """The values of the fields this panel owns."""

        return {name: control.value() for name, control in self._controls.items()}

    def apply_to(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return ``params`` updated with this panel's values.

        Keys the panel does not own are copied through untouched.
        """

        merged = dict(params)
        merged.update(self.params())
        return merged

    # --- validation ---------------------------------------------------------

    def show_issues(self, issues: Iterable[Any]) -> None:
        """Bind structured validation issues to the fields that produced them."""

        for control in self._controls.values():
            control.show_issue(None)
        for issue in issues:
            control = self._controls.get(getattr(issue, "path", ""))
            if control is not None:
                control.show_issue(issue.message, getattr(issue, "severity", "error"))

    # --- signals ------------------------------------------------------------

    def _emit_params(self) -> None:
        self.paramsChanged.emit(self.params())

    def flush(self) -> None:
        """Publish any pending coalesced edit immediately."""

        if self._coalesce.isActive():
            self._coalesce.stop()
        self._emit_params()

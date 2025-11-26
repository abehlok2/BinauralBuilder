from copy import deepcopy
from collections import OrderedDict
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

try:
    from src.utils.preferences import Preferences
except ImportError:  # Fallback for legacy execution contexts
    from utils.preferences import Preferences

from src.synth_functions import sound_creator
from src.ui.voice_editor_dialog import (
    FLANGE_TOOLTIPS,
    SPATIAL_TOOLTIPS,
    get_default_params_for_function,
)


def is_flanging_or_spatial_param(param_name: str) -> bool:
    name_lower = param_name.lower()
    return (
        name_lower.startswith("flange")
        or name_lower.startswith("spatial")
        or param_name in FLANGE_TOOLTIPS
        or param_name in SPATIAL_TOOLTIPS
    )


def get_default_display_flag(param_name: str) -> bool:
    """Return the default display flag for a parameter name."""
    return not is_flanging_or_spatial_param(param_name)


class VoiceDetailDisplayDialog(QDialog):
    """Configure which synth parameters appear in the Selected Voice Details panel."""

    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Voice Detail Display")
        self._prefs = prefs
        self.settings: Dict[str, Dict[str, Dict[str, object]]] = deepcopy(
            getattr(prefs, "voice_detail_display", {}) or {}
        )
        self.current_function: str | None = None
        self.param_widgets: Dict[str, Dict[str, object]] = {}

        layout = QVBoxLayout(self)
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Synth Function:"))
        self.function_combo = QComboBox()
        try:
            func_names = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
        except Exception:
            func_names = []
        if func_names:
            self.function_combo.addItems(func_names)
        else:
            self.function_combo.addItem("(No functions available)")
            self.function_combo.setEnabled(False)
        self.function_combo.currentTextChanged.connect(self._on_function_change)
        header_layout.addWidget(self.function_combo, 1)
        layout.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QFormLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if func_names:
            self._load_function(func_names[0])

    def get_settings(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        return deepcopy(self.settings)

    # ---- Internal helpers ----
    def _ensure_defaults_for_function(self, func_name: str) -> OrderedDict:
        default_params = get_default_params_for_function(
            func_name, func_name.endswith("_transition")
        )
        func_settings = self.settings.setdefault(func_name, {})
        for name, value in default_params.items():
            if name not in func_settings:
                func_settings[name] = {
                    "display": get_default_display_flag(name),
                    "default": value,
                }
        return default_params

    def _clear_form(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _load_function(self, func_name: str):
        self._clear_form()
        self.param_widgets.clear()
        self.current_function = func_name

        defaults = self._ensure_defaults_for_function(func_name)
        if not defaults:
            self.scroll_layout.addRow(QLabel("No parameters available for this synth function."))
            return

        for name, default_value in defaults.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            display_box = QCheckBox("Display")
            display_box.setChecked(
                self.settings.get(func_name, {}).get(name, {}).get(
                    "display", get_default_display_flag(name)
                )
            )
            row_layout.addWidget(display_box)

            row_layout.addWidget(QLabel(name))

            value_edit = QLineEdit(str(default_value))
            stored_default = self.settings.get(func_name, {}).get(name, {}).get(
                "default", default_value
            )
            value_edit.setText("" if stored_default is None else str(stored_default))
            row_layout.addWidget(value_edit)

            self.param_widgets[name] = {
                "display": display_box,
                "value": value_edit,
            }
            self.scroll_layout.addRow(row_widget)

    def _on_function_change(self, func_name: str):
        self._save_current_edits()
        if func_name:
            self._load_function(func_name)

    def _save_current_edits(self):
        if not self.current_function:
            return
        func_settings = self.settings.setdefault(self.current_function, {})
        for name, widgets in self.param_widgets.items():
            display_flag = widgets["display"].isChecked()
            text_value = widgets["value"].text()
            func_settings[name] = {
                "display": display_flag,
                "default": self._parse_value(
                    text_value, func_settings.get(name, {}).get("default")
                ),
            }

    def _parse_value(self, text: str, fallback):
        stripped = text.strip()
        if not stripped:
            return fallback
        lower = stripped.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return float(stripped)
        except ValueError:
            return stripped

    def _accept(self):
        self._save_current_edits()
        self._prefs.voice_detail_display = deepcopy(self.settings)
        self.accept()

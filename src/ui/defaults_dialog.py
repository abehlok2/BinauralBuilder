"""Dialog for configuring session builder defaults."""

import json
import os
from typing import Dict, Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QSlider,
    QDialogButtonBox,
    QLabel,
    QHBoxLayout,
    QWidget,
    QComboBox,
)

DEFAULTS_FILE = "session_defaults.json"

DEFAULT_SETTINGS = {
    "step_duration": 300.0,
    "normalization_level": 0.95,
    "crossfade_duration": 10.0,
    "default_binaural_id": None,
    "default_noise_id": None,
}

def load_defaults() -> Dict[str, Any]:
    """Load defaults from file or return standard defaults."""
    if os.path.exists(DEFAULTS_FILE):
        try:
            with open(DEFAULTS_FILE, "r") as f:
                data = json.load(f)
                defaults = DEFAULT_SETTINGS.copy()
                defaults.update(data)
                return defaults
        except Exception as e:
            print(f"Error loading defaults: {e}")
    return DEFAULT_SETTINGS.copy()

def save_defaults(defaults: Dict[str, Any]) -> None:
    """Save defaults to file."""
    try:
        with open(DEFAULTS_FILE, "w") as f:
            json.dump(defaults, f, indent=4)
    except Exception as e:
        print(f"Error saving defaults: {e}")


class DefaultsDialog(QDialog):
    """Dialog to set default session parameters."""

    def __init__(self, binaural_catalog: Dict[str, Any], noise_catalog: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Defaults")
        self.resize(450, 300)
        
        self._binaural_catalog = binaural_catalog
        self._noise_catalog = noise_catalog
        self._defaults = load_defaults()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # Step Duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setSingleStep(10.0)
        self.duration_spin.setValue(float(self._defaults.get("step_duration", 300.0)))
        self.duration_spin.setSuffix(" s")
        form_layout.addRow("Default Step Duration:", self.duration_spin)
        
        # Normalization Level
        norm_widget = QWidget()
        norm_layout = QHBoxLayout(norm_widget)
        norm_layout.setContentsMargins(0, 0, 0, 0)
        
        self.norm_slider = QSlider(Qt.Horizontal)
        self.norm_slider.setRange(0, 100)
        self.norm_val = float(self._defaults.get("normalization_level", 0.95))
        self.norm_slider.setValue(int(self.norm_val * 100))
        
        self.norm_spin = QDoubleSpinBox()
        self.norm_spin.setRange(0.0, 1.0)
        self.norm_spin.setSingleStep(0.01)
        self.norm_spin.setValue(self.norm_val)
        
        self.norm_slider.valueChanged.connect(lambda v: self.norm_spin.setValue(v / 100.0))
        self.norm_spin.valueChanged.connect(lambda v: self.norm_slider.setValue(int(v * 100)))
        
        norm_layout.addWidget(self.norm_slider)
        norm_layout.addWidget(self.norm_spin)
        form_layout.addRow("Default Normalization:", norm_widget)
        
        # Crossfade Duration
        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(0.0, 60.0)
        self.crossfade_spin.setSingleStep(1.0)
        self.crossfade_spin.setValue(float(self._defaults.get("crossfade_duration", 10.0)))
        self.crossfade_spin.setSuffix(" s")
        form_layout.addRow("Default Crossfade:", self.crossfade_spin)
        
        # Default Binaural Preset
        self.binaural_combo = QComboBox()
        # Sort by label
        sorted_binaural = sorted(self._binaural_catalog.items(), key=lambda x: x[1].label)
        for pid, preset in sorted_binaural:
            self.binaural_combo.addItem(preset.label, pid)
            
        current_binaural = self._defaults.get("default_binaural_id")
        if current_binaural:
            idx = self.binaural_combo.findData(current_binaural)
            if idx >= 0:
                self.binaural_combo.setCurrentIndex(idx)
        form_layout.addRow("Default Binaural Preset:", self.binaural_combo)

        # Default Noise Preset
        self.noise_combo = QComboBox()
        self.noise_combo.addItem("None", None)
        sorted_noise = sorted(self._noise_catalog.items(), key=lambda x: x[1].label)
        for pid, preset in sorted_noise:
            self.noise_combo.addItem(preset.label, pid)
            
        current_noise = self._defaults.get("default_noise_id")
        if current_noise:
            idx = self.noise_combo.findData(current_noise)
            if idx >= 0:
                self.noise_combo.setCurrentIndex(idx)
        form_layout.addRow("Default Noise Preset:", self.noise_combo)
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_defaults(self) -> Dict[str, Any]:
        """Return the configured defaults."""
        return {
            "step_duration": self.duration_spin.value(),
            "normalization_level": self.norm_spin.value(),
            "crossfade_duration": self.crossfade_spin.value(),
            "default_binaural_id": self.binaural_combo.currentData(),
            "default_noise_id": self.noise_combo.currentData(),
        }

    def accept(self):
        """Save defaults on accept."""
        defaults = self.get_defaults()
        save_defaults(defaults)
        super().accept()

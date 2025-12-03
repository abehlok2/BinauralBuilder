from typing import Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from PyQt5.QtCore import QBuffer, QIODevice
try:
    from PyQt5.QtMultimedia import (
        QAudioOutput,
        QAudioFormat,
        QAudioDeviceInfo,
        QAudio,
    )
    QT_MULTIMEDIA_AVAILABLE = True
except ImportError:
    QT_MULTIMEDIA_AVAILABLE = False

import numpy as np

from src.utils.colored_noise import (
    DEFAULT_COLOR_PRESETS,
    ColoredNoiseGenerator,
    apply_preset_to_generator,
    generator_to_preset,
    load_custom_color_presets,
    save_custom_color_presets,
)


class ColoredNoiseDialog(QDialog):
    """Dialog for configuring customizable colored noise presets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colored Noise Presets")
        self.resize(400, 0)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._custom_presets = load_custom_color_presets()

        color_layout = QHBoxLayout()
        self.color_combo = QComboBox()
        self.save_color_btn = QPushButton("Save Color")
        self.delete_color_btn = QPushButton("Delete Color")
        self.save_color_btn.clicked.connect(self.on_save_color)
        self.delete_color_btn.clicked.connect(self.on_delete_color)
        self.color_combo.currentTextChanged.connect(self.on_color_selected)
        color_layout.addWidget(self.color_combo, 1)
        color_layout.addWidget(self.save_color_btn)
        color_layout.addWidget(self.delete_color_btn)
        form.addRow("Noise Color:", color_layout)

        self.exponent_spin = QDoubleSpinBox()
        self.exponent_spin.setRange(-3.0, 3.0)
        self.exponent_spin.setValue(1.0)
        self.exponent_spin.setToolTip("Power-law exponent applied at low frequencies (e.g. 1=pink)")
        form.addRow("Low Exponent:", self.exponent_spin)

        self.high_exponent_spin = QDoubleSpinBox()
        self.high_exponent_spin.setRange(-3.0, 3.0)
        self.high_exponent_spin.setValue(1.0)
        self.high_exponent_spin.setToolTip("Exponent to reach at the top of the spectrum")
        form.addRow("High Exponent:", self.high_exponent_spin)

        self.distribution_curve_spin = QDoubleSpinBox()
        self.distribution_curve_spin.setRange(0.1, 5.0)
        self.distribution_curve_spin.setSingleStep(0.1)
        self.distribution_curve_spin.setValue(1.0)
        self.distribution_curve_spin.setToolTip("Curve shaping how quickly the exponent transitions across frequencies")
        form.addRow("Distribution Curve:", self.distribution_curve_spin)

        self.lowcut_spin = QDoubleSpinBox()
        self.lowcut_spin.setRange(0.0, 20000.0)
        self.lowcut_spin.setValue(0.0)
        form.addRow("Low Cut (Hz):", self.lowcut_spin)

        self.highcut_spin = QDoubleSpinBox()
        self.highcut_spin.setRange(0.0, 20000.0)
        self.highcut_spin.setValue(0.0)
        form.addRow("High Cut (Hz):", self.highcut_spin)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 10.0)
        self.amplitude_spin.setValue(1.0)
        form.addRow("Amplitude:", self.amplitude_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)
        form.addRow("Seed:", self.seed_spin)

        layout.addLayout(form)

        button_layout = QHBoxLayout()
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.on_test)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.test_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        self.audio_output = None
        self.audio_buffer = None

        if not QT_MULTIMEDIA_AVAILABLE:
            self.test_btn.setEnabled(False)

        self._refresh_color_presets()

    def _collect_params(self) -> ColoredNoiseGenerator:
        lowcut = self.lowcut_spin.value() or None
        highcut = self.highcut_spin.value() or None
        seed_val = self.seed_spin.value()
        seed = seed_val if seed_val != -1 else None
        return ColoredNoiseGenerator(
            exponent=float(self.exponent_spin.value()),
            high_exponent=float(self.high_exponent_spin.value()),
            distribution_curve=float(self.distribution_curve_spin.value()),
            lowcut=lowcut,
            highcut=highcut,
            amplitude=float(self.amplitude_spin.value()),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------
    def _refresh_color_presets(self, *, selected: Optional[str] = None) -> None:
        current = selected or self.color_combo.currentText()
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        for name in sorted(DEFAULT_COLOR_PRESETS):
            self.color_combo.addItem(name)
        for name in sorted(self._custom_presets):
            self.color_combo.addItem(name)
        if self.color_combo.count():
            idx = self.color_combo.findText(current)
            self.color_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.color_combo.blockSignals(False)
        self.on_color_selected(self.color_combo.currentText())

    def on_color_selected(self, name: str) -> None:
        preset = self._get_preset(name)
        if preset is not None:
            self._apply_preset(preset)
        self.delete_color_btn.setEnabled(name in self._custom_presets)

    def _get_preset(self, name: str) -> dict | None:
        if name in self._custom_presets:
            return self._custom_presets[name]
        return DEFAULT_COLOR_PRESETS.get(name)

    def _apply_preset(self, preset: dict) -> None:
        gen = apply_preset_to_generator(self._collect_params(), preset)
        self.exponent_spin.setValue(float(gen.exponent))
        self.high_exponent_spin.setValue(float(gen.high_exponent or 0.0))
        self.distribution_curve_spin.setValue(float(gen.distribution_curve))
        self.lowcut_spin.setValue(float(gen.lowcut or 0.0))
        self.highcut_spin.setValue(float(gen.highcut or 0.0))
        self.amplitude_spin.setValue(float(gen.amplitude))
        self.seed_spin.setValue(int(gen.seed) if gen.seed is not None else -1)

    def on_save_color(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Noise Color", "Name for this color:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a non-empty name for the color.")
            return
        if name in DEFAULT_COLOR_PRESETS:
            QMessageBox.warning(
                self,
                "Reserved Name",
                "That name is reserved for a built-in color. Please choose another.",
            )
            return
        if name in self._custom_presets:
            overwrite = QMessageBox.question(
                self,
                "Overwrite Color",
                f"A custom color named '{name}' already exists. Overwrite it?",
            )
            if overwrite != QMessageBox.Yes:
                return
        params = generator_to_preset(self._collect_params())
        self._custom_presets[name] = params
        save_custom_color_presets(self._custom_presets)
        self._refresh_color_presets(selected=name)

    def on_delete_color(self) -> None:
        name = self.color_combo.currentText()
        if name not in self._custom_presets:
            return
        confirm = QMessageBox.question(
            self,
            "Delete Noise Color",
            f"Remove the custom noise color '{name}'?",
        )
        if confirm != QMessageBox.Yes:
            return
        self._custom_presets.pop(name, None)
        save_custom_color_presets(self._custom_presets)
        self._refresh_color_presets()

    def on_test(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            return

        try:
            # Generate 5 seconds of noise for testing
            gen = self._collect_params()
            gen.duration = 5.0
            noise = gen.generate()
            
            # Convert to int16 PCM
            audio_int16 = (np.clip(noise, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            fmt = QAudioFormat()
            fmt.setCodec("audio/pcm")
            fmt.setSampleRate(gen.sample_rate)
            fmt.setSampleSize(16)
            fmt.setChannelCount(1) # Mono
            fmt.setByteOrder(QAudioFormat.LittleEndian)
            fmt.setSampleType(QAudioFormat.SignedInt)

            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(fmt):
                # Try stereo if mono fails
                fmt.setChannelCount(2)
                if not device_info.isFormatSupported(fmt):
                    QMessageBox.warning(self, "Test Error", "Default output device does not support the required format")
                    return
                # Duplicate for stereo
                stereo = np.column_stack((noise, noise))
                audio_int16 = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

            if self.audio_output:
                self.on_stop()

            self.audio_output = QAudioOutput(fmt, self)
            self.audio_output.stateChanged.connect(self.on_audio_state_changed)
            self.audio_buffer = QBuffer()
            self.audio_buffer.setData(audio_bytes)
            self.audio_buffer.open(QIODevice.ReadOnly)
            self.audio_output.start(self.audio_buffer)
            self.stop_btn.setEnabled(True)
            
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def on_stop(self):
        if self.audio_output:
            self.audio_output.stop()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer = None
        self.stop_btn.setEnabled(False)

    def on_audio_state_changed(self, state):
        if state in (QAudio.IdleState, QAudio.StoppedState):
            self.on_stop()

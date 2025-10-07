"""Dialog UI for binaural encoder functionality."""

from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

from src.utils.binaural_processing import (
    BinauralProcessingConfig,
    binaural_encode,
    parse_band_ranges,
)


class BinauralEncoderDialog(QDialog):
    """Dialog to apply binaural encoding to audio files."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Binaural Encoder")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        file_group = QGroupBox("Audio Files")
        file_layout = QGridLayout()
        file_group.setLayout(file_layout)

        self.input_path_edit = QLineEdit()
        self.output_path_edit = QLineEdit()
        browse_in_btn = QPushButton("Browse…")
        browse_out_btn = QPushButton("Browse…")
        browse_in_btn.clicked.connect(self._choose_input_file)
        browse_out_btn.clicked.connect(self._choose_output_file)

        file_layout.addWidget(QLabel("Input Audio:"), 0, 0)
        file_layout.addWidget(self.input_path_edit, 0, 1)
        file_layout.addWidget(browse_in_btn, 0, 2)
        file_layout.addWidget(QLabel("Output Audio:"), 1, 0)
        file_layout.addWidget(self.output_path_edit, 1, 1)
        file_layout.addWidget(browse_out_btn, 1, 2)
        layout.addWidget(file_group)

        offsets_group = QGroupBox("Frequency Offsets")
        offsets_layout = QFormLayout()
        offsets_group.setLayout(offsets_layout)

        self.beat_frequency_spin = QDoubleSpinBox()
        self.beat_frequency_spin.setRange(0.1, 40.0)
        self.beat_frequency_spin.setDecimals(2)
        self.beat_frequency_spin.setValue(4.0)
        self.left_offset_spin = QDoubleSpinBox()
        self.left_offset_spin.setRange(-50.0, 50.0)
        self.left_offset_spin.setDecimals(3)
        self.left_offset_spin.setValue(-1.5)
        self.right_offset_spin = QDoubleSpinBox()
        self.right_offset_spin.setRange(-50.0, 50.0)
        self.right_offset_spin.setDecimals(3)
        self.right_offset_spin.setValue(1.5)

        offsets_layout.addRow("Beat Frequency (Hz)", self.beat_frequency_spin)
        offsets_layout.addRow("Left Offset (Hz)", self.left_offset_spin)
        offsets_layout.addRow("Right Offset (Hz)", self.right_offset_spin)
        layout.addWidget(offsets_group)

        spectral_group = QGroupBox("Spectral Panning")
        spectral_layout = QFormLayout()
        spectral_group.setLayout(spectral_layout)

        self.band_ranges_edit = QLineEdit()
        self.band_ranges_edit.setPlaceholderText("e.g. 100-300,450-700,1200-1800")
        self.band_phase_spin = QDoubleSpinBox()
        self.band_phase_spin.setRange(0.0, 3.14159)
        self.band_phase_spin.setDecimals(3)
        self.band_phase_spin.setValue(0.4)
        self.band_detune_spin = QDoubleSpinBox()
        self.band_detune_spin.setRange(0.0, 5.0)
        self.band_detune_spin.setDecimals(3)
        self.band_detune_spin.setValue(0.25)

        spectral_layout.addRow("Band Ranges (Hz)", self.band_ranges_edit)
        spectral_layout.addRow("Phase Spread (rad)", self.band_phase_spin)
        spectral_layout.addRow("Band Detune (Hz)", self.band_detune_spin)
        layout.addWidget(spectral_group)

        lfo_group = QGroupBox("Dynamic Panning LFO")
        lfo_layout = QFormLayout()
        lfo_group.setLayout(lfo_layout)

        self.lfo_depth_spin = QDoubleSpinBox()
        self.lfo_depth_spin.setRange(0.0, 1.0)
        self.lfo_depth_spin.setSingleStep(0.05)
        self.lfo_depth_spin.setValue(0.35)
        self.lfo_multiplier_spin = QDoubleSpinBox()
        self.lfo_multiplier_spin.setRange(0.05, 8.0)
        self.lfo_multiplier_spin.setDecimals(2)
        self.lfo_multiplier_spin.setValue(0.5)

        lfo_layout.addRow("Depth", self.lfo_depth_spin)
        lfo_layout.addRow("Rate Multiplier", self.lfo_multiplier_spin)
        layout.addWidget(lfo_group)

        level_group = QGroupBox("Level Management")
        level_layout = QFormLayout()
        level_group.setLayout(level_layout)

        self.target_rms_spin = QDoubleSpinBox()
        self.target_rms_spin.setRange(0.0, 0.7)
        self.target_rms_spin.setDecimals(3)
        self.target_rms_spin.setValue(0.18)
        self.crest_spin = QDoubleSpinBox()
        self.crest_spin.setRange(0.0, 4.0)
        self.crest_spin.setDecimals(2)
        self.crest_spin.setValue(1.5)

        level_layout.addRow("Target RMS", self.target_rms_spin)
        level_layout.addRow("Soft Knee", self.crest_spin)
        layout.addWidget(level_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        button_row = QHBoxLayout()
        self.process_button = QPushButton("Process Audio")
        self.process_button.clicked.connect(self._process_audio)
        cancel_button = QPushButton("Close")
        cancel_button.clicked.connect(self.reject)
        button_row.addStretch(1)
        button_row.addWidget(self.process_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

    def _choose_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            os.path.expanduser("~"),
            "Audio Files (*.wav *.flac *.mp3)",
        )
        if path:
            self.input_path_edit.setText(path)
            if not self.output_path_edit.text():
                base, ext = os.path.splitext(path)
                self.output_path_edit.setText(f"{base}_binaural{ext}")

    def _choose_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Audio",
            self.output_path_edit.text() or os.path.expanduser("~"),
            "Audio Files (*.wav *.flac *.mp3)",
        )
        if path:
            self.output_path_edit.setText(path)

    def _build_config(self) -> BinauralProcessingConfig:
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        if not input_path or not os.path.exists(input_path):
            raise ValueError("Please choose a valid input audio file.")
        if not output_path:
            raise ValueError("Please choose a destination for the processed audio.")

        band_ranges = parse_band_ranges(self.band_ranges_edit.text())

        return BinauralProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            beat_frequency=self.beat_frequency_spin.value(),
            left_offset_hz=self.left_offset_spin.value(),
            right_offset_hz=self.right_offset_spin.value(),
            band_ranges=band_ranges,
            band_phase_spread=self.band_phase_spin.value(),
            band_detune=self.band_detune_spin.value(),
            lfo_depth=self.lfo_depth_spin.value(),
            lfo_rate_multiplier=self.lfo_multiplier_spin.value(),
            target_rms=self.target_rms_spin.value(),
            crest_soft_knee=self.crest_spin.value(),
        )

    def _process_audio(self) -> None:
        try:
            config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Binaural Encoder", str(exc))
            return

        self.progress_bar.show()
        self.process_button.setEnabled(False)
        self.repaint()
        try:
            binaural_encode(config)
        except Exception as exc:  # noqa: PIE786 - propagate rich context
            QMessageBox.critical(self, "Processing Error", str(exc))
        else:
            QMessageBox.information(self, "Binaural Encoder", "Processing complete!")
        finally:
            self.progress_bar.hide()
            self.process_button.setEnabled(True)

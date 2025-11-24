from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QMessageBox,
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
except Exception as e:  # noqa: PIE786 - broad for missing backends
    print(
        "WARNING: PyQt5.QtMultimedia could not be imported.\n"
        "ColoredNoiseDialog will have audio preview disabled.\n"
        f"Original error: {e}"
    )
    QT_MULTIMEDIA_AVAILABLE = False

import soundfile as sf
import numpy as np

from src.utils.colored_noise import ColoredNoiseGenerator, plot_spectrogram


class ColoredNoiseDialog(QDialog):
    """Dialog for generating customizable colored noise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colored Noise Generator")
        self.resize(400, 0)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit("colored_noise.wav")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Output File:", file_layout)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 3600.0)
        self.duration_spin.setValue(60.0)
        form.addRow("Duration (s):", self.duration_spin)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(44100)
        form.addRow("Sample Rate:", self.sample_rate_spin)

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
        self.spectro_btn = QPushButton("Spectrogram")
        self.spectro_btn.clicked.connect(self.on_spectrogram)
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.on_test)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.spectro_btn)
        button_layout.addWidget(self.test_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(close_btn)
        button_layout.addWidget(self.generate_btn)
        layout.addLayout(button_layout)

        self.audio_output = None
        self.audio_buffer = None

        if not QT_MULTIMEDIA_AVAILABLE:
            self.test_btn.setEnabled(False)

    def browse_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def _collect_params(self) -> ColoredNoiseGenerator:
        lowcut = self.lowcut_spin.value() or None
        highcut = self.highcut_spin.value() or None
        seed_val = self.seed_spin.value()
        seed = seed_val if seed_val != -1 else None
        return ColoredNoiseGenerator(
            sample_rate=int(self.sample_rate_spin.value()),
            duration=float(self.duration_spin.value()),
            exponent=float(self.exponent_spin.value()),
            high_exponent=float(self.high_exponent_spin.value()),
            distribution_curve=float(self.distribution_curve_spin.value()),
            lowcut=lowcut,
            highcut=highcut,
            amplitude=float(self.amplitude_spin.value()),
            seed=seed,
        )

    def on_generate(self) -> None:
        try:
            gen = self._collect_params()
            noise = gen.generate()
            sf.write(self.file_edit.text() or "colored_noise.wav", noise, gen.sample_rate)
            QMessageBox.information(self, "Success", f"Generated {self.file_edit.text()}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def on_spectrogram(self) -> None:
        try:
            gen = self._collect_params()
            noise = gen.generate()
            plot_spectrogram(noise, gen.sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def on_test(self) -> None:
        if not QT_MULTIMEDIA_AVAILABLE:
            QMessageBox.critical(
                self,
                "PyQt5 Multimedia Missing",
                "PyQt5.QtMultimedia is required for audio preview, but it could not be loaded.",
            )
            return
        try:
            gen = self._collect_params()
            gen.duration = min(gen.duration, 10.0)
            noise = gen.generate()
            audio_int16 = (np.clip(noise, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            fmt = QAudioFormat()
            fmt.setCodec("audio/pcm")
            fmt.setSampleRate(int(gen.sample_rate))
            fmt.setSampleSize(16)
            fmt.setChannelCount(1)
            fmt.setByteOrder(QAudioFormat.LittleEndian)
            fmt.setSampleType(QAudioFormat.SignedInt)

            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(fmt):
                QMessageBox.warning(self, "Noise Test", "Default output device does not support the required format")
                return

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

    def on_stop(self) -> None:
        if self.audio_output:
            self.audio_output.stop()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer = None
        self.stop_btn.setEnabled(False)

    def on_audio_state_changed(self, state) -> None:
        if state in (QAudio.IdleState, QAudio.StoppedState):
            self.on_stop()

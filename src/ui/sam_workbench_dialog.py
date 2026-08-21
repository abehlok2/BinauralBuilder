"""The SAM/HRTF workbench dialog.

A `QDialog` owned by the existing voice editor - not a second main window. It
edits a **copy** of the voice dictionary and commits through the existing save
path, so Cancel cannot leave a half-applied edit in the live project.

Preview rendering happens in a worker thread and plays through the same
QtMultimedia path the step tester already uses; no second audio backend is
introduced. Preview and export both call
:mod:`src.audio.sam_workbench`, so what you hear is what gets written.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from PyQt5.QtCore import QBuffer, QIODevice, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.parameters import (
    ADVANCED,
    BASIC,
    EXPERT,
    SAM2_FIELDS,
    fields_for_mode,
    validate_sam2_params,
)
from src.audio.sam_workbench.preview import PreviewResult, render_voice_preview, to_pcm16_bytes

from .sam_analysis_panel import SamAnalysisPanel
from .sam_basic_panel import SamBasicPanel
from .sam_path_panel import SamPathPanel

try:  # pragma: no cover - import guard mirrors the rest of the UI package
    from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioOutput

    AUDIO_OUTPUT_AVAILABLE = True
except Exception:  # pragma: no cover - headless or minimal Qt install
    QAudio = QAudioDeviceInfo = QAudioFormat = QAudioOutput = None
    AUDIO_OUTPUT_AVAILABLE = False

__all__ = ["SamWorkbenchDialog", "PreviewWorker", "SAM_FUNCTION_NAMES"]

#: Voices this workbench can edit.
SAM_FUNCTION_NAMES = (
    "spatial_angle_modulation_sam2",
    "spatial_angle_modulation_sam2_transition",
)

DEFAULT_PREVIEW_SECONDS = 5.0


class PreviewWorker(QObject):
    """Renders a preview off the GUI thread."""

    rendered = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, voice_data: Mapping[str, Any], sample_rate: int, duration_s: float) -> None:
        super().__init__()
        self._voice_data = copy.deepcopy(dict(voice_data))
        self._sample_rate = int(sample_rate)
        self._duration_s = float(duration_s)

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = render_voice_preview(
                self._voice_data,
                sample_rate_hz=self._sample_rate,
                duration_s=self._duration_s,
            )
        except Exception as error:  # pragma: no cover - defensive, surfaced in the UI
            self.failed.emit(str(error))
            return
        self.rendered.emit(result)


class SamWorkbenchDialog(QDialog):
    """SAM parameters, analysis, and compatibility views over one voice copy."""

    voiceApplied = pyqtSignal(dict)

    def __init__(
        self,
        voice_data: Mapping[str, Any],
        *,
        sample_rate: int = 44_100,
        parent: QWidget | None = None,
        on_apply: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAM/HRTF Workbench")
        self.resize(940, 720)
        if parent is not None:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        # Everything below edits this copy. The caller's dictionary is only
        # touched when Apply or OK is pressed.
        self._voice = copy.deepcopy(dict(voice_data))
        self._voice.setdefault("params", {})
        self._original = copy.deepcopy(self._voice)
        self._sample_rate = int(sample_rate)
        self._on_apply = on_apply
        self._is_transition = bool(self._voice.get("is_transition", False))

        self._preview_thread: QThread | None = None
        self._preview_worker: PreviewWorker | None = None
        self._audio_output = None
        self._audio_buffer: QBuffer | None = None
        self._last_preview: PreviewResult | None = None

        self._build_ui()
        self._load_params()
        self._revalidate()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Voice: {self._voice.get('synth_function_name', 'unknown')}"))
        header.addWidget(QLabel("Renderer:"))
        self.renderer_combo = QComboBox()
        self.renderer_combo.addItem("Abstract phase modulation", "abstract_pm")
        self.renderer_combo.addItem("Geometric binaural", "geometric")
        self.renderer_combo.setToolTip("Select the algorithm used by both Python preview and export.")
        self.renderer_combo.currentIndexChanged.connect(self._on_params_changed)
        header.addWidget(self.renderer_combo)
        header.addStretch(1)
        header.addWidget(QLabel("Disclosure:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([BASIC, ADVANCED, EXPERT])
        self.mode_combo.setCurrentText(EXPERT)
        self.mode_combo.setToolTip("How much of the parameter space to show.")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        header.addWidget(self.mode_combo)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.parameter_panel = SamBasicPanel(mode=EXPERT, is_transition=self._is_transition)
        # Compatibility aliases for integrations that previously addressed the
        # two duplicated panels directly. Both now refer to the same editor.
        self.basic_panel = self.parameter_panel
        self.advanced_panel = self.parameter_panel
        self.analysis_panel = SamAnalysisPanel()
        self.path_panel = SamPathPanel()
        self.compatibility_view = QTextEdit()
        self.compatibility_view.setReadOnly(True)

        self.tabs.addTab(self._scrolled(self.parameter_panel), "Parameters")
        self.tabs.addTab(self._scrolled(self.path_panel), "Path & Geometry")
        self.tabs.addTab(self.analysis_panel, "Analysis")
        self.tabs.addTab(self.compatibility_view, "Compatibility")
        layout.addWidget(self.tabs, 1)

        self.parameter_panel.paramsChanged.connect(self._on_params_changed)
        self.path_panel.paramsChanged.connect(self._on_params_changed)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview length:"))
        self.preview_seconds = QDoubleSpinBox()
        self.preview_seconds.setRange(0.25, 60.0)
        self.preview_seconds.setDecimals(2)
        self.preview_seconds.setValue(DEFAULT_PREVIEW_SECONDS)
        self.preview_seconds.setSuffix(" s")
        preview_row.addWidget(self.preview_seconds)
        self.preview_button = QPushButton("Render preview")
        self.preview_button.clicked.connect(self.start_preview)
        preview_row.addWidget(self.preview_button)
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_preview)
        preview_row.addWidget(self.play_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_preview)
        preview_row.addWidget(self.stop_button)
        preview_row.addStretch(1)
        layout.addLayout(preview_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(self.buttons)

    @staticmethod
    def _scrolled(widget: QWidget) -> QWidget:
        from PyQt5.QtWidgets import QScrollArea

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _on_mode_changed(self, mode: str) -> None:
        """Apply progressive disclosure within the shared parameter view."""

        self.parameter_panel.set_mode(mode)

    # --- parameters ---------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._voice.get("params", {}))

    def _load_params(self) -> None:
        params = self.params
        renderer_index = self.renderer_combo.findData(params.get("rendererMode", "abstract_pm"))
        self.renderer_combo.setCurrentIndex(max(0, renderer_index))
        self.parameter_panel.set_params(params)
        self.path_panel.set_params(params)
        self._on_mode_changed(self.mode_combo.currentText())

    def collect_params(self) -> dict[str, Any]:
        """Merge both panels' values over the original parameters.

        Anything the panels do not own - custom path profiles, transition
        timing, unknown keys from an older or external extension - is carried
        through untouched.
        """

        merged = dict(self._original.get("params", {}))
        merged["rendererMode"] = self.renderer_combo.currentData()
        panel_params = self.parameter_panel.params()
        visible_names = {
            name
            for entry in fields_for_mode(self.mode_combo.currentText())
            for name in entry.names_for(self._is_transition)
        }
        merged.update({name: value for name, value in panel_params.items() if name in visible_names})
        merged.update(self.path_panel.params())
        return merged

    def voice_data(self) -> dict[str, Any]:
        """The edited voice dictionary, ready for the existing save path."""

        voice = copy.deepcopy(self._original)
        voice["params"] = self.collect_params()
        return voice

    def _on_params_changed(self, _params: dict[str, Any]) -> None:
        self._voice["params"] = self.collect_params()
        self._revalidate()

    # --- validation ---------------------------------------------------------

    def issues(self) -> tuple[Any, ...]:
        return validate_sam2_params(
            self.collect_params(),
            is_transition=self._is_transition,
            sample_rate_hz=self._sample_rate,
        )

    def _revalidate(self) -> None:
        issues = self.issues()
        self.parameter_panel.show_issues(issues)

        errors = [issue for issue in issues if issue.severity == "error"]
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not errors)
        self.buttons.button(QDialogButtonBox.Apply).setEnabled(not errors)
        if errors:
            self.status_label.setText(
                "Cannot apply: " + "; ".join(f"{issue.path}: {issue.message}" for issue in errors)
            )
        elif self.status_label.text().startswith("Cannot apply"):
            self.status_label.setText("")
        self._refresh_compatibility(issues)

    def _refresh_compatibility(self, issues: tuple[Any, ...]) -> None:
        params = self.collect_params()
        known = {name for entry in SAM2_FIELDS for name in (entry.name, *entry.aliases, entry.start_name, entry.end_name)}
        preserved = {key: value for key, value in params.items() if key not in known}
        lines = [
            f"Renderer: canonical SAM core (src.audio.sam_workbench), sample rate {self._sample_rate} Hz",
            f"Voice: {self._voice.get('synth_function_name', 'unknown')}"
            f" ({'transition' if self._is_transition else 'static'})",
            f"Schema version: {params.get('samSchemaVersion', 'unversioned (legacy ear polarity)')}",
            "",
            "Parameters preserved but not edited here:",
        ]
        if preserved:
            lines.extend(f"  {key} = {value!r}" for key, value in sorted(preserved.items()))
        else:
            lines.append("  (none)")
        if issues:
            lines.extend(["", "Validation:"])
            lines.extend(f"  [{issue.severity}] {issue.path}: {issue.message}" for issue in issues)
        self.compatibility_view.setPlainText("\n".join(lines))

    # --- preview ------------------------------------------------------------

    def start_preview(self) -> None:
        """Render a preview in a worker thread; the dialog stays responsive."""

        if self._preview_thread is not None:
            self.status_label.setText("A preview render is already running.")
            return
        self.stop_preview()
        self.preview_button.setEnabled(False)
        self.status_label.setText("Rendering preview…")

        worker = PreviewWorker(self.voice_data(), self._sample_rate, self.preview_seconds.value())
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.rendered.connect(self._on_preview_rendered)
        worker.failed.connect(self._on_preview_failed)
        worker.rendered.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_preview_thread_finished)
        self._preview_worker = worker
        self._preview_thread = thread
        thread.start()

    @pyqtSlot(object)
    def _on_preview_rendered(self, result: PreviewResult) -> None:
        self._last_preview = result
        self.analysis_panel.set_audio(result.audio, result.sample_rate_hz)
        self.play_button.setEnabled(AUDIO_OUTPUT_AVAILABLE)
        message = (
            f"Preview ready: {result.duration_s:.2f} s, peak {result.peak:.3f}"
            f"{', limited' if result.limited else ''}"
            f"{', truncated to the preview cap' if result.truncated else ''}"
        )
        if not AUDIO_OUTPUT_AVAILABLE:
            message += " — audio output unavailable, showing analysis only"
        self.status_label.setText(message)

    @pyqtSlot(str)
    def _on_preview_failed(self, message: str) -> None:
        self.status_label.setText(f"Preview failed: {message}")

    @pyqtSlot()
    def _on_preview_thread_finished(self) -> None:
        if self._preview_thread is not None:
            self._preview_thread.deleteLater()
        self._preview_thread = None
        self._preview_worker = None
        self.preview_button.setEnabled(True)

    @property
    def last_preview(self) -> PreviewResult | None:
        return self._last_preview

    def play_preview(self) -> None:
        """Play the rendered preview through the existing QtMultimedia path."""

        if self._last_preview is None:
            self.status_label.setText("Render a preview first.")
            return
        if not AUDIO_OUTPUT_AVAILABLE:
            self.status_label.setText("Audio output is unavailable in this environment.")
            return

        audio_format = QAudioFormat()
        audio_format.setCodec("audio/pcm")
        audio_format.setSampleRate(self._last_preview.sample_rate_hz)
        audio_format.setSampleSize(16)
        audio_format.setChannelCount(2)
        audio_format.setByteOrder(QAudioFormat.LittleEndian)
        audio_format.setSampleType(QAudioFormat.SignedInt)

        device = QAudioDeviceInfo.defaultOutputDevice()
        if device.isNull() or not device.isFormatSupported(audio_format):
            self.status_label.setText(
                "The default audio device does not support 16-bit stereo playback at this rate."
            )
            return

        self.stop_preview()
        self._audio_buffer = QBuffer(self)
        self._audio_buffer.setData(to_pcm16_bytes(self._last_preview.audio))
        self._audio_buffer.open(QIODevice.ReadOnly)
        self._audio_output = QAudioOutput(audio_format, self)
        self._audio_output.stateChanged.connect(self._on_audio_state_changed)
        self._audio_output.start(self._audio_buffer)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Playing preview…")

    def stop_preview(self) -> None:
        if self._audio_output is not None:
            try:
                self._audio_output.stop()
            except RuntimeError:  # pragma: no cover - already torn down
                pass
            self._audio_output = None
        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer = None
        self.stop_button.setEnabled(False)

    def _on_audio_state_changed(self, state) -> None:  # pragma: no cover - needs a device
        if AUDIO_OUTPUT_AVAILABLE and state in (QAudio.IdleState, QAudio.StoppedState):
            self.stop_preview()

    # --- commit -------------------------------------------------------------

    def apply_changes(self) -> dict[str, Any]:
        """Publish the edited voice without closing the dialog."""

        self.basic_panel.flush()
        voice = self.voice_data()
        self._voice = copy.deepcopy(voice)
        if self._on_apply is not None:
            self._on_apply(copy.deepcopy(voice))
        self.voiceApplied.emit(copy.deepcopy(voice))
        self.status_label.setText("Applied to the voice editor.")
        return voice

    def accept(self) -> None:  # noqa: D102 - Qt slot
        self.apply_changes()
        self.stop_preview()
        super().accept()

    def reject(self) -> None:  # noqa: D102 - Qt slot
        self.stop_preview()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.stop_preview()
        if self._preview_thread is not None:
            self._preview_thread.quit()
            self._preview_thread.wait(2000)
        super().closeEvent(event)

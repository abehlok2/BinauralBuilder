"""Advanced HRTF acquisition: measure a head, or import one that was simulated.

Both routes here are advanced by design and off by default. The specification is
blunt about the order: choosing among generic HRTFs by localization test costs
nothing and should be tried first, because most of the benefit is there and
neither of these routes is forgiving. So the measurement tab stays disabled
until :data:`ADVANCED_FLAG` is set in the environment, and says why rather than
simply greying out.

The Mesh2HRTF tab runs no solver - the specification puts the solve outside the
application, and nothing here pretends otherwise. What it does is receive the
SOFA that came out and ask the questions that catch the errors this route
actually makes: a mesh scaled in millimetres, swapped ears, a grid that never
leaves the horizon. Each of those produces a file that loads cleanly and sounds
wrong, so a validation that only checked whether the file parses would pass
every one of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.hrtf.measurement import (
    ADVANCED_FLAG,
    MeasurementError,
    advanced_enabled,
    exponential_sweep,
    import_measurement,
)
from src.audio.sam_workbench.hrtf.mesh2hrtf import (
    compare_to_baseline,
    import_guidance,
    validate_simulated,
)

__all__ = ["SamMeasurementDialog"]


class SamMeasurementDialog(QDialog):
    """Swept-sine measurement import, and Mesh2HRTF validation, side by side."""

    responseImported = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Measure or import an HRTF")
        self.resize(820, 640)
        if parent is not None:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        self._recordings: list[Path] = []
        self._reference: Path | None = None
        self._response: np.ndarray | None = None
        self._simulated = None
        self._baseline = None

        self._build_ui()
        self._apply_gate()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_measure(), "Measure (Route B)")
        self.tabs.addTab(self._build_mesh(), "Simulate (Route C)")
        layout.addWidget(self.tabs, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_measure(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        self.gate_label = QLabel("")
        self.gate_label.setWordWrap(True)
        self.gate_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.gate_label)

        self.measure_body = QWidget()
        body = QVBoxLayout(self.measure_body)

        sweep_box = QGroupBox("Excitation")
        form = QFormLayout(sweep_box)
        self.start_hz = self._spin(20.0, 1.0, 1000.0, " Hz", "Lowest frequency in the sweep.")
        form.addRow("Start:", self.start_hz)
        self.end_hz = self._spin(20_000.0, 1000.0, 96_000.0, " Hz", "Highest frequency in the sweep.")
        form.addRow("End:", self.end_hz)
        self.duration_s = self._spin(2.0, 0.1, 30.0, " s", "Longer sweeps buy signal-to-noise.")
        form.addRow("Duration:", self.duration_s)
        self.rate_hz = self._spin(48_000.0, 8000.0, 192_000.0, " Hz", "Sample rate of the recording.")
        self.rate_hz.setDecimals(0)
        form.addRow("Sample rate:", self.rate_hz)

        save_sweep = QPushButton("Save sweep to play…")
        save_sweep.setToolTip("Write the excitation signal so it can be played through the rig.")
        save_sweep.clicked.connect(self.save_sweep)
        form.addRow("", save_sweep)
        body.addWidget(sweep_box)

        takes_box = QGroupBox("Recordings")
        takes = QVBoxLayout(takes_box)
        self.recording_list = QListWidget()
        self.recording_list.setToolTip(
            "Several takes of the same direction. Repeatability across takes is "
            "part of the quality check, so one take is never enough to trust."
        )
        takes.addWidget(self.recording_list)
        row = QHBoxLayout()
        add = QPushButton("Add takes…")
        add.clicked.connect(self.add_recordings)
        row.addWidget(add)
        reference = QPushButton("Set reference…")
        reference.setToolTip(
            "A recording of the loudspeaker and microphone without the head, "
            "divided out so the result is the head's response and not the rig's."
        )
        reference.clicked.connect(self.set_reference)
        row.addWidget(reference)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_recordings)
        row.addWidget(clear)
        row.addStretch(1)
        takes.addLayout(row)
        body.addWidget(takes_box, 1)

        action = QHBoxLayout()
        self.import_button = QPushButton("Deconvolve and check")
        self.import_button.clicked.connect(self.run_import)
        action.addWidget(self.import_button)
        self.save_response_button = QPushButton("Save impulse response…")
        self.save_response_button.setEnabled(False)
        self.save_response_button.clicked.connect(self.save_response)
        action.addWidget(self.save_response_button)
        action.addStretch(1)
        body.addLayout(action)

        self.quality_view = QTextEdit()
        self.quality_view.setReadOnly(True)
        self.quality_view.setMinimumHeight(140)
        body.addWidget(self.quality_view)

        outer.addWidget(self.measure_body, 1)
        return page

    def _build_mesh(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        steps = QGroupBox("Workflow")
        steps_layout = QVBoxLayout(steps)
        guidance = QLabel("\n".join(import_guidance()))
        guidance.setWordWrap(True)
        guidance.setToolTip(
            "Every step happens in the scanner or the solver. This application "
            "receives the result and checks it."
        )
        steps_layout.addWidget(guidance)
        outer.addWidget(steps)

        row = QHBoxLayout()
        load = QPushButton("Load simulated SOFA…")
        load.clicked.connect(self.load_simulated)
        row.addWidget(load)
        baseline = QPushButton("Load generic baseline…")
        baseline.setToolTip(
            "A known-good generic set. Comparing the interaural cues against it "
            "catches a rotation or a left/right swap that peak values agree on."
        )
        baseline.clicked.connect(self.load_baseline)
        row.addWidget(baseline)
        row.addStretch(1)
        outer.addLayout(row)

        self.issue_table = QTableWidget(0, 3)
        self.issue_table.setHorizontalHeaderLabels(["Severity", "Where", "What"])
        self.issue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.issue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.issue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.issue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.issue_table.setWordWrap(True)
        outer.addWidget(self.issue_table, 1)

        self.comparison_view = QTextEdit()
        self.comparison_view.setReadOnly(True)
        self.comparison_view.setMaximumHeight(140)
        outer.addWidget(self.comparison_view)
        return page

    @staticmethod
    def _spin(value: float, low: float, high: float, suffix: str, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setToolTip(tooltip)
        return spin

    # --- the advanced gate --------------------------------------------------

    def _apply_gate(self) -> None:
        enabled = advanced_enabled()
        self.measure_body.setEnabled(enabled)
        if enabled:
            self.gate_label.setText(
                "Advanced measurement tools are enabled. A measurement that fails "
                "its quality checks is refused rather than imported with a warning: "
                "a file written from a bad take outlives the warning."
            )
        else:
            self.gate_label.setText(
                "Measuring your own HRTF is an advanced feature and is off by "
                f"default. Set {ADVANCED_FLAG}=1 in the environment to enable it.\n\n"
                "Try the localization test first: choosing among generic HRTFs by "
                "listening costs nothing, needs no rig, and is where most of the "
                "benefit is."
            )

    # --- measurement --------------------------------------------------------

    def save_sweep(self, path: str | None = None) -> Path | None:
        import soundfile as sf

        if path is None:
            path, _filter = QFileDialog.getSaveFileName(
                self, "Save excitation sweep", "sweep.wav", "WAV (*.wav)"
            )
        if not path:
            return None
        sweep = exponential_sweep(
            float(self.start_hz.value()),
            float(self.end_hz.value()),
            float(self.duration_s.value()),
            float(self.rate_hz.value()),
        )
        destination = Path(path)
        sf.write(str(destination), sweep, int(self.rate_hz.value()))
        self.status_label.setText(f"Sweep written to {destination}")
        return destination

    def add_recordings(self, paths=None) -> None:
        if paths is None:
            paths, _filter = QFileDialog.getOpenFileNames(
                self, "Add recorded takes", "", "Audio (*.wav *.flac);;All files (*)"
            )
        for path in paths or ():
            self._recordings.append(Path(path))
            self.recording_list.addItem(str(path))

    def set_reference(self, path: str | None = None) -> None:
        if path is None:
            path, _filter = QFileDialog.getOpenFileName(
                self, "Set reference recording", "", "Audio (*.wav *.flac);;All files (*)"
            )
        if not path:
            return
        self._reference = Path(path)
        self.status_label.setText(f"Reference: {self._reference.name}")

    def clear_recordings(self) -> None:
        self._recordings.clear()
        self._reference = None
        self.recording_list.clear()
        self.status_label.setText("")

    def run_import(self) -> bool:
        """Deconvolve, window, and check. A failing measurement is refused."""

        import soundfile as sf

        if not self._recordings:
            self.status_label.setText("Add at least one recorded take.")
            return False
        try:
            takes = []
            for path in self._recordings:
                audio, _rate = sf.read(str(path), dtype="float64", always_2d=False)
                takes.append(np.asarray(audio, dtype=np.float64).reshape(-1))
            reference = None
            if self._reference is not None:
                data, _rate = sf.read(str(self._reference), dtype="float64", always_2d=False)
                reference = np.asarray(data, dtype=np.float64).reshape(-1)

            response, report = import_measurement(
                takes,
                sample_rate_hz=float(self.rate_hz.value()),
                sweep_range_hz=(float(self.start_hz.value()), float(self.end_hz.value())),
                sweep_duration_s=float(self.duration_s.value()),
                reference=reference,
            )
        except MeasurementError as error:
            self._response = None
            self.save_response_button.setEnabled(False)
            self.quality_view.setPlainText(str(error))
            self.status_label.setText("Measurement refused.")
            return False
        except Exception as error:  # pragma: no cover - file and codec failures
            self.quality_view.setPlainText(str(error))
            self.status_label.setText("Could not read the recordings.")
            return False

        self._response = response
        self.save_response_button.setEnabled(True)
        self.quality_view.setPlainText(
            "\n".join(
                f"{key}: {value}" for key, value in sorted(report.describe().items())
            )
        )
        # The window zeroes the tail rather than truncating it, so the array
        # length says nothing. What matters is where the arrival sits and how
        # much of it survived the cut.
        rate = float(self.rate_hz.value())
        peak = int(np.argmax(np.abs(response)))
        retained = int(np.max(np.nonzero(response)[0]) + 1) if np.any(response) else 0
        self.status_label.setText(
            f"Passed: arrival at tap {peak}, kept to tap {retained} "
            f"({max(0, retained - peak) / rate * 1000.0:.1f} ms after the peak), "
            f"signal-to-noise {report.snr_db:.1f} dB."
        )
        self.responseImported.emit(response)
        return True

    def save_response(self, path: str | None = None) -> Path | None:
        import soundfile as sf

        if self._response is None:
            return None
        if path is None:
            path, _filter = QFileDialog.getSaveFileName(
                self, "Save impulse response", "measured.wav", "WAV (*.wav)"
            )
        if not path:
            return None
        destination = Path(path)
        sf.write(str(destination), self._response, int(self.rate_hz.value()))
        self.status_label.setText(f"Impulse response written to {destination}")
        return destination

    # --- Mesh2HRTF ----------------------------------------------------------

    def load_simulated(self, path: str | None = None) -> bool:
        from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

        if path is None:
            path, _filter = QFileDialog.getOpenFileName(
                self, "Load simulated SOFA", "", "SOFA files (*.sofa);;All files (*)"
            )
        if not path:
            return False
        try:
            self._simulated = load_sofa(path)
        except Exception as error:
            self.status_label.setText(f"Could not read that file: {error}")
            return False
        self.validate()
        return True

    def load_baseline(self, path: str | None = None) -> bool:
        from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

        if path is None:
            path, _filter = QFileDialog.getOpenFileName(
                self, "Load baseline SOFA", "", "SOFA files (*.sofa);;All files (*)"
            )
        if not path:
            return False
        try:
            self._baseline = load_sofa(path)
        except Exception as error:
            self.status_label.setText(f"Could not read that file: {error}")
            return False
        self.validate()
        return True

    def validate(self) -> dict[str, Any]:
        """Check the simulated set, and compare it with the baseline if there is one."""

        if self._simulated is None:
            return {}
        issues = validate_simulated(self._simulated)
        self.issue_table.setRowCount(len(issues))
        for row, issue in enumerate(issues):
            for column, text in enumerate((issue.severity, issue.path, issue.message)):
                item = QTableWidgetItem(str(text))
                item.setToolTip(str(issue.message))
                self.issue_table.setItem(row, column, item)
        self.issue_table.resizeRowsToContents()

        comparison = compare_to_baseline(self._simulated, self._baseline)
        lines = [
            f"Largest interaural delay: {comparison.max_itd_us:.0f} µs",
            f"Largest interaural level difference: {comparison.max_ild_db:.1f} dB",
        ]
        if self._baseline is not None:
            lines += [
                f"Baseline delay: {comparison.baseline_max_itd_us:.0f} µs "
                f"(ratio {comparison.itd_ratio:.2f})",
                f"Delay-against-azimuth correlation with the baseline: "
                f"{comparison.correlation:.2f}",
            ]
        else:
            lines.append("No baseline loaded, so only physical plausibility was checked.")
        for issue in comparison.issues:
            lines.append(f"[{issue.severity}] {issue.message}")
        self.comparison_view.setPlainText("\n".join(lines))

        errors = [issue for issue in (*issues, *comparison.issues) if issue.severity == "error"]
        self.status_label.setText(
            f"{len(errors)} error(s), {len(issues) + len(comparison.issues) - len(errors)} other note(s)."
            if issues or comparison.issues
            else "No problems found."
        )
        return comparison.describe()

"""A blinded localization test: point at what you heard, one trial at a time.

The specification's argument for this is the cheapest thing in the whole plan:
before measuring a head or simulating a mesh, find out which generic HRTF you
already localise best with. That is a listening test, and a listening test is
only worth anything if it is blind - so the runner shows a scrambled label and
never the dataset name, and the reveal happens once, at the end, next to the
scores.

The answer is given by pointing at a head rather than typing an azimuth. That
is not decoration: the thing being measured is where the sound seemed to be,
and a picture of a head is the interface that asks that question directly.
Front/back reversals - the failure that most separates one HRTF from another -
are impossible to express in a slider that stops at ninety degrees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.experiment.localization import (
    LocalizationSession,
    LocalizationTrial,
    TrialResponse,
    build_session,
    score_responses,
)

from .head_view import HeadDirectionSelector

__all__ = ["SamLocalizationDialog"]

SIGNAL_TYPES = ("pink_noise_burst", "click", "speech")


class SamLocalizationDialog(QDialog):
    """Set up, run, and score a blinded localization test."""

    sessionFinished = pyqtSignal(dict)

    def __init__(
        self,
        candidates: Sequence[str] = (),
        *,
        play_trial: Callable[[LocalizationTrial], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Localization test")
        self.resize(760, 620)
        if parent is not None:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        self._play_trial = play_trial
        self._session: LocalizationSession | None = None
        self._responses: list[TrialResponse] = []
        self._index = 0

        self._build_ui()
        self.set_candidates(candidates)

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_setup())
        self.pages.addWidget(self._build_runner())
        self.pages.addWidget(self._build_results())
        layout.addWidget(self.pages, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _build_setup(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        intro = QLabel(
            "Choosing among generic HRTFs by listening costs nothing and is the "
            "first thing worth trying. Pick the candidates to compare; the test "
            "hides which is which until it is scored."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        box = QGroupBox("Candidates")
        box_layout = QVBoxLayout(box)
        self.candidate_list = QListWidget()
        self.candidate_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.candidate_list.setToolTip("Select two or more datasets to compare.")
        box_layout.addWidget(self.candidate_list)
        outer.addWidget(box, 1)

        form = QFormLayout()
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 10)
        self.repeats_spin.setValue(2)
        self.repeats_spin.setToolTip(
            "How many times each direction is presented per candidate. Repeats "
            "are what let the score separate a real difference from a guess."
        )
        form.addRow("Repeats:", self.repeats_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setToolTip("Fixes the trial order and the blinded labels, so a run can be repeated.")
        form.addRow("Seed:", self.seed_spin)

        self.distance_check = QCheckBox("ask for distance as well as direction")
        self.distance_check.setChecked(True)
        form.addRow("", self.distance_check)
        outer.addLayout(form)

        signals = QGroupBox("Signals")
        signal_row = QHBoxLayout(signals)
        self.signal_checks: dict[str, QCheckBox] = {}
        for name in SIGNAL_TYPES:
            check = QCheckBox(name.replace("_", " "))
            check.setChecked(name != "speech")
            self.signal_checks[name] = check
            signal_row.addWidget(check)
        outer.addWidget(signals)

        start_row = QHBoxLayout()
        start_row.addStretch(1)
        self.start_button = QPushButton("Start blinded test")
        self.start_button.clicked.connect(self.start)
        start_row.addWidget(self.start_button)
        outer.addLayout(start_row)
        return page

    def _build_runner(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        header = QHBoxLayout()
        self.trial_label = QLabel("")
        header.addWidget(self.trial_label, 1)
        self.progress = QProgressBar()
        header.addWidget(self.progress, 1)
        outer.addLayout(header)

        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Present this trial. You may replay it as often as you like.")
        self.play_button.clicked.connect(self.play_current)
        outer.addWidget(self.play_button)

        outer.addWidget(QLabel("Point at where you heard it:"))
        self.selector = HeadDirectionSelector()
        outer.addWidget(self.selector, 1)

        form = QFormLayout()
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.1, 20.0)
        self.distance_spin.setDecimals(2)
        self.distance_spin.setValue(1.0)
        self.distance_spin.setSuffix(" m")
        form.addRow("Distance:", self.distance_spin)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 5)
        self.confidence_slider.setValue(3)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setToolTip(
            "How sure you are. A confident wrong answer counts for more than an "
            "unsure one, which is what makes the score mean something."
        )
        self.confidence_label = QLabel("3 of 5")
        confidence_row = QHBoxLayout()
        confidence_row.addWidget(self.confidence_slider, 1)
        confidence_row.addWidget(self.confidence_label)
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_label.setText(f"{value} of 5")
        )
        form.addRow("Confidence:", confidence_row)
        outer.addLayout(form)

        row = QHBoxLayout()
        row.addStretch(1)
        self.skip_button = QPushButton("Skip")
        self.skip_button.setToolTip("Record nothing for this trial and move on.")
        self.skip_button.clicked.connect(self.skip_current)
        row.addWidget(self.skip_button)
        self.next_button = QPushButton("Record and continue")
        self.next_button.clicked.connect(self.record_current)
        row.addWidget(self.next_button)
        outer.addLayout(row)
        return page

    def _build_results(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("Best localised first. The labels are revealed here, not before."))

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["Candidate", "Angular error", "Front/back", "Elevation", "Distance", "Confidence"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        outer.addWidget(self.results_table, 1)

        row = QHBoxLayout()
        self.export_button = QPushButton("Export results…")
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        again = QPushButton("Run another test")
        again.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        row.addWidget(again)
        row.addStretch(1)
        outer.addLayout(row)
        return page

    # --- setup --------------------------------------------------------------

    def set_candidates(self, candidates: Sequence[str]) -> None:
        self.candidate_list.clear()
        for name in candidates:
            self.candidate_list.addItem(str(name))
        for row in range(self.candidate_list.count()):
            self.candidate_list.item(row).setSelected(True)

    def selected_candidates(self) -> tuple[str, ...]:
        return tuple(item.text() for item in self.candidate_list.selectedItems())

    # --- running ------------------------------------------------------------

    def start(self) -> bool:
        candidates = self.selected_candidates()
        if len(candidates) < 2:
            self.status_label.setText("Select at least two candidates to compare.")
            return False
        signals = tuple(name for name, check in self.signal_checks.items() if check.isChecked())
        if not signals:
            self.status_label.setText("Select at least one signal type.")
            return False

        self._session = build_session(
            candidates,
            seed=int(self.seed_spin.value()),
            repeats=int(self.repeats_spin.value()),
            signal_types=signals,
            include_distance=bool(self.distance_check.isChecked()),
        )
        self._responses = []
        self._index = 0
        self.progress.setRange(0, len(self._session.trials))
        self.pages.setCurrentIndex(1)
        self._load_trial()
        self.status_label.setText(
            f"{len(self._session.trials)} trials over {len(candidates)} candidates."
        )
        return True

    @property
    def session(self) -> LocalizationSession | None:
        return self._session

    @property
    def responses(self) -> tuple[TrialResponse, ...]:
        return tuple(self._responses)

    def current_trial(self) -> LocalizationTrial | None:
        if self._session is None or self._index >= len(self._session.trials):
            return None
        return self._session.trials[self._index]

    def _load_trial(self) -> None:
        trial = self.current_trial()
        if trial is None:
            self.finish()
            return
        total = len(self._session.trials) if self._session else 0
        # The blinded label is the only identity shown. Naming the dataset here
        # would turn the test into a preference for a familiar name.
        self.trial_label.setText(
            f"Trial {self._index + 1} of {total} — set {trial.label}, "
            f"{trial.signal_type.replace('_', ' ')}"
        )
        self.progress.setValue(self._index)
        self.distance_spin.setEnabled(bool(self.distance_check.isChecked()))
        self.play_button.setEnabled(self._play_trial is not None)
        if self._play_trial is None:
            self.play_button.setText("Play (no renderer connected)")
        self.play_current()

    def play_current(self) -> None:
        trial = self.current_trial()
        if trial is None or self._play_trial is None:
            return
        try:
            self._play_trial(trial)
        except Exception as error:  # pragma: no cover - surfaced, not swallowed
            self.status_label.setText(f"Could not present this trial: {error}")

    def record_current(self) -> None:
        trial = self.current_trial()
        if trial is None:
            return
        self._responses.append(
            TrialResponse(
                trial_id=trial.id,
                azimuth_deg=float(self.selector.azimuth_deg),
                elevation_deg=float(self.selector.elevation_deg),
                distance_m=(
                    float(self.distance_spin.value()) if self.distance_check.isChecked() else None
                ),
                confidence=int(self.confidence_slider.value()),
            )
        )
        self._advance()

    def skip_current(self) -> None:
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        if self._session is not None and self._index >= len(self._session.trials):
            self.finish()
        else:
            self._load_trial()

    # --- scoring ------------------------------------------------------------

    def finish(self) -> dict[str, Any]:
        """Score what was answered and reveal which set was which."""

        if self._session is None:
            return {}
        scores = score_responses(self._session, self._responses)
        self.results_table.setRowCount(len(scores))
        for row, score in enumerate(scores):
            values = (
                score.candidate,
                f"{score.angular_error_deg:.1f}°",
                f"{score.front_back_reversals} ({score.front_back_rate * 100:.0f}%)",
                f"{score.elevation_error_deg:.1f}°",
                f"{score.distance_error_m:.2f} m",
                f"{score.mean_confidence:.1f}",
            )
            for column, text in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(text))

        payload = {
            "session": self._session.describe(),
            "responses": [response.describe() for response in self._responses],
            "scores": [score.describe() for score in scores],
        }
        self._last_payload = payload
        self.progress.setValue(len(self._session.trials))
        self.pages.setCurrentIndex(2)
        self.status_label.setText(
            f"Best: {scores[0].candidate}" if scores else "No responses were recorded."
        )
        self.sessionFinished.emit(payload)
        return payload

    def export(self, path: str | None = None) -> Path | None:
        payload = getattr(self, "_last_payload", None)
        if not payload:
            self.status_label.setText("Nothing to export yet.")
            return None
        if path is None:
            path, _filter = QFileDialog.getSaveFileName(
                self, "Export localization results", "localization.json", "JSON (*.json)"
            )
        if not path:
            return None
        destination = Path(path)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.status_label.setText(f"Written to {destination}")
        return destination

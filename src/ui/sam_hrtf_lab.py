"""The HRTF Lab: choosing an asset, inspecting it, and auditioning subjects.

Three jobs in one panel:

* **selection** - point at a dataset folder, pick a subject, processing
  variant, sample rate, delay policy, interpolation mode, and headphone
  correction;
* **inspection** - the ingest report, so a user sees what an asset actually
  contains before trusting what they hear;
* **audition** - the structured subject test, because HRTF compatibility is a
  perceptual question and the only honest answer comes from listening.

Storage lives in ``QStandardPaths.AppLocalDataLocation``, not in the
repository: the dataset is large and licensed, and a checkout is the wrong
place for it. Only the chosen SOFA path and subject identifier are written into
the project.

Without a SOFA backend installed the panel still opens; the controls disable
themselves and say what to install, rather than the application failing to
start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import QStandardPaths, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.hrtf import (
    AUDITION_AZIMUTHS_DEG,
    AUDITION_ELEVATIONS_DEG,
    CORRECTION_MODES,
    DELAY_POLICIES,
    INTERPOLATION_MODES,
    KEMAR_VARIANTS,
    PROCESSING_VARIANTS,
    RATING_CRITERIA,
    SIGNAL_TYPES,
    SONICOM,
    SONICOM_SAMPLE_RATES_HZ,
    RatingStore,
    SubjectRating,
    available_backend,
    datasets_root,
    describe_subject,
    discover_assets,
    missing_dependency_message,
    rank_subjects,
    set_data_root,
    storage_report,
)

__all__ = ["SamHrtfLab", "SOURCE_FILTERS", "application_data_root"]

#: How to narrow a subject list: everything, the human subjects, or the KEMAR
#: dummy heads that make a stable generic fallback and a pinna-size comparison.
SOURCE_FILTERS = ("Measured", "KEMAR", "Synthetic")

_RATING_MINIMUM = 1
_RATING_MAXIMUM = 5


def application_data_root() -> Path:
    """The per-user application data directory, as Qt reports it.

    Qt is the authority here because it already knows each platform's
    convention; the Qt-free core is pointed at the same place so both halves
    agree on one location.
    """

    location = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    return Path(location) if location else Path.home() / ".binauralbuilder"


class SamHrtfLab(QWidget):
    """Asset selection, the ingest report, and the subject audition test."""

    assetSelected = pyqtSignal(dict)
    auditionRequested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None, *, sample_rate_hz: int = 44_100) -> None:
        super().__init__(parent)
        set_data_root(application_data_root())

        self._sample_rate_hz = int(sample_rate_hz)
        self._assets: list[Any] = []
        self._dataset_root: Path = datasets_root()
        self._report: Any = None
        self._prepared: Any = None
        self._ratings = RatingStore().load()

        self._build_ui()
        self._apply_backend_state()
        self.refresh_assets()
        self._refresh_ranking()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Where the data lives.
        location_box = QGroupBox("Dataset location")
        location_layout = QHBoxLayout(location_box)
        self.location_label = QLabel(str(self._dataset_root))
        self.location_label.setWordWrap(True)
        self.location_label.setToolTip(
            "Datasets live in the application data directory, not in the project or the "
            "repository. Only the chosen file path is saved with a project."
        )
        location_layout.addWidget(self.location_label, 1)
        self.browse_button = QPushButton("Choose folder…")
        self.browse_button.clicked.connect(self.choose_dataset_folder)
        location_layout.addWidget(self.browse_button)
        layout.addWidget(location_box)

        # Selection.
        selection_box = QGroupBox("Asset")
        selection_form = QFormLayout(selection_box)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([SONICOM])
        selection_form.addRow("Dataset:", self.dataset_combo)

        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_FILTERS))
        self.source_combo.setToolTip(
            "KEMAR is a dummy head: a stable generic fallback, and the way to compare "
            "pinna sizes or head-versus-torso rotation."
        )
        self.source_combo.currentTextChanged.connect(lambda _: self._populate_subjects())
        selection_form.addRow("Source:", self.source_combo)

        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(lambda _: self._on_subject_changed())
        selection_form.addRow("Subject:", self.subject_combo)

        self.processing_combo = QComboBox()
        self.processing_combo.addItems(["Any", *PROCESSING_VARIANTS])
        self.processing_combo.currentTextChanged.connect(lambda _: self._populate_subjects())
        selection_form.addRow("Processing:", self.processing_combo)

        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["Any", *[f"{rate} Hz" for rate in SONICOM_SAMPLE_RATES_HZ]])
        self.rate_combo.currentTextChanged.connect(lambda _: self._populate_subjects())
        selection_form.addRow("Sample rate:", self.rate_combo)

        self.delay_combo = QComboBox()
        self.delay_combo.addItems(list(DELAY_POLICIES))
        self.delay_combo.setToolTip(
            "A nonzero Data_Delay is never ignored: either it is applied to the "
            "responses on load, or it is kept and applied at render time."
        )
        selection_form.addRow("Interaural delay:", self.delay_combo)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(list(INTERPOLATION_MODES))
        self.interpolation_combo.setToolTip(
            "Nearest with crossfade never blends raw responses, so it cannot smear a "
            "transient. The others blend only after delay alignment."
        )
        selection_form.addRow("Interpolation:", self.interpolation_combo)

        self.headphone_combo = QComboBox()
        self.headphone_combo.addItems(list(CORRECTION_MODES))
        self.headphone_combo.setToolTip(
            "Applied after binaural rendering. The HD650 measurement is correct only "
            "for HD650 headphones."
        )
        selection_form.addRow("Headphone correction:", self.headphone_combo)

        action_row = QHBoxLayout()
        self.load_button = QPushButton("Load and inspect")
        self.load_button.clicked.connect(self.load_selected_asset)
        action_row.addWidget(self.load_button)
        self.quality_label = QLabel("Quality: no asset loaded")
        self.quality_label.setToolTip(
            "Normal, suspected outlier, or missing resources - from the ingest report."
        )
        action_row.addWidget(self.quality_label, 1)
        selection_form.addRow(action_row)
        layout.addWidget(selection_box)

        # The ingest report.
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMinimumHeight(120)
        layout.addWidget(self.report_view, 1)

        layout.addWidget(self._build_audition_box(), 1)

    def _build_audition_box(self) -> QWidget:
        box = QGroupBox("Subject audition")
        layout = QVBoxLayout(box)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Direction:"))
        self.azimuth_combo = QComboBox()
        self.azimuth_combo.addItems([f"{value:+.0f}°" for value in AUDITION_AZIMUTHS_DEG])
        controls.addWidget(self.azimuth_combo)
        self.elevation_combo = QComboBox()
        self.elevation_combo.addItems([f"{value:+.0f}°" for value in AUDITION_ELEVATIONS_DEG])
        self.elevation_combo.setCurrentIndex(list(AUDITION_ELEVATIONS_DEG).index(0.0))
        controls.addWidget(self.elevation_combo)

        controls.addWidget(QLabel("Signal:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(list(SIGNAL_TYPES))
        self.signal_combo.setToolTip(
            "Each signal stresses something different: clicks and percussion expose "
            "smearing, noise exposes spectral cues, motion exposes interpolation."
        )
        controls.addWidget(self.signal_combo)

        controls.addWidget(QLabel("Length:"))
        self.audition_seconds = QDoubleSpinBox()
        self.audition_seconds.setRange(0.25, 10.0)
        self.audition_seconds.setValue(2.0)
        self.audition_seconds.setSuffix(" s")
        controls.addWidget(self.audition_seconds)

        self.audition_button = QPushButton("Audition")
        self.audition_button.clicked.connect(self.request_audition)
        controls.addWidget(self.audition_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Ratings.
        ratings_widget = QWidget()
        ratings_layout = QGridLayout(ratings_widget)
        self.rating_sliders: dict[str, QSlider] = {}
        self.rating_values: dict[str, QLabel] = {}
        for row, criterion in enumerate(RATING_CRITERIA):
            label = QLabel(criterion.replace("_", " ").capitalize())
            slider = QSlider(Qt.Horizontal)
            slider.setRange(_RATING_MINIMUM, _RATING_MAXIMUM)
            slider.setValue(3)
            slider.setTickPosition(QSlider.TicksBelow)
            value_label = QLabel("3")
            slider.valueChanged.connect(
                lambda value, target=value_label: target.setText(str(value))
            )
            ratings_layout.addWidget(label, row // 2, (row % 2) * 3)
            ratings_layout.addWidget(slider, row // 2, (row % 2) * 3 + 1)
            ratings_layout.addWidget(value_label, row // 2, (row % 2) * 3 + 2)
            self.rating_sliders[criterion] = slider
            self.rating_values[criterion] = value_label
        layout.addWidget(ratings_widget)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Profile:"))
        self.profile_edit = QLineEdit("default")
        self.profile_edit.setToolTip(
            "Ratings are filed per profile: a subject that suits broadband noise may "
            "be wrong for a low-frequency SAM carrier."
        )
        save_row.addWidget(self.profile_edit)
        self.save_rating_button = QPushButton("Save rating")
        self.save_rating_button.clicked.connect(self.save_rating)
        save_row.addWidget(self.save_rating_button)
        save_row.addStretch(1)
        layout.addLayout(save_row)

        self.ranking_table = QTableWidget(0, 3)
        self.ranking_table.setHorizontalHeaderLabels(["Subject", "Score", "Ratings"])
        self.ranking_table.setMinimumHeight(120)
        layout.addWidget(self.ranking_table)
        return box

    # --- dependency state ---------------------------------------------------

    def _apply_backend_state(self) -> None:
        """Disable what cannot work, and say what to install."""

        self.backend = available_backend()
        if self.backend is not None:
            return
        for widget in (self.load_button, self.audition_button, self.subject_combo):
            widget.setEnabled(False)
        self.quality_label.setText("Quality: missing resources")
        self.report_view.setPlainText(missing_dependency_message())

    # --- assets -------------------------------------------------------------

    @property
    def dataset_root(self) -> Path:
        return self._dataset_root

    def choose_dataset_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose the HRTF dataset folder", str(self._dataset_root)
        )
        if chosen:
            self.set_dataset_root(chosen)

    def set_dataset_root(self, root: str | Path) -> None:
        self._dataset_root = Path(root)
        self.location_label.setText(str(self._dataset_root))
        self.refresh_assets()

    def refresh_assets(self) -> None:
        """Rescan the dataset folder and repopulate the subject list."""

        self._assets = discover_assets(self._dataset_root)
        self._populate_subjects()

    def _matching_assets(self) -> list[Any]:
        source = self.source_combo.currentText()
        processing = self.processing_combo.currentText()
        rate_text = self.rate_combo.currentText()

        matches = []
        for asset in self._assets:
            if source == "KEMAR" and not asset.is_kemar:
                continue
            if source == "Measured" and asset.is_kemar:
                continue
            if source == "Synthetic" and asset.subject:
                # Anything the naming does not identify as a dataset subject.
                continue
            if processing != "Any" and asset.processing != processing:
                continue
            if rate_text != "Any" and asset.sample_rate_hz != int(rate_text.split()[0]):
                continue
            matches.append(asset)
        return matches

    def _populate_subjects(self) -> None:
        matches = self._matching_assets()
        blocked = self.subject_combo.blockSignals(True)
        try:
            self.subject_combo.clear()
            for asset in matches:
                self.subject_combo.addItem(asset.label, asset)
        finally:
            self.subject_combo.blockSignals(blocked)

        if not matches:
            message = (
                f"No SOFA assets found under {self._dataset_root}."
                if self.backend
                else missing_dependency_message()
            )
            self.report_view.setPlainText(message)
        self.load_button.setEnabled(bool(matches) and self.backend is not None)

    def _on_subject_changed(self) -> None:
        asset = self.subject_combo.currentData()
        if asset is not None and asset.subject:
            self.report_view.setPlainText(
                f"{describe_subject(asset.subject)}\n{asset.path}\n\n"
                "Load and inspect to read the file's own report."
            )

    @property
    def selected_asset(self) -> Any:
        return self.subject_combo.currentData()

    def load_selected_asset(self) -> dict[str, Any] | None:
        """Load, validate, and prepare the selected asset."""

        asset = self.selected_asset
        if asset is None:
            return None
        return self.load_asset_path(asset.path, subject=asset.subject)

    def load_asset_path(self, path: str | Path, *, subject: str = "") -> dict[str, Any] | None:
        """Load a specific SOFA file, whatever the selector currently shows."""

        from src.audio.sam_workbench.hrtf import (
            default_cache,
            load_sofa,
            validate_sofa_file,
        )

        try:
            sofa_file = load_sofa(path)
            self._report = validate_sofa_file(sofa_file, project_sample_rate_hz=self._sample_rate_hz)
            self._prepared = (
                default_cache().get(
                    path,
                    sample_rate_hz=self._sample_rate_hz,
                    delay_policy=self.delay_combo.currentText(),
                    subject=subject,
                )
                if self._report.is_renderable
                else None
            )
        except Exception as error:
            self._report = None
            self._prepared = None
            self.quality_label.setText("Quality: missing resources")
            self.report_view.setPlainText(f"Could not load {path}:\n{error}")
            return None

        self._show_report()
        selection = self.asset_selection()
        self.assetSelected.emit(selection)
        return selection

    def _show_report(self) -> None:
        if self._report is None:
            return
        metadata = self._report.metadata
        lines = [
            f"{metadata.get('database_name', 'unknown database')}  ·  "
            f"subject {metadata.get('subject', '?')}",
            f"{metadata.get('path', '')}",
            f"sha256 {metadata.get('sha256', '')[:16]}…  ·  read with {metadata.get('backend', '?')}",
            "",
            f"Convention: {metadata.get('convention', '?')} {metadata.get('convention_version', '')}",
            f"Measurements: {metadata.get('measurements', 0)}  ·  receivers "
            f"{metadata.get('receivers', 0)}  ·  {metadata.get('taps', 0)} taps",
            f"Sample rate: {metadata.get('sample_rate_hz', 0):.0f} Hz "
            f"(project {self._sample_rate_hz} Hz)",
            f"Azimuth {metadata.get('azimuth_deg_range', [0, 0])}  ·  elevation "
            f"{metadata.get('elevation_deg_range', [0, 0])}",
            f"Direction spacing: {metadata.get('median_direction_spacing_deg', float('nan')):.1f}°",
            f"Data_Delay: shape {metadata.get('data_delay_shape', [])}, nonzero "
            f"{metadata.get('data_delay_nonzero', False)}  ·  policy "
            f"{self.delay_combo.currentText()}",
            f"Licence: {metadata.get('license', '(not stated)')}",
        ]
        if self._report.issues:
            lines.extend(["", "Report:"])
            lines.extend(
                f"  [{issue.severity}] {issue.path}: {issue.message}" for issue in self._report.issues
            )
        self.report_view.setPlainText("\n".join(lines))
        self.quality_label.setText(f"Quality: {self._report.quality}")

    def asset_selection(self) -> dict[str, Any]:
        """What the project should store about this choice."""

        asset = self.selected_asset
        return {
            "hrtfAsset": str(asset.path) if asset is not None else "",
            "hrtfSubject": asset.subject if asset is not None else "",
            "hrtfInterpolation": self.interpolation_combo.currentText(),
            "hrtfDelayPolicy": self.delay_combo.currentText(),
            "headphoneCorrection": self.headphone_combo.currentText(),
            "quality": self._report.quality if self._report is not None else "missing resources",
            "prepared": self._prepared is not None,
        }

    @property
    def prepared_dataset(self) -> Any:
        return self._prepared

    @property
    def report(self) -> Any:
        return self._report

    # --- audition -----------------------------------------------------------

    def audition_request(self) -> dict[str, Any]:
        """The direction, signal, and asset to audition."""

        azimuth = AUDITION_AZIMUTHS_DEG[max(0, self.azimuth_combo.currentIndex())]
        elevation = AUDITION_ELEVATIONS_DEG[max(0, self.elevation_combo.currentIndex())]
        asset = self.selected_asset
        return {
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "signal_type": self.signal_combo.currentText(),
            "duration_s": float(self.audition_seconds.value()),
            "asset": str(asset.path) if asset is not None else "",
            "subject": asset.subject if asset is not None else "",
            "interpolation": self.interpolation_combo.currentText(),
        }

    def request_audition(self) -> dict[str, Any]:
        request = self.audition_request()
        self.auditionRequested.emit(request)
        return request

    # --- ratings ------------------------------------------------------------

    def current_scores(self) -> dict[str, int]:
        return {criterion: slider.value() for criterion, slider in self.rating_sliders.items()}

    def save_rating(self) -> SubjectRating | None:
        """Record the current scores for the selected subject."""

        asset = self.selected_asset
        subject = (asset.subject if asset is not None else "") or "unnamed"
        rating = SubjectRating(
            subject=subject,
            scores=self.current_scores(),
            profile=self.profile_edit.text().strip() or "default",
            signal_type=self.signal_combo.currentText(),
            asset_path=str(asset.path) if asset is not None else "",
        )
        self._ratings.add(rating)
        self._ratings.save()
        self._refresh_ranking()
        return rating

    @property
    def ratings(self) -> RatingStore:
        return self._ratings

    def _refresh_ranking(self) -> None:
        profile = self.profile_edit.text().strip() or "default"
        ranked = rank_subjects(self._ratings.ratings, profile=profile)
        self.ranking_table.setRowCount(len(ranked))
        for row, entry in enumerate(ranked):
            self.ranking_table.setItem(row, 0, QTableWidgetItem(describe_subject(entry.subject)))
            self.ranking_table.setItem(row, 1, QTableWidgetItem(f"{entry.score:.2f}"))
            self.ranking_table.setItem(row, 2, QTableWidgetItem(str(entry.ratings)))

    def storage_summary(self) -> dict[str, Any]:
        """Where everything is kept on this machine, for the status line."""

        summary = storage_report().to_dict()
        summary["dataset_root"] = str(self._dataset_root)
        summary["kemar_variants"] = dict(KEMAR_VARIANTS)
        return summary

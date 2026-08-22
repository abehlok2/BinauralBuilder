"""The HRTF Lab: choose a subject, audition it, modify it, and rank what you heard.

Four tabs, matching the order the work actually happens in:

* **Asset** - point at a SONICOM library, narrow it down to one SOFA file, and
  set the policies that decide how it is rendered.
* **Audition** - listen to that subject at the directions and on the material
  that reveal whether it suits *you*, and score it.
* **Modify** - scale the cues the measurement encodes, see what that does, and
  write the result out as a derived asset.
* **Ranking** - see which subjects your own scores favour, per profile.

The dataset is large and licensed, so nothing here bundles or downloads it.
The library folder defaults to the platform's application-data location rather
than anywhere inside the repository; only the chosen path and subject
identifier are written into the project.

Heavy imports stay inside the methods that need them. Selecting a renderer that
is not HRTF, or opening this tab without a dataset installed, must not require
``sofar`` or ``h5py`` to be present.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .hrtf_coverage_plot import CoveragePlotWidget
from .sam_hrtf_modify_panel import SamHrtfModifyPanel

__all__ = ["SamHrtfLab", "AuditionWorker", "application_data_root"]

#: Options schema written into the project alongside the asset path.
HRTF_OPTIONS_SCHEMA_VERSION = 1

DEFAULT_PROFILE = "default"

#: Label, stored value. The order is the order the modes are meant to be
#: adopted, so the list doubles as guidance.
INTERPOLATION_CHOICES = (
    ("Nearest with crossfade", "nearest"),
    ("Three-neighbour (unit-sphere distance)", "three_neighbor"),
    ("Spherical triangular (barycentric)", "spherical_triangular"),
    ("Delay and magnitude (minimum phase)", "delay_magnitude"),
    ("Spherical harmonic", "spherical_harmonic"),
)

DELAY_CHOICES = (
    ("Embedded in HRIR", "bake_delay_into_ir"),
    ("SOFA Data.Delay", "preserve_external_delay"),
)

SOURCE_CHOICES = (
    ("Measured", "measured"),
    ("Synthetic", "synthetic"),
    ("KEMAR", "kemar"),
)

PROCESSING_LABELS = {
    "Windowed": "Windowed",
    "FreeFieldComp": "Free-field compensated",
    "FreeFieldCompMinPhase": "Free-field compensated minimum-phase",
}

HEADPHONE_CHOICES = (
    ("Off", "off"),
    ("SONICOM Sennheiser HD650", "sonicom_hd650"),
    ("User impulse response", "user_impulse_response"),
    ("User equalisation curve", "user_equalization_curve"),
)

_QUALITY_COLOURS = {
    "normal": "#2f7d32",
    "suspected outlier": "#b26a00",
    "missing resources": "#b3261e",
}


def application_data_root() -> Path:
    """The per-user application data folder Qt reports for this application.

    Falls back to the core's own platform logic when Qt cannot answer, which
    keeps the two in step rather than letting the GUI and the renderer disagree
    about where ratings live.
    """

    try:
        from PyQt5.QtCore import QStandardPaths

        location = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        if location:
            return Path(location)
    except Exception:  # pragma: no cover - minimal Qt build
        pass
    from src.audio.sam_workbench.hrtf.storage import data_root

    return data_root()


class AuditionWorker(QObject):
    """Renders one audition clip off the GUI thread."""

    rendered = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        asset_path: str,
        direction_deg: tuple[float, float],
        signal_type: str,
        options: dict[str, Any],
        duration_s: float = 1.5,
        material: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._asset_path = str(asset_path)
        self._direction_deg = (float(direction_deg[0]), float(direction_deg[1]))
        self._signal_type = str(signal_type)
        self._options = dict(options)
        self._duration_s = float(duration_s)
        self._material = None if material is None else np.asarray(material, dtype=np.float64)

    @pyqtSlot()
    def run(self) -> None:
        try:
            audio, sample_rate = self.render()
        except Exception as error:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(error))
            return
        self.rendered.emit((audio, sample_rate))

    def render(self) -> tuple[np.ndarray, int]:
        """Render the clip. Separated so tests can call it without a thread."""

        from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
        from src.audio.sam_workbench.hrtf.subject_test import make_test_signal
        from src.audio.sam_workbench.render.hrtf import SpatialHrtfSpec, render_spatial_hrtf

        target_rate = int(self._options.get("sampleRateHz") or 0) or None
        dataset = load_sofa(
            self._asset_path,
            target_sample_rate_hz=target_rate,
            delay_policy=self._options.get("delayPolicy", "bake_delay_into_ir"),
        )
        sample_rate = int(round(dataset.sample_rate_hz))

        azimuth, elevation = np.radians(self._direction_deg[0]), np.radians(self._direction_deg[1])
        pointing = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )

        mono = make_test_signal(
            self._signal_type,
            duration_s=self._duration_s,
            sample_rate_hz=sample_rate,
            material=self._material,
        )

        if self._signal_type == "moving_broadband":
            # The one signal whose point is motion: sweep it past the chosen
            # direction so the crossfade between filters is what gets judged.
            frames = mono.size
            sweep = np.radians(
                np.linspace(self._direction_deg[0] - 60.0, self._direction_deg[0] + 60.0, frames)
            )
            path = np.stack(
                [
                    np.cos(elevation) * np.cos(sweep),
                    np.cos(elevation) * np.sin(sweep),
                    np.full(frames, np.sin(elevation)),
                ],
                axis=1,
            )
        else:
            path = pointing

        harmonic_order = self._options.get("harmonicOrder")
        spec = SpatialHrtfSpec(
            interpolation=self._options.get("interpolation", "nearest"),
            neighbor_count=int(self._options.get("neighborCount", 3)),
            harmonic_order=None if harmonic_order is None else int(harmonic_order),
            crossfade_ms=float(self._options.get("crossfadeMs", 12.0)),
            headphone=self._headphone_correction(sample_rate),
        )
        audio = render_spatial_hrtf(mono, path, dataset, sample_rate, spec=spec)
        return audio, sample_rate

    def _headphone_correction(self, sample_rate: int):
        """Build the correction the user explicitly asked for, and only that."""

        from src.audio.sam_workbench.hrtf.headphones import HeadphoneCorrection

        mode = self._options.get("headphoneCorrection", "off")
        asset = self._options.get("headphoneAsset") or ""
        if mode == "off" or not asset:
            return None
        if mode in ("sonicom_hd650", "user_impulse_response"):
            model = "Sennheiser HD650" if mode == "sonicom_hd650" else ""
            correction = HeadphoneCorrection.from_sofa(asset, sample_rate, model=model)
            # from_sofa stamps its own mode; keep the user's choice authoritative.
            return correction
        if mode == "user_equalization_curve":
            curve = np.loadtxt(asset, delimiter=",", ndmin=2)
            return HeadphoneCorrection.from_equalization_curve(
                curve[:, 0], curve[:, 1], sample_rate
            )
        return None


class SamHrtfLab(QWidget):
    """Asset selection, rendering policy, auditioning, and subject ranking."""

    paramsChanged = pyqtSignal(dict)
    #: ``(audio, sample_rate_hz)`` for the host dialog's existing audio path.
    auditionRendered = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params: dict[str, Any] = {}
        self._loading = False
        self._assets: list[Any] = []
        self._dataset_cache: Any = None
        self._audition_thread: QThread | None = None
        self._audition_worker: AuditionWorker | None = None
        self._last_audition: tuple[np.ndarray, int] | None = None
        self._session_material: np.ndarray | None = None

        self._build_ui()
        self._refresh_library()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_asset_tab(), "Asset")
        self.tabs.addTab(self._build_audition_tab(), "Audition")
        self.modify_panel = SamHrtfModifyPanel()
        self.modify_panel.paramsChanged.connect(self._on_modify_changed)
        self.modify_panel.auditionRendered.connect(self.auditionRendered)
        self.tabs.addTab(self.modify_panel, "Modify")
        self.tabs.addTab(self._build_ranking_tab(), "Ranking")
        self.tabs.addTab(self._build_tests_tab(), "Tests && import")
        layout.addWidget(self.tabs, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_asset_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        library = QGroupBox("SONICOM library")
        library_form = QFormLayout(library)
        root_row = QHBoxLayout()
        self.library_edit = QLineEdit()
        self.library_edit.setPlaceholderText("Folder containing the SONICOM SOFA files")
        self.library_edit.editingFinished.connect(self._refresh_library)
        browse_root = QPushButton("Browse…")
        browse_root.clicked.connect(self._browse_library)
        reset_root = QPushButton("App data folder")
        reset_root.setToolTip(
            "Use this machine's application-data folder. The dataset is never "
            "assumed to live inside the project or the repository."
        )
        reset_root.clicked.connect(self._use_default_library)
        root_row.addWidget(self.library_edit, 1)
        root_row.addWidget(browse_root)
        root_row.addWidget(reset_root)
        library_form.addRow("Library folder", root_row)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("SONICOM", "SONICOM")
        self.dataset_combo.setToolTip("The dataset vocabulary used to interpret file names.")
        library_form.addRow("Dataset", self.dataset_combo)

        self.source_combo = QComboBox()
        for label, value in SOURCE_CHOICES:
            self.source_combo.addItem(label, value)
        self.source_combo.setToolTip(
            "Measured lists human subjects, KEMAR the manikin variants "
            "(large and small ears), Synthetic anything modelled."
        )
        library_form.addRow("Source", self.source_combo)

        self.processing_combo = QComboBox()
        for value, label in PROCESSING_LABELS.items():
            self.processing_combo.addItem(label, value)
        # Free-field compensated is the variant most SONICOM work starts from;
        # defaulting to Windowed made a full library look empty.
        self.processing_combo.setCurrentIndex(
            max(0, self.processing_combo.findData("FreeFieldComp"))
        )
        library_form.addRow("Processing", self.processing_combo)

        self.rate_combo = QComboBox()
        for rate in (44_100, 48_000, 96_000):
            self.rate_combo.addItem(f"{rate} Hz", rate)
        library_form.addRow("Sample rate", self.rate_combo)

        self.subject_combo = QComboBox()
        self.subject_combo.setToolTip(
            "SONICOM provides pinna measurements for 93 human subjects; the "
            "KEMAR large-ear scan is anthropometry identifier zero."
        )
        library_form.addRow("Subject", self.subject_combo)

        asset_row = QHBoxLayout()
        self.asset_label = QLabel("No explicit SOFA asset selected")
        self.asset_label.setWordWrap(True)
        browse_asset = QPushButton("Select SOFA…")
        browse_asset.clicked.connect(self._browse_asset)
        asset_row.addWidget(self.asset_label, 1)
        asset_row.addWidget(browse_asset)
        library_form.addRow("HRTF asset", asset_row)
        outer.addWidget(library)

        rendering = QGroupBox("Rendering")
        rendering_form = QFormLayout(rendering)
        self.delay_combo = QComboBox()
        for label, value in DELAY_CHOICES:
            self.delay_combo.addItem(label, value)
        self.delay_combo.setToolTip(
            "Whether the direction-dependent delay is baked into the impulse "
            "response or kept in SOFA Data.Delay and applied separately."
        )
        rendering_form.addRow("Interaural delay", self.delay_combo)

        self.interpolation_combo = QComboBox()
        for label, value in INTERPOLATION_CHOICES:
            self.interpolation_combo.addItem(label, value)
        self.interpolation_combo.setToolTip(
            "Nearest never blends, so it cannot smear a transient. Every other "
            "mode aligns the responses before blending them."
        )
        self.interpolation_combo.currentIndexChanged.connect(
            self._update_interpolation_settings
        )
        rendering_form.addRow("Interpolation", self.interpolation_combo)

        # The two modes that take a setting of their own. Shown only for the
        # mode they belong to: a neighbour count next to "nearest" would invite
        # someone to set a number that changes nothing.
        self.neighbor_spin = QSpinBox()
        self.neighbor_spin.setRange(2, 16)
        self.neighbor_spin.setValue(3)
        self.neighbor_spin.setToolTip(
            "How many measurements the three-neighbour blend uses. More is "
            "smoother and blurrier; the weights taper to nothing at the edge "
            "of the set either way, so the blend stays continuous."
        )
        self.neighbor_label = QLabel("Neighbours")
        rendering_form.addRow(self.neighbor_label, self.neighbor_spin)

        self.harmonic_spin = QSpinBox()
        self.harmonic_spin.setRange(0, 40)
        self.harmonic_spin.setValue(0)
        self.harmonic_spin.setSpecialValueText("automatic")
        self.harmonic_spin.setToolTip(
            "Spherical-harmonic order. Automatic uses the highest the dataset "
            "can determine, which is what its measurement count allows: an "
            "order needing more coefficients than there are measurements is "
            "capped, and the asset panel says when that happened."
        )
        self.harmonic_label = QLabel("Harmonic order")
        rendering_form.addRow(self.harmonic_label, self.harmonic_spin)

        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(0.0, 100.0)
        self.crossfade_spin.setSingleStep(1.0)
        self.crossfade_spin.setValue(12.0)
        self.crossfade_spin.setSuffix(" ms")
        self.crossfade_spin.setToolTip(
            "Length of the fade when the selected direction changes. Clamped "
            "to 5-20 ms when rendering."
        )
        rendering_form.addRow("Filter crossfade", self.crossfade_spin)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 4096)
        self.interval_spin.setValue(128)
        self.interval_spin.setSuffix(" samples")
        rendering_form.addRow("Direction interval", self.interval_spin)

        self.headphone_combo = QComboBox()
        for label, value in HEADPHONE_CHOICES:
            self.headphone_combo.addItem(label, value)
        self.headphone_combo.setToolTip(
            "Applied after binaural rendering. The HD650 measurement is only "
            "correct for HD650s and is never selected for you."
        )
        rendering_form.addRow("Headphone correction", self.headphone_combo)

        headphone_row = QHBoxLayout()
        self.headphone_label = QLabel("No correction file selected")
        self.headphone_label.setWordWrap(True)
        browse_headphone = QPushButton("Select…")
        browse_headphone.clicked.connect(self._browse_headphone)
        headphone_row.addWidget(self.headphone_label, 1)
        headphone_row.addWidget(browse_headphone)
        rendering_form.addRow("Correction file", headphone_row)
        outer.addWidget(rendering)

        status = QGroupBox("Asset status")
        status_layout = QVBoxLayout(status)
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality status:"))
        self.quality_label = QLabel("missing resources")
        self.quality_label.setToolTip(
            "Normal - no issues found. Suspected outlier - the file loads but "
            "something looks unusual. Missing resources - it could not be read."
        )
        quality_row.addWidget(self.quality_label)
        quality_row.addStretch(1)
        status_layout.addLayout(quality_row)
        self.coverage_plot = CoveragePlotWidget()
        status_layout.addWidget(self.coverage_plot)
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setPlaceholderText(
            "Select an asset to see its metadata, delay status, and validation."
        )
        status_layout.addWidget(self.report_view, 1)
        outer.addWidget(status, 1)

        for widget in (
            self.dataset_combo,
            self.source_combo,
            self.processing_combo,
            self.rate_combo,
            self.delay_combo,
            self.interpolation_combo,
            self.headphone_combo,
        ):
            widget.currentIndexChanged.connect(self._on_control_changed)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        for spin in (
            self.crossfade_spin,
            self.interval_spin,
            self.neighbor_spin,
            self.harmonic_spin,
        ):
            spin.valueChanged.connect(self._on_control_changed)
        for widget in (self.source_combo, self.processing_combo, self.rate_combo):
            widget.currentIndexChanged.connect(self._populate_subjects)
        self._update_interpolation_settings()
        return page

    def _build_audition_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        controls = QGroupBox("Test signal and direction")
        form = QFormLayout(controls)
        self.azimuth_combo = QComboBox()
        self.elevation_combo = QComboBox()
        self.signal_combo = QComboBox()
        form.addRow("Azimuth", self.azimuth_combo)
        form.addRow("Elevation", self.elevation_combo)
        form.addRow("Signal", self.signal_combo)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.25, 10.0)
        self.duration_spin.setValue(1.5)
        self.duration_spin.setSuffix(" s")
        form.addRow("Clip length", self.duration_spin)

        button_row = QHBoxLayout()
        self.audition_button = QPushButton("Render audition")
        self.audition_button.clicked.connect(self.start_audition)
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._emit_audition)
        button_row.addWidget(self.audition_button)
        button_row.addWidget(self.play_button)
        button_row.addStretch(1)
        form.addRow(button_row)
        outer.addWidget(controls)

        rating = QGroupBox("Rate this subject")
        rating_layout = QVBoxLayout(rating)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_edit = QLineEdit(DEFAULT_PROFILE)
        self.profile_edit.setToolTip(
            "Scores are kept per profile. A subject that externalises "
            "broadband noise well may not suit a low-frequency carrier, so "
            "rate them separately and keep both."
        )
        profile_row.addWidget(self.profile_edit, 1)
        rating_layout.addLayout(profile_row)

        grid = QGridLayout()
        self._criteria_sliders: dict[str, QSlider] = {}
        from src.audio.sam_workbench.hrtf.subject_test import RATING_CRITERIA

        for row, criterion in enumerate(RATING_CRITERIA):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, 5)
            slider.setValue(3)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            value_label = QLabel("3")
            slider.valueChanged.connect(
                lambda value, label=value_label: label.setText(str(value))
            )
            grid.addWidget(QLabel(criterion.replace("_", " ").capitalize()), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
            self._criteria_sliders[criterion] = slider
        rating_layout.addLayout(grid)

        save_row = QHBoxLayout()
        self.save_rating_button = QPushButton("Save rating")
        self.save_rating_button.clicked.connect(self.save_rating)
        save_row.addWidget(self.save_rating_button)
        save_row.addStretch(1)
        rating_layout.addLayout(save_row)
        # Absorb the leftover height here rather than letting it open gaps
        # between the profile field, the sliders, and the save button.
        rating_layout.addStretch(1)
        outer.addWidget(rating, 1)

        self._populate_audition_choices()
        for widget in (self.azimuth_combo, self.elevation_combo):
            widget.currentIndexChanged.connect(self._update_highlight)
        return page

    def _build_ranking_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        row = QHBoxLayout()
        row.addWidget(QLabel("Profile:"))
        self.ranking_profile_combo = QComboBox()
        self.ranking_profile_combo.setEditable(False)
        self.ranking_profile_combo.currentIndexChanged.connect(self.refresh_ranking)
        row.addWidget(self.ranking_profile_combo, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_ranking)
        row.addWidget(refresh)
        outer.addLayout(row)

        self.ranking_table = QTableWidget(0, 4)
        self.ranking_table.setHorizontalHeaderLabels(["Subject", "Score", "Ratings", "Description"])
        self.ranking_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ranking_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.ranking_table.horizontalHeader()
        # Subject identifiers are the point of the table; let them show in full
        # and give the leftover width to the description.
        for column in (0, 1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        outer.addWidget(self.ranking_table, 1)

        self.ranking_hint = QLabel(
            "No ratings yet for this profile. Score a subject on the Audition "
            "tab and it will appear here."
        )
        self.ranking_hint.setWordWrap(True)
        outer.addWidget(self.ranking_hint)

        self.ratings_location_label = QLabel("")
        self.ratings_location_label.setWordWrap(True)
        outer.addWidget(self.ratings_location_label)
        return page

    def _build_tests_tab(self) -> QWidget:
        """Listening tests, and the two advanced acquisition routes.

        Ordered the way the specification orders them: the localization test
        first, because it costs nothing and settles most of the question, and
        the measurement and simulation routes last, behind their own flag.
        """

        page = QWidget()
        outer = QVBoxLayout(page)

        intro = QLabel(
            "Which HRTF works for you is a listening question, not a specification "
            "one. Start with the localization test over the assets in your "
            "library; measuring or simulating a head is a much larger undertaking "
            "for a smaller remaining gain."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        localization = QPushButton("Run a blinded localization test…")
        localization.setToolTip(
            "Present directions through each candidate and point at where you "
            "heard them. Scores the candidates on angular error and front/back "
            "reversals."
        )
        localization.clicked.connect(self.open_localization_test)
        outer.addWidget(localization)

        experiment = QPushButton("Run a listening experiment…")
        experiment.setToolTip(
            "Blinded subjective comparison across conditions, with an export "
            "that records enough to reproduce what was compared."
        )
        experiment.clicked.connect(self.open_experiment)
        outer.addWidget(experiment)

        acquisition = QPushButton("Measure or import an HRTF…")
        acquisition.setToolTip(
            "Swept-sine measurement (advanced, off by default) and Mesh2HRTF "
            "import validation."
        )
        acquisition.clicked.connect(self.open_measurement)
        outer.addWidget(acquisition)

        self.tests_hint = QLabel("")
        self.tests_hint.setWordWrap(True)
        outer.addWidget(self.tests_hint)
        outer.addStretch(1)
        return page

    # --- test and import dialogs -------------------------------------------

    def candidate_subjects(self) -> list[str]:
        """The subjects the current filters leave available, for a comparison."""

        subjects = []
        for entry in self._matching_assets():
            name = entry.subject or Path(str(entry.path)).stem
            if name and name not in subjects:
                subjects.append(name)
        return subjects

    def candidate_assets(self) -> list[dict[str, Any]]:
        """The matching assets as condition material, with provenance kept."""

        assets = []
        for entry in self._matching_assets():
            path = str(getattr(entry, "path", ""))
            assets.append(
                {
                    "id": entry.subject or Path(path).stem,
                    "subject": entry.subject or "",
                    "path": path,
                    "sha256": str(getattr(entry, "sha256", "") or ""),
                    "processing": str(getattr(entry, "processing", "") or ""),
                    "sampleRateHz": float(getattr(entry, "sample_rate_hz", 0) or 48_000.0),
                }
            )
        return assets

    def open_localization_test(self) -> Any:
        from .sam_localization_dialog import SamLocalizationDialog

        candidates = self.candidate_subjects()
        if len(candidates) < 2:
            self.tests_hint.setText(
                "A comparison needs at least two candidates. Widen the filters on "
                "the Asset tab, or point the library at a folder with more SOFA files."
            )
            return None
        dialog = SamLocalizationDialog(candidates, parent=self)
        dialog.exec_()
        return dialog

    def open_experiment(self) -> Any:
        from .sam_experiment_dialog import SamExperimentDialog

        dialog = SamExperimentDialog(self.candidate_assets(), parent=self)
        dialog.exec_()
        return dialog

    def open_measurement(self) -> Any:
        from .sam_measurement_dialog import SamMeasurementDialog

        dialog = SamMeasurementDialog(parent=self)
        dialog.exec_()
        return dialog

    def _update_interpolation_settings(self) -> None:
        """Show each mode's own setting, and only when it applies."""

        mode = self.interpolation_combo.currentData()
        uses_neighbours = mode == "three_neighbor"
        uses_harmonics = mode == "spherical_harmonic"
        for widget, applies in (
            (self.neighbor_label, uses_neighbours),
            (self.neighbor_spin, uses_neighbours),
            (self.harmonic_label, uses_harmonics),
            (self.harmonic_spin, uses_harmonics),
        ):
            widget.setVisible(applies)

    def _populate_audition_choices(self) -> None:
        from src.audio.sam_workbench.hrtf.subject_test import (
            AUDITION_AZIMUTHS_DEG,
            AUDITION_ELEVATIONS_DEG,
            SIGNAL_TYPES,
        )

        for azimuth in AUDITION_AZIMUTHS_DEG:
            self.azimuth_combo.addItem(f"{azimuth:+.0f}°", float(azimuth))
        for elevation in AUDITION_ELEVATIONS_DEG:
            self.elevation_combo.addItem(f"{elevation:+.0f}°", float(elevation))
        self.elevation_combo.setCurrentIndex(
            max(0, list(AUDITION_ELEVATIONS_DEG).index(0.0)) if 0.0 in AUDITION_ELEVATIONS_DEG else 0
        )
        for signal_type in SIGNAL_TYPES:
            self.signal_combo.addItem(signal_type.replace("_", " ").capitalize(), signal_type)

    # --- library and subjects ----------------------------------------------

    def _browse_library(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select the SONICOM library folder")
        if folder:
            self.library_edit.setText(folder)
            self._refresh_library()

    def _use_default_library(self) -> None:
        from src.audio.sam_workbench.hrtf.storage import datasets_root

        try:
            self.library_edit.setText(str(datasets_root()))
        except Exception:
            self.library_edit.setText(str(application_data_root()))
        self._refresh_library()

    def library_root(self) -> Path | None:
        text = self.library_edit.text().strip()
        return Path(text) if text else None

    def _refresh_library(self) -> None:
        """Discover what is actually on disk, tolerating an absent dataset."""

        root = self.library_root()
        self._assets = []
        if root is None:
            self.status_label.setText(
                "No library folder set. Choose one, or select a SOFA file directly."
            )
            self._populate_subjects()
            return
        if not root.is_dir():
            # Saying "not set" while the field shows a path sends the user
            # looking for the wrong problem.
            self.status_label.setText(
                f"{root} does not exist yet. Install the dataset there, choose "
                "another folder, or select a SOFA file directly."
            )
            self._populate_subjects()
            return
        try:
            from src.audio.sam_workbench.hrtf.datasets import discover_assets

            self._assets = discover_assets(root)
        except Exception as error:
            self.status_label.setText(f"Could not scan the library: {error}")
        else:
            self.status_label.setText(
                f"Found {len(self._assets)} SOFA file(s) under {root}."
                if self._assets
                else f"No SOFA files found under {root}."
            )
        self._populate_subjects()

    def _matching_assets(self) -> list[Any]:
        from src.audio.sam_workbench.hrtf.datasets import is_kemar_subject

        source = self.source_combo.currentData()
        processing = self.processing_combo.currentData()
        rate = self.rate_combo.currentData()

        matches = []
        for entry in self._assets:
            subject = entry.subject or ""
            kemar = is_kemar_subject(subject) if subject else False
            if source == "kemar" and not kemar:
                continue
            if source == "measured" and (kemar or not subject.upper().startswith("P")):
                continue
            if source == "synthetic" and (kemar or subject.upper().startswith("P")):
                continue
            if entry.processing and processing and entry.processing != processing:
                continue
            if entry.sample_rate_hz and rate and int(entry.sample_rate_hz) != int(rate):
                continue
            matches.append(entry)
        return matches

    def _populate_subjects(self) -> None:
        previous = self.subject_combo.currentData()
        blocked = self.subject_combo.blockSignals(True)
        self.subject_combo.clear()

        from src.audio.sam_workbench.hrtf.datasets import describe_subject

        seen: dict[str, Any] = {}
        for entry in self._matching_assets():
            if entry.subject and entry.subject not in seen:
                seen[entry.subject] = entry
        for subject, entry in sorted(seen.items()):
            self.subject_combo.addItem(f"{subject} — {describe_subject(subject)}", str(entry.path))
        if not seen:
            self.subject_combo.addItem("(no matching subjects)", "")
            if self._assets:
                # Files are present but the filters excluded all of them. Saying
                # which filters is the difference between "wrong folder" and
                # "wrong processing variant".
                self.status_label.setText(
                    f"{len(self._assets)} SOFA file(s) found, but none match "
                    f"{self.source_combo.currentText()} / "
                    f"{self.processing_combo.currentText()} / "
                    f"{self.rate_combo.currentText()}. Adjust the filters above."
                )

        if previous:
            index = self.subject_combo.findData(previous)
            if index >= 0:
                self.subject_combo.setCurrentIndex(index)
        self.subject_combo.blockSignals(blocked)

    def _on_subject_changed(self, _index: int) -> None:
        path = self.subject_combo.currentData()
        if path:
            self._params["hrtfAsset"] = str(path)
            self.asset_label.setText(str(path))
            self.inspect_asset()
        self._on_control_changed()

    def _browse_asset(self) -> None:
        start = str(self.library_root() or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select explicit SOFA HRTF", start, "SOFA files (*.sofa);;All files (*)"
        )
        if path:
            self._params["hrtfAsset"] = path
            self.asset_label.setText(path)
            self.inspect_asset()
            self._on_control_changed()

    def _browse_headphone(self) -> None:
        mode = self.headphone_combo.currentData()
        if mode == "user_equalization_curve":
            caption, filters = "Select an equalisation curve", "CSV curves (*.csv *.txt);;All files (*)"
        else:
            caption, filters = "Select a headphone measurement", "SOFA files (*.sofa);;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, caption, str(self.library_root() or ""), filters)
        if path:
            self._params.setdefault("hrtfOptions", {})["headphoneAsset"] = path
            self.headphone_label.setText(path)
            self._on_control_changed()

    # --- inspection ---------------------------------------------------------

    def _set_quality(self, quality: str) -> None:
        self.quality_label.setText(quality)
        colour = _QUALITY_COLOURS.get(quality, "")
        self.quality_label.setStyleSheet(f"color: {colour}; font-weight: bold;" if colour else "")

    def inspect_asset(self) -> None:
        """Load the selected asset and report what it contains."""

        path = self._params.get("hrtfAsset")
        if not path:
            self._set_quality("missing resources")
            self.coverage_plot.clear()
            self.report_view.setPlainText("")
            return

        from src.audio.sam_workbench.hrtf.validation import quality_from_issues, validate_dataset

        try:
            from src.audio.sam_workbench.hrtf.coordinates import cartesian_to_sofa_spherical
            from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

            dataset = load_sofa(path, delay_policy=self.delay_combo.currentData())
        except Exception as error:
            self._dataset_cache = None
            self.modify_panel.set_dataset(None)
            self._set_quality("missing resources")
            self.coverage_plot.clear("This asset could not be read.")
            self.report_view.setPlainText(
                f"Asset unavailable or invalid:\n{error}\n\n"
                "Other renderers remain available; only HRTF rendering needs this file."
            )
            return

        self._dataset_cache = dataset
        self.modify_panel.set_dataset(dataset)
        self._params["hrtfAssetHash"] = dataset.content_hash
        issues = validate_dataset(dataset)
        self._set_quality(quality_from_issues(issues))

        spherical = cartesian_to_sofa_spherical(dataset.positions_m)
        self.coverage_plot.set_directions(spherical[:, 0], spherical[:, 1])
        self._update_highlight()

        report = dataset.metadata_report()
        lines = [f"{key}: {value}" for key, value in report.items()]
        lines += [
            "",
            "Spherical coverage:",
            f"  azimuth   {spherical[:, 0].min():.1f}° … {spherical[:, 0].max():.1f}°",
            f"  elevation {spherical[:, 1].min():.1f}° … {spherical[:, 1].max():.1f}°",
            f"  distance  {spherical[:, 2].min():.3f} … {spherical[:, 2].max():.3f} m",
        ]
        if issues:
            lines += ["", "Validation:"]
            lines += [f"  [{issue.severity}] {issue.path}: {issue.message}" for issue in issues]
        self.report_view.setPlainText("\n".join(lines))

    def _update_highlight(self) -> None:
        azimuth = self.azimuth_combo.currentData()
        elevation = self.elevation_combo.currentData()
        if azimuth is None or elevation is None:
            return
        self.coverage_plot.set_highlight(float(azimuth), float(elevation))

    # --- audition -----------------------------------------------------------

    def audition_options(self) -> dict[str, Any]:
        return dict(self.params().get("hrtfOptions", {}))

    def set_session_material(self, audio, sample_rate_hz: int | None = None) -> None:
        """Supply the session's own carrier or noise for auditioning.

        A subject that suits clicks and pink noise may still be wrong for the
        material a listener will actually spend an hour with, so the audition
        list offers that material directly. The host dialog feeds this from its
        rendered preview.
        """

        block = np.asarray(audio, dtype=np.float64)
        if block.size == 0:
            self._session_material = None
            return
        # Channel-major stereo comes back from the renderer; audition material
        # is mono, and downmixing keeps both ears' content rather than picking
        # one and silently discarding the other.
        self._session_material = block.mean(axis=0) if block.ndim == 2 else block.reshape(-1)

    @property
    def has_session_material(self) -> bool:
        return self._session_material is not None

    def start_audition(self) -> None:
        """Render the current direction and signal in a worker thread."""

        path = self._params.get("hrtfAsset")
        if not path:
            self.status_label.setText("Select a SOFA asset before auditioning.")
            return
        if self.signal_combo.currentData() == "session_material" and not self.has_session_material:
            self.status_label.setText(
                "Render a preview first: session material auditions this voice's "
                "own audio, so there has to be some."
            )
            return
        if self._audition_thread is not None:
            self.status_label.setText("An audition render is already running.")
            return

        self.audition_button.setEnabled(False)
        self.status_label.setText("Rendering audition…")
        worker = AuditionWorker(
            path,
            (float(self.azimuth_combo.currentData()), float(self.elevation_combo.currentData())),
            str(self.signal_combo.currentData()),
            self.audition_options(),
            duration_s=self.duration_spin.value(),
            material=self._session_material,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.rendered.connect(self._on_audition_rendered)
        worker.failed.connect(self._on_audition_failed)
        worker.rendered.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_audition_finished)
        self._audition_worker = worker
        self._audition_thread = thread
        thread.start()

    @pyqtSlot(object)
    def _on_audition_rendered(self, payload) -> None:
        audio, sample_rate = payload
        self._last_audition = (audio, int(sample_rate))
        self.play_button.setEnabled(True)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        self.status_label.setText(
            f"Audition ready: {audio.shape[1] / max(1, sample_rate):.2f} s at "
            f"{sample_rate} Hz, peak {peak:.3f}."
        )
        self._emit_audition()

    @pyqtSlot(str)
    def _on_audition_failed(self, message: str) -> None:
        self.status_label.setText(f"Audition failed: {message}")

    @pyqtSlot()
    def _on_audition_finished(self) -> None:
        if self._audition_thread is not None:
            self._audition_thread.deleteLater()
        self._audition_thread = None
        self._audition_worker = None
        self.audition_button.setEnabled(True)

    def _emit_audition(self) -> None:
        if self._last_audition is not None:
            self.auditionRendered.emit(self._last_audition)

    @property
    def last_audition(self) -> tuple[np.ndarray, int] | None:
        return self._last_audition

    # --- ratings ------------------------------------------------------------

    def _rating_store(self):
        from src.audio.sam_workbench.hrtf.subject_test import RatingStore

        store = RatingStore()
        try:
            store.load()
        except Exception:  # pragma: no cover - unreadable or absent file
            pass
        return store

    def current_scores(self) -> dict[str, float]:
        return {name: float(slider.value()) for name, slider in self._criteria_sliders.items()}

    def current_subject(self) -> str:
        """The subject identifier for whatever asset is selected."""

        explicit = self._params.get("hrtfSubject")
        if explicit:
            return str(explicit)
        path = self._params.get("hrtfAsset")
        if not path:
            return ""
        from src.audio.sam_workbench.hrtf.datasets import parse_asset_name

        return parse_asset_name(path).subject or Path(path).stem

    def save_rating(self) -> None:
        """Store this subject's scores locally, under the chosen profile."""

        subject = self.current_subject()
        if not subject:
            self.status_label.setText("Select a SOFA asset before saving a rating.")
            return

        from src.audio.sam_workbench.hrtf.subject_test import SubjectRating

        store = self._rating_store()
        store.add(
            SubjectRating(
                subject=subject,
                scores=self.current_scores(),
                profile=self.profile_edit.text().strip() or DEFAULT_PROFILE,
                signal_type=str(self.signal_combo.currentData()),
                asset_path=str(self._params.get("hrtfAsset") or ""),
            )
        )
        try:
            location = store.save()
        except Exception as error:
            self.status_label.setText(f"Could not save the rating: {error}")
            return
        self.status_label.setText(f"Saved a rating for {subject} to {location}.")
        self.refresh_profiles()

    def refresh_profiles(self) -> None:
        store = self._rating_store()
        profiles = list(store.profiles()) or [DEFAULT_PROFILE]
        previous = self.ranking_profile_combo.currentText()
        blocked = self.ranking_profile_combo.blockSignals(True)
        self.ranking_profile_combo.clear()
        self.ranking_profile_combo.addItems(profiles)
        if previous in profiles:
            self.ranking_profile_combo.setCurrentText(previous)
        self.ranking_profile_combo.blockSignals(blocked)
        self.refresh_ranking()

    def refresh_ranking(self) -> None:
        """Rank the subjects this user has scored, for the selected profile."""

        from src.audio.sam_workbench.hrtf.datasets import describe_subject
        from src.audio.sam_workbench.hrtf.storage import ratings_path
        from src.audio.sam_workbench.hrtf.subject_test import rank_subjects

        store = self._rating_store()
        profile = self.ranking_profile_combo.currentText() or DEFAULT_PROFILE
        ranked = rank_subjects(store.for_profile(profile))

        self.ranking_table.setRowCount(len(ranked))
        for row, entry in enumerate(ranked):
            self.ranking_table.setItem(row, 0, QTableWidgetItem(entry.subject))
            self.ranking_table.setItem(row, 1, QTableWidgetItem(f"{entry.score:.2f}"))
            self.ranking_table.setItem(row, 2, QTableWidgetItem(str(entry.ratings)))
            self.ranking_table.setItem(row, 3, QTableWidgetItem(describe_subject(entry.subject)))
        self.ranking_hint.setVisible(not ranked)
        try:
            self.ratings_location_label.setText(f"Ratings are stored at {ratings_path()}.")
        except Exception:  # pragma: no cover - unwritable location
            self.ratings_location_label.setText("")

    # --- parameter contract -------------------------------------------------

    def set_params(self, params) -> None:
        self._loading = True
        try:
            self._params = copy.deepcopy(dict(params))
            options = dict(self._params.get("hrtfOptions") or {})

            path = self._params.get("hrtfAsset")
            self.asset_label.setText(str(path) if path else "No explicit SOFA asset selected")

            if options.get("libraryRoot"):
                self.library_edit.setText(str(options["libraryRoot"]))
            elif not self.library_edit.text().strip():
                self._use_default_library()

            self._select_data(self.dataset_combo, options.get("dataset", "SONICOM"))
            self._select_data(self.source_combo, options.get("source", "measured"))
            self._select_data(self.processing_combo, options.get("processing", "FreeFieldComp"))
            self._select_data(self.rate_combo, int(options.get("sampleRateHz", 48_000) or 48_000))
            self._select_data(self.delay_combo, options.get("delayPolicy", "bake_delay_into_ir"))
            self._select_data(self.interpolation_combo, options.get("interpolation", "nearest"))
            self.neighbor_spin.setValue(int(options.get("neighborCount", 3)))
            harmonic_order = options.get("harmonicOrder")
            self.harmonic_spin.setValue(0 if harmonic_order is None else int(harmonic_order))
            self._select_data(self.headphone_combo, options.get("headphoneCorrection", "off"))
            self.crossfade_spin.setValue(float(options.get("crossfadeMs", 12.0)))
            self.interval_spin.setValue(int(options.get("controlIntervalSamples", 128)))

            headphone_asset = options.get("headphoneAsset")
            self.headphone_label.setText(
                str(headphone_asset) if headphone_asset else "No correction file selected"
            )

            self.modify_panel.set_params(self._params)

            subject = self._params.get("hrtfSubject")
            if subject:
                index = self.subject_combo.findText(str(subject), Qt.MatchStartsWith)
                if index >= 0:
                    self.subject_combo.setCurrentIndex(index)
        finally:
            self._loading = False

        if path:
            self.inspect_asset()
        self.refresh_profiles()

    @staticmethod
    def _select_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def params(self) -> dict[str, Any]:
        """The HRTF selection and policies, for the project dictionary.

        Only the asset path and the subject identifier identify the dataset;
        the measurements themselves stay where the user installed them.
        """

        options = dict(self._params.get("hrtfOptions") or {})
        options.update(
            {
                "schemaVersion": HRTF_OPTIONS_SCHEMA_VERSION,
                "dataset": self.dataset_combo.currentData(),
                "source": self.source_combo.currentData(),
                "processing": self.processing_combo.currentData(),
                "sampleRateHz": int(self.rate_combo.currentData()),
                "delayPolicy": self.delay_combo.currentData(),
                "interpolation": self.interpolation_combo.currentData(),
                "neighborCount": int(self.neighbor_spin.value()),
                # Zero is the spin box's "automatic", not an order of zero.
                "harmonicOrder": (
                    None if self.harmonic_spin.value() == 0 else int(self.harmonic_spin.value())
                ),
                "crossfadeMs": float(self.crossfade_spin.value()),
                "controlIntervalSamples": int(self.interval_spin.value()),
                "headphoneCorrection": self.headphone_combo.currentData(),
            }
        )
        root = self.library_edit.text().strip()
        if root:
            options["libraryRoot"] = root

        # The modify panel owns the cue and anchor sub-objects.
        options.update(self.modify_panel.params().get("hrtfOptions", {}))

        result: dict[str, Any] = {"hrtfOptions": options}
        if self._params.get("hrtfAsset"):
            result["hrtfAsset"] = self._params["hrtfAsset"]
        if self._params.get("hrtfAssetHash"):
            result["hrtfAssetHash"] = self._params["hrtfAssetHash"]
        subject = self.current_subject()
        if subject:
            result["hrtfSubject"] = subject
        return result

    def _on_modify_changed(self, _params) -> None:
        if not self._loading:
            self.paramsChanged.emit(self.params())

    def _on_control_changed(self, *_args) -> None:
        if self._loading:
            return
        mode = self.headphone_combo.currentData()
        self.headphone_label.setEnabled(mode != "off")
        self.paramsChanged.emit(self.params())

    # --- teardown -----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._audition_thread is not None:
            self._audition_thread.quit()
            self._audition_thread.wait(2000)
        super().closeEvent(event)

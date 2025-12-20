"""Session builder main window with streaming and export controls."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, MutableSequence, Optional

from PyQt5.QtCore import QEvent, QModelIndex, QObject, Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.session_engine import SessionAssembler
from src.audio.session_model import Session, SessionPresetChoice, SessionStep, MAX_INDIVIDUAL_GAIN
from PyQt5.QtWidgets import QApplication

from src.audio.session_stream import SessionStreamPlayer
from src.audio.rust_stream_player import (
    HybridStreamPlayer,
    RustStreamPlayer,
    is_rust_backend_available,
)
from src.models.models import StepModel
from . import themes
from .defaults_dialog import DefaultsDialog, load_defaults


MAX_NORMALIZATION_UI = 0.95


class ComboBoxClickFilter(QObject):
    """Event filter that makes clicking anywhere on a combo box open its dropdown."""

    def __init__(self, combo: QComboBox) -> None:
        super().__init__(combo)
        self._combo = combo

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress:
            self._combo.showPopup()
            return True
        return super().eventFilter(obj, event)


class SessionStepModel(StepModel):
    """Table model wrapper bridging :class:`SessionStep` objects to the view."""

    headers = ["Duration (s)", "Binaural Preset", "Noise Preset", "Description"]

    def __init__(self, steps: MutableSequence[SessionStep], preset_lookup: Mapping[str, SessionPresetChoice], noise_lookup: Mapping[str, SessionPresetChoice]):
        super().__init__(list(steps))
        self._preset_lookup = preset_lookup
        self._noise_lookup = noise_lookup

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        if index.row() >= len(self.steps):
            return None
        step: SessionStep = self.steps[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return f"{step.duration:.2f}"
            if index.column() == 1:
                if not step.binaural_preset_id:
                    return "None"
                preset = self._preset_lookup.get(step.binaural_preset_id)
                return preset.label if preset else step.binaural_preset_id
            if index.column() == 2:
                if not step.noise_preset_id:
                    return "None"
                preset = self._noise_lookup.get(step.noise_preset_id)
                return preset.label if preset else step.noise_preset_id
            if index.column() == 3:
                return step.description
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole):  # type: ignore[override]
        if not index.isValid() or role != Qt.EditRole:
            return False
        if index.row() >= len(self.steps):
            return False
        step: SessionStep = self.steps[index.row()]
        if index.column() == 3:
            step.description = str(value)
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index: QModelIndex):  # type: ignore[override]
        base_flags = super().flags(index)
        if index.column() == 3:
            base_flags |= Qt.ItemIsEditable
        return base_flags

    def refresh(self, steps: Optional[MutableSequence[SessionStep]] = None):  # type: ignore[override]
        if steps is not None:
            self.steps = list(steps)
        super().refresh(self.steps)


def _session_to_dict(session: Session) -> dict:
    data = asdict(session)
    return data


def _session_from_dict(data: Mapping[str, object]) -> Session:
    kwargs = dict(data)
    raw_steps = list(kwargs.pop("steps", []))
    steps = []
    for raw in raw_steps:
        if isinstance(raw, Mapping):
            step_kwargs = dict(raw)
        else:
            continue
        steps.append(SessionStep(**step_kwargs))
    kwargs["steps"] = steps
    return Session(**kwargs)


class SessionBuilderWindow(QMainWindow):
    """Main window that allows building, previewing, and exporting sessions."""

    def __init__(
        self,
        session: Optional[Session] = None,
        *,
        binaural_catalog: Optional[Mapping[str, SessionPresetChoice]] = None,
        noise_catalog: Optional[Mapping[str, SessionPresetChoice]] = None,
        stream_player_factory: Optional[Callable[[dict], SessionStreamPlayer]] = None,
        assembler_factory: Optional[
            Callable[[Session, Mapping[str, SessionPresetChoice], Mapping[str, SessionPresetChoice]], SessionAssembler]
        ] = None,
        theme_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Session Builder")
        self.resize(1000, 700)

        self._session = session or Session()
        self._binaural_catalog = dict(binaural_catalog or {})
        self._noise_catalog = dict(noise_catalog or {})

        # Load defaults
        self._defaults = load_defaults()
        is_new_session = session is None
        if is_new_session:
            norm_default = float(self._defaults.get("normalization_level", 0.95))
            self._session.normalization_level = max(0.0, min(norm_default, MAX_NORMALIZATION_UI))
            self._session.crossfade_duration = float(self._defaults.get("crossfade_duration", 10.0))

        self._assembler_factory = assembler_factory or (lambda s, b, n, **opts: SessionAssembler(s, b, n, **opts))
        # Use HybridStreamPlayer which prefers Rust backend when available
        self._stream_player_factory = stream_player_factory or (
            lambda track_data: HybridStreamPlayer.create(track_data, self, prefer_rust=True)
        )
        self._stream_player: Optional[SessionStreamPlayer] = None
        self._using_rust_backend = is_rust_backend_available()
        self._current_assembler: Optional[SessionAssembler] = None
        self._is_paused = False  # Track pause state for reliable play/pause toggling

        # Coalesce rapid UI changes before pushing live updates to the backend
        self._stream_update_timer = QTimer(self)
        self._stream_update_timer.setInterval(200)
        self._stream_update_timer.setSingleShot(True)
        self._stream_update_timer.timeout.connect(self._push_stream_update)

        self._init_actions()
        self._init_menu()
        self._init_ui()
        self._load_session(self._session)

        app = QApplication.instance()
        if app is not None and theme_name:
            themes.apply_theme(app, theme_name)

    # ------------------------------------------------------------------
    # UI creation helpers
    # ------------------------------------------------------------------
    def _init_actions(self) -> None:
        self.save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save Session", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._save_session)
        
        self.load_action = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Load Session", self)
        self.load_action.setShortcut("Ctrl+O")
        self.load_action.triggered.connect(self._load_session_from_file)
        
        self.addAction(self.save_action)
        self.addAction(self.load_action)

    def _init_menu(self) -> None:
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.load_action)
        
        file_menu.addSeparator()
        defaults_action = QAction("Set Defaults...", self)
        defaults_action.triggered.connect(self._open_defaults_dialog)
        file_menu.addAction(defaults_action)
        
        file_menu.addSeparator()
        
        # Themes Submenu
        themes_menu = file_menu.addMenu("Themes")
        theme_group = QActionGroup(self)
        
        # Get available themes from themes module
        available_themes = sorted(themes.THEMES.keys())
        
        for theme_name in available_themes:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.setData(theme_name)
            action.triggered.connect(self._change_theme)
            themes_menu.addAction(action)
            theme_group.addAction(action)
            
            # Check if this is the current theme (approximation)
            # In a real app we might track current theme name
            if theme_name == "Modern Dark": 
                action.setChecked(True)

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        # --- Top Panel: Control & Playback (Card Style) ---
        control_panel = QFrame()
        control_panel.setObjectName("control_panel")
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(30)

        # Section 1: Session Settings
        settings_layout = QGridLayout()
        settings_layout.setHorizontalSpacing(15)
        settings_layout.setVerticalSpacing(10)
        
        settings_header = QLabel("Session Settings")
        settings_header.setObjectName("panel_header")
        settings_layout.addWidget(settings_header, 0, 0, 1, 3)

        self.crossfade_slider = QSlider(Qt.Horizontal)
        self.crossfade_slider.setRange(0, 300)
        self.crossfade_slider.setToolTip("Global crossfade duration applied between steps (seconds).")
        
        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setDecimals(2)
        self.crossfade_spin.setRange(0.0, 30.0)
        self.crossfade_spin.setSuffix(" s")
        self.crossfade_spin.setSingleStep(0.1)
        self.crossfade_spin.setToolTip("Precise crossfade duration in seconds.")
        
        self.crossfade_curve_combo = QComboBox()
        self.crossfade_curve_combo.addItems(["linear", "equal_power"])
        self.crossfade_curve_combo.setToolTip("Choose crossfade curve applied between steps.")

        self.normalization_slider = QSlider(Qt.Horizontal)
        self.normalization_slider.setRange(0, int(MAX_NORMALIZATION_UI * 100))
        self.normalization_slider.setToolTip(
            "Target normalization ceiling for rendered audio (0.00 – 0.95)."
        )
        
        self.normalization_label = QLabel("0.00")
        self.normalization_label.setToolTip("Current normalization ceiling applied during rendering.")

        settings_layout.addWidget(QLabel("Crossfade:"), 1, 0)
        settings_layout.addWidget(self.crossfade_slider, 1, 1)
        settings_layout.addWidget(self.crossfade_spin, 1, 2)
        
        settings_layout.addWidget(QLabel("Curve:"), 2, 0)
        settings_layout.addWidget(self.crossfade_curve_combo, 2, 1, 1, 2)
        
        settings_layout.addWidget(QLabel("Normalize:"), 3, 0)
        settings_layout.addWidget(self.normalization_slider, 3, 1)
        settings_layout.addWidget(self.normalization_label, 3, 2)
        
        control_layout.addLayout(settings_layout, 1) # Stretch factor 1

        # Vertical Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # Section 2: Playback & Export
        playback_layout = QVBoxLayout()
        playback_layout.setSpacing(10)

        playback_header = QLabel("Playback & Export")
        playback_header.setObjectName("panel_header")
        playback_layout.addWidget(playback_header)

        session_buttons = QHBoxLayout()
        session_buttons.setSpacing(8)
        self.save_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save Session")
        self.save_btn.setToolTip("Save the current session steps and settings to disk.")
        self.save_btn.setProperty("class", "secondary")
        self.load_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Load Session")
        self.load_btn.setToolTip("Load session steps and settings from disk.")
        session_buttons.addWidget(self.save_btn)
        session_buttons.addWidget(self.load_btn)
        playback_layout.addLayout(session_buttons)

        self.export_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Export Session")
        self.export_btn.setToolTip("Render the session to an audio file.")
        self.export_btn.setProperty("class", "primary") # Apply primary style

        playback_layout.addWidget(self.export_btn)
        
        control_layout.addLayout(playback_layout, 0) # No stretch, fixed width

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2) # Make handle visible but thin

        # Left: Steps List (Directly in splitter, styled by theme)
        step_container = QWidget()
        step_layout = QVBoxLayout(step_container)
        step_layout.setContentsMargins(0, 0, 10, 0) # Right margin for spacing
        step_layout.setSpacing(10)

        step_header = QLabel("Session Steps")
        step_header.setObjectName("panel_header")
        step_layout.addWidget(step_header)

        self.step_model = SessionStepModel(self._session.steps, self._binaural_catalog, self._noise_catalog)
        self.step_table = QTableView()
        self.step_table.setModel(self.step_model)
        self.step_table.setSelectionBehavior(QTableView.SelectRows)
        self.step_table.setSelectionMode(QTableView.SingleSelection)
        self.step_table.horizontalHeader().setStretchLastSection(True)
        self.step_table.setToolTip("List of steps with their duration and presets.")
        self.step_table.setAlternatingRowColors(True) # Better readability
        self.step_table.verticalHeader().setVisible(False)
        step_layout.addWidget(self.step_table)
        
        # Session Duration Label
        self.session_duration_label = QLabel("Total Session Duration: 00:00")
        self.session_duration_label.setAlignment(Qt.AlignRight)
        self.session_duration_label.setStyleSheet("font-weight: bold; color: #888;")
        step_layout.addWidget(self.session_duration_label)
        
        # Connect model changes to duration update
        self.step_model.dataChanged.connect(self._update_total_duration)
        self.step_model.rowsInserted.connect(self._update_total_duration)
        self.step_model.rowsRemoved.connect(self._update_total_duration)
        self.step_model.modelReset.connect(self._update_total_duration)

        step_buttons = QHBoxLayout()
        step_buttons.setContentsMargins(0, 10, 0, 15)  # Add top and bottom padding
        self.add_step_btn = QPushButton(self.style().standardIcon(QStyle.SP_FileDialogNewFolder), "Add")
        self.add_step_btn.setToolTip("Insert a new step.")
        self.remove_step_btn = QPushButton(self.style().standardIcon(QStyle.SP_TrashIcon), "Remove")
        self.remove_step_btn.setToolTip("Remove selected step.")
        self.remove_step_btn.setProperty("class", "destructive")

        self.move_up_btn = QPushButton(self.style().standardIcon(QStyle.SP_ArrowUp), "Up")
        self.move_up_btn.setToolTip("Move step earlier.")
        self.move_down_btn = QPushButton(self.style().standardIcon(QStyle.SP_ArrowDown), "Down")
        self.move_down_btn.setToolTip("Move step later.")
        
        step_buttons.addWidget(self.add_step_btn)
        step_buttons.addWidget(self.remove_step_btn)
        step_buttons.addStretch()
        step_buttons.addWidget(self.move_up_btn)
        step_buttons.addWidget(self.move_down_btn)
        step_layout.addLayout(step_buttons)
        
        splitter.addWidget(step_container)

        # Right: Step Details (Card Style)
        self.editor_panel = QFrame()
        self.editor_panel.setObjectName("editor_panel")
        editor_main_layout = QVBoxLayout(self.editor_panel)
        editor_main_layout.setContentsMargins(20, 20, 20, 20)
        editor_main_layout.setSpacing(15)

        self.editor_header = QLabel("Step Details")
        self.editor_header.setObjectName("panel_header")
        editor_main_layout.addWidget(self.editor_header)

        # === Three-column layout for Binaural, Noise, Background ===
        columns_widget = QWidget()
        columns_layout = QHBoxLayout(columns_widget)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(20)

        # --- Binaural Column ---
        binaural_col = QWidget()
        binaural_col_layout = QVBoxLayout(binaural_col)
        binaural_col_layout.setContentsMargins(0, 0, 0, 0)
        binaural_col_layout.setSpacing(8)
        binaural_col_layout.setAlignment(Qt.AlignHCenter)

        binaural_label = QLabel("Binaural")
        binaural_label.setObjectName("column_header")
        binaural_label.setAlignment(Qt.AlignCenter)
        binaural_col_layout.addWidget(binaural_label)

        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Select the binaural preset used for this step.")
        self.preset_combo.setMinimumWidth(100)
        self.preset_combo.setMaximumWidth(140)
        self.preset_combo.setEditable(True)
        self.preset_combo.lineEdit().setReadOnly(True)
        self.preset_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self._preset_combo_filter = ComboBoxClickFilter(self.preset_combo)
        self.preset_combo.lineEdit().installEventFilter(self._preset_combo_filter)
        binaural_col_layout.addWidget(self.preset_combo, alignment=Qt.AlignCenter)

        self.binaural_vol_slider = QSlider(Qt.Vertical)
        self.binaural_vol_slider.setRange(0, 100)
        self.binaural_vol_slider.setToolTip("Volume of the binaural preset (0-100%).")
        self.binaural_vol_slider.setMinimumHeight(120)
        self.binaural_vol_slider.setFixedWidth(30)
        binaural_col_layout.addWidget(self.binaural_vol_slider, alignment=Qt.AlignCenter)

        self.binaural_vol_display = QLabel("0.00")
        self.binaural_vol_display.setAlignment(Qt.AlignCenter)
        self.binaural_vol_display.setVisible(False)
        binaural_col_layout.addWidget(self.binaural_vol_display, alignment=Qt.AlignCenter)

        self.binaural_vol_spin = QDoubleSpinBox()
        self.binaural_vol_spin.setDecimals(2)
        self.binaural_vol_spin.setRange(0.0, 1.0)
        self.binaural_vol_spin.setSingleStep(0.02)
        self.binaural_vol_spin.setToolTip("Binaural volume (0.0-1.0).")
        self.binaural_vol_spin.setFixedWidth(70)
        self.binaural_vol_spin.setAlignment(Qt.AlignCenter)
        self.binaural_vol_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        binaural_col_layout.addWidget(self.binaural_vol_spin, alignment=Qt.AlignCenter)

        columns_layout.addWidget(binaural_col)

        # --- Noise Column ---
        noise_col = QWidget()
        noise_col_layout = QVBoxLayout(noise_col)
        noise_col_layout.setContentsMargins(0, 0, 0, 0)
        noise_col_layout.setSpacing(8)
        noise_col_layout.setAlignment(Qt.AlignHCenter)

        noise_label = QLabel("Noise")
        noise_label.setObjectName("column_header")
        noise_label.setAlignment(Qt.AlignCenter)
        noise_col_layout.addWidget(noise_label)

        self.noise_combo = QComboBox()
        self.noise_combo.setToolTip("Optional noise preset blended with the step.")
        self.noise_combo.setMinimumWidth(100)
        self.noise_combo.setMaximumWidth(140)
        self.noise_combo.setEditable(True)
        self.noise_combo.lineEdit().setReadOnly(True)
        self.noise_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self._noise_combo_filter = ComboBoxClickFilter(self.noise_combo)
        self.noise_combo.lineEdit().installEventFilter(self._noise_combo_filter)
        noise_col_layout.addWidget(self.noise_combo, alignment=Qt.AlignCenter)

        self.noise_vol_slider = QSlider(Qt.Vertical)
        self.noise_vol_slider.setRange(0, 100)
        self.noise_vol_slider.setToolTip("Volume of the noise preset (0-100%).")
        self.noise_vol_slider.setMinimumHeight(120)
        self.noise_vol_slider.setFixedWidth(30)
        noise_col_layout.addWidget(self.noise_vol_slider, alignment=Qt.AlignCenter)

        self.noise_vol_display = QLabel("0.00")
        self.noise_vol_display.setAlignment(Qt.AlignCenter)
        self.noise_vol_display.setVisible(False)
        noise_col_layout.addWidget(self.noise_vol_display, alignment=Qt.AlignCenter)

        self.noise_vol_spin = QDoubleSpinBox()
        self.noise_vol_spin.setDecimals(2)
        self.noise_vol_spin.setRange(0.0, 1.0)
        self.noise_vol_spin.setSingleStep(0.02)
        self.noise_vol_spin.setToolTip("Noise volume (0.0-1.0).")
        self.noise_vol_spin.setFixedWidth(70)
        self.noise_vol_spin.setAlignment(Qt.AlignCenter)
        self.noise_vol_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        noise_col_layout.addWidget(self.noise_vol_spin, alignment=Qt.AlignCenter)

        columns_layout.addWidget(noise_col)

        # --- Background Column ---
        bg_col = QWidget()
        bg_col_layout = QVBoxLayout(bg_col)
        bg_col_layout.setContentsMargins(0, 0, 0, 0)
        bg_col_layout.setSpacing(8)
        bg_col_layout.setAlignment(Qt.AlignHCenter)

        bg_label = QLabel("Background")
        bg_label.setObjectName("column_header")
        bg_label.setAlignment(Qt.AlignCenter)
        bg_col_layout.addWidget(bg_label)

        self.bg_audio_btn = QPushButton("None")
        self.bg_audio_btn.setObjectName("preset_button")
        self.bg_audio_btn.setToolTip("Click to choose a background audio file from disk.")
        self.bg_audio_btn.setMinimumWidth(100)
        self.bg_audio_btn.setMaximumWidth(140)
        self.bg_audio_btn.setStyleSheet("text-align: center;")
        bg_col_layout.addWidget(self.bg_audio_btn, alignment=Qt.AlignCenter)

        # Hidden line edit to store the actual path
        self.bg_audio_edit = QLineEdit()
        self.bg_audio_edit.setVisible(False)

        self.bg_audio_vol_slider = QSlider(Qt.Vertical)
        self.bg_audio_vol_slider.setRange(0, 100)
        self.bg_audio_vol_slider.setToolTip("Volume of the background audio (0-100%).")
        self.bg_audio_vol_slider.setMinimumHeight(120)
        self.bg_audio_vol_slider.setFixedWidth(30)
        bg_col_layout.addWidget(self.bg_audio_vol_slider, alignment=Qt.AlignCenter)

        self.bg_audio_vol_display = QLabel("0.00")
        self.bg_audio_vol_display.setAlignment(Qt.AlignCenter)
        self.bg_audio_vol_display.setVisible(False)
        bg_col_layout.addWidget(self.bg_audio_vol_display, alignment=Qt.AlignCenter)

        self.bg_audio_vol_spin = QDoubleSpinBox()
        self.bg_audio_vol_spin.setDecimals(2)
        self.bg_audio_vol_spin.setRange(0.0, 1.0)
        self.bg_audio_vol_spin.setSingleStep(0.02)
        self.bg_audio_vol_spin.setToolTip("Background audio volume (0.0-1.0).")
        self.bg_audio_vol_spin.setFixedWidth(70)
        self.bg_audio_vol_spin.setAlignment(Qt.AlignCenter)
        self.bg_audio_vol_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        bg_col_layout.addWidget(self.bg_audio_vol_spin, alignment=Qt.AlignCenter)

        columns_layout.addWidget(bg_col)

        editor_main_layout.addWidget(columns_widget)

        # === Bottom controls row: Duration, Crossfade, Browse, Clear, Extend ===
        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.setContentsMargins(0, 10, 0, 0)
        bottom_layout.setSpacing(15)

        # Duration
        duration_widget = QWidget()
        duration_layout = QVBoxLayout(duration_widget)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(4)
        duration_label = QLabel("Duration")
        duration_label.setAlignment(Qt.AlignCenter)
        duration_layout.addWidget(duration_label)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(2)
        self.duration_spin.setRange(1.0, 7200.0)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setToolTip("Duration of the current step in seconds.")
        self.duration_spin.setFixedWidth(80)
        duration_layout.addWidget(self.duration_spin, alignment=Qt.AlignCenter)
        bottom_layout.addWidget(duration_widget)

        # Crossfade - label centered above spin box, checkbox to the right
        crossfade_widget = QWidget()
        crossfade_outer_layout = QHBoxLayout(crossfade_widget)
        crossfade_outer_layout.setContentsMargins(0, 0, 0, 0)
        crossfade_outer_layout.setSpacing(6)
        crossfade_outer_layout.setAlignment(Qt.AlignBottom)

        # Column with label + spin box
        crossfade_spin_col = QWidget()
        crossfade_spin_layout = QVBoxLayout(crossfade_spin_col)
        crossfade_spin_layout.setContentsMargins(0, 0, 0, 0)
        crossfade_spin_layout.setSpacing(4)
        crossfade_label = QLabel("Crossfade")
        crossfade_label.setAlignment(Qt.AlignCenter)
        crossfade_spin_layout.addWidget(crossfade_label)
        self.step_crossfade_spin = QDoubleSpinBox()
        self.step_crossfade_spin.setDecimals(2)
        self.step_crossfade_spin.setRange(0.0, 30.0)
        self.step_crossfade_spin.setSingleStep(0.1)
        self.step_crossfade_spin.setToolTip("Crossfade override for this step (seconds).")
        self.step_crossfade_spin.setFixedWidth(80)
        crossfade_spin_layout.addWidget(self.step_crossfade_spin, alignment=Qt.AlignCenter)
        crossfade_outer_layout.addWidget(crossfade_spin_col)

        # Checkbox to the right
        self.step_crossfade_use_global = QCheckBox("Use Global")
        self.step_crossfade_use_global.setToolTip("Use the session's global crossfade duration for this step.")
        crossfade_outer_layout.addWidget(self.step_crossfade_use_global, alignment=Qt.AlignBottom)

        # Hidden slider for compatibility (not displayed in new UI)
        self.step_crossfade_slider = QSlider(Qt.Horizontal)
        self.step_crossfade_slider.setRange(0, 300)
        self.step_crossfade_slider.setVisible(False)
        # Hidden curve combo for compatibility
        self.step_crossfade_curve_combo = QComboBox()
        self.step_crossfade_curve_combo.addItems(["Use Session", "linear", "equal_power"])
        self.step_crossfade_curve_combo.setVisible(False)
        bottom_layout.addWidget(crossfade_widget)

        bottom_layout.addStretch()

        # Browse button
        browse_widget = QWidget()
        browse_layout = QVBoxLayout(browse_widget)
        browse_layout.setContentsMargins(0, 0, 0, 0)
        browse_layout.setSpacing(4)
        browse_label = QLabel("Browse")
        browse_label.setAlignment(Qt.AlignCenter)
        browse_layout.addWidget(browse_label)
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.browse_btn.setToolTip("Choose a background audio file.")
        self.browse_btn.setFixedSize(40, 30)
        browse_layout.addWidget(self.browse_btn, alignment=Qt.AlignCenter)
        bottom_layout.addWidget(browse_widget)

        # Clear button
        clear_widget = QWidget()
        clear_layout = QVBoxLayout(clear_widget)
        clear_layout.setContentsMargins(0, 0, 0, 0)
        clear_layout.setSpacing(4)
        clear_label = QLabel("Clear")
        clear_label.setAlignment(Qt.AlignCenter)
        clear_layout.addWidget(clear_label)
        self.bg_audio_clear_btn = QPushButton()
        self.bg_audio_clear_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.bg_audio_clear_btn.setToolTip("Clear background audio selection.")
        self.bg_audio_clear_btn.setFixedSize(40, 30)
        clear_layout.addWidget(self.bg_audio_clear_btn, alignment=Qt.AlignCenter)
        bottom_layout.addWidget(clear_widget)

        # Extend checkbox
        extend_widget = QWidget()
        extend_layout = QVBoxLayout(extend_widget)
        extend_layout.setContentsMargins(0, 0, 0, 0)
        extend_layout.setSpacing(4)
        extend_label = QLabel("Extend")
        extend_label.setAlignment(Qt.AlignCenter)
        extend_layout.addWidget(extend_label)
        self.bg_audio_extend_checkbox = QCheckBox()
        self.bg_audio_extend_checkbox.setToolTip(
            "When checked, the loaded background audio continues into subsequent steps "
            "if it is longer than this step's duration."
        )
        self.bg_audio_extend_checkbox.setChecked(True)
        extend_layout.addWidget(self.bg_audio_extend_checkbox, alignment=Qt.AlignCenter)
        bottom_layout.addWidget(extend_widget)

        editor_main_layout.addWidget(bottom_row)

        # === Description section ===
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_layout.setContentsMargins(0, 10, 0, 0)
        desc_layout.setSpacing(4)

        desc_label = QLabel("Description")
        desc_layout.addWidget(desc_label)

        self.description_edit = QTextEdit()
        self.description_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.description_edit.setToolTip("Notes about the intention or feel of the step.")
        self.description_edit.setMaximumHeight(80)
        desc_layout.addWidget(self.description_edit)

        editor_main_layout.addWidget(desc_widget)
        editor_main_layout.addStretch()

        splitter.addWidget(self.editor_panel)
        self.editor_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set initial splitter sizes (approx 40% list, 60% details)
        splitter.setSizes([400, 600])

        # --- Bottom Panel: Playback Controls ---
        playback_panel = QFrame()
        playback_panel.setObjectName("playback_panel")
        playback_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        playback_layout = QHBoxLayout(playback_panel)
        playback_layout.setContentsMargins(10, 10, 10, 10)
        playback_layout.setSpacing(15)

        self.skip_back_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipBackward), "")
        self.skip_back_btn.setToolTip("Skip to previous step.")
        
        self.play_pause_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.play_pause_btn.setToolTip("Play/Pause")
        
        self.stop_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self.stop_btn.setToolTip("Stop playback.")
        
        self.skip_fwd_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipForward), "")
        self.skip_fwd_btn.setToolTip("Skip to next step.")

        self.time_label = QLabel("00:00")
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setToolTip("Seek position.")
        self.total_time_label = QLabel("00:00")

        self.vol_icon = QLabel()
        self.vol_icon.setPixmap(self.style().standardIcon(QStyle.SP_MediaVolume).pixmap(16, 16))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.setToolTip("Playback volume.")

        playback_layout.addWidget(self.skip_back_btn)
        playback_layout.addWidget(self.play_pause_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(self.skip_fwd_btn)
        playback_layout.addWidget(self.time_label)
        playback_layout.addWidget(self.seek_slider, 1)
        playback_layout.addWidget(self.total_time_label)
        playback_layout.addSpacing(10)
        playback_layout.addWidget(self.vol_icon)
        playback_layout.addWidget(self.vol_slider)

        # Status label with backend indicator
        backend_type = "Rust" if self._using_rust_backend else "Python"
        self.status_label = QLabel(f"Ready (Streaming: {backend_type} backend)", central)
        self.status_label.setStyleSheet("color: #888888;")
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # --- Vertical Splitter to improve vertical resize behavior ---
        bottom_container = QWidget()
        bottom_container_layout = QVBoxLayout(bottom_container)
        bottom_container_layout.setContentsMargins(0, 0, 0, 0)
        bottom_container_layout.setSpacing(6)
        bottom_container_layout.addWidget(playback_panel)
        bottom_container_layout.addWidget(self.status_label)

        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.setHandleWidth(6)
        vertical_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vertical_splitter.addWidget(control_panel)
        vertical_splitter.addWidget(splitter)
        vertical_splitter.addWidget(bottom_container)
        vertical_splitter.setStretchFactor(0, 0)
        vertical_splitter.setStretchFactor(1, 1)
        vertical_splitter.setStretchFactor(2, 0)
        vertical_splitter.setSizes(
            [
                control_panel.sizeHint().height(),
                max(600, splitter.sizeHint().height()),
                playback_panel.sizeHint().height() + self.status_label.sizeHint().height(),
            ]
        )

        # Add scrollable container so the window remains usable when vertically constrained
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)

        content_wrapper = QWidget()
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(vertical_splitter)

        scroll_area.setWidget(content_wrapper)

        # Add scroll area to central layout
        main_layout.addWidget(scroll_area, 1)

        self._populate_presets()
        self._apply_initial_defaults()
        self._bind_signals()

    def _apply_initial_defaults(self) -> None:
        """Apply defaults to UI controls on startup."""
        # Duration
        def_dur = float(self._defaults.get("step_duration", 300.0))
        self.duration_spin.setValue(def_dur)
        
        # Binaural Preset
        def_bin = self._defaults.get("default_binaural_id")
        if def_bin:
            idx = self.preset_combo.findData(def_bin)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
                
        # Noise Preset
        def_noise = self._defaults.get("default_noise_id")
        if def_noise:
            idx = self.noise_combo.findData(def_noise)
            if idx >= 0:
                self.noise_combo.setCurrentIndex(idx)

    def _center_combo_items(self, combo: QComboBox) -> None:
        """Center the display text for all items in the combo box."""
        for idx in range(combo.count()):
            combo.setItemData(idx, Qt.AlignCenter, Qt.TextAlignmentRole)

    def _populate_presets(self) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItem("None", None)
        for preset_id, preset in sorted(self._binaural_catalog.items()):
            self.preset_combo.addItem(preset.label, preset_id)
        self._center_combo_items(self.preset_combo)
        self.noise_combo.clear()
        self.noise_combo.addItem("None", None)
        for preset_id, preset in sorted(self._noise_catalog.items()):
            self.noise_combo.addItem(preset.label, preset_id)
        self._center_combo_items(self.noise_combo)

    def _bind_signals(self) -> None:
        self.crossfade_slider.valueChanged.connect(self._sync_crossfade_spin_from_slider)
        self.crossfade_spin.valueChanged.connect(self._sync_crossfade_slider_from_spin)
        self.crossfade_curve_combo.currentTextChanged.connect(self._on_crossfade_curve_changed)
        self.normalization_slider.valueChanged.connect(self._on_normalization_changed)

        self.step_table.selectionModel().selectionChanged.connect(self._on_step_selection_changed)
        self.add_step_btn.clicked.connect(self._add_step)
        self.remove_step_btn.clicked.connect(self._remove_step)
        self.move_up_btn.clicked.connect(lambda: self._move_step(-1))
        self.move_down_btn.clicked.connect(lambda: self._move_step(1))

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.binaural_vol_slider.valueChanged.connect(self._sync_binaural_vol_spin_from_slider)
        self.binaural_vol_spin.valueChanged.connect(self._sync_binaural_vol_slider_from_spin)
        self.noise_combo.currentIndexChanged.connect(self._on_noise_changed)
        self.noise_vol_slider.valueChanged.connect(self._sync_noise_vol_spin_from_slider)
        self.noise_vol_spin.valueChanged.connect(self._sync_noise_vol_slider_from_spin)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        self.step_crossfade_slider.valueChanged.connect(self._sync_step_crossfade_spin_from_slider)
        self.step_crossfade_spin.valueChanged.connect(self._sync_step_crossfade_slider_from_spin)
        self.step_crossfade_curve_combo.currentIndexChanged.connect(self._on_step_curve_changed)
        self.step_crossfade_use_global.stateChanged.connect(self._on_use_global_crossfade_changed)
        self.bg_audio_btn.clicked.connect(self._choose_background_audio_file)
        self.browse_btn.clicked.connect(self._choose_background_audio_file)
        self.bg_audio_clear_btn.clicked.connect(self._clear_background_audio)
        self.bg_audio_vol_slider.valueChanged.connect(self._sync_bg_audio_vol_spin_from_slider)
        self.bg_audio_vol_spin.valueChanged.connect(self._sync_bg_audio_vol_slider_from_spin)
        self.bg_audio_extend_checkbox.stateChanged.connect(self._on_bg_audio_extend_changed)
        self.description_edit.textChanged.connect(self._on_description_changed)

        self.save_btn.clicked.connect(self._save_session)
        self.load_btn.clicked.connect(self._load_session_from_file)
        self.export_btn.clicked.connect(self._export_session)

        # Playback controls
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self.stop_btn.clicked.connect(self._stop_playback)
        self.skip_back_btn.clicked.connect(self._skip_backward)
        self.skip_fwd_btn.clicked.connect(self._skip_forward)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100)
        self._playback_timer.timeout.connect(self._update_playback_ui)
        self._is_seeking = False

    # ------------------------------------------------------------------
    # Session & model synchronization
    # ------------------------------------------------------------------
    def _invalidate_assembler(self) -> None:
        """Clear any cached assembler so exports rebuild from current state."""

        self._current_assembler = None
        self._schedule_stream_update()

    def _schedule_stream_update(self) -> None:
        """Debounce live stream refreshes while sliders are being dragged."""

        if not isinstance(self._stream_player, RustStreamPlayer):
            self._stream_update_timer.stop()
            return

        # Restart the timer so only the final value after a drag triggers an update
        self._stream_update_timer.start()

    def _push_stream_update(self) -> None:
        """Send the latest track data to the Rust backend while streaming.

        The realtime Rust backend supports live updates via ``update_track``.
        Whenever the GUI changes a session parameter, we rebuild the assembler
        payload and push it to the backend so playback immediately reflects the
        current settings. Python fallback players ignore these updates.
        """

        if not isinstance(self._stream_player, RustStreamPlayer):
            return

        try:
            assembler = self._create_assembler()
        except Exception:
            return

        try:
            self._stream_player.update_track(assembler.track_data)
        except Exception:
            # Best-effort update; keep playback running even if live refresh fails
            pass

    def _refresh_step_model(self, *, preserve_selection: bool = True, select_index: Optional[int] = None) -> None:
        """Refresh the step model and keep the current editing row highlighted."""

        if preserve_selection and select_index is None:
            select_index = self._selected_step_index()

        self.step_model.refresh(self._session.steps)

        if select_index is not None and 0 <= select_index < len(self._session.steps):
            self.step_table.selectRow(select_index)

    def _load_session(self, session: Session) -> None:
        self._session = session
        self._invalidate_assembler()
        self.crossfade_spin.blockSignals(True)
        self.crossfade_slider.blockSignals(True)
        self.crossfade_spin.setValue(float(session.crossfade_duration))
        self.crossfade_slider.setValue(int(round(session.crossfade_duration * 10)))
        self.crossfade_spin.blockSignals(False)
        self.crossfade_slider.blockSignals(False)
        idx = self.crossfade_curve_combo.findText(session.crossfade_curve or "linear")
        if idx >= 0:
            self.crossfade_curve_combo.setCurrentIndex(idx)
        self.normalization_slider.blockSignals(True)
        current_norm = getattr(session, "normalization_level", 0.95)
        current_norm = max(0.0, min(current_norm, MAX_NORMALIZATION_UI))
        self.normalization_slider.setValue(int(round(current_norm * 100)))
        self.normalization_slider.blockSignals(False)
        self._update_normalization_label(self.normalization_slider.value())
        self.step_model.refresh(session.steps)
        if session.steps:
            self.step_table.selectRow(0)
        else:
            self._clear_step_editors()
        self._update_total_duration()

    def _open_defaults_dialog(self) -> None:
        dlg = DefaultsDialog(self._binaural_catalog, self._noise_catalog, self)
        if dlg.exec_():
            self._defaults = load_defaults()

            # Apply to current session immediately
            new_norm = float(self._defaults.get("normalization_level", 0.95))
            new_norm = max(0.0, min(new_norm, MAX_NORMALIZATION_UI))
            new_cross = float(self._defaults.get("crossfade_duration", 10.0))
            new_dur = float(self._defaults.get("step_duration", 300.0))

            # Update Session
            self._session.normalization_level = new_norm
            self._session.crossfade_duration = new_cross
            
            # Update UI controls
            self.normalization_slider.blockSignals(True)
            self.normalization_slider.setValue(int(round(new_norm * 100)))
            self.normalization_slider.blockSignals(False)
            self._update_normalization_label(int(round(new_norm * 100)))
            
            self.crossfade_spin.setValue(new_cross) # Signals will update slider

            # Update duration spinbox for next step
            self.duration_spin.setValue(new_dur)

            self._invalidate_assembler()

            self.status_label.setText("Defaults updated and applied.")
            
    def _update_total_duration(self, *args) -> None:
        total_seconds = 0.0
        # Simple sum for now, could account for crossfade overlap if needed for precise timeline
        # But usually "session duration" implies the timeline length.
        # Timeline length = sum(durations) - sum(crossfades) ?
        # Actually, in this engine, steps are sequential. Crossfade eats into the previous step or overlaps?
        # Let's check session_stream logic:
        # total += samples (first)
        # total += max(samples - prev_crossfade, 0) (subsequent)
        
        # Let's replicate that logic roughly
        crossfade = self._session.crossfade_duration
        first = True
        for step in self._session.steps:
            dur = step.duration
            if first:
                total_seconds += dur
                first = False
            else:
                # If crossfade is global, use session.crossfade_duration
                # If per-step, use step.crossfade_duration
                # The engine uses step.crossfade or global default.
                step_xf = step.crossfade_duration if step.crossfade_duration is not None else crossfade
                total_seconds += max(0, dur - step_xf)
        
        m = int(total_seconds // 60)
        s = int(total_seconds % 60)
        self.session_duration_label.setText(f"Total Session Duration: {m:02d}:{s:02d}")

    def _selected_step_index(self) -> Optional[int]:
        sel = self.step_table.selectionModel().selectedRows()
        if not sel:
            return None
        return sel[0].row()

    def _get_selected_step(self) -> Optional[SessionStep]:
        index = self._selected_step_index()
        if index is None:
            return None
        if index < 0 or index >= len(self._session.steps):
            return None
        return self._session.steps[index]

    def _on_step_selection_changed(self, *_args) -> None:
        step = self._get_selected_step()
        if step is None:
            self._clear_step_editors()
            return
        self._load_step_into_editors(step)

    def _load_step_into_editors(self, step: SessionStep) -> None:
        self.preset_combo.blockSignals(True)
        self.noise_combo.blockSignals(True)
        self.noise_vol_slider.blockSignals(True)
        self.noise_vol_spin.blockSignals(True)
        self.duration_spin.blockSignals(True)
        self.step_crossfade_slider.blockSignals(True)
        self.step_crossfade_spin.blockSignals(True)
        self.step_crossfade_use_global.blockSignals(True)
        self.step_crossfade_curve_combo.blockSignals(True)
        self.bg_audio_vol_slider.blockSignals(True)
        self.bg_audio_vol_spin.blockSignals(True)
        self.bg_audio_extend_checkbox.blockSignals(True)

        idx = self.preset_combo.findData(step.binaural_preset_id)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        else:
            # None or unknown preset - select "None" which is index 0
            self.preset_combo.setCurrentIndex(0)

        binaural_vol = getattr(step, "binaural_volume", MAX_INDIVIDUAL_GAIN)
        # Internal value (0-MAX_INDIVIDUAL_GAIN) -> display value (0-1.0)
        normalized_binaural = binaural_vol / MAX_INDIVIDUAL_GAIN
        self.binaural_vol_slider.setValue(int(round(normalized_binaural * 100)))
        self.binaural_vol_spin.setValue(normalized_binaural)

        idx = self.noise_combo.findData(step.noise_preset_id)
        self.noise_combo.setCurrentIndex(idx if idx >= 0 else 0)

        noise_vol = getattr(step, "noise_volume", MAX_INDIVIDUAL_GAIN)
        # Internal value (0-MAX_INDIVIDUAL_GAIN) -> display value (0-1.0)
        normalized_noise = noise_vol / MAX_INDIVIDUAL_GAIN
        self.noise_vol_slider.setValue(int(round(normalized_noise * 100)))
        self.noise_vol_spin.setValue(normalized_noise)

        self.duration_spin.setValue(step.duration)
        crossfade_duration = step.crossfade_duration if step.crossfade_duration is not None else 0.0
        self.step_crossfade_slider.setValue(int(round(crossfade_duration * 10)))
        self.step_crossfade_spin.setValue(crossfade_duration)
        if step.crossfade_curve:
            idx = self.step_crossfade_curve_combo.findText(step.crossfade_curve)
            self.step_crossfade_curve_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.step_crossfade_curve_combo.setCurrentIndex(0)
        use_global_crossfade = step.crossfade_duration is None
        self.step_crossfade_use_global.setChecked(use_global_crossfade)
        self._update_crossfade_editor_state(use_global_crossfade)

        # Load background audio path (with fallback to legacy warmup_clip_path)
        bg_audio_path = step.background_audio_path or step.warmup_clip_path or ""
        self.bg_audio_edit.setText(bg_audio_path)
        self._update_bg_audio_btn_text(bg_audio_path)

        # Load background audio volume
        bg_audio_vol = getattr(step, "background_audio_volume", MAX_INDIVIDUAL_GAIN)
        # Internal value (0-MAX_INDIVIDUAL_GAIN) -> display value (0-1.0)
        normalized_bg = bg_audio_vol / MAX_INDIVIDUAL_GAIN
        self.bg_audio_vol_slider.setValue(int(round(normalized_bg * 100)))
        self.bg_audio_vol_spin.setValue(normalized_bg)

        # Load background audio extend flag (default True for backwards compatibility)
        bg_audio_extend = getattr(step, "background_audio_extend", True)
        self.bg_audio_extend_checkbox.setChecked(bg_audio_extend)

        self.description_edit.blockSignals(True)
        self.description_edit.setPlainText(step.description)
        self.description_edit.blockSignals(False)

        self.preset_combo.blockSignals(False)
        self.binaural_vol_slider.blockSignals(False)
        self.binaural_vol_spin.blockSignals(False)
        self.noise_combo.blockSignals(False)
        self.noise_vol_slider.blockSignals(False)
        self.noise_vol_spin.blockSignals(False)
        self.duration_spin.blockSignals(False)
        self.step_crossfade_slider.blockSignals(False)
        self.step_crossfade_spin.blockSignals(False)
        self.step_crossfade_use_global.blockSignals(False)
        self.step_crossfade_curve_combo.blockSignals(False)
        self.bg_audio_vol_slider.blockSignals(False)
        self.bg_audio_vol_spin.blockSignals(False)
        self.bg_audio_extend_checkbox.blockSignals(False)
        self._refresh_volume_displays()

    def _clear_step_editors(self) -> None:
        default_duration = float(self._defaults.get("step_duration", 1.0))
        default_binaural = self._defaults.get("default_binaural_id")
        default_noise = self._defaults.get("default_noise_id")

        # Binaural preset selection
        if default_binaural:
            idx = self.preset_combo.findData(default_binaural)
            self.preset_combo.setCurrentIndex(idx if idx >= 0 else -1)
        else:
            self.preset_combo.setCurrentIndex(-1)

        # Default to max volume (slider at 100, spin at 1.0)
        self.binaural_vol_slider.setValue(100)
        self.binaural_vol_spin.setValue(1.0)

        # Noise preset selection ("None" entry is index 0)
        if default_noise:
            idx = self.noise_combo.findData(default_noise)
            self.noise_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.noise_combo.setCurrentIndex(0 if self.noise_combo.count() else -1)

        # Default to max volume (slider at 100, spin at 1.0)
        self.noise_vol_slider.setValue(100)
        self.noise_vol_spin.setValue(1.0)
        self.duration_spin.setValue(default_duration)
        self.step_crossfade_slider.setValue(0)
        self.step_crossfade_spin.setValue(0.0)
        self.step_crossfade_use_global.setChecked(True)
        self._update_crossfade_editor_state(True)
        self.step_crossfade_curve_combo.setCurrentIndex(0)
        # Clear background audio
        self.bg_audio_edit.clear()
        self._update_bg_audio_btn_text("")
        self.bg_audio_vol_slider.setValue(100)
        self.bg_audio_vol_spin.setValue(1.0)
        self.bg_audio_extend_checkbox.setChecked(True)
        self.description_edit.blockSignals(True)
        self.description_edit.clear()
        self.description_edit.blockSignals(False)
        self._refresh_volume_displays()

    def _sync_crossfade_spin_from_slider(self, value: int) -> None:
        seconds = value / 10.0
        self.crossfade_spin.setValue(seconds)
        self._session.crossfade_duration = seconds
        self._invalidate_assembler()
        self._refresh_global_crossfade_display_for_step()

    def _sync_crossfade_slider_from_spin(self, value: float) -> None:
        self.crossfade_slider.setValue(int(round(value * 10)))
        self._session.crossfade_duration = float(value)
        self._invalidate_assembler()
        self._refresh_global_crossfade_display_for_step()

    def _on_crossfade_curve_changed(self, text: str) -> None:
        self._session.crossfade_curve = text
        self._invalidate_assembler()

    def _on_normalization_changed(self, value: int) -> None:
        value = max(0, min(value, self.normalization_slider.maximum()))
        self._session.normalization_level = value / 100.0
        self._update_normalization_label(value)
        self._invalidate_assembler()

    def _update_normalization_label(self, slider_value: int) -> None:
        self.normalization_label.setText(f"{slider_value / 100:.2f}")

    def _update_volume_display_label(self, label: QLabel, slider_value: int) -> None:
        normalized = min(slider_value, 99) / 100.0
        label.setText(f"{normalized:.2f}")

    def _refresh_volume_displays(self) -> None:
        self._update_volume_display_label(self.binaural_vol_display, self.binaural_vol_slider.value())
        self._update_volume_display_label(self.noise_vol_display, self.noise_vol_slider.value())
        self._update_volume_display_label(self.bg_audio_vol_display, self.bg_audio_vol_slider.value())

    def _on_preset_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None or index < 0:
            return
        preset_id = self.preset_combo.itemData(index)
        # preset_id can be None for the "None" option
        step.binaural_preset_id = preset_id
        self._refresh_step_model()
        self._invalidate_assembler()

    def _on_noise_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.noise_preset_id = self.noise_combo.itemData(index)
        self._invalidate_assembler()

    def _sync_noise_vol_spin_from_slider(self, value: int) -> None:
        # Slider 0-100 maps to spin 0-1.0 (display), which maps to 0-MAX_INDIVIDUAL_GAIN (internal)
        normalized = value / 100.0
        self.noise_vol_spin.setValue(normalized)
        internal_vol = normalized * MAX_INDIVIDUAL_GAIN
        self._set_noise_volume(internal_vol)

    def _sync_noise_vol_slider_from_spin(self, value: float) -> None:
        # Spin 0-1.0 maps to slider 0-100
        self.noise_vol_slider.setValue(int(round(value * 100)))
        internal_vol = value * MAX_INDIVIDUAL_GAIN
        self._set_noise_volume(internal_vol)

    def _set_noise_volume(self, value: float) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.noise_volume = float(value)
        self._invalidate_assembler()

    def _sync_binaural_vol_spin_from_slider(self, value: int) -> None:
        # Slider 0-100 maps to spin 0-1.0 (display), which maps to 0-MAX_INDIVIDUAL_GAIN (internal)
        normalized = value / 100.0
        self.binaural_vol_spin.setValue(normalized)
        internal_vol = normalized * MAX_INDIVIDUAL_GAIN
        self._set_binaural_volume(internal_vol)

    def _sync_binaural_vol_slider_from_spin(self, value: float) -> None:
        # Spin 0-1.0 maps to slider 0-100
        self.binaural_vol_slider.setValue(int(round(value * 100)))
        internal_vol = value * MAX_INDIVIDUAL_GAIN
        self._set_binaural_volume(internal_vol)

    def _set_binaural_volume(self, value: float) -> None:
        step = self._get_selected_step()
        if step:
            step.binaural_volume = float(value)
            self._invalidate_assembler()

    def _on_duration_changed(self, value: float) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.duration = float(value)
        self._refresh_step_model()
        self._invalidate_assembler()

    def _sync_step_crossfade_spin_from_slider(self, value: int) -> None:
        seconds = value / 10.0
        self.step_crossfade_spin.setValue(seconds)
        self._set_step_crossfade(seconds)

    def _sync_step_crossfade_slider_from_spin(self, value: float) -> None:
        self.step_crossfade_slider.setValue(int(round(value * 10)))
        self._set_step_crossfade(value)

    def _set_step_crossfade(self, value: float) -> None:
        if self.step_crossfade_use_global.isChecked():
            self._apply_global_crossfade_display()
            step = self._get_selected_step()
            if step is not None:
                step.crossfade_duration = None
                self._invalidate_assembler()
            return
        step = self._get_selected_step()
        if step is None:
            return
        if value <= 0.0:
            step.crossfade_duration = None
        else:
            step.crossfade_duration = float(value)
        self._invalidate_assembler()

    def _on_step_curve_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        if index <= 0:
            step.crossfade_curve = None
        else:
            step.crossfade_curve = self.step_crossfade_curve_combo.itemText(index)
        self._invalidate_assembler()

    def _choose_background_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Background Audio", "src/presets/audio", "Audio Files (*.wav *.flac *.mp3 *.ogg *.aac);;All Files (*)")
        if not path:
            return
        self.bg_audio_edit.setText(path)
        self._update_bg_audio_btn_text(path)
        step = self._get_selected_step()
        if step is not None:
            step.background_audio_path = path
            self._invalidate_assembler()

    def _clear_background_audio(self) -> None:
        self.bg_audio_edit.clear()
        self._update_bg_audio_btn_text("")
        step = self._get_selected_step()
        if step is not None:
            step.background_audio_path = None
            self._invalidate_assembler()

    def _update_bg_audio_btn_text(self, path: str) -> None:
        """Update the background audio button text to show filename or indicate no selection."""
        if path:
            filename = Path(path).name
            # Truncate if too long
            if len(filename) > 12:
                filename = filename[:9] + "..."
            self.bg_audio_btn.setText(f"<{filename}>")
        else:
            self.bg_audio_btn.setText("None")

    def _sync_bg_audio_vol_spin_from_slider(self, value: int) -> None:
        # Slider 0-100 maps to spin 0-1.0 (display), which maps to 0-MAX_INDIVIDUAL_GAIN (internal)
        normalized = value / 100.0
        self.bg_audio_vol_spin.setValue(normalized)
        internal_vol = normalized * MAX_INDIVIDUAL_GAIN
        self._set_background_audio_volume(internal_vol)

    def _sync_bg_audio_vol_slider_from_spin(self, value: float) -> None:
        # Spin 0-1.0 maps to slider 0-100
        self.bg_audio_vol_slider.setValue(int(round(value * 100)))
        internal_vol = value * MAX_INDIVIDUAL_GAIN
        self._set_background_audio_volume(internal_vol)

    def _set_background_audio_volume(self, value: float) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.background_audio_volume = float(value)
        self._invalidate_assembler()

    def _on_bg_audio_extend_changed(self, state: int) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.background_audio_extend = bool(state)
        self._invalidate_assembler()

    def _on_description_changed(self) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.description = self.description_edit.toPlainText()
        self._refresh_step_model()
        self._invalidate_assembler()

    def _on_use_global_crossfade_changed(self, state: int) -> None:
        use_global = bool(state)
        self._update_crossfade_editor_state(use_global)
        step = self._get_selected_step()
        if step is None:
            return
        if use_global:
            step.crossfade_duration = None
            self._apply_global_crossfade_display()
        else:
            step.crossfade_duration = float(self.step_crossfade_spin.value())
        self._invalidate_assembler()

    def _update_crossfade_editor_state(self, use_global: bool) -> None:
        self.step_crossfade_spin.setEnabled(not use_global)
        self.step_crossfade_slider.setEnabled(not use_global)
        self.step_crossfade_curve_combo.setEnabled(not use_global)
        if use_global:
            self._apply_global_crossfade_display()

    def _apply_global_crossfade_display(self) -> None:
        value = float(self._session.crossfade_duration or 0.0)
        self.step_crossfade_spin.blockSignals(True)
        self.step_crossfade_slider.blockSignals(True)
        self.step_crossfade_spin.setValue(value)
        self.step_crossfade_slider.setValue(int(round(value * 10)))
        self.step_crossfade_spin.blockSignals(False)
        self.step_crossfade_slider.blockSignals(False)

    def _refresh_global_crossfade_display_for_step(self) -> None:
        if not self.step_crossfade_use_global.isChecked():
            return
        self._apply_global_crossfade_display()
        step = self._get_selected_step()
        if step is not None:
            step.crossfade_duration = None

    # ------------------------------------------------------------------
    # Step list manipulation
    # ------------------------------------------------------------------
    def _add_step(self) -> None:
        # Determine preset IDs: use UI selection if valid, else default, else first available
        ui_preset_id = self.preset_combo.itemData(self.preset_combo.currentIndex())
        default_preset_id = self._defaults.get("default_binaural_id")
        
        # Logic: If UI has a selection, use it? Or if adding a NEW step, should we reset to default?
        # Usually "Add Step" adds what's currently configured in the "Add Step" form.
        # But the user wants "default binaural" to set those defaults in the step GUI.
        # So we should probably update the UI to match defaults when the window opens or defaults change.
        # And _add_step just reads from the UI.
        
        # However, if the user just opened the app, the UI should already reflect defaults.
        # Let's assume _add_step reads from UI.
        
        # preset_id can be None if user explicitly selected "None"
        preset_id = ui_preset_id
        # Only fall back to default if combo is at initial state (not explicitly selected "None")
        if preset_id is None and self.preset_combo.currentIndex() != 0 and default_preset_id:
             # Try to find default in combo
             idx = self.preset_combo.findData(default_preset_id)
             if idx >= 0:
                 self.preset_combo.setCurrentIndex(idx)
                 preset_id = default_preset_id

        # Noise
        ui_noise_id = self.noise_combo.itemData(self.noise_combo.currentIndex())
        noise_id = ui_noise_id

        step = SessionStep(
            binaural_preset_id=preset_id,  # Can be None for no binaural
            duration=self.duration_spin.value(),
            noise_preset_id=noise_id,
            noise_volume=self.noise_vol_spin.value() * MAX_INDIVIDUAL_GAIN,  # Convert display (0-1) to internal
            binaural_volume=self.binaural_vol_spin.value() * MAX_INDIVIDUAL_GAIN,  # Convert display (0-1) to internal
            crossfade_duration=None,
            crossfade_curve=None,
            background_audio_path=self.bg_audio_edit.text() or None,
            background_audio_volume=self.bg_audio_vol_spin.value() * MAX_INDIVIDUAL_GAIN,  # Convert display (0-1) to internal
            background_audio_extend=self.bg_audio_extend_checkbox.isChecked(),
            description=self.description_edit.toPlainText(),
        )
        self._session.steps.append(step)
        self._refresh_step_model(select_index=len(self._session.steps) - 1)
        self._invalidate_assembler()

    def _remove_step(self) -> None:
        index = self._selected_step_index()
        if index is None:
            return
        if 0 <= index < len(self._session.steps):
            del self._session.steps[index]
            self._refresh_step_model(select_index=min(index, len(self._session.steps) - 1))
            if not self._session.steps:
                self._clear_step_editors()
            self._invalidate_assembler()

    def _move_step(self, direction: int) -> None:
        index = self._selected_step_index()
        if index is None:
            return
        new_index = index + direction
        if new_index < 0 or new_index >= len(self._session.steps):
            return
        steps = self._session.steps
        steps[index], steps[new_index] = steps[new_index], steps[index]
        self._refresh_step_model(select_index=new_index)
        self._invalidate_assembler()

    # ------------------------------------------------------------------
    # Save/load handling
    # ------------------------------------------------------------------
    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "session.session", "Session Files (*.session *.json)")
        if not path:
            return
        data = _session_to_dict(self._session)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Save Error", f"Failed to save session: {exc}")
            return
        self.status_label.setText(f"Session saved to {Path(path).name}")

    def _load_session_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "Session Files (*.session *.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Load Error", f"Failed to load session: {exc}")
            return
        session = _session_from_dict(data)
        self._load_session(session)
        self.status_label.setText(f"Session loaded from {Path(path).name}")

    # ------------------------------------------------------------------
    # Audio preview/export
    # ------------------------------------------------------------------
    def _current_normalization(self) -> float:
        return float(self.normalization_slider.value()) / 100.0

    def _create_assembler(self) -> SessionAssembler:
        assembler = self._assembler_factory(
            self._session,
            self._binaural_catalog,
            self._noise_catalog,
            sample_rate=self._session.sample_rate,
            crossfade_curve=self._session.crossfade_curve,
            normalization_ceiling=self._current_normalization(),
        )
        self._current_assembler = assembler
        return assembler

    def _toggle_playback(self) -> None:
        if self._stream_player is None:
            self._start_playback()
        else:
            # Use explicit state tracking for reliable pause/resume toggling
            if not self._is_paused:
                self._stream_player.pause()
                self._is_paused = True
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self._stream_player.resume()
                self._is_paused = False
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def _start_playback(self) -> None:
        try:
            track_data = self._create_assembler().track_data
            # Inject normalization if needed (already handled in session_to_track_data via session.normalization_level)
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", f"Failed to assemble session: {exc}")
            return

        self._stream_player = self._stream_player_factory(track_data)
        self._stream_player.set_volume(self.vol_slider.value() / 100.0)

        # Use prebuffer for scrubbable playback
        self._stream_player.start(use_prebuffer=False)

        self._playback_timer.start()
        self._is_paused = False
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        # Indicate which backend is being used
        backend_name = "Rust" if isinstance(self._stream_player, RustStreamPlayer) else "Python"
        self.status_label.setText(f"Playing... ({backend_name} backend)")

    def _stop_playback(self) -> None:
        if self._stream_player:
            self._stream_player.stop()
            self._stream_player = None
        self._playback_timer.stop()
        self._is_paused = False
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.seek_slider.setValue(0)
        self.time_label.setText("00:00")
        backend_type = "Rust" if self._using_rust_backend else "Python"
        self.status_label.setText(f"Stopped (Streaming: {backend_type} backend)")

    def _on_seek_pressed(self) -> None:
        self._is_seeking = True

    def _on_seek_released(self) -> None:
        if self._stream_player:
            val = self.seek_slider.value()
            max_val = self.seek_slider.maximum()
            duration = self._stream_player.duration
            if duration > 0:
                target_time = (val / max_val) * duration
                self._stream_player.seek(target_time)
        self._is_seeking = False

    def _on_volume_changed(self, value: int) -> None:
        if self._stream_player:
            self._stream_player.set_volume(value / 100.0)

    def _skip_forward(self) -> None:
        if not self._stream_player:
            return
        current_pos = self._stream_player.position
        # Find next step start
        accum_time = 0.0
        for step in self._session.steps:
            duration = step.duration
            if accum_time > current_pos + 0.1: # Threshold
                self._stream_player.seek(accum_time)
                return
            accum_time += duration

    def _skip_backward(self) -> None:
        if not self._stream_player:
            return
        current_pos = self._stream_player.position
        # Find prev step start
        accum_time = 0.0
        prev_time = 0.0
        for step in self._session.steps:
            duration = step.duration
            if accum_time + duration > current_pos - 0.1:
                # We are in this step.
                # If we are close to start, go to prev step.
                if current_pos - accum_time < 2.0: # 2 seconds threshold
                     self._stream_player.seek(prev_time)
                else:
                     self._stream_player.seek(accum_time)
                return
            prev_time = accum_time
            accum_time += duration
        # If at end, go to start of last step
        self._stream_player.seek(prev_time)

    def _update_playback_ui(self) -> None:
        if not self._stream_player:
            return
        
        duration = self._stream_player.duration
        position = self._stream_player.position
        
        # Update labels
        def fmt_time(s):
            m = int(s // 60)
            sec = int(s % 60)
            return f"{m:02d}:{sec:02d}"
            
        self.time_label.setText(fmt_time(position))
        self.total_time_label.setText(fmt_time(duration))
        
        # Update slider
        if not self._is_seeking and duration > 0:
            self.seek_slider.blockSignals(True)
            val = int((position / duration) * self.seek_slider.maximum())
            self.seek_slider.setValue(val)
            self.seek_slider.blockSignals(False)
            
        # Check if finished (simple heuristic if position >= duration)
        # Or better, check if audio output is idle?
        # For now, just rely on user stopping or loop?
        # SessionStreamPlayer stops automatically at end.
        # We can check if position >= duration - epsilon
        if position >= duration and duration > 0:
             self._stop_playback()

    def _export_session(self) -> None:
        self._invalidate_assembler()
        if self._current_assembler is None:
            self._current_assembler = self._create_assembler()
        path, _ = QFileDialog.getSaveFileName(self, "Export Session", self._session.output_filename, "Audio Files (*.wav *.flac *.mp3)")
        if not path:
            return
        assembler = self._current_assembler
        try:
            success = assembler.render_to_file(path)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Export Error", f"Failed to export session: {exc}")
            return
        if success:
            self.status_label.setText(f"Exported to {Path(path).name}")
        else:
            self.status_label.setText("Export failed")

    def _change_theme(self) -> None:
        action = self.sender()
        if action and isinstance(action, QAction):
            theme_name = action.data()
            app = QApplication.instance()
            if app:
                themes.apply_theme(app, theme_name)
                self.status_label.setText(f"Theme changed to {theme_name}")


__all__ = ["SessionBuilderWindow"]

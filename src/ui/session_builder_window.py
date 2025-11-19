"""Session builder main window with streaming and export controls."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, MutableSequence, Optional

from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
)

from src.audio.session_engine import SessionAssembler
from src.audio.session_model import Session, SessionPresetChoice, SessionStep
from PyQt5.QtWidgets import QApplication

from src.audio.session_stream import SessionStreamPlayer
from src.models.models import StepModel
from . import themes


class SessionStepModel(StepModel):
    """Table model wrapper bridging :class:`SessionStep` objects to the view."""

    headers = ["Duration (s)", "Binaural Preset", "Description"]

    def __init__(self, steps: MutableSequence[SessionStep], preset_lookup: Mapping[str, SessionPresetChoice]):
        super().__init__(list(steps))
        self._preset_lookup = preset_lookup

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
                preset = self._preset_lookup.get(step.binaural_preset_id)
                return preset.label if preset else step.binaural_preset_id
            if index.column() == 2:
                return step.description
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole):  # type: ignore[override]
        if not index.isValid() or role != Qt.EditRole:
            return False
        if index.row() >= len(self.steps):
            return False
        step: SessionStep = self.steps[index.row()]
        if index.column() == 2:
            step.description = str(value)
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index: QModelIndex):  # type: ignore[override]
        base_flags = super().flags(index)
        if index.column() == 2:
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

        self._session = session or Session()
        self._binaural_catalog = dict(binaural_catalog or {})
        self._noise_catalog = dict(noise_catalog or {})

        self._assembler_factory = assembler_factory or (lambda s, b, n, **opts: SessionAssembler(s, b, n, **opts))
        self._stream_player_factory = stream_player_factory or (lambda track_data: SessionStreamPlayer(track_data, self))
        self._stream_player: Optional[SessionStreamPlayer] = None
        self._current_assembler: Optional[SessionAssembler] = None

        self._init_actions()
        self._init_ui()
        self._load_session(self._session)

        app = QApplication.instance()
        if app is not None and theme_name:
            themes.apply_theme(app, theme_name)

    # ------------------------------------------------------------------
    # UI creation helpers
    # ------------------------------------------------------------------
    def _init_actions(self) -> None:
        save_action = QAction("Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_session)
        load_action = QAction("Load Session", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_session_from_file)
        self.addAction(save_action)
        self.addAction(load_action)

    def _init_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        session_group = QGroupBox("Session Settings", central)
        session_group.setToolTip("Configure global session playback and export options.")
        session_layout = QGridLayout(session_group)

        self.crossfade_slider = QSlider(Qt.Horizontal, session_group)
        self.crossfade_slider.setRange(0, 300)
        self.crossfade_slider.setToolTip("Global crossfade duration applied between steps (seconds).")
        self.crossfade_spin = QDoubleSpinBox(session_group)
        self.crossfade_spin.setDecimals(2)
        self.crossfade_spin.setRange(0.0, 30.0)
        self.crossfade_spin.setSuffix(" s")
        self.crossfade_spin.setSingleStep(0.1)
        self.crossfade_spin.setToolTip("Precise crossfade duration in seconds.")
        self.crossfade_curve_combo = QComboBox(session_group)
        self.crossfade_curve_combo.addItems(["linear", "equal_power"])
        self.crossfade_curve_combo.setToolTip("Choose crossfade curve applied between steps.")

        self.normalization_slider = QSlider(Qt.Horizontal, session_group)
        self.normalization_slider.setRange(0, 75)
        self.normalization_slider.setToolTip("Target normalization ceiling for rendered audio (0.00 – 0.75).")
        self.normalization_label = QLabel("0.00", session_group)
        self.normalization_label.setToolTip("Current normalization ceiling applied during rendering.")

        session_layout.addWidget(QLabel("Crossfade Duration:"), 0, 0)
        session_layout.addWidget(self.crossfade_slider, 0, 1)
        session_layout.addWidget(self.crossfade_spin, 0, 2)
        session_layout.addWidget(QLabel("Crossfade Curve:"), 1, 0)
        session_layout.addWidget(self.crossfade_curve_combo, 1, 1, 1, 2)
        session_layout.addWidget(QLabel("Normalization Target:"), 2, 0)
        session_layout.addWidget(self.normalization_slider, 2, 1)
        session_layout.addWidget(self.normalization_label, 2, 2)

        layout.addWidget(session_group)

        step_group = QGroupBox("Session Steps", central)
        step_group.setToolTip("Timeline of session steps. Select a row to edit details below.")
        step_layout = QVBoxLayout(step_group)

        self.step_model = SessionStepModel(self._session.steps, self._binaural_catalog)
        self.step_table = QTableView(step_group)
        self.step_table.setModel(self.step_model)
        self.step_table.setSelectionBehavior(QTableView.SelectRows)
        self.step_table.setSelectionMode(QTableView.SingleSelection)
        self.step_table.horizontalHeader().setStretchLastSection(True)
        self.step_table.setToolTip("List of steps with their duration and presets.")

        step_layout.addWidget(self.step_table)

        step_buttons = QHBoxLayout()
        self.add_step_btn = QPushButton("Add Step", step_group)
        self.add_step_btn.setToolTip("Insert a new step using the selected presets.")
        self.remove_step_btn = QPushButton("Remove Step", step_group)
        self.remove_step_btn.setToolTip("Remove the currently selected step from the session.")
        self.move_up_btn = QPushButton("Move Up", step_group)
        self.move_up_btn.setToolTip("Move the selected step earlier in the timeline.")
        self.move_down_btn = QPushButton("Move Down", step_group)
        self.move_down_btn.setToolTip("Move the selected step later in the timeline.")
        step_buttons.addWidget(self.add_step_btn)
        step_buttons.addWidget(self.remove_step_btn)
        step_buttons.addWidget(self.move_up_btn)
        step_buttons.addWidget(self.move_down_btn)
        step_layout.addLayout(step_buttons)

        layout.addWidget(step_group)

        editor_group = QGroupBox("Step Details", central)
        editor_group.setToolTip("Edit parameters for the selected step.")
        editor_layout = QGridLayout(editor_group)

        self.preset_combo = QComboBox(editor_group)
        self.preset_combo.setToolTip("Select the binaural preset used for this step.")
        self.noise_combo = QComboBox(editor_group)
        self.noise_combo.setToolTip("Optional noise preset blended with the step.")
        self.duration_spin = QDoubleSpinBox(editor_group)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setRange(1.0, 7200.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setToolTip("Duration of the current step in seconds.")
        self.step_crossfade_slider = QSlider(Qt.Horizontal, editor_group)
        self.step_crossfade_slider.setRange(0, 300)
        self.step_crossfade_slider.setToolTip("Crossfade duration for this step (seconds).")
        self.step_crossfade_spin = QDoubleSpinBox(editor_group)
        self.step_crossfade_spin.setDecimals(2)
        self.step_crossfade_spin.setRange(0.0, 30.0)
        self.step_crossfade_spin.setSuffix(" s")
        self.step_crossfade_spin.setSingleStep(0.1)
        self.step_crossfade_spin.setToolTip("Precise crossfade override for this step.")
        self.step_crossfade_curve_combo = QComboBox(editor_group)
        self.step_crossfade_curve_combo.addItems(["Use Session", "linear", "equal_power"])
        self.step_crossfade_curve_combo.setToolTip("Override the crossfade curve for this step.")
        self.warmup_edit = QLineEdit(editor_group)
        self.warmup_edit.setToolTip("Optional warmup audio file path used before the step starts.")
        self.warmup_btn = QPushButton("Browse", editor_group)
        self.warmup_btn.setToolTip("Choose a warmup audio file from disk.")
        self.description_edit = QTextEdit(editor_group)
        self.description_edit.setToolTip("Notes about the intention or feel of the step.")

        editor_layout.addWidget(QLabel("Binaural Preset:"), 0, 0)
        editor_layout.addWidget(self.preset_combo, 0, 1, 1, 2)
        editor_layout.addWidget(QLabel("Noise Preset:"), 1, 0)
        editor_layout.addWidget(self.noise_combo, 1, 1, 1, 2)
        editor_layout.addWidget(QLabel("Duration:"), 2, 0)
        editor_layout.addWidget(self.duration_spin, 2, 1, 1, 2)
        editor_layout.addWidget(QLabel("Step Crossfade:"), 3, 0)
        editor_layout.addWidget(self.step_crossfade_slider, 3, 1)
        editor_layout.addWidget(self.step_crossfade_spin, 3, 2)
        editor_layout.addWidget(QLabel("Step Curve:"), 4, 0)
        editor_layout.addWidget(self.step_crossfade_curve_combo, 4, 1, 1, 2)
        editor_layout.addWidget(QLabel("Warmup Audio:"), 5, 0)
        editor_layout.addWidget(self.warmup_edit, 5, 1)
        editor_layout.addWidget(self.warmup_btn, 5, 2)
        editor_layout.addWidget(QLabel("Description:"), 6, 0)
        editor_layout.addWidget(self.description_edit, 6, 1, 1, 2)

        layout.addWidget(editor_group)

        controls_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Stream", central)
        self.preview_btn.setToolTip("Render the current session and stream audio preview.")
        self.stop_btn = QPushButton("Stop", central)
        self.stop_btn.setToolTip("Stop streaming playback.")
        self.export_btn = QPushButton("Export Session", central)
        self.export_btn.setToolTip("Render the session to an audio file.")
        controls_layout.addWidget(self.preview_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.export_btn)
        layout.addLayout(controls_layout)

        self.status_label = QLabel("Ready", central)
        layout.addWidget(self.status_label)

        self.setCentralWidget(central)

        self._populate_presets()
        self._bind_signals()

    def _populate_presets(self) -> None:
        self.preset_combo.clear()
        for preset_id, preset in sorted(self._binaural_catalog.items()):
            self.preset_combo.addItem(preset.label, preset_id)
        self.noise_combo.clear()
        self.noise_combo.addItem("None", None)
        for preset_id, preset in sorted(self._noise_catalog.items()):
            self.noise_combo.addItem(preset.label, preset_id)

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
        self.noise_combo.currentIndexChanged.connect(self._on_noise_changed)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        self.step_crossfade_slider.valueChanged.connect(self._sync_step_crossfade_spin_from_slider)
        self.step_crossfade_spin.valueChanged.connect(self._sync_step_crossfade_slider_from_spin)
        self.step_crossfade_curve_combo.currentIndexChanged.connect(self._on_step_curve_changed)
        self.warmup_btn.clicked.connect(self._choose_warmup_file)
        self.description_edit.textChanged.connect(self._on_description_changed)

        self.preview_btn.clicked.connect(self._preview_session)
        self.stop_btn.clicked.connect(self._stop_stream)
        self.export_btn.clicked.connect(self._export_session)

    # ------------------------------------------------------------------
    # Session & model synchronization
    # ------------------------------------------------------------------
    def _load_session(self, session: Session) -> None:
        self._session = session
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
        current_norm = getattr(self, "_normalization", 0.25)
        current_norm = max(0.0, min(current_norm, 0.75))
        self.normalization_slider.setValue(int(round(current_norm * 100)))
        self.normalization_slider.blockSignals(False)
        self._update_normalization_label(self.normalization_slider.value())
        self.step_model.refresh(session.steps)
        if session.steps:
            self.step_table.selectRow(0)
        else:
            self._clear_step_editors()

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
        self.duration_spin.blockSignals(True)
        self.step_crossfade_slider.blockSignals(True)
        self.step_crossfade_spin.blockSignals(True)
        self.step_crossfade_curve_combo.blockSignals(True)

        idx = self.preset_combo.findData(step.binaural_preset_id)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        idx = self.noise_combo.findData(step.noise_preset_id)
        self.noise_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.duration_spin.setValue(step.duration)
        crossfade_duration = step.crossfade_duration if step.crossfade_duration is not None else 0.0
        self.step_crossfade_slider.setValue(int(round(crossfade_duration * 10)))
        self.step_crossfade_spin.setValue(crossfade_duration)
        if step.crossfade_curve:
            idx = self.step_crossfade_curve_combo.findText(step.crossfade_curve)
            self.step_crossfade_curve_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.step_crossfade_curve_combo.setCurrentIndex(0)
        self.warmup_edit.setText(step.warmup_clip_path or "")
        self.description_edit.blockSignals(True)
        self.description_edit.setPlainText(step.description)
        self.description_edit.blockSignals(False)

        self.preset_combo.blockSignals(False)
        self.noise_combo.blockSignals(False)
        self.duration_spin.blockSignals(False)
        self.step_crossfade_slider.blockSignals(False)
        self.step_crossfade_spin.blockSignals(False)
        self.step_crossfade_curve_combo.blockSignals(False)

    def _clear_step_editors(self) -> None:
        self.preset_combo.setCurrentIndex(-1)
        self.noise_combo.setCurrentIndex(0 if self.noise_combo.count() else -1)
        self.duration_spin.setValue(1.0)
        self.step_crossfade_slider.setValue(0)
        self.step_crossfade_spin.setValue(0.0)
        self.step_crossfade_curve_combo.setCurrentIndex(0)
        self.warmup_edit.clear()
        self.description_edit.blockSignals(True)
        self.description_edit.clear()
        self.description_edit.blockSignals(False)

    def _sync_crossfade_spin_from_slider(self, value: int) -> None:
        seconds = value / 10.0
        self.crossfade_spin.setValue(seconds)
        self._session.crossfade_duration = seconds

    def _sync_crossfade_slider_from_spin(self, value: float) -> None:
        self.crossfade_slider.setValue(int(round(value * 10)))
        self._session.crossfade_duration = float(value)

    def _on_crossfade_curve_changed(self, text: str) -> None:
        self._session.crossfade_curve = text

    def _on_normalization_changed(self, value: int) -> None:
        value = max(0, min(value, 75))
        self._normalization = value / 100.0
        self._update_normalization_label(value)

    def _update_normalization_label(self, slider_value: int) -> None:
        self.normalization_label.setText(f"{slider_value / 100:.2f}")

    def _on_preset_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None or index < 0:
            return
        preset_id = self.preset_combo.itemData(index)
        if preset_id:
            step.binaural_preset_id = preset_id
            self.step_model.refresh(self._session.steps)

    def _on_noise_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.noise_preset_id = self.noise_combo.itemData(index)

    def _on_duration_changed(self, value: float) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.duration = float(value)
        self.step_model.refresh(self._session.steps)

    def _sync_step_crossfade_spin_from_slider(self, value: int) -> None:
        seconds = value / 10.0
        self.step_crossfade_spin.setValue(seconds)
        self._set_step_crossfade(seconds)

    def _sync_step_crossfade_slider_from_spin(self, value: float) -> None:
        self.step_crossfade_slider.setValue(int(round(value * 10)))
        self._set_step_crossfade(value)

    def _set_step_crossfade(self, value: float) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        if value <= 0.0:
            step.crossfade_duration = None
        else:
            step.crossfade_duration = float(value)

    def _on_step_curve_changed(self, index: int) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        if index <= 0:
            step.crossfade_curve = None
        else:
            step.crossfade_curve = self.step_crossfade_curve_combo.itemText(index)

    def _choose_warmup_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Warmup Audio", "src/presets/audio", "Audio Files (*.wav *.flac *.mp3);;All Files (*)")
        if not path:
            return
        self.warmup_edit.setText(path)
        step = self._get_selected_step()
        if step is not None:
            step.warmup_clip_path = path

    def _on_description_changed(self) -> None:
        step = self._get_selected_step()
        if step is None:
            return
        step.description = self.description_edit.toPlainText()
        self.step_model.refresh(self._session.steps)

    # ------------------------------------------------------------------
    # Step list manipulation
    # ------------------------------------------------------------------
    def _add_step(self) -> None:
        preset_id = self.preset_combo.itemData(self.preset_combo.currentIndex())
        if not preset_id and self.preset_combo.count():
            preset_id = self.preset_combo.itemData(0)
        step = SessionStep(
            binaural_preset_id=str(preset_id or ""),
            duration=self.duration_spin.value(),
            noise_preset_id=self.noise_combo.itemData(self.noise_combo.currentIndex()),
            crossfade_duration=None,
            crossfade_curve=None,
            warmup_clip_path=self.warmup_edit.text() or None,
            description=self.description_edit.toPlainText(),
        )
        self._session.steps.append(step)
        self.step_model.refresh(self._session.steps)
        self.step_table.selectRow(len(self._session.steps) - 1)

    def _remove_step(self) -> None:
        index = self._selected_step_index()
        if index is None:
            return
        if 0 <= index < len(self._session.steps):
            del self._session.steps[index]
            self.step_model.refresh(self._session.steps)
            if self._session.steps:
                self.step_table.selectRow(min(index, len(self._session.steps) - 1))
            else:
                self._clear_step_editors()

    def _move_step(self, direction: int) -> None:
        index = self._selected_step_index()
        if index is None:
            return
        new_index = index + direction
        if new_index < 0 or new_index >= len(self._session.steps):
            return
        steps = self._session.steps
        steps[index], steps[new_index] = steps[new_index], steps[index]
        self.step_model.refresh(steps)
        self.step_table.selectRow(new_index)

    # ------------------------------------------------------------------
    # Save/load handling
    # ------------------------------------------------------------------
    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "session.json", "Session Files (*.json)")
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
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "Session Files (*.json)")
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

    def _preview_session(self) -> None:
        assembler = self._create_assembler()
        track_data = assembler.track_data
        if self._stream_player is not None:
            self._stream_player.stop()
        try:
            player = self._stream_player_factory(track_data)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Preview Error", f"Failed to start preview: {exc}")
            return
        self._stream_player = player
        if hasattr(player, "start"):
            player.start()
        self.status_label.setText("Streaming preview…")

    def _stop_stream(self) -> None:
        if self._stream_player is not None and hasattr(self._stream_player, "stop"):
            self._stream_player.stop()
        self.status_label.setText("Preview stopped")

    def _export_session(self) -> None:
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


__all__ = ["SessionBuilderWindow"]


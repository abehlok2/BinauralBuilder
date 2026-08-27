"""Cue controls, original-against-modified plots, A/B/X, and derived export.

This is the part of the HRTF Lab where a measured asset stops being fixed. The
controls scale the interaural and spectral cues the measurement encodes; the
plots show the measured curve and the modified one together, because a slider
whose effect you cannot see is a slider you cannot use deliberately.

Two things are deliberate about the layout:

* Every control is neutral where it starts, and a Reset returns to exactly
  that. A panel whose defaults already alter the asset would make "unmodified"
  impossible to get back to.
* The comparison is blinded and level-matched. Judging whether a change helped
  is exactly the situation where knowing which one you are listening to
  decides the answer for you, so X is drawn at random and the two candidates
  are matched in level before playback.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.hrtf.modification import (
    BASIC_RANGES,
    CUE_PARAMETERS,
    EXPERT_RANGES,
    NEUTRAL,
    CueTransform,
)
from src.audio.sam_workbench.render.anchor import (
    ANCHOR_PATH_MODES,
    ANCHOR_SOURCE_TYPES,
    AnchorSpec,
)

from .head_view import HeadDirectionSelector
from .hrtf_cue_plot import Curve, HrtfCurvePlot, PlotThrottle

__all__ = ["SamHrtfModifyPanel", "AbxTrial", "DeriveWorker"]

#: Slider steps per unit, so a float cue fits an integer slider.
_SLIDER_STEPS = 100

_CUE_LABELS = {
    "itd_scale": "ITD scale",
    "ild_scale": "ILD scale",
    "pinna_scale": "Pinna scale",
    "distance_scale": "Distance scale",
    "coherence": "Coherence",
    "extra_delay_ms": "Extra delay",
}

_CUE_TOOLTIPS = {
    "itd_scale": "Scales the interaural time difference. 1.0 is the measurement.",
    "ild_scale": "Scales the interaural level difference. 1.0 is the measurement.",
    "pinna_scale": "Scales the high-frequency fine structure that carries "
    "elevation and front/back. 0 removes those cues.",
    "distance_scale": "Scales the common delay and the shared level, leaving "
    "both interaural cues alone.",
    "coherence": "1.0 keeps the ears as measured; lower decorrelates them, "
    "widening the image and blurring its direction.",
    "extra_delay_ms": "An extra differential delay on top of the measured one. "
    "Beyond about 0.7 ms it exceeds a natural head width.",
}

_ORIGINAL_COLOUR = QColor(150, 150, 155)
_MODIFIED_COLOUR = QColor(70, 130, 200)
_RIGHT_COLOUR = QColor(200, 110, 70)


class AbxTrial:
    """One blinded comparison between two renders.

    ``X`` is one of the two, chosen at random and not revealed until the
    listener commits to an answer.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self._x_is_b = bool(self._random.getrandbits(1))
        self.answered = False
        self.correct: bool | None = None
        self.trials = getattr(self, "trials", 0)
        self.hits = getattr(self, "hits", 0)

    @property
    def x_is_b(self) -> bool:
        return self._x_is_b

    def answer(self, guess: str) -> bool:
        """Record a guess of ``"A"`` or ``"B"`` and say whether it was right."""

        if guess not in ("A", "B"):
            raise ValueError(f"an A/B/X answer is 'A' or 'B', got {guess!r}")
        truth = "B" if self._x_is_b else "A"
        self.correct = guess == truth
        self.answered = True
        self.trials += 1
        self.hits += int(self.correct)
        return self.correct

    @property
    def score(self) -> str:
        if self.trials == 0:
            return "no trials yet"
        return f"{self.hits} of {self.trials} correct"


class DeriveWorker(QObject):
    """Applies a cue transform to a whole dataset off the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, dataset: Any, transform: CueTransform) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    @pyqtSlot()
    def run(self) -> None:
        try:
            from src.audio.sam_workbench.hrtf.modification import transform_dataset

            result = transform_dataset(self._dataset, self._transform)
        except Exception as error:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class SamHrtfModifyPanel(QWidget):
    """Cue controls with live plots, a blinded comparison, and export."""

    paramsChanged = pyqtSignal(dict)
    #: ``(audio, sample_rate_hz)`` for the host dialog's audio path.
    auditionRendered = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: Any = None
        self._params: dict[str, Any] = {}
        self._loading = False
        self._modified_cache: Any = None
        self._abx = AbxTrial()
        self._abx_renders: dict[str, tuple[np.ndarray, int]] = {}
        self._derive_thread: QThread | None = None
        self._derive_worker: DeriveWorker | None = None

        # One throttle for the whole panel: a drag redraws the view once.
        self.throttle = PlotThrottle(parent=self)
        self.throttle.triggered.connect(self.refresh_plots)

        self._build_ui()
        self._set_controls_enabled(False)

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        cues = QGroupBox("Cue transform")
        cue_layout = QVBoxLayout(cues)

        header = QHBoxLayout()
        self.expert_check = QCheckBox("Expert ranges")
        self.expert_check.setToolTip(
            "Widen every control past its basic range. These are interaction "
            "limits, not physiological claims."
        )
        self.expert_check.toggled.connect(self._apply_ranges)
        header.addWidget(self.expert_check)
        header.addStretch(1)
        self.reset_button = QPushButton("Reset to measured")
        self.reset_button.setToolTip("Return every cue to its neutral value.")
        self.reset_button.clicked.connect(self.reset_cues)
        header.addWidget(self.reset_button)
        cue_layout.addLayout(header)

        grid = QGridLayout()
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        for row, name in enumerate(CUE_PARAMETERS):
            slider = QSlider(Qt.Horizontal)
            slider.setToolTip(_CUE_TOOLTIPS[name])
            slider.valueChanged.connect(self._on_cue_moved)
            value_label = QLabel()
            value_label.setMinimumWidth(58)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(QLabel(_CUE_LABELS[name]), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
            self._sliders[name] = slider
            self._value_labels[name] = value_label
        cue_layout.addLayout(grid)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b26a00;")
        cue_layout.addWidget(self.warning_label)
        layout.addWidget(cues)

        anchor = QGroupBox("Broadband spatial anchor")
        anchor_form = QFormLayout(anchor)
        self.anchor_check = QCheckBox("Enabled")
        self.anchor_check.setToolTip(
            "A low-frequency carrier cannot express pinna cues. A quiet "
            "broadband companion on the same path can, and pulls the carrier "
            "out with it. Off unless you turn it on."
        )
        self.anchor_check.toggled.connect(self._on_control_changed)
        anchor_form.addRow(self.anchor_check)

        self.anchor_source = QComboBox()
        for source_type in ANCHOR_SOURCE_TYPES:
            self.anchor_source.addItem(source_type.replace("_", " ").capitalize(), source_type)
        self.anchor_source.setToolTip(
            "What the anchor is made of. Its job is to have energy where the "
            "pinna works, which a low carrier does not, so the listener has "
            "something to localise by. Pink and shaped noise are continuous; "
            "sparse grains are intermittent and less intrusive; a harmonic "
            "bank keeps a pitched character; imported uses your own file."
        )
        self.anchor_source.currentIndexChanged.connect(self._on_control_changed)
        anchor_form.addRow("Source", self.anchor_source)

        self.anchor_level = QDoubleSpinBox()
        self.anchor_level.setRange(-80.0, 0.0)
        self.anchor_level.setValue(-30.0)
        self.anchor_level.setSuffix(" dB")
        self.anchor_level.setToolTip("Level relative to the primary source.")
        self.anchor_level.valueChanged.connect(self._on_control_changed)
        anchor_form.addRow("Level", self.anchor_level)

        self.anchor_path = QComboBox()
        for mode in ANCHOR_PATH_MODES:
            self.anchor_path.addItem(mode.capitalize(), mode)
        self.anchor_path.setToolTip(
            "Where the anchor sits relative to the source it supports. "
            "'Same' puts it on the same path, which is what pulls the "
            "carrier outward; 'offset' and 'mirrored' place it elsewhere, "
            "which weakens that pull but keeps the two from masking."
        )
        self.anchor_path.currentIndexChanged.connect(self._on_control_changed)
        anchor_form.addRow("Path", self.anchor_path)
        layout.addWidget(anchor)

        plots = QGroupBox("Measured against modified")
        plot_layout = QVBoxLayout(plots)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("View:"))
        self.plot_selector = QComboBox()
        self.plot_selector.setToolTip(
            "Which view of the measured and modified responses to compare. "
            "Impulse response shows them in time; magnitude shows the "
            "spectral notches that carry elevation; group delay and phase "
            "show timing, where interaural differences live."
        )
        for label, key in (
            ("Impulse response", "impulse"),
            ("Magnitude", "magnitude"),
            ("Group delay", "group_delay"),
            ("Phase", "phase"),
            ("ITD across azimuth", "itd"),
            ("ILD across azimuth", "ild"),
        ):
            self.plot_selector.addItem(label, key)
        self.plot_selector.currentIndexChanged.connect(self.refresh_plots)
        selector.addWidget(self.plot_selector, 1)
        plot_layout.addLayout(selector)

        # A direction is a place around the head, so it is chosen on a picture
        # of a head rather than typed. The faint dots are the directions this
        # dataset actually measured; the plots snap to the nearest of them.
        plot_layout.addWidget(QLabel("Direction (single-direction views):"))
        self.direction_selector = HeadDirectionSelector()
        self.direction_selector.setMaximumHeight(190)
        self.direction_selector.directionChanged.connect(self._on_direction_changed)
        plot_layout.addWidget(self.direction_selector)

        self.direction_label = QLabel("")
        self.direction_label.setWordWrap(True)
        plot_layout.addWidget(self.direction_label)

        self.plot = HrtfCurvePlot()
        # The head view sits above the plot now, so the plot needs a floor of
        # its own or the two share the space until neither is readable.
        self.plot.setMinimumHeight(200)
        plot_layout.addWidget(self.plot, 1)
        layout.addWidget(plots, 1)

        comparison = QGroupBox("Blinded A/B/X")
        comparison_layout = QVBoxLayout(comparison)
        self.abx_hint = QLabel(
            "A is the measured asset, B is your modified one, and X is one of "
            "them chosen at random. Both are matched in level first."
        )
        self.abx_hint.setWordWrap(True)
        comparison_layout.addWidget(self.abx_hint)

        buttons = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare trial")
        self.prepare_button.clicked.connect(self.prepare_abx)
        buttons.addWidget(self.prepare_button)
        for label in ("A", "B", "X"):
            button = QPushButton(f"Play {label}")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, which=label: self.play(which))
            buttons.addWidget(button)
            setattr(self, f"play_{label.lower()}_button", button)
        buttons.addStretch(1)
        comparison_layout.addLayout(buttons)

        answers = QHBoxLayout()
        answers.addWidget(QLabel("X was:"))
        self.answer_a_button = QPushButton("A")
        self.answer_b_button = QPushButton("B")
        for button, guess in ((self.answer_a_button, "A"), (self.answer_b_button, "B")):
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, value=guess: self.answer(value))
            answers.addWidget(button)
        self.abx_result = QLabel("no trials yet")
        answers.addWidget(self.abx_result, 1)
        comparison_layout.addLayout(answers)
        layout.addWidget(comparison)

        actions = QHBoxLayout()
        self.derive_button = QPushButton("Derive and save SOFA…")
        self.derive_button.setToolTip(
            "Write the modified dataset with its provenance. The file is only "
            "reported as saved once it passes verification."
        )
        self.derive_button.clicked.connect(self.derive_and_save)
        actions.addWidget(self.derive_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._apply_ranges()
        self.reset_cues()

    # --- ranges and values --------------------------------------------------

    def _ranges(self) -> dict[str, tuple[float, float]]:
        return EXPERT_RANGES if self.expert_check.isChecked() else BASIC_RANGES

    def _apply_ranges(self) -> None:
        """Re-scale every slider, keeping the value it already holds."""

        previous = self.cue_transform() if self._sliders else None
        self._loading = True
        try:
            for name, slider in self._sliders.items():
                low, high = self._ranges()[name]
                slider.setRange(int(low * _SLIDER_STEPS), int(high * _SLIDER_STEPS))
                if previous is not None:
                    value = float(np.clip(getattr(previous, name), low, high))
                    slider.setValue(int(round(value * _SLIDER_STEPS)))
        finally:
            self._loading = False
        self._refresh_value_labels()

    def _refresh_value_labels(self) -> None:
        for name, slider in self._sliders.items():
            value = slider.value() / _SLIDER_STEPS
            suffix = " ms" if name == "extra_delay_ms" else ""
            label = self._value_labels[name]
            label.setText(f"{value:.2f}{suffix}")
            neutral = abs(value - NEUTRAL[name]) < 1e-9
            label.setStyleSheet("" if neutral else "font-weight: bold;")

    def cue_transform(self) -> CueTransform:
        """The transform the controls currently describe."""

        values = {
            name: slider.value() / _SLIDER_STEPS for name, slider in self._sliders.items()
        }
        return CueTransform(**values)

    def anchor_spec(self) -> AnchorSpec:
        return AnchorSpec(
            enabled=self.anchor_check.isChecked(),
            source_type=str(self.anchor_source.currentData()),
            level_db=float(self.anchor_level.value()),
            path_mode=str(self.anchor_path.currentData()),
        )

    def reset_cues(self) -> None:
        """Back to the measured asset, exactly."""

        self._loading = True
        try:
            for name, slider in self._sliders.items():
                slider.setValue(int(round(NEUTRAL[name] * _SLIDER_STEPS)))
        finally:
            self._loading = False
        self._modified_cache = None
        self._refresh_value_labels()
        self._on_control_changed()

    @property
    def is_neutral(self) -> bool:
        return self.cue_transform().is_neutral

    # --- dataset ------------------------------------------------------------

    def set_dataset(self, dataset: Any) -> None:
        """Attach the loaded asset the controls act on."""

        self._dataset = dataset
        self._modified_cache = None
        self._abx_renders.clear()
        self._set_controls_enabled(dataset is not None)
        if dataset is None:
            self.direction_selector.set_marks(())
            self.plot.clear("Select an asset on the Asset tab.")
            return
        self._seed_direction()
        self.refresh_plots()

    def _seed_direction(self) -> None:
        """Show where this dataset measured, and open on a useful direction."""

        from src.audio.sam_workbench.hrtf.coordinates import cartesian_to_sofa_spherical

        spherical = cartesian_to_sofa_spherical(self._dataset.positions_m)
        self.direction_selector.set_marks(
            [(float(row[0]), float(row[1])) for row in spherical]
        )
        opening = spherical[self._default_plot_index()]
        self.direction_selector.set_direction(float(opening[0]), float(opening[1]))
        self._describe_direction()

    def _on_direction_changed(self, _azimuth: float, _elevation: float) -> None:
        if self._dataset is None:
            return
        self._describe_direction()
        self.refresh_plots()

    def _describe_direction(self) -> None:
        from src.audio.sam_workbench.hrtf.coordinates import cartesian_to_sofa_spherical

        spherical = cartesian_to_sofa_spherical(self._dataset.positions_m)
        row = spherical[self._plot_index()]
        self.direction_label.setText(
            f"Nearest measured direction: azimuth {float(row[0]):+.1f}°, "
            f"elevation {float(row[1]):+.1f}°"
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            *self._sliders.values(),
            self.reset_button,
            self.derive_button,
            self.prepare_button,
            self.expert_check,
        ):
            widget.setEnabled(enabled)

    @property
    def dataset(self) -> Any:
        return self._dataset

    def modified_hrirs(self) -> np.ndarray | None:
        """The modified responses, computed on demand and cached."""

        if self._dataset is None:
            return None
        transform = self.cue_transform()
        if transform.is_neutral:
            return np.asarray(self._dataset.ir, dtype=np.float64)
        if self._modified_cache is not None and self._modified_cache[0] == transform:
            return self._modified_cache[1]

        from src.audio.sam_workbench.hrtf.modification import transform_dataset

        result = transform_dataset(self._dataset, transform)
        self._modified_cache = (transform, result.hrirs)
        return result.hrirs

    # --- plots --------------------------------------------------------------

    def _default_plot_index(self) -> int:
        """The direction to open on: most lateral on the horizon.

        The interaural cues are largest there, so a change to them is visible
        rather than subtle - a better opening view than straight ahead, where
        ITD and ILD are both near zero and nothing appears to happen.
        """

        from src.audio.sam_workbench.hrtf.coordinates import cartesian_to_sofa_spherical

        spherical = cartesian_to_sofa_spherical(self._dataset.positions_m)
        horizon = np.abs(spherical[:, 1]) < 10.0
        candidates = np.flatnonzero(horizon) if np.any(horizon) else np.arange(len(spherical))
        lateral = np.abs(np.sin(np.radians(spherical[candidates, 0])))
        return int(candidates[int(np.argmax(lateral))])

    def _plot_index(self) -> int:
        """The measured direction nearest the one chosen on the head view.

        Snapping rather than interpolating is deliberate: these plots show what
        the file contains, and an interpolated curve would show what the
        renderer would make of it instead.
        """

        from src.audio.sam_workbench.hrtf.coordinates import (
            sofa_positions_to_cartesian,
            unit_vectors,
        )

        wanted = unit_vectors(
            sofa_positions_to_cartesian(
                [[self.direction_selector.azimuth_deg, self.direction_selector.elevation_deg, 1.0]],
                "spherical",
                "degree, degree, metre",
            )
        )[0]
        directions = unit_vectors(self._dataset.positions_m)
        # The nearest direction on the sphere is the one whose unit vector has
        # the largest projection onto the wanted one.
        return int(np.argmax(directions @ wanted))

    def refresh_plots(self) -> None:
        """Redraw the selected view with both curves on it."""

        if self._dataset is None:
            self.plot.clear("Select an asset on the Asset tab.")
            return

        from src.audio.sam_workbench.analysis import hrtf_curves as curves

        rate = float(self._dataset.sample_rate_hz)
        modified = self.modified_hrirs()
        if modified is None:
            return
        view = self.plot_selector.currentData()
        index = self._plot_index()
        original_pair = np.asarray(self._dataset.ir[index], dtype=np.float64)
        modified_pair = np.asarray(modified[index], dtype=np.float64)

        if view == "impulse":
            times, left, _ = curves.impulse_response_curves(original_pair, rate)
            _, modified_left, _ = curves.impulse_response_curves(modified_pair, rate)
            self.plot.set_curves(
                times,
                [
                    Curve("measured", left, _ORIGINAL_COLOUR, dashed=True),
                    Curve("modified", modified_left, _MODIFIED_COLOUR),
                ],
                x_label="milliseconds",
                y_label="left ear impulse response",
            )
        elif view in ("magnitude", "phase", "group_delay"):
            function = {
                "magnitude": curves.magnitude_db,
                "phase": curves.unwrapped_phase_deg,
                "group_delay": curves.group_delay_samples,
            }[view]
            frequencies, left, _ = function(original_pair, rate)
            _, modified_left, _ = function(modified_pair, rate)
            label = {
                "magnitude": "left ear magnitude, dB",
                "phase": "left ear phase, degrees",
                "group_delay": "left ear group delay, samples",
            }[view]
            self.plot.set_curves(
                frequencies,
                [
                    Curve("measured", left, _ORIGINAL_COLOUR, dashed=True),
                    Curve("modified", modified_left, _MODIFIED_COLOUR),
                ],
                x_label="hertz",
                y_label=label,
                log_x=True,
            )
        else:
            azimuths, original_itd, original_ild = curves.sweep_across_azimuth(
                self._dataset.ir, self._dataset.positions_m, rate
            )
            _, modified_itd, modified_ild = curves.sweep_across_azimuth(
                modified, self._dataset.positions_m, rate
            )
            if view == "itd":
                series, modified_series, label = (
                    original_itd,
                    modified_itd,
                    "ITD, microseconds (positive = left leads)",
                )
            else:
                series, modified_series, label = (
                    original_ild,
                    modified_ild,
                    "ILD, dB (positive = left louder)",
                )
            self.plot.set_curves(
                azimuths,
                [
                    Curve("measured", series, _ORIGINAL_COLOUR, dashed=True),
                    Curve("modified", modified_series, _MODIFIED_COLOUR),
                ],
                x_label="azimuth, degrees",
                y_label=label,
            )

    # --- A/B/X --------------------------------------------------------------

    def prepare_abx(self) -> None:
        """Render both candidates, level-matched, and pick X at random."""

        if self._dataset is None:
            self.status_label.setText("Select an asset before comparing.")
            return
        if self.is_neutral and not self.anchor_check.isChecked():
            self.status_label.setText(
                "Nothing to compare yet: move a cue control or enable the anchor."
            )
            return

        from src.audio.sam_workbench.hrtf.subject_test import make_test_signal
        from src.audio.sam_workbench.render.hybrid import HybridSpec, render_hybrid

        rate = int(round(float(self._dataset.sample_rate_hz)))
        try:
            mono = make_test_signal("pink_noise_burst", duration_s=1.5, sample_rate_hz=rate)
            direction = np.asarray(self._dataset.positions_m[self._plot_index()], dtype=np.float64)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)

            modified_spec = HybridSpec(cue=self.cue_transform(), anchor=self.anchor_spec())
            candidate_a = render_hybrid(
                mono, direction, self._dataset, rate, modified_spec.as_physical()
            ).audio
            candidate_b = render_hybrid(mono, direction, self._dataset, rate, modified_spec).audio
        except Exception as error:
            self.status_label.setText(f"Could not prepare the comparison: {error}")
            return

        candidate_a, candidate_b = _match_levels(candidate_a, candidate_b)
        self._abx.reset()
        self._abx_renders = {
            "A": (candidate_a, rate),
            "B": (candidate_b, rate),
            "X": ((candidate_b if self._abx.x_is_b else candidate_a), rate),
        }
        for label in ("a", "b", "x"):
            getattr(self, f"play_{label}_button").setEnabled(True)
        self.answer_a_button.setEnabled(True)
        self.answer_b_button.setEnabled(True)
        self.status_label.setText(
            "Trial ready. A is measured, B is modified, X is one of them."
        )

    def play(self, which: str) -> None:
        """Emit one candidate for the host dialog to play."""

        render = self._abx_renders.get(which)
        if render is None:
            self.status_label.setText("Prepare a trial first.")
            return
        self.auditionRendered.emit(render)

    def answer(self, guess: str) -> None:
        correct = self._abx.answer(guess)
        truth = "B" if self._abx.x_is_b else "A"
        self.abx_result.setText(
            f"X was {truth} - {'correct' if correct else 'incorrect'} ({self._abx.score})"
        )
        self.answer_a_button.setEnabled(False)
        self.answer_b_button.setEnabled(False)

    @property
    def abx(self) -> AbxTrial:
        return self._abx

    # --- derive -------------------------------------------------------------

    def derive_and_save(self) -> None:
        """Write the modified dataset, and only claim success if it verifies."""

        if self._dataset is None:
            self.status_label.setText("Select an asset before deriving one.")
            return
        if self.is_neutral:
            self.status_label.setText(
                "The cue controls are neutral, so a derived file would only "
                "duplicate the measurement."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save derived SOFA", "", "SOFA files (*.sofa);;All files (*)"
        )
        if not path:
            return

        from src.audio.sam_workbench.hrtf.derived import write_derived_sofa
        from src.audio.sam_workbench.hrtf.modification import transform_dataset

        try:
            modified = transform_dataset(self._dataset, self.cue_transform())
            result = write_derived_sofa(
                modified,
                path,
                source=self._dataset,
                delay_policy=str(getattr(getattr(self._dataset, "delay_policy", None), "value", "")),
            )
        except Exception as error:
            self.status_label.setText(f"Could not write the derived file: {error}")
            return

        if result.verified:
            self.status_label.setText(f"Saved and verified: {result.path}")
        else:
            # Written but not trustworthy: say so rather than reporting success.
            self.status_label.setText(
                f"Written to {result.path}, but verification did not pass: {result.message}"
            )

    # --- parameter contract -------------------------------------------------

    def set_params(self, params) -> None:
        self._loading = True
        try:
            self._params = copy.deepcopy(dict(params))
            options = dict(self._params.get("hrtfOptions") or {})
            cue = CueTransform.from_mapping(options.get("cue") or {})
            anchor = AnchorSpec.from_mapping(options.get("anchor") or {})

            if any(
                not (self._ranges()[name][0] <= getattr(cue, name) <= self._ranges()[name][1])
                for name in CUE_PARAMETERS
            ):
                # A stored project may sit outside the basic range; widen rather
                # than silently clamping what the user saved.
                self.expert_check.setChecked(True)
                self._apply_ranges()

            for name, slider in self._sliders.items():
                low, high = self._ranges()[name]
                value = float(np.clip(getattr(cue, name), low, high))
                slider.setValue(int(round(value * _SLIDER_STEPS)))

            self.anchor_check.setChecked(anchor.enabled)
            index = self.anchor_source.findData(anchor.source_type)
            if index >= 0:
                self.anchor_source.setCurrentIndex(index)
            self.anchor_level.setValue(anchor.level_db)
            index = self.anchor_path.findData(anchor.path_mode)
            if index >= 0:
                self.anchor_path.setCurrentIndex(index)
        finally:
            self._loading = False
        self._refresh_value_labels()
        self._modified_cache = None
        self.throttle.request()

    def params(self) -> dict[str, Any]:
        options = dict(self._params.get("hrtfOptions") or {})
        options["cue"] = self.cue_transform().describe()
        options["anchor"] = self.anchor_spec().describe()
        return {"hrtfOptions": options}

    # --- events -------------------------------------------------------------

    def _on_cue_moved(self, _value: int) -> None:
        self._refresh_value_labels()
        self._on_control_changed()

    def _on_control_changed(self, *_args) -> None:
        if self._loading:
            return
        self._modified_cache = None
        # A changed transform invalidates any trial already rendered.
        self._abx_renders.clear()
        for label in ("a", "b", "x"):
            getattr(self, f"play_{label}_button").setEnabled(False)

        transform = self.cue_transform()
        warnings = transform.warnings()
        self.warning_label.setText(
            "\n".join(f"{w.parameter}: {w.message}" for w in warnings) if warnings else ""
        )
        self.throttle.request()
        self.paramsChanged.emit(self.params())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._derive_thread is not None:
            self._derive_thread.quit()
            self._derive_thread.wait(2000)
        super().closeEvent(event)


def _match_levels(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bring two renders to the same root-mean-square level.

    Without this a comparison measures whichever one came out louder, which is
    the one thing a blinded test is supposed to rule out.
    """

    def rms(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0

    first_rms, second_rms = rms(first), rms(second)
    if first_rms <= 0.0 or second_rms <= 0.0:
        return first, second
    target = 0.5 * (first_rms + second_rms)
    return first * (target / first_rms), second * (target / second_rms)

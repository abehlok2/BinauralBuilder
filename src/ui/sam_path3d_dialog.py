"""The three-dimensional path designer.

Two ideas shape this dialog.

The first is that *depth is ambiguous in a single view*, so there are four:
a perspective view to see the shape, and top, front and side views in which
every axis is directly editable.  Selecting a point in any of them selects it
in all of them, and the numeric editor beside them is a fifth view of the same
point - in metres and in azimuth/elevation/distance at once, because the two
ways of thinking about a position are both worth having and neither is the
stored format on its own.

The second is that *path geometry and path traversal are different things*.
Where the source can be, and how it moves along that shape over time, are
edited in separate panels and stored separately.  A circle stays one circle
whether it is walked at constant speed, eased, reversed or driven from the
stage timeline; keeping the two apart is what lets either grow without
disturbing the other.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.audio.sam_workbench.render.hybrid import SIGNAL_CHAIN_TEXT
from src.audio.sam_workbench.scene_state import MODULATOR_WAVEFORMS
from src.audio.sam_workbench.stages import CURVES
from src.audio.sam_workbench.trajectory import (
    PRIMITIVE_TYPES,
    Keyframe,
    KeyframedPath,
    PathModel,
    cartesian_array_to_spherical,
    geometry_to_dict,
    keyframes_from_csv,
    keyframes_from_json,
    path_model_from_dict,
    spherical_to_cartesian,
)
from src.audio.sam_workbench.trajectory.parameter_catalog import (
    GEOMETRY_PARAMETER_SPECS,
    PATH_PREFIX,
    TRANSFORM_PREFIX,
    TRANSFORM_PARAMETER_SPECS,
    primitive_component_fields,
    route_leaf,
    split_parameter_path,
)
from src.audio.sam_workbench.modulation import ModulationMatrix, ModulationRoute
from src.audio.sam_workbench.path_automation import (
    is_reserved_path,
    compile_bound_trajectory,
)
from src.audio.sam_workbench.trajectory.serialization import _SPATIAL

#: Curves that *shape* a modulator. ``hold`` is a stage edge: applied to a
#: route it maps every value to zero, so offering it here would let a row
#: look armed while silently contributing nothing.
_MOTION_CURVES: tuple[str, ...] = tuple(curve for curve in CURVES if curve != "hold")

from .sam_path3d_views import PLANES, OrthographicPathView, PerspectivePathView

#: Geometry kinds edited as draggable control points rather than as parameters.
_POINT_KINDS = ("polyline", "polygon", "spline", "bezier", "line")

#: How many samples the drawn curve uses. Enough that a dome with several turns
#: reads as a curve rather than as a polygon, cheap enough to redraw on drag.
_CURVE_SAMPLES = 400

#: Help for the fields shared by the parametric geometry dataclasses.  The
#: form itself is generated from those dataclasses, so this stays keyed by the
#: serialized field name rather than by primitive.  A newly introduced field
#: still receives a useful fallback in :func:`_parameter_tooltip`.
_PARAMETER_TOOLTIPS = {
    "radius_m": "Radius of the orbit in metres, measured from its centre.",
    "major_radius_m": "Distance in metres from the torus centre to the centre of its tube.",
    "minor_radius_m": "Radius in metres of the smaller circle that winds around the torus.",
    "length_m": "Distance in metres from the pendulum pivot to the moving source.",
    "distance_m": "Constant distance in metres from the listener to the source path.",
    "start_distance_m": "Source distance from the listener at the start of the path, in metres.",
    "end_distance_m": "Source distance from the listener at the end or widest part of the path, in metres.",
    "minimum_distance_m": "Minimum allowed distance in metres from the walk's centre; prevents the path crossing the listener.",
    "azimuth_deg": "Fixed azimuth in degrees: 0° is front and positive angles move toward the left.",
    "start_azimuth_deg": "Starting azimuth in degrees: 0° is front and positive angles move toward the left.",
    "end_azimuth_deg": "Ending azimuth in degrees: 0° is front and positive angles move toward the left.",
    "centre_azimuth_deg": "Azimuth of the figure's centre in degrees: 0° is front and positive is left.",
    "plane_azimuth_deg": "Horizontal direction of the orbit or swing plane: 0° is front and 90° is left.",
    "tilt_axis_azimuth_deg": "Horizontal direction toward which the orbit is tilted: 0° is front and 90° is left.",
    "elevation_deg": "Elevation in degrees: 0° is ear height and positive angles move upward.",
    "start_elevation_deg": "Elevation at the start of the path: 0° is ear height and positive is upward.",
    "end_elevation_deg": "Elevation at the end of the path: 0° is ear height and positive is upward.",
    "centre_elevation_deg": "Elevation of the figure's centre: 0° is ear height and positive is upward.",
    "azimuth_extent_deg": "Maximum horizontal angular extent of the figure-eight from its centre.",
    "elevation_extent_deg": "Maximum vertical angular extent of the figure-eight from its centre.",
    "start_angle_deg": "Angle around the orbit at which the path begins, in degrees.",
    "tilt_deg": "Rotation in degrees that tilts the path out of its unrotated plane.",
    "swing_deg": "Maximum pendulum angle in degrees on either side of its resting position.",
    "turns": "Number of complete revolutions made while traversing the path once.",
    "major_turns": "Number of complete orbits around the torus centre during one traversal.",
    "minor_turns": "Number of windings through the torus tube during one traversal.",
    "cycles": "Number of distance expansion-and-contraction cycles during one traversal.",
    "swings": "Number of complete back-and-forth pendulum swings during one traversal.",
    "centre_m": "Path centre as X / Y / Z metres: +x forward, +y left, +z up.",
    "pivot_m": "Pendulum pivot as X / Y / Z metres: +x forward, +y left, +z up.",
    "extent_m": "Full X / Y / Z dimensions of the random-walk volume in metres.",
    "steps": "Number of deterministic waypoints used to construct the random walk.",
    "seed": "Non-negative random seed; the same seed produces the same path.",
    "smooth": "Use a smooth periodic spline between random-walk waypoints instead of straight segments.",
    "pass_over": "Continue over the zenith and descend behind the listener instead of stopping overhead.",
}


def _parameter_tooltip(field_name):
    """Return user-facing help for a generated primitive parameter."""

    return _PARAMETER_TOOLTIPS.get(
        field_name,
        f"Controls the {field_name.replace('_', ' ')} value for this path primitive.",
    )


class _AxisRow(QWidget):
    """Three linked spin boxes, for a vector-valued parameter."""

    changed = pyqtSignal()

    def __init__(self, suffix="", minimum=-1e4, maximum=1e4, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        for _ in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(suffix)
            spin.valueChanged.connect(self.changed)
            layout.addWidget(spin)
            self.spins.append(spin)

    def value(self):
        return [spin.value() for spin in self.spins]

    def set_value(self, values):
        for spin, value in zip(self.spins, values):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)


class SamPath3DDialog(QDialog):
    """Multi-view editor for a canonical three-dimensional trajectory."""

    #: Measured positions of the dataset this path will be rendered against,
    #: when one has been selected. Without it the coverage shell is drawn at a
    #: radius guessed from the path itself, which tells the user nothing about
    #: the dataset - a path can sit exactly on a guessed shell and still be
    #: nowhere near a measurement.
    _dataset_positions = None
    _dataset_label = ""

    def __init__(self, trajectory_spec=None, parent=None, modulation=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 3D Path Designer")
        self.resize(1180, 820)
        self._spec = copy.deepcopy(trajectory_spec or {})
        # Scene context, when the host can provide one: a stable source
        # identifier, a callable returning the current scene, and a callable
        # that persists an edited scene. Without it the Motion group is built
        # but disabled, so its absence is stated rather than silent.
        self._motion = dict(modulation or {})
        self._motion_disclosure = str(self._motion.get("disclosure", "advanced"))
        self._updating = False
        self._selected = -1
        self._points: list[list[float]] = []
        self._keyframes: list[Keyframe] = []
        self._parameters: dict[str, object] = {}
        self._parameter_widgets: dict[str, object] = {}
        self._motion_rows: dict[str, dict[str, object]] = {}
        self._preview_time = 0.0

        self._build_ui()
        self._load_spec()
        self.set_disclosure(self._motion_disclosure)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance_preview)

    # ------------------------------------------------------------------ setup

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_header())
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_views())
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        chain = QLabel(f"Signal chain:  {SIGNAL_CHAIN_TEXT}")
        chain.setToolTip(
            "Phase manipulation before binaural filtering and after it produce "
            "meaningfully different results, so the order is stated explicitly."
        )
        chain.setStyleSheet("color: #9aa0aa;")
        layout.addWidget(chain)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_header(self):
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Geometry:"))
        self.primitive_combo = QComboBox()
        self.primitive_combo.addItem("— points —")
        for kind in _POINT_KINDS:
            self.primitive_combo.addItem(kind)
        self.primitive_combo.insertSeparator(self.primitive_combo.count())
        self.primitive_combo.addItem("keyframes")
        self.primitive_combo.insertSeparator(self.primitive_combo.count())
        for kind in PRIMITIVE_TYPES:
            self.primitive_combo.addItem(kind)
        self.primitive_combo.setToolTip(
            "Point kinds are edited by dragging. Three-dimensional primitives "
            "are edited by their own numbers, so a dome stays a dome when the "
            "project is reopened instead of becoming a spline through samples "
            "of one."
        )
        self.primitive_combo.currentTextChanged.connect(self._primitive_changed)
        row.addWidget(self.primitive_combo)

        self.shell_check = QCheckBox("HRTF coverage shell")
        self.shell_check.setToolTip(
            "Draw the sphere an HRTF dataset measures on. A path that leaves it "
            "is being extrapolated rather than reproduced."
        )
        self.shell_check.toggled.connect(self._shell_toggled)
        row.addWidget(self.shell_check)

        self.preview_button = QPushButton("Preview motion")
        self.preview_button.setCheckable(True)
        self.preview_button.setToolTip(
            "Animate the marker along the compiled trajectory - the positions "
            "the renderer is actually sent, including easing, direction and "
            "the constant-speed law, not the authored curve."
        )
        self.preview_button.toggled.connect(self._preview_toggled)
        row.addWidget(self.preview_button)
        row.addStretch(1)
        return header

    def _build_views(self):
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        self.perspective = PerspectivePathView()
        self.views = {"perspective": self.perspective}
        grid.addWidget(self.perspective, 0, 0)
        for index, plane in enumerate(("top", "front", "side")):
            view = OrthographicPathView(plane)
            view.pointMoved.connect(self._point_dragged)
            self.views[plane] = view
            grid.addWidget(view, (index + 1) // 2, (index + 1) % 2)
        for view in self.views.values():
            view.selectionChanged.connect(self._select_point)
        return container

    def _build_side_panel(self):
        tabs = QTabWidget()
        tabs.addTab(self._scrolled(self._build_position_tab()), "Position")
        tabs.addTab(self._scrolled(self._build_geometry_tab()), "Geometry")
        tabs.addTab(self._scrolled(self._build_traversal_tab()), "Traversal")
        return tabs

    @staticmethod
    def _scrolled(widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # --- position tab -------------------------------------------------------

    def _build_position_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        numeric = QGroupBox("Selected point")
        form = QFormLayout(numeric)
        self.cartesian_row = _AxisRow(" m")
        self.cartesian_row.changed.connect(self._cartesian_edited)
        form.addRow("X / Y / Z", self.cartesian_row)

        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(-180.0, 180.0)
        self.azimuth_spin.setSuffix(" °")
        self.elevation_spin = QDoubleSpinBox()
        self.elevation_spin.setRange(-90.0, 90.0)
        self.elevation_spin.setSuffix(" °")
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.0, 1000.0)
        self.distance_spin.setDecimals(3)
        self.distance_spin.setSuffix(" m")
        for spin in (self.azimuth_spin, self.elevation_spin, self.distance_spin):
            spin.valueChanged.connect(self._spherical_edited)
        form.addRow("Azimuth", self.azimuth_spin)
        form.addRow("Elevation", self.elevation_spin)
        form.addRow("Distance", self.distance_spin)

        self.keyframe_time_spin = QDoubleSpinBox()
        self.keyframe_time_spin.setRange(0.0, 36000.0)
        self.keyframe_time_spin.setDecimals(3)
        self.keyframe_time_spin.setSuffix(" s")
        self.keyframe_time_spin.valueChanged.connect(self._keyframe_time_edited)
        form.addRow("Keyframe time", self.keyframe_time_spin)
        layout.addWidget(numeric)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time (s)", "X", "Y", "Z"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._table_selection_changed)
        self.table.itemChanged.connect(self._table_edited)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        for text, slot, tip in (
            ("Add", self._add_point, "Append a point after the selection"),
            ("Remove", self._remove_point, "Delete the selected point"),
            ("Import…", self._import_points, "Read points from CSV or JSON"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        helpers = QGroupBox("Whole-path helpers")
        helper_layout = QVBoxLayout(helpers)
        for text, slot, tip in (
            ("Snap to horizontal plane", self._snap_horizontal,
             "Set every height to zero, flattening the path into the ear-height plane"),
            ("Snap to listener height", self._snap_listener_height,
             "Move the path vertically so its mean height is the listener's"),
            ("Normalize distance", self._normalize_distance,
             "Scale the path so its mean distance is one metre"),
            ("Maintain constant distance", self._constant_distance,
             "Push every point onto one sphere, keeping its direction. "
             "This is the surface an HRTF dataset measures."),
            ("Reverse direction", self._reverse,
             "Traverse the same geometry the other way round"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            helper_layout.addWidget(button)
        layout.addWidget(helpers)
        return page

    # --- geometry tab -------------------------------------------------------

    def _build_geometry_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.parameter_box = QGroupBox("Primitive parameters")
        self.parameter_form = QFormLayout(self.parameter_box)
        layout.addWidget(self.parameter_box)

        layout.addWidget(self._build_motion_group())

        transform = QGroupBox("Path transform")
        form = QFormLayout(transform)
        self.offset_row = _AxisRow(" m")
        self.offset_row.changed.connect(self._refresh)
        form.addRow("Centre offset", self.offset_row)
        self.scale_row = _AxisRow("")
        self.scale_row.set_value((1.0, 1.0, 1.0))
        self.scale_row.changed.connect(self._refresh)
        for spin in self.scale_row.spins:
            spin.setRange(-100.0, 100.0)
        form.addRow("Scale per axis", self.scale_row)
        self.rotation_row = _AxisRow(" °", -360.0, 360.0)
        self.rotation_row.changed.connect(self._refresh)
        form.addRow("Yaw / pitch / roll", self.rotation_row)
        transform.setToolTip(
            "Applied to the whole path after its own geometry: tilt an orbit, "
            "stretch it on one axis, or move its centre away from the listener."
        )
        layout.addWidget(transform)

        frame = QGroupBox("Coordinate model")
        frame_form = QFormLayout(frame)
        self.frame_combo = QComboBox()
        self.frame_combo.addItem("Listener-relative", "listener_relative_cartesian")
        self.frame_combo.addItem("World", "world_cartesian")
        self.frame_combo.setToolTip(
            "Listener-relative coordinates follow the head; world coordinates "
            "stay fixed in the room and are resolved against the listener pose."
        )
        frame_form.addRow("Coordinates", self.frame_combo)
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["hold", "linear", "cubic", "catmull_rom"])
        self.interpolation_combo.setCurrentText("cubic")
        self.interpolation_combo.currentTextChanged.connect(self._refresh)
        frame_form.addRow("Interpolation", self.interpolation_combo)
        self.closed_check = QCheckBox("Closed path")
        self.closed_check.toggled.connect(self._refresh)
        frame_form.addRow(self.closed_check)
        frame_form.addRow(
            QLabel(
                "Right-handed metres: +x forward, +y left, +z up.\n"
                "Azimuth 0° is in front and increases to the left."
            )
        )
        layout.addWidget(frame)
        layout.addStretch(1)
        return page

    # --- traversal tab ------------------------------------------------------

    def _build_traversal_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Traversal is how the source moves along the geometry over time. "
            "Changing it never changes the shape."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa0aa;")
        layout.addWidget(note)

        box = QGroupBox("Time law")
        form = QFormLayout(box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["loop", "ping_pong", "one_shot", "discontinuous"])
        self.mode_combo.currentTextChanged.connect(self._refresh)
        form.addRow("Looping mode", self.mode_combo)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.01, 36000.0)
        self.duration_spin.setValue(5.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.valueChanged.connect(self._refresh)
        form.addRow("Cycle duration", self.duration_spin)

        self.easing_combo = QComboBox()
        self.easing_combo.addItems(["linear", "sine", "smoothstep"])
        self.easing_combo.setToolTip(
            "Ease in and out of each cycle. 'sine' and 'smoothstep' start and "
            "end slowly; 'linear' holds one speed throughout."
        )
        self.easing_combo.currentTextChanged.connect(self._refresh)
        form.addRow("Ease in / out", self.easing_combo)

        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Forward", 1)
        self.direction_combo.addItem("Reverse", -1)
        self.direction_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("Direction", self.direction_combo)

        self.speed_combo = QComboBox()
        self.speed_combo.addItem("Constant linear speed", "constant_speed")
        self.speed_combo.addItem("Curve parameter speed", "parameter_speed")
        self.speed_combo.setToolTip(
            "Constant speed advances by physical distance, so the source covers "
            "metres at an even rate. Parameter speed advances along the curve's "
            "own parameter, which on a circle is constant angular speed instead."
        )
        self.speed_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("Speed law", self.speed_combo)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 128)
        self.steps_spin.setValue(8)
        self.steps_spin.valueChanged.connect(self._refresh)
        form.addRow("Jump positions", self.steps_spin)

        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(0.0, 10.0)
        self.crossfade_spin.setDecimals(3)
        self.crossfade_spin.setSuffix(" s")
        self.crossfade_spin.valueChanged.connect(self._refresh)
        form.addRow("Jump crossfade", self.crossfade_spin)
        layout.addWidget(box)

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

        # Coverage sits beside the path, not only in the render report: the
        # moment to learn that a dome leaves the measured region is while it is
        # being dragged.
        self.coverage_label = QLabel()
        self.coverage_label.setWordWrap(True)
        self.coverage_label.setToolTip(
            "What the selected HRTF dataset can and cannot support about this "
            "path. Directions outside the measured region are extrapolated "
            "rather than reproduced."
        )
        layout.addWidget(self.coverage_label)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------- load

    def _load_spec(self):
        geometry = self._spec.get("geometry", {}) or {}
        kind = str(geometry.get("type", "spline"))
        self._parameters = dict(geometry.get("parameters", {}) or {})
        if kind == "keyframes":
            self._keyframes = [
                Keyframe.from_mapping(dict(entry))
                for entry in geometry.get("keyframes", ())
            ] or self._default_keyframes()
        points = geometry.get("controlPointsM")
        self._points = (
            [list(map(float, point[:3])) for point in points]
            if isinstance(points, list) and len(points) >= 2
            else [[1.0, 1.0, 0.0], [1.5, 0.0, 0.4], [1.0, -1.0, 0.0]]
        )

        self.primitive_combo.blockSignals(True)
        index = self.primitive_combo.findText(kind)
        self.primitive_combo.setCurrentIndex(index if index >= 0 else self.primitive_combo.findText("spline"))
        self.primitive_combo.blockSignals(False)

        traversal = self._spec.get("traversal", {}) or {}
        self.mode_combo.setCurrentText(str(traversal.get("mode", "loop")))
        self.duration_spin.setValue(float(traversal.get("durationS", 5.0)))
        self.easing_combo.setCurrentText(str(traversal.get("easing", "linear")))
        self.direction_combo.setCurrentIndex(0 if int(traversal.get("direction", 1)) == 1 else 1)
        self.steps_spin.setValue(int(traversal.get("steps", 8)))
        self.crossfade_spin.setValue(float(traversal.get("crossfadeS", 0.0)))
        self.closed_check.setChecked(bool(geometry.get("closed", False)))
        self.interpolation_combo.setCurrentText(str(self._spec.get("interpolation", "cubic")))

        law = str(self._spec.get("speedLaw", "constant_speed" if self._spec.get("arcLength", True) else "parameter_speed"))
        self.speed_combo.setCurrentIndex(0 if law == "constant_speed" else 1)
        frame = str(self._spec.get("coordinateSystem", "listener_relative_cartesian"))
        self.frame_combo.setCurrentIndex(0 if frame == "listener_relative_cartesian" else 1)

        transform = self._spec.get("transform", {}) or {}
        self.offset_row.set_value(transform.get("translationM", (0.0, 0.0, 0.0)))
        self.scale_row.set_value(transform.get("scale", (1.0, 1.0, 1.0)))
        self.rotation_row.set_value(transform.get("yawPitchRollDegrees", (0.0, 0.0, 0.0)))

        self._rebuild_parameter_form()
        self._refresh()

    @staticmethod
    def _default_keyframes():
        return [
            Keyframe(0.0, (0.0, 1.0, 0.0)),
            Keyframe(5.0, (1.0, 0.0, 1.5)),
            Keyframe(10.0, (0.0, -1.0, 0.5)),
        ]

    # ------------------------------------------------------------------- mode

    @property
    def kind(self):
        return self.primitive_combo.currentText()

    @property
    def is_parametric(self):
        return self.kind in _SPATIAL

    @property
    def is_keyframed(self):
        return self.kind == "keyframes"

    def _primitive_changed(self, kind):
        if kind.startswith("—"):
            return
        if kind == "keyframes" and not self._keyframes:
            self._keyframes = self._default_keyframes()
        if kind in _SPATIAL:
            # Start from the primitive's own defaults rather than from whatever
            # numbers the previous primitive happened to use.
            self._parameters = {}
        self._selected = -1
        self._rebuild_parameter_form()
        self._refresh()

    def _rebuild_parameter_form(self):
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self._parameter_widgets = {}
        factory = _SPATIAL.get(self.kind)
        self.parameter_box.setVisible(factory is not None)
        if factory is None:
            return
        # Generated from the dataclass, so a primitive added to the core shows
        # up here without the dialog having to be edited to know about it.
        for field in dataclasses.fields(factory):
            widget = self._parameter_widget(field)
            if widget is None:
                continue
            widget.setToolTip(_parameter_tooltip(field.name))
            label = field.name.replace("_", " ").replace(" deg", " (°)").replace(" m", " (m)")
            label_widget = QLabel(label.capitalize())
            label_widget.setToolTip(widget.toolTip())
            self.parameter_form.addRow(label_widget, widget)
            self._parameter_widgets[field.name] = widget
        self._rebuild_motion_rows()

    def _parameter_widget(self, field):
        current = self._parameters.get(field.name, field.default)
        if isinstance(current, (list, tuple)) or isinstance(
            getattr(field, "default", None), tuple
        ):
            row = _AxisRow(" m")
            row.set_value(current if isinstance(current, (list, tuple)) else (0.0, 0.0, 0.0))
            row.changed.connect(self._parameter_edited)
            return row
        if isinstance(current, bool):
            check = QCheckBox()
            check.setChecked(bool(current))
            check.toggled.connect(self._parameter_edited)
            return check
        if isinstance(current, int) and not isinstance(current, bool):
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.setValue(int(current))
            spin.valueChanged.connect(self._parameter_edited)
            return spin
        spin = QDoubleSpinBox()
        spin.setRange(-3600.0, 3600.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setValue(float(current if current is not None else 0.0))
        spin.valueChanged.connect(self._parameter_edited)
        return spin

    def _parameter_edited(self):
        if self._updating:
            return
        for name, widget in self._parameter_widgets.items():
            if isinstance(widget, _AxisRow):
                self._parameters[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                self._parameters[name] = widget.isChecked()
            else:
                self._parameters[name] = widget.value()
        self._refresh()

    # --- parameter motion ----------------------------------------------------

    def set_disclosure(self, mode: str) -> None:
        """Hide the Motion group in Basic disclosure, but keep it built."""

        self._motion_disclosure = str(mode or "advanced").lower()
        self.motion_box.setVisible(self._motion_disclosure != "basic")

    def _has_motion_context(self) -> bool:
        return bool(
            callable(self._motion.get("scene")) and callable(self._motion.get("commit"))
        )

    def _motion_scene(self):
        provider = self._motion.get("scene")
        if not callable(provider):
            return None
        try:
            scene = provider()
        except Exception:  # noqa: BLE001 - a broken host must not break editing
            return None
        from src.audio.sam_workbench.scene_state import normalize_sam_scene

        return normalize_sam_scene(scene) if isinstance(scene, Mapping) else None

    @staticmethod
    def _route_range(route) -> tuple[float, float]:
        """The pair of numbers to show for a route, in the order it sweeps.

        A route saved before ranges existed carries only a depth, which meant
        base-to-base-plus-depth. Shown as that same interval so an existing
        document opens reading exactly as it renders.
        """

        if getattr(route, "has_range", False):
            low, high = float(route.minimum), float(route.maximum)
            return (high, low) if route.polarity < 0 else (low, high)
        return (0.0, float(route.depth) * float(route.polarity))

    def _motion_route_key(self, section: str, field: str) -> str:
        prefix = PATH_PREFIX if section == "geometry" else TRANSFORM_PREFIX
        return f"{prefix}{field}"

    def _build_motion_group(self):
        self.motion_box = QGroupBox("Parameter motion")
        box_layout = QVBoxLayout(self.motion_box)
        self.motion_status = QLabel()
        self.motion_status.setWordWrap(True)
        box_layout.addWidget(self.motion_status)
        self.motion_form = QFormLayout()
        box_layout.addLayout(self.motion_form)
        box_layout.addWidget(self._build_modulator_editor())
        self.motion_box.setToolTip(
            "Drive this path's own numbers from the scene's modulators. The "
            "value here swings around the number above by the depth shown, "
            "and the routes live in the scene's modulation matrix."
        )
        return self.motion_box

    def _build_modulator_editor(self):
        """Edit a modulator definition itself: shape, speed, phase, seed."""

        self.modulator_box = QGroupBox("Modulator")
        form = QFormLayout(self.modulator_box)

        self.motion_modulator_combo = QComboBox()
        self.motion_modulator_combo.setToolTip(
            "Which scene modulator to edit. Rows above reference these by name."
        )
        self.motion_modulator_combo.currentIndexChanged.connect(
            self._motion_modulator_selected
        )
        form.addRow("Name", self.motion_modulator_combo)

        self.motion_waveform_combo = QComboBox()
        self.motion_waveform_combo.addItems(list(MODULATOR_WAVEFORMS))
        self.motion_waveform_combo.setToolTip(
            "Sine, triangle and square sweep smoothly back and forth; random "
            "wanders between seeded values, smoothed at each step."
        )
        self.motion_waveform_combo.currentTextChanged.connect(
            lambda _text: self._modulator_definition_edited()
        )
        form.addRow("Waveform", self.motion_waveform_combo)

        self.motion_rate_spin = QDoubleSpinBox()
        self.motion_rate_spin.setRange(0.0, 1000.0)
        self.motion_rate_spin.setDecimals(3)
        self.motion_rate_spin.setSingleStep(0.05)
        self.motion_rate_spin.setSuffix(" Hz")
        self.motion_rate_spin.setToolTip("Cycles per second of this modulator.")
        self.motion_rate_spin.valueChanged.connect(lambda _v: self._modulator_definition_edited())
        form.addRow("Rate", self.motion_rate_spin)

        self.motion_phase_spin = QDoubleSpinBox()
        self.motion_phase_spin.setRange(-360.0, 360.0)
        self.motion_phase_spin.setDecimals(1)
        self.motion_phase_spin.setSuffix(" °")
        self.motion_phase_spin.setToolTip(
            "Where in its cycle the modulator starts. Offset two modulators "
            "to drive related parameters out of step."
        )
        self.motion_phase_spin.valueChanged.connect(lambda _v: self._modulator_definition_edited())
        form.addRow("Phase", self.motion_phase_spin)

        self.motion_seed_spin = QSpinBox()
        self.motion_seed_spin.setRange(0, 2_147_483_647)
        self.motion_seed_spin.setToolTip(
            "Only used by the random waveform: the same seed always produces "
            "the same wander."
        )
        self.motion_seed_spin.valueChanged.connect(lambda _v: self._modulator_definition_edited())
        form.addRow("Seed", self.motion_seed_spin)
        return self.modulator_box

    def _motion_fields(self):
        """Catalogue rows for the selected primitive, then the transform."""

        rows = []
        factory = _SPATIAL.get(self.kind)
        if factory is not None:
            for field_name in sorted(primitive_component_fields(factory)):
                spec = GEOMETRY_PARAMETER_SPECS.get(field_name)
                if spec is not None:
                    rows.append(("geometry", field_name, spec.label, spec.unit))
        for field_name in sorted(TRANSFORM_PARAMETER_SPECS):
            spec = TRANSFORM_PARAMETER_SPECS[field_name]
            rows.append(("transform", field_name, spec.label, spec.unit))
        return rows

    def _motion_routes_by_field(self) -> dict[str, Any]:
        scene = self._motion_scene()
        if scene is None:
            return {}
        source_id = str(self._motion.get("source_id", "") or "")
        matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
        found: dict[str, Any] = {}
        for route in matrix.routes:
            if route.target_id != source_id or not is_reserved_path(route.parameter_path):
                continue
            try:
                section, field_name = split_parameter_path(route.parameter_path)
            except ValueError:
                continue
            found[self._motion_route_key(section, field_name)] = route
        return found

    def _motion_modulator_ids(self) -> list[str]:
        scene = self._motion_scene()
        if scene is None:
            return []
        return [
            str(item.get("id"))
            for item in scene.get("modulators", ())
            if str(item.get("id") or "").strip()
        ]

    def _rebuild_motion_rows(self) -> None:
        previous = self._updating
        self._updating = True
        try:
            while self.motion_form.rowCount():
                self.motion_form.removeRow(0)
            self._motion_rows = {}

            context_ok = self._has_motion_context()
            routes = self._motion_routes_by_field()
            modulator_ids = self._motion_modulator_ids()

            for section, field_name, label, unit in self._motion_fields():
                key = self._motion_route_key(section, field_name)
                widgets = self._motion_row_widget(label, unit, modulator_ids)
                route = routes.get(key)
                widgets["enable"].setChecked(route is not None)
                if route is not None:
                    index = widgets["mod"].findData(route.modulator_id)
                    if index >= 0:
                        widgets["mod"].setCurrentIndex(index)
                    low, high = self._route_range(route)
                    widgets["low"].setValue(low)
                    widgets["high"].setValue(high)
                    curve_index = widgets["curve"].findText(route.curve)
                    if curve_index >= 0:
                        widgets["curve"].setCurrentIndex(curve_index)
                if not context_ok or not modulator_ids:
                    for widget in widgets.values():
                        widget.setEnabled(False)
                self.motion_form.addRow(f"{label}", widgets["container"])
                self._motion_rows[key] = widgets
        finally:
            self._updating = previous
        self._refresh_motion_status()
        self._refresh_modulator_editor()

    def _motion_row_widget(self, label: str, unit: str, modulator_ids: list[str]) -> dict[str, Any]:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        enable = QCheckBox()
        enable.setToolTip(f"Drive {label} from a scene modulator.")
        enable.toggled.connect(self._motion_edited)
        row.addWidget(enable)

        modulator = QComboBox()
        for identifier in modulator_ids:
            modulator.addItem(identifier, identifier)
        modulator.addItem("(new sine LFO)", "@new")
        modulator.setToolTip(
            "The scene modulator to use. '(new sine LFO)' adds a 0.25 Hz "
            "sine definition to the scene."
        )
        modulator.currentIndexChanged.connect(self._motion_edited)
        row.addWidget(modulator, 1)

        suffix = f" {unit}" if unit else ""

        def _spin(tip: str) -> QDoubleSpinBox:
            box = QDoubleSpinBox()
            box.setRange(-10_000.0, 10_000.0)
            box.setDecimals(3)
            box.setSingleStep(0.1)
            box.setSuffix(suffix)
            box.setToolTip(tip)
            box.valueChanged.connect(self._motion_edited)
            return box

        low = _spin(
            f"The value {label} falls to at the bottom of the modulator's "
            "cycle. It may be negative, and it may be above the high end - "
            "the path simply travels the other way."
        )
        high = _spin(
            f"The value {label} rises to at the top of the modulator's cycle. "
            "The pair is a range the parameter sweeps between, so it can "
            "cross zero; the number above is left behind while this row is on."
        )
        row.addWidget(QLabel("from"))
        row.addWidget(low)
        row.addWidget(QLabel("to"))
        row.addWidget(high)

        curve = QComboBox()
        curve.addItems(list(_MOTION_CURVES))
        curve.setCurrentText("linear")
        curve.setToolTip(
            "Shape applied to the modulator before scaling. Linear keeps "
            "the swing even; smooth eases the ends; exponential leans into "
            "the top of each cycle."
        )
        curve.currentTextChanged.connect(self._motion_edited)
        row.addWidget(curve)

        return {
            "container": container,
            "enable": enable,
            "mod": modulator,
            "low": low,
            "high": high,
            "curve": curve,
        }

    def _refresh_motion_status(self) -> None:
        active = [key for key, row in self._motion_rows.items() if row["enable"].isChecked()]
        if not self._has_motion_context():
            self.motion_status.setText(
                "This editor was opened without its scene, so path parameters "
                "cannot be driven from here. Open it through a voice's Path "
                "panel inside the SAM workbench."
            )
        elif not self.kind in _SPATIAL:
            self.motion_status.setText(
                "Point-drawn shapes have no named numbers to drive; pick a "
                "primitive to expose its parameters."
            )
        elif active:
            self.motion_status.setText("Driven: " + ", ".join(sorted(active)))
        else:
            self.motion_status.setText("No parameters driven.")

    def _create_modulator(self, definitions: list) -> str | None:
        existing = {
            str(item.get("id"))
            for item in definitions
            if str(item.get("id") or "").strip()
        }
        serial = 1
        while f"lfo{serial}" in existing:
            serial += 1
        identifier = f"lfo{serial}"
        definitions.append(
            {"id": identifier, "waveform": "sine", "rateHz": 0.25, "phaseDeg": 0.0, "seed": 0}
        )
        return identifier

    def _refresh_modulator_editor(self, select_id: str | None = None) -> None:
        """Rebuild the definition editor's names and fields from the scene."""

        previous = self._updating
        self._updating = True
        try:
            combo = self.motion_modulator_combo
            combo.blockSignals(True)
            wanted = select_id or combo.currentData()
            combo.clear()
            definitions = self._motion_definitions()
            for identifier in sorted(
                str(item.get("id")) for item in definitions if str(item.get("id") or "").strip()
            ):
                combo.addItem(identifier, identifier)
            if wanted is not None:
                index = combo.findData(wanted)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)

            has_context = self._has_motion_context() and bool(definitions)
            self.modulator_box.setEnabled(self._has_motion_context())
            definition = next(
                (
                    item
                    for item in definitions
                    if str(item.get("id")) == str(combo.currentData())
                ),
                None,
            )
            waveform = str((definition or {}).get("waveform", "sine"))
            index = self.motion_waveform_combo.findText(waveform)
            self.motion_waveform_combo.setCurrentIndex(index if index >= 0 else 0)
            self.motion_rate_spin.setValue(float((definition or {}).get("rateHz", 1.0)))
            self.motion_phase_spin.setValue(float((definition or {}).get("phaseDeg", 0.0)))
            self.motion_seed_spin.setValue(int((definition or {}).get("seed", 0) or 0))
            editable = has_context and definition is not None
            for widget in (
                self.motion_waveform_combo,
                self.motion_rate_spin,
                self.motion_phase_spin,
            ):
                widget.setEnabled(editable)
            self.motion_seed_spin.setEnabled(editable and waveform == "random")
        finally:
            self._updating = previous

    def _motion_definitions(self) -> list[dict[str, Any]]:
        scene = self._motion_scene()
        if scene is None:
            return []
        return [
            dict(item)
            for item in scene.get("modulators", ())
            if str(item.get("id") or "").strip()
        ]

    def _motion_modulator_selected(self, _index: int) -> None:
        if self._updating:
            return
        self._refresh_modulator_editor()

    def _modulator_definition_edited(self, *_args) -> None:
        if self._updating or not self._has_motion_context():
            return
        scene = self._motion_scene()
        if scene is None:
            return
        identifier = str(self.motion_modulator_combo.currentData() or "")
        if not identifier:
            return

        definitions = list(scene.get("modulators") or ())
        for item in definitions:
            if str(item.get("id")) != identifier:
                continue
            item["waveform"] = self.motion_waveform_combo.currentText()
            item["rateHz"] = float(self.motion_rate_spin.value())
            item["phaseDeg"] = float(self.motion_phase_spin.value())
            item["seed"] = int(self.motion_seed_spin.value())
            break
        else:
            return
        scene["modulators"] = definitions
        # Keep the seed control honest about whether it does anything.
        self.motion_seed_spin.setEnabled(
            self.motion_waveform_combo.currentText() == "random"
        )
        try:
            self._motion["commit"](scene)
        except Exception as error:  # noqa: BLE001 - reported rather than lost
            QMessageBox.warning(self, "Could not save modulator", str(error))
            return
        self._refresh()

    def _motion_edited(self, *_args) -> None:
        if self._updating or not self._has_motion_context():
            return
        scene = self._motion_scene()
        if scene is None:
            return
        source_id = str(self._motion.get("source_id", "") or "")
        matrix = ModulationMatrix.from_mapping(scene.get("modulation"))
        kept = tuple(
            route
            for route in matrix.routes
            if not (route.target_id == source_id and is_reserved_path(route.parameter_path))
        )
        matrix = dataclasses.replace(matrix, routes=kept)
        modulators = list(scene.get("modulators") or ())

        previous = self._updating
        self._updating = True
        created: str | None = None
        try:
            for key, row in sorted(self._motion_rows.items()):
                if not row["enable"].isChecked():
                    continue
                low = float(row["low"].value())
                high = float(row["high"].value())
                if low == high:
                    # Inert rather than refused: the user may be mid-edit, and
                    # a range with no width simply holds the value still.
                    continue
                modulator_id = row["mod"].currentData()
                if modulator_id == "@new":
                    modulator_id = self._create_modulator(modulators)
                    if modulator_id is None:
                        continue
                    created = modulator_id
                    index = row["mod"].findData(modulator_id)
                    if index >= 0:
                        row["mod"].setCurrentIndex(index)
                # Stored low-to-high with polarity carrying the direction,
                # so a range entered backwards still means "sweep the other
                # way" rather than being refused.
                inverted = high < low
                route = ModulationRoute(
                    modulator_id,
                    source_id,
                    key,
                    depth=abs(high - low),
                    polarity=-1 if inverted else 1,
                    curve=row["curve"].currentText(),
                    minimum=min(low, high),
                    maximum=max(low, high),
                )
                cycle = matrix.find_cycle(route)
                if cycle:
                    QMessageBox.warning(
                        self,
                        "Modulation cycle",
                        f"Driving {key} would close a loop:\n"
                        + " -> ".join(cycle),
                    )
                    continue
                matrix = matrix.with_route(route)
        finally:
            self._updating = previous

        scene["modulation"] = matrix.describe()
        scene["modulators"] = modulators
        try:
            self._motion["commit"](scene)
        except Exception as error:  # noqa: BLE001 - reported rather than lost
            QMessageBox.warning(self, "Could not save motion", str(error))
            return
        self._refresh_motion_status()
        self._refresh_modulator_editor(select_id=created)
        self._refresh()

    # -------------------------------------------------------------- selection

    def _select_point(self, index):
        self._selected = int(index)
        for view in self.views.values():
            view.set_selected(self._selected)
        self._updating = True
        if 0 <= self._selected < self.table.rowCount():
            self.table.selectRow(self._selected)
        self._updating = False
        self._refresh_numeric()

    def _table_selection_changed(self):
        if self._updating:
            return
        rows = {index.row() for index in self.table.selectedIndexes()}
        if rows:
            self._select_point(min(rows))

    def _editable_points(self):
        """The points the user can move, whichever geometry kind is active."""

        if self.is_keyframed:
            return [list(key.position_m) for key in self._keyframes]
        if self.is_parametric:
            return []
        return self._points

    def _set_editable_point(self, index, position):
        if self.is_keyframed:
            self._keyframes[index] = Keyframe(
                self._keyframes[index].time_s, tuple(float(v) for v in position)
            )
        else:
            self._points[index] = [float(value) for value in position]

    def _point_dragged(self, index, position):
        points = self._editable_points()
        if not (0 <= index < len(points)):
            return
        self._set_editable_point(index, position)
        self._selected = index
        self._refresh()

    # ------------------------------------------------------------ numeric edit

    def _refresh_numeric(self):
        points = self._editable_points()
        active = 0 <= self._selected < len(points)
        for widget in (
            self.cartesian_row,
            self.azimuth_spin,
            self.elevation_spin,
            self.distance_spin,
        ):
            widget.setEnabled(active)
        self.keyframe_time_spin.setEnabled(active and self.is_keyframed)
        if not active:
            return
        point = points[self._selected]
        self._updating = True
        self.cartesian_row.set_value(point)
        azimuth, elevation, distance = cartesian_array_to_spherical(np.asarray([point]))[0]
        self.azimuth_spin.setValue(float(azimuth))
        self.elevation_spin.setValue(float(elevation))
        self.distance_spin.setValue(float(distance))
        if self.is_keyframed:
            self.keyframe_time_spin.setValue(self._keyframes[self._selected].time_s)
        self._updating = False

    def _cartesian_edited(self):
        if self._updating or not (0 <= self._selected < len(self._editable_points())):
            return
        self._set_editable_point(self._selected, self.cartesian_row.value())
        self._refresh()

    def _spherical_edited(self):
        if self._updating or not (0 <= self._selected < len(self._editable_points())):
            return
        # Spherical entry is converted here, at the boundary, so what is stored
        # stays Cartesian and there is never a second position format to
        # reconcile.
        position = spherical_to_cartesian(
            self.azimuth_spin.value(),
            self.elevation_spin.value(),
            self.distance_spin.value(),
        )
        self._set_editable_point(self._selected, [float(value) for value in position])
        self._refresh()

    def _keyframe_time_edited(self):
        if self._updating or not self.is_keyframed:
            return
        if not (0 <= self._selected < len(self._keyframes)):
            return
        existing = self._keyframes[self._selected]
        self._keyframes[self._selected] = Keyframe(
            self.keyframe_time_spin.value(), existing.position_m
        )
        self._keyframes.sort(key=lambda key: key.time_s)
        self._refresh()

    def _table_edited(self, item):
        if self._updating:
            return
        points = self._editable_points()
        if not (0 <= item.row() < len(points)):
            return
        try:
            value = float(item.text())
        except ValueError:
            self._refresh()
            return
        if item.column() == 0:
            if self.is_keyframed:
                self._keyframes[item.row()] = Keyframe(
                    value, self._keyframes[item.row()].position_m
                )
                self._keyframes.sort(key=lambda key: key.time_s)
        else:
            point = list(points[item.row()])
            point[item.column() - 1] = value
            self._set_editable_point(item.row(), point)
        self._refresh()

    # ------------------------------------------------------------------ points

    def _add_point(self):
        points = self._editable_points()
        if self.is_parametric:
            return
        anchor = points[self._selected] if 0 <= self._selected < len(points) else [1.0, 0.0, 0.0]
        new = [anchor[0], anchor[1], anchor[2] + 0.25]
        if self.is_keyframed:
            last = self._keyframes[-1].time_s if self._keyframes else 0.0
            self._keyframes.append(Keyframe(last + 1.0, tuple(new)))
            self._selected = len(self._keyframes) - 1
        else:
            self._points.insert(self._selected + 1 if self._selected >= 0 else len(self._points), new)
            self._selected = min(self._selected + 1, len(self._points) - 1)
        self._refresh()

    def _remove_point(self):
        points = self._editable_points()
        # Two points is the minimum any path geometry can be built from.
        if len(points) <= 2 or not (0 <= self._selected < len(points)):
            return
        if self.is_keyframed:
            self._keyframes.pop(self._selected)
        else:
            self._points.pop(self._selected)
        self._selected = min(self._selected, len(self._editable_points()) - 1)
        self._refresh()

    def _import_points(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import path", "", "Path data (*.csv *.json *.txt);;All files (*)"
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            keyframes = (
                keyframes_from_json(text)
                if Path(path).suffix.lower() == ".json"
                else keyframes_from_csv(text)
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Import failed", str(error))
            return
        self._keyframes = list(keyframes)
        self.primitive_combo.setCurrentText("keyframes")
        self._selected = 0
        self._rebuild_parameter_form()
        self._refresh()

    # ----------------------------------------------------------------- helpers

    def _apply_to_points(self, transform):
        """Run a whole-path helper over whichever points are editable."""

        points = self._editable_points()
        if not points:
            QMessageBox.information(
                self,
                "Not available for this geometry",
                "This primitive is defined by its parameters rather than by "
                "points. Adjust its numbers on the Geometry tab instead.",
            )
            return
        moved = transform(np.asarray(points, dtype=float))
        for index, point in enumerate(moved):
            self._set_editable_point(index, point)
        self._refresh()

    def _snap_horizontal(self):
        self._apply_to_points(lambda points: np.column_stack((points[:, :2], np.zeros(len(points)))))

    def _snap_listener_height(self):
        def shift(points):
            result = points.copy()
            result[:, 2] -= float(np.mean(result[:, 2]))
            return result

        self._apply_to_points(shift)

    def _normalize_distance(self):
        def scale(points):
            mean = float(np.mean(np.linalg.norm(points, axis=1)))
            return points if mean <= 1e-9 else points / mean

        self._apply_to_points(scale)

    def _constant_distance(self):
        def project(points):
            radii = np.linalg.norm(points, axis=1, keepdims=True)
            target = float(np.mean(radii))
            # A point at the listener's own position has no direction to keep,
            # so it is left where it is rather than sent somewhere arbitrary.
            return np.where(radii > 1e-9, points / np.maximum(radii, 1e-9) * target, points)

        self._apply_to_points(project)

    def _reverse(self):
        index = self.direction_combo.currentIndex()
        self.direction_combo.setCurrentIndex(1 - index)

    def set_hrtf_dataset(self, dataset, label=""):
        """Draw and check coverage against the dataset this path will use.

        ``dataset`` may be a loaded dataset, a path to a SOFA file, or an array
        of measured positions. Passing ``None`` returns the view to the guessed
        shell and stops reporting coverage, which is honest: with no dataset
        selected there is nothing to be outside of.
        """

        positions = None
        if dataset is not None:
            positions = getattr(dataset, "positions_m", None)
            if positions is None and isinstance(dataset, (str, Path)):
                try:
                    from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

                    loaded = load_sofa(str(dataset))
                    positions = loaded.positions_m
                    label = label or Path(str(dataset)).name
                except Exception as error:  # noqa: BLE001 - reported in the label
                    self._dataset_positions = None
                    self._dataset_label = f"{dataset} could not be read: {error}"
                    self._refresh_coverage()
                    return
            if positions is None:
                positions = np.asarray(dataset, dtype=np.float64)
        self._dataset_positions = None if positions is None else np.asarray(
            positions, dtype=np.float64
        )
        self._dataset_label = label
        self._shell_toggled(self.shell_check.isChecked())
        self._refresh_coverage()

    def _shell_radius(self):
        """The radius the shell is drawn at, and where it came from."""

        if self._dataset_positions is not None and len(self._dataset_positions):
            radii = np.linalg.norm(self._dataset_positions, axis=1)
            radii = radii[radii > 0.0]
            if len(radii):
                return float(np.median(radii)), True
        points = self._sample_curve()
        if len(points):
            return float(np.mean(np.linalg.norm(points, axis=1))) or 1.5, False
        return 1.5, False

    def _shell_toggled(self, visible):
        radius, _measured = self._shell_radius()
        for view in self.views.values():
            view.set_coverage_shell(visible, radius)

    def _refresh_coverage(self):
        """Report what the selected dataset cannot support about this path.

        Shown beside the path rather than only at render time: the moment to
        learn that a dome leaves the measured region is while dragging it.
        """

        label = getattr(self, "coverage_label", None)
        if label is None:
            return
        if self._dataset_positions is None:
            label.setText(
                self._dataset_label
                or "No HRTF dataset selected, so coverage is not being checked."
            )
            return
        curve = self._sample_curve()
        if len(curve) < 2:
            label.setText("")
            return
        try:
            from src.audio.sam_workbench.hrtf.coverage import assess_path_coverage

            report = assess_path_coverage(self._dataset_positions, curve)
        except Exception as error:  # noqa: BLE001 - advice must not block editing
            label.setText(f"Coverage could not be assessed: {error}")
            return
        if not report.issues:
            name = self._dataset_label or "the selected dataset"
            label.setText(f"Fully covered by {name}.")
            return
        label.setText(
            "\n".join(f"• {issue.message}" for issue in report.issues)
        )

    # ----------------------------------------------------------------- refresh

    def _geometry_dict(self):
        if self.is_keyframed:
            return {
                "type": "keyframes",
                "interpolation": self.interpolation_combo.currentText(),
                "keyframes": [key.describe() for key in self._keyframes],
            }
        if self.is_parametric:
            return {"type": self.kind, "parameters": dict(self._parameters)}
        kind = self.kind if self.kind in _POINT_KINDS else "spline"
        return {
            "type": kind,
            "controlPointsM": [list(map(float, point)) for point in self._points],
            "closed": self.closed_check.isChecked(),
        }

    def trajectory_spec(self):
        """The saved form: geometry, traversal, and the metadata to read them."""

        return {
            "schemaVersion": 2,
            "coordinateSystem": self.frame_combo.currentData(),
            "handedness": "right",
            "units": "metres",
            "interpolation": self.interpolation_combo.currentText(),
            "speedLaw": self.speed_combo.currentData(),
            "closed": self.closed_check.isChecked(),
            "geometry": self._geometry_dict(),
            "transform": {
                "translationM": self.offset_row.value(),
                "yawPitchRollDegrees": self.rotation_row.value(),
                "scale": self.scale_row.value(),
                "shear": [0.0, 0.0, 0.0],
            },
            "traversal": {
                "mode": self.mode_combo.currentText(),
                "durationS": self.duration_spin.value(),
                "easing": self.easing_combo.currentText(),
                "direction": self.direction_combo.currentData(),
                "steps": self.steps_spin.value(),
                "crossfadeS": self.crossfade_spin.value(),
            },
            # Kept for builds that read the schema version 1 spelling.
            "arcLength": self.speed_combo.currentData() == "constant_speed",
        }

    def path_model(self):
        """The compiled model, or ``None`` when the current edit is invalid."""

        try:
            return path_model_from_dict(self.trajectory_spec())
        except (ValueError, TypeError, KeyError):
            return None

    def _preview_model(self):
        """What the renderer would move along: the path with the scene's
        motion attached, or the static path when nothing binds."""

        base = self.path_model()
        if base is None or not self._has_motion_context():
            return base
        scene = self._motion_scene()
        if scene is None:
            return base
        try:
            bound = compile_bound_trajectory(
                self.trajectory_spec(),
                scene,
                str(self._motion.get("source_id", "") or ""),
                sample_rate_hz=48_000.0,
                origin_sample=0,
                params={},
            )
        except Exception:  # noqa: BLE001 - a broken route must not break drawing
            return base
        model = bound.model
        return model if getattr(model, "bindings", None) else base

    def _sample_curve(self):
        """The *authored* shape, without motion - what coverage is judged on."""

        model = self.path_model()
        return self._positions_over(model)

    def _positions_over(self, model, count=_CURVE_SAMPLES):
        if model is None:
            return np.zeros((0, 3))
        # Sampled through the traversal, not the raw geometry: what is drawn is
        # what the renderer will be sent.
        times = np.linspace(0.0, self.duration_spin.value(), count)
        try:
            return np.asarray(model.positions(times), dtype=float)
        except (ValueError, TypeError):
            return np.zeros((0, 3))

    def _refresh(self):
        if self._updating:
            return
        bound = self._preview_model()
        base = self.path_model()
        moving = getattr(bound, "bindings", None)
        curve = self._positions_over(bound)
        # The dashed twin beneath the sweep is what makes modulation visible:
        # without it a breathing orbit reads as a slightly thick ring.
        reference = (
            self._positions_over(base) if moving and base is not None else np.zeros((0, 3))
        )
        points = self._editable_points()
        for view in self.views.values():
            view.set_reference_curve(reference)
            view.set_path(points, curve)
            view.set_selected(self._selected)
        self._refresh_table(points)
        self._refresh_numeric()
        self._refresh_metrics(curve)
        self._refresh_coverage()
        if self.shell_check.isChecked():
            self._shell_toggled(True)

    def _refresh_table(self, points):
        self._updating = True
        self.table.setRowCount(len(points))
        for row, point in enumerate(points):
            time_text = f"{self._keyframes[row].time_s:.3f}" if self.is_keyframed else "—"
            item = QTableWidgetItem(time_text)
            if not self.is_keyframed:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            for column, value in enumerate(point):
                self.table.setItem(row, column + 1, QTableWidgetItem(f"{value:.4f}"))
        self._updating = False

    def _refresh_metrics(self, curve):
        if len(curve) < 2:
            self.metrics_label.setText("Path is not valid yet.")
            return
        spherical = cartesian_array_to_spherical(curve)
        duration = max(self.duration_spin.value(), 1e-6)
        length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
        self.metrics_label.setText(
            f"Length {length:.2f} m over {duration:.2f} s "
            f"(mean {length / duration:.2f} m/s).\n"
            f"Elevation {spherical[:, 1].min():.1f}° to {spherical[:, 1].max():.1f}°, "
            f"distance {spherical[:, 2].min():.2f} m to {spherical[:, 2].max():.2f} m."
        )

    # ----------------------------------------------------------------- preview

    def _preview_toggled(self, running):
        if running:
            self._preview_time = 0.0
            self._timer.start()
        else:
            self._timer.stop()
            for view in self.views.values():
                view.set_marker(None)
                view.set_live_shape(None)
            self._refresh_motion_status()

    def _live_shape(self, model, time_s):
        """The path's shape at ``time_s``, or ``None`` when nothing moves it.

        A static path has one shape, already drawn; asking for an instant of it
        would put a second identical curve on top.
        """

        shape_at = getattr(model, "shape_at", None)
        if not callable(shape_at) or not getattr(model, "bindings", None):
            return None
        try:
            return np.asarray(shape_at(float(time_s), _CURVE_SAMPLES), dtype=float)
        except (ValueError, TypeError):
            return None

    def _live_motion_text(self) -> str:
        """The driven parameters' values right now, for the status line."""

        model = self._preview_model()
        if not getattr(model, "sample_parameter", None):
            return ""
        bits = []
        for key, row in sorted(self._motion_rows.items()):
            if not row["enable"].isChecked():
                continue
            values = model.sample_parameter(key, np.array([self._preview_time]))
            if values is None or not len(values):
                continue
            bits.append(f"{key.split('.', 1)[1]}={float(values[0]):.3g}")
        if not bits:
            return ""
        return f" · t {self._preview_time:.2f} s: " + ", ".join(bits)

    def _advance_preview(self):
        model = self._preview_model()
        if model is None:
            return
        self._preview_time += self._timer.interval() / 1000.0
        duration = max(self.duration_spin.value(), 1e-6)
        if self.mode_combo.currentText() == "one_shot" and self._preview_time > duration:
            self._preview_time = 0.0
        try:
            # Evaluated through the same model the renderer uses, so the marker
            # follows the trajectory actually being sent rather than the drawn
            # curve. With easing or reverse in play the two differ.
            position = np.asarray(model.positions(np.array([self._preview_time])))[0]
        except (ValueError, TypeError):
            return
        # The shape the path has at this instant, so a modulated path is seen
        # to breathe rather than sitting still under a moving dot. The marker
        # comes from the same model at the same time, so it rides the shape
        # being drawn rather than floating beside it.
        shape = self._live_shape(model, self._preview_time)
        for view in self.views.values():
            view.set_marker(position)
            view.set_live_shape(shape)
        live = self._live_motion_text()
        if live:
            driven = ", ".join(
                key for key, row in sorted(self._motion_rows.items())
                if row["enable"].isChecked()
            )
            self.motion_status.setText(f"Driven: {driven}{live}")

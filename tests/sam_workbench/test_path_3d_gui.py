"""The three-dimensional path designer.

The rules this file protects:

* the four views are views of one path - a selection or an edit in any of them
  is a selection or an edit in all of them;
* an orthographic view edits exactly the two axes it shows and leaves the third
  alone, which is the entire reason there are three of them;
* spherical entry converts on the way in, so the stored path stays Cartesian;
* geometry and traversal are edited and stored separately;
* the preview marker follows the compiled trajectory, not the drawn curve.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtCore import QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402

from src.audio.sam_workbench.trajectory import (  # noqa: E402
    PRIMITIVE_TYPES,
    path_model_from_dict,
    spherical_to_cartesian,
)
from src.ui.sam_path3d_dialog import SamPath3DDialog  # noqa: E402
from src.ui.sam_path3d_views import PLANES, OrthographicPathView  # noqa: E402

TIMES = np.linspace(0.0, 5.0, 64)


@pytest.fixture
def dialog(qtbot):
    widget = SamPath3DDialog()
    qtbot.addWidget(widget)
    widget.resize(1180, 820)
    return widget


def _press(view, position, button=Qt.LeftButton, buttons=Qt.LeftButton):
    return QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(position), button, buttons, Qt.NoModifier)


def _move(view, position):
    return QMouseEvent(QMouseEvent.MouseMove, QPointF(position), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def _release(view, position):
    return QMouseEvent(QMouseEvent.MouseButtonRelease, QPointF(position), Qt.LeftButton, Qt.NoButton, Qt.NoModifier)


# --- the views are views of one path ----------------------------------------


def test_the_designer_offers_a_perspective_view_and_three_editable_planes(dialog):
    assert set(dialog.views) == {"perspective", "top", "front", "side"}


def test_selecting_a_point_in_one_view_selects_it_everywhere(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog.views["top"].selectionChanged.emit(1)
    assert [view._selected for view in dialog.views.values()] == [1, 1, 1, 1]
    assert {index.row() for index in dialog.table.selectedIndexes()} == {1}


@pytest.mark.parametrize("plane", sorted(PLANES))
def test_an_orthographic_drag_leaves_the_axis_it_does_not_show_alone(dialog, plane):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._select_point(1)
    dialog._set_editable_point(1, [1.0, 0.3, 0.7])
    dialog._refresh()

    view = dialog.views[plane]
    view.resize(300, 240)
    before = list(dialog._editable_points()[1])
    target = view._project([before])[0]
    view.mousePressEvent(_press(view, target))
    view.mouseMoveEvent(_move(view, target + QPointF(25.0, -18.0)))
    view.mouseReleaseEvent(_release(view, target))

    after = dialog._editable_points()[1]
    locked = PLANES[plane]["locked"]
    assert after[locked] == pytest.approx(before[locked]), "the hidden axis moved"
    shown = [axis for axis in range(3) if axis != locked]
    assert any(abs(after[axis] - before[axis]) > 1e-6 for axis in shown)


def test_the_perspective_view_is_for_reading_rather_than_editing(dialog):
    """A drag there would be a guess about depth, which is what it must avoid."""

    assert dialog.views["perspective"]._editable is False


# --- numeric editing --------------------------------------------------------


def test_spherical_entry_is_converted_to_the_cartesian_store(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._select_point(1)
    dialog.azimuth_spin.setValue(45.0)
    dialog.elevation_spin.setValue(30.0)
    dialog.distance_spin.setValue(1.5)
    assert dialog._editable_points()[1] == pytest.approx(
        np.asarray(spherical_to_cartesian(45.0, 30.0, 1.5)), abs=1e-9
    )
    # Nothing spherical is stored; the saved geometry is metres.
    assert "azimuth" not in str(dialog.trajectory_spec()["geometry"]).lower()


def test_the_cartesian_and_spherical_editors_show_the_same_point(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._select_point(0)
    dialog.cartesian_row.set_value((1.0, 1.0, 0.5))
    dialog._cartesian_edited()
    dialog._select_point(0)
    assert dialog.distance_spin.value() == pytest.approx(np.linalg.norm([1.0, 1.0, 0.5]), abs=1e-3)
    assert dialog.azimuth_spin.value() == pytest.approx(45.0, abs=1e-2)


# --- geometry and traversal stay separate -----------------------------------


@pytest.mark.parametrize("kind", PRIMITIVE_TYPES)
def test_every_three_dimensional_primitive_compiles_and_reloads(dialog, kind):
    dialog.primitive_combo.setCurrentText(kind)
    model = dialog.path_model()
    assert model is not None
    positions = model.positions(TIMES)
    assert np.all(np.isfinite(positions))
    assert path_model_from_dict(dialog.trajectory_spec()).positions(TIMES) == pytest.approx(positions)


def test_the_parameter_form_is_generated_from_the_geometry_itself(dialog):
    """A primitive added to the core needs no edit here to become editable."""

    dialog.primitive_combo.setCurrentText("dome_traversal")
    assert set(dialog._parameter_widgets) >= {"distance_m", "turns", "end_elevation_deg"}


@pytest.mark.parametrize("kind", PRIMITIVE_TYPES)
def test_every_generated_primitive_parameter_has_a_tooltip(dialog, kind):
    dialog.primitive_combo.setCurrentText(kind)
    assert dialog._parameter_widgets
    assert all(widget.toolTip().strip() for widget in dialog._parameter_widgets.values())


def test_parameter_tooltips_explain_units_and_coordinate_signs(dialog):
    dialog.primitive_combo.setCurrentText("horizontal_orbit")
    assert "metres" in dialog._parameter_widgets["radius_m"].toolTip()
    assert "+x forward" in dialog._parameter_widgets["centre_m"].toolTip()
    assert "positive" in dialog._parameter_widgets["start_azimuth_deg"].toolTip()


def test_changing_traversal_does_not_change_the_geometry(dialog):
    dialog.primitive_combo.setCurrentText("horizontal_orbit")
    before = dialog.trajectory_spec()["geometry"]
    dialog.duration_spin.setValue(11.0)
    dialog.easing_combo.setCurrentText("sine")
    dialog.mode_combo.setCurrentText("ping_pong")
    assert dialog.trajectory_spec()["geometry"] == before


def test_the_saved_form_carries_the_metadata_needed_to_read_it(dialog):
    spec = dialog.trajectory_spec()
    assert spec["units"] == "metres"
    assert spec["handedness"] == "right"
    assert spec["coordinateSystem"] == "listener_relative_cartesian"
    assert spec["speedLaw"] in ("constant_speed", "parameter_speed")


def test_reversing_flips_direction_without_touching_the_shape(dialog):
    dialog.primitive_combo.setCurrentText("horizontal_orbit")
    geometry = dialog.trajectory_spec()["geometry"]
    dialog._reverse()
    assert dialog.trajectory_spec()["traversal"]["direction"] == -1
    assert dialog.trajectory_spec()["geometry"] == geometry


# --- whole-path helpers -----------------------------------------------------


def test_snapping_to_the_horizontal_plane_removes_all_height(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._snap_horizontal()
    assert all(point[2] == pytest.approx(0.0) for point in dialog._editable_points())


def test_maintaining_constant_distance_puts_every_point_on_one_sphere(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._constant_distance()
    radii = np.linalg.norm(np.asarray(dialog._editable_points()), axis=1)
    assert np.ptp(radii) == pytest.approx(0.0, abs=1e-9)


def test_normalizing_distance_gives_a_mean_radius_of_one_metre(dialog):
    dialog.primitive_combo.setCurrentText("spline")
    dialog._normalize_distance()
    radii = np.linalg.norm(np.asarray(dialog._editable_points()), axis=1)
    assert float(np.mean(radii)) == pytest.approx(1.0)


def test_a_parametric_primitive_keeps_at_least_two_points_editable_elsewhere(dialog):
    """Point helpers do not apply to a primitive defined by its numbers."""

    dialog.primitive_combo.setCurrentText("dome_traversal")
    assert dialog._editable_points() == []


# --- preview ----------------------------------------------------------------


def test_the_preview_marker_follows_the_compiled_trajectory(dialog):
    """Not the drawn curve: with easing or reverse in play the two differ."""

    dialog.primitive_combo.setCurrentText("horizontal_orbit")
    dialog.easing_combo.setCurrentText("sine")
    dialog.preview_button.setChecked(True)
    dialog._advance_preview()
    dialog._advance_preview()
    expected = dialog.path_model().positions(np.array([dialog._preview_time]))[0]
    for view in dialog.views.values():
        assert view._marker == pytest.approx(expected)


def test_stopping_the_preview_clears_the_marker(dialog):
    dialog.preview_button.setChecked(True)
    dialog._advance_preview()
    dialog.preview_button.setChecked(False)
    assert all(view._marker is None for view in dialog.views.values())


# --- the panel entry point --------------------------------------------------


def test_the_path_panel_offers_the_three_dimensional_designer(qtbot):
    from src.ui.sam_path_panel import SamPathPanel

    panel = SamPathPanel()
    qtbot.addWidget(panel)
    panel.set_params(
        {
            "canonicalTrajectory": {
                "geometry": {"type": "dome_traversal", "parameters": {"turns": 3}},
                "traversal": {"durationS": 6.0},
            }
        }
    )
    assert panel.designer_3d_button is not None
    # The panel previews a three-dimensional primitive without flattening it.
    assert np.ptp(panel.preview._points[:, 2]) > 0.1


def test_an_orthographic_view_renders_without_a_display(qtbot):
    view = OrthographicPathView("front")
    qtbot.addWidget(view)
    view.resize(280, 220)
    view.set_path([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], None)
    view.set_coverage_shell(True, 1.5)
    view.set_marker([0.5, 0.5, 0.5])
    assert not view.grab().isNull()

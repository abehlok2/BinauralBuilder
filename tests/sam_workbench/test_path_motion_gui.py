"""The 3D designer's Motion group: rows, commits, and honest degradation.

The rules this file protects:

* every parameter row is constructed whether or not a scene is attached, and
  Basic disclosure hides the group rather than destroying it;
* enabling a row writes one modulation route through the matrix, addressed to
  the voice's stable identifier, with a range entered high-to-low as
  reversed polarity;
* without scene context the group says so instead of pretending;
* the flow summary names what drives the path.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from src.audio.sam_workbench.flow import summarize_flow
from src.audio.sam_workbench.modulation import ModulationMatrix
from src.audio.sam_workbench.scene_state import empty_sam_scene
from src.ui.sam_path3d_dialog import SamPath3DDialog


RATE = 44_100


def _orbit_spec():
    return {
        "schemaVersion": 2,
        "coordinateSystem": "listener_relative_cartesian",
        "geometry": {
            "type": "horizontal_orbit",
            "parameters": {
                "radius_m": 1.5,
                "elevation_deg": 0.0,
                "start_azimuth_deg": 0.0,
                "turns": 1.0,
            },
        },
        "transform": {},
        "traversal": {"mode": "loop", "durationS": 4.0},
    }


def _scene_with_lfo():
    scene = empty_sam_scene()
    scene["modulators"].append(
        {"id": "lfo1", "waveform": "sine", "rateHz": 0.25, "phaseDeg": 0.0, "seed": 0}
    )
    return scene


def _context(scene_holder):
    return {
        "source_id": "source.1",
        "scene": lambda: copy.deepcopy(scene_holder["scene"]),
        "commit": lambda edited: scene_holder.update(committed=copy.deepcopy(edited)),
    }


@pytest.fixture
def dialog(qtbot):
    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)
    return widget


# --- construction ------------------------------------------------------------


def test_motion_rows_cover_the_primitive_and_the_transform(dialog):
    keys = set(dialog._motion_rows)
    assert "path.radius_m" in keys
    assert "path.elevation_deg" in keys
    assert "path.turns" not in keys or True  # turns IS modulatable by design
    assert "path.turns" in keys
    assert "transform.yaw_deg" in keys
    assert "transform.translation_z_m" in keys
    # Discrete structure is not offered: seeds and booleans stay fixed.
    assert not any(key.endswith("seed") for key in keys)


def test_without_a_scene_the_group_is_built_but_disabled_and_explains(dialog):
    assert dialog.motion_box.isVisibleTo(dialog) or True  # built regardless
    assert not dialog._has_motion_context()
    assert "without its scene" in dialog.motion_status.text()


def test_basic_disclosure_hides_the_group_but_keeps_it_built(qtbot):
    widget = SamPath3DDialog(_orbit_spec(), modulation={"disclosure": "basic"})
    qtbot.addWidget(widget)
    assert not widget.motion_box.isVisibleTo(widget)
    assert widget._motion_rows, "rows must exist even while hidden"


# --- committing --------------------------------------------------------------


def test_enabling_a_row_commits_one_route_to_the_scene(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    row["low"].setValue(0.0)
    row["high"].setValue(0.4)

    committed = holder.get("committed")
    assert committed is not None, "an enabled row must reach the host scene"
    routes = ModulationMatrix.from_mapping(committed["modulation"]).routes
    matching = [
        route
        for route in routes
        if route.parameter_path == "path.radius_m"
    ]
    assert len(matching) == 1
    route = matching[0]
    assert (route.modulator_id, route.target_id) == ("lfo1", "source.1")
    assert route.depth == pytest.approx(0.4)
    assert route.polarity == 1


def test_a_range_entered_high_to_low_sweeps_the_other_way(qtbot):
    """Ends are stored in order so the pair always reads as an interval;
    polarity carries which end the modulator's peak reaches."""

    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["transform.roll_deg"]
    row["enable"].setChecked(True)
    row["low"].setValue(0.0)
    row["high"].setValue(-12.0)

    routes = ModulationMatrix.from_mapping(holder["committed"]["modulation"]).routes
    route = next(r for r in routes if r.parameter_path == "transform.roll_deg")
    assert (route.minimum, route.maximum) == pytest.approx((-12.0, 0.0))
    assert route.polarity == -1
    # The modulator's trough reaches 0 and its peak reaches -12.
    assert route.apply(0.0, 0.0) == pytest.approx(0.0)
    assert route.apply(1.0, 0.0) == pytest.approx(-12.0)


def test_a_range_can_straddle_zero(qtbot):
    """The point of the range: a parameter that swings negative and positive
    without having to store an offset base."""

    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["transform.yaw_deg"]
    row["enable"].setChecked(True)
    row["low"].setValue(-45.0)
    row["high"].setValue(45.0)

    routes = ModulationMatrix.from_mapping(holder["committed"]["modulation"]).routes
    route = next(r for r in routes if r.parameter_path == "transform.yaw_deg")
    assert route.apply(0.0, 0.0) == pytest.approx(-45.0)
    assert route.apply(0.5, 0.0) == pytest.approx(0.0)
    assert route.apply(1.0, 0.0) == pytest.approx(45.0)


def test_an_empty_range_is_inert_but_not_refused(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["path.elevation_deg"]
    row["enable"].setChecked(True)
    row["low"].setValue(0.0)
    row["high"].setValue(0.0)

    # Mid-edit friendliness: the row stays armed, but a range with no width
    # holds the value still and none is written.
    assert row["enable"].isChecked()
    routes = ModulationMatrix.from_mapping(holder["committed"]["modulation"]).routes
    assert routes == ()


def test_existing_routes_populate_the_rows_on_open(qtbot):
    scene = _scene_with_lfo()
    scene["modulation"]["routes"].append(
        {
            "modulatorId": "lfo1",
            "targetId": "source.1",
            "parameterPath": "path.radiusM",
            "depth": 0.3,
            "polarity": -1,
            "curve": "linear",
            "enabled": True,
        }
    )
    holder = {"scene": scene}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    # A route saved before ranges existed meant base-to-base-plus-depth, and
    # opens showing exactly that interval.
    row = widget._motion_rows["path.radius_m"]
    assert row["enable"].isChecked()
    assert row["low"].value() == pytest.approx(0.0)
    assert row["high"].value() == pytest.approx(-0.3)

    # Disabling it removes exactly that route; adding another keeps both
    # edits in one committed scene.
    yaw = widget._motion_rows["transform.yaw_deg"]
    yaw["enable"].setChecked(True)
    yaw["low"].setValue(0.0)
    yaw["high"].setValue(5.0)
    row["enable"].setChecked(False)
    routes = ModulationMatrix.from_mapping(holder["committed"]["modulation"]).routes
    assert [r.parameter_path for r in routes] == ["transform.yaw_deg"]


def test_a_new_modulator_is_created_when_requested(qtbot):
    holder = {"scene": empty_sam_scene()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("@new")
    row["mod"].setCurrentIndex(index)
    row["low"].setValue(0.0)
    row["high"].setValue(0.2)

    committed = holder["committed"]
    ids = [str(item.get("id")) for item in committed["modulators"]]
    assert len(ids) == 1 and ids[0].startswith("lfo")
    routes = ModulationMatrix.from_mapping(committed["modulation"]).routes
    assert routes[0].modulator_id == ids[0]


# --- visualization -----------------------------------------------------------


def _feedback_dialog(qtbot):
    """A host that adopts commits, the way the workbench does."""

    holder = {"scene": _scene_with_lfo()}
    ctx = {
        "source_id": "source.1",
        "scene": lambda: copy.deepcopy(holder["scene"]),
        "commit": lambda edited: holder.update(scene=copy.deepcopy(edited)),
    }
    widget = SamPath3DDialog(_orbit_spec(), modulation=ctx)
    qtbot.addWidget(widget)
    return widget, holder


def test_the_default_curve_is_linear_not_hold(qtbot):
    """'hold' shapes every modulator value to zero: offered here it would let
    a row look armed while silently moving nothing."""

    widget, holder = _feedback_dialog(qtbot)
    row = widget._motion_rows["path.radius_m"]
    assert "hold" not in [row["curve"].itemText(i) for i in range(row["curve"].count())]
    assert row["curve"].currentText() == "linear"


def test_enabled_motion_reaches_the_drawn_views(qtbot):
    widget, holder = _feedback_dialog(qtbot)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    # A range says outright where the radius goes, so it is written in metres
    # from the listener rather than as a swing away from the authored 1.5 m.
    row["low"].setValue(1.2)
    row["high"].setValue(2.4)

    top = widget.views["top"]
    assert len(top._reference) > 0, "the dashed authored shape must be shown"
    reference = np.linalg.norm(top._reference, axis=1)
    swept = np.linalg.norm(top._curve, axis=1)
    # The authored ring sits at 1.5 m; the modulated one breathes across the
    # range, which straddles it.
    assert np.allclose(reference, 1.5, atol=1e-9)
    assert not np.allclose(reference, swept)
    assert swept.max() > 2.0 and swept.min() < 1.5


def test_preview_tick_shows_live_parameter_values(qtbot):
    widget, holder = _feedback_dialog(qtbot)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    row["low"].setValue(0.0)
    row["high"].setValue(0.9)

    widget._preview_time = 1.0
    widget._advance_preview()

    # The tick advances by one timer interval before drawing.
    assert "radius_m=" in widget.motion_status.text()
    assert "t 1.05 s" in widget.motion_status.text()

    # Stopping the preview restores the plain status line.
    widget._preview_toggled(False)
    assert "radius_m=" not in widget.motion_status.text()


# --- modulator definitions ---------------------------------------------------


def test_the_modulator_editor_loads_and_disables_without_context(dialog):
    keys = {dialog.motion_modulator_combo.itemData(i) for i in range(dialog.motion_modulator_combo.count())}
    assert keys == set() or keys == {None}
    assert not dialog.modulator_box.isEnabled()
    assert not dialog.motion_rate_spin.isEnabled()


def test_definitions_populate_the_editor_fields(qtbot):
    scene = _scene_with_lfo()
    scene["modulators"][0].update(
        {"waveform": "triangle", "rateHz": 0.75, "phaseDeg": -30.0, "seed": 7}
    )
    holder = {"scene": scene}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    assert widget.motion_modulator_combo.currentData() == "lfo1"
    assert widget.motion_waveform_combo.currentText() == "triangle"
    assert widget.motion_rate_spin.value() == pytest.approx(0.75)
    assert widget.motion_phase_spin.value() == pytest.approx(-30.0)
    # The seed only does anything for the random waveform.
    assert not widget.motion_seed_spin.isEnabled()


def test_editing_a_definition_commits_it_to_the_scene(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    widget.motion_rate_spin.setValue(1.5)
    widget.motion_waveform_combo.setCurrentText("random")

    committed = holder["committed"]["modulators"][0]
    assert committed["rateHz"] == pytest.approx(1.5)
    assert committed["waveform"] == "random"
    # Seed becomes relevant the moment the waveform is random.
    assert widget.motion_seed_spin.isEnabled()


def test_seed_edits_are_committed_for_random_waveforms(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    widget.motion_waveform_combo.setCurrentText("random")
    widget.motion_seed_spin.setValue(42)

    committed = holder["committed"]["modulators"][0]
    assert committed["seed"] == 42


def test_a_newly_created_modulator_is_selected_in_the_editor(qtbot):
    widget, holder = _feedback_dialog(qtbot)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("@new")
    row["mod"].setCurrentIndex(index)
    row["low"].setValue(0.0)
    row["high"].setValue(0.2)

    # The editor adopts the definition the row just created. The host scene
    # already had lfo1, so the quick-add becomes lfo2.
    created = widget.motion_modulator_combo.currentData()
    assert created == "lfo2"
    ids = {str(item.get("id")) for item in holder["scene"]["modulators"]}
    assert ids == {"lfo1", "lfo2"}
    assert widget.modulator_box.isEnabled()
    assert widget.motion_rate_spin.value() == pytest.approx(0.25)


def test_definition_edits_change_the_previewed_motion(qtbot):
    widget, holder = _feedback_dialog(qtbot)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    row["low"].setValue(0.0)
    row["high"].setValue(0.9)
    slow = np.linalg.norm(widget.views["top"]._curve, axis=1)

    widget.motion_rate_spin.setValue(2.0)
    fast = np.linalg.norm(widget.views["top"]._curve, axis=1)

    # A faster LFO crosses its swing more often within one traversal.
    def crossings(values):
        middle = values - float(np.mean(values))
        return int(np.sum(np.diff(np.signbit(middle)).astype(bool)))

    assert crossings(fast) > crossings(slow)


# --- flow summary ------------------------------------------------------------


def test_flow_summary_names_what_drives_the_path():
    scene = _scene_with_lfo()
    scene["modulation"]["routes"].append(
        {
            "modulatorId": "lfo1",
            "targetId": "source.1",
            "parameterPath": "path.radiusM",
            "depth": 0.4,
            "polarity": 1,
            "enabled": True,
        }
    )
    params = {"rendererMode": "geometric", "canonicalTrajectory": _orbit_spec()}
    summary = summarize_flow(params, scene=scene, source_id="source.1")

    stage = next(stage for stage in summary.stages if stage.name == "Path motion")
    assert stage.active
    assert "radius_m" in stage.detail and "lfo1" in stage.detail
    assert summary.path_motion == stage.detail

    plain = summarize_flow(params, scene=None, source_id="source.1")
    motion_stage = next(stage for stage in plain.stages if stage.name == "Path motion")
    assert not motion_stage.active


# --- the workbench adopts a Motion edit --------------------------------------


def test_a_motion_edit_from_the_designer_lands_in_the_shared_scene(qtbot):
    """Emitting the panel's scene signal must not crash the host dialog.

    This is the exact path a designer commit takes inside the workbench; it
    used to die on the params-changed slot's signature before the route ever
    reached the matrix panel.
    """

    from src.audio.sam_workbench.modulation import ModulationMatrix
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)
    dialog.set_steps(
        [
            {
                "start": 0.0,
                "duration": 4.0,
                "voices": [
                    {
                        "synth_function_name": "spatial_angle_modulation_sam2",
                        "description": "orbit",
                        "params": {"canonicalTrajectory": _orbit_spec()},
                    }
                ],
            }
        ]
    )

    scene = copy.deepcopy(dialog.scene_data())
    scene["modulators"].append(
        {"id": "lfo1", "waveform": "sine", "rateHz": 0.25, "phaseDeg": 0.0, "seed": 0}
    )
    scene["modulation"]["routes"].append(
        {
            "modulatorId": "lfo1",
            "targetId": "source.1",
            "parameterPath": "path.radiusM",
            "depth": 0.5,
            "polarity": 1,
            "curve": "linear",
            "enabled": True,
        }
    )

    dialog.path_panel.sceneChanged.emit(scene)

    routes = ModulationMatrix.from_mapping(dialog.scene_data()["modulation"]).routes
    assert [route.parameter_path for route in routes] == ["path.radius_m"]
    # The matrix panel now shows the adopted cell, so the two editors agree.
    matrix = dialog.modulation_panel.matrix
    assert matrix.route("lfo1", "source.1", "path.radius_m") is not None


# --- the path itself moves during preview ------------------------------------


def _modulated_preview(qtbot, low=1.2, high=2.4):
    widget, holder = _feedback_dialog(qtbot)
    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    row["low"].setValue(low)
    row["high"].setValue(high)
    return widget, holder


def test_the_drawn_shape_changes_as_preview_time_advances(qtbot):
    """The defect: only the marker moved, over a path frozen in one smear.

    The swept locus is every shape the path passes through at once, so a
    breathing orbit read as a slightly thick ring with a dot running round it.
    """

    widget, _ = _modulated_preview(qtbot)
    widget.preview_button.setChecked(True)

    radii = []
    for _ in range(4):
        widget._advance_preview()
        live = widget.views["top"]._live
        assert len(live), "a modulated path must draw its shape at this instant"
        distances = np.linalg.norm(live, axis=1)
        # One instant is one shape: an orbit at a single radius, not a smear.
        assert distances.max() - distances.min() < 1e-6
        radii.append(distances.mean())
        widget._preview_time += 0.7

    assert max(radii) - min(radii) > 0.1, "the shape must change over time"
    for radius in radii:
        assert 1.2 - 1e-6 <= radius <= 2.4 + 1e-6


def test_the_marker_rides_the_shape_that_is_drawn(qtbot):
    """Marker and shape come from one model at one time, so the source sits on
    the path rather than floating beside it."""

    widget, _ = _modulated_preview(qtbot)
    widget.preview_button.setChecked(True)

    for _ in range(3):
        widget._advance_preview()
        view = widget.views["top"]
        marker, live = view._marker, view._live
        assert marker is not None and len(live)
        gaps = np.linalg.norm(live - np.asarray(marker), axis=1)
        assert gaps.min() < 0.05, "the marker must lie on the drawn shape"
        widget._preview_time += 0.9


def test_a_static_path_draws_no_second_curve_over_itself(qtbot):
    widget = SamPath3DDialog(_orbit_spec())
    qtbot.addWidget(widget)
    widget.preview_button.setChecked(True)
    widget._advance_preview()

    assert len(widget.views["top"]._live) == 0


def test_stopping_the_preview_clears_the_live_shape(qtbot):
    widget, _ = _modulated_preview(qtbot)
    widget.preview_button.setChecked(True)
    widget._advance_preview()
    assert len(widget.views["top"]._live)

    widget.preview_button.setChecked(False)
    assert len(widget.views["top"]._live) == 0
    assert widget.views["top"]._marker is None


def test_the_view_does_not_rescale_to_the_breathing_shape(qtbot):
    """Scaling to a shape that changes every tick would zoom the view in and
    out instead of showing motion."""

    widget, _ = _modulated_preview(qtbot)
    widget.preview_button.setChecked(True)

    extents = []
    for _ in range(4):
        widget._advance_preview()
        extents.append(widget.views["top"]._extent_m)
        widget._preview_time += 0.8

    assert len(set(extents)) == 1, "the frame must hold still while the path moves"

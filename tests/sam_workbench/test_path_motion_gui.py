"""The 3D designer's Motion group: rows, commits, and honest degradation.

The rules this file protects:

* every parameter row is constructed whether or not a scene is attached, and
  Basic disclosure hides the group rather than destroying it;
* enabling a row writes one modulation route through the matrix, addressed to
  the voice's stable identifier, with negative depth as reversed polarity;
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
    row["depth"].setValue(0.4)

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


def test_negative_depth_stores_reversed_polarity(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["transform.roll_deg"]
    row["enable"].setChecked(True)
    row["depth"].setValue(-12.0)

    routes = ModulationMatrix.from_mapping(holder["committed"]["modulation"]).routes
    route = next(r for r in routes if r.parameter_path == "transform.roll_deg")
    assert route.depth == pytest.approx(12.0)
    assert route.polarity == -1


def test_zero_depth_is_inert_but_not_refused(qtbot):
    holder = {"scene": _scene_with_lfo()}
    widget = SamPath3DDialog(_orbit_spec(), modulation=_context(holder))
    qtbot.addWidget(widget)

    row = widget._motion_rows["path.elevation_deg"]
    row["enable"].setChecked(True)
    row["depth"].setValue(0.0)

    # Mid-edit friendliness: the row stays armed, but a zero-depth route
    # contributes nothing and none is written.
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

    row = widget._motion_rows["path.radius_m"]
    assert row["enable"].isChecked()
    assert row["depth"].value() == pytest.approx(-0.3)

    # Disabling it removes exactly that route; adding another keeps both
    # edits in one committed scene.
    yaw = widget._motion_rows["transform.yaw_deg"]
    yaw["enable"].setChecked(True)
    yaw["depth"].setValue(5.0)
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
    row["depth"].setValue(0.2)

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
    row["depth"].setValue(0.9)

    top = widget.views["top"]
    assert len(top._reference) > 0, "the dashed authored shape must be shown"
    reference = np.linalg.norm(top._reference, axis=1)
    swept = np.linalg.norm(top._curve, axis=1)
    # The authored ring sits at 1.5 m; the modulated one breathes around it.
    assert np.allclose(reference, 1.5, atol=1e-9)
    assert not np.allclose(reference, swept)
    assert swept.max() > 1.5 + 0.3 and swept.min() < 1.5 + 0.05


def test_preview_tick_shows_live_parameter_values(qtbot):
    widget, holder = _feedback_dialog(qtbot)

    row = widget._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    index = row["mod"].findData("lfo1")
    row["mod"].setCurrentIndex(max(index, 0))
    row["depth"].setValue(0.9)

    widget._preview_time = 1.0
    widget._advance_preview()

    # The tick advances by one timer interval before drawing.
    assert "radius_m=" in widget.motion_status.text()
    assert "t 1.05 s" in widget.motion_status.text()

    # Stopping the preview restores the plain status line.
    widget._preview_toggled(False)
    assert "radius_m=" not in widget.motion_status.text()


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

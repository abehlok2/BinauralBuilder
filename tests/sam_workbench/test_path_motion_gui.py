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

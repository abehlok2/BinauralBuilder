"""Regression contracts for the 3D authoring review fixes."""

import copy
from dataclasses import replace
import numpy as np
import pytest
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QDialogButtonBox
from src.audio.sam_workbench.trajectory import path_model_from_dict
from src.audio.sam_workbench.trajectory.authoring import PathConstraints, loop_findings
from src.ui.sam_path3d_dialog import SamPath3DDialog
from src.ui.sam_path3d_views import PerspectivePathView
from src.ui.sam_path_panel import SamPathPanel


def _orbit_spec():
    return {
        "schemaVersion": 2,
        "geometry": {"type": "horizontal_orbit", "parameters": {"radius_m": 1.5}},
        "traversal": {"mode": "loop", "durationS": 4.0},
    }


def _scene_with_lfo():
    from src.audio.sam_workbench.scene_state import empty_sam_scene

    scene = empty_sam_scene()
    scene["modulators"].append(
        {"id": "lfo1", "waveform": "sine", "rateHz": 0.25, "phaseDeg": 0.0, "seed": 0}
    )
    return scene


def spec():
    return {
        "schemaVersion": 2,
        "geometry": {"type": "polyline", "controlPointsM": [[1, 0, 0], [0, 0, 1]]},
        "speedLaw": "parameter_speed",
        "traversal": {"durationS": 2, "mode": "one_shot"},
    }


def test_distance_constraint_survives_coordinate_crossfade():
    payload = spec()
    payload.update(
        schemaVersion=3, constraints={"distance_m": 2}, coordinateSmoothing=True
    )
    payload["traversal"].update(mode="discontinuous", steps=3, crossfadeS=0.1)
    points = path_model_from_dict(payload).positions(np.linspace(0, 2, 1001))
    np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 2, atol=1e-12)


def test_angular_speed_rejects_radial_only_sections():
    payload = spec()
    payload.update(schemaVersion=3, speedLaw="angular_speed")
    payload["geometry"]["controlPointsM"] = [[1, 0, 0], [2, 0, 0], [0, 0, 2]]
    with pytest.raises(ValueError, match="radial-only"):
        path_model_from_dict(payload).positions([0, 1])


def test_legacy_launch_promotes_shape_without_committing_on_cancel(qtbot, monkeypatch):
    from src.audio.sam_workbench.trajectory.legacy_paths import (
        promote_profile_to_trajectory,
    )

    seen = []

    class Designer:
        def __init__(self, initial, *args, **kwargs):
            seen.append(copy.deepcopy(initial))

        def set_render_context(self, context):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr("src.ui.sam_path_panel.SamPath3DDialog", Designer)
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    profile = {"points": [[0, 0], [1, 1]], "closedLoop": False}
    panel.set_params({"customPathProfile": profile})
    original = panel.params()
    panel.open_3d_designer()
    assert seen == [promote_profile_to_trajectory(profile)]
    assert panel.params() == original
    panel.set_params({"canonicalTrajectory": spec()})
    assert not panel.designer_button.isEnabled()


@pytest.mark.parametrize("pitch", [-80, 0, 22, 80])
def test_positive_height_projects_up(qtbot, pitch):
    view = PerspectivePathView()
    qtbot.addWidget(view)
    view.set_orbit(35, pitch)
    origin, above = view._project([[0, 0, 0], [0, 0, 1]])
    assert above.y() < origin.y()


def test_noop_preserves_world_pose_shear_and_extensions(qtbot):
    payload = spec()
    payload.update(
        coordinateSystem="world_cartesian",
        listener={
            "positionM": [2, 1, 0.5],
            "yawPitchRollDegrees": [20, 10, 5],
            "earSpacingM": 0.21,
            "extension": "listener",
        },
        sourceOrientation={"mode": "toward_listener", "extension": 42},
        coordinateSmoothing=True,
        custom={"abc": [1, 2]},
    )
    payload["transform"] = {
        "translationM": [0.5, 0.3, 0.1],
        "yawPitchRollDegrees": [10, 20, 30],
        "scale": [1, 2, 1],
        "shear": [0.1, 0.2, 0.3],
        "unknown": "keep",
    }
    payload["geometry"]["extension"] = {"keep": True}
    dialog = SamPath3DDialog(payload)
    qtbot.addWidget(dialog)
    saved = dialog.trajectory_spec()
    for key in (
        "listener",
        "sourceOrientation",
        "coordinateSmoothing",
        "custom",
        "transform",
    ):
        assert saved[key] == payload[key]
    assert saved["geometry"]["extension"] == payload["geometry"]["extension"]
    times = np.linspace(0, 2, 11)
    np.testing.assert_allclose(
        path_model_from_dict(saved).positions(times),
        path_model_from_dict(payload).positions(times),
    )


def test_handles_and_numeric_editor_use_transformed_listener_frame(qtbot):
    payload = spec()
    payload["transform"] = {
        "translationM": [1, 2, 3],
        "yawPitchRollDegrees": [30, 20, 10],
        "scale": [2, 1, 1],
    }
    dialog = SamPath3DDialog(payload)
    qtbot.addWidget(dialog)
    expected = dialog.path_model().transform.apply(
        payload["geometry"]["controlPointsM"]
    )
    np.testing.assert_allclose(dialog.views["front"]._points, expected)
    dialog._select_point(0)
    np.testing.assert_allclose(dialog.cartesian_row.value(), expected[0], atol=1e-3)
    moved = expected[0] + np.array([0, 0, 0.5])
    dialog._point_dragged(0, moved)
    np.testing.assert_allclose(dialog._to_display(dialog._editable_points())[0], moved)


def test_main_panel_matches_full_path_model(qtbot):
    payload = spec()
    payload.update(
        coordinateSystem="world_cartesian",
        listener={"positionM": [1, 0, 0]},
        transform={"translationM": [0, 1, 1]},
    )
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    panel.set_params({"canonicalTrajectory": payload})
    np.testing.assert_allclose(
        panel.preview._points,
        path_model_from_dict(payload).positions(np.linspace(0, 2, 512)),
    )
    assert "Canonical 3D path" in panel.metadata_label.text()


def test_drag_freezes_scale_and_is_one_undo(qtbot):
    dialog = SamPath3DDialog(spec())
    qtbot.addWidget(dialog)
    original = copy.deepcopy(dialog.trajectory_spec())
    extent = dialog.views["front"]._extent_m
    dialog._begin_drag()
    for z in (5, 10, 20):
        dialog._point_dragged(0, [1, 0, z])
    assert dialog.views["front"]._extent_m == extent
    dialog._end_drag()
    edited = dialog.trajectory_spec()
    dialog.undo()
    assert dialog.trajectory_spec() == original
    dialog.redo()
    assert dialog.trajectory_spec() == edited


def test_invalid_and_future_paths_cannot_be_accepted(qtbot):
    dialog = SamPath3DDialog(spec())
    qtbot.addWidget(dialog)
    dialog.scale_row.set_value([0, 1, 1])
    dialog._refresh()
    assert not dialog.button_box.button(QDialogButtonBox.Ok).isEnabled()
    dialog.accept()
    assert dialog.result() != dialog.Accepted
    payload = spec()
    payload["schemaVersion"] = 99
    future = SamPath3DDialog(payload)
    qtbot.addWidget(future)
    assert not future.button_box.button(QDialogButtonBox.Ok).isEnabled()
    with pytest.raises(ValueError, match="schemaVersion"):
        path_model_from_dict(payload)


def test_motion_cancel_is_transactional_and_undo_restores_draft(qtbot):

    scene = _scene_with_lfo()
    original = copy.deepcopy(scene)
    commits = []
    dialog = SamPath3DDialog(
        _orbit_spec(),
        modulation={
            "source_id": "voice-a",
            "scene": lambda: scene,
            "commit": commits.append,
        },
    )
    qtbot.addWidget(dialog)
    draft = copy.deepcopy(dialog._draft_scene)
    draft["extensions"] = {"edit": True}
    dialog._stage_scene(draft)
    dialog._refresh()
    assert not commits and scene == original
    dialog.undo()
    assert dialog._draft_scene == original
    dialog.redo()
    assert dialog._draft_scene["extensions"] == {"edit": True}
    dialog.reject()
    assert not commits


def test_elapsed_clock_and_one_shot_endpoint(qtbot):
    class Clock:
        def isValid(self):
            return True

        def elapsed(self):
            return 2500

    dialog = SamPath3DDialog(spec())
    qtbot.addWidget(dialog)
    dialog._preview_clock = Clock()
    dialog._advance_preview()
    assert dialog._preview_time == 2
    for view in dialog.views.values():
        np.testing.assert_allclose(view._marker, [0, 0, 1])
    assert not dialog._timer.isActive()


def test_compiled_preview_is_cached_between_ticks(qtbot):
    dialog = SamPath3DDialog(spec())
    qtbot.addWidget(dialog)
    first = dialog._preview_model()
    assert dialog._preview_model() is first
    dialog.offset_row.set_value([0, 0, 1])
    dialog._refresh()
    assert dialog._preview_model() is not first


def test_spherical_interpolation_stays_on_sphere_and_survives_roundtrip():
    payload = spec()
    payload.update(schemaVersion=3, interpolation="spherical")
    payload["geometry"]["interpolation"] = "spherical"
    model = path_model_from_dict(payload)
    times = np.linspace(0, 2, 101)
    points = model.positions(times)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(points[50], [2**-0.5, 0, 2**-0.5])
    np.testing.assert_allclose(
        path_model_from_dict(model.describe()).positions(times), points
    )


@pytest.mark.parametrize("points", [[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [-1, 0, 0]]])
def test_spherical_undefined_directions_are_rejected(points):
    payload = spec()
    payload["geometry"].update(controlPointsM=points, interpolation="spherical")
    with pytest.raises(ValueError):
        path_model_from_dict(payload)


def test_constraints_apply_between_keys_and_after_transform():
    payload = spec()
    payload["schemaVersion"] = 3
    payload["constraints"] = {
        "distance_m": 2,
        "azimuth_deg": 30,
        "minimum_height_m": 0.2,
        "maximum_height_m": 1,
    }
    payload["transform"] = {"translationM": [0.2, 0.1, 0]}
    points = path_model_from_dict(payload).positions(np.linspace(0, 2, 101))
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 2)
    np.testing.assert_allclose(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 30)
    assert points[:, 2].min() >= 0.2 and points[:, 2].max() <= 1


def test_conflicting_constraints_are_rejected():
    with pytest.raises(ValueError):
        PathConstraints(distance_m=1, minimum_height_m=2)
    with pytest.raises(ValueError):
        PathConstraints(distance_m=1, elevation_deg=0, minimum_height_m=0.5)


def test_angular_speed_uses_effective_direction_and_is_block_invariant():
    payload = spec()
    payload.update(schemaVersion=3, speedLaw="angular_speed")
    payload["geometry"]["controlPointsM"] = [[3, 0, 0], [0, 0, 1]]
    model = path_model_from_dict(payload)
    times = np.linspace(0, 2, 101)
    points = model.positions(times)
    directions = points / np.linalg.norm(points, axis=1)[:, None]
    delta = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
    np.testing.assert_allclose(delta, np.pi / 200, atol=2e-6)
    np.testing.assert_allclose(
        np.vstack([model.positions(part) for part in np.array_split(times, 7)]), points
    )


def test_authored_timestamps_are_not_arclength_remapped():
    payload = spec()
    payload.update(schemaVersion=3, speedLaw="authored_timing")
    payload["geometry"] = {
        "type": "keyframes",
        "interpolation": "linear",
        "keyframes": [
            {"timeSeconds": 1, "position": [1, 0, 0]},
            {"timeSeconds": 9, "position": [1, 0, 1]},
            {"timeSeconds": 10, "position": [1, 0, 3]},
        ],
    }
    model = path_model_from_dict(payload)
    np.testing.assert_allclose(
        model.positions([0, 1, 9, 10]), [[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 3]]
    )


def test_open_loop_diagnostic():
    payload = spec()
    payload["traversal"]["mode"] = "loop"
    assert "Loop jumps" in loop_findings(path_model_from_dict(payload))[0]


def test_production_hrtf_adapter_uses_constraints_and_timing():
    from src.audio.sam_workbench.compat import _hrtf_trajectory

    payload = spec()
    payload.update(schemaVersion=3, constraints={"distance_m": 2})
    times = np.linspace(0, 2, 11)
    np.testing.assert_allclose(
        _hrtf_trajectory({"canonicalTrajectory": payload}, {})(times),
        path_model_from_dict(payload).positions(times),
    )


def test_launch_passes_current_hrtf_context(qtbot, monkeypatch):
    seen = {}

    class Designer:
        def __init__(self, *args, **kwargs):
            pass

        def set_render_context(self, context):
            seen.update(context)

        def exec_(self):
            return 0

    monkeypatch.setattr("src.ui.sam_path_panel.SamPath3DDialog", Designer)
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    panel.set_render_context_provider(
        lambda: {
            "params": {"hrtfAsset": "subject.sofa"},
            "sample_rate_hz": 48000,
            "origin_sample": 48000,
        }
    )
    panel.open_3d_designer()
    assert (
        seen["params"]["hrtfAsset"] == "subject.sofa" and seen["origin_sample"] == 48000
    )


def test_coverage_checks_modulated_height(qtbot):

    scene = _scene_with_lfo()
    scene["modulation"]["routes"] = [
        {
            "modulatorId": "lfo1",
            "targetId": "voice-a",
            "parameterPath": "transform.translation_z_m",
            "depth": 1,
            "minimum": 0,
            "maximum": 2,
        }
    ]
    dialog = SamPath3DDialog(
        _orbit_spec(),
        modulation={
            "source_id": "voice-a",
            "scene": lambda: scene,
            "commit": lambda s: None,
        },
    )
    qtbot.addWidget(dialog)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    measurements = np.c_[np.cos(angles), np.sin(angles), np.zeros(32)]
    dialog.set_hrtf_dataset(measurements)
    qtbot.waitUntil(lambda: dialog._coverage_future is None, timeout=10000)
    assert "elevation" in dialog.coverage_label.text().lower()
    assert np.any(dialog.views["top"]._coverage_bad)


def test_accept_commits_motion_once(qtbot):

    scene = _scene_with_lfo()
    commits = []
    dialog = SamPath3DDialog(
        _orbit_spec(),
        modulation={
            "source_id": "source.1",
            "scene": lambda: scene,
            "commit": commits.append,
        },
    )
    qtbot.addWidget(dialog)
    row = dialog._motion_rows["path.radius_m"]
    row["enable"].setChecked(True)
    row["high"].setValue(2)
    assert not commits
    expected = dialog._motion_scene()
    dialog.accept()
    assert commits == [expected]


def test_noop_preserves_transform_precision(qtbot):
    payload = spec()
    payload["transform"] = {
        "translationM": [0.123456789, 0, 0],
        "shear": [0.000011, 0, 0],
    }
    dialog = SamPath3DDialog(payload)
    qtbot.addWidget(dialog)
    assert (
        dialog.trajectory_spec()["transform"]["translationM"]
        == payload["transform"]["translationM"]
    )


@pytest.mark.parametrize("renderer", ["geometric", "hrtf"])
def test_new_path_render_is_block_invariant(renderer):
    from pathlib import Path
    from src.audio.sam_workbench.compat import render_sam2_voice

    payload = spec()
    payload.update(
        schemaVersion=3, constraints={"distance_m": 1.5}, interpolation="spherical"
    )
    payload["geometry"]["interpolation"] = "spherical"
    payload["traversal"]["durationS"] = 0.1
    params = {
        "rendererMode": renderer,
        "canonicalTrajectory": payload,
        "carrierFreq": 250,
        "amp": 0.1,
        "hrtfAsset": str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"),
        "hrtfOptions": {"propagationDelay": False},
    }
    whole = render_sam2_voice(0.1, 8000, params=params, block_size=800)
    blocked = render_sam2_voice(0.1, 8000, params=params, block_size=127)
    assert whole.shape == (800, 2) and np.all(np.isfinite(whole))
    np.testing.assert_allclose(whole, blocked, atol=2e-6)


def test_vertical_audition_is_explicit_and_keeps_voice_options(qtbot):
    from src.ui.sam_hrtf_lab import SamHrtfLab

    lab = SamHrtfLab()
    qtbot.addWidget(lab)
    lab.set_params({"carrierFreq": 123, "hrtfOptions": {"schemaVersion": 1}})
    before = copy.deepcopy(lab.params())
    assert not lab.vertical_check.isChecked()
    lab.vertical_check.setChecked(True)
    lab.vertical_material.setCurrentIndex(
        lab.vertical_material.findData("carrier_anchor")
    )
    assert lab.audition_options()["verticalMaterial"] == "carrier_anchor"
    assert lab.audition_options()["carrierFreq"] == 123
    assert lab.params() == before


def test_vertical_audition_uses_same_arc_for_comparison(monkeypatch):
    from pathlib import Path
    from src.ui.sam_hrtf_lab import AuditionWorker
    import src.audio.sam_workbench.render.hrtf as renderer

    captures = []
    monkeypatch.setattr(
        renderer,
        "render_spatial_hrtf",
        lambda mono, path, *args, **kwargs: captures.append((mono.copy(), path.copy()))
        or np.vstack((mono, mono)),
    )
    # The synthetic grid is deliberately sparse; isolate route equivalence from coverage policy.
    from src.audio.sam_workbench.hrtf.coverage import CoverageReport

    monkeypatch.setattr(
        "src.audio.sam_workbench.hrtf.coverage.assess_path_coverage",
        lambda *a, **kw: CoverageReport(),
    )
    asset = str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa")
    for material in ("broadband", "carrier", "carrier_anchor"):
        AuditionWorker(
            asset,
            (0, 0),
            "pink_noise_burst",
            {
                "sampleRateHz": 8000,
                "verticalCheck": True,
                "verticalMaterial": material,
                "carrierFreq": 250,
            },
            duration_s=0.1,
        ).render()
    for mono, path in captures:
        assert np.all(np.isfinite(mono))
        np.testing.assert_allclose(
            np.linalg.norm(path, axis=1), np.linalg.norm(path[0])
        )
        assert path[-1, 2] > path[0, 2]
    np.testing.assert_array_equal(captures[0][1], captures[2][1])
    assert not np.array_equal(captures[1][0], captures[2][0])


# --- opening the designer on a profile with nothing drawn in it --------------


UNDRAWN_PROFILES = {
    "no points key": {"schemaVersion": 2},
    "points empty": {"schemaVersion": 2, "points": []},
    "a single point": {"points": [[10.0, 20.0]]},
    "two identical points": {"points": [[5.0, 5.0], [5.0, 5.0]]},
    "entries that are not coordinates": {"points": ["x", None, [1]]},
}


@pytest.mark.parametrize("label", sorted(UNDRAWN_PROFILES))
def test_a_profile_with_no_path_in_it_still_opens_the_designer(qtbot, monkeypatch, label):
    """A 2-D profile created but not drawn used to make the button raise.

    ``promote_profile_to_trajectory`` refuses a profile describing no path,
    which is right, but the panel asked for the promotion whenever a profile
    existed at all. The ValueError reached the top, so a source whose profile
    had been started and left empty could not be given a 3-D path at all.
    """

    seen = []

    class Designer:
        def __init__(self, initial, *args, **kwargs):
            seen.append(copy.deepcopy(initial))

        def set_render_context(self, context):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr("src.ui.sam_path_panel.SamPath3DDialog", Designer)
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    panel.set_params({"customPathProfile": UNDRAWN_PROFILES[label]})

    panel.open_3d_designer()

    assert len(seen) == 1, "the designer must open rather than raise"
    # Nothing to carry over, so the designer starts on its own default path.
    assert not seen[0]


def test_a_profile_that_does_describe_a_path_is_still_promoted(qtbot, monkeypatch):
    """The guard must not cost the promotion it was added to protect."""

    from src.audio.sam_workbench.trajectory.legacy_paths import (
        promote_profile_to_trajectory,
    )

    seen = []

    class Designer:
        def __init__(self, initial, *args, **kwargs):
            seen.append(copy.deepcopy(initial))

        def set_render_context(self, context):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr("src.ui.sam_path_panel.SamPath3DDialog", Designer)
    panel = SamPathPanel()
    qtbot.addWidget(panel)
    profile = {"points": [[0.0, 0.0], [100.0, 50.0], [40.0, 120.0]], "closedLoop": True}
    panel.set_params({"customPathProfile": profile})

    panel.open_3d_designer()

    assert seen == [promote_profile_to_trajectory(profile)]


@pytest.mark.parametrize("label", sorted(UNDRAWN_PROFILES))
def test_promotability_is_decided_by_the_evaluator_not_by_counting(label):
    """Two of these carry two or more entries and still describe no path, so a
    caller that counted points would let them through and get the exception it
    was trying to avoid."""

    from src.audio.sam_workbench.trajectory.legacy_paths import (
        legacy_profile_is_promotable,
        promote_profile_to_trajectory,
    )

    profile = UNDRAWN_PROFILES[label]
    assert legacy_profile_is_promotable(profile) is False
    with pytest.raises(ValueError):
        promote_profile_to_trajectory(profile)


@pytest.mark.parametrize("profile", [None, "nonsense", 42, {"points": "not a list"}])
def test_an_unreadable_profile_is_simply_not_promotable(profile):
    from src.audio.sam_workbench.trajectory.legacy_paths import (
        legacy_profile_is_promotable,
    )

    assert legacy_profile_is_promotable(profile) is False


def test_a_drawable_profile_is_promotable():
    from src.audio.sam_workbench.trajectory.legacy_paths import (
        legacy_profile_is_promotable,
    )

    assert legacy_profile_is_promotable({"points": [[0.0, 0.0], [100.0, 50.0]]}) is True

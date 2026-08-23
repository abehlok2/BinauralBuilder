"""Scene automation of path parameters, through both front doors.

The rules this file protects:

* a modulation route targeting ``path.radiusM`` reaches the rendered motion -
  the compiled plan binds it and the compatibility bridge binds it the same
  way, so preview and export cannot disagree;
* reserved names never leak into generator controls, where they would be
  silently ignored;
* stage overrides set the parameter's absolute value, blending from the stored
  base, exactly as they do for every other automatable parameter;
* readiness says when motion is held at its limit and when constant-speed
  traversal has given way.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.plan import plan_from_track
from src.audio.sam_workbench.readiness import assess_readiness
from src.audio.sam_workbench.render.executor import execute_plan, render_source_window
from src.audio.sam_workbench.scene_state import empty_sam_scene
from src.audio.sam_workbench.stages import ParameterBinding, StageConfig, Timeline
from src.audio.sam_workbench.trajectory import ModulatedPath

RATE = 44_100


def _orbit_payload():
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


def _project(routes=(), extra_stages=(), radius_value=1.5, include_stage=True, lfo_hz=0.25):
    """A one-step track whose single voice orbits, plus its scene edits."""

    scene = empty_sam_scene()
    if include_stage:
        scene["stages"] = Timeline(
            stages=(
                StageConfig(
                    id="base-radius",
                    name="Base orbit radius",
                    start_s=0.0,
                    duration_s=8.0,
                    parameter_overrides=(
                        ParameterBinding("source.1", "path.radiusM", float(radius_value)),
                    ),
                ),
            )
        ).describe()
    if routes:
        scene["modulators"].append(
            {"id": "lfo1", "waveform": "sine", "rateHz": lfo_hz, "phaseDeg": 0.0, "seed": 0}
        )
        for route in routes:
            scene["modulation"]["routes"].append({"modulatorId": "lfo1", **route})

    voice = {
        "synth_function_name": "spatial_angle_modulation_sam2",
        "description": "orbit",
        "params": {
            "rendererMode": "geometric",
            "carrierFreq": 220.0,
            "amp": 0.4,
            "canonicalTrajectory": _orbit_payload(),
        },
    }
    track = {
        "global_settings": {"sample_rate": RATE},
        "steps": [{"start": 0.0, "duration": 2.0, "voices": [voice]}],
        "sam_scene": scene,
    }
    return track, scene, voice


# --- compilation -------------------------------------------------------------


def test_scene_routes_bind_to_the_path_and_not_the_generator():
    track, _scene, _voice = _project(routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.4}])
    plan = plan_from_track(track)

    source = plan.source("source.1")
    assert isinstance(source.trajectory, ModulatedPath)
    assert source.trajectory.parameter_paths() == ("path.radius_m",)
    assert source.controls == {}
    assert not any(
        path.startswith(("path.", "transform.")) for path in source.controls
    )


def test_a_route_to_a_field_this_geometry_lacks_is_reported_and_dropped():
    track, _scene, _voice = _project(
        routes=[
            {"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.4},
            {"targetId": "source.1", "parameterPath": "path.major_radius_m", "depth": 0.2},
        ]
    )
    plan = plan_from_track(track)
    messages = [issue.message for issue in plan.warnings]
    assert any("not a parameter" in message for message in messages)
    source = plan.source("source.1")
    assert source.trajectory.parameter_paths() == ("path.radius_m",)


def test_stage_overrides_set_the_absolute_radius():
    track, _scene, _voice = _project(radius_value=2.8)
    plan = plan_from_track(track)
    source = plan.source("source.1")

    times = np.array([0.5, 1.0, 1.5])
    spherical = source.trajectory.spherical(times)
    assert spherical[:, 2] == pytest.approx(2.8, abs=1e-9)


def test_an_unbound_track_keeps_a_plain_path_model():
    from src.audio.sam_workbench.trajectory import PathModel

    track, _scene, _voice = _project(include_stage=False)
    plan = plan_from_track(track)
    assert isinstance(plan.source("source.1").trajectory, PathModel)


# --- rendering ---------------------------------------------------------------


def test_render_is_invariant_under_blocking():
    track, _scene, _voice = _project(routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.4}])
    plan = plan_from_track(track)
    frames = int(1.6 * RATE)

    whole = execute_plan(plan.for_window(0, frames), apply_routing=False, block_size=4096).audio
    middle = frames // 2
    first = execute_plan(plan.for_window(0, middle), apply_routing=False, block_size=777).audio
    second = execute_plan(plan.for_window(middle, frames - middle), apply_routing=False, block_size=777).audio
    blocked = np.concatenate((first, second), axis=1)

    assert whole.shape == blocked.shape
    assert np.allclose(whole, blocked, rtol=1e-5, atol=1e-7)


def test_compat_door_matches_plan_door():
    track, scene, voice = _project(routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.4}])
    plan = plan_from_track(track)
    source = plan.source("source.1")
    frames = int(1.2 * RATE)

    planned = render_source_window(source, sample_rate_hz=RATE, start_sample=0, frames=frames)
    bridged = render_sam2_voice(
        float(frames) / RATE,
        RATE,
        params=dict(voice["params"]),
        sam_scene=scene,
        source_id="source.1",
        apply_routing=False,
    )

    assert bridged.shape == (frames, 2)
    assert np.allclose(planned, np.asarray(bridged).T, rtol=1e-4, atol=1e-6)


def test_the_bound_motion_actually_reaches_the_audio():
    still, _scene_still, _v = _project()
    moving, _scene_move, _v2 = _project(routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.9}])
    frames = int(1.0 * RATE)

    static_peak = np.abs(execute_plan(plan_from_track(still).for_window(0, frames), apply_routing=False).audio)
    moving_peak = np.abs(execute_plan(plan_from_track(moving).for_window(0, frames), apply_routing=False).audio)
    # A breathing radius changes level and delay; a frozen one does not move.
    assert not np.allclose(static_peak, moving_peak, rtol=1e-3, atol=1e-5)


# --- readiness ---------------------------------------------------------------


def test_readiness_reports_speed_law_fallback():
    _track_unused, scene, voice = _project(routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 0.4}])
    report = assess_readiness(voice["params"], scene=scene, sample_rate_hz=RATE, source_id="source.1")
    assert any("parameter speed" in issue.message for issue in report.warnings)


def test_readiness_reports_motion_held_at_its_limit():
    _track_unused, scene, voice = _project(
        # A scene modulator swings 0..1, so reversed polarity with depth 3
        # pulls the radius from its 1.5 m base down through the floor.
        routes=[{"targetId": "source.1", "parameterPath": "path.radiusM", "depth": 3.0, "polarity": -1}],
        lfo_hz=0.5,
    )
    report = assess_readiness(voice["params"], scene=scene, sample_rate_hz=RATE, source_id="source.1")
    assert any("documented" in issue.message and "limit" in issue.message for issue in report.warnings)


def test_readiness_errors_on_a_route_the_geometry_cannot_follow():
    _track_unused, scene, voice = _project(
        routes=[{"targetId": "source.1", "parameterPath": "path.swing_deg", "depth": 10.0}],
        include_stage=False,
    )
    report = assess_readiness(voice["params"], scene=scene, sample_rate_hz=RATE, source_id="source.1")
    assert any("not a parameter" in issue.message for issue in report.errors)

"""The compiled scene plan: one description of what is to be rendered.

The rules this file protects:

* both front doors - a BinauralBuilder track and a standalone Project - compile
  to the same type, so nothing downstream needs to know which one a project
  came through;
* a plan is immutable and derived, so compiling twice is always the same answer
  and nothing can edit one behind a renderer's back;
* automation is compiled into functions of the absolute sample index, not
  resolved once per block, so where the timeline is cut cannot change a value;
* a trajectory is carried as its PathModel and never materialized at the audio
  rate;
* renderer configuration, assets, latency and tail come from the registry
  rather than being recomputed here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.audio.sam_workbench.model import Project, Source
from src.audio.sam_workbench.modulation import ModulationMatrix, ModulationRoute
from src.audio.sam_workbench.plan import (
    CompiledControl,
    CompiledScenePlan,
    intersect_window,
    plan_from_project,
    plan_from_track,
)
from src.audio.sam_workbench.scene_state import empty_sam_scene
from src.audio.sam_workbench.stages import ParameterBinding, StageConfig, Timeline
from src.audio.sam_workbench.trajectory import PathModel

RATE = 44100
DOME = {
    "geometry": {"type": "dome_traversal", "parameters": {"turns": 2}},
    "traversal": {"durationS": 2.0},
}


def _voice(name, mode="abstract_pm", **extra):
    params = {
        "amp": 0.5,
        "carrierFreq": 440.0,
        "modFreq": 4.0,
        "duration": 2.0,
        "rendererMode": mode,
    }
    params.update(extra)
    return {
        "synth_function_name": "spatial_angle_modulation_sam2",
        "description": name,
        "params": params,
    }


def _automated_scene():
    scene = empty_sam_scene()
    scene["modulators"] = [{"id": "lfo", "rateHz": 0.5}]
    scene["stages"] = Timeline(
        stages=(
            StageConfig(
                id="st",
                start_s=0.0,
                duration_s=2.0,
                transition_in_s=0.5,
                parameter_overrides=(
                    ParameterBinding(target_id="source.1", parameter_path="modFreq", value=9.0),
                    ParameterBinding(target_id="source.1", parameter_path="amp", value=0.4),
                ),
            ),
        )
    ).describe()
    scene["modulation"] = ModulationMatrix(
        routes=(
            ModulationRoute(
                modulator_id="lfo", target_id="source.1", parameter_path="carrierFreq", depth=50.0
            ),
        )
    ).describe()
    scene["environment"] = {"speedOfSoundMS": 343.0}
    scene["routing"]["buses"] = [{"id": "master", "gainDb": -3.0}]
    scene["routing"]["sources"] = [{"sourceId": "source.1", "busId": "master", "gainDb": -6.0}]
    return scene


def _track(scene=None, voices=None):
    track = {
        "global_settings": {"sample_rate": RATE, "random_seed": 7},
        "steps": [{"duration": 2.0, "voices": voices or [_voice("Drone")]}],
    }
    if scene is not None:
        track["sam_scene"] = scene
    return track


# --- the window intersection is shared --------------------------------------


def test_the_plan_and_the_scene_renderer_share_one_intersection():
    from src.audio.sam_workbench.conventions import intersect_window as canonical

    assert intersect_window is canonical


@pytest.mark.parametrize(
    "window_start, window_frames, expected",
    [
        (0, 100, None),            # entirely before
        (50, 200, (150, 0, 50)),   # overlapping the beginning
        (220, 30, (0, 20, 30)),    # inside
        (250, 200, (0, 50, 50)),   # overlapping the end
        (400, 100, None),          # entirely after
        (300, 100, None),          # exactly at the end
    ],
)
def test_the_intersection_covers_every_window_position(window_start, window_frames, expected):
    assert intersect_window(200, 300, window_start, window_frames) == expected


def test_a_source_with_no_end_runs_as_long_as_the_window_asks():
    assert intersect_window(100, None, 0, 500) == (100, 0, 400)


# --- both front doors -------------------------------------------------------


def test_a_project_compiles_to_a_plan():
    plan = plan_from_project(Project(sources=(Source(id="s", start_s=1.0, duration_s=1.0),)))
    assert isinstance(plan, CompiledScenePlan)
    assert plan.sample_rate_hz == RATE
    assert [entry.source_id for entry in plan.sources] == ["s"]
    assert plan.diagnostics["origin"] == "project"


def test_a_track_compiles_to_a_plan():
    plan = plan_from_track(_track())
    assert isinstance(plan, CompiledScenePlan)
    assert plan.diagnostics["origin"] == "track"
    assert [entry.name for entry in plan.sources] == ["Drone"]


def test_both_front_doors_place_the_same_source_at_the_same_samples():
    """A source one second in, one second long, is that in either document."""

    project = plan_from_project(
        Project(sources=(Source(id="s", start_s=1.0, duration_s=1.0),))
    )
    track = plan_from_track(
        {
            "global_settings": {"sample_rate": RATE},
            "steps": [{"start": 1.0, "duration": 1.0, "voices": [_voice("s")]}],
        }
    )
    assert track.sources[0].start_sample == project.sources[0].start_sample == RATE
    assert track.sources[0].end_sample == project.sources[0].end_sample == 2 * RATE


def test_a_source_window_matches_the_shared_intersection():
    plan = plan_from_project(Project(sources=(Source(id="s", start_s=1.0, duration_s=1.0),)))
    source = plan.sources[0]
    assert source.window(0, 3 * RATE) == (RATE, 0, RATE)
    assert source.window(3 * RATE, RATE) is None


# --- immutability and derivation --------------------------------------------


def test_a_plan_cannot_be_edited():
    plan = plan_from_track(_track())
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.sources[0].source_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.frames = 0


def test_compiling_the_same_track_twice_gives_the_same_plan():
    track = _track(_automated_scene())
    first, second = plan_from_track(track), plan_from_track(track)
    assert first.describe() == second.describe()


def test_restricting_a_plan_to_a_window_moves_only_the_render_range():
    plan = plan_from_track(_track())
    moved = plan.for_window(RATE, RATE // 2)
    assert (moved.start_sample, moved.frames) == (RATE, RATE // 2)
    assert [entry.start_sample for entry in moved.sources] == [
        entry.start_sample for entry in plan.sources
    ]


def test_active_sources_are_the_ones_that_overlap_the_window():
    plan = plan_from_project(
        Project(
            sources=(
                Source(id="early", start_s=0.0, duration_s=1.0),
                Source(id="late", start_s=5.0, duration_s=1.0),
            )
        )
    )
    assert [entry.source_id for entry in plan.active_sources(0, RATE)] == ["early"]
    assert [entry.source_id for entry in plan.active_sources(5 * RATE, RATE)] == ["late"]


# --- renderer configuration comes from the registry -------------------------


def test_the_renderer_configuration_is_validated_and_defaulted():
    plan = plan_from_track(
        _track(voices=[_voice("H", "hrtf", hrtfAsset="a.sofa", hrtfAssetHash="abc")])
    )
    config = plan.sources[0].renderer_config
    assert config["interpolation"] == "nearest"
    assert config["delayPolicy"] == "bake_delay_into_ir"


def test_a_missing_required_asset_becomes_a_plan_error():
    plan = plan_from_track(_track(voices=[_voice("H", "hrtf")]))
    assert not plan.is_renderable
    assert any("sofa asset" in issue.message for issue in plan.errors)


def test_an_unknown_renderer_is_reported_rather_than_silently_substituted():
    plan = plan_from_track(_track(voices=[_voice("X", "nonsense")]))
    assert any(issue.path.endswith("rendererMode") for issue in plan.errors)


def test_assets_are_collected_with_their_hashes():
    plan = plan_from_track(
        _track(voices=[_voice("H", "hrtf", hrtfAsset="a.sofa", hrtfAssetHash="deadbeef")])
    )
    assets = plan.assets()
    assert [(entry.kind, entry.path, entry.sha256) for entry in assets] == [
        ("sofa", "a.sofa", "deadbeef")
    ]


def test_the_same_asset_used_twice_is_listed_once():
    plan = plan_from_track(
        _track(
            voices=[
                _voice("A", "hrtf", hrtfAsset="a.sofa"),
                _voice("B", "hrtf", hrtfAsset="a.sofa"),
            ]
        )
    )
    assert len(plan.assets()) == 1


def test_latency_and_tail_come_from_the_renderers():
    plan = plan_from_track(
        _track(voices=[_voice("H", "hrtf", hrtfAsset="a.sofa"), _voice("D")])
    )
    # The HRTF source rings out; the abstract one does not, and the plan takes
    # the longest so a render does not cut the tail off.
    assert plan.tail_samples > 0
    assert plan.tail_samples == max(entry.tail_samples for entry in plan.sources)


# --- the trajectory system is reused, not rebuilt ----------------------------


def test_a_stored_path_is_carried_as_the_canonical_path_model():
    plan = plan_from_track(
        _track(voices=[_voice("H", "hrtf", hrtfAsset="a.sofa", canonicalTrajectory=DOME)])
    )
    assert isinstance(plan.sources[0].trajectory, PathModel)


def test_the_path_is_not_materialized_at_the_audio_rate():
    """A path is a function of time; sampling it per audio frame is a cost."""

    plan = plan_from_track(
        _track(voices=[_voice("H", "hrtf", hrtfAsset="a.sofa", canonicalTrajectory=DOME)])
    )
    model = plan.sources[0].trajectory
    assert not any(
        isinstance(value, np.ndarray) and value.size > 4096
        for value in vars(model).values()
    )
    # It still answers at whatever rate a renderer asks for.
    assert model.positions(np.linspace(0.0, 2.0, 32)).shape == (32, 3)


def test_a_source_without_a_path_has_no_trajectory():
    assert plan_from_track(_track()).sources[0].trajectory is None


def test_an_unreadable_path_is_reported_rather_than_replaced_by_a_default():
    """Substituting a default would move the source somewhere nobody asked for."""

    plan = plan_from_track(
        _track(
            voices=[
                _voice("H", "hrtf", hrtfAsset="a.sofa", canonicalTrajectory={"geometry": {"type": "no_such_shape"}})
            ]
        )
    )
    assert plan.sources[0].trajectory is None
    assert any("unreadable path" in issue.message for issue in plan.errors)


# --- compiled automation ----------------------------------------------------


def test_automated_parameters_become_controls():
    plan = plan_from_track(_track(_automated_scene()))
    assert sorted(plan.sources[0].controls) == ["carrierFreq", "modFreq"]


def test_an_automated_parameter_is_removed_from_the_constant_generator():
    """Otherwise a stale constant sits beside the control that supersedes it."""

    source = plan_from_track(_track(_automated_scene())).sources[0]
    assert "modFreq" not in source.generator
    assert source.generator["duration"] == 2.0


def test_a_control_follows_the_stage_transition():
    source = plan_from_track(_track(_automated_scene())).sources[0]
    control = source.controls["modFreq"]
    assert control.value_at(0) == pytest.approx(4.0)          # the base value
    assert control.value_at(RATE // 2) == pytest.approx(9.0)  # the stage value


def test_gain_is_always_present_even_without_a_scene():
    source = plan_from_track(_track()).sources[0]
    assert source.gain.is_constant
    assert source.gain.value_at(0) == pytest.approx(1.0)


def test_gain_carries_the_stage_and_the_routing_together():
    source = plan_from_track(_track(_automated_scene())).sources[0]
    # 0.4 from the stage, -6 dB source and -3 dB bus from the routing.
    expected = 0.4 * 10.0 ** (-6.0 / 20.0) * 10.0 ** (-3.0 / 20.0)
    assert source.gain.value_at(RATE) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "cuts", [[0, 6000], [0, 3000, 6000], [0, 13, 777, 5995, 6000]],
    ids=["whole", "halves", "awkward"],
)
def test_a_compiled_control_does_not_depend_on_block_partitioning(cuts):
    """The defect this replaces resolved each parameter once per chunk."""

    source = plan_from_track(_track(_automated_scene())).sources[0]
    for control in (*source.controls.values(), source.gain):
        whole = control.at(0, 6000)
        joined = np.concatenate(
            [control.at(cuts[i], cuts[i + 1] - cuts[i]) for i in range(len(cuts) - 1)]
        )
        assert np.array_equal(joined, whole), control.path


def test_automating_a_parameter_that_cannot_be_automated_is_reported():
    scene = empty_sam_scene()
    scene["stages"] = Timeline(
        stages=(
            StageConfig(
                id="st",
                duration_s=2.0,
                parameter_overrides=(
                    ParameterBinding(target_id="source.1", parameter_path="pathShape", value=1.0),
                ),
            ),
        )
    ).describe()
    plan = plan_from_track(_track(scene))
    assert "pathShape" not in plan.sources[0].controls
    assert any("not an automatable parameter" in issue.message for issue in plan.warnings)


def test_a_constant_control_answers_without_evaluating_anything():
    control = CompiledControl("x", constant=2.5)
    assert control.is_constant
    assert np.array_equal(control.at(0, 3), np.full(3, 2.5))


def test_a_control_refuses_to_return_the_wrong_length():
    control = CompiledControl("x", _evaluate=lambda start, frames: np.zeros(frames + 1))
    with pytest.raises(ValueError, match="expected"):
        control.at(0, 4)


# --- scene sections and seeds -----------------------------------------------


def test_the_plan_carries_routing_and_bands_for_the_signal_path():
    scene = _automated_scene()
    scene["routing"]["bands"] = {"crossoversHz": [300.0]}
    plan = plan_from_track(_track(scene))
    assert plan.routing["source.1"]["gainDb"] == -6.0
    assert plan.band_routing == {"crossoversHz": [300.0]}
    assert [bus["id"] for bus in plan.buses] == ["master"]


def test_the_plan_carries_the_environment_and_experiment_sections():
    plan = plan_from_track(_track(_automated_scene()))
    assert plan.environment == {"speedOfSoundMS": 343.0}
    assert plan.experiment == {}


def test_per_source_seeds_are_derived_from_the_identifier():
    """Adding a source must not move another source's noise."""

    one = plan_from_track(_track(voices=[_voice("A")]))
    two = plan_from_track(_track(voices=[_voice("A"), _voice("B")]))
    assert two.sources[0].seed == one.sources[0].seed
    assert two.sources[1].seed != two.sources[0].seed


def test_the_seed_follows_the_project_seed():
    track = _track()
    first = plan_from_track(track).sources[0].seed
    track["global_settings"]["random_seed"] = 99
    assert plan_from_track(track).sources[0].seed != first


def test_a_plan_describes_itself_for_a_manifest():
    described = plan_from_track(_track(_automated_scene())).describe()
    assert set(described) >= {
        "sampleRateHz", "startSample", "frames", "latencySamples", "tailSamples",
        "seed", "renderers", "sources", "buses", "bandRouting", "assets", "warnings",
    }


def test_a_track_without_a_scene_still_compiles():
    plan = plan_from_track(_track())
    assert plan.is_renderable
    assert plan.sources[0].controls == {}

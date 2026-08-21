"""Stages over the existing steps, the modulation matrix, and source coupling.

The rules this file protects:

* a stage is a label over existing steps, so the two views cannot disagree
  about when anything happens;
* everything is a pure function of time, so a staged render is deterministic
  and can be chunked;
* bindings address stable identifiers, never display names or list indices;
* a modulation route that would close a loop is refused before it is accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.modulation import (
    ModulationCycleError,
    ModulationMatrix,
    ModulationRoute,
    search_parameters,
)
from src.audio.sam_workbench.stages import (
    CURVES,
    ParameterBinding,
    StageConfig,
    Timeline,
    apply_curve,
    stages_from_steps,
)
from src.audio.sam_workbench.trajectory.coupling import (
    COUPLING_MODES,
    CouplingSpec,
    couple_positions,
)

STEPS = [
    {"duration": 60.0, "description": "settle"},
    {"duration": 120.0, "description": "build"},
    {"duration": 300.0, "description": "hold"},
    {"duration": 90.0, "description": "return"},
]


def azimuth_of(vector) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


# --- stages over the existing steps -----------------------------------------


def test_every_step_becomes_a_stage_with_the_step_s_own_timing():
    timeline = stages_from_steps(STEPS)

    assert [stage.name for stage in timeline.stages] == ["settle", "build", "hold", "return"]
    assert [stage.start_s for stage in timeline.stages] == [0.0, 60.0, 180.0, 480.0]
    # The stage view cannot drift from the step view, because it is derived.
    assert timeline.duration_s == sum(step["duration"] for step in STEPS)


def test_contiguous_steps_can_be_grouped_into_one_named_stage():
    timeline = stages_from_steps(
        STEPS, groups=[(0, 1), (2,), (3,)], names=["preparation", "induction", "return"]
    )

    assert [stage.name for stage in timeline.stages] == ["preparation", "induction", "return"]
    assert timeline.stages[0].duration_s == 180.0
    assert timeline.stages[1].start_s == 180.0


def test_a_stage_may_not_cover_a_gap_in_the_steps():
    """A stage names a span, so its steps have to be next to each other."""

    with pytest.raises(ValueError):
        stages_from_steps(STEPS, groups=[(0, 2)])


def test_a_group_beyond_the_steps_is_rejected():
    with pytest.raises(ValueError):
        stages_from_steps(STEPS, groups=[(0, 1, 2, 3, 4)])


def test_grouping_gives_the_inner_edges_transitions():
    timeline = stages_from_steps(STEPS, groups=[(0,), (1,), (2, 3)], transition_s=10.0)

    assert timeline.stages[0].transition_in_s == 0.0  # nothing to fade from
    assert timeline.stages[0].transition_out_s == 10.0
    assert timeline.stages[-1].transition_out_s == 0.0  # nothing to fade to


# --- stage weights ----------------------------------------------------------


def test_a_stage_weight_is_a_pure_function_of_time():
    stage = StageConfig(id="a", start_s=10.0, duration_s=20.0, transition_in_s=5.0, transition_out_s=5.0)

    assert stage.weight_at(5.0) == 0.0
    assert stage.weight_at(12.5) == pytest.approx(0.5)
    assert stage.weight_at(20.0) == pytest.approx(1.0)
    assert stage.weight_at(27.5) == pytest.approx(0.5)
    assert stage.weight_at(35.0) == 0.0

    # Evaluating a span at once must match evaluating each instant alone, or a
    # chunked render would not match a whole one.
    times = np.linspace(0.0, 40.0, 41)
    assert stage.weight_at(times) == pytest.approx([stage.weight_at(float(t)) for t in times])


def test_transitions_longer_than_the_stage_are_scaled_rather_than_overshooting():
    stage = StageConfig(id="t", duration_s=4.0, transition_in_s=10.0, transition_out_s=10.0)
    weights = stage.weight_at(np.linspace(0.0, 4.0, 33))

    assert np.max(weights) <= 1.0 + 1e-12
    assert np.all(weights >= 0.0)


def test_a_stage_with_no_duration_never_applies():
    stage = StageConfig(id="empty", start_s=5.0, duration_s=0.0)
    assert stage.weight_at(5.0) == 0.0

    issues = Timeline((stage,)).validate()
    assert any("never applies" in issue.message for issue in issues)


@pytest.mark.parametrize("curve", CURVES)
def test_every_curve_runs_from_zero_to_one(curve):
    assert apply_curve(0.0, curve) == pytest.approx(0.0)
    assert apply_curve(1.0, curve) == pytest.approx(1.0)
    # Monotone, so a fade never reverses part way through.
    values = [float(apply_curve(p, curve)) for p in np.linspace(0.0, 1.0, 21)]
    assert values == sorted(values)


def test_hold_switches_rather_than_sweeps():
    assert apply_curve(0.99, "hold") == 0.0
    assert apply_curve(1.0, "hold") == 1.0


def test_an_unknown_curve_is_rejected():
    with pytest.raises(ValueError):
        apply_curve(0.5, "sideways")


# --- bindings ---------------------------------------------------------------


def test_a_binding_needs_a_stable_identity():
    """Never a display name, never a list index."""

    with pytest.raises(ValueError):
        ParameterBinding("", "sam.beatFreq", 4.0)
    with pytest.raises(ValueError):
        ParameterBinding("src1", "", 4.0)
    with pytest.raises(ValueError):
        ParameterBinding("src1", "sam.beatFreq", float("nan"))


def test_overlapping_stages_blend_their_bindings():
    early = StageConfig(
        id="a",
        duration_s=20.0,
        transition_out_s=10.0,
        parameter_overrides=(ParameterBinding("src1", "sam.beatFreq", 4.0),),
    )
    late = StageConfig(
        id="b",
        start_s=10.0,
        duration_s=20.0,
        transition_in_s=10.0,
        parameter_overrides=(ParameterBinding("src1", "sam.beatFreq", 10.0),),
    )
    timeline = Timeline((early, late))
    key = ("src1", "sam.beatFreq")

    assert timeline.resolve(5.0)[key] == pytest.approx(4.0)
    assert timeline.resolve(15.0)[key] == pytest.approx(7.0)  # halfway between
    assert timeline.resolve(25.0)[key] == pytest.approx(10.0)


def test_an_overlap_is_reported_as_a_warning_not_an_error():
    timeline = Timeline(
        (
            StageConfig(id="a", duration_s=20.0),
            StageConfig(id="b", start_s=10.0, duration_s=20.0),
        )
    )
    issues = timeline.validate()

    assert issues
    assert all(issue.severity == "warning" for issue in issues)
    assert any("overlap" in issue.message for issue in issues)


def test_a_duplicate_stage_id_is_an_error():
    timeline = Timeline((StageConfig(id="a", duration_s=1.0), StageConfig(id="a", duration_s=1.0)))
    assert any(issue.severity == "error" for issue in timeline.validate())


def test_a_partly_faded_stage_blends_from_the_default():
    """A fade-in moves a value away from where it was, not from zero."""

    stage = StageConfig(
        id="a",
        duration_s=20.0,
        transition_in_s=10.0,
        parameter_overrides=(ParameterBinding("src1", "sam.beatFreq", 10.0),),
    )
    timeline = Timeline((stage,))
    key = ("src1", "sam.beatFreq")

    resolved = timeline.resolve(5.0, defaults={key: 2.0})
    assert 2.0 < resolved[key] < 10.0


def test_a_timeline_round_trips_through_a_mapping():
    timeline = stages_from_steps(STEPS, groups=[(0, 1), (2, 3)], transition_s=5.0)
    timeline = Timeline(
        (
            timeline.stages[0],
            StageConfig(
                id="x",
                name="named",
                start_s=180.0,
                duration_s=390.0,
                parameter_overrides=(ParameterBinding("src1", "sam.depth", 0.7, "smooth"),),
                notes="a note",
            ),
        )
    )
    restored = Timeline.from_mapping(timeline.describe())

    assert restored == timeline
    assert restored.stage("x").parameter_overrides[0].curve == "smooth"


# --- modulation matrix ------------------------------------------------------


def test_the_matrix_is_rows_of_modulators_over_columns_of_targets():
    matrix = (
        ModulationMatrix()
        .with_route(ModulationRoute("lfo1", "src1", "sam.beatFreq", depth=2.0))
        .with_route(ModulationRoute("lfo1", "src2", "sam.carrierFreq", depth=-30.0))
        .with_route(ModulationRoute("env1", "src1", "sam.depth", depth=0.5))
    )

    assert matrix.modulators == ("env1", "lfo1")
    assert matrix.targets == ("src1", "src2")
    assert sorted(matrix.as_grid()["lfo1"]) == ["src1.sam.beatFreq", "src2.sam.carrierFreq"]
    assert len(matrix.routes_from("lfo1")) == 2
    assert len(matrix.routes_for_target("src1")) == 2


def test_a_route_that_would_close_a_loop_is_refused_before_it_is_accepted():
    matrix = ModulationMatrix().with_route(ModulationRoute("a", "b", "p", depth=1.0))
    closing = ModulationRoute("b", "a", "p", depth=1.0)

    assert matrix.would_cycle(closing)
    with pytest.raises(ModulationCycleError) as caught:
        matrix.with_route(closing)
    assert caught.value.cycle[0] == caught.value.cycle[-1]
    # The matrix is unchanged: nothing was half-applied.
    assert len(matrix.routes) == 1


def test_a_long_cycle_is_found_and_named():
    matrix = (
        ModulationMatrix()
        .with_route(ModulationRoute("a", "b", "p", depth=1.0))
        .with_route(ModulationRoute("b", "c", "p", depth=1.0))
    )
    with pytest.raises(ModulationCycleError) as caught:
        matrix.with_route(ModulationRoute("c", "a", "p", depth=1.0))

    assert "a" in caught.value.cycle and "c" in caught.value.cycle


def test_self_modulation_is_a_cycle():
    assert ModulationMatrix().would_cycle(ModulationRoute("a", "a", "p", depth=1.0))


def test_a_disabled_route_cannot_close_a_loop():
    """It carries no value, so refusing it would block a parked setting."""

    matrix = ModulationMatrix().with_route(ModulationRoute("a", "b", "p", depth=1.0))
    parked = ModulationRoute("b", "a", "p", depth=1.0, enabled=False)

    assert not matrix.would_cycle(parked)
    assert len(matrix.with_route(parked).routes) == 2


def test_several_routes_on_one_parameter_add_up():
    matrix = (
        ModulationMatrix()
        .with_route(ModulationRoute("lfo1", "src1", "sam.beatFreq", depth=2.0))
        .with_route(ModulationRoute("env1", "src1", "sam.beatFreq", depth=1.0))
    )
    base = {("src1", "sam.beatFreq"): 4.0}

    assert matrix.evaluate({"lfo1": 0.0, "env1": 0.0}, base)[("src1", "sam.beatFreq")] == 4.0
    assert matrix.evaluate({"lfo1": 1.0, "env1": 1.0}, base)[("src1", "sam.beatFreq")] == 7.0


def test_polarity_flips_direction_without_a_negative_depth():
    positive = ModulationRoute("l", "t", "p", depth=2.0, polarity=1)
    negative = ModulationRoute("l", "t", "p", depth=2.0, polarity=-1)

    assert positive.apply(1.0, 4.0) == pytest.approx(6.0)
    assert negative.apply(1.0, 4.0) == pytest.approx(2.0)


def test_a_route_with_no_depth_does_nothing_and_says_so():
    matrix = ModulationMatrix(routes=(ModulationRoute("a", "b", "p", depth=0.0),))
    issues = matrix.validate()

    assert any(issue.severity == "warning" and "does nothing" in issue.message for issue in issues)
    assert matrix.evaluate({"a": 1.0}, {("b", "p"): 5.0})[("b", "p")] == 5.0


def test_an_invalid_route_is_rejected():
    with pytest.raises(ValueError):
        ModulationRoute("a", "b", "p", polarity=0)
    with pytest.raises(ValueError):
        ModulationRoute("a", "b", "p", curve="sideways")
    with pytest.raises(ValueError):
        ModulationRoute("", "b", "p")


def test_the_matrix_round_trips_and_a_route_can_be_removed():
    matrix = ModulationMatrix().with_route(ModulationRoute("a", "b", "p", depth=1.0))

    assert ModulationMatrix.from_mapping(matrix.describe()) == matrix
    assert matrix.without_route("a", "b", "p").routes == ()


# --- parameter search -------------------------------------------------------


def test_search_finds_parameters_by_name():
    hits = search_parameters("carrier")
    assert hits
    assert any("carrier" in entry.name.lower() for entry in hits)


def test_search_is_bounded_and_ignores_an_empty_query():
    assert search_parameters("") == ()
    assert len(search_parameters("a", limit=3)) <= 3


# --- source coupling --------------------------------------------------------


def test_every_coupling_mode_is_a_pure_function_of_the_leader():
    leader = np.array([1.0, 0.0, 0.0])
    own = np.array([0.0, 1.0, 0.0])

    for mode in COUPLING_MODES:
        spec = CouplingSpec(mode=mode, leader_id="" if mode == "independent" else "lead", distance_m=0.3)
        first = couple_positions(spec, leader, own=own, times_s=0.25)
        second = couple_positions(spec, leader, own=own, times_s=0.25)
        assert first == pytest.approx(second)
        assert np.all(np.isfinite(first))


def test_shared_mirrored_and_offset_place_the_follower():
    leader = np.array([1.0, 0.0, 0.0])
    left = np.array([0.0, 1.0, 0.0])

    assert couple_positions(CouplingSpec("shared", "lead"), leader) == pytest.approx(leader)
    assert azimuth_of(
        couple_positions(CouplingSpec("offset", "lead", angle_deg=90.0), leader)
    ) == pytest.approx(90.0)
    assert azimuth_of(
        couple_positions(CouplingSpec("mirrored", "lead"), left)
    ) == pytest.approx(-90.0)
    assert azimuth_of(
        couple_positions(CouplingSpec("phase_locked", "lead", angle_deg=180.0), leader)
    ) == pytest.approx(180.0)


def test_an_offset_can_push_out_along_the_radius():
    leader = np.array([1.0, 0.0, 0.0])
    pushed = couple_positions(CouplingSpec("offset", "lead", distance_m=0.5), leader)

    assert np.linalg.norm(pushed) == pytest.approx(1.5)


def test_orbit_circles_the_leader_at_its_own_rate():
    frames = 5
    times = np.linspace(0.0, 1.0, frames)
    leader = np.stack([np.ones(frames), np.zeros(frames), np.zeros(frames)], axis=1)
    orbit = couple_positions(
        CouplingSpec("orbit", "lead", distance_m=0.3, rate_hz=1.0), leader, times_s=times
    )

    # One revolution per second: a full turn brings it back where it started.
    assert orbit[0] == pytest.approx(orbit[-1], abs=1e-9)
    radii = np.linalg.norm(orbit - leader, axis=1)
    assert radii == pytest.approx(np.full(frames, 0.3))


def test_attraction_never_pulls_past_the_leader():
    """Overshooting would swap which side of the head the sources are on."""

    leader = np.array([1.0, 0.0, 0.0])
    close = np.array([1.05, 0.0, 0.0])
    pulled = couple_positions(CouplingSpec("attracted", "lead", distance_m=10.0), leader, own=close)

    assert pulled == pytest.approx(leader, abs=1e-9)


def test_repulsion_pushes_away_and_coincident_sources_still_separate():
    leader = np.array([1.0, 0.0, 0.0])
    away = couple_positions(
        CouplingSpec("repelled", "lead", distance_m=0.4), leader, own=np.array([0.0, 1.0, 0.0])
    )
    assert np.linalg.norm(away - leader) > np.linalg.norm(np.array([0.0, 1.0, 0.0]) - leader)

    # No direction to move along, so it must pick one rather than divide by zero.
    coincident = couple_positions(
        CouplingSpec("repelled", "lead", distance_m=0.4), leader, own=leader.copy()
    )
    assert np.all(np.isfinite(coincident))
    assert np.linalg.norm(coincident - leader) == pytest.approx(0.4)


def test_a_coupled_path_can_be_rendered_in_chunks():
    """Which is what a multi-source render needs in order to be deterministic."""

    frames = 9
    times = np.linspace(0.0, 2.0, frames)
    leader = np.stack([np.ones(frames), np.zeros(frames), np.zeros(frames)], axis=1)
    spec = CouplingSpec("orbit", "lead", distance_m=0.3, rate_hz=0.5)

    whole = couple_positions(spec, leader, times_s=times)
    chunked = np.concatenate(
        [
            couple_positions(spec, leader[:4], times_s=times[:4]),
            couple_positions(spec, leader[4:], times_s=times[4:]),
        ]
    )
    assert whole == pytest.approx(chunked)


def test_a_follower_needs_a_leader_and_an_independent_source_needs_a_path():
    with pytest.raises(ValueError):
        CouplingSpec("shared", "")
    with pytest.raises(ValueError):
        CouplingSpec("nonsense", "lead")
    with pytest.raises(ValueError):
        couple_positions(CouplingSpec(), np.array([1.0, 0.0, 0.0]))


def test_coupling_round_trips():
    spec = CouplingSpec("orbit", "lead", angle_deg=30.0, distance_m=0.3, rate_hz=2.0, strength=0.5)
    assert CouplingSpec.from_mapping(spec.describe()) == spec

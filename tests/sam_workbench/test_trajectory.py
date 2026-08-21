"""Canonical trajectories: geometry, traversal, transforms, and the legacy bridge.

The properties that matter here are the ones a listener can hear or a user can
lose: a path that is the shape it claims to be, a timing law that does not
restart at a block boundary, and a legacy pixel path that maps into metres
reproducibly and comes back unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.conventions import cartesian_to_spherical_deg
from src.audio.sam_workbench.trajectory import (
    PROFILE_VERSION,
    ArcGeometry,
    CircleGeometry,
    EllipseGeometry,
    HelixGeometry,
    LegacyPathTransform,
    LineGeometry,
    LissajousGeometry,
    ListenerFrame,
    PointCloudGeometry,
    PolylineGeometry,
    SpiralGeometry,
    SplineGeometry,
    TrajectorySpec,
    TransformSpec,
    TraversalSpec,
    constant_speed_rate_hz,
    geometry_from_dict,
    legacy_azimuth_to_canonical_deg,
    profile_points,
    profile_round_trip,
    profile_schema_version,
    profile_to_geometry,
    profile_transform,
    rotation_matrix,
    segment_positions,
    segment_to_trajectory,
    segments_duration_s,
    segments_positions,
    trajectory_from_dict,
    upgrade_profile,
)

SAMPLE_RATE = 44_100

GEOMETRIES = [
    LineGeometry(),
    ArcGeometry(),
    CircleGeometry(radius_m=1.5),
    EllipseGeometry(),
    SpiralGeometry(),
    HelixGeometry(),
    LissajousGeometry(),
    PolylineGeometry(points_m=((1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.5))),
    SplineGeometry(points_m=((1.0, -1.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
    PointCloudGeometry(points_m=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
]


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.kind)
def test_every_geometry_returns_finite_metres(geometry):
    points = geometry.evaluate(np.linspace(0.0, 1.0, 64))
    assert points.shape == (3, 64)
    assert np.all(np.isfinite(points))


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.kind)
def test_every_geometry_round_trips_through_its_serialized_form(geometry):
    restored = geometry_from_dict(geometry.to_dict())
    np.testing.assert_allclose(
        restored.evaluate(np.linspace(0.0, 1.0, 32)), geometry.evaluate(np.linspace(0.0, 1.0, 32))
    )


def test_circle_places_the_cardinal_directions_where_they_belong():
    circle = CircleGeometry(radius_m=1.0)
    front, left, rear, right = circle.evaluate([0.0, 0.25, 0.5, 0.75]).T
    assert cartesian_to_spherical_deg(front)[0] == pytest.approx(0.0)
    assert cartesian_to_spherical_deg(left)[0] == pytest.approx(90.0)
    assert abs(cartesian_to_spherical_deg(rear)[0]) == pytest.approx(180.0)
    assert cartesian_to_spherical_deg(right)[0] == pytest.approx(-90.0)


def test_circle_length_matches_the_analytic_circumference():
    assert CircleGeometry(radius_m=1.5).length_m() == pytest.approx(2 * math.pi * 1.5, rel=1e-4)


def test_line_length_is_the_distance_between_its_ends():
    line = LineGeometry(start_m=(0.0, 0.0, 0.0), end_m=(3.0, 4.0, 0.0))
    assert line.length_m() == pytest.approx(5.0, rel=1e-6)


def test_arc_length_parameterization_gives_constant_steps():
    """Equal steps in `u` are not equal steps in metres on an ellipse."""

    ellipse = EllipseGeometry(forward_radius_m=2.0, lateral_radius_m=0.5)
    raw = ellipse.evaluate(np.linspace(0.0, 1.0, 200))
    even = ellipse.evaluate_by_arc_length(np.linspace(0.0, 1.0, 200))

    raw_steps = np.linalg.norm(np.diff(raw, axis=1), axis=0)
    even_steps = np.linalg.norm(np.diff(even, axis=1), axis=0)
    assert raw_steps.max() / raw_steps.min() > 3.0
    assert even_steps.max() / even_steps.min() < 1.15


def test_closed_geometries_declare_themselves_closed():
    assert CircleGeometry().is_closed
    assert EllipseGeometry().is_closed
    assert not LineGeometry().is_closed
    assert PolylineGeometry(points_m=((0, 0, 0), (1, 0, 0)), closed=True).is_closed


def test_polyline_and_spline_pass_through_their_endpoints():
    points = ((1.0, -1.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    for geometry in (PolylineGeometry(points_m=points), SplineGeometry(points_m=points)):
        ends = geometry.evaluate([0.0, 1.0])
        np.testing.assert_allclose(ends[:, 0], points[0], atol=1e-9)
        np.testing.assert_allclose(ends[:, 1], points[-1], atol=1e-9)


def test_point_cloud_holds_each_point_then_jumps():
    cloud = PointCloudGeometry(points_m=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    values = cloud.evaluate([0.0, 0.4, 0.6, 0.9])
    np.testing.assert_allclose(values[:, 0], values[:, 1])
    np.testing.assert_allclose(values[:, 2], values[:, 3])
    assert not np.allclose(values[:, 1], values[:, 2])


def test_geometry_needs_enough_points():
    with pytest.raises(ValueError):
        PolylineGeometry(points_m=((0.0, 0.0, 0.0),)).evaluate([0.0])
    with pytest.raises(ValueError):
        geometry_from_dict({"kind": "hyperbola"})


# --- traversal --------------------------------------------------------------


@pytest.mark.parametrize("mode", ["loop", "one_shot", "ping_pong", "reverse"])
def test_traversal_positions_stay_in_range(mode):
    traversal = TraversalSpec(rate_hz=0.7, mode=mode)
    position = traversal.positions(np.linspace(0.0, 10.0, 501)).position
    assert position.min() >= -1e-12
    assert position.max() <= 1.0 + 1e-12


def test_traversal_is_block_size_invariant():
    traversal = TraversalSpec(rate_hz=0.3, mode="ping_pong", easing="sine")
    whole = traversal.positions(traversal.sample_times(0, 1_000, SAMPLE_RATE)).position
    first = traversal.positions(traversal.sample_times(0, 137, SAMPLE_RATE)).position
    second = traversal.positions(traversal.sample_times(137, 863, SAMPLE_RATE)).position
    np.testing.assert_array_equal(whole, np.concatenate((first, second)))


def test_one_shot_holds_its_end_state():
    position = TraversalSpec(rate_hz=1.0, mode="one_shot").positions([0.0, 0.5, 1.0, 5.0]).position
    assert position[-1] == pytest.approx(1.0)
    assert position[-2] == pytest.approx(1.0)


def test_ping_pong_returns_the_way_it_came():
    position = TraversalSpec(rate_hz=0.5, mode="ping_pong").positions([0.0, 1.0, 2.0, 3.0, 4.0]).position
    np.testing.assert_allclose(position, [0.0, 0.5, 1.0, 0.5, 0.0])


def test_reverse_runs_backwards():
    position = TraversalSpec(rate_hz=0.25, mode="reverse").positions([0.0, 1.0, 2.0]).position
    assert position[1] < position[0] or position[1] == pytest.approx(0.75)


def test_easing_changes_the_shape_not_the_endpoints():
    times = np.linspace(0.0, 1.0, 5)
    linear = TraversalSpec(rate_hz=1.0, mode="one_shot").positions(times).position
    eased = TraversalSpec(rate_hz=1.0, mode="one_shot", easing="smooth").positions(times).position
    assert eased[0] == pytest.approx(linear[0])
    assert eased[-1] == pytest.approx(linear[-1])
    assert eased[1] < linear[1]


def test_keyframed_traversal_follows_its_keyframes():
    traversal = TraversalSpec(easing="keyframed", keyframes=((0.0, 0.0), (2.0, 1.0)))
    np.testing.assert_allclose(traversal.positions([0.0, 1.0, 2.0, 3.0]).position, [0.0, 0.5, 1.0, 1.0])


def test_keyframed_traversal_requires_keyframes():
    with pytest.raises(ValueError):
        TraversalSpec(easing="keyframed")


def test_jumps_are_reported_and_move_the_position():
    traversal = TraversalSpec(rate_hz=0.0, jump_times_s=(1.0, 2.0), jump_step=0.25)
    result = traversal.positions(np.arange(0.0, 4.0, 0.5))
    assert result.jump_indices.size == 2
    assert result.position[0] == pytest.approx(0.0)
    assert result.position[-1] == pytest.approx(0.5)
    assert result.crossfade_ms > 0.0


def test_absolute_jump_positions_are_honoured():
    traversal = TraversalSpec(rate_hz=0.0, jump_times_s=(1.0,), jump_positions=(0.75,))
    result = traversal.positions([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(result.position, [0.0, 0.0, 0.75, 0.75])


def test_unknown_traversal_settings_are_rejected():
    with pytest.raises(ValueError):
        TraversalSpec(mode="spiral")
    with pytest.raises(ValueError):
        TraversalSpec(easing="bouncy")


def test_constant_speed_rate_matches_the_path_length():
    circle = CircleGeometry(radius_m=1.0)
    rate = constant_speed_rate_hz(circle.length_m(), speed_m_s=2.0)
    assert rate == pytest.approx(2.0 / (2 * math.pi), rel=1e-4)


# --- transforms -------------------------------------------------------------


def test_yaw_turns_toward_the_left():
    turned = rotation_matrix(90.0) @ np.array([[1.0], [0.0], [0.0]])
    np.testing.assert_allclose(turned.ravel(), [0.0, 1.0, 0.0], atol=1e-12)


def test_transform_applies_translation_rotation_and_scale():
    transform = TransformSpec(translation_m=(1.0, 0.0, 0.0), yaw_deg=90.0, scale=(2.0, 2.0, 1.0))
    moved = transform.apply(np.array([[1.0], [0.0], [0.0]]))
    np.testing.assert_allclose(moved.ravel(), [1.0, 2.0, 0.0], atol=1e-12)


def test_identity_transform_changes_nothing():
    points = np.array([[1.0, 2.0], [0.5, -0.5], [0.0, 1.0]])
    assert TransformSpec().is_identity
    np.testing.assert_array_equal(TransformSpec().apply(points), points)


def test_reflection_mirrors_the_chosen_axis():
    mirrored = TransformSpec(reflect_y=True).apply(np.array([[1.0], [1.0], [0.0]]))
    np.testing.assert_allclose(mirrored.ravel(), [1.0, -1.0, 0.0])


def test_listener_frame_round_trips_between_world_and_head():
    listener = ListenerFrame(position_m=(1.0, 2.0, 0.0), yaw_deg=35.0)
    world = np.array([[3.0, -1.0], [1.0, 2.0], [0.5, 0.0]])
    np.testing.assert_allclose(listener.listener_to_world(listener.world_to_listener(world)), world, atol=1e-12)


def test_listener_ears_sit_at_the_head_radius():
    ears = ListenerFrame(head_radius_m=0.09).ear_positions_m
    np.testing.assert_allclose(ears[:, 0], [0.0, 0.09, 0.0])
    np.testing.assert_allclose(ears[:, 1], [0.0, -0.09, 0.0])


# --- the legacy pixel bridge ------------------------------------------------


def test_legacy_scene_maps_marker_a_to_the_left_ear():
    """The editor's A/B markers are the left and right ears; this pins that."""

    bridge = LegacyPathTransform()
    canonical = bridge.to_canonical(np.array([[-65.0, 0.0], [65.0, 0.0], [0.0, -100.0]]))
    assert cartesian_to_spherical_deg(canonical[:, 0])[0] == pytest.approx(90.0)
    assert cartesian_to_spherical_deg(canonical[:, 1])[0] == pytest.approx(-90.0)
    assert cartesian_to_spherical_deg(canonical[:, 2])[0] == pytest.approx(0.0)


def test_default_scale_puts_the_ear_markers_at_the_head_radius():
    canonical = LegacyPathTransform().to_canonical(np.array([[-65.0, 0.0]]))
    assert canonical[1, 0] == pytest.approx(ListenerFrame().head_radius_m, rel=1e-9)


def test_legacy_scene_round_trips_exactly():
    bridge = LegacyPathTransform()
    points = np.array([[-120.0, -20.0], [0.0, 0.0], [95.5, 37.25]])
    np.testing.assert_allclose(bridge.from_canonical(bridge.to_canonical(points)), points, atol=1e-12)


def test_scene_units_scale_the_result_predictably():
    points = np.array([[100.0, 0.0]])
    near = LegacyPathTransform(scene_units_per_metre=100.0).to_canonical(points)
    far = LegacyPathTransform(scene_units_per_metre=50.0).to_canonical(points)
    assert far[1, 0] == pytest.approx(2.0 * near[1, 0])


def test_unknown_axis_convention_is_rejected():
    with pytest.raises(ValueError):
        LegacyPathTransform(axis_convention="x_forward_y_left")


# --- legacy custom path profiles --------------------------------------------

LEGACY_PROFILE = {
    "kind": "linear",
    "closedLoop": False,
    "smoothingPasses": 1,
    "smoothingRatio": 0.25,
    "points": [[-120.0, -20.0], [0.0, 0.0], [120.0, 20.0]],
}


def test_unversioned_profiles_report_version_one():
    assert profile_schema_version(LEGACY_PROFILE) == 1
    assert profile_schema_version({}) == 1
    assert profile_schema_version(upgrade_profile(LEGACY_PROFILE)) == PROFILE_VERSION


def test_profile_points_survive_the_canonical_round_trip():
    np.testing.assert_allclose(
        profile_round_trip(LEGACY_PROFILE), profile_points(LEGACY_PROFILE), atol=1e-9
    )


def test_profile_becomes_canonical_geometry_in_metres():
    geometry = profile_to_geometry(LEGACY_PROFILE)
    assert geometry is not None
    points = geometry.evaluate(np.linspace(0.0, 1.0, 32))
    assert np.all(np.isfinite(points))
    assert np.max(np.abs(points)) < 1.0  # a pixel path is a small path in metres


def test_spline_profiles_become_splines():
    spline = dict(LEGACY_PROFILE, kind="spline")
    assert type(profile_to_geometry(spline)).__name__ == "SplineGeometry"
    assert type(profile_to_geometry(LEGACY_PROFILE)).__name__ == "PolylineGeometry"


def test_migration_is_explicit_and_non_destructive():
    original = dict(LEGACY_PROFILE)
    upgraded = upgrade_profile(original)
    assert upgraded["schemaVersion"] == PROFILE_VERSION
    assert upgraded["axisConvention"] == "x_right_y_down"
    assert upgraded["points"] == original["points"]
    assert "schemaVersion" not in original


def test_versioned_profiles_declare_their_own_bridge():
    upgraded = upgrade_profile(LEGACY_PROFILE, LegacyPathTransform(scene_units_per_metre=200.0))
    assert profile_transform(upgraded).scene_units_per_metre == pytest.approx(200.0)
    # A version-1 profile declares nothing, so the documented default applies.
    assert profile_transform(LEGACY_PROFILE).scene_units_per_metre == pytest.approx(
        LegacyPathTransform().scene_units_per_metre
    )


def test_an_empty_profile_has_no_geometry():
    assert profile_to_geometry({"points": []}) is None
    assert profile_points("not json").shape == (0, 2)


# --- legacy trajectory segments ---------------------------------------------


def test_legacy_azimuth_sign_is_flipped_once():
    assert legacy_azimuth_to_canonical_deg(90.0) == pytest.approx(-90.0)
    assert legacy_azimuth_to_canonical_deg(-45.0) == pytest.approx(45.0)


def test_rotate_segment_turns_the_way_the_legacy_engine_did():
    """Legacy azimuth grows toward the right, so canonical azimuth decreases."""

    segment = {"mode": "rotate", "start_deg": 0.0, "speed_deg_per_s": 90.0, "distance_m": 1.0, "seconds": 4.0}
    positions = segment_positions(segment, np.array([0.0, 1.0, 2.0]))
    azimuths = [cartesian_to_spherical_deg(positions[:, index])[0] for index in range(3)]
    assert azimuths[0] == pytest.approx(0.0)
    assert azimuths[1] == pytest.approx(-90.0)
    assert abs(azimuths[2]) == pytest.approx(180.0)


def test_oscillate_segment_sweeps_around_its_centre():
    segment = {
        "mode": "oscillate",
        "center_deg": 0.0,
        "extent_deg": 90.0,
        "period_s": 4.0,
        "distance_m": 1.0,
        "seconds": 4.0,
    }
    positions = segment_positions(segment, np.linspace(0.0, 4.0, 9))
    azimuths = np.array(
        [cartesian_to_spherical_deg(positions[:, index])[0] for index in range(positions.shape[1])]
    )
    assert azimuths.max() == pytest.approx(45.0, abs=0.5)
    assert azimuths.min() == pytest.approx(-45.0, abs=0.5)


def test_distance_ranges_ramp_over_the_segment():
    segment = {"mode": "rotate", "speed_deg_per_s": 0.0, "distance_m": [1.0, 3.0], "seconds": 2.0}
    positions = segment_positions(segment, np.array([0.0, 1.0, 2.0]))
    radii = np.linalg.norm(positions, axis=0)
    np.testing.assert_allclose(radii, [1.0, 2.0, 3.0], atol=1e-9)


def test_segment_lists_play_one_after_another():
    segments = [
        {"mode": "rotate", "start_deg": 0.0, "speed_deg_per_s": 0.0, "distance_m": 1.0, "seconds": 1.0},
        {"mode": "rotate", "start_deg": 90.0, "speed_deg_per_s": 0.0, "distance_m": 2.0, "seconds": 1.0},
    ]
    assert segments_duration_s(segments) == pytest.approx(2.0)
    positions = segments_positions(segments, np.array([0.5, 1.5, 5.0]))
    radii = np.linalg.norm(positions, axis=0)
    assert radii[0] == pytest.approx(1.0)
    assert radii[1] == pytest.approx(2.0)
    assert radii[2] == pytest.approx(2.0)  # past the end, the last position holds


def test_rotate_and_oscillate_have_geometry_and_traversal_forms():
    rotate = segment_to_trajectory(
        {"mode": "rotate", "start_deg": 0.0, "speed_deg_per_s": 90.0, "distance_m": 1.0, "seconds": 4.0}
    )
    assert rotate is not None
    geometry, traversal = rotate
    assert geometry.radius_m == pytest.approx(1.0)
    assert traversal.rate_hz == pytest.approx(-0.25)

    assert segment_to_trajectory({"mode": "oscillate", "extent_deg": 90.0, "period_s": 2.0}) is not None
    # Two superimposed rates have no single geometry-plus-traversal form.
    assert segment_to_trajectory({"mode": "rotating_arc"}) is None


# --- the composed trajectory ------------------------------------------------


def test_trajectory_evaluates_from_absolute_samples():
    trajectory = TrajectorySpec(
        geometry=CircleGeometry(radius_m=1.0), traversal=TraversalSpec(rate_hz=0.5)
    )
    whole = trajectory.positions_for_block(0, 1_000, SAMPLE_RATE).positions_m
    tail = trajectory.positions_for_block(500, 500, SAMPLE_RATE).positions_m
    np.testing.assert_allclose(whole[:, 500:], tail, atol=1e-12)


def test_trajectory_applies_its_transform():
    plain = TrajectorySpec(geometry=CircleGeometry(radius_m=1.0))
    moved = TrajectorySpec(
        geometry=CircleGeometry(radius_m=1.0), transform=TransformSpec(translation_m=(2.0, 0.0, 0.0))
    )
    np.testing.assert_allclose(
        moved.sample_path(16)[0] - plain.sample_path(16)[0], np.full(16, 2.0), atol=1e-12
    )


def test_world_frame_paths_move_when_the_listener_turns():
    trajectory = TrajectorySpec(
        geometry=CircleGeometry(radius_m=1.0),
        traversal=TraversalSpec(rate_hz=0.0),
        coordinate_frame="world",
    )
    facing_forward = trajectory.positions_at([0.0], listener=ListenerFrame()).positions_m
    turned = trajectory.positions_at([0.0], listener=ListenerFrame(yaw_deg=90.0)).positions_m
    assert cartesian_to_spherical_deg(facing_forward[:, 0])[0] == pytest.approx(0.0)
    assert cartesian_to_spherical_deg(turned[:, 0])[0] == pytest.approx(-90.0)


def test_trajectory_round_trips_through_its_serialized_form():
    trajectory = TrajectorySpec(
        geometry=EllipseGeometry(forward_radius_m=2.0),
        traversal=TraversalSpec(rate_hz=0.4, mode="ping_pong"),
        transform=TransformSpec(yaw_deg=30.0),
    )
    restored = trajectory_from_dict(trajectory.to_dict())
    np.testing.assert_allclose(restored.sample_path(32), trajectory.sample_path(32), atol=1e-12)


def test_spherical_view_reports_azimuth_elevation_and_radius():
    trajectory = TrajectorySpec(
        geometry=CircleGeometry(radius_m=2.0), traversal=TraversalSpec(rate_hz=0.0)
    )
    spherical = trajectory.positions_at([0.0, 1.0]).spherical_deg()
    assert spherical.shape == (3, 2)
    np.testing.assert_allclose(spherical[2], 2.0)


def test_unknown_coordinate_frame_is_rejected():
    with pytest.raises(ValueError):
        TrajectorySpec(coordinate_frame="galactic")

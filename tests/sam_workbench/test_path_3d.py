"""Three-dimensional path creation, from authored geometry to rendered audio.

The rules this file protects:

* a position is stored Cartesian and once - spherical entry converts at the
  boundary rather than becoming a second saved format;
* every primitive compiles to the same kind of timestamped trajectory, and
  survives a save and reload unchanged;
* geometry and traversal stay separable - re-timing a path must not reshape it;
* the HRTF renderer receives the whole path, including elevation and distance,
  and says so when the dataset cannot support where the path goes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.trajectory import (
    PRIMITIVE_TYPES,
    DomeTraversal,
    FigureEight3D,
    HorizontalOrbit,
    Keyframe,
    KeyframedPath,
    OverheadSweep,
    PathModel,
    RisingArc,
    SphericalPosition,
    TiltedOrbit,
    Torus,
    Traversal,
    VerticalOrbit,
    cartesian_array_to_spherical,
    cartesian_to_spherical,
    keyframes_from_csv,
    keyframes_from_json,
    path_model_from_dict,
    spherical_to_cartesian,
)
from src.audio.sam_workbench.trajectory.serialization import _SPATIAL

TIMES = np.linspace(0.0, 5.0, 128)


# --- coordinates ------------------------------------------------------------


def test_spherical_entry_round_trips_through_the_cartesian_store():
    original = SphericalPosition(45.0, 30.0, 1.5)
    recovered = SphericalPosition.from_cartesian(original.to_cartesian())
    assert recovered.azimuth_deg == pytest.approx(45.0)
    assert recovered.elevation_deg == pytest.approx(30.0)
    assert recovered.distance_m == pytest.approx(1.5)


def test_documented_spherical_example_converts_as_specified():
    """The GUI's spherical form is accepted under its documented key names."""

    position = SphericalPosition.from_mapping(
        {"azimuthDegrees": 45.0, "elevationDegrees": 30.0, "distanceMetres": 1.5}
    )
    x, y, z = position.to_cartesian()
    assert math.hypot(math.hypot(x, y), z) == pytest.approx(1.5)
    assert z == pytest.approx(1.5 * math.sin(math.radians(30.0)))


@pytest.mark.parametrize(
    "point, azimuth, elevation",
    [
        ((1.0, 0.0, 0.0), 0.0, 0.0),      # in front
        ((0.0, 1.0, 0.0), 90.0, 0.0),     # to the left
        ((-1.0, 0.0, 0.0), 180.0, 0.0),   # behind
        ((0.0, 0.0, 2.0), 0.0, 90.0),     # overhead
        ((0.0, 0.0, -2.0), 0.0, -90.0),   # underneath
    ],
)
def test_canonical_directions_read_back_as_expected(point, azimuth, elevation):
    got_azimuth, got_elevation, distance = cartesian_to_spherical(point)
    assert got_azimuth == pytest.approx(azimuth)
    assert got_elevation == pytest.approx(elevation)
    assert distance == pytest.approx(np.linalg.norm(point))


def test_the_listener_position_itself_stays_finite():
    """A path may pass through the origin; it must not produce a NaN there."""

    assert cartesian_to_spherical((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


# --- keyframes --------------------------------------------------------------


def test_documented_keyframe_example_loads_and_reaches_its_positions():
    keyframes = keyframes_from_json(
        {
            "coordinateSystem": "listener_relative_cartesian",
            "units": "metres",
            "interpolation": "cubic",
            "keyframes": [
                {"timeSeconds": 0.0, "position": [0.0, 1.0, 0.0]},
                {"timeSeconds": 5.0, "position": [1.0, 0.0, 1.5]},
                {"timeSeconds": 10.0, "position": [0.0, -1.0, 0.5]},
            ],
        }
    )
    path = KeyframedPath(keyframes, "cubic")
    assert path.at_time([0.0, 5.0, 10.0]) == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.5], [0.0, -1.0, 0.5]]), abs=1e-9
    )


@pytest.mark.parametrize("interpolation", ["hold", "linear", "cubic", "catmull_rom"])
def test_every_interpolation_passes_through_its_keys(interpolation):
    keys = (
        Keyframe(0.0, (1.0, 0.0, 0.0)),
        Keyframe(2.0, (0.0, 1.0, 0.5)),
        Keyframe(6.0, (-1.0, 0.0, -0.5)),
    )
    path = KeyframedPath(keys, interpolation)
    # ``hold`` steps at each key rather than approaching it, so it is checked
    # at the key it holds from rather than the one it is heading to.
    expected = np.array([key.position_m for key in keys])
    assert path.at_time([0.0, 2.0, 6.0]) == pytest.approx(expected, abs=1e-9)


def test_uneven_key_spacing_is_honoured_rather_than_redistributed():
    """A key nine tenths of the way through time sits nine tenths along u."""

    path = KeyframedPath(
        (Keyframe(0.0, (0.0, 0.0, 0.0)), Keyframe(9.0, (9.0, 0.0, 0.0)), Keyframe(10.0, (10.0, 0.0, 0.0))),
        "linear",
    )
    assert float(path.evaluate(0.9)[0]) == pytest.approx(9.0)


def test_csv_import_accepts_cartesian_and_spherical_columns():
    cartesian = keyframes_from_csv("time,x,y,z\n0,1,0,0\n1,0,1,0.5\n")
    spherical = keyframes_from_csv(
        "time,azimuth,elevation,distance\n0,0,0,1\n1,90,30,1\n"
    )
    assert cartesian[0].position_m == pytest.approx((1.0, 0.0, 0.0))
    assert spherical[0].position_m == pytest.approx((1.0, 0.0, 0.0))
    assert cartesian_to_spherical(spherical[1].position_m)[0] == pytest.approx(90.0)


def test_a_keyframe_cannot_carry_two_position_formats_at_once():
    with pytest.raises(ValueError, match="not both"):
        Keyframe.from_mapping(
            {"timeSeconds": 0.0, "position": [1, 0, 0], "azimuthDegrees": 45.0}
        )


# --- primitives -------------------------------------------------------------


@pytest.mark.parametrize("kind", PRIMITIVE_TYPES)
def test_every_primitive_compiles_to_a_finite_three_dimensional_trajectory(kind):
    model = PathModel(geometry=_SPATIAL[kind](), traversal=Traversal(duration_s=5.0))
    positions = model.positions(TIMES)
    assert positions.shape == (len(TIMES), 3)
    assert np.all(np.isfinite(positions))


@pytest.mark.parametrize(
    "geometry",
    [VerticalOrbit(), TiltedOrbit(), RisingArc(), OverheadSweep(), DomeTraversal(), Torus()],
    ids=lambda value: type(value).__name__,
)
def test_the_three_dimensional_primitives_actually_leave_the_horizontal_plane(geometry):
    """A 3-D primitive that never changes height would be a 2-D one."""

    heights = geometry.evaluate(np.linspace(0.0, 1.0, 257))[:, 2]
    assert np.ptp(heights) > 0.1


def test_a_tilted_orbit_at_zero_tilt_is_a_horizontal_orbit():
    progress = np.linspace(0.0, 1.0, 64)
    assert TiltedOrbit(radius_m=1.5, tilt_deg=0.0).evaluate(progress) == pytest.approx(
        np.asarray(HorizontalOrbit(radius_m=1.5).evaluate(progress)), abs=1e-9
    )


def test_the_overhead_sweep_runs_from_the_front_over_the_top_to_the_back():
    spherical = cartesian_array_to_spherical(
        OverheadSweep().evaluate(np.array([0.0, 0.5, 1.0]))
    )
    assert spherical[0, 0] == pytest.approx(0.0)     # starts in front
    assert spherical[1, 1] == pytest.approx(90.0)    # passes through the zenith
    assert abs(spherical[2, 0]) == pytest.approx(180.0)  # ends behind


def test_the_figure_eight_stays_off_the_listeners_own_position():
    """A Cartesian lemniscate crosses through the head, where gain is unbounded."""

    distances = np.linalg.norm(FigureEight3D().evaluate(np.linspace(0.0, 1.0, 257)), axis=1)
    assert distances.min() == pytest.approx(1.5)


def test_a_tilted_figure_eight_alternates_above_left_with_below_right():
    spherical = cartesian_array_to_spherical(
        FigureEight3D(tilt_deg=45.0).evaluate(np.linspace(0.0, 1.0, 9))
    )
    left_high = spherical[3]   # azimuth positive (left), elevation positive
    right_low = spherical[7]   # the opposite lobe
    assert left_high[0] > 0.0 and left_high[1] > 0.0
    assert right_low[0] < 0.0 and right_low[1] < 0.0


def test_a_seeded_random_walk_is_reproducible_and_stays_in_its_volume():
    from src.audio.sam_workbench.trajectory import RandomWalkVolume

    walk = RandomWalkVolume(extent_m=(2.0, 2.0, 1.0), seed=7, minimum_distance_m=0.3)
    progress = np.linspace(0.0, 1.0, 128)
    assert walk.evaluate(progress) == pytest.approx(
        RandomWalkVolume(extent_m=(2.0, 2.0, 1.0), seed=7, minimum_distance_m=0.3).evaluate(progress)
    )
    # Never inside the listener's head, which is what the minimum enforces.
    assert np.min(np.linalg.norm(walk.evaluate(progress), axis=1)) > 0.2


# --- the path model ---------------------------------------------------------


def test_the_saved_form_names_its_frame_units_and_handedness():
    """A project opened elsewhere must not have to guess what the numbers mean."""

    described = PathModel(geometry=DomeTraversal()).describe()
    assert described["coordinateSystem"] == "listener_relative_cartesian"
    assert described["handedness"] == "right"
    assert described["units"] == "metres"
    assert set(described["axes"]) == {"x", "y", "z"}
    assert described["speedLaw"] in ("constant_speed", "parameter_speed")


@pytest.mark.parametrize("kind", PRIMITIVE_TYPES)
def test_a_primitive_survives_a_save_and_reload_exactly(kind):
    model = PathModel(geometry=_SPATIAL[kind](), traversal=Traversal(duration_s=5.0))
    reloaded = path_model_from_dict(model.describe())
    assert reloaded.positions(TIMES) == pytest.approx(model.positions(TIMES))


def test_a_keyframed_path_survives_a_save_and_reload_exactly():
    model = PathModel(
        geometry=KeyframedPath(
            (Keyframe(0.0, (0.0, 1.0, 0.0)), Keyframe(5.0, (1.0, 0.0, 1.5))), "cubic"
        )
    )
    reloaded = path_model_from_dict(model.describe())
    assert reloaded.positions(TIMES) == pytest.approx(model.positions(TIMES))


def test_retiming_a_path_does_not_reshape_it():
    """Geometry and traversal are separable, which is the whole design."""

    model = PathModel(geometry=HorizontalOrbit(), traversal=Traversal(duration_s=5.0))
    faster = model.with_traversal(Traversal(duration_s=1.0, easing="sine"))
    # Same set of positions visited, reached at different times.
    original = np.sort(np.linalg.norm(model.positions(TIMES), axis=1))
    retimed = np.sort(np.linalg.norm(faster.positions(TIMES), axis=1))
    assert original == pytest.approx(retimed, abs=1e-9)
    assert not np.allclose(model.positions(TIMES), faster.positions(TIMES))


def test_a_world_frame_path_is_resolved_against_the_listener_pose():
    from src.audio.sam_workbench.trajectory import ListenerTransform

    listener = ListenerTransform(position_m=(1.0, 0.0, 0.0))
    model = PathModel(
        geometry=HorizontalOrbit(radius_m=1.0),
        coordinate_system="world_cartesian",
        listener=listener,
    )
    # A world orbit centred on the origin is off-centre from a listener who is
    # standing a metre in front of it.
    distances = np.linalg.norm(model.positions(TIMES), axis=1)
    assert np.ptp(distances) > 0.5


def test_a_spherical_keyframe_and_its_cartesian_twin_describe_one_path():
    spherical = path_model_from_dict(
        {
            "geometry": {
                "type": "keyframes",
                "keyframes": [
                    {"timeSeconds": 0.0, "azimuthDegrees": 45.0, "elevationDegrees": 30.0, "distanceMetres": 1.5},
                    {"timeSeconds": 4.0, "azimuthDegrees": -90.0, "elevationDegrees": 0.0, "distanceMetres": 1.0},
                ],
            }
        }
    )
    cartesian = path_model_from_dict(
        {
            "geometry": {
                "type": "keyframes",
                "keyframes": [
                    {"timeSeconds": 0.0, "position": list(spherical_to_cartesian(45.0, 30.0, 1.5))},
                    {"timeSeconds": 4.0, "position": list(spherical_to_cartesian(-90.0, 0.0, 1.0))},
                ],
            }
        }
    )
    assert spherical.positions(TIMES) == pytest.approx(cartesian.positions(TIMES))

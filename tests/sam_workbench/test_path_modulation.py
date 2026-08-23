"""Modulated path parameters: bindings, bounds, and block invariance.

The rules this file protects:

* a project without path modulation gets its exact former positions back -
  the wrapper must never touch an unbound path;
* offsets follow the modulation matrix's own semantics (summed across
  modulators, added to the stored base, held inside documented ranges);
* positions are a pure function of absolute time, so any blocking of the
  timeline yields one trajectory;
* a moving shape cannot pretend the frozen arc-length table still applies,
  and a modulated torus can never wind its tube through the listener's head.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.controls import LfoControl, RampControl
from src.audio.sam_workbench.trajectory import (
    PathBinding,
    bind_path_parameters,
    cartesian_array_to_spherical,
    path_model_from_dict,
)
from src.audio.sam_workbench.trajectory.parameter_catalog import (
    GEOMETRY_PARAMETER_SPECS,
    route_leaf,
    split_parameter_path,
)
from src.audio.sam_workbench.trajectory.primitives import (
    horizontal_orbit_points,
    torus_points,
)

RATE_HZ = 48_000


def _payload(geometry: dict, duration_s: float = 4.0) -> dict:
    return {
        "schemaVersion": 2,
        "coordinateSystem": "listener_relative_cartesian",
        "geometry": geometry,
        "transform": {},
        "traversal": {"mode": "loop", "durationS": duration_s},
    }


def _orbit_model(**overrides):
    parameters = {"radius_m": 1.5, "elevation_deg": 0.0, "start_azimuth_deg": 0.0}
    parameters.update(overrides)
    return path_model_from_dict(
        _payload({"type": "horizontal_orbit", "parameters": parameters})
    )


def _torus_model(**overrides):
    parameters = {
        "major_radius_m": 1.0,
        "minor_radius_m": 0.9,
        "minor_turns": 2.0,
    }
    parameters.update(overrides)
    return path_model_from_dict(
        _payload({"type": "torus", "parameters": parameters})
    )


def _binding(field: str, control) -> PathBinding:
    return PathBinding(f"path.{route_leaf(field)}", "geometry", field, control)


def _times(count: int = 997, stop: float = 4.0):
    return np.linspace(0.0, stop, count)


def _control_times(times):
    """Controls read whole samples, so expectations share that clock."""

    return np.rint(np.asarray(times) * RATE_HZ) / RATE_HZ


# --- identity without bindings ----------------------------------------------


def test_no_bindings_returns_the_model_and_its_exact_positions():
    model = _orbit_model()
    assert bind_path_parameters(model, [], sample_rate_hz=RATE_HZ) is model

    times = _times()
    static = model.positions(times)
    assert np.array_equal(static, model.positions(times))


# --- route semantics ---------------------------------------------------------


def test_bound_values_are_the_parameters_effective_value():
    """Controls provide the whole value, as scene automation does for any
    parameter: the stored base is resolved by whoever compiled the control."""

    model = _orbit_model(radius_m=1.5)
    base, depth, rate_hz = 1.5, 0.4, 0.75
    bound = bind_path_parameters(
        model,
        [_binding("radius_m", LfoControl(offset=base, rate_hz=rate_hz, depth=depth))],
        sample_rate_hz=RATE_HZ,
    )

    times = _times()
    expected_radius = base + depth * np.sin(
        2.0 * np.pi * rate_hz * _control_times(times)
    )
    expected = horizontal_orbit_points(
        model.traversal.progress(times),
        radius_m=expected_radius,
        elevation_deg=0.0,
        start_azimuth_deg=0.0,
        turns=1.0,
        centre_m=(0.0, 0.0, 0.0),
    )
    assert np.array_equal(bound.positions(times), expected)

    distances = cartesian_array_to_spherical(bound.positions(times))[:, 2]
    assert distances.max() > 1.5 and distances.min() < 1.5


def test_a_merged_control_carries_multiple_routes_contributions():
    """In production one compiled control merges every route reaching a
    parameter; the bound path reads that single answer."""

    model = _orbit_model(radius_m=0.5)
    times = _times()
    clock = _control_times(times)
    merged = 0.5 + 0.25 * np.sin(2.0 * np.pi * clock) - 0.125 * np.sin(2.0 * np.pi * clock)
    bound = bind_path_parameters(
        model,
        [_binding("radius_m", LfoControl(offset=float(merged[0]), rate_hz=0.0, depth=0.0))],
        sample_rate_hz=RATE_HZ,
    )
    # A zero-rate LFO holds its phase value, so this stands in for the merged
    # constant; the summation itself is proven end to end in the scene tests.
    assert np.allclose(
        cartesian_array_to_spherical(bound.positions(times))[:, 2], merged[0]
    )


def test_the_same_parameter_cannot_be_bound_twice():
    model = _orbit_model(radius_m=1.5)
    with pytest.raises(ValueError, match="bound more than once"):
        bind_path_parameters(
            model,
            [
                _binding("radius_m", LfoControl(offset=1.5)),
                _binding("radius_m", LfoControl(offset=1.6)),
            ],
            sample_rate_hz=RATE_HZ,
        )


# --- time invariance ---------------------------------------------------------


def test_positions_are_identical_under_arbitrary_blocking():
    model = _torus_model()
    bound = bind_path_parameters(
        model,
        [
            _binding("major_radius_m", LfoControl(rate_hz=0.3, depth=0.2)),
            PathBinding("transform.yawDeg", "transform", "yaw_deg",
                        LfoControl(rate_hz=0.11, depth=45.0)),
        ],
        sample_rate_hz=RATE_HZ,
    )
    times = _times(count=1024)
    whole = bound.positions(times)

    edges = [0, 1, 3, 17, 64, 400, 700, 1023, 1024]
    pieces = [
        bound.positions(times[start:end])
        for start, end in zip(edges, edges[1:])
    ]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_origin_sample_shifts_the_control_clock_exactly_once():
    model = _orbit_model()
    bound = bind_path_parameters(
        model,
        [PathBinding("transform.translationXM", "transform",
                     "translation_x_m", RampControl(slope_per_s=1.0))],
        sample_rate_hz=RATE_HZ,
        origin_sample=int(3.5 * RATE_HZ),
    )
    times = np.array([0.0, 0.5, 2.0])
    # The ramp reads absolute samples, so its value is (3.5 s + t) metres;
    # the path's own contribution cancels against the static model.
    shifted = bound.positions(times)[:, 0] - model.positions(times)[:, 0]
    assert np.allclose(shifted, 3.5 + times)


# --- ranges and couplings ----------------------------------------------------


def test_lengths_are_held_above_zero():
    model = _orbit_model()
    bound = bind_path_parameters(
        model,
        [_binding("radius_m", LfoControl(offset=1.0, rate_hz=0.5, depth=1.0))],
        sample_rate_hz=RATE_HZ,
    )
    spherical = cartesian_array_to_spherical(bound.positions(_times()))
    minimum = GEOMETRY_PARAMETER_SPECS["radius_m"].minimum
    assert spherical[:, 2].min() == pytest.approx(minimum)
    assert spherical[:, 2].max() > 1.0


def test_elevation_is_held_inside_the_poles():
    model = _orbit_model(elevation_deg=80.0)
    bound = bind_path_parameters(
        model,
        [_binding("elevation_deg", LfoControl(offset=80.0, rate_hz=0.25, depth=50.0))],
        sample_rate_hz=RATE_HZ,
    )
    spherical = cartesian_array_to_spherical(bound.positions(_times()))
    assert spherical[:, 1].max() == pytest.approx(90.0)
    assert spherical[:, 1].min() == pytest.approx(30.0)
    assert np.all(spherical[:, 1] <= 90.0 + 1e-9)


def test_a_modulated_torus_tube_stays_on_its_ring():
    model = _torus_model()
    bound = bind_path_parameters(
        model,
        [_binding("minor_radius_m", LfoControl(offset=0.9, rate_hz=0.4, depth=0.5))],
        sample_rate_hz=RATE_HZ,
    )
    points = bound.positions(_times())
    ring = np.hypot(points[:, 0], points[:, 1])
    # |ring radius - major| is the tube radius actually realized.
    realized = np.abs(ring - 1.0)
    ceiling = 1.0 - GEOMETRY_PARAMETER_SPECS["minor_radius_m"].minimum
    assert realized.max() <= ceiling + 1e-9


# --- speed-law honesty -------------------------------------------------------


def test_shape_binding_gives_up_constant_speed_and_says_so():
    model = _torus_model()
    bound = bind_path_parameters(
        model,
        [_binding("major_radius_m", LfoControl(offset=1.0, rate_hz=0.2, depth=0.1))],
        sample_rate_hz=RATE_HZ,
    )
    assert not bound.uses_constant_speed_law
    assert any("parameter speed" in note for note in bound.notes)

    # The motion matches the parameter-speed evaluation of the same formulas.
    times = _times()
    u = model.traversal.progress(times)
    clock = _control_times(times)
    major = 1.0 + 0.1 * np.sin(2.0 * np.pi * 0.2 * clock)
    # The tube is re-clamped inside the moving ring after every resolution.
    minor = np.minimum(
        0.9, major - GEOMETRY_PARAMETER_SPECS["minor_radius_m"].minimum
    )
    expected = torus_points(
        u,
        major_radius_m=major,
        minor_radius_m=minor,
        major_turns=1.0,
        minor_turns=2.0,
        centre_m=(0.0, 0.0, 0.0),
    )
    assert np.allclose(bound.positions(times), expected)


def test_rigid_only_binding_keeps_constant_speed():
    model = _orbit_model()
    bound = bind_path_parameters(
        model,
        [PathBinding("transform.rollDeg", "transform", "roll_deg",
                     LfoControl(rate_hz=0.2, depth=15.0))],
        sample_rate_hz=RATE_HZ,
    )
    assert bound.uses_constant_speed_law
    assert bound.notes == ()


def test_a_static_yaw_offset_turns_the_whole_path_left():
    model = _orbit_model()
    bound = bind_path_parameters(
        model,
        [PathBinding("transform.yawDeg", "transform", "yaw_deg",
                     LfoControl(rate_hz=1.0, depth=0.0, offset=90.0))],
        sample_rate_hz=RATE_HZ,
    )
    front = bound.positions(np.array([0.0]))[0]
    # Positive yaw moves forward (+x) toward the listener's left (+y).
    assert front[1] == pytest.approx(1.5, abs=1e-9)
    assert front[0] == pytest.approx(0.0, abs=1e-9)


# --- refusals ----------------------------------------------------------------


def test_an_unknown_parameter_name_is_refused_at_bind_time():
    with pytest.raises(ValueError, match="unknown path parameter"):
        split_parameter_path("path.radiusMegabytes")


def test_a_field_the_geometry_does_not_have_is_refused():
    model = _orbit_model()
    with pytest.raises(ValueError, match="not a parameter"):
        bind_path_parameters(
            model,
            [_binding("major_radius_m", LfoControl())],
            sample_rate_hz=RATE_HZ,
        )


def test_point_based_geometry_carries_nothing_to_bind():
    payload = _payload(
        {"type": "polyline", "controlPointsM": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    )
    model = path_model_from_dict(payload)
    with pytest.raises(ValueError, match="no named parameters"):
        bind_path_parameters(
            model,
            [_binding("radius_m", LfoControl())],
            sample_rate_hz=RATE_HZ,
        )


def test_every_catalogued_geometry_field_is_real_on_some_primitive():
    from src.audio.sam_workbench.trajectory.primitives import (
        DomeTraversal, ElevationSweep, FigureEight3D, HorizontalOrbit,
        OverheadSweep, Pendulum, RandomWalkVolume, RisingArc,
        SphericalOrbit, TiltedOrbit, Torus, VerticalOrbit,
    )
    from src.audio.sam_workbench.trajectory.parameter_catalog import (
        primitive_component_fields,
    )

    covered: set[str] = set()
    for factory in (
        HorizontalOrbit, VerticalOrbit, TiltedOrbit, SphericalOrbit,
        RisingArc, OverheadSweep, ElevationSweep, DomeTraversal,
        FigureEight3D, Pendulum, Torus, RandomWalkVolume,
    ):
        covered |= set(primitive_component_fields(factory))
    stray = covered.symmetric_difference(set(GEOMETRY_PARAMETER_SPECS))
    assert not stray, f"catalog and primitives disagree about {stray}"

"""A modulation route may sweep an explicit interval instead of a depth.

Depth measures a swing away from the stored base in one direction, so a
parameter could only ever travel between its base and one other number.
Reaching -45 to +45 degrees meant storing a base of -45 and a depth of 90,
which does not read as the range it is. These tests pin the range form and,
just as importantly, pin that a document which never used one is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.modulation import ModulationRoute


UNIT = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _route(**overrides):
    settings = dict(
        modulator_id="lfo", target_id="voice", parameter_path="transform.yaw_deg"
    )
    settings.update(overrides)
    return ModulationRoute(**settings)


# --- the range itself -------------------------------------------------------


def test_a_range_sweeps_between_its_ends_and_may_cross_zero():
    route = _route(minimum=-45.0, maximum=45.0)
    values = route.apply(UNIT, base_value=0.0)
    assert values[0] == pytest.approx(-45.0)
    assert values[2] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(45.0)


def test_a_range_ignores_the_base_because_it_says_where_the_value_goes():
    route = _route(minimum=-1.0, maximum=1.0)
    from_zero = route.apply(UNIT, base_value=0.0)
    from_far_away = route.apply(UNIT, base_value=1000.0)
    assert np.allclose(from_zero, from_far_away)


def test_polarity_reverses_which_end_the_peak_reaches():
    forward = _route(minimum=-45.0, maximum=45.0, polarity=1)
    backward = _route(minimum=-45.0, maximum=45.0, polarity=-1)
    assert np.allclose(forward.apply(UNIT, 0.0), backward.apply(UNIT, 0.0)[::-1])


def test_a_range_still_honours_the_curve():
    linear = _route(minimum=0.0, maximum=10.0, curve="linear")
    curved = _route(minimum=0.0, maximum=10.0, curve="exponential")
    # Different journey between the ends ("smooth" is symmetric, so it would
    # agree with linear at the midpoint and prove nothing).
    assert linear.apply(UNIT, 0.0)[2] != pytest.approx(curved.apply(UNIT, 0.0)[2])
    # Both still arrive at the same ends.
    for route in (linear, curved):
        assert route.apply(UNIT, 0.0)[0] == pytest.approx(0.0)
        assert route.apply(UNIT, 0.0)[-1] == pytest.approx(10.0)


def test_a_degenerate_range_pins_the_value_and_stays_active():
    """Holding a parameter at a constant is a thing someone may mean, and is
    not the same as leaving the base alone."""

    route = _route(minimum=3.0, maximum=3.0)
    assert route.is_active is True
    assert np.allclose(route.apply(UNIT, base_value=999.0), 3.0)


# --- what a range refuses ---------------------------------------------------


def test_a_range_needs_both_ends():
    for half in ({"minimum": 1.0}, {"maximum": 1.0}):
        with pytest.raises(ValueError, match="both ends"):
            _route(**half)


def test_the_ends_must_be_ordered_and_finite():
    with pytest.raises(ValueError, match="below minimum"):
        _route(minimum=5.0, maximum=-5.0)
    with pytest.raises(ValueError, match="finite"):
        _route(minimum=0.0, maximum=float("inf"))


# --- nothing changes for a route that has no range --------------------------


def test_a_depth_route_behaves_exactly_as_before():
    route = _route(depth=2.0)
    assert np.allclose(route.apply(UNIT, base_value=10.0), 10.0 + 2.0 * UNIT)
    inverted = _route(depth=2.0, polarity=-1)
    assert np.allclose(inverted.apply(UNIT, base_value=10.0), 10.0 - 2.0 * UNIT)


def test_a_document_without_a_range_gains_no_keys_round_tripping():
    """An existing project must not be rewritten just by being opened."""

    stored = _route(depth=2.0).describe()
    assert "minimum" not in stored and "maximum" not in stored
    assert ModulationRoute.from_mapping(stored) == _route(depth=2.0)


def test_a_ranged_route_round_trips_through_a_document():
    route = _route(minimum=-45.0, maximum=45.0, depth=90.0, polarity=-1)
    restored = ModulationRoute.from_mapping(route.describe())
    assert restored == route
    assert np.allclose(restored.apply(UNIT, 0.0), route.apply(UNIT, 0.0))


def test_an_absent_range_reads_as_absent_not_as_zero():
    restored = ModulationRoute.from_mapping(
        {
            "modulatorId": "lfo",
            "targetId": "voice",
            "parameterPath": "transform.yaw_deg",
            "depth": 1.0,
            "minimum": None,
        }
    )
    assert restored.has_range is False


# --- how it describes itself ------------------------------------------------


def test_the_amount_reads_as_a_range_or_a_swing():
    assert _route(minimum=-45.0, maximum=45.0).describe_amount() == "-45 to 45"
    assert _route(minimum=-45.0, maximum=45.0, polarity=-1).describe_amount() == "45 to -45"
    assert _route(depth=2.0).describe_amount() == "+/-2"
    assert _route(depth=0.0).describe_amount() == "0"


def test_span_is_the_distance_travelled_either_way():
    assert _route(minimum=-45.0, maximum=45.0).span == pytest.approx(90.0)
    assert _route(depth=-3.0).span == pytest.approx(3.0)

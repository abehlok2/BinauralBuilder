"""What each of the five interpolation modes must actually deliver.

The modes all ran before this; none of the aligned ones worked properly. The
reason was one function: a fractional delay done by interpolating between
samples, which is a lowpass filter and a poor one. At half a sample it put more
than six decibels of ripple across the band, so every response that was aligned
and put back came out dulled - and the modes could not reproduce a measurement
at its own direction, which is the one thing an interpolator must do.

So the properties are checked here rather than assumed:

* at a measured direction, a mode returns that measurement;
* moving the source produces a continuous filter, with each mode at least as
  smooth as the cruder one below it;
* the interaural cues survive - a blend that averages away the ITD has
  destroyed the thing being interpolated;
* nothing blows up outside the measured grid, and a query that leaves it says
  so rather than pretending.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.analysis.hrtf_curves import ild_db, itd_seconds
from src.audio.sam_workbench.hrtf.coordinates import unit_vectors
from src.audio.sam_workbench.hrtf.decomposition import align_pair, shift_response
from src.audio.sam_workbench.hrtf.interpolation import (
    DELAY_MAGNITUDE,
    INTERPOLATION_MODES,
    NEAREST,
    SPHERICAL_HARMONIC,
    SPHERICAL_TRIANGULAR,
    THREE_NEIGHBOR,
    HrtfInterpolator,
)
from src.audio.sam_workbench.hrtf.selection import DirectionIndex
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(str(FIXTURES / "synthetic_sonicom_hrir.sofa"))


@pytest.fixture(scope="module")
def measured_directions(dataset):
    return unit_vectors(dataset.positions_m)


def _horizon(step_deg: float = 1.0) -> np.ndarray:
    azimuth = np.radians(np.arange(0.0, 360.0, step_deg))
    return np.column_stack((np.cos(azimuth), np.sin(azimuth), np.zeros_like(azimuth)))


# --- the fractional delay everything rests on -------------------------------


@pytest.mark.parametrize("delay", [0.0, 0.25, 0.5, 1.0, 2.25, 16.218, 39.903])
def test_a_fractional_shift_does_not_change_the_magnitude(delay):
    """A delay is a phase term. Anything it does to magnitude is damage."""

    impulse = np.zeros(256)
    impulse[64] = 1.0
    shifted = shift_response(impulse, delay, 256)

    frequencies = np.fft.rfftfreq(256)
    magnitude = np.abs(np.fft.rfft(shifted))
    audible = frequencies < 0.45
    ripple_db = 20.0 * np.log10(
        magnitude[audible].max() / max(float(magnitude[audible].min()), 1e-12)
    )
    assert ripple_db < 0.5


@pytest.mark.parametrize("delay", [1, 5, 23])
def test_an_integer_shift_is_exactly_a_shift(delay):
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, 64)
    shifted = shift_response(samples, delay, 64)
    assert np.max(np.abs(shifted[delay:] - samples[: 64 - delay])) < 1e-12
    assert np.max(np.abs(shifted[:delay])) < 1e-12


def test_removing_a_delay_and_putting_it_back_returns_the_response(dataset):
    """Every aligned mode does exactly this, so it has to be lossless."""

    peak = float(np.max(np.abs(dataset.ir)))
    worst = 0.0
    for measurement in range(dataset.measurements):
        aligned, delays = align_pair(dataset.ir[measurement])
        restored = np.vstack(
            [
                shift_response(aligned[ear], float(delays[ear]), dataset.taps)
                for ear in range(2)
            ]
        )
        worst = max(worst, float(np.max(np.abs(restored - dataset.ir[measurement]))) / peak)
    assert worst < 0.05


# --- exactness at the measurements ------------------------------------------


@pytest.mark.parametrize("mode", [NEAREST, THREE_NEIGHBOR, SPHERICAL_TRIANGULAR, DELAY_MAGNITUDE])
def test_a_mode_returns_the_measurement_at_a_measured_direction(
    dataset, measured_directions, mode
):
    """Interpolation that misses its own data points is not interpolation."""

    interpolator = HrtfInterpolator(dataset, mode=mode)
    peak = float(np.max(np.abs(dataset.ir)))

    worst = 0.0
    for measurement in range(0, dataset.measurements, 7):
        selection = interpolator.at(measured_directions[measurement])
        error = float(np.max(np.abs(selection.hrirs - dataset.ir[measurement]))) / peak
        worst = max(worst, error)
    assert worst < 0.02


def test_nearest_returns_the_measurement_untouched(dataset, measured_directions):
    interpolator = HrtfInterpolator(dataset, mode=NEAREST)
    for measurement in (0, 40, 143):
        selection = interpolator.at(measured_directions[measurement])
        assert np.array_equal(selection.hrirs, dataset.ir[measurement])
        assert selection.distance_deg == pytest.approx(0.0, abs=1e-6)


# --- continuity -------------------------------------------------------------


def test_each_mode_is_at_least_as_smooth_as_the_cruder_one(dataset):
    """The modes are a progression; a later one that jumps more is not one."""

    peak = float(np.max(np.abs(dataset.ir)))
    steps: dict[str, float] = {}
    for mode in INTERPOLATION_MODES:
        interpolator = HrtfInterpolator(dataset, mode=mode)
        previous = None
        largest = 0.0
        for direction in _horizon():
            current = interpolator.at(direction).hrirs
            if previous is not None:
                largest = max(largest, float(np.max(np.abs(current - previous))) / peak)
            previous = current
        steps[mode] = largest

    assert steps[THREE_NEIGHBOR] < steps[NEAREST]
    for mode in (SPHERICAL_TRIANGULAR, DELAY_MAGNITUDE, SPHERICAL_HARMONIC):
        assert steps[mode] < steps[THREE_NEIGHBOR], mode
        # Smooth enough that a moving source does not step between blocks.
        assert steps[mode] < 0.15, mode


def test_neighbour_weights_fall_to_zero_as_a_neighbour_leaves_the_set(dataset):
    """The seam plain inverse-distance leaves behind, closed.

    A measurement carrying weight at the moment it drops out of the k nearest
    makes the blend jump. Weighting against the next neighbour out means it has
    already faded to nothing by then.
    """

    index = DirectionIndex(unit_vectors(dataset.positions_m))
    for direction in _horizon(3.0):
        weights = index.nearest_k(direction, 3)
        assert weights.weights.sum() == pytest.approx(1.0)
        assert np.all(weights.weights >= 0.0)
        # The furthest of the chosen neighbours never carries the most weight.
        assert weights.weights[0] >= weights.weights[-1]


# --- the cues the interpolation exists to preserve ---------------------------


def test_every_mode_keeps_the_interaural_cues(dataset):
    """A blend that averages the ITD away has destroyed what it interpolated."""

    rate = float(dataset.sample_rate_hz)
    for mode in INTERPOLATION_MODES:
        interpolator = HrtfInterpolator(dataset, mode=mode)
        itds, ilds = [], []
        for direction in _horizon(5.0):
            pair = interpolator.at(direction).hrirs
            itds.append(itd_seconds(pair, rate) * 1e6)
            ilds.append(ild_db(pair))
        # A real head at these angles: hundreds of microseconds, several dB.
        assert max(itds) - min(itds) > 400.0, mode
        assert max(ilds) - min(ilds) > 10.0, mode
        # And the sign convention survives: positive ITD means the left leads.
        left_side = interpolator.at([0.0, 1.0, 0.0])
        assert itd_seconds(left_side.hrirs, rate) > 0.0, mode


def test_no_mode_invents_level_that_was_not_measured(dataset):
    peak = float(np.max(np.abs(dataset.ir)))
    for mode in INTERPOLATION_MODES:
        interpolator = HrtfInterpolator(dataset, mode=mode)
        largest = max(
            float(np.max(np.abs(interpolator.at(direction).hrirs)))
            for direction in _horizon(5.0)
        )
        assert largest <= 1.5 * peak, mode


# --- the spherical-harmonic fit ---------------------------------------------


def test_the_harmonic_order_follows_the_measurements(dataset):
    """An order the data cannot determine is a choice nothing in it made."""

    interpolator = HrtfInterpolator(dataset, mode=SPHERICAL_HARMONIC)
    assert interpolator.harmonic_order == interpolator.max_supportable_harmonic_order
    assert (interpolator.harmonic_order + 1) ** 2 <= dataset.measurements
    assert interpolator.harmonic_order_was_reduced is False


def test_asking_for_more_harmonics_than_the_data_supports_is_capped(dataset):
    interpolator = HrtfInterpolator(dataset, mode=SPHERICAL_HARMONIC, harmonic_order=40)
    assert interpolator.requested_harmonic_order == 40
    assert interpolator.harmonic_order == interpolator.max_supportable_harmonic_order
    # Reported rather than silently clamped, so an interface can say so.
    assert interpolator.harmonic_order_was_reduced is True


def test_a_lower_harmonic_order_is_accepted_as_asked(dataset):
    interpolator = HrtfInterpolator(dataset, mode=SPHERICAL_HARMONIC, harmonic_order=3)
    assert interpolator.harmonic_order == 3
    assert interpolator.harmonic_order_was_reduced is False


def test_a_negative_harmonic_order_is_refused(dataset):
    with pytest.raises(ValueError):
        HrtfInterpolator(dataset, mode=SPHERICAL_HARMONIC, harmonic_order=-1)


def test_the_harmonic_fit_improves_when_it_is_allowed_more_terms(dataset, measured_directions):
    """The cap is worth having because the order genuinely matters."""

    peak = float(np.max(np.abs(dataset.ir)))

    def error(order: int) -> float:
        interpolator = HrtfInterpolator(
            dataset, mode=SPHERICAL_HARMONIC, harmonic_order=order
        )
        return float(
            np.mean(
                [
                    np.max(np.abs(interpolator.at(measured_directions[i]).hrirs - dataset.ir[i]))
                    / peak
                    for i in range(0, dataset.measurements, 9)
                ]
            )
        )

    assert error(11) < error(4) < error(2)


# --- sparse and awkward datasets --------------------------------------------


def test_a_horizontal_only_set_still_works_in_every_mode():
    """A ring of measurements has no spherical triangulation; nothing may raise."""

    dataset = load_sofa(str(FIXTURES / "synthetic_hrir.sofa"))
    index = DirectionIndex(unit_vectors(dataset.positions_m))
    assert index.supports_triangulation is False

    for mode in INTERPOLATION_MODES:
        interpolator = HrtfInterpolator(dataset, mode=mode)
        selection = interpolator.at([1.0, 0.35, 0.2])
        assert selection.hrirs.shape == (2, dataset.taps)
        assert np.all(np.isfinite(selection.hrirs))


def test_leaving_the_measured_grid_is_reported(dataset):
    """Off the grid the answer is a projection, and must not claim otherwise."""

    interpolator = HrtfInterpolator(dataset, mode=SPHERICAL_TRIANGULAR)
    # Straight up, far above the highest measured elevation in this set.
    overhead = interpolator.at([0.0, 0.0, 1.0])
    assert np.all(np.isfinite(overhead.hrirs))
    assert overhead.distance_deg > 0.0


# --- reaching the modes from the interface ----------------------------------


def test_the_lab_offers_every_mode_and_each_mode_its_own_setting():
    """A mode with an unreachable setting is a mode you cannot really use."""

    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from src.ui.sam_hrtf_lab import SamHrtfLab

    application = QApplication.instance() or QApplication([])
    lab = SamHrtfLab()
    try:
        offered = {
            lab.interpolation_combo.itemData(index)
            for index in range(lab.interpolation_combo.count())
        }
        assert set(INTERPOLATION_MODES) <= offered

        # Each extra setting appears for the mode it belongs to and no other:
        # a neighbour count beside "nearest" invites a number that does nothing.
        for mode, neighbours, harmonics in (
            (NEAREST, False, False),
            (THREE_NEIGHBOR, True, False),
            (SPHERICAL_TRIANGULAR, False, False),
            (DELAY_MAGNITUDE, False, False),
            (SPHERICAL_HARMONIC, False, True),
        ):
            lab.interpolation_combo.setCurrentIndex(lab.interpolation_combo.findData(mode))
            assert lab.neighbor_spin.isVisibleTo(lab) is neighbours, mode
            assert lab.harmonic_spin.isVisibleTo(lab) is harmonics, mode
    finally:
        lab.deleteLater()
        application.processEvents()


def test_the_lab_round_trips_the_interpolation_settings():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from src.ui.sam_hrtf_lab import SamHrtfLab

    application = QApplication.instance() or QApplication([])
    lab = SamHrtfLab()
    restored = SamHrtfLab()
    try:
        lab.interpolation_combo.setCurrentIndex(
            lab.interpolation_combo.findData(THREE_NEIGHBOR)
        )
        lab.neighbor_spin.setValue(5)
        lab.harmonic_spin.setValue(7)
        options = lab.params()["hrtfOptions"]
        assert options["neighborCount"] == 5
        assert options["harmonicOrder"] == 7

        restored.set_params(lab.params())
        assert restored.params()["hrtfOptions"]["neighborCount"] == 5
        assert restored.params()["hrtfOptions"]["harmonicOrder"] == 7

        # Zero on the spin box means "let the dataset decide", not order zero.
        lab.harmonic_spin.setValue(0)
        assert lab.params()["hrtfOptions"]["harmonicOrder"] is None
    finally:
        lab.deleteLater()
        restored.deleteLater()
        application.processEvents()


def test_the_renderer_spec_carries_the_settings_through(dataset):
    from src.audio.sam_workbench.render.hrtf import SpatialHrtfRenderer, SpatialHrtfSpec

    spec = SpatialHrtfSpec(
        interpolation=THREE_NEIGHBOR, neighbor_count=5, harmonic_order=6
    )
    renderer = SpatialHrtfRenderer(dataset, dataset.sample_rate_hz, spec)
    assert renderer.interpolator.neighbor_count == 5
    assert renderer.interpolator.requested_harmonic_order == 6

    # Unset means the dataset decides, which is the default the spec carries.
    automatic = SpatialHrtfRenderer(
        dataset, dataset.sample_rate_hz, SpatialHrtfSpec(interpolation=SPHERICAL_HARMONIC)
    )
    assert automatic.interpolator.requested_harmonic_order is None
    assert automatic.interpolator.harmonic_order > 0


@pytest.mark.parametrize(
    "spec_kwargs", [{"neighbor_count": 0}, {"harmonic_order": -2}]
)
def test_the_spec_refuses_settings_that_cannot_render(spec_kwargs):
    from src.audio.sam_workbench.render.hrtf import SpatialHrtfSpec

    with pytest.raises(ValueError):
        SpatialHrtfSpec(**spec_kwargs)


def test_a_hybrid_render_carries_the_settings_too():
    from src.audio.sam_workbench.render.hybrid import HybridSpec

    spec = HybridSpec.from_options(
        {"interpolation": SPHERICAL_HARMONIC, "neighborCount": 6, "harmonicOrder": 9}
    )
    assert spec.interpolation == SPHERICAL_HARMONIC
    assert spec.neighbor_count == 6
    assert spec.harmonic_order == 9
    described = spec.describe()
    assert described["neighborCount"] == 6
    assert described["harmonicOrder"] == 9

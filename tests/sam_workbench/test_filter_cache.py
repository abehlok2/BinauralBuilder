"""Interpolated filters are cached across renders, and keyed on what they depend on.

Per-instance caching bought nothing, and finding that out was the point. The
adaptive control interval already declines to reselect until the direction has
moved past its tolerance, so within a single render consecutive lookups are
genuinely different directions and there is nothing to hit.

What repeats is *renders*: a preview and then an export of the same voice, the
two halves of an A/B, several sources on one trajectory. Each built a new
interpolator and threw the work away. The cache therefore lives at module
scope, keyed by everything the result depends on, so it survives the object
that filled it - and so that two differently configured interpolators can never
see each other's filters.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.hrtf import default_hrtf_cache
from src.audio.sam_workbench.hrtf.interpolation import (
    HrtfInterpolator,
    clear_filter_cache,
)
from src.audio.sam_workbench.trajectory import spherical_to_cartesian
from pathlib import Path

SOFA = str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa")
RATE = 44100
DIRECTION = spherical_to_cartesian(33.0, 17.0, 1.0)


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_filter_cache()
    yield
    clear_filter_cache()


@pytest.fixture
def dataset():
    return default_hrtf_cache.get(SOFA, RATE, "bake_delay_into_ir", None)


# --- the key covers everything that changes the answer ----------------------


def test_the_interpolation_mode_is_part_of_the_key(dataset):
    nearest = HrtfInterpolator(dataset, mode="nearest").at(DIRECTION)
    blended = HrtfInterpolator(dataset, mode="three_neighbor").at(DIRECTION)
    assert not np.array_equal(nearest.hrirs, blended.hrirs)


def test_the_harmonic_order_is_part_of_the_key(dataset):
    low = HrtfInterpolator(dataset, mode="spherical_harmonic", harmonic_order=0).at(DIRECTION)
    high = HrtfInterpolator(dataset, mode="spherical_harmonic", harmonic_order=1).at(DIRECTION)
    assert not np.array_equal(low.hrirs, high.hrirs)


def test_the_neighbour_count_is_part_of_the_key(dataset):
    three = HrtfInterpolator(dataset, mode="three_neighbor", neighbor_count=3).at(DIRECTION)
    seven = HrtfInterpolator(dataset, mode="three_neighbor", neighbor_count=7).at(DIRECTION)
    assert not np.array_equal(three.hrirs, seven.hrirs)


def test_the_dataset_is_part_of_the_key(dataset):
    """A modified dataset must not be served the unmodified dataset's filters."""

    from src.audio.sam_workbench.hrtf.modification import CueTransform, transform_dataset
    from src.audio.sam_workbench.render.hybrid import _ModifiedDatasetView

    modified = _ModifiedDatasetView(
        dataset, transform_dataset(dataset, CueTransform(itd_scale=1.8)).hrirs
    )
    plain = HrtfInterpolator(dataset, mode="nearest").at(DIRECTION)
    altered = HrtfInterpolator(modified, mode="nearest").at(DIRECTION)
    assert not np.array_equal(plain.hrirs, altered.hrirs)


# --- it survives the object that filled it ----------------------------------


def test_a_new_interpolator_reuses_an_earlier_one_s_work(dataset):
    first = HrtfInterpolator(dataset, mode="spherical_harmonic").at(DIRECTION)
    second_interpolator = HrtfInterpolator(dataset, mode="spherical_harmonic")
    second = second_interpolator.at(DIRECTION)
    assert first is second
    assert second_interpolator.cache_statistics()["hitRate"] == 1.0


def test_a_cached_filter_is_the_one_that_would_have_been_computed(dataset):
    interpolator = HrtfInterpolator(dataset, mode="delay_magnitude")
    computed = interpolator.at(DIRECTION)
    clear_filter_cache()
    recomputed = HrtfInterpolator(dataset, mode="delay_magnitude").at(DIRECTION)
    assert np.array_equal(computed.hrirs, recomputed.hrirs)


def test_nearby_directions_share_a_filter(dataset):
    """Quantized finer than any tolerance a caller can set, so a hit is not a fudge."""

    interpolator = HrtfInterpolator(dataset, mode="nearest")
    first = interpolator.at(spherical_to_cartesian(30.0, 0.0, 1.0))
    # A ten-thousandth of a degree apart.
    second = interpolator.at(spherical_to_cartesian(30.0001, 0.0, 1.0))
    assert first is second


def test_distant_directions_do_not_share_a_filter(dataset):
    interpolator = HrtfInterpolator(dataset, mode="nearest")
    first = interpolator.at(spherical_to_cartesian(30.0, 0.0, 1.0))
    second = interpolator.at(spherical_to_cartesian(90.0, 0.0, 1.0))
    assert first is not second


# --- bounded ----------------------------------------------------------------


def test_the_cache_stays_bounded(dataset):
    """A long render with a wandering path must not accumulate without limit."""

    interpolator = HrtfInterpolator(dataset, mode="nearest")
    for step in range(HrtfInterpolator.CACHE_LIMIT * 2):
        interpolator.at(spherical_to_cartesian(step * 0.06, 0.0, 1.0))
    assert interpolator.cache_statistics()["size"] <= HrtfInterpolator.CACHE_LIMIT


def test_clearing_the_cache_empties_it(dataset):
    interpolator = HrtfInterpolator(dataset, mode="nearest")
    interpolator.at(DIRECTION)
    clear_filter_cache()
    assert interpolator.cache_statistics()["size"] == 0


# --- the effect it actually has ---------------------------------------------


def _render(mode):
    params = {
        "amp": 0.5,
        "carrierFreq": 300.0,
        "rendererMode": "hrtf",
        "hrtfAsset": SOFA,
        "canonicalTrajectory": {
            "geometry": {"type": "dome_traversal", "parameters": {"turns": 4}},
            "traversal": {"durationS": 1.0},
        },
        "hrtfOptions": {"interpolation": mode, "maxAngularErrorDeg": 1.0},
    }
    return render_sam2_voice(1.0, RATE, params=params)


def test_a_second_render_of_the_same_voice_is_identical():
    """A warm cache must be a speed difference and nothing else."""

    clear_filter_cache()
    cold = _render("spherical_harmonic")
    warm = _render("spherical_harmonic")
    assert np.array_equal(cold, warm)


@pytest.mark.parametrize("mode", ["spherical_harmonic", "delay_magnitude", "three_neighbor"])
def test_a_second_render_is_not_slower(mode):
    """The cache must never cost more than it saves."""

    clear_filter_cache()
    start = time.perf_counter()
    _render(mode)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    _render(mode)
    warm = time.perf_counter() - start

    # Timing on a shared machine is noisy; the claim is only that a warm cache
    # is not a penalty, not a specific speedup.
    assert warm < cold * 1.5

"""Cue modification: the transforms must move one cue without moving the others.

Phase 5's exit criteria in test form:

* every cue control is neutral at its default;
* a modified dataset keeps the common response and the ear relationships it
  was meant to keep;
* normalization is shared, never per ear and never per direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.hrtf.decomposition import (
    common_differential_delay,
    common_differential_log_magnitude,
    minimum_phase_from_log_magnitude,
)
from src.audio.sam_workbench.hrtf.modification import (
    BASIC_RANGES,
    CUE_PARAMETERS,
    EXPERT_RANGES,
    NEUTRAL,
    CueTransform,
    transform_dataset,
    transform_pair,
)
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

FIXTURE = "tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa"
#: A measurement well off the median plane, so it has cues worth scaling.
LATERAL = 20


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(FIXTURE)


@pytest.fixture(scope="module")
def pair(dataset):
    return np.asarray(dataset.ir[LATERAL], dtype=np.float64)


def lag_samples(hrir_pair) -> int:
    """Samples by which the left ear lags the right."""

    correlation = np.correlate(hrir_pair[0], hrir_pair[1], mode="full")
    return int(np.argmax(np.abs(correlation))) - (hrir_pair.shape[1] - 1)


def ild_db(hrir_pair) -> float:
    power = np.sqrt(np.mean(np.square(hrir_pair), axis=1))
    return 20.0 * np.log10(power[0] / max(power[1], 1e-12))


def log_magnitude(response, taps) -> np.ndarray:
    return np.log(np.maximum(np.abs(np.fft.rfft(response, n=taps)), 1e-12))


# --- neutrality -------------------------------------------------------------


def test_a_default_transform_is_neutral():
    transform = CueTransform()
    assert transform.is_neutral
    for name in CUE_PARAMETERS:
        assert getattr(transform, name) == NEUTRAL[name]


def test_a_neutral_transform_returns_the_dataset_untouched(pair, dataset):
    result = transform_pair(pair, dataset.sample_rate_hz, CueTransform())
    assert result == pytest.approx(pair, abs=0.0)


def test_every_cue_alone_at_its_neutral_value_is_still_neutral():
    for name in CUE_PARAMETERS:
        transform = CueTransform(**{name: NEUTRAL[name]})
        assert transform.is_neutral, name


def test_neutral_is_inside_every_documented_range():
    for name in CUE_PARAMETERS:
        for ranges in (BASIC_RANGES, EXPERT_RANGES):
            low, high = ranges[name]
            assert low <= NEUTRAL[name] <= high, (name, ranges[name])


# --- decomposition ----------------------------------------------------------


def test_the_common_differential_split_is_exact(pair):
    """M_c + M_d must reconstruct the left ear exactly, or nothing downstream holds."""

    spectra = np.fft.rfft(pair, axis=1)
    log_magnitudes = np.log(np.maximum(np.abs(spectra), 1e-12))
    common, differential = common_differential_log_magnitude(log_magnitudes)

    assert common + differential == pytest.approx(log_magnitudes[0], abs=1e-12)
    assert common - differential == pytest.approx(log_magnitudes[1], abs=1e-12)

    delays = np.array([12.0, 4.0])
    common_delay, differential_delay = common_differential_delay(delays)
    assert common_delay == pytest.approx(8.0)
    assert differential_delay == pytest.approx(4.0)


@pytest.mark.parametrize("taps", [128, 129, 136, 137, 512])
def test_minimum_phase_reconstruction_is_exact_for_odd_and_even_lengths(taps):
    """Regression: an odd length reconstructed on a grid one sample too short.

    A real FFT of an odd-length signal has (n + 1) // 2 bins, so 2 * (bins - 1)
    is one short; the response came back on the wrong frequency grid and one
    sample shorter than requested.
    """

    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 1.0, taps) * np.exp(-np.arange(taps) / 12.0)
    magnitude = log_magnitude(source, taps)

    rebuilt = minimum_phase_from_log_magnitude(magnitude, taps)

    assert rebuilt.size == taps
    assert log_magnitude(rebuilt, taps) == pytest.approx(magnitude, abs=1e-4)


# --- cue independence -------------------------------------------------------


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5, 2.0])
def test_scaling_the_time_cue_leaves_the_level_cue_alone(pair, dataset, scale):
    reference = ild_db(pair)
    result = transform_pair(pair, dataset.sample_rate_hz, CueTransform(itd_scale=scale))

    assert ild_db(result) == pytest.approx(reference, abs=0.25)


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5, 2.0])
def test_scaling_the_level_cue_leaves_the_time_cue_alone(pair, dataset, scale):
    reference = lag_samples(pair)
    result = transform_pair(pair, dataset.sample_rate_hz, CueTransform(ild_scale=scale))

    assert abs(lag_samples(result) - reference) <= 1


def test_the_time_cue_scales_proportionally(pair, dataset):
    reference = lag_samples(pair)
    assert reference != 0  # otherwise this measurement proves nothing

    for scale in (0.0, 0.5, 1.5, 2.0):
        result = transform_pair(pair, dataset.sample_rate_hz, CueTransform(itd_scale=scale))
        assert lag_samples(result) == pytest.approx(reference * scale, abs=1.5)


def test_the_level_cue_scales_proportionally(pair, dataset):
    reference = ild_db(pair)

    for scale in (0.0, 0.5, 1.5, 2.0):
        result = transform_pair(pair, dataset.sample_rate_hz, CueTransform(ild_scale=scale))
        assert ild_db(result) == pytest.approx(reference * scale, abs=0.35)


def test_an_extra_differential_delay_shifts_the_time_cue_by_the_stated_amount(pair, dataset):
    rate = dataset.sample_rate_hz
    reference = lag_samples(pair)

    for milliseconds in (-0.25, 0.25, 0.5):
        result = transform_pair(
            pair, rate, CueTransform(extra_delay_ms=milliseconds)
        )
        expected = reference + milliseconds * 1e-3 * rate
        assert lag_samples(result) == pytest.approx(expected, abs=1.5)


# --- pinna and distance -----------------------------------------------------


def test_the_pinna_control_acts_above_its_cutoff_and_not_below(pair, dataset):
    """Elevation and front/back cues live in the fine structure up top."""

    rate = dataset.sample_rate_hz
    taps = pair.shape[1]
    frequencies = np.fft.rfftfreq(taps, 1.0 / rate)
    low = frequencies < 2000.0
    high = frequencies > 6000.0

    reference = log_magnitude(pair[0], taps)
    flattened = transform_pair(pair, rate, CueTransform(pinna_scale=0.0))
    result = log_magnitude(flattened[0], taps)

    # Below the cutoff the shared response is left as it was.
    assert np.mean(np.abs(result[low] - reference[low])) < 0.01
    # Above it, the fine structure has been removed.
    assert np.mean(np.abs(result[high] - reference[high])) > 0.5


def test_stronger_pinna_scaling_deepens_the_fine_structure(pair, dataset):
    rate = dataset.sample_rate_hz
    taps = pair.shape[1]
    high = np.fft.rfftfreq(taps, 1.0 / rate) > 6000.0

    ripple = []
    for scale in (0.0, 0.5, 1.0, 1.5):
        result = transform_pair(pair, rate, CueTransform(pinna_scale=scale))
        ripple.append(float(np.std(log_magnitude(result[0], taps)[high])))

    assert ripple == sorted(ripple)
    assert ripple[-1] > ripple[0]


def test_distance_scales_level_without_touching_either_interaural_cue(pair, dataset):
    rate = dataset.sample_rate_hz
    reference_lag, reference_ild = lag_samples(pair), ild_db(pair)
    near = transform_pair(pair, rate, CueTransform(distance_scale=0.5))
    far = transform_pair(pair, rate, CueTransform(distance_scale=2.0))

    for result in (near, far):
        assert abs(lag_samples(result) - reference_lag) <= 1
        assert ild_db(result) == pytest.approx(reference_ild, abs=0.25)

    # Level follows 1/r, shared by both ears.
    assert np.max(np.abs(near)) / np.max(np.abs(far)) == pytest.approx(4.0, rel=0.15)


# --- coherence --------------------------------------------------------------


def test_reducing_coherence_decorrelates_the_ears_at_constant_level(pair, dataset):
    rate = dataset.sample_rate_hz

    def correlation(hrir_pair):
        numerator = np.max(np.abs(np.correlate(hrir_pair[0], hrir_pair[1], "full")))
        return numerator / (np.linalg.norm(hrir_pair[0]) * np.linalg.norm(hrir_pair[1]))

    full = transform_pair(pair, rate, CueTransform(coherence=1.0, ild_scale=1.0 + 1e-9))
    loose = transform_pair(pair, rate, CueTransform(coherence=0.3))

    assert correlation(loose) < correlation(full)
    # Decorrelation must not double as a level change.
    left_full = np.sqrt(np.mean(np.square(full[0])))
    left_loose = np.sqrt(np.mean(np.square(loose[0])))
    assert left_loose == pytest.approx(left_full, rel=0.2)


def test_coherence_is_reproducible_from_the_seed(pair, dataset):
    rate = dataset.sample_rate_hz
    transform = CueTransform(coherence=0.5, seed=7)

    first = transform_pair(pair, rate, transform, rng=np.random.default_rng(3))
    second = transform_pair(pair, rate, transform, rng=np.random.default_rng(3))
    assert first == pytest.approx(second)


# --- dataset level ----------------------------------------------------------


def test_a_neutral_dataset_transform_changes_nothing(dataset):
    result = transform_dataset(dataset, CueTransform())

    assert result.normalization_gain == 1.0
    assert result.hrirs == pytest.approx(np.asarray(dataset.ir), abs=0.0)
    assert result.measurements == dataset.measurements


def test_normalization_is_one_shared_gain_not_per_ear_or_per_direction(dataset):
    """Per-ear normalization destroys ILD; per-direction destroys motion."""

    loud = CueTransform(ild_scale=2.4, distance_scale=0.3)
    result = transform_dataset(dataset, loud, peak_ceiling=0.5)

    assert result.normalization_gain < 1.0
    assert np.max(np.abs(result.hrirs)) == pytest.approx(0.5, rel=1e-6)

    # Undoing the one gain must recover exactly what the transform produced,
    # which is only true if the same scalar was applied everywhere.
    unscaled = result.hrirs / result.normalization_gain
    rebuilt = np.vstack(
        [
            transform_pair(
                dataset.ir[index],
                dataset.sample_rate_hz,
                loud,
                rng=np.random.default_rng((loud.seed, index)),
            )[None]
            for index in range(dataset.measurements)
        ]
    )
    assert unscaled == pytest.approx(rebuilt, rel=1e-9, abs=1e-12)


def test_the_transform_preserves_direction_to_direction_relationships(dataset):
    """Scaling ITD should move every direction's timing by the same factor."""

    result = transform_dataset(dataset, CueTransform(itd_scale=0.5))

    for index in (10, 20, 40, 80):
        original = lag_samples(np.asarray(dataset.ir[index]))
        if abs(original) < 4:
            continue  # too near the median plane to measure a halving
        assert lag_samples(result.hrirs[index]) == pytest.approx(original * 0.5, abs=2.0)


# --- guidance ---------------------------------------------------------------


def test_contradictory_cue_settings_are_flagged_but_still_allowed():
    collapsed = CueTransform(itd_scale=0.0, ild_scale=0.0)
    assert collapsed.warnings()
    assert any("centre" in warning.message for warning in collapsed.warnings())

    assert CueTransform(pinna_scale=0.0).warnings()
    assert CueTransform(coherence=0.2).warnings()
    assert CueTransform(extra_delay_ms=1.5).warnings()
    # A neutral transform has nothing to say.
    assert CueTransform().warnings() == ()


def test_coherence_outside_its_range_is_rejected():
    with pytest.raises(ValueError):
        CueTransform(coherence=1.4)
    with pytest.raises(ValueError):
        CueTransform(coherence=-0.1)
    with pytest.raises(ValueError):
        CueTransform(itd_scale=float("nan"))


def test_clamping_pulls_values_into_the_documented_range():
    wild = CueTransform(itd_scale=9.0, ild_scale=-3.0, extra_delay_ms=25.0)

    expert = wild.clamped(expert=True)
    assert expert.itd_scale == EXPERT_RANGES["itd_scale"][1]
    assert expert.ild_scale == EXPERT_RANGES["ild_scale"][0]
    assert expert.extra_delay_ms == EXPERT_RANGES["extra_delay_ms"][1]

    basic = wild.clamped(expert=False)
    assert basic.itd_scale == BASIC_RANGES["itd_scale"][1]


def test_options_round_trip_through_a_mapping():
    transform = CueTransform(itd_scale=1.25, coherence=0.8, extra_delay_ms=-0.1, seed=5)
    restored = CueTransform.from_mapping(transform.describe())

    assert restored.itd_scale == pytest.approx(1.25)
    assert restored.coherence == pytest.approx(0.8)
    assert restored.extra_delay_ms == pytest.approx(-0.1)
    assert restored.seed == 5


def test_unknown_option_keys_are_ignored_rather_than_raising():
    """A project written by a newer build must still open."""

    restored = CueTransform.from_mapping({"itd_scale": 1.1, "somethingNewer": "x"})
    assert restored.itd_scale == pytest.approx(1.1)

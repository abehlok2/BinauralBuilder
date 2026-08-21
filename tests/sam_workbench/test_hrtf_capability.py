"""Behaviour of the HRTF capability layer re-landed on main's foundation.

These cover the parts the specification is explicit about: never averaging
unaligned HRIRs, preserving both convolution histories across a filter switch,
applying headphone correction after rendering rather than before, and keeping
the HD650 measurement from being applied to headphones it was not measured on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.dsp.convolution import CrossfadingConvolver, OverlapSaveConvolver
from src.audio.sam_workbench.hrtf import datasets as asset_names
from src.audio.sam_workbench.hrtf import subject_test
from src.audio.sam_workbench.hrtf.coordinates import angular_distance_deg, unit_vectors
from src.audio.sam_workbench.hrtf.headphones import (
    OFF,
    SONICOM_HD650,
    HeadphoneCorrection,
    apply_correction,
)
from src.audio.sam_workbench.hrtf.interpolation import (
    INTERPOLATION_MODES,
    NEAREST,
    HrtfInterpolator,
)
from src.audio.sam_workbench.hrtf.selection import DirectionIndex
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
from src.audio.sam_workbench.render.hrtf import SpatialHrtfSpec, render_spatial_hrtf

FIXTURE = "tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa"


def direction(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def lag_samples(stereo: np.ndarray) -> int:
    """Samples by which the left channel *lags* the right."""

    correlation = np.correlate(stereo[0], stereo[1], mode="full")
    return int(np.argmax(np.abs(correlation))) - (stereo.shape[1] - 1)


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(FIXTURE)


@pytest.fixture(scope="module")
def interpolators(dataset):
    # The aligned representation is the expensive part; build each mode once.
    return {mode: HrtfInterpolator(dataset, mode=mode) for mode in INTERPOLATION_MODES}


# --- SOFA ingestion ---------------------------------------------------------


def test_data_delay_in_samples_is_not_scaled_by_the_sample_rate(dataset):
    """Regression: "samples" also starts with "s" and was read as seconds.

    A five sample delay became five seconds, padding the impulse responses to
    220632 taps. Real SONICOM assets declare samples.
    """

    preserved = load_sofa(FIXTURE, delay_policy="preserve_external_delay")
    assert preserved.delay_samples.max() == pytest.approx(5.0)
    assert preserved.taps == 128
    # Baking pads by the delay plus a small guard, not by the sample rate.
    assert dataset.taps == 137


# --- direction lookup -------------------------------------------------------


def test_unit_vectors_and_angular_distance_use_the_canonical_frame():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert unit_vectors(points) == pytest.approx(np.eye(3))
    # +y is the left ear, so it sits 90 degrees from straight ahead.
    assert angular_distance_deg(np.array([1.0, 0.0, 0.0]), np.eye(3)) == pytest.approx(
        [0.0, 90.0, 90.0]
    )


def test_direction_index_returns_one_measurement_when_the_query_is_on_grid(dataset):
    index = DirectionIndex(unit_vectors(dataset.positions_m))
    on_grid = unit_vectors(dataset.positions_m)[20]

    for weights in (index.nearest(on_grid), index.nearest_k(on_grid, 3)):
        # Blending a coincident measurement with its neighbours would blur a
        # measurement that needs no blending at all.
        assert weights.indices.tolist() == [20]
        assert weights.weights == pytest.approx([1.0])
        assert weights.distance_deg == pytest.approx(0.0, abs=1e-6)


def test_direction_index_weights_are_normalised_off_grid(dataset):
    index = DirectionIndex(unit_vectors(dataset.positions_m))
    query = direction(37.5, 7.5)

    for weights in (index.nearest_k(query, 3), index.barycentric(query)):
        assert weights.weights.sum() == pytest.approx(1.0)
        assert np.all(weights.weights >= 0.0)
        assert not weights.extrapolated


def test_planar_grids_have_no_triangulation_and_say_so():
    horizontal = np.stack(
        [direction(azimuth) for azimuth in range(0, 360, 30)], axis=0
    )
    index = DirectionIndex(horizontal)
    assert not index.supports_triangulation
    # Degrading to the nearest three is fine; pretending it interpolated is not.
    assert index.barycentric(direction(15.0)).extrapolated


# --- interpolation ----------------------------------------------------------


@pytest.mark.parametrize("mode", INTERPOLATION_MODES)
def test_every_mode_produces_a_finite_pair_of_the_right_length(interpolators, mode, dataset):
    selection = interpolators[mode].at(direction(37.5))
    assert selection.hrirs.shape == (2, dataset.taps)
    assert np.all(np.isfinite(selection.hrirs))
    assert selection.mode == mode


def test_modes_agree_where_the_direction_is_measured(interpolators):
    """On a measured direction the blending modes must not smear the filter."""

    reference = interpolators[NEAREST].at(direction(30.0))
    reference_lag = lag_samples(reference.hrirs)

    for mode, interpolator in interpolators.items():
        selection = interpolator.at(direction(30.0))
        assert abs(lag_samples(selection.hrirs) - reference_lag) <= 2, mode


def test_blended_modes_interpolate_between_their_neighbours(interpolators):
    """An off-grid direction should land between the measurements around it."""

    lower = lag_samples(interpolators[NEAREST].at(direction(30.0)).hrirs)
    upper = lag_samples(interpolators[NEAREST].at(direction(45.0)).hrirs)
    low, high = sorted((lower, upper))

    for mode in ("three_neighbor", "spherical_triangular", "delay_magnitude"):
        middle = lag_samples(interpolators[mode].at(direction(37.5)).hrirs)
        assert low <= middle <= high, mode


def test_alignment_keeps_energy_that_naive_averaging_would_cancel(dataset, interpolators):
    """Averaging raw HRIRs comb-filters; aligning first does not."""

    index = DirectionIndex(unit_vectors(dataset.positions_m))
    weights = index.nearest_k(direction(37.5), 3)
    naive = np.tensordot(weights.weights, dataset.ir[weights.indices], axes=(0, 0))
    aligned = interpolators["three_neighbor"].at(direction(37.5)).hrirs

    naive_peak = np.max(np.abs(naive))
    aligned_peak = np.max(np.abs(aligned))
    nearest_peak = np.max(np.abs(interpolators[NEAREST].at(direction(37.5)).hrirs))

    # The aligned blend keeps a peak comparable to a real measurement; the
    # unaligned average loses height as the wavefronts partially cancel.
    assert aligned_peak > naive_peak
    assert aligned_peak == pytest.approx(nearest_peak, rel=0.35)


# --- convolution ------------------------------------------------------------


def test_overlap_save_matches_direct_convolution_and_is_block_size_invariant():
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, 3000)
    taps = rng.normal(0.0, 1.0, 64)
    expected = np.convolve(signal, taps)[: signal.size]

    for block_size in (1, 7, 64, 512, 4096):
        convolver = OverlapSaveConvolver(filter_taps=taps)
        output = np.concatenate(
            [
                convolver.process(signal[start : start + block_size])
                for start in range(0, signal.size, block_size)
            ]
        )
        assert output == pytest.approx(expected, abs=1e-10)


def test_a_filter_switch_preserves_both_histories_and_settles_on_the_new_filter():
    rng = np.random.default_rng(1)
    signal = rng.normal(0.0, 1.0, 8000)
    first = rng.normal(0.0, 1.0, 48)
    second = rng.normal(0.0, 1.0, 48)

    convolver = CrossfadingConvolver(crossfade_ms=10.0, sample_rate_hz=44_100.0)
    convolver.set_filter(first)
    warm = np.concatenate(
        [convolver.process(signal[start : start + 256]) for start in range(0, 4096, 256)]
    )
    assert convolver.set_filter(second) is True

    faded = np.concatenate(
        [convolver.process(signal[start : start + 256]) for start in range(4096, 8000, 256)]
    )
    output = np.concatenate([warm, faded])

    # No step at the switch: the fade mixes two continuous signals rather than
    # restarting one of them from an empty history.
    assert np.all(np.isfinite(output))
    step = np.max(np.abs(np.diff(output)))
    assert step < 8.0 * np.max(np.abs(np.diff(signal))) * np.max(np.abs(second))

    # Once the fade is over the output is exactly the new filter's.
    tail_start = 7000
    reference = np.convolve(signal, second)[: signal.size]
    assert output[tail_start:] == pytest.approx(reference[tail_start:], abs=1e-8)


def test_a_change_arriving_mid_fade_is_queued_rather_than_restarting_the_fade():
    convolver = CrossfadingConvolver(crossfade_ms=20.0, sample_rate_hz=44_100.0)
    # reset() installs the starting filter outright; set_filter() would fade up
    # from the convolver's default passthrough instead.
    convolver.reset(np.array([1.0, 0.0]))
    convolver.process(np.zeros(16))
    assert convolver.set_filter(np.array([0.0, 1.0])) is True
    assert convolver.is_fading
    # Restarting a fade from a partly mixed signal is the artefact the fade exists
    # to prevent, so the request waits.
    assert convolver.set_filter(np.array([0.5, 0.5])) is False


# --- rendering --------------------------------------------------------------


def test_rendering_places_the_source_on_the_correct_side(dataset):
    rng = np.random.default_rng(2)
    mono = rng.normal(0.0, 0.1, 8192)
    rate = int(dataset.sample_rate_hz)

    left = render_spatial_hrtf(mono, direction(90.0), dataset, rate, block_size=512)
    right = render_spatial_hrtf(mono, direction(-90.0), dataset, rate, block_size=512)
    front = render_spatial_hrtf(mono, direction(0.0), dataset, rate, block_size=512)

    def ild_db(stereo):
        power = np.sqrt(np.mean(np.square(stereo), axis=1))
        return 20.0 * np.log10(power[0] / max(power[1], 1e-12))

    assert ild_db(left) > 6.0  # louder in the left ear
    assert ild_db(right) < -6.0
    assert ild_db(front) == pytest.approx(0.0, abs=0.5)

    # The left channel leads for a source on the left and lags for one on the right.
    assert lag_samples(left) < 0
    assert lag_samples(right) > 0
    assert ild_db(left) == pytest.approx(-ild_db(right), abs=0.5)


def test_a_moving_source_introduces_no_discontinuity(dataset):
    rng = np.random.default_rng(3)
    frames = 16384
    mono = rng.normal(0.0, 0.1, frames)
    azimuths = np.linspace(0.0, 180.0, frames)
    path = np.stack(
        [
            np.cos(np.radians(azimuths)),
            np.sin(np.radians(azimuths)),
            np.zeros(frames),
        ],
        axis=1,
    )
    output = render_spatial_hrtf(mono, path, dataset, int(dataset.sample_rate_hz), block_size=512)

    assert np.all(np.isfinite(output))
    # Crossfading between filters must not add a step larger than the source's own.
    assert np.max(np.abs(np.diff(output, axis=1))) < np.max(np.abs(np.diff(mono)))


def test_block_size_does_not_change_a_static_render(dataset):
    rng = np.random.default_rng(4)
    mono = rng.normal(0.0, 0.1, 4096)
    rate = int(dataset.sample_rate_hz)
    reference = render_spatial_hrtf(mono, direction(30.0), dataset, rate, block_size=4096)

    for block_size in (128, 512, 1024):
        candidate = render_spatial_hrtf(mono, direction(30.0), dataset, rate, block_size=block_size)
        assert candidate == pytest.approx(reference, abs=1e-9)


# --- headphone correction ---------------------------------------------------


def test_correction_is_off_by_default_and_leaves_audio_untouched():
    audio = np.random.default_rng(5).normal(0.0, 0.1, (2, 512))
    correction = HeadphoneCorrection.off()

    assert correction.mode == OFF
    assert not correction.is_active
    assert apply_correction(audio, correction) == pytest.approx(audio)


def test_hd650_correction_is_never_selected_implicitly(dataset):
    """The HD650 measurement must not be applied to other headphones."""

    spec = SpatialHrtfSpec()
    assert spec.headphone is None

    rng = np.random.default_rng(6)
    mono = rng.normal(0.0, 0.1, 2048)
    rate = int(dataset.sample_rate_hz)
    plain = render_spatial_hrtf(mono, direction(30.0), dataset, rate)
    explicit_off = render_spatial_hrtf(
        mono, direction(30.0), dataset, rate, spec=SpatialHrtfSpec(headphone=HeadphoneCorrection.off())
    )
    assert plain == pytest.approx(explicit_off)


def test_correction_is_applied_after_binaural_rendering(dataset):
    """Correcting first would change what the HRIRs are convolved with."""

    rate = int(dataset.sample_rate_hz)
    frequencies = np.array([20.0, 1000.0, 20_000.0])
    correction = HeadphoneCorrection.from_equalization_curve(
        frequencies, np.array([6.0, 6.0, 6.0]), rate, taps=127
    )
    assert correction.is_active

    rng = np.random.default_rng(7)
    mono = rng.normal(0.0, 0.1, 4096)
    uncorrected = render_spatial_hrtf(mono, direction(30.0), dataset, rate)
    corrected = render_spatial_hrtf(
        mono, direction(30.0), dataset, rate, spec=SpatialHrtfSpec(headphone=correction)
    )

    expected = apply_correction(uncorrected, correction)
    assert corrected == pytest.approx(expected)


def test_a_correction_pair_preserves_the_interaural_relationship():
    """Each ear gets its own filter, but that is not per-ear normalisation."""

    rate = 44_100
    measurement = np.zeros((2, 64))
    measurement[0, 0] = 1.0
    measurement[1, 0] = 0.5  # the two ears differ by 6 dB
    correction = HeadphoneCorrection.from_measurement(
        measurement, rate, mode=SONICOM_HD650, taps=127
    )
    assert correction.mode == SONICOM_HD650
    assert correction.filters.shape[0] == 2
    # Inverting a quieter measurement yields a louder correction for that ear.
    assert np.max(np.abs(correction.filters[1])) > np.max(np.abs(correction.filters[0]))


# --- asset naming and subject ratings ---------------------------------------


@pytest.mark.parametrize(
    ("name", "subject", "processing"),
    [
        ("P0001_FreeFieldComp_48kHz.sofa", "P0001", "FreeFieldComp"),
        ("P0102_Windowed_44kHz.sofa", "P0102", "Windowed"),
        ("KB0065_FreeFieldCompMinPhase_96kHz.sofa", "KB0065", "FreeFieldCompMinPhase"),
    ],
)
def test_asset_names_are_parsed_into_subject_and_processing(name, subject, processing):
    entry = asset_names.parse_asset_name(name)
    assert entry.subject == subject
    assert entry.processing == processing


def test_kemar_subjects_are_recognised_with_their_ear_size():
    assert asset_names.is_kemar_subject("KB0065")
    assert not asset_names.is_kemar_subject("P0001")
    assert "large ears" in asset_names.describe_subject("KB0065")
    assert "small ears" in asset_names.describe_subject("KB0060")


def test_the_audition_grid_matches_the_requested_directions():
    directions = subject_test.audition_directions()
    azimuths = {round(entry.azimuth_deg) for entry in directions}
    elevations = {round(entry.elevation_deg) for entry in directions}

    assert azimuths == {0, 30, -30, 60, -60, 90, -90, 135, -135, 180}
    assert elevations == {-45, -30, 0, 30, 45, 60}


@pytest.mark.parametrize("signal_type", subject_test.SIGNAL_TYPES)
def test_every_audition_signal_is_finite_and_audible(signal_type):
    material = np.sin(np.linspace(0.0, 200.0, 4096))
    signal = subject_test.make_test_signal(
        signal_type, duration_s=0.25, sample_rate_hz=44_100, material=material
    )
    assert signal.size > 0
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_ratings_round_trip_and_rank_subjects(tmp_path):
    store = subject_test.RatingStore(tmp_path / "ratings.json")
    strong = {criterion: 5.0 for criterion in subject_test.RATING_CRITERIA}
    weak = {criterion: 2.0 for criterion in subject_test.RATING_CRITERIA}

    store.add(subject_test.SubjectRating(subject="P0001", scores=strong, profile="carrier"))
    store.add(subject_test.SubjectRating(subject="P0002", scores=weak, profile="carrier"))
    store.save()

    reloaded = subject_test.RatingStore(tmp_path / "ratings.json")
    reloaded.load()
    assert set(reloaded.subjects()) == {"P0001", "P0002"}

    ranked = subject_test.rank_subjects(reloaded.for_profile("carrier"))
    assert [entry.subject for entry in ranked] == ["P0001", "P0002"]
    assert ranked[0].score > ranked[1].score


def test_profiles_are_kept_separate_so_a_voice_can_prefer_a_different_subject(tmp_path):
    """A subject that suits broadband noise need not suit a low carrier."""

    store = subject_test.RatingStore(tmp_path / "ratings.json")
    high = {criterion: 5.0 for criterion in subject_test.RATING_CRITERIA}
    low = {criterion: 1.0 for criterion in subject_test.RATING_CRITERIA}

    store.add(subject_test.SubjectRating(subject="P0001", scores=high, profile="noise"))
    store.add(subject_test.SubjectRating(subject="P0001", scores=low, profile="carrier"))
    store.add(subject_test.SubjectRating(subject="P0002", scores=low, profile="noise"))
    store.add(subject_test.SubjectRating(subject="P0002", scores=high, profile="carrier"))

    assert [entry.subject for entry in subject_test.rank_subjects(store.for_profile("noise"))][0] == "P0001"
    assert [entry.subject for entry in subject_test.rank_subjects(store.for_profile("carrier"))][0] == "P0002"


def test_the_first_block_installs_its_filter_without_fading_in(dataset):
    """A render must not open by fading up from the dry mono source.

    The convolver starts on a passthrough filter, so fading into the first
    HRIR would let the uncoloured mono signal through at the top of every
    render - audible as a moment of un-spatialised sound.
    """

    rate = int(dataset.sample_rate_hz)
    mono = np.ones(2048)  # steady, so any fade-in shows up as a ramp

    at_once = render_spatial_hrtf(mono, direction(90.0), dataset, rate, block_size=2048)
    in_blocks = render_spatial_hrtf(mono, direction(90.0), dataset, rate, block_size=256)
    assert at_once == pytest.approx(in_blocks, abs=1e-9)

    # The opening samples already carry the full interaural level difference.
    opening = at_once[:, 64:256]
    settled = at_once[:, 1024:]
    opening_ild = 20.0 * np.log10(
        np.max(np.abs(opening[0])) / max(np.max(np.abs(opening[1])), 1e-12)
    )
    settled_ild = 20.0 * np.log10(
        np.max(np.abs(settled[0])) / max(np.max(np.abs(settled[1])), 1e-12)
    )
    assert opening_ild == pytest.approx(settled_ild, abs=1.0)

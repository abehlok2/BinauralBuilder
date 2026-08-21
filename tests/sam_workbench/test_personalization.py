"""Localization testing, experiment export, measurement import, mesh validation.

The rules this file protects:

* a blinded localization session ranks candidates reproducibly;
* an experiment export carries complete acoustic provenance;
* measurement tools enforce their calibration and quality checks rather than
  advising about them;
* a simulated HRTF is checked against physical plausibility before it is used.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from scipy.signal import fftconvolve

from src.audio.sam_workbench.experiment.localization import (
    LocalizationSession,
    TrialResponse,
    angular_error_deg,
    build_session,
    is_front_back_reversal,
    rank_candidates,
    score_responses,
)
from src.audio.sam_workbench.experiment.session import (
    Condition,
    ExperimentDesign,
    ExperimentResult,
    Rating,
    conditions_from_assets,
    export_result,
    load_result,
)
from src.audio.sam_workbench.hrtf import measurement as measure
from src.audio.sam_workbench.hrtf import mesh2hrtf
from src.audio.sam_workbench.hrtf.sofa_io import load_sofa

FIXTURE = "tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa"
CANDIDATES = ["P0001", "P0002", "KB0065"]


@pytest.fixture(scope="module")
def dataset():
    return load_sofa(FIXTURE)


@pytest.fixture
def session() -> LocalizationSession:
    return build_session(CANDIDATES, seed=42, repeats=2)


def responses_for(session: LocalizationSession, behaviour) -> list[TrialResponse]:
    """Simulate a listener whose accuracy depends on the candidate."""

    rng = np.random.default_rng(0)
    out = []
    for trial in session.trials:
        out.append(behaviour(trial, rng))
    return out


# --- localization sessions --------------------------------------------------


def test_a_session_is_balanced_across_candidates(session):
    """No candidate may be advantaged by getting easier or fewer trials."""

    counts = {
        candidate: sum(1 for trial in session.trials if trial.candidate == candidate)
        for candidate in session.candidates
    }
    assert len(set(counts.values())) == 1

    kinds = {trial.kind for trial in session.trials}
    assert kinds == {"front_back", "lateral", "elevation", "distance"}


def test_labels_are_opaque_and_the_listener_never_sees_the_candidate(session):
    blinded = session.blinded()

    assert all("candidate" not in entry for entry in blinded)
    assert all(entry["label"].startswith("set-") for entry in blinded)
    # No candidate name may leak into a label.
    for candidate in CANDIDATES:
        assert not any(candidate in entry["label"] for entry in blinded)


def test_the_same_seed_reproduces_the_same_session():
    first = build_session(CANDIDATES, seed=42, repeats=2)
    second = build_session(CANDIDATES, seed=42, repeats=2)
    different = build_session(CANDIDATES, seed=7, repeats=2)

    assert [trial.id for trial in first.trials] == [trial.id for trial in second.trials]
    assert [trial.id for trial in first.trials] != [trial.id for trial in different.trials]


def test_a_session_needs_candidates():
    with pytest.raises(ValueError):
        build_session([])


# --- scoring ----------------------------------------------------------------


def test_angular_error_is_measured_on_the_sphere():
    """Near the poles a large azimuth difference is a small angle."""

    assert angular_error_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(10.0, abs=1e-6)
    assert angular_error_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(10.0, abs=1e-6)
    # Ninety degrees of azimuth at eighty degrees elevation is not ninety degrees.
    assert angular_error_deg(0.0, 80.0, 90.0, 80.0) < 20.0
    # And the wrap is handled: 350 and 10 are twenty degrees apart.
    assert angular_error_deg(350.0, 0.0, 10.0, 0.0) == pytest.approx(20.0, abs=1e-6)


def test_front_back_reversals_are_categorical_and_exclude_the_lateral_axis():
    assert is_front_back_reversal(30.0, 150.0)
    assert is_front_back_reversal(150.0, 30.0)
    assert is_front_back_reversal(0.0, 180.0)
    assert not is_front_back_reversal(30.0, 40.0)
    # At ninety degrees the front/back distinction is not what is being tested.
    assert not is_front_back_reversal(90.0, 95.0)
    assert not is_front_back_reversal(-90.0, -85.0)


def test_a_perfect_listener_scores_zero_and_ranks_first(session):
    def perfect(trial, _rng):
        return TrialResponse(trial.id, trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 5)

    scores = score_responses(session, responses_for(session, perfect))

    for score in scores:
        assert score.weighted_error == pytest.approx(0.0, abs=1e-9)
        assert score.front_back_reversals == 0


def test_reversals_and_guesses_rank_below_accuracy(session):
    def behaviour(trial, rng):
        if trial.candidate == "P0001":
            return TrialResponse(trial.id, trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 5)
        if trial.candidate == "P0002":  # confuses front with back
            return TrialResponse(trial.id, 180.0 - trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 4)
        return TrialResponse(  # guesses
            trial.id, rng.uniform(-180.0, 180.0), rng.uniform(-45.0, 60.0), trial.distance_m, 2
        )

    scores = {score.candidate: score for score in score_responses(session, responses_for(session, behaviour))}

    assert rank_candidates(session, responses_for(session, behaviour))[0] == "P0001"
    assert scores["P0001"].weighted_error < scores["KB0065"].weighted_error
    assert scores["P0002"].front_back_reversals > scores["KB0065"].front_back_reversals
    # The reverser is confident and wrong, which should cost it.
    assert scores["P0002"].weighted_error > scores["P0001"].weighted_error


def test_confidence_weights_a_wrong_answer(session):
    def wrong(confidence):
        def behaviour(trial, _rng):
            return TrialResponse(trial.id, trial.azimuth_deg + 40.0, trial.elevation_deg, trial.distance_m, confidence)

        return behaviour

    confident = score_responses(session, responses_for(session, wrong(5)))[0]
    hesitant = score_responses(session, responses_for(session, wrong(1)))[0]

    assert confident.weighted_error > hesitant.weighted_error


def test_confidence_weighting_can_be_switched_off(session):
    def behaviour(trial, _rng):
        return TrialResponse(trial.id, trial.azimuth_deg + 20.0, trial.elevation_deg, trial.distance_m, 5)

    weighted = score_responses(session, responses_for(session, behaviour))[0]
    plain = score_responses(session, responses_for(session, behaviour), use_confidence=False)[0]

    assert weighted.weighted_error != pytest.approx(plain.weighted_error)


def test_inconsistent_repeats_are_penalised(session):
    """A set that answers differently each time is unreliable even if unbiased."""

    def steady(trial, _rng):
        return TrialResponse(trial.id, trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 3)

    def erratic(trial, rng):
        jitter = 40.0 * (1 if trial.repeat_index else -1)
        return TrialResponse(trial.id, trial.azimuth_deg + jitter, trial.elevation_deg, trial.distance_m, 3)

    calm = score_responses(session, responses_for(session, steady))[0]
    jumpy = score_responses(session, responses_for(session, erratic))[0]

    assert jumpy.inconsistency_deg > calm.inconsistency_deg
    assert jumpy.weighted_error > calm.weighted_error


def test_an_unfinished_session_is_visibly_unfinished(session):
    def perfect(trial, _rng):
        return TrialResponse(trial.id, trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 5)

    partial = responses_for(session, perfect)[:10]
    scores = score_responses(session, partial)

    assert any(score.responses < score.trials for score in scores)
    assert all(score.trials > 0 for score in scores)


def test_a_candidate_with_no_responses_ranks_last(session):
    def only_one(trial, _rng):
        if trial.candidate != "P0001":
            return TrialResponse(trial.id, trial.azimuth_deg, trial.elevation_deg, trial.distance_m, 3)
        return TrialResponse("not-a-trial", 0.0)

    scores = score_responses(session, responses_for(session, only_one))
    unanswered = [score for score in scores if score.candidate == "P0001"][0]

    assert unanswered.responses == 0
    assert unanswered.weighted_error == float("inf")
    assert scores[-1].candidate == "P0001"


def test_an_impossible_confidence_is_rejected():
    with pytest.raises(ValueError):
        TrialResponse("t", 0.0, confidence=0)
    with pytest.raises(ValueError):
        TrialResponse("t", 0.0, confidence=6)


# --- experiment design and export -------------------------------------------


@pytest.fixture
def design() -> ExperimentDesign:
    conditions = conditions_from_assets(
        [
            {"subject": subject, "path": f"/data/{subject}.sofa", "sha256": "a" * 64, "dataset": "SONICOM"}
            for subject in CANDIDATES
        ]
    )
    return ExperimentDesign(conditions=conditions, repeats=2, seed=11, name="subject comparison")


def test_presentation_order_is_balanced_and_reproducible(design):
    runs = design.build_runs()
    counts = {
        condition.id: sum(1 for run in runs if run.condition_id == condition.id)
        for condition in design.conditions
    }

    assert len(runs) == len(design.conditions) * design.repeats
    assert set(counts.values()) == {design.repeats}
    assert [run.condition_id for run in runs] == [run.condition_id for run in design.build_runs()]


def test_the_listener_view_hides_which_condition_is_playing(design):
    run = design.build_runs()[0]

    assert "conditionId" not in run.describe(reveal=False)
    assert run.describe(reveal=True)["conditionId"] == run.condition_id


def test_duplicate_condition_ids_are_rejected():
    with pytest.raises(ValueError):
        ExperimentDesign(conditions=(Condition(id="a"), Condition(id="a")))
    with pytest.raises(ValueError):
        ExperimentDesign(conditions=())


def test_an_export_carries_complete_acoustic_provenance(design, tmp_path):
    """Exit criterion: someone else must be able to say what was played."""

    runs = design.build_runs()
    ratings = tuple(Rating(run.index, {"preference": 4.0}) for run in runs)
    result = ExperimentResult(design=design, runs=runs, ratings=ratings, listener="L01")

    destination = export_result(result, tmp_path / "experiment.json")
    payload = load_result(destination)

    assert payload["applicationVersion"]
    assert payload["design"]["seed"] == 11
    assert payload["completion"] == pytest.approx(1.0)

    condition = payload["design"]["conditions"][0]
    for key in (
        "assetPath",
        "assetSha256",
        "renderer",
        "interpolation",
        "delayPolicy",
        "sampleRateHz",
        "crossfadeMs",
        "cue",
        "anchor",
        "headphoneCorrection",
        "outputGainDb",
    ):
        assert key in condition, key


def test_a_blinded_export_withholds_the_mapping(design, tmp_path):
    runs = design.build_runs()
    result = ExperimentResult(design=design, runs=runs)

    payload = load_result(export_result(result, tmp_path / "blind.json", reveal=False))

    assert "reveal" not in payload
    assert all("conditionId" not in run for run in payload["runs"])
    # The conditions themselves are still described: what was played is not secret,
    # only which label was which.
    assert payload["design"]["conditions"]


def test_a_condition_without_a_hash_is_named_as_unreproducible():
    design = ExperimentDesign(
        conditions=(
            Condition(id="pinned", renderer="hrtf", asset_path="/a.sofa", asset_sha256="a" * 64),
            Condition(id="loose", renderer="hrtf", asset_path="/b.sofa"),
            Condition(id="abstract", renderer="abstract_pm"),
        )
    )

    assert design.unreproducible() == ("loose",)


def test_the_export_is_written_atomically(design, tmp_path):
    result = ExperimentResult(design=design, runs=design.build_runs())
    destination = export_result(result, tmp_path / "nested" / "experiment.json")

    assert destination.exists()
    assert not list(tmp_path.glob("**/*.partial"))


def test_partial_completion_is_reported(design):
    runs = design.build_runs()
    result = ExperimentResult(design=design, runs=runs, ratings=(Rating(runs[0].index, {"preference": 3.0}),))

    assert result.completion == pytest.approx(1.0 / len(runs))
    assert result.rating_for(runs[0].index) is not None
    assert result.rating_for(9999) is None


# --- measurement ------------------------------------------------------------

RATE = 48_000.0
SWEEP_S = 0.5


@pytest.fixture(scope="module")
def sweep_pair():
    return (
        measure.exponential_sweep(20.0, 20_000.0, SWEEP_S, RATE),
        measure.inverse_sweep(20.0, 20_000.0, SWEEP_S, RATE),
    )


def test_a_sweep_deconvolved_with_its_inverse_is_an_impulse(sweep_pair):
    """Which is the property the whole measurement chain rests on."""

    sweep, inverse = sweep_pair
    result = measure.deconvolve(sweep, inverse)
    peak = int(np.argmax(np.abs(result)))

    assert result[peak] == pytest.approx(1.0, abs=1e-9)
    away = np.concatenate([result[: peak - 32], result[peak + 32 :]])
    assert np.max(np.abs(away)) < 0.05


def test_a_known_response_is_recovered_from_a_swept_recording(sweep_pair):
    sweep, inverse = sweep_pair
    known = np.zeros(256)
    known[10], known[35], known[70] = 1.0, -0.4, 0.2

    recovered = measure.deconvolve(fftconvolve(sweep, known), inverse)
    peak = int(np.argmax(np.abs(recovered)))
    segment = recovered[peak - 10 : peak - 10 + known.size]
    segment = segment / np.max(np.abs(segment))

    assert segment[10] == pytest.approx(1.0, abs=0.05)
    assert segment[35] == pytest.approx(-0.4, abs=0.08)
    assert segment[70] == pytest.approx(0.2, abs=0.08)


def test_dividing_by_the_reference_removes_the_loudspeaker(sweep_pair):
    sweep, inverse = sweep_pair
    known = np.zeros(256)
    known[10], known[35] = 1.0, -0.4
    speaker = np.zeros(64)
    speaker[0], speaker[3], speaker[9] = 1.0, 0.6, -0.3

    raw = measure.deconvolve(fftconvolve(sweep, fftconvolve(known, speaker)), inverse)
    reference = measure.deconvolve(fftconvolve(sweep, speaker), inverse)
    corrected = measure.divide_reference(raw, reference, sample_rate_hz=RATE)

    def deviation(response):
        length = 2048
        peak = int(np.argmax(np.abs(response)))
        segment = response[max(0, peak - 10) : max(0, peak - 10) + 256]
        measured = np.abs(np.fft.rfft(segment, length))
        truth = np.abs(np.fft.rfft(known, length))
        frequencies = np.fft.rfftfreq(length, 1.0 / RATE)
        band = (frequencies > 200.0) & (frequencies < 15_000.0)
        measured = measured / np.mean(measured[band])
        truth = truth / np.mean(truth[band])
        return float(
            np.mean(np.abs(20.0 * np.log10(np.maximum(measured[band], 1e-9) / np.maximum(truth[band], 1e-9))))
        )

    assert deviation(corrected) < deviation(raw) / 4.0


def test_windowing_cuts_before_a_room_reflection():
    response = np.zeros(2048)
    response[100], response[140], response[600] = 1.0, 0.35, 0.5

    windowed, cut = measure.window_before_reflection(response, RATE, max_length_ms=20.0)

    assert cut == 140
    assert windowed[100] == pytest.approx(1.0)
    assert np.all(windowed[cut:] == 0.0)


def test_the_measurement_tools_are_off_by_default(monkeypatch, sweep_pair):
    """Route A costs nothing and should be tried first; this one has a risk."""

    monkeypatch.delenv(measure.ADVANCED_FLAG, raising=False)
    assert not measure.advanced_enabled()

    sweep, _ = sweep_pair
    with pytest.raises(measure.MeasurementError) as caught:
        measure.import_measurement([sweep], sample_rate_hz=RATE, sweep_duration_s=SWEEP_S)
    assert "advanced" in str(caught.value)
    assert "localization test" in str(caught.value)


@pytest.fixture
def advanced(monkeypatch):
    monkeypatch.setenv(measure.ADVANCED_FLAG, "1")
    return True


def test_a_clean_measurement_is_accepted(advanced, sweep_pair):
    sweep, _ = sweep_pair
    known = np.zeros(128)
    known[10] = 1.0
    recording = fftconvolve(sweep, known)
    recording = recording / np.max(np.abs(recording)) * 0.5

    response, report = measure.import_measurement(
        [recording, recording], sample_rate_hz=RATE, sweep_duration_s=SWEEP_S
    )

    assert report.passes
    assert not report.clipped
    assert report.repeatability == pytest.approx(1.0)
    assert np.any(response != 0.0)


def test_a_clipped_recording_is_refused(advanced, sweep_pair):
    sweep, _ = sweep_pair
    clipped = np.clip(sweep * 4.0, -1.0, 1.0)

    with pytest.raises(measure.MeasurementError) as caught:
        measure.import_measurement([clipped, clipped], sample_rate_hz=RATE, sweep_duration_s=SWEEP_S)
    assert "peaks at" in str(caught.value)


def test_recordings_that_disagree_are_refused(advanced, sweep_pair):
    """A subject who moved between takes has not measured one head twice."""

    sweep, _ = sweep_pair
    recording = fftconvolve(sweep, np.array([1.0])) * 0.5

    with pytest.raises(measure.MeasurementError) as caught:
        measure.import_measurement(
            [recording, np.roll(recording, 60)], sample_rate_hz=RATE, sweep_duration_s=SWEEP_S
        )
    assert "correlate" in str(caught.value) or "onsets" in str(caught.value)


def test_a_noisy_measurement_is_refused_on_signal_to_noise(advanced, sweep_pair):
    sweep, _ = sweep_pair
    rng = np.random.default_rng(0)
    takes = [(sweep * 0.02 + rng.normal(0.0, 0.2, sweep.size)) for _ in range(2)]

    with pytest.raises(measure.MeasurementError):
        measure.import_measurement(takes, sample_rate_hz=RATE, sweep_duration_s=SWEEP_S)


def test_a_sweep_above_nyquist_is_rejected():
    with pytest.raises(ValueError):
        measure.exponential_sweep(20.0, 30_000.0, 1.0, RATE)
    with pytest.raises(ValueError):
        measure.exponential_sweep(0.0, 20_000.0, 1.0, RATE)


# --- Mesh2HRTF -------------------------------------------------------------


def simulated(dataset, **overrides):
    """A stand-in for a solver's output, with one thing set wrong at a time."""

    fake = types.SimpleNamespace(
        ir=dataset.ir,
        positions_m=dataset.positions_m,
        receiver_positions_m=np.array([[0.0, 0.09, 0.0], [0.0, -0.09, 0.0]]),
        sample_rate_hz=dataset.sample_rate_hz,
    )
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


def test_the_import_guidance_is_numbered_and_complete():
    guidance = mesh2hrtf.import_guidance()

    assert len(guidance) == len(mesh2hrtf.IMPORT_STEPS)
    assert guidance[0].startswith("1.")
    assert any("scale" in step.lower() for step in guidance)
    assert any("receiver" in step.lower() for step in guidance)


def test_a_well_formed_simulation_passes(dataset):
    assert mesh2hrtf.validate_simulated(simulated(dataset)) == ()


def test_a_mesh_scaled_in_millimetres_is_caught(dataset):
    wrong = simulated(dataset, receiver_positions_m=np.array([[0.0, 90.0, 0.0], [0.0, -90.0, 0.0]]))
    issues = mesh2hrtf.validate_simulated(wrong)

    assert issues
    assert any("scale" in issue.message for issue in issues)


def test_swapped_ears_are_caught(dataset):
    swapped = simulated(dataset, receiver_positions_m=np.array([[0.0, -0.09, 0.0], [0.0, 0.09, 0.0]]))
    issues = mesh2hrtf.validate_simulated(swapped)

    assert any("mirror" in issue.message for issue in issues)


def test_wrong_axes_are_caught(dataset):
    rotated = simulated(
        dataset, receiver_positions_m=np.array([[0.09, 0.001, 0.0], [-0.09, -0.001, 0.0]])
    )
    issues = mesh2hrtf.validate_simulated(rotated)

    assert any("convention" in issue.message for issue in issues)


def test_a_horizon_only_grid_is_caught(dataset):
    azimuths = np.radians(np.arange(0.0, 360.0, 15.0))
    flat = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)], axis=1)
    issues = mesh2hrtf.validate_simulated(simulated(dataset, positions_m=flat))

    assert any("elevation" in issue.message for issue in issues)


def test_missing_or_silent_data_is_an_error(dataset):
    one_ear = mesh2hrtf.validate_simulated(
        simulated(dataset, receiver_positions_m=np.array([[0.0, 0.09, 0.0]]))
    )
    silent = mesh2hrtf.validate_simulated(simulated(dataset, ir=np.zeros_like(dataset.ir)))

    assert any(issue.severity == "error" for issue in one_ear)
    assert any(issue.severity == "error" for issue in silent)


def test_strict_mode_promotes_plausibility_warnings(dataset):
    wrong = simulated(dataset, receiver_positions_m=np.array([[0.0, 90.0, 0.0], [0.0, -90.0, 0.0]]))

    assert all(issue.severity == "warning" for issue in mesh2hrtf.validate_simulated(wrong))
    assert any(issue.severity == "error" for issue in mesh2hrtf.validate_simulated(wrong, strict=True))


def test_a_simulation_compared_against_itself_is_plausible(dataset):
    comparison = mesh2hrtf.compare_to_baseline(dataset, dataset)

    assert comparison.plausible
    assert comparison.itd_ratio == pytest.approx(1.0)
    assert comparison.correlation == pytest.approx(1.0, abs=1e-6)


def test_swapped_ears_show_up_as_a_reversed_cue(dataset):
    """Caught a second way: the delay runs opposite to the baseline."""

    swapped = types.SimpleNamespace(
        ir=dataset.ir[:, ::-1, :], positions_m=dataset.positions_m, sample_rate_hz=dataset.sample_rate_hz
    )
    comparison = mesh2hrtf.compare_to_baseline(swapped, dataset)

    assert comparison.correlation < 0.0
    assert not comparison.plausible
    assert any("swapped" in issue.message or "reversed" in issue.message for issue in comparison.issues)


def test_an_implausible_interaural_delay_is_an_error(dataset):
    """Ten times a real head's delay is a scale error, not an anatomy."""

    stretched = types.SimpleNamespace(
        ir=dataset.ir, positions_m=dataset.positions_m, sample_rate_hz=dataset.sample_rate_hz / 10.0
    )
    comparison = mesh2hrtf.compare_to_baseline(stretched, dataset)

    assert comparison.max_itd_us > mesh2hrtf.PLAUSIBLE_MAX_ITD_US[1]
    assert not comparison.plausible

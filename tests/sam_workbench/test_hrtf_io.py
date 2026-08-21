"""SOFA ingest: reading, validating, preparing, and caching an HRTF asset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.conventions import spherical_to_cartesian_m
from src.audio.sam_workbench.hrtf import (
    BAKE_DELAY,
    KEMAR_VARIANTS,
    PRESERVE_DELAY,
    PROCESSING_VARIANTS,
    SONICOM_SAMPLE_RATES_HZ,
    HrtfCache,
    angular_distance_deg,
    available_backend,
    describe_subject,
    discover_assets,
    file_sha256,
    is_kemar_subject,
    load_sofa,
    parse_asset_name,
    positions_to_canonical,
    prepare_from_path,
    storage_report,
    unit_vectors,
    validate_sofa_file,
)
from src.audio.sam_workbench.hrtf import storage

pytestmark = pytest.mark.skipif(
    available_backend() is None, reason="no SOFA backend (sofar or h5py) installed"
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SMALL_SOFA = FIXTURES / "synthetic_hrir.sofa"
SONICOM_SOFA = FIXTURES / "synthetic_sonicom_hrir.sofa"
SAMPLE_RATE = 44_100


# --- coordinates ------------------------------------------------------------


@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected"),
    [(0.0, 0.0, (1.0, 0.0, 0.0)), (90.0, 0.0, (0.0, 1.0, 0.0)), (0.0, 90.0, (0.0, 0.0, 1.0))],
)
def test_spherical_sofa_positions_reach_the_canonical_frame(azimuth, elevation, expected):
    positions = positions_to_canonical(
        np.array([[azimuth, elevation, 1.0]]), "spherical", "degree, degree, metre"
    )
    np.testing.assert_allclose(positions[:, 0], expected, atol=1e-12)


def test_radian_and_centimetre_units_are_honoured():
    radians = positions_to_canonical(
        np.array([[np.pi / 2, 0.0, 1.0]]), "spherical", "radian, radian, metre"
    )
    np.testing.assert_allclose(radians[:, 0], (0.0, 1.0, 0.0), atol=1e-12)

    centimetres = positions_to_canonical(
        np.array([[0.0, 0.0, 150.0]]), "spherical", "degree, degree, centimetre"
    )
    assert centimetres[0, 0] == pytest.approx(1.5)


def test_cartesian_sofa_positions_pass_through():
    positions = positions_to_canonical(np.array([[1.0, 2.0, 3.0]]), "cartesian", "metre")
    np.testing.assert_allclose(positions[:, 0], (1.0, 2.0, 3.0))


def test_unknown_position_type_is_rejected():
    with pytest.raises(ValueError):
        positions_to_canonical(np.array([[0.0, 0.0, 1.0]]), "cylindrical", "metre")


def test_angular_distance_handles_the_azimuth_wrap():
    """179° and -179° are two degrees apart, not 358."""

    left = spherical_to_cartesian_m(179.0, 0.0, 1.0)
    right = spherical_to_cartesian_m(-179.0, 0.0, 1.0)
    distance = angular_distance_deg(left, right.reshape(3, 1))
    assert float(distance.ravel()[0]) == pytest.approx(2.0, abs=1e-6)


def test_unit_vectors_normalize_without_dividing_by_zero():
    vectors = unit_vectors(np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_allclose(vectors[:, 0], (1.0, 0.0, 0.0))
    assert np.all(np.isfinite(vectors))


# --- reading ----------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sofar", "h5py"])
def test_both_backends_read_the_same_file(backend):
    pytest.importorskip(backend)
    first = load_sofa(SONICOM_SOFA, backend="sofar")
    second = load_sofa(SONICOM_SOFA, backend=backend)
    np.testing.assert_allclose(first.impulse_responses, second.impulse_responses, atol=1e-12)
    assert first.sample_rate_hz == second.sample_rate_hz
    assert first.sha256 == second.sha256


def test_reading_records_provenance():
    sofa_file = load_sofa(SONICOM_SOFA)
    assert sofa_file.convention == "SimpleFreeFieldHRIR"
    assert sofa_file.measurements == 144
    assert sofa_file.receivers == 2
    assert sofa_file.database_name
    assert sofa_file.license_note
    assert sofa_file.sha256 == file_sha256(SONICOM_SOFA)
    assert sofa_file.has_external_delay is True


def test_reading_is_silent():
    """A library that prints on every read would spam the application console."""

    import contextlib
    import io

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        load_sofa(SMALL_SOFA)
    assert captured.getvalue() == ""


def test_a_missing_file_says_so():
    with pytest.raises(FileNotFoundError):
        load_sofa(FIXTURES / "not_here.sofa")


def test_a_non_sofa_file_is_rejected(tmp_path):
    decoy = tmp_path / "decoy.sofa"
    decoy.write_bytes(b"not a SOFA file")
    with pytest.raises(Exception):
        load_sofa(decoy)


# --- validation -------------------------------------------------------------


def test_a_good_asset_is_renderable():
    report = validate_sofa_file(load_sofa(SONICOM_SOFA), project_sample_rate_hz=SAMPLE_RATE)
    assert report.is_renderable
    assert report.quality == "normal"
    assert report.fatal == ()


def test_the_report_describes_what_the_file_contains():
    report = validate_sofa_file(load_sofa(SONICOM_SOFA))
    metadata = report.metadata
    assert metadata["measurements"] == 144
    assert metadata["receivers"] == 2
    assert metadata["data_delay_nonzero"] is True
    assert metadata["elevation_deg_range"][0] < 0 < metadata["elevation_deg_range"][1]
    assert metadata["median_direction_spacing_deg"] < 20.0
    assert metadata["license"]


def test_a_sample_rate_mismatch_is_a_warning_not_a_blocker():
    report = validate_sofa_file(load_sofa(SONICOM_SOFA), project_sample_rate_hz=96_000)
    assert report.is_renderable
    assert any("resampled" in issue.message for issue in report.warnings)
    assert report.quality == "suspected outlier"


def test_a_horizontal_only_asset_warns_about_elevation():
    report = validate_sofa_file(load_sofa(SMALL_SOFA))
    assert report.is_renderable
    assert any("horizontal only" in issue.message for issue in report.warnings)


def test_the_report_serializes_for_the_manifest():
    payload = validate_sofa_file(load_sofa(SONICOM_SOFA)).to_dict()
    assert payload["renderable"] is True
    assert payload["quality"] in ("normal", "suspected outlier")
    assert "metadata" in payload and "issues" in payload


# --- preparation ------------------------------------------------------------


def test_preparation_converts_coordinates_and_keeps_provenance():
    prepared = prepare_from_path(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    assert prepared.measurements == 144
    assert prepared.directions.shape == (3, 144)
    np.testing.assert_allclose(np.linalg.norm(prepared.directions, axis=0), 1.0, atol=1e-12)
    assert prepared.asset.sha256 == file_sha256(SONICOM_SOFA)
    assert prepared.asset.subject == "P0001"
    assert prepared.describe()["delay_policy"] == BAKE_DELAY


def test_baking_the_delay_moves_the_responses_and_zeroes_the_field():
    baked = prepare_from_path(SONICOM_SOFA, delay_policy=BAKE_DELAY)
    kept = prepare_from_path(SONICOM_SOFA, delay_policy=PRESERVE_DELAY)

    assert baked.has_external_delay is False
    assert kept.has_external_delay is True
    np.testing.assert_allclose(kept.delays_samples[0], (3.0, 5.0))

    for ear, expected in enumerate((3, 5)):
        moved = int(np.argmax(np.abs(baked.hrirs[0, ear]))) - int(
            np.argmax(np.abs(kept.hrirs[0, ear]))
        )
        assert moved == expected


def test_a_nonzero_delay_is_never_silently_dropped():
    """Whichever policy is chosen, the delay ends up somewhere."""

    for policy in (BAKE_DELAY, PRESERVE_DELAY):
        prepared = prepare_from_path(SONICOM_SOFA, delay_policy=policy)
        if policy == PRESERVE_DELAY:
            assert prepared.has_external_delay
        else:
            assert prepared.taps > load_sofa(SONICOM_SOFA).taps


def test_resampling_happens_once_at_preparation():
    prepared = prepare_from_path(SONICOM_SOFA, sample_rate_hz=48_000)
    assert prepared.sample_rate_hz == 48_000
    assert prepared.taps > 128  # resampled upward from 44.1 kHz
    assert prepared.asset.source_sample_rate_hz == pytest.approx(44_100.0)


def test_unknown_delay_policy_is_rejected():
    with pytest.raises(ValueError):
        prepare_from_path(SONICOM_SOFA, delay_policy="ignore_it")


def test_cache_keys_separate_every_meaningful_difference():
    prepared = prepare_from_path(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    other_rate = prepare_from_path(SONICOM_SOFA, sample_rate_hz=48_000)
    other_policy = prepare_from_path(SONICOM_SOFA, delay_policy=PRESERVE_DELAY)
    keys = {prepared.cache_key(), other_rate.cache_key(), other_policy.cache_key()}
    assert len(keys) == 3


# --- cache ------------------------------------------------------------------


def test_cache_reuses_memory_then_disk(tmp_path):
    cache = HrtfCache(directory=tmp_path / "cache")
    first = cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    assert cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE) is first

    cache.clear_memory()
    from_disk = cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    assert from_disk is not first
    np.testing.assert_allclose(from_disk.hrirs, first.hrirs)
    assert from_disk.asset.sha256 == first.asset.sha256


def test_a_damaged_cache_entry_is_replaced_not_raised(tmp_path):
    cache = HrtfCache(directory=tmp_path / "cache")
    cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    cache.clear_memory()
    for entry in (tmp_path / "cache").glob("*.npz"):
        entry.write_bytes(b"corrupt")
    prepared = cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    assert prepared.measurements == 144


def test_memory_cache_is_bounded(tmp_path):
    cache = HrtfCache(memory_entries=1, directory=tmp_path / "cache")
    cache.get(SONICOM_SOFA, sample_rate_hz=SAMPLE_RATE)
    cache.get(SONICOM_SOFA, sample_rate_hz=48_000)
    assert cache.describe()["memory_entries"] == 1


# --- dataset vocabulary -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "subject", "processing", "rate"),
    [
        ("P0001_FreeFieldComp_48kHz.sofa", "P0001", "FreeFieldComp", 48_000),
        ("KB0065_Windowed_96kHz.sofa", "KB0065", "Windowed", 96_000),
        ("P0042_FreeFieldCompMinPhase_44kHz.sofa", "P0042", "FreeFieldCompMinPhase", 44_100),
        ("anything.sofa", "", "", None),
    ],
)
def test_asset_names_are_parsed_tolerantly(name, subject, processing, rate):
    entry = parse_asset_name(name)
    assert entry.subject == subject
    assert entry.processing == processing
    assert entry.sample_rate_hz == rate


def test_the_filename_wins_over_its_folder():
    entry = parse_asset_name("P0001/HRTF/KB0065_Windowed_44kHz.sofa")
    assert entry.subject == "KB0065"
    assert entry.is_kemar


def test_kemar_variants_are_known():
    assert is_kemar_subject("KB0065")
    assert not is_kemar_subject("P0001")
    assert "large ears" in KEMAR_VARIANTS["KB0065"]
    assert "small ears" in KEMAR_VARIANTS["KB0060"]
    assert "KEMAR" in describe_subject("KB0065")
    assert "human" in describe_subject("P0001")


def test_processing_variants_and_rates_are_declared():
    assert PROCESSING_VARIANTS == ("Windowed", "FreeFieldComp", "FreeFieldCompMinPhase")
    assert SONICOM_SAMPLE_RATES_HZ == (44_100, 48_000, 96_000)


def test_discovery_finds_assets_and_tolerates_a_missing_folder(tmp_path):
    import shutil

    (tmp_path / "P0001").mkdir()
    shutil.copy(SONICOM_SOFA, tmp_path / "P0001" / "P0001_Windowed_44kHz.sofa")
    shutil.copy(SONICOM_SOFA, tmp_path / "KB0065_FreeFieldComp_48kHz.sofa")

    found = discover_assets(tmp_path)
    assert [entry.subject for entry in found] == ["KB0065", "P0001"]
    assert discover_assets(tmp_path / "nowhere") == []


# --- storage ----------------------------------------------------------------


def test_storage_lives_outside_the_repository():
    report = storage_report()
    repository = Path(__file__).resolve().parents[2]
    assert repository not in Path(report.data_root).parents
    assert Path(report.datasets_root).is_relative_to(Path(report.data_root))


def test_storage_root_can_be_pointed_anywhere():
    with tempfile.TemporaryDirectory() as directory:
        storage.set_data_root(directory)
        try:
            assert str(storage.data_root()) == directory
            assert storage.cache_root().is_relative_to(directory)
        finally:
            storage.set_data_root(None)


def test_reading_missing_local_state_returns_the_default():
    assert storage.read_json("/nonexistent/state.json", default={"ratings": []}) == {"ratings": []}

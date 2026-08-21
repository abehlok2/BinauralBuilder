"""Phase 0 behaviour of the canonical project document beyond the contract test."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from src.audio.sam_workbench import conventions
from src.audio.sam_workbench.cli import main
from src.audio.sam_workbench.migrations import migrate_project_dict
from src.audio.sam_workbench.model import (
    Project,
    ProjectValidationError,
    SamModulationSpec,
    SignalSpec,
    Source,
    load_project,
    project_from_dict,
    save_project,
)
from src.audio.sam_workbench.version import SCHEMA_VERSION


# --- conventions -----------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (90.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0), (-90.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, 90.0, 1.0)),
        ((-1.0, 0.0, 0.0), (180.0, 0.0, 1.0)),
    ],
    ids=["front", "left", "right", "above", "rear"],
)
def test_cardinal_directions_round_trip(point, expected):
    assert conventions.cartesian_to_spherical_deg(point) == pytest.approx(expected)
    np.testing.assert_allclose(
        conventions.spherical_to_cartesian_m(*expected), point, atol=1e-15
    )


def test_origin_maps_to_zero_without_raising():
    assert conventions.cartesian_to_spherical_deg((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("angle_deg", "expected"), [(180.0, 180.0), (-180.0, 180.0), (270.0, -90.0), (540.0, 180.0), (-450.0, -90.0)]
)
def test_azimuth_wraps_into_the_canonical_range(angle_deg, expected):
    assert conventions.wrap_degrees(angle_deg) == pytest.approx(expected)
    assert conventions.wrap_radians(math.radians(angle_deg)) == pytest.approx(math.radians(expected))


def test_level_conversions_round_trip():
    assert conventions.db_to_linear(0.0) == pytest.approx(1.0)
    assert conventions.db_to_linear(conventions.SILENCE_DB) == 0.0
    assert conventions.linear_to_db(conventions.db_to_linear(-12.5)) == pytest.approx(-12.5)
    assert conventions.linear_to_db(0.0) == conventions.SILENCE_DB


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0.0, 0.0), (1.0, 44_100), (0.5, 22_050), (1.0 / 44_100, 1)]
)
def test_time_and_sample_conversions_are_consistent(seconds, expected):
    samples = conventions.seconds_to_samples(seconds, 44_100)
    assert samples == expected
    assert conventions.samples_to_seconds(samples, 44_100) == pytest.approx(seconds, abs=1e-9)


def test_seconds_to_samples_rounds_half_away_from_zero():
    assert conventions.seconds_to_samples(0.5 / 44_100, 44_100) == 1
    assert conventions.seconds_to_samples(1.5 / 44_100, 44_100) == 2


def test_channel_major_and_frame_major_convert_exactly_once():
    frame_major = np.arange(8, dtype=np.float32).reshape(4, 2)
    channel_major = conventions.to_channel_major(frame_major)
    assert channel_major.shape == (2, 4)
    np.testing.assert_array_equal(conventions.to_frame_major(channel_major), frame_major)
    with pytest.raises(ValueError):
        conventions.to_channel_major(channel_major)


# --- document ---------------------------------------------------------------


def test_default_project_is_safe_and_valid():
    project = Project()
    project.validate()
    assert project.schema_version == SCHEMA_VERSION
    assert project.output.limiter_enabled is True
    assert project.output.limiter_ceiling_dbfs == -1.0
    assert project.output.headphone_compensation_enabled is False
    assert Source().amplitude_linear <= 1.0
    assert project.audio.preview_block_size == 512
    assert project.audio.offline_block_size == 4_096


def test_populated_project_round_trips_through_json():
    project = Project(
        name="Populated",
        sources=(
            Source(
                id="carrier",
                name="Carrier",
                start_s=1.5,
                duration_s=30.0,
                amplitude_linear=0.4,
                signal=SignalSpec(carrier_frequency_hz=432.0, phase_rad=0.25),
                sam=SamModulationSpec(rate_hz=6.0, depth_rad=1.4, phase_rad=-0.5),
                tags=("lead",),
            ),
        ),
        metadata={"notes": "phase zero"},
    )
    restored = project_from_dict(json.loads(project.to_json()))
    assert restored == project


def test_unknown_fields_survive_a_round_trip():
    document = Project().to_dict()
    document["future_section"] = {"kind": "unknown"}
    document["audio"]["future_audio_field"] = 7
    document["sources"] = [dict(Source(id="s0").to_dict(), future_source_field="keep me")]

    project = project_from_dict(document)
    assert project.extras == {"future_section": {"kind": "unknown"}}
    assert project.audio.extras == {"future_audio_field": 7}
    assert project.sources[0].extras == {"future_source_field": "keep me"}
    assert project.to_dict() == document


def test_unversioned_documents_are_migrated_explicitly():
    document = Project().to_dict()
    document.pop("schema_version")
    assert migrate_project_dict(document)["schema_version"] == SCHEMA_VERSION
    assert project_from_dict(document).schema_version == SCHEMA_VERSION


def test_unknown_schema_version_is_refused():
    document = Project().to_dict()
    document["schema_version"] = "99.0"
    with pytest.raises(ProjectValidationError) as caught:
        project_from_dict(document)
    assert [issue.path for issue in caught.value.issues] == ["schema_version"]


def test_validation_reports_settings_and_source_issues_together():
    project = Project(
        name="",
        audio=replace(Project().audio, sample_rate_hz=10, preview_block_size=3),
        sources=(Source(id="", signal=SignalSpec(carrier_frequency_hz=-1.0)),),
    )
    with pytest.raises(ProjectValidationError) as caught:
        project.validate()
    paths = {issue.path for issue in caught.value.issues}
    assert {
        "name",
        "audio.sample_rate_hz",
        "audio.preview_block_size",
        "sources[0].id",
        "sources[0].signal.carrier_frequency_hz",
    } <= paths


def test_carrier_above_nyquist_is_reported():
    project = Project(sources=(Source(signal=SignalSpec(carrier_frequency_hz=30_000.0)),))
    with pytest.raises(ProjectValidationError) as caught:
        project.validate()
    assert any(issue.path == "sources[0].signal.carrier_frequency_hz" for issue in caught.value.issues)


def test_malformed_documents_report_paths_rather_than_crashing():
    document = Project().to_dict()
    document["audio"] = {"sample_rate_hz": "fast"}
    document["sources"] = [{"id": "a", "amplitude_linear": "loud"}]
    with pytest.raises(ProjectValidationError) as caught:
        project_from_dict(document)
    paths = {issue.path for issue in caught.value.issues}
    assert "audio.sample_rate_hz" in paths
    assert "sources[0].amplitude_linear" in paths


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "atomic.sam.json"
    save_project(Project(name="First"), path)
    save_project(Project(name="Second"), path)
    assert load_project(path).name == "Second"
    assert [entry.name for entry in tmp_path.iterdir()] == ["atomic.sam.json"]


def test_save_refuses_an_invalid_project_without_writing(tmp_path):
    path = tmp_path / "invalid.sam.json"
    with pytest.raises(ProjectValidationError):
        save_project(Project(sources=(Source(amplitude_linear=-1.0),)), path)
    assert list(tmp_path.iterdir()) == []


def test_load_reports_broken_json_with_a_path(tmp_path):
    path = tmp_path / "broken.sam.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectValidationError):
        load_project(path)


# --- command-line shell -----------------------------------------------------


def test_cli_refuses_to_overwrite_without_force(tmp_path, capsys):
    path = tmp_path / "demo.sam.json"
    assert main(["new", str(path), "--name", "Original"]) == 0
    assert main(["new", str(path), "--name", "Replacement"]) == 1
    assert load_project(path).name == "Original"
    assert main(["new", str(path), "--name", "Replacement", "--force"]) == 0
    assert load_project(path).name == "Replacement"
    capsys.readouterr()


def test_cli_new_honours_the_sample_rate_and_rejects_invalid_ones(tmp_path, capsys):
    path = tmp_path / "rate.sam.json"
    assert main(["new", str(path), "--sample-rate", "48000"]) == 0
    assert load_project(path).audio.sample_rate_hz == 48_000
    assert main(["new", str(tmp_path / "bad.sam.json"), "--sample-rate", "3"]) == 1
    assert not (tmp_path / "bad.sam.json").exists()
    assert "audio.sample_rate_hz" in capsys.readouterr().err


def test_cli_validate_reports_every_issue(tmp_path, capsys):
    path = tmp_path / "invalid.sam.json"
    document = Project().to_dict()
    document["sources"] = [
        Source(id="dup", amplitude_linear=-1.0).to_dict(),
        Source(id="dup", duration_s=0.0).to_dict(),
    ]
    path.write_text(json.dumps(document), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    errors = capsys.readouterr().err
    assert "sources[0].amplitude_linear" in errors
    assert "sources[1].duration_s" in errors
    assert "sources[1].id" in errors


def test_cli_validate_reports_a_missing_file(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "absent.sam.json")]) == 1
    assert "Could not read" in capsys.readouterr().err

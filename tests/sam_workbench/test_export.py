"""WAV export, the reconstruction manifest, and the render CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from src.audio.sam_workbench.cli import main
from src.audio.sam_workbench.conventions import db_to_linear
from src.audio.sam_workbench.export import export_wav, project_sha256
from src.audio.sam_workbench.model import (
    Project,
    SamModulationSpec,
    SignalSpec,
    Source,
    save_project,
)
from src.audio.sam_workbench.render import render_project

SAMPLE_RATE = 44_100


def _project(**overrides) -> Project:
    source = Source(
        id="carrier",
        amplitude_linear=0.4,
        signal=SignalSpec(carrier_frequency_hz=220.0),
        sam=SamModulationSpec(rate_hz=4.0, depth_rad=1.0),
    )
    return Project(name="Export test", sources=(source,), **overrides)


def test_export_writes_audio_and_manifest(tmp_path):
    wav_path = tmp_path / "render.wav"
    manifest_path = export_wav(_project(), 0.25, wav_path)

    audio, sample_rate = sf.read(wav_path, always_2d=True)
    assert sample_rate == SAMPLE_RATE
    assert audio.shape == (11_025, 2)
    assert manifest_path == tmp_path / "render.manifest.json"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["renderer"] == "exact_symmetric_sam"
    assert manifest["frames"] == 11_025
    assert manifest["duration_s"] == pytest.approx(0.25)
    assert manifest["channels"] == 2
    assert manifest["channel_order"] == ["left", "right"]
    assert manifest["bit_depth"] == 32
    assert manifest["audio_file"] == "render.wav"


def test_manifest_carries_everything_needed_to_reconstruct(tmp_path):
    project = _project()
    manifest = json.loads(export_wav(project, 0.05, tmp_path / "r.wav").read_text())

    assert manifest["schema_version"] == project.schema_version
    assert manifest["application_version"]
    assert manifest["project_id"] == project.id
    assert manifest["project_sha256"] == project_sha256(project)
    assert len(manifest["project_sha256"]) == 64
    assert manifest["seeds"]["project_random_seed"] == project.audio.random_seed
    assert manifest["renderer_options"]["block_size_invariant"] is True
    assert manifest["master_processing"]["master_gain_db"] == pytest.approx(-6.0)
    assert manifest["master_processing"]["shared_gain_both_ears"] is True
    assert manifest["master_processing"]["per_channel_normalization"] is False
    assert manifest["master_processing"]["headphone_compensation_enabled"] is False
    assert manifest["sources"][0]["id"] == "carrier"
    assert "rendered_utc" in manifest["nondeterministic"]
    assert "elapsed_s" in manifest["nondeterministic"]


def test_manifest_hash_tracks_the_project_document(tmp_path):
    first = json.loads(export_wav(_project(), 0.02, tmp_path / "a.wav").read_text())
    renamed = json.loads(
        export_wav(Project(name="Other", sources=_project().sources), 0.02, tmp_path / "b.wav").read_text()
    )
    assert first["project_sha256"] != renamed["project_sha256"]


def test_export_is_deterministic_apart_from_timing_fields(tmp_path):
    project = _project()
    first_manifest = json.loads(export_wav(project, 0.05, tmp_path / "first.wav").read_text())
    second_manifest = json.loads(
        export_wav(project, 0.05, tmp_path / "second.wav", block_size=97).read_text()
    )

    first_audio, _ = sf.read(tmp_path / "first.wav", always_2d=True)
    second_audio, _ = sf.read(tmp_path / "second.wav", always_2d=True)
    np.testing.assert_array_equal(first_audio, second_audio)

    for manifest in (first_manifest, second_manifest):
        manifest.pop("nondeterministic")
        manifest.pop("audio_file")
        manifest["renderer_options"].pop("block_size")
    assert first_manifest == second_manifest


def test_master_gain_and_limiter_are_shared_between_ears(tmp_path):
    quiet = Project(sources=(Source(id="s", amplitude_linear=0.5),))
    audio, report = render_project(quiet, 0.05)
    assert report.master_gain_db == pytest.approx(-6.0)
    assert report.peak == pytest.approx(0.5 * db_to_linear(-6.0), rel=1e-3)

    loud = Project(
        sources=(Source(id="a", amplitude_linear=1.5), Source(id="b", amplitude_linear=1.5)),
    )
    limited, loud_report = render_project(loud, 0.05)
    assert loud_report.limiter is not None
    assert loud_report.limiter.max_gain_reduction_db > 0.0
    assert loud_report.peak <= db_to_linear(loud.output.limiter_ceiling_dbfs) + 1e-9
    assert np.all(np.isfinite(limited))


def test_render_warns_when_modulation_approaches_nyquist():
    project = Project(
        sources=(
            Source(
                id="deep",
                signal=SignalSpec(carrier_frequency_hz=18_000.0),
                sam=SamModulationSpec(rate_hz=2_000.0, depth_rad=6.0),
            ),
        )
    )
    _, report = render_project(project, 0.01)
    assert any("Nyquist" in issue.message for issue in report.warnings)
    assert all(issue.severity == "warning" for issue in report.warnings)


def test_sources_are_placed_at_their_start_time():
    project = Project(
        sources=(Source(id="late", amplitude_linear=0.5, start_s=0.05, duration_s=0.05),)
    )
    audio, report = render_project(project, 0.2, apply_limiter=False)
    assert report.sources[0].start_sample == 2205
    assert np.all(audio[:, :2205] == 0.0)
    assert np.any(audio[:, 2205:4410] != 0.0)
    assert np.all(audio[:, 4410:] == 0.0)


def test_disabled_sources_are_not_rendered():
    project = Project(sources=(Source(id="off", enabled=False, amplitude_linear=0.5),))
    audio, report = render_project(project, 0.02)
    assert report.sources == ()
    assert np.all(audio == 0.0)


# --- command line -----------------------------------------------------------


def test_cli_render_produces_audio_and_manifest(tmp_path, capsys):
    project_path = tmp_path / "cli.sam.json"
    save_project(_project(), project_path)
    output = tmp_path / "cli.wav"

    assert main(["render", str(project_path), str(output), "--duration", "0.1"]) == 0
    assert output.exists()
    manifest = json.loads((tmp_path / "cli.manifest.json").read_text())
    assert manifest["frames"] == 4410
    printed = capsys.readouterr().out
    assert "Rendered" in printed
    assert "Wrote manifest" in printed


def test_cli_render_can_write_a_project_snapshot(tmp_path, capsys):
    project_path = tmp_path / "snap.sam.json"
    save_project(_project(), project_path)
    assert (
        main(
            [
                "render",
                str(project_path),
                str(tmp_path / "snap.wav"),
                "--duration",
                "0.05",
                "--snapshot",
            ]
        )
        == 0
    )
    snapshot = tmp_path / "snap.project.json"
    assert snapshot.exists()
    assert json.loads(snapshot.read_text())["name"] == "Export test"
    capsys.readouterr()


def test_cli_render_reports_a_silent_project(tmp_path, capsys):
    project_path = tmp_path / "empty.sam.json"
    save_project(Project(), project_path)
    assert main(["render", str(project_path), str(tmp_path / "empty.wav"), "--duration", "0.05"]) == 0
    assert "no enabled sources" in capsys.readouterr().err


def test_cli_render_rejects_a_non_positive_duration(tmp_path, capsys):
    project_path = tmp_path / "bad.sam.json"
    save_project(_project(), project_path)
    assert main(["render", str(project_path), str(tmp_path / "bad.wav"), "--duration", "0"]) == 1
    assert "must be positive" in capsys.readouterr().err


def test_cli_new_with_source_renders_audibly(tmp_path, capsys):
    project_path = tmp_path / "audible.sam.json"
    assert main(["new", str(project_path), "--with-source"]) == 0
    assert main(["render", str(project_path), str(tmp_path / "audible.wav"), "--duration", "0.05"]) == 0
    audio, _ = sf.read(tmp_path / "audible.wav", always_2d=True)
    assert np.max(np.abs(audio)) > 0.05
    capsys.readouterr()

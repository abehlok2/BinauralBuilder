"""An exported file that can say what produced it.

The typed-project exporter has written manifests since Phase 1. Normal track
export wrote a WAV and nothing else, so a finished file carried no record of
which dataset, which interpolation, which path or which seeds produced it -
which is the difference between an artifact and a result.

"Sufficient to reconstruct the acoustic condition" is not provable by counting
fields, so the central test here rebuilds a track from the manifest alone and
renders it. If the manifest were missing something that matters, the two
renders would differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audio.render_job import RenderSnapshot, run_render
from src.audio.track_manifest import (
    build_track_manifest,
    manifest_path_for,
    reconstruct_track,
    write_track_manifest,
)

RATE = 8000
SOFA = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")


def _track(renderer="abstract_pm", **params):
    voice_params = {
        "amp_left": 0.2, "amp_right": 0.2,
        "baseFreq": 200.0, "beatFreq": 4.0,
        "rendererMode": renderer,
    }
    voice_params.update(params)
    return {
        "global_settings": {
            "sample_rate": RATE,
            "crossfade_duration": 0.0,
            "crossfade_curve": "linear",
            "random_seed": 4242,
        },
        "steps": [
            {
                "duration": 0.4,
                "description": "induction",
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "description": "Lead",
                        "sam_source_id": "sam.1",
                        "params": voice_params,
                    }
                ],
            }
        ],
    }


# --- the claim ---------------------------------------------------------------


def test_a_track_rebuilt_from_its_manifest_renders_the_same_audio(tmp_path):
    """The whole point, checked rather than asserted."""

    from src.synth_functions.sound_creator import assemble_track_from_data

    track = _track()
    manifest = build_track_manifest(track, audio_path=tmp_path / "out.wav")
    rebuilt = reconstruct_track(manifest)

    original = assemble_track_from_data(track, RATE, 0.0)
    restored = assemble_track_from_data(rebuilt, RATE, 0.0)

    assert original.shape == restored.shape
    assert np.array_equal(original, restored)


def test_reconstruction_keeps_parameters_a_newer_build_added(tmp_path):
    """Filtering to a known list drops exactly the keys a reader would need."""

    track = _track(someFutureKey={"nested": [1, 2, 3]})
    rebuilt = reconstruct_track(
        build_track_manifest(track, audio_path=tmp_path / "out.wav")
    )
    params = rebuilt["steps"][0]["voices"][0]["params"]
    assert params["someFutureKey"] == {"nested": [1, 2, 3]}


def test_identifiers_and_names_survive_the_round_trip(tmp_path):
    track = _track()
    rebuilt = reconstruct_track(
        build_track_manifest(track, audio_path=tmp_path / "out.wav")
    )
    voice = rebuilt["steps"][0]["voices"][0]
    assert voice["sam_source_id"] == "sam.1"
    assert voice["description"] == "Lead"


# --- what it records ---------------------------------------------------------


def test_the_manifest_records_the_schema_versions(tmp_path):
    manifest = build_track_manifest(_track(), audio_path=tmp_path / "out.wav")
    assert manifest["trackSchemaVersion"] >= 1
    assert manifest["sceneSchemaVersion"] >= 1
    assert manifest["manifestVersion"]


def test_the_manifest_names_the_renderer_and_what_it_claims(tmp_path):
    manifest = build_track_manifest(_track("hrtf"), audio_path=tmp_path / "out.wav")
    modes = manifest["renderer"]["modes"]
    assert "hrtf" in modes
    assert modes["hrtf"]["physicalElevation"] is True
    # The honesty note travels with the render it describes.
    assert modes["hrtf"]["honestyNote"]
    assert manifest["renderer"]["engineVersion"]


def test_an_unknown_renderer_is_recorded_as_unknown(tmp_path):
    """A manifest from a newer build must not be silently reinterpreted."""

    manifest = build_track_manifest(
        _track("some_future_renderer"), audio_path=tmp_path / "out.wav"
    )
    assert manifest["renderer"]["modes"]["some_future_renderer"] == {
        "unknownToThisBuild": True
    }


def test_a_sofa_asset_is_referenced_by_hash_not_just_by_name(tmp_path):
    manifest = build_track_manifest(
        _track("hrtf", hrtfAsset=SOFA), audio_path=tmp_path / "out.wav"
    )
    sofa = manifest["sources"][0]["sofa"]
    assert sofa["path"] == SOFA
    assert len(sofa["sha256"]) == 64
    assert sofa["measurements"] > 0


def test_an_unreadable_asset_says_so_rather_than_vanishing(tmp_path):
    """"I could not hash this" and "there was nothing" are different facts."""

    manifest = build_track_manifest(
        _track("hrtf", hrtfAsset=str(tmp_path / "missing.sofa")),
        audio_path=tmp_path / "out.wav",
    )
    sofa = manifest["sources"][0]["sofa"]
    assert "unreadable" in sofa
    assert "sha256" not in sofa


def test_the_hrtf_configuration_travels_with_the_source(tmp_path):
    options = {
        "interpolation": "delay_magnitude",
        "delayPolicy": "preserve_external_delay",
        "distanceLaw": "inverse_square",
        "cue": {"neutral": False, "itdScale": 1.4},
        "anchor": {"enabled": True},
    }
    manifest = build_track_manifest(
        _track("hybrid", hrtfAsset=SOFA, hrtfOptions=options,
               headphoneAsset="/phones.wav"),
        audio_path=tmp_path / "out.wav",
    )
    params = manifest["sources"][0]["parameters"]
    assert params["hrtfOptions"] == options
    assert params["headphoneAsset"] == "/phones.wav"


def test_the_trajectory_and_listener_are_recorded(tmp_path):
    trajectory = {"geometry": {"type": "torus"}, "traversal": {"durationS": 4.0}}
    listener = {"positionM": [0.0, 0.0, 0.0], "yawPitchRollDegrees": [10.0, 0.0, 0.0]}
    manifest = build_track_manifest(
        _track("hrtf", hrtfAsset=SOFA, canonicalTrajectory=trajectory, listener=listener),
        audio_path=tmp_path / "out.wav",
    )
    record = manifest["sources"][0]
    assert record["trajectory"] == trajectory
    assert record["listener"] == listener


def test_the_scene_sections_are_recorded(tmp_path):
    track = _track()
    track["sam_scene"] = {
        "sources": [{"id": "sam.1", "name": "Lead"}],
        "modulators": [{"id": "lfo.slow", "waveform": "sine", "rateHz": 0.05}],
        "routing": {"buses": [{"id": "master"}], "sources": [{"sourceId": "sam.1"}]},
    }
    manifest = build_track_manifest(track, audio_path=tmp_path / "out.wav")
    assert manifest["scene"]["modulators"][0]["waveform"] == "sine"
    assert manifest["scene"]["routing"]["sources"][0]["sourceId"] == "sam.1"


def test_seeds_gain_and_output_policy_are_recorded(tmp_path):
    manifest = build_track_manifest(
        _track(), audio_path=tmp_path / "out.wav", target_level=0.5
    )
    assert manifest["seeds"]["trackRandomSeed"] == 4242
    assert manifest["output"]["targetLevel"] == pytest.approx(0.5)
    assert manifest["output"]["perChannelNormalization"] is False
    assert manifest["output"]["qualityProfile"]


def test_coverage_warnings_reach_the_manifest(tmp_path):
    """A path the dataset cannot support is part of the acoustic condition."""

    trajectory = {
        "geometry": {"type": "dome_traversal", "parameters": {"turns": 40}},
        "traversal": {"durationS": 1.0},
    }
    manifest = build_track_manifest(
        _track("hrtf", hrtfAsset=SOFA, canonicalTrajectory=trajectory),
        audio_path=tmp_path / "out.wav",
    )
    assert manifest["sources"][0]["coverageWarnings"]


def test_the_manifest_is_json_and_stable(tmp_path):
    manifest = build_track_manifest(_track(), audio_path=tmp_path / "out.wav")
    written = write_track_manifest(manifest, tmp_path / "out.wav")
    assert written == manifest_path_for(tmp_path / "out.wav")
    reloaded = json.loads(written.read_text())
    assert reloaded["trackSha256"] == manifest["trackSha256"]


# --- the export path writes one ---------------------------------------------


def test_an_export_writes_its_manifest_beside_the_audio(tmp_path):
    destination = tmp_path / "out.wav"
    outcome = run_render(
        RenderSnapshot.of(_track(), str(destination), write_manifest=True)
    )

    assert outcome.succeeded
    assert outcome.manifest_path
    written = Path(outcome.manifest_path)
    assert written.exists()

    manifest = json.loads(written.read_text())
    assert manifest["audioFile"] == "out.wav"
    # The measured cost of this very render, not an estimate.
    assert manifest["diagnostics"]["wallS"] > 0.0


def test_no_manifest_is_written_unless_asked(tmp_path):
    destination = tmp_path / "out.wav"
    outcome = run_render(RenderSnapshot.of(_track(), str(destination)))
    assert outcome.succeeded
    assert outcome.manifest_path == ""
    assert not manifest_path_for(destination).exists()


def test_a_cancelled_export_writes_no_manifest(tmp_path):
    """A manifest must never describe a file that does not exist."""

    destination = tmp_path / "out.wav"
    outcome = run_render(
        RenderSnapshot.of(_track(), str(destination), write_manifest=True),
        should_cancel=lambda: True,
    )
    assert outcome.cancelled
    assert not manifest_path_for(destination).exists()

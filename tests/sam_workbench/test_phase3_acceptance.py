"""Phase 3's completion criteria, as tests rather than as a claim.

Each test here corresponds to a line in the specification's completion list.
They deliberately go through the production entry points rather than through
internals, because the criteria are about what a user can reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audio.render_job import RenderSnapshot, run_render
from src.audio.track_manifest import manifest_path_for, reconstruct_track
from src.audio.sam_workbench.render.registry import REGISTRY

RATE = 8000
SOFA = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")
TRAJECTORY = {
    "geometry": {"type": "dome_traversal", "parameters": {"turns": 1}},
    "traversal": {"durationS": 0.4},
}


def _voice_params(renderer):
    params = {
        "amp": 0.4,
        "carrierFreq": 200.0,
        "rendererMode": renderer,
        "canonicalTrajectory": TRAJECTORY,
    }
    if renderer in ("hrtf", "hybrid"):
        params["hrtfAsset"] = SOFA
        params["hrtfOptions"] = {"interpolation": "nearest"}
    return params


def _track(renderer):
    return {
        "global_settings": {"sample_rate": RATE, "crossfade_duration": 0.0},
        "steps": [
            {
                "duration": 0.4,
                "description": renderer,
                "voices": [
                    {
                        "synth_function_name": "spatial_angle_modulation_sam2",
                        "description": f"{renderer} voice",
                        "params": _voice_params(renderer),
                    }
                ],
            }
        ],
    }


ALL_RENDERERS = [entry.identifier for entry in REGISTRY.voice_renderable]


# --- all four renderers are reachable ---------------------------------------


def test_the_build_offers_the_four_named_renderers():
    assert {"abstract_pm", "geometric", "hrtf", "hybrid"} <= set(ALL_RENDERERS)


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_every_renderer_previews_through_the_workbench(renderer):
    from src.audio.sam_workbench.preview import render_voice_preview

    result = render_voice_preview(
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "params": _voice_params(renderer),
        },
        sample_rate_hz=RATE,
        duration_s=0.2,
    )
    assert result.audio.shape[0] > 0
    assert np.all(np.isfinite(result.audio))


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_every_renderer_exports_through_the_background_job(renderer, tmp_path):
    destination = tmp_path / f"{renderer}.wav"
    outcome = run_render(
        RenderSnapshot.of(_track(renderer), str(destination), write_manifest=True)
    )
    assert outcome.succeeded, outcome.error
    assert destination.exists()
    assert Path(outcome.manifest_path).exists()


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_preview_and_export_agree_for_every_renderer(renderer, tmp_path):
    """Auditioning something other than what is exported is the old failure."""

    import soundfile as sf

    from src.audio.sam_workbench.preview import render_voice_preview

    destination = tmp_path / f"{renderer}.wav"
    run_render(RenderSnapshot.of(_track(renderer), str(destination)))
    exported, _rate = sf.read(str(destination), dtype="float32", always_2d=True)

    preview = render_voice_preview(
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "params": _voice_params(renderer),
        },
        sample_rate_hz=RATE,
        duration_s=0.4,
    ).audio

    span = min(len(exported), len(preview))
    assert span > 0
    # Export normalizes to a target level, so compare shape rather than scale.
    for channel in range(2):
        a = exported[:span, channel].astype(np.float64)
        b = preview[:span, channel].astype(np.float64)
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        correlation = float(np.corrcoef(a, b)[0, 1])
        assert correlation > 0.99, f"{renderer} channel {channel}: {correlation:.4f}"


# --- formats ----------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_both_export_formats_write_readable_audio(suffix, tmp_path):
    import soundfile as sf

    destination = tmp_path / f"track{suffix}"
    outcome = run_render(RenderSnapshot.of(_track("abstract_pm"), str(destination)))

    assert outcome.succeeded, outcome.error
    audio, rate = sf.read(str(destination), always_2d=True)
    assert rate == RATE
    assert audio.shape[1] == 2
    assert np.all(np.isfinite(audio))


# --- the manifest reconstructs ----------------------------------------------


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_a_manifest_reconstructs_its_own_render(renderer, tmp_path):
    from src.synth_functions.sound_creator import assemble_track_from_data

    destination = tmp_path / f"{renderer}.wav"
    outcome = run_render(
        RenderSnapshot.of(_track(renderer), str(destination), write_manifest=True)
    )
    assert outcome.succeeded

    manifest = json.loads(Path(outcome.manifest_path).read_text())
    rebuilt = reconstruct_track(manifest)

    original = assemble_track_from_data(_track(renderer), RATE, 0.0)
    restored = assemble_track_from_data(rebuilt, RATE, 0.0)
    assert np.array_equal(original, restored)


# --- long renders stay bounded ----------------------------------------------


def test_chunking_bounds_what_a_long_render_costs_on_top_of_its_output():
    """What chunking actually buys, stated as what it is.

    It does not stop peak memory growing with length: assemble_track_from_data
    holds a step buffer and a track buffer, and both scale with duration, so
    growth is close to twice the output buffer either way. What chunking bounds
    is the synth's working set on top of those - one chunk wide rather than one
    step wide - which is where the multi-gigabyte peaks came from.
    """

    import tracemalloc

    import src.synth_functions.sound_creator as sound_creator

    def peak_for(chunked):
        track = {
            "global_settings": {"sample_rate": 44100},
            "steps": [
                {
                    "duration": 300.0,
                    "voices": [
                        {
                            "synth_function_name": "binaural_beat",
                            "params": {"amp_left": 0.3, "amp_right": 0.3,
                                       "baseFreq": 200.0, "beatFreq": 4.0},
                        }
                    ],
                }
            ],
        }
        saved = sound_creator.ENABLE_SEQUENTIAL_CHUNKING
        sound_creator.ENABLE_SEQUENTIAL_CHUNKING = chunked
        try:
            tracemalloc.start()
            sound_creator.assemble_track_from_data(track, 44100, 0.0)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak
        finally:
            sound_creator.ENABLE_SEQUENTIAL_CHUNKING = saved

    chunked, whole = peak_for(True), peak_for(False)
    output_buffer = 300.0 * 44100 * 2 * 4

    assert chunked < whole
    # The saving scales with the step, so it is the working set that went away
    # rather than a fixed overhead.
    assert whole - chunked > output_buffer
    # And what remains is a small multiple of the output buffer rather than a
    # multiple of the step's length. Measured at 3.3x on this fixture; the
    # bound is a regression guard, not a derived limit.
    assert chunked < 4.0 * output_buffer


# --- cancellation and cleanup -----------------------------------------------


def test_cancelling_leaves_no_file_and_says_so(tmp_path):
    destination = tmp_path / "cancelled.wav"
    outcome = run_render(
        RenderSnapshot.of(_track("abstract_pm"), str(destination), write_manifest=True),
        should_cancel=lambda: True,
    )
    assert outcome.cancelled
    assert not destination.exists()
    assert not manifest_path_for(destination).exists()


def test_an_interrupted_export_can_simply_be_run_again(tmp_path):
    """Recovery after an interruption is that nothing was left to clean up."""

    destination = tmp_path / "again.wav"
    cancelled = run_render(
        RenderSnapshot.of(_track("abstract_pm"), str(destination)),
        should_cancel=lambda: True,
    )
    assert cancelled.cancelled and not destination.exists()

    second = run_render(RenderSnapshot.of(_track("abstract_pm"), str(destination)))
    assert second.succeeded and destination.exists()


# --- closing during a job ---------------------------------------------------


def test_a_manager_with_a_running_job_reports_itself_busy(qtbot, tmp_path):
    from src.ui.render_job import RenderJobManager

    manager = RenderJobManager()
    outcomes = []
    manager.finished.connect(lambda _w, outcome: outcomes.append(outcome))

    manager.start(
        RenderSnapshot.of(_track("abstract_pm"), str(tmp_path / "busy.wav"))
    )
    assert manager.busy is True
    assert manager.active_count == 1

    manager.cancel_all()
    qtbot.waitUntil(lambda: bool(outcomes), timeout=30_000)
    assert manager.wait_for_idle(5_000) is True
    assert manager.busy is False


# --- compatibility ----------------------------------------------------------


def test_an_older_voice_without_new_keys_still_renders(tmp_path):
    """Existing tracks and presets remain compatible."""

    legacy = {
        "global_settings": {"sample_rate": RATE},
        "steps": [
            {
                "duration": 0.3,
                "voices": [
                    {
                        "synth_function_name": "spatial_angle_modulation_sam2",
                        "params": {"amp": 0.4, "carrierFreq": 200.0},
                    }
                ],
            }
        ],
    }
    outcome = run_render(RenderSnapshot.of(legacy, str(tmp_path / "legacy.wav")))
    assert outcome.succeeded, outcome.error


def test_unknown_serialized_fields_survive_an_export_round_trip(tmp_path):
    from src.audio.track_manifest import build_track_manifest

    track = _track("abstract_pm")
    track["steps"][0]["voices"][0]["params"]["someFutureKey"] = {"a": [1, 2]}
    track["steps"][0]["voices"][0]["unknownVoiceKey"] = "kept"

    manifest = build_track_manifest(track, audio_path=tmp_path / "x.wav")
    rebuilt = reconstruct_track(manifest)
    assert rebuilt["steps"][0]["voices"][0]["params"]["someFutureKey"] == {"a": [1, 2]}

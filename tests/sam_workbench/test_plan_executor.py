"""The compiled plan, executed - and proved equal to what it replaces.

The plan has been authoritative for validation, timing, assets and controls
since Phase 1, and nothing rendered from it: production reached the renderers
through `compat.render_sam2_voice`, one voice at a time. So the structure a
project was validated against and the structure it was rendered from were two
readings of one document, kept in agreement only by testing.

The executor closes that. It contains no renderer of its own - every source
goes through the same `render_voice_channels` dispatch the per-voice path
calls - so the two are equal by construction rather than by luck. These tests
hold that construction in place, and check the things a second reading of the
document could still get wrong: windows, gains, ordering, and failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice, render_voice_channels
from src.audio.sam_workbench.plan import plan_from_track
from src.audio.sam_workbench.render.executor import execute_plan, render_source_window

RATE = 8000
SOFA = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")
TRAJECTORY = {
    "geometry": {"type": "dome_traversal", "parameters": {"turns": 1}},
    "traversal": {"durationS": 0.5},
}
ALL_RENDERERS = ("abstract_pm", "geometric", "hrtf", "hybrid")


def _params(renderer):
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


def _track(renderer, duration=0.5, start=0.0, sources=1):
    voices = [
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "description": f"{renderer} {index}",
            "sam_source_id": f"sam.{index}",
            "params": _params(renderer),
        }
        for index in range(1, sources + 1)
    ]
    return {
        "global_settings": {"sample_rate": RATE},
        "steps": [{"duration": duration, "description": "step", "voices": voices}],
    }


# --- one dispatch, two callers ----------------------------------------------


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_the_executor_renders_what_the_voice_path_renders(renderer):
    """Equal by construction; this is the construction being checked."""

    frames = int(0.5 * RATE)
    plan = plan_from_track(_track(renderer), start_sample=0, frames=frames)
    executed = execute_plan(plan, block_size=512)

    direct = render_voice_channels(
        _params(renderer), frames, RATE, duration=0.5, block_size=512
    )

    assert executed.audio.shape == (2, frames)
    assert np.allclose(executed.audio, direct, atol=1e-12)


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_the_executor_matches_the_public_compatibility_entry_point(renderer):
    """Frame-major out of one, channel-major out of the other, same audio."""

    frames = int(0.5 * RATE)
    plan = plan_from_track(_track(renderer), start_sample=0, frames=frames)
    executed = execute_plan(plan, block_size=512)

    legacy = render_sam2_voice(0.5, RATE, params=_params(renderer), block_size=512)

    assert np.allclose(executed.audio.T, legacy, atol=1e-6)


# --- windows ----------------------------------------------------------------


@pytest.mark.parametrize("renderer", ALL_RENDERERS)
def test_an_arbitrary_window_is_the_same_audio_as_the_whole(renderer):
    """A source continues from its own elapsed time, not the window's start."""

    frames = int(0.5 * RATE)
    plan = plan_from_track(_track(renderer), start_sample=0, frames=frames)

    whole = execute_plan(plan, block_size=512).audio
    offset = int(0.2 * RATE)
    span = int(0.2 * RATE)
    windowed = execute_plan(
        plan, start_sample=offset, frames=span, block_size=512
    ).audio

    assert windowed.shape == (2, span)
    assert np.allclose(windowed, whole[:, offset : offset + span], atol=2e-6)


def test_a_source_that_starts_later_is_silent_until_it_does():
    track = _track("abstract_pm", duration=0.3)
    track["steps"].insert(0, {"duration": 0.2, "description": "lead-in", "voices": []})

    frames = int(0.5 * RATE)
    plan = plan_from_track(track, start_sample=0, frames=frames)
    executed = execute_plan(plan)

    silent_until = int(0.2 * RATE)
    assert np.allclose(executed.audio[:, :silent_until], 0.0)
    assert np.max(np.abs(executed.audio[:, silent_until:])) > 0.0


def test_a_window_entirely_after_a_source_is_silent():
    frames = int(0.5 * RATE)
    plan = plan_from_track(_track("abstract_pm", duration=0.5), frames=frames)
    executed = execute_plan(plan, start_sample=frames * 2, frames=frames)
    assert np.allclose(executed.audio, 0.0)


# --- several sources --------------------------------------------------------


def test_two_sources_sum_and_are_both_reported():
    frames = int(0.4 * RATE)
    plan = plan_from_track(_track("abstract_pm", duration=0.4, sources=2), frames=frames)
    executed = execute_plan(plan)

    assert len(executed.stems) == 2
    assert {entry.source_id for entry in executed.sources} == {"sam.1", "sam.2"}

    summed = sum(executed.stems.values())
    assert np.allclose(executed.audio, summed, atol=1e-9)


def test_the_renderers_used_are_reported_from_the_plan():
    frames = int(0.3 * RATE)
    plan = plan_from_track(_track("hrtf", duration=0.3), frames=frames)
    executed = execute_plan(plan)
    assert executed.renderer_ids == ("hrtf",)


# --- routing ----------------------------------------------------------------


def test_a_muted_source_is_silent_in_the_mix_but_kept_as_a_stem():
    """An inspector still needs to see what the muted source would sound like."""

    track = _track("abstract_pm", duration=0.3)
    track["sam_scene"] = {
        "sources": [{"id": "sam.1", "name": "Lead"}],
        "routing": {
            "buses": [{"id": "master", "name": "Master"}],
            "sources": [{"sourceId": "sam.1", "busId": "master", "muted": True}],
        },
    }
    frames = int(0.3 * RATE)
    plan = plan_from_track(track, frames=frames)

    executed = execute_plan(plan)
    assert np.allclose(executed.audio, 0.0)
    assert np.max(np.abs(executed.stems["sam.1"])) > 0.0


def test_routing_can_be_bypassed_for_an_inspector():
    track = _track("abstract_pm", duration=0.3)
    track["sam_scene"] = {
        "sources": [{"id": "sam.1"}],
        "routing": {
            "buses": [{"id": "master"}],
            "sources": [{"sourceId": "sam.1", "busId": "master", "muted": True}],
        },
    }
    frames = int(0.3 * RATE)
    plan = plan_from_track(track, frames=frames)

    assert np.max(np.abs(execute_plan(plan, apply_routing=False).audio)) > 0.0


def test_a_bus_gain_reaches_the_mix():
    track = _track("abstract_pm", duration=0.3)
    track["sam_scene"] = {
        "sources": [{"id": "sam.1"}],
        "routing": {
            "buses": [{"id": "master", "gainDb": -6.0}],
            "sources": [{"sourceId": "sam.1", "busId": "master"}],
        },
    }
    frames = int(0.3 * RATE)
    plan = plan_from_track(track, frames=frames)

    quiet = execute_plan(plan).audio
    plain = execute_plan(plan_from_track(_track("abstract_pm", duration=0.3),
                                        frames=frames)).audio
    ratio = np.max(np.abs(quiet)) / max(np.max(np.abs(plain)), 1e-12)
    assert ratio == pytest.approx(10 ** (-6.0 / 20.0), rel=0.02)


# --- failure ----------------------------------------------------------------


def test_one_broken_source_does_not_lose_the_rest_of_the_scene():
    """Five working sources and one missing asset should render five."""

    track = _track("abstract_pm", duration=0.3, sources=2)
    track["steps"][0]["voices"][1]["params"] = {
        **_params("hrtf"),
        "hrtfAsset": "/nonexistent/missing.sofa",
    }
    frames = int(0.3 * RATE)
    plan = plan_from_track(track, frames=frames)

    executed = execute_plan(plan)
    assert "sam.1" in executed.stems
    assert "sam.2" not in executed.stems
    assert executed.diagnostics["sourcesFailed"] == 1
    assert any("sam.2" in issue.path for issue in executed.warnings)
    assert np.max(np.abs(executed.audio)) > 0.0


def test_the_plan_s_own_warnings_travel_with_the_render():
    track = _track("abstract_pm", duration=0.3)
    track["steps"][0]["voices"][0]["params"]["rendererMode"] = "not_a_renderer"
    plan = plan_from_track(track, frames=int(0.3 * RATE))

    executed = execute_plan(plan)
    assert any("not_a_renderer" in issue.message for issue in executed.warnings)


# --- reporting --------------------------------------------------------------


def test_the_result_describes_itself_for_a_manifest():
    plan = plan_from_track(_track("abstract_pm", duration=0.3), frames=int(0.3 * RATE))
    described = execute_plan(plan).describe()

    import json

    assert json.loads(json.dumps(described))["renderers"] == ["abstract_pm"]
    assert described["sources"][0]["renderer"] == "abstract_pm"


def test_rendering_one_source_directly_places_it_in_the_window():
    frames = int(0.5 * RATE)
    plan = plan_from_track(_track("abstract_pm", duration=0.5), frames=frames)
    source = plan.sources[0]

    stem = render_source_window(
        source, sample_rate_hz=RATE, start_sample=0, frames=frames
    )
    assert stem.shape == (2, frames)
    assert np.max(np.abs(stem)) > 0.0


# --- preview goes through the plan ------------------------------------------


def test_a_scene_preview_is_what_an_export_of_that_window_produces():
    """Auditioning the project and exporting it must not be two readings."""

    from src.audio.sam_workbench.preview import render_scene_preview

    track = _track("abstract_pm", duration=0.4, sources=2)
    frames = int(0.4 * RATE)

    preview = render_scene_preview(
        track, duration_s=0.4, fade_ms=0.0, ceiling_dbfs=0.0
    )
    plan = plan_from_track(track, start_sample=0, frames=frames)
    executed = execute_plan(plan, start_sample=0, frames=frames)

    span = min(preview.frames, executed.frames)
    assert span > 0
    assert np.allclose(
        preview.audio[:span].astype(np.float64), executed.audio[:, :span].T, atol=1e-6
    )


def test_a_scene_preview_hears_every_source_not_just_the_first():
    """The old preview auditioned one voice and called it the project."""

    from src.audio.sam_workbench.preview import render_scene_preview

    one = render_scene_preview(
        _track("abstract_pm", duration=0.4, sources=1), duration_s=0.4, fade_ms=0.0
    )
    two = render_scene_preview(
        _track("abstract_pm", duration=0.4, sources=2), duration_s=0.4, fade_ms=0.0
    )
    assert two.peak > one.peak


def test_a_scene_preview_respects_mute():
    from src.audio.sam_workbench.preview import render_scene_preview

    track = _track("abstract_pm", duration=0.3)
    track["sam_scene"] = {
        "sources": [{"id": "sam.1"}],
        "routing": {
            "buses": [{"id": "master"}],
            "sources": [{"sourceId": "sam.1", "busId": "master", "muted": True}],
        },
    }
    preview = render_scene_preview(track, duration_s=0.3, fade_ms=0.0)
    assert preview.peak == pytest.approx(0.0, abs=1e-9)


def test_a_scene_preview_of_a_later_window_is_that_window():
    from src.audio.sam_workbench.preview import render_scene_preview

    track = _track("abstract_pm", duration=0.6)
    whole = render_scene_preview(track, duration_s=0.6, fade_ms=0.0, ceiling_dbfs=0.0)
    later = render_scene_preview(
        track, duration_s=0.2, start_time_s=0.3, fade_ms=0.0, ceiling_dbfs=0.0
    )

    offset = int(0.3 * RATE)
    span = later.frames
    assert np.allclose(
        later.audio.astype(np.float64),
        whole.audio[offset : offset + span].astype(np.float64),
        atol=2e-6,
    )


def test_a_scene_preview_is_capped_like_any_other():
    from src.audio.sam_workbench.preview import MAX_PREVIEW_SECONDS, render_scene_preview

    preview = render_scene_preview(
        _track("abstract_pm", duration=0.2), duration_s=MAX_PREVIEW_SECONDS + 30.0
    )
    assert preview.truncated is True
    assert preview.duration_s <= MAX_PREVIEW_SECONDS

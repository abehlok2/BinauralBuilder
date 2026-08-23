"""Export as a background job, and what that has to guarantee.

Final export used to run on the GUI thread and keep the window alive by calling
``QApplication.processEvents()`` from inside the render. That is not a
background job: it reads the live project while the user can still edit it, it
cannot be refused, and nothing is watching for a cancel. These tests hold the
replacement to the four things that makes it a job - an immutable snapshot, a
cancel that takes effect, no partial file left behind, and honest metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.render_job import (
    RenderCancelled,
    RenderMetrics,
    RenderSnapshot,
    estimate_peak_bytes,
    estimate_render_seconds,
    run_render,
)

RATE = 8000


def _track(duration=0.4, steps=3, renderer="abstract_pm"):
    return {
        "global_settings": {"sample_rate": RATE},
        "steps": [
            {
                "duration": duration,
                "voices": [
                    {
                        "synth_function_name": "binaural_beat",
                        "params": {
                            "amp_left": 0.2, "amp_right": 0.2,
                            "baseFreq": 200.0, "beatFreq": 4.0,
                            "rendererMode": renderer,
                        },
                    }
                ],
            }
            for _ in range(steps)
        ],
    }


# --- the snapshot -----------------------------------------------------------


def test_the_snapshot_does_not_follow_later_edits():
    """The file that appears is the project as it was at the button press."""

    track = _track()
    snapshot = RenderSnapshot.of(track, "/tmp/out.wav")

    track["steps"][0]["duration"] = 999.0
    track["global_settings"]["sample_rate"] = 96_000
    track["steps"].append({"duration": 5.0, "voices": []})

    assert snapshot.sample_rate_hz == RATE
    assert snapshot.duration_s == pytest.approx(1.2)


def test_the_snapshot_reads_its_own_shape():
    snapshot = RenderSnapshot.of(_track(renderer="hrtf"), "/tmp/out.wav")
    assert snapshot.renderer_ids == ("hrtf",)
    assert snapshot.duration_s == pytest.approx(1.2)


def test_a_malformed_duration_does_not_break_the_estimate():
    track = _track()
    track["steps"][1]["duration"] = "not a number"
    snapshot = RenderSnapshot.of(track, "/tmp/out.wav")
    assert snapshot.duration_s == pytest.approx(0.8)


# --- estimates --------------------------------------------------------------


def test_the_slowest_renderer_decides_the_estimate():
    """The slowest is what the user waits for."""

    fast = RenderSnapshot.of(_track(renderer="abstract_pm"), "/tmp/a.wav")
    slow = RenderSnapshot.of(_track(renderer="hybrid"), "/tmp/b.wav")
    assert estimate_render_seconds(slow) > estimate_render_seconds(fast)


def test_a_measured_throughput_replaces_the_guess():
    snapshot = RenderSnapshot.of(_track(duration=60.0, steps=1), "/tmp/a.wav")
    assert estimate_render_seconds(snapshot, throughput=30.0) == pytest.approx(2.0)


def test_the_memory_estimate_grows_with_the_track():
    short = RenderSnapshot.of(_track(duration=1.0, steps=1), "/tmp/a.wav")
    long = RenderSnapshot.of(_track(duration=600.0, steps=1), "/tmp/b.wav")
    assert estimate_peak_bytes(long) > estimate_peak_bytes(short)


# --- running it -------------------------------------------------------------


def test_a_render_writes_its_file_and_reports_what_it_cost(tmp_path):
    destination = tmp_path / "out.wav"
    seen: list[float] = []

    outcome = run_render(
        RenderSnapshot.of(_track(), str(destination)), progress=seen.append
    )

    assert outcome.succeeded and destination.exists()
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert outcome.metrics.wall_s > 0.0
    assert outcome.metrics.audio_seconds_per_second > 0.0
    assert "realtime" in outcome.metrics.summary()


def test_a_cancel_stops_the_render_and_leaves_no_file(tmp_path):
    """A half-written export plays, sounds right, and stops early."""

    destination = tmp_path / "out.wav"
    calls = {"n": 0}

    def cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    outcome = run_render(
        RenderSnapshot.of(_track(duration=0.4, steps=8), str(destination)),
        should_cancel=cancel_after_first,
    )

    assert outcome.cancelled is True
    assert outcome.succeeded is False
    assert not destination.exists()


def test_a_failing_render_leaves_no_file(tmp_path, monkeypatch):
    destination = tmp_path / "out.wav"

    def explode(*_args, **_kwargs):
        destination.write_bytes(b"partial")
        raise RuntimeError("the engine gave up")

    monkeypatch.setattr(
        "src.synth_functions.sound_creator.generate_audio", explode
    )
    outcome = run_render(RenderSnapshot.of(_track(), str(destination)))

    assert outcome.failed is True
    assert "gave up" in outcome.error
    assert not destination.exists()


def test_a_failed_render_does_not_delete_an_earlier_export(tmp_path, monkeypatch):
    """The user's previous file is not this render's to remove."""

    destination = tmp_path / "out.wav"
    destination.write_bytes(b"the good one")

    def explode(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(
        "src.synth_functions.sound_creator.generate_audio", explode
    )
    outcome = run_render(RenderSnapshot.of(_track(), str(destination)))

    assert outcome.failed is True
    assert destination.read_bytes() == b"the good one"


def test_an_engine_that_reports_failure_is_a_failure(tmp_path, monkeypatch):
    """A falsy return is a failure even though nothing was raised."""

    destination = tmp_path / "out.wav"
    monkeypatch.setattr(
        "src.synth_functions.sound_creator.generate_audio",
        lambda *_args, **_kwargs: False,
    )
    outcome = run_render(RenderSnapshot.of(_track(), str(destination)))

    assert outcome.succeeded is False
    assert outcome.failed is True
    assert outcome.error
    # Still measured, so a failure that took ten minutes says so.
    assert outcome.metrics.wall_s > 0.0


# --- cancellation reaches the engine ----------------------------------------


def test_the_engine_propagates_cancellation_rather_than_swallowing_it():
    """Every progress site tolerates a raising callback; this one must not be."""

    import src.synth_functions.sound_creator as sound_creator

    seen = {"n": 0}

    def cancel(_fraction):
        seen["n"] += 1
        if seen["n"] > 1:
            raise RenderCancelled()

    with pytest.raises(RenderCancelled):
        sound_creator.assemble_track_from_data(
            _track(duration=0.3, steps=6), RATE, 0.0, progress_callback=cancel
        )


def test_an_ordinary_callback_error_is_still_tolerated():
    """A buggy progress callback must not lose an otherwise fine render."""

    import src.synth_functions.sound_creator as sound_creator

    def buggy(_fraction):
        raise ValueError("badly written callback")

    audio = sound_creator.assemble_track_from_data(
        _track(duration=0.2, steps=2), RATE, 0.0, progress_callback=buggy
    )
    assert isinstance(audio, np.ndarray) and audio.size > 0


# --- metrics ----------------------------------------------------------------


def test_throughput_is_audio_over_wall_clock():
    assert RenderMetrics(duration_s=60.0, wall_s=15.0).audio_seconds_per_second == 4.0
    assert RenderMetrics().audio_seconds_per_second == 0.0
    assert RenderMetrics().summary() == "not measured"


# --- the Qt side -----------------------------------------------------------


def test_the_manager_runs_a_job_off_the_calling_thread(qtbot, tmp_path):
    from PyQt5.QtCore import QThread

    from src.ui.render_job import RenderJobManager

    destination = tmp_path / "out.wav"
    manager = RenderJobManager()
    outcomes: list[object] = []
    manager.finished.connect(lambda _worker, outcome: outcomes.append(outcome))

    worker = manager.start(RenderSnapshot.of(_track(), str(destination)))
    assert manager.busy and manager.active_count == 1
    # The worker lives on its own thread, which is the whole point.
    assert worker.thread() is not QThread.currentThread()

    qtbot.waitUntil(lambda: bool(outcomes), timeout=30_000)
    assert outcomes[0].succeeded
    assert manager.busy is False
    assert manager.active_count == 0


def test_the_manager_measures_throughput_for_the_next_estimate(qtbot, tmp_path):
    from src.ui.render_job import RenderJobManager

    manager = RenderJobManager()
    outcomes: list[object] = []
    manager.finished.connect(lambda _worker, outcome: outcomes.append(outcome))

    before = manager.estimate(RenderSnapshot.of(_track(), str(tmp_path / "a.wav")))
    assert before["measured"] is False

    manager.start(RenderSnapshot.of(_track(), str(tmp_path / "b.wav")))
    qtbot.waitUntil(lambda: bool(outcomes), timeout=30_000)

    after = manager.estimate(RenderSnapshot.of(_track(), str(tmp_path / "c.wav")))
    assert after["measured"] is True


def test_cancelling_through_the_manager_leaves_no_file(qtbot, tmp_path):
    from src.ui.render_job import RenderJobManager

    destination = tmp_path / "out.wav"
    manager = RenderJobManager()
    outcomes: list[object] = []
    manager.finished.connect(lambda _worker, outcome: outcomes.append(outcome))

    manager.start(RenderSnapshot.of(_track(duration=0.5, steps=40), str(destination)))
    manager.cancel_all()

    qtbot.waitUntil(lambda: bool(outcomes), timeout=30_000)
    assert outcomes[0].cancelled is True
    assert not destination.exists()


def test_the_worker_never_needs_a_widget():
    """Workers must not touch the GUI; the module must not even import one."""

    source = Path("src/ui/render_job.py").read_text()
    assert "QWidget" not in source
    assert "QMessageBox" not in source


# --- step audition ----------------------------------------------------------


def _step(duration=0.3):
    return {
        "duration": duration,
        "description": "test step",
        "voices": [
            {
                "synth_function_name": "binaural_beat",
                "params": {
                    "amp_left": 0.2, "amp_right": 0.2,
                    "baseFreq": 200.0, "beatFreq": 4.0,
                },
            }
        ],
    }


def test_a_step_preview_snapshot_does_not_follow_later_edits():
    """The same discipline as a render, for the same reason."""

    from src.audio.render_job import StepPreviewSnapshot

    step = _step()
    settings = {"sample_rate": RATE}
    snapshot = StepPreviewSnapshot.of(step, settings, 0.3, step_index=2)

    step["duration"] = 99.0
    step["voices"].clear()
    settings["sample_rate"] = 96_000

    assert snapshot.sample_rate_hz == RATE
    assert snapshot.step_data["duration"] == pytest.approx(0.3)
    assert snapshot.step_data["voices"]
    assert snapshot.step_index == 2


def test_a_step_preview_produces_stereo_audio():
    from src.audio.render_job import StepPreviewSnapshot, run_step_preview

    outcome = run_step_preview(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3)
    )
    assert outcome.succeeded
    assert outcome.audio.shape[1] == 2
    assert outcome.audio.shape[0] > 0
    assert outcome.metrics.wall_s > 0.0


def test_a_step_preview_cancelled_before_it_starts_does_no_work():
    from src.audio.render_job import StepPreviewSnapshot, run_step_preview

    outcome = run_step_preview(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3),
        should_cancel=lambda: True,
    )
    assert outcome.cancelled is True
    assert outcome.audio is None


def test_a_step_preview_cancelled_while_running_discards_its_result():
    """It cannot be interrupted; what it must not do is arrive and play."""

    from src.audio.render_job import StepPreviewSnapshot, run_step_preview

    calls = {"n": 0}

    def cancel_after_the_check_before_work() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    outcome = run_step_preview(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3),
        should_cancel=cancel_after_the_check_before_work,
    )
    assert outcome.cancelled is True
    assert outcome.audio is None


def test_a_failing_step_preview_reports_rather_than_raises(monkeypatch):
    from src.audio.render_job import StepPreviewSnapshot, run_step_preview

    monkeypatch.setattr(
        "src.synth_functions.sound_creator.generate_single_step_audio_segment",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no engine")),
    )
    outcome = run_step_preview(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3)
    )
    assert outcome.failed is True
    assert "no engine" in outcome.error


def test_an_empty_step_preview_is_a_failure_not_a_success(monkeypatch):
    from src.audio.render_job import StepPreviewSnapshot, run_step_preview

    monkeypatch.setattr(
        "src.synth_functions.sound_creator.generate_single_step_audio_segment",
        lambda *_a, **_k: np.zeros((0, 2), dtype=np.float32),
    )
    outcome = run_step_preview(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3)
    )
    assert outcome.failed is True


def test_the_step_preview_job_runs_off_the_calling_thread(qtbot):
    from PyQt5.QtCore import QThread

    from src.audio.render_job import StepPreviewSnapshot
    from src.ui.render_job import StepPreviewJob

    job = StepPreviewJob()
    outcomes = []
    job.finished.connect(outcomes.append)

    worker = job.start(
        StepPreviewSnapshot.of(_step(), {"sample_rate": RATE}, 0.3)
    )
    assert job.busy is True
    assert worker.thread() is not QThread.currentThread()

    qtbot.waitUntil(lambda: bool(outcomes), timeout=30_000)
    assert outcomes[0].succeeded
    assert job.busy is False


def test_starting_a_second_step_preview_cancels_the_first(qtbot):
    """Clicking through steps must not play the one you moved away from."""

    from src.audio.render_job import StepPreviewSnapshot
    from src.ui.render_job import StepPreviewJob

    job = StepPreviewJob()
    outcomes = []
    job.finished.connect(outcomes.append)

    first = job.start(
        StepPreviewSnapshot.of(_step(1.0), {"sample_rate": RATE}, 1.0, step_index=0)
    )
    job.start(
        StepPreviewSnapshot.of(_step(0.2), {"sample_rate": RATE}, 0.2, step_index=1)
    )
    assert first.cancelled is True

    qtbot.waitUntil(lambda: len(outcomes) >= 2, timeout=60_000)
    playable = [entry for entry in outcomes if entry.succeeded]
    assert playable and all(entry.step_index == 1 for entry in playable)

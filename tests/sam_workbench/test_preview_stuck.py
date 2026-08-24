"""Previews must come back - quickly, at any Start-at time.

The rules this file protects:

* reconstructing HRTF history for a late window costs a *bounded* pre-roll,
  never a re-render of everything before it (the bug that left previews
  spinning on "Rendering…" whenever Start-at grew);
* a bounded pre-roll is deep enough that the audible result matches what a
  full from-zero render would have produced;
* a slot raising while handling a result cannot wedge the worker thread -
  quitting the thread is connected first and directly.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtCore import QEventLoop, QTimer

from src.audio.sam_workbench.compat import _hrtf_preroll_frames
from src.ui.sam_workbench_dialog import SamWorkbenchDialog

FIXTURE = "tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa"
RATE = 44_100


def _orbit():
    return {
        "schemaVersion": 2,
        "coordinateSystem": "listener_relative_cartesian",
        "geometry": {
            "type": "horizontal_orbit",
            "parameters": {"radius_m": 1.5},
        },
        "traversal": {"mode": "loop", "durationS": 4.0},
    }


def _hrtf_voice(**overrides):
    params = {
        "rendererMode": "hrtf",
        "hrtfAsset": FIXTURE,
        "carrierFreq": 220.0,
        "amp": 0.4,
        "canonicalTrajectory": _orbit(),
    }
    params.update(overrides)
    return {"synth_function_name": "spatial_angle_modulation_sam2", "params": params}


def _pump(qtbot, seconds: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec_()


# --- the bounded pre-roll ----------------------------------------------------


def test_preroll_is_bounded_and_covers_every_delay():
    from src.audio.sam_workbench.render.hrtf import HRTFRendererSpec

    plain = HRTFRendererSpec(sofa_path=FIXTURE, trajectory=lambda t: np.zeros((len(np.atleast_1d(t)), 3)))
    far = HRTFRendererSpec(
        sofa_path=FIXTURE,
        trajectory=lambda t: np.zeros((len(np.atleast_1d(t)), 3)),
        propagation_delay=True,
        maximum_distance_m=100.0,
    )
    small = _hrtf_preroll_frames(plain, RATE)
    large = _hrtf_preroll_frames(far, RATE)

    assert 0 < small < 20_000
    assert large > small + 10_000  # the 100 m delay line dominates
    # Whatever the configuration, a couple of seconds of history is plenty.
    assert large < 2.0 * RATE


def test_a_late_window_matches_full_history(monkeypatch):
    """A bounded pre-roll must be indistinguishable from unbounded history.

    The same window is rendered twice: once with the production pre-roll,
    once forced to rebuild history from absolute zero (the pre-bug behaviour).
    The trajectory loops every 4 s, so both see identical directions; only
    renderer memory differs. Any seam would show up here.
    """

    import src.audio.sam_workbench.compat as compat

    voice = _hrtf_voice()
    frames = 8192
    start_sample = int(12.0 * RATE)

    bounded = np.asarray(
        compat._render_hrtf_voice(voice["params"], frames, float(RATE),
                                  start_sample=start_sample, block_size=1024)
    )
    monkeypatch.setattr(compat, "_HRTF_PREROLL_TAPS", start_sample)
    full = np.asarray(
        compat._render_hrtf_voice(voice["params"], frames, float(RATE),
                                  start_sample=start_sample, block_size=1024)
    )

    assert bounded.shape == full.shape == (2, frames)
    assert np.all(np.isfinite(bounded))
    np.testing.assert_allclose(bounded, full, rtol=1e-9, atol=1e-9)


def test_a_late_window_renders_quickly():
    """The old code re-rendered every sample since zero; this must not."""

    from src.audio.sam_workbench.compat import _render_hrtf_voice

    voice = _hrtf_voice()
    started = time.perf_counter()
    _render_hrtf_voice(
        voice["params"], 8192, float(RATE),
        start_sample=int(600.0 * RATE), block_size=2048,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"a ten-minute-offset render took {elapsed:.1f} s"


# --- the dialog end to end ---------------------------------------------------


@pytest.mark.parametrize("mode,start_at", [
    ("hrtf", 300.0),
    ("hybrid", 300.0),
])
def test_previews_at_late_start_times_return(qtbot, mode, start_at):
    voice = _hrtf_voice()
    voice["params"]["rendererMode"] = mode
    dialog = SamWorkbenchDialog(dict(voice))
    qtbot.addWidget(dialog)

    dialog.preview_seconds.setValue(0.4)
    dialog.preview_start_seconds.setValue(start_at)
    dialog.start_preview()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if "Rendering" not in dialog.status_label.text():
            break
        _pump(qtbot, 0.05)

    assert "Preview ready" in dialog.status_label.text(), dialog.status_label.text()


def test_a_raising_slot_cannot_wedge_the_worker_thread(qtbot):
    dialog = SamWorkbenchDialog(_hrtf_voice())
    qtbot.addWidget(dialog)

    original = dialog.analysis_panel.set_audio
    def explode(audio, rate):
        raise RuntimeError("analysis exploded")
    dialog.analysis_panel.set_audio = explode

    dialog.preview_seconds.setValue(0.3)
    dialog.start_preview()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if dialog.preview_button.isEnabled():
            break
        _pump(qtbot, 0.05)

    # The thread finished, the button came back, and the failure was named.
    assert dialog.preview_button.isEnabled(), "worker thread never quit"
    assert "could not be shown" in dialog.status_label.text()
    dialog.analysis_panel.set_audio = original

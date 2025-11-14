from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import SessionStreamPlayer
import src.audio.session_stream as session_stream


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, state):
        for callback in list(self._callbacks):
            callback(state)


class FakeAudioOutput:
    def __init__(self, fmt, parent=None):
        self.format = fmt
        self.parent = parent
        self.stateChanged = FakeSignal()
        self.started_devices = []
        self.suspended = False
        self.stopped = False
        self._error = getattr(session_stream, "QAudio", None)

    def start(self, device):
        self.started_devices.append(device)

    def stop(self):
        self.stopped = True

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.suspended = False

    def error(self):
        if self._error is None:
            return 0
        return self._error.NoError


@pytest.fixture
def track_data():
    sample_rate = 4000
    return {
        "global_settings": {
            "sample_rate": sample_rate,
            "crossfade_duration": 0.05,
            "crossfade_curve": "linear",
        },
        "steps": [
            {"duration": 0.30, "crossfade_duration": 0.05, "level": 0.25},
            {"duration": 0.20, "crossfade_duration": 0.0, "level": -0.5},
        ],
    }


@pytest.fixture(autouse=True)
def stub_single_step(monkeypatch, track_data):
    sample_rate = track_data["global_settings"]["sample_rate"]

    def fake_generate(step, global_settings, target_duration, duration_override=None):
        n = int(target_duration * sample_rate)
        value = float(step.get("level", 0.0))
        return np.full((n, 2), value, dtype=np.float32)

    monkeypatch.setattr(session_stream, "generate_single_step_audio_segment", fake_generate)
    return fake_generate


def test_stream_player_emits_expected_pcm(track_data):
    sample_rate = track_data["global_settings"]["sample_rate"]
    step1 = track_data["steps"][0]
    step2 = track_data["steps"][1]

    player = SessionStreamPlayer(
        track_data,
        use_prebuffer=False,
        ring_buffer_seconds=10.0,
        audio_output_factory=FakeAudioOutput,
        validate_format=False,
    )

    progress_updates = []
    remaining_updates = []
    player.set_progress_callback(progress_updates.append)
    player.set_time_remaining_callback(remaining_updates.append)

    player.start()

    device = player._buffer_device
    assert device is not None
    data = device.take_all()

    crossfade_samples = int(step1["crossfade_duration"] * sample_rate)
    expected_samples = (
        int(step1["duration"] * sample_rate)
        + int(step2["duration"] * sample_rate)
        - crossfade_samples
    )

    assert len(data) == expected_samples * 4
    assert progress_updates
    assert progress_updates[-1] == pytest.approx(1.0)
    assert remaining_updates[-1] == pytest.approx(0.0, abs=1e-6)

    player.stop()


def test_stream_player_pause_and_stop(track_data):
    player = SessionStreamPlayer(
        track_data,
        use_prebuffer=False,
        ring_buffer_seconds=1.0,
        audio_output_factory=FakeAudioOutput,
        validate_format=False,
    )

    player.start()
    audio_output = player._audio_output
    assert audio_output is not None

    player.pause()
    assert audio_output.suspended is True

    player.resume()
    assert audio_output.suspended is False

    player.stop()
    assert audio_output.stopped is True
    assert player._buffer_device is None

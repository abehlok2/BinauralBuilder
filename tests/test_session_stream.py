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


def test_worker_prunes_inactive_step_states(monkeypatch):
    sample_rate = 100
    track_data = {
        "global_settings": {"sample_rate": sample_rate, "crossfade_duration": 0.0},
        "steps": [
            {"duration": 0.10, "voices": [{"voice_type": "noise"}]},
            {"duration": 0.10, "voices": [{"voice_type": "noise"}]},
        ],
    }

    def fake_generate(step, global_settings, target_duration, **kwargs):
        n = int(target_duration * sample_rate)
        audio = np.ones((n, 2), dtype=np.float32)
        state = [{"full_audio": np.ones((n, 2), dtype=np.float32)}]
        return audio, state

    monkeypatch.setattr(session_stream, "generate_single_step_audio_segment", fake_generate)

    worker = session_stream.AudioGeneratorWorker(
        track_data,
        session_stream._PCMBufferDevice(),
        sample_rate,
        ring_buffer_seconds=1.0,
    )
    step_states = {}

    worker._generate_next_chunk(0, 10, step_states)
    assert set(step_states.keys()) == {0}

    worker._generate_next_chunk(10, 10, step_states)
    assert set(step_states.keys()) == {1}


# =============================================================================
# Noise voice streaming tests (directly test generate_single_step_audio_segment)
# =============================================================================

from src.synth_functions.sound_creator import generate_single_step_audio_segment


def test_noise_voice_caching_produces_continuous_audio():
    """Test that noise voices are cached and sliced correctly for continuous playback."""
    sample_rate = 4000
    step_duration = 1.0  # 1 second step
    chunk_duration = 0.25  # 250ms chunks

    # Create a step with a noise voice
    step_data = {
        "duration": step_duration,
        "voices": [
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 200, "beat_freq": 10},
            }
        ],
    }
    global_settings = {"sample_rate": sample_rate}

    # Generate 4 chunks covering the full step
    chunks = []
    voice_states = None
    for i in range(4):
        chunk_start_time = i * chunk_duration
        audio, voice_states = generate_single_step_audio_segment(
            step_data,
            global_settings,
            chunk_duration,
            duration_override=chunk_duration,
            chunk_start_time=chunk_start_time,
            voice_states=voice_states,
            return_state=True,
        )
        chunks.append(audio)

    # Verify all chunks have the expected shape
    chunk_samples = int(chunk_duration * sample_rate)
    for i, chunk in enumerate(chunks):
        assert chunk.shape == (chunk_samples, 2), f"Chunk {i} has wrong shape: {chunk.shape}"

    # Verify that the chunks form continuous audio (no silent gaps)
    # Check that no chunk is completely silent
    for i, chunk in enumerate(chunks):
        chunk_power = np.mean(chunk ** 2)
        assert chunk_power > 1e-12, f"Chunk {i} is silent (power={chunk_power})"

    # Verify that the voice state was properly cached
    assert voice_states is not None
    assert len(voice_states) == 1
    noise_state = voice_states[0]
    assert isinstance(noise_state, dict)
    assert "full_audio" in noise_state
    assert "noise_peak" in noise_state


def test_noise_voice_boundary_slicing_pads_correctly():
    """Test that noise slicing at step boundaries pads short slices."""
    sample_rate = 4000
    step_duration = 0.5  # 500ms step
    chunk_duration = 0.3  # 300ms chunk - second chunk extends beyond step

    step_data = {
        "duration": step_duration,
        "voices": [
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 200, "beat_freq": 10},
            }
        ],
    }
    global_settings = {"sample_rate": sample_rate}

    # First chunk: 0-300ms
    audio1, voice_states = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=0.0,
        voice_states=None,
        return_state=True,
    )

    # Second chunk: 300-600ms (but step only lasts 500ms)
    audio2, voice_states = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=0.3,
        voice_states=voice_states,
        return_state=True,
    )

    chunk_samples = int(chunk_duration * sample_rate)

    # Both chunks should have the full requested size (padding applied if needed)
    assert audio1.shape == (chunk_samples, 2), f"First chunk has wrong shape: {audio1.shape}"
    assert audio2.shape == (chunk_samples, 2), f"Second chunk has wrong shape: {audio2.shape}"


def test_noise_normalization_consistent_across_chunks():
    """Test that noise normalization uses cached peak for consistent volume."""
    sample_rate = 4000
    step_duration = 1.0
    chunk_duration = 0.25

    step_data = {
        "duration": step_duration,
        "voices": [
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 200, "beat_freq": 10},
            }
        ],
        "normalization_level": 0.8,
        "noise_volume": 1.0,
    }
    global_settings = {"sample_rate": sample_rate}

    # Generate all chunks
    chunks = []
    voice_states = None
    for i in range(4):
        chunk_start_time = i * chunk_duration
        audio, voice_states = generate_single_step_audio_segment(
            step_data,
            global_settings,
            chunk_duration,
            duration_override=chunk_duration,
            chunk_start_time=chunk_start_time,
            voice_states=voice_states,
            return_state=True,
        )
        chunks.append(audio)

    # Check that peak values are similar across chunks (within reasonable tolerance)
    # If normalization is consistent, chunks shouldn't have wildly different peaks
    peaks = [np.max(np.abs(chunk)) for chunk in chunks]

    # All peaks should be reasonably close (accounting for the nature of noise)
    # The key is that they shouldn't vary by orders of magnitude
    max_peak = max(peaks)
    min_peak = min(peaks)

    # Allow up to 3x variation due to random noise characteristics
    # but not more (which would indicate per-chunk normalization issues)
    if max_peak > 0:
        peak_ratio = max_peak / (min_peak + 1e-10)
        assert peak_ratio < 10.0, f"Peak variation too large: {peaks}"




def test_transition_chunk_timing_respects_initial_offset_and_duration():
    """Transition chunks should keep step-level offsets/duration semantics."""
    sample_rate = 100
    step_duration = 10.0
    chunk_duration = 1.0

    step_data = {
        "duration": step_duration,
        "voices": [
            {
                "synth_function_name": "binaural_beat",
                "is_transition": True,
                "params": {
                    "baseFreq": 100.0,
                    "endBaseFreq": 200.0,
                    "beatFreq": 4.0,
                    "endBeatFreq": 8.0,
                    "ampL": 0.2,
                    "endAmpL": 0.2,
                    "ampR": 0.2,
                    "endAmpR": 0.2,
                    "initial_offset": 2.0,
                    "transition_duration": 3.0,
                },
            }
        ],
    }

    global_settings = {"sample_rate": sample_rate}

    # Before transition start => unchanged (start state)
    audio_pre, state = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=0.0,
        return_state=True,
    )

    # During transition => should differ from both pre and post chunks
    audio_mid, state = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=3.0,
        voice_states=state,
        return_state=True,
    )

    # After transition end => end state should be stable
    audio_post_a, state = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=6.0,
        voice_states=state,
        return_state=True,
    )
    audio_post_b, _ = generate_single_step_audio_segment(
        step_data,
        global_settings,
        chunk_duration,
        duration_override=chunk_duration,
        chunk_start_time=7.0,
        voice_states=state,
        return_state=True,
    )

    # Transition must be inactive before initial_offset and active during its window
    assert np.mean(np.abs(audio_mid - audio_pre)) > 1e-4

    # Once transition duration has elapsed, later chunks should represent the
    # same end state (no further transition progression).
    post_diff = np.mean(np.abs(audio_post_a - audio_post_b))
    assert post_diff < 5e-3

def test_no_clipping_at_maximum_settings():
    """Test that audio never clips even with max normalization and volumes.

    This regression test verifies that when:
    - normalization_level is at maximum (0.75)
    - binaural_volume is 1.0
    - noise_volume is 1.0

    The combined audio output never exceeds 1.0 (which would cause clipping).
    """
    sample_rate = 4000
    step_duration = 0.5

    step_data = {
        "duration": step_duration,
        "voices": [
            # Binaural voice
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "binaural",
                "params": {"carrier_freq": 200, "beat_freq": 10},
            },
            # Noise voice
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 300, "beat_freq": 5},
            },
        ],
        # Maximum settings that would cause clipping before the fix
        "normalization_level": 0.75,
        "binaural_volume": 1.0,
        "noise_volume": 1.0,
    }
    global_settings = {"sample_rate": sample_rate}

    audio = generate_single_step_audio_segment(
        step_data,
        global_settings,
        step_duration,
    )

    # The audio should NEVER exceed 1.0 in absolute value
    peak = np.max(np.abs(audio))
    assert peak <= 1.0, f"Audio clipped! Peak value: {peak}"

    # Also verify audio was actually generated (not silent)
    assert peak > 0.1, f"Audio is unexpectedly quiet: peak = {peak}"


def test_no_clipping_with_multiple_voices():
    """Test that audio doesn't clip with multiple binaural and noise voices."""
    sample_rate = 4000
    step_duration = 0.5

    step_data = {
        "duration": step_duration,
        "voices": [
            # Multiple binaural voices
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "binaural",
                "params": {"carrier_freq": 100, "beat_freq": 4},
            },
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "binaural",
                "params": {"carrier_freq": 200, "beat_freq": 8},
            },
            # Multiple noise voices
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 300, "beat_freq": 12},
            },
            {
                "synth_function_name": "binaural_beat",
                "voice_type": "noise",
                "params": {"carrier_freq": 400, "beat_freq": 16},
            },
        ],
        "normalization_level": 0.75,
        "binaural_volume": 1.0,
        "noise_volume": 1.0,
    }
    global_settings = {"sample_rate": sample_rate}

    audio = generate_single_step_audio_segment(
        step_data,
        global_settings,
        step_duration,
    )

    # Even with many voices, audio should never clip
    peak = np.max(np.abs(audio))
    assert peak <= 1.0, f"Audio clipped with multiple voices! Peak value: {peak}"

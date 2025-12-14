from pathlib import Path
import sys

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.session_engine import SessionAssembler
from src.audio.session_model import Session, SessionPresetChoice, SessionStep
from src.synth_functions import sound_creator


def _register_constant_voice(monkeypatch):
    def constant_voice(duration, sample_rate, amplitude=0.5, channel="both"):
        n = int(duration * sample_rate)
        audio = np.zeros((n, 2), dtype=np.float32)
        if channel == "left":
            audio[:, 0] = amplitude
        elif channel == "right":
            audio[:, 1] = amplitude
        else:
            audio[:, :] = amplitude
        return audio

    monkeypatch.setitem(sound_creator.SYNTH_FUNCTIONS, "test_constant_voice", constant_voice)
    try:
        import binauralbuilder_core.synth_functions.sound_creator as core_sound_creator

        monkeypatch.setitem(
            core_sound_creator.SYNTH_FUNCTIONS, "test_constant_voice", constant_voice
        )
    except Exception:
        pass


def _patch_noise_generation(monkeypatch, noise_value=0.05):
    class DummyParams:
        def __init__(self):
            self.duration_seconds = 0.0
            self.sample_rate = 0
            self.transition = False
            self.lfo_freq = 0.0
            self.sweeps = [{
                "start_min": 200,
                "start_max": 400,
                "start_q": 30,
                "start_casc": 10,
            }]
            self.start_lfo_phase_offset_deg = 0
            self.start_intra_phase_offset_deg = 0
            self.end_lfo_freq = 0.0
            self.end_lfo_phase_offset_deg = 0
            self.end_intra_phase_offset_deg = 0
            self.initial_offset = 0.0
            self.duration = 0.0
            self.input_audio_path = ""
            self.noise_parameters = {"name": "white"}
            self.lfo_waveform = "sine"
            self.static_notches = []

    def fake_params_loader(path):
        return DummyParams()

    def fake_noise_array(duration_seconds, sample_rate, *args, **kwargs):
        n = int(duration_seconds * sample_rate)
        return np.full((n, 2), noise_value, dtype=np.float32), duration_seconds

    monkeypatch.setattr(sound_creator, "load_noise_params", fake_params_loader)
    monkeypatch.setattr(sound_creator, "_generate_swept_notch_arrays", fake_noise_array)
    monkeypatch.setattr(sound_creator, "_generate_swept_notch_arrays_transition", fake_noise_array)


def test_session_assembler_render_with_warmup_crossfade_and_noise(tmp_path, monkeypatch):
    _register_constant_voice(monkeypatch)
    _patch_noise_generation(monkeypatch, noise_value=0.05)

    sample_rate = 8000

    warmup_path = tmp_path / "warmup.wav"
    warmup_samples = int(0.25 * sample_rate)
    warmup_audio = np.full((warmup_samples, 2), 0.2, dtype=np.float32)
    sf.write(warmup_path, warmup_audio, sample_rate)

    binaural_catalog = {
        "presetA": SessionPresetChoice(
            id="presetA",
            label="Const 0.5",
            kind="binaural",
            payload={
                "voice_data": {
                    "synth_function_name": "test_constant_voice",
                    "params": {"amplitude": 0.5},
                }
            },
        ),
        "presetB": SessionPresetChoice(
            id="presetB",
            label="Const 1.0",
            kind="binaural",
            payload={
                "voice_data": {
                    "synth_function_name": "test_constant_voice",
                    "params": {"amplitude": 1.0},
                }
            },
        ),
    }

    noise_catalog = {
        "noise:test": SessionPresetChoice(
            id="noise:test",
            label="Noise",
            kind="noise",
            payload={"params_path": "dummy.noise"},
        )
    }

    session = Session(
        sample_rate=sample_rate,
        crossfade_duration=0.1,
        crossfade_curve="linear",
        background_noise_preset_id="noise:test",
        background_noise_gain=0.5,
        background_noise_start_time=0.0,
        background_noise_fade_in=0.0,
        background_noise_fade_out=0.0,
        background_noise_amp_envelope=[[0.0, 0.5], [2.0, 0.5]],
        steps=[
            SessionStep(
                binaural_preset_id="presetA",
                duration=1.0,
                description="Step 1",
                warmup_clip_path=str(warmup_path),
                crossfade_duration=0.2,
            ),
            SessionStep(
                binaural_preset_id="presetB",
                duration=1.0,
                description="Step 2",
            ),
        ],
    )

    assembler = SessionAssembler(
        session,
        binaural_catalog,
        noise_catalog,
        sample_rate=sample_rate,
        normalization_ceiling=0.9,
    )

    assert assembler.normalization_target == 0.75

    buffer = assembler.render_to_array()

    assert buffer.ndim == 2 and buffer.shape[1] == 2

    expected_duration = session.steps[0].duration + session.steps[1].duration - session.steps[0].crossfade_duration
    expected_samples = int(expected_duration * sample_rate)
    assert buffer.shape[0] == expected_samples

    crossfade_samples = int(session.steps[0].crossfade_duration * sample_rate)
    crossfade_start = int(session.steps[0].duration * sample_rate) - crossfade_samples
    crossfade_mid = crossfade_start + crossfade_samples // 2
    crossfade_end = crossfade_start + crossfade_samples - 1

    start_val = buffer[crossfade_start, 0]
    mid_val = buffer[crossfade_mid, 0]
    end_val = buffer[crossfade_end, 0]

    assert mid_val > start_val
    assert mid_val > end_val
    assert np.isclose(np.max(np.abs(buffer)), assembler.normalization_target, atol=1e-4)

    bg = assembler.track_data["background_noise"]
    assert bg["amp_envelope"] == session.background_noise_amp_envelope

    overlay_paths = [clip["path"] for clip in assembler.track_data["overlay_clips"]]
    assert str(warmup_path) in overlay_paths

    # Warmup plus noise ensures early samples are non-zero after normalization
    assert np.any(buffer[:warmup_samples] > 0.0)

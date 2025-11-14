from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.session_model import (
    Session,
    SessionPresetChoice,
    SessionStep,
    build_binaural_preset_catalog,
    build_noise_preset_catalog,
    session_to_track_data,
)
from src.utils.noise_file import NoiseParams, save_noise_params
from src.utils.voice_file import VoicePreset, save_voice_preset


@pytest.fixture()
def sample_presets(tmp_path: Path):
    voice = VoicePreset(
        synth_function_name="binaural_beat",
        params={"baseFreq": 432.0, "beatFreq": 7.0, "ampL": 0.6, "ampR": 0.6},
        description="Custom Voice",
    )
    voice_path = tmp_path / "custom.voice"
    save_voice_preset(voice, str(voice_path))

    noise = NoiseParams(duration_seconds=90.0, noise_type="brown")
    noise_path = tmp_path / "soothing.noise"
    save_noise_params(noise, str(noise_path))

    return voice_path, noise_path


def test_catalog_builders_include_builtin_and_files(sample_presets, tmp_path):
    voice_path, noise_path = sample_presets
    binaural_catalog = build_binaural_preset_catalog(preset_dirs=[tmp_path])
    noise_catalog = build_noise_preset_catalog(preset_dirs=[tmp_path])

    assert "builtin:theta" in binaural_catalog
    assert f"voice:{voice_path.stem}" in binaural_catalog
    assert f"noise:{noise_path.stem}" in noise_catalog

    theta_voice = binaural_catalog["builtin:theta"].payload["voice_data"]
    assert theta_voice["synth_function_name"] == "binaural_beat"
    assert pytest.approx(theta_voice["params"]["baseFreq"]) == 200.0
    assert pytest.approx(theta_voice["params"]["beatFreq"]) == 5.0


def test_session_to_track_data_conversion(sample_presets, tmp_path):
    voice_path, noise_path = sample_presets
    binaural_catalog = build_binaural_preset_catalog(preset_dirs=[tmp_path])
    noise_catalog = build_noise_preset_catalog(preset_dirs=[tmp_path])

    session = Session(
        steps=[
            SessionStep(
                binaural_preset_id="builtin:alpha",
                duration=120.0,
                description="Alpha Warmup",
                warmup_clip_path=str(tmp_path / "warmup.wav"),
            ),
            SessionStep(
                binaural_preset_id=f"voice:{voice_path.stem}",
                duration=180.0,
                start=120.0,
                noise_preset_id=f"noise:{noise_path.stem}",
                description="Custom Focus",
            ),
        ],
        sample_rate=48000,
        crossfade_duration=5.0,
        crossfade_curve="equal_power",
        output_filename="focus_session.flac",
        background_noise_preset_id=f"noise:{noise_path.stem}",
        background_noise_gain=0.4,
        background_noise_start_time=15.0,
    )

    track_data = session_to_track_data(session, binaural_catalog, noise_catalog)

    globals_cfg = track_data["global_settings"]
    assert globals_cfg["sample_rate"] == 48000
    assert globals_cfg["crossfade_duration"] == pytest.approx(5.0)
    assert globals_cfg["crossfade_curve"] == "equal_power"
    assert globals_cfg["output_filename"] == "focus_session.flac"

    assert track_data["background_noise"]["noise_file"] == str(noise_path)
    assert track_data["background_noise"]["gain"] == pytest.approx(0.4)
    assert track_data["background_noise"]["start_time"] == pytest.approx(15.0)

    assert len(track_data["steps"]) == 2
    first_step = track_data["steps"][0]
    second_step = track_data["steps"][1]

    assert first_step["voices"][0]["synth_function_name"] == "binaural_beat"
    assert first_step["duration"] == pytest.approx(120.0)
    assert first_step["start"] == pytest.approx(0.0)

    assert second_step["voices"][0]["synth_function_name"] == "binaural_beat"
    assert second_step["start"] == pytest.approx(120.0)
    assert second_step["noise_preset_id"] == f"noise:{noise_path.stem}"

    assert track_data["clips"][0]["file_path"] == str(tmp_path / "warmup.wav")
    assert track_data["clips"][0]["start"] == pytest.approx(0.0)

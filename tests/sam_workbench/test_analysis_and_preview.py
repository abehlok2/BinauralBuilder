"""Analysis of rendered audio, and the preview path the GUI plays.

The analysis must agree with the model: `dsp.source` computes instantaneous
frequency and IPD from the specification, this module measures them from the
rendered samples, and a disagreement means the renderer is not doing what the
specification says.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.audio.sam_workbench.analysis import (
    channel_peaks,
    estimate_itd_seconds,
    instantaneous_frequency_hz,
    interaural_level_difference_db,
    interaural_phase_difference_rad,
    magnitude_spectrum_db,
    peak_envelope,
    peak_level_db,
    rms_envelope,
    spectrogram_db,
    summarize_cues,
)
from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.conventions import to_frame_major
from src.audio.sam_workbench.dsp import render_source
from src.audio.sam_workbench.preview import (
    MAX_PREVIEW_SECONDS,
    render_voice_preview,
    to_pcm16_bytes,
)
from src.audio.sam_workbench.render.abstract_pm import preset_exact_symmetric

SAMPLE_RATE = 44_100
VOICE = {
    "synth_function_name": "spatial_angle_modulation_sam2",
    "is_transition": False,
    "params": {"amp": 0.6, "carrierFreq": 220.0, "modFreq": 4.0, "arcWidthDeg": 90.0, "spatialScale": 1.0},
}


@pytest.fixture(scope="module")
def sam_audio() -> np.ndarray:
    return render_sam2_voice(1.0, SAMPLE_RATE, params=VOICE["params"])


# --- waveform ---------------------------------------------------------------


def test_peak_envelope_keeps_the_extremes_of_every_bucket():
    audio = np.zeros((2, 10_000))
    audio[0, 5_000] = 0.9
    audio[1, 5_001] = -0.8
    minimum, maximum = peak_envelope(audio, 100)
    assert maximum.shape == (2, 100)
    assert maximum[0].max() == pytest.approx(0.9)
    assert minimum[1].min() == pytest.approx(-0.8)


def test_envelope_of_short_audio_is_the_audio():
    audio = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    minimum, maximum = peak_envelope(audio, 512)
    np.testing.assert_array_equal(maximum, audio)
    np.testing.assert_array_equal(minimum, audio)
    assert peak_envelope(np.zeros((2, 0)), 8)[0].shape == (2, 0)


def test_rms_and_peak_levels(sam_audio):
    levels = rms_envelope(sam_audio, 64)
    assert levels.shape == (2, 64)
    assert np.all(levels > 0.0)
    np.testing.assert_allclose(channel_peaks(sam_audio), [0.6, 0.6], atol=1e-3)
    assert peak_level_db(sam_audio) == pytest.approx(-4.44, abs=0.05)


def test_frame_major_and_channel_major_input_agree(sam_audio):
    channel_major = np.ascontiguousarray(sam_audio.T)
    np.testing.assert_allclose(channel_peaks(sam_audio), channel_peaks(channel_major))


# --- spectrum ---------------------------------------------------------------


def test_spectrum_peaks_at_the_carrier(sam_audio):
    frequencies, magnitudes = magnitude_spectrum_db(sam_audio, SAMPLE_RATE)
    assert frequencies[int(np.argmax(magnitudes[0]))] == pytest.approx(220.0, abs=2.0)
    assert magnitudes.shape[0] == 2


def test_full_scale_sine_reads_near_zero_dbfs():
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = np.vstack((np.sin(2 * math.pi * 1_000.0 * times),) * 2)
    _, magnitudes = magnitude_spectrum_db(tone, SAMPLE_RATE)
    assert magnitudes.max() == pytest.approx(0.0, abs=0.5)


def test_spectrogram_returns_a_time_frequency_grid(sam_audio):
    frequencies, times, values = spectrogram_db(sam_audio, SAMPLE_RATE, window_frames=1024)
    assert values.shape == (frequencies.size, times.size)
    assert np.all(np.isfinite(values))
    assert times[-1] <= 1.0


def test_spectrum_of_too_short_audio_is_empty():
    frequencies, magnitudes = magnitude_spectrum_db(np.zeros((2, 1)), SAMPLE_RATE)
    assert frequencies.size == 0 and magnitudes.shape[1] == 0


# --- binaural cues ----------------------------------------------------------


def test_measured_instantaneous_frequency_matches_the_model():
    source = preset_exact_symmetric(carrier_hz=220.0, modulation_hz=4.0, depth_rad=1.0)
    audio = render_source(source, SAMPLE_RATE, SAMPLE_RATE, dtype=np.float64)
    measured = instantaneous_frequency_hz(audio, SAMPLE_RATE)
    trim = 2_000
    inside = measured[:, trim:-trim]
    assert inside[0].max() == pytest.approx(224.0, abs=0.5)
    assert inside[0].min() == pytest.approx(216.0, abs=0.5)
    # Opposed ears: the two deviations cancel at every instant.
    np.testing.assert_allclose(inside[0] + inside[1], 440.0, atol=0.5)


def test_measured_ipd_matches_the_model():
    source = preset_exact_symmetric(carrier_hz=220.0, modulation_hz=4.0, depth_rad=0.5)
    audio = render_source(source, SAMPLE_RATE, SAMPLE_RATE, dtype=np.float64)
    measured = interaural_phase_difference_rad(audio)[2_000:-2_000]
    times = np.arange(SAMPLE_RATE)[2_000:-2_000] / SAMPLE_RATE
    expected = 2 * 0.5 * np.sin(2 * math.pi * 4.0 * times)
    np.testing.assert_allclose(measured, expected, atol=0.02)


def test_diotic_audio_has_no_interaural_difference():
    times = np.arange(4_096) / SAMPLE_RATE
    tone = np.vstack((np.sin(2 * math.pi * 200.0 * times),) * 2)
    np.testing.assert_allclose(interaural_phase_difference_rad(tone), 0.0, atol=1e-9)
    np.testing.assert_allclose(
        interaural_level_difference_db(tone, SAMPLE_RATE)[100:-100], 0.0, atol=1e-9
    )


def test_level_difference_follows_the_channel_gains():
    times = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    tone = np.sin(2 * math.pi * 300.0 * times)
    audio = np.vstack((tone, 0.5 * tone))
    measured = interaural_level_difference_db(audio, SAMPLE_RATE)[1_000:-1_000]
    np.testing.assert_allclose(measured, 20 * np.log10(2.0), atol=0.2)


def test_itd_estimate_recovers_a_known_delay():
    """Positive ITD means the left ear leads, the conventional sign."""

    generator = np.random.default_rng(7)
    noise = generator.normal(size=SAMPLE_RATE // 2)
    delay = 30
    left_leads = np.vstack((np.concatenate((noise[delay:], np.zeros(delay))), noise))
    assert estimate_itd_seconds(left_leads, SAMPLE_RATE) == pytest.approx(delay / SAMPLE_RATE, abs=1e-5)

    right_leads = np.vstack((noise, np.concatenate((noise[delay:], np.zeros(delay)))))
    assert estimate_itd_seconds(right_leads, SAMPLE_RATE) == pytest.approx(-delay / SAMPLE_RATE, abs=1e-5)
    assert estimate_itd_seconds(np.zeros((2, 1_000)), SAMPLE_RATE) == 0.0


def test_summary_trims_the_analytic_signal_edges():
    """A faded preview's edges would otherwise dominate the reported range."""

    faded = render_voice_preview(VOICE, sample_rate_hz=SAMPLE_RATE, duration_s=1.0).audio
    untrimmed = summarize_cues(faded, SAMPLE_RATE)
    trimmed = summarize_cues(faded, SAMPLE_RATE, edge_trim_frames=2_205)
    assert untrimmed["frequency_hz_min"] < 100.0  # the fade, not the signal
    assert trimmed["frequency_hz_max"] == pytest.approx(227.3, abs=1.0)
    assert trimmed["frequency_hz_min"] == pytest.approx(212.8, abs=1.0)


# --- preview ----------------------------------------------------------------


def test_preview_uses_the_same_renderer_as_export():
    result = render_voice_preview(VOICE, sample_rate_hz=SAMPLE_RATE, duration_s=1.0, fade_ms=0.0)
    reference = render_sam2_voice(1.0, SAMPLE_RATE, params=VOICE["params"])
    np.testing.assert_allclose(result.audio, reference, rtol=0, atol=1e-6)
    assert result.sample_rate_hz == SAMPLE_RATE
    assert result.frames == SAMPLE_RATE


def test_preview_fades_its_edges_so_playback_cannot_click():
    result = render_voice_preview(VOICE, duration_s=0.5)
    assert abs(float(result.audio[0, 0])) < 1e-6
    assert abs(float(result.audio[-1, 0])) < 1e-3
    interior = result.audio[len(result.audio) // 2]
    assert abs(float(interior[0])) > 0.0


def test_preview_honours_a_start_offset():
    whole = render_voice_preview(VOICE, duration_s=1.0, fade_ms=0.0)
    tail = render_voice_preview(VOICE, duration_s=0.5, start_time_s=0.5, fade_ms=0.0)
    np.testing.assert_allclose(tail.audio, whole.audio[SAMPLE_RATE // 2 :], rtol=0, atol=1e-6)


def test_preview_is_capped_and_reports_it():
    result = render_voice_preview(VOICE, duration_s=MAX_PREVIEW_SECONDS + 30.0)
    assert result.truncated is True
    assert result.duration_s == pytest.approx(MAX_PREVIEW_SECONDS, abs=0.01)


def test_loud_preview_is_limited_and_reported():
    loud = {"params": {**VOICE["params"], "amp": 1.0}}
    result = render_voice_preview(loud, duration_s=0.2, ceiling_dbfs=-6.0)
    assert result.limited is True
    assert result.peak <= 10 ** (-6.0 / 20.0) + 1e-6


def test_transition_voice_preview_uses_the_transition_renderer():
    voice = {
        "is_transition": True,
        "params": {
            "amp": 0.5,
            "startCarrierFreq": 200.0,
            "endCarrierFreq": 400.0,
            "duration": 1.0,
        },
    }
    result = render_voice_preview(voice, duration_s=1.0, fade_ms=0.0)
    expected = render_sam2_voice(
        1.0, SAMPLE_RATE, params=voice["params"], is_transition=True, initial_offset=0.0, transition_duration=1.0
    )
    np.testing.assert_allclose(result.audio, expected, rtol=0, atol=1e-6)


def test_pcm_conversion_is_interleaved_little_endian_16_bit():
    audio = np.array([[1.0, -1.0], [0.0, 0.5]], dtype=np.float32)
    raw = to_pcm16_bytes(audio)
    assert len(raw) == 2 * 2 * 2
    decoded = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    assert decoded[0, 0] == 32767
    assert decoded[0, 1] == -32767
    assert decoded[1, 1] == pytest.approx(16383, abs=1)


def test_pcm_conversion_clips_rather_than_wrapping():
    decoded = np.frombuffer(to_pcm16_bytes(np.array([[4.0, -4.0]]), gain=2.0), dtype="<i2")
    assert decoded.tolist() == [32767, -32767]


def test_pcm_conversion_rejects_channel_major_input(sam_audio):
    with pytest.raises(ValueError):
        to_pcm16_bytes(np.ascontiguousarray(sam_audio.T[:, :3]))
    assert to_pcm16_bytes(to_frame_major(np.ascontiguousarray(sam_audio.T)))

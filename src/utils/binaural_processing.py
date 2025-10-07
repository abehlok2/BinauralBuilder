"""Utility functions for binaural audio encoding and processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
import librosa


@dataclass
class BinauralProcessingConfig:
    """Configuration data for binaural encoding."""

    input_path: str
    output_path: str
    beat_frequency: float
    left_offset_hz: float
    right_offset_hz: float
    band_ranges: Sequence[Tuple[float, float]]
    band_phase_spread: float
    band_detune: float
    lfo_depth: float
    lfo_rate_multiplier: float
    target_rms: float
    crest_soft_knee: float = 1.5


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio data is 2 x N (stereo)."""

    if audio.ndim == 1:
        return np.vstack([audio, audio])
    if audio.shape[0] == 2:
        return audio
    if audio.shape[1] == 2:  # librosa can return Nx2
        return audio.T
    raise ValueError("Unsupported audio shape for stereo conversion: %s" % (audio.shape,))


def _analytic_frequency_shift(signal_in: np.ndarray, sample_rate: int, offset_hz: float) -> np.ndarray:
    """Apply a frequency shift using an analytic signal approach."""

    if abs(offset_hz) < 1e-6:
        return signal_in
    analytic = signal.hilbert(signal_in)
    time = np.arange(len(signal_in)) / float(sample_rate)
    shift = np.exp(1j * 2.0 * np.pi * offset_hz * time)
    shifted = np.real(analytic * shift)
    return shifted.astype(signal_in.dtype, copy=False)


def _bandpass_filter(data: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    low = max(low, 1.0)
    high = min(high, nyquist - 1.0)
    if low >= high:
        return np.zeros_like(data)
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data)


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def _normalize_stereo(left: np.ndarray, right: np.ndarray, target_rms: float) -> Tuple[np.ndarray, np.ndarray]:
    if target_rms <= 0:
        return left, right
    current = max(_rms(left), _rms(right), 1e-9)
    gain = target_rms / current
    return left * gain, right * gain


def _apply_soft_knee(data: np.ndarray, knee: float) -> np.ndarray:
    if knee <= 0:
        return data
    return np.tanh(data * knee) / np.tanh(knee)


def parse_band_ranges(text: str) -> List[Tuple[float, float]]:
    """Parse a user-provided band definition string."""

    if not text.strip():
        return []
    ranges: List[Tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            continue
        low_str, high_str = part.split("-", 1)
        try:
            low = float(low_str)
            high = float(high_str)
        except ValueError:
            continue
        if low >= high:
            continue
        ranges.append((low, high))
    return ranges


def binaural_encode(config: BinauralProcessingConfig) -> None:
    """Process the input file and write the binaural-encoded output."""

    audio, sample_rate = librosa.load(config.input_path, sr=None, mono=False)
    audio = _ensure_stereo(np.asarray(audio))
    left, right = audio[0].astype(np.float32), audio[1].astype(np.float32)

    left = _analytic_frequency_shift(left, sample_rate, config.left_offset_hz)
    right = _analytic_frequency_shift(right, sample_rate, config.right_offset_hz)

    mono_source = 0.5 * (left + right)
    time = np.arange(len(mono_source)) / float(sample_rate)

    if config.band_ranges:
        for index, (low, high) in enumerate(config.band_ranges):
            band_signal = _bandpass_filter(mono_source, sample_rate, low, high)
            if not np.any(band_signal):
                continue
            band_phase = config.band_phase_spread * index
            detune = config.band_detune * index
            phase_l = 2.0 * np.pi * (config.beat_frequency + detune) * time + band_phase
            phase_r = 2.0 * np.pi * (config.beat_frequency - detune) * time - band_phase
            left += 0.5 * band_signal * np.cos(phase_l)
            right += 0.5 * band_signal * np.cos(phase_r)

    if config.lfo_depth > 0:
        lfo_rate = max(config.beat_frequency * config.lfo_rate_multiplier, 0.01)
        lfo = np.sin(2.0 * np.pi * lfo_rate * time)
        depth = np.clip(config.lfo_depth, 0.0, 1.0)
        left *= 1.0 - depth * 0.5 + depth * (0.5 - 0.5 * lfo)
        right *= 1.0 - depth * 0.5 + depth * (0.5 + 0.5 * lfo)

    left, right = _normalize_stereo(left, right, config.target_rms)
    stereo = np.vstack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 1.0:
        stereo /= peak

    stereo = _apply_soft_knee(stereo, config.crest_soft_knee)

    sf.write(config.output_path, stereo.T, sample_rate)

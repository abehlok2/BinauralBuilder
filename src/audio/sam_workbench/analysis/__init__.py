"""Analysis of rendered audio, for the GUI's views and for reports.

Everything here is Qt-free and takes plain arrays, so the same functions serve
the analysis panels, the export report, and the tests.
"""

from __future__ import annotations

from .binaural_cues import (
    analytic_phase,
    estimate_itd_seconds,
    instantaneous_frequency_hz,
    interaural_level_difference_db,
    interaural_phase_difference_rad,
    summarize_cues,
)
from .spectrum import magnitude_spectrum_db, spectrogram_db
from .waveform import channel_peaks, peak_envelope, peak_level_db, rms_envelope

__all__ = [
    "analytic_phase",
    "channel_peaks",
    "estimate_itd_seconds",
    "instantaneous_frequency_hz",
    "interaural_level_difference_db",
    "interaural_phase_difference_rad",
    "magnitude_spectrum_db",
    "peak_envelope",
    "peak_level_db",
    "rms_envelope",
    "spectrogram_db",
    "summarize_cues",
]

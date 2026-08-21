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
from .trajectory_metrics import (
    curvature_1_m,
    path_length_m,
    predicted_ear_distances_m,
    predicted_ild_db,
    predicted_itd_s,
    speeds_m_s,
    trajectory_summary,
)
from .waveform import channel_peaks, peak_envelope, peak_level_db, rms_envelope

__all__ = [
    "analytic_phase",
    "channel_peaks",
    "curvature_1_m",
    "estimate_itd_seconds",
    "instantaneous_frequency_hz",
    "interaural_level_difference_db",
    "interaural_phase_difference_rad",
    "magnitude_spectrum_db",
    "path_length_m",
    "peak_envelope",
    "peak_level_db",
    "predicted_ear_distances_m",
    "predicted_ild_db",
    "predicted_itd_s",
    "rms_envelope",
    "spectrogram_db",
    "speeds_m_s",
    "summarize_cues",
    "trajectory_metrics",
    "ComparisonResult",
    "render_ab_comparison",
    "render_legacy_migration_preview",
]
from .trajectory_metrics import trajectory_metrics
from .comparison import ComparisonResult, render_ab_comparison, render_legacy_migration_preview

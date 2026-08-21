"""Preview rendering for the GUI, without any Qt dependency.

The preview must use the same renderer as the export - a preview that sounds
different from the file it is previewing is worse than no preview - so this
module goes through the same compatibility adapter that `sound_creator` calls,
and only adds what playback needs: a duration cap, a safety limiter, and the
conversion to the interleaved 16-bit PCM the existing QtMultimedia path plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .compat import render_sam2_voice
from .conventions import (
    AUDIO_DTYPE,
    DEFAULT_LIMITER_CEILING_DBFS,
    DEFAULT_SAMPLE_RATE_HZ,
    seconds_to_samples,
)
from .dsp.envelopes import apply_fade_in, apply_fade_out, milliseconds_to_frames
from .dsp.limiter import safety_clamp

__all__ = [
    "MAX_PREVIEW_SECONDS",
    "PREVIEW_FADE_MS",
    "PreviewResult",
    "preview_frames",
    "render_voice_preview",
    "to_pcm16_bytes",
]

#: A preview is an audition, not a render; longer requests are truncated.
MAX_PREVIEW_SECONDS = 60.0
#: Fade applied at both ends so starting and stopping cannot click.
PREVIEW_FADE_MS = 15.0


@dataclass(frozen=True)
class PreviewResult:
    """A rendered preview and what the renderer did to it."""

    audio: NDArray[np.float32]  # (frames, 2), frame-major for playback
    sample_rate_hz: int
    duration_s: float
    peak: float
    limited: bool
    truncated: bool

    @property
    def frames(self) -> int:
        return int(self.audio.shape[0])


def render_voice_preview(
    voice_data: Mapping[str, Any],
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    duration_s: float = 5.0,
    start_time_s: float = 0.0,
    ceiling_dbfs: float = DEFAULT_LIMITER_CEILING_DBFS,
    fade_ms: float = PREVIEW_FADE_MS,
) -> PreviewResult:
    """Render a BinauralBuilder voice dictionary for audition.

    ``voice_data`` is the familiar ``{"synth_function_name", "is_transition",
    "params"}`` mapping. ``start_time_s`` previews a later part of the step: it
    becomes the absolute offset the renderer starts from, exactly as a chunked
    render would.
    """

    requested = max(0.0, float(duration_s))
    truncated = requested > MAX_PREVIEW_SECONDS
    span = min(requested, MAX_PREVIEW_SECONDS)

    params = dict(voice_data.get("params", {}) or {})
    is_transition = bool(voice_data.get("is_transition", False))
    transition_duration = params.get("duration") if is_transition else None
    if is_transition:
        # A transition voice stores its span under `duration`; the runtime
        # argument is `transition_duration`, and `initial_offset` is the
        # transition's start relative to the previewed chunk.
        initial_offset = float(params.get("initial_offset", 0.0) or 0.0) - float(start_time_s)
        if not transition_duration:
            transition_duration = span
    else:
        initial_offset = float(start_time_s)

    audio = render_sam2_voice(
        span,
        sample_rate_hz,
        params=params,
        is_transition=is_transition,
        initial_offset=initial_offset,
        transition_duration=float(transition_duration) if transition_duration else None,
    )

    channel_major = np.ascontiguousarray(audio.T, dtype=np.float64)
    fade_frames = milliseconds_to_frames(fade_ms, sample_rate_hz)
    if fade_frames and channel_major.shape[1] > 2 * fade_frames:
        apply_fade_in(channel_major, fade_frames)
        apply_fade_out(channel_major, fade_frames)

    clamped, report = safety_clamp(channel_major, ceiling_dbfs)
    frames = channel_major.shape[1]
    return PreviewResult(
        audio=np.ascontiguousarray(clamped.T, dtype=AUDIO_DTYPE),
        sample_rate_hz=int(sample_rate_hz),
        duration_s=frames / float(sample_rate_hz),
        peak=float(np.max(np.abs(clamped))) if frames else 0.0,
        limited=report.max_gain_reduction_db > 0.0,
        truncated=truncated,
    )


def to_pcm16_bytes(audio: NDArray[np.floating], gain: float = 1.0) -> bytes:
    """Convert frame-major float audio to interleaved little-endian 16-bit PCM.

    This is the format the existing QtMultimedia preview path already plays; no
    second audio backend is introduced for SAM.
    """

    block = np.asarray(audio, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != 2:
        raise ValueError(f"expected frame-major stereo audio, got shape {block.shape}")
    scaled = np.clip(block * float(gain), -1.0, 1.0) * 32767.0
    return np.ascontiguousarray(scaled.astype("<i2")).tobytes()


def preview_frames(duration_s: float, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> int:
    """Frames a preview of ``duration_s`` will produce, after the cap."""

    return seconds_to_samples(min(max(0.0, duration_s), MAX_PREVIEW_SECONDS), sample_rate_hz)

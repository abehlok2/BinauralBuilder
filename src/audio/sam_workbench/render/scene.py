"""Project-level rendering: sources to stems, stems to one stereo mix.

The scene applies the output-safety rules: one shared master gain, an optional
look-ahead limiter, and never a per-ear or per-source normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.conventions import CHANNEL_COUNT, seconds_to_samples
from src.audio.sam_workbench.dsp.limiter import LimiterReport, limit_lookahead, true_peak_estimate_db
from src.audio.sam_workbench.dsp.mixer import apply_shared_gain, mix_stems, peak_level
from src.audio.sam_workbench.dsp.source import (
    EAR_POLARITY_CANONICAL,
    CompiledSource,
    compile_source,
    render_source,
)
from src.audio.sam_workbench.validation import ValidationIssue
from src.audio.sam_workbench.render.abstract_pm import RENDERER_NAME, peak_frequency_deviation_hz

__all__ = ["SceneReport", "SourceRender", "render_project", "render_warnings"]


@dataclass(frozen=True)
class SourceRender:
    """One source's contribution, positioned on the project timeline."""

    source_id: str
    start_sample: int
    frames: int
    peak: float


@dataclass(frozen=True)
class SceneReport:
    """What a scene render produced, for the manifest and the GUI."""

    renderer: str
    frames: int
    sample_rate_hz: int
    block_size: int
    sources: tuple[SourceRender, ...]
    master_gain_db: float
    peak: float
    true_peak_dbfs: float
    limiter: LimiterReport | None
    warnings: tuple[ValidationIssue, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "renderer": self.renderer,
            "frames": self.frames,
            "sample_rate_hz": self.sample_rate_hz,
            "block_size": self.block_size,
            "master_gain_db": self.master_gain_db,
            "peak": self.peak,
            "true_peak_dbfs": self.true_peak_dbfs,
            "sources": [
                {
                    "id": entry.source_id,
                    "start_sample": entry.start_sample,
                    "frames": entry.frames,
                    "peak": entry.peak,
                }
                for entry in self.sources
            ],
            "limiter": None if self.limiter is None else vars(self.limiter),
            "warnings": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.warnings
            ],
        }


def render_warnings(project) -> tuple[ValidationIssue, ...]:
    """Render-time checks that are advisory rather than fatal.

    Peak instantaneous-frequency deviation ``|beta * f_m|`` is compared against
    Nyquist and against zero: phase modulation deep enough to push the carrier
    past either bound aliases or folds, and the user should be told before the
    render rather than after listening to it.
    """

    issues: list[ValidationIssue] = []
    nyquist = project.audio.sample_rate_hz / 2.0
    for position, source in enumerate(project.sources):
        if not source.enabled:
            continue
        compiled = compile_source(source)
        deviation = peak_frequency_deviation_hz(compiled)
        carrier = float(source.signal.carrier_frequency_hz)
        path = f"sources[{position}].sam.depth_rad"
        if carrier + deviation >= nyquist:
            issues.append(
                ValidationIssue(
                    path,
                    f"peak instantaneous frequency {carrier + deviation:.1f} Hz reaches Nyquist "
                    f"({nyquist:.1f} Hz); reduce depth or modulation rate",
                    severity="warning",
                )
            )
        elif carrier + deviation >= 0.8 * nyquist:
            issues.append(
                ValidationIssue(
                    path,
                    f"peak instantaneous frequency {carrier + deviation:.1f} Hz is within 20% of "
                    f"Nyquist ({nyquist:.1f} Hz)",
                    severity="warning",
                )
            )
        if carrier - deviation <= 0.0:
            issues.append(
                ValidationIssue(
                    path,
                    f"instantaneous frequency swings through 0 Hz (deviation {deviation:.1f} Hz "
                    f"against a {carrier:.1f} Hz carrier); the ear phases will fold",
                    severity="warning",
                )
            )
    return tuple(issues)


def _source_window(
    source, sample_rate: float, window_start: int, window_frames: int
) -> tuple[int, int, int, int] | None:
    """Intersect a source's own absolute interval with the requested window.

    Returns ``(source_start, placement, source_offset, frames)`` where
    ``placement`` is the offset into the window, ``source_offset`` is how far
    into the source's own timeline the window opens, and ``frames`` is the
    overlap - or ``None`` when the two do not overlap at all.

    Both ends have to be intersected, not just the near one. Clamping the
    length to the source's duration without first subtracting the part of the
    source that had already elapsed let a window opening late run past the
    source's end, and a window opening entirely after the source had finished
    rendered it from its own beginning as though it were only now starting.
    """

    source_start = seconds_to_samples(source.start_s, sample_rate)
    window_end = window_start + window_frames
    # A source with no declared duration runs to the end of any window.
    source_end = window_end
    if source.duration_s is not None:
        source_end = source_start + seconds_to_samples(source.duration_s, sample_rate)

    begin = max(source_start, window_start)
    end = min(source_end, window_end)
    if end <= begin:
        return None
    return (source_start, begin - window_start, begin - source_start, end - begin)


def render_project(
    project,
    duration_s: float | None = None,
    *,
    frames: int | None = None,
    block_size: int | None = None,
    start_sample: int = 0,
    apply_master_gain: bool = True,
    apply_limiter: bool | None = None,
    ear_polarity: str = EAR_POLARITY_CANONICAL,
) -> tuple[NDArray[np.float64], SceneReport]:
    """Render a validated project to ``(2, frames)`` float64 audio.

    Sources are rendered as independent stems from their own absolute sample
    positions and summed; the master gain and limiter are applied once, to both
    channels together.
    """

    project.validate()
    sample_rate = project.audio.sample_rate_hz

    if frames is None:
        if duration_s is None:
            raise ValueError("render_project needs either duration_s or frames")
        frames = seconds_to_samples(duration_s, sample_rate)
    frames = int(frames)
    if frames < 0:
        raise ValueError(f"frames must not be negative, got {frames}")
    span = int(block_size) if block_size else project.audio.offline_block_size

    stems: list[NDArray[np.float64]] = []
    rendered: list[SourceRender] = []
    for source in project.sources:
        if not source.enabled:
            continue
        window = _source_window(source, sample_rate, int(start_sample), frames)
        if window is None:
            continue
        source_start, placement, source_offset, source_frames = window

        compiled: CompiledSource = compile_source(source, ear_polarity=ear_polarity)
        audio = render_source(
            compiled,
            sample_rate,
            source_frames,
            span,
            start_sample=source_offset,
            dtype=np.float64,
        )
        stem = np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
        stem[:, placement : placement + source_frames] = audio
        stems.append(stem)
        rendered.append(
            SourceRender(
                source_id=source.id,
                start_sample=source_start,
                frames=source_frames,
                peak=peak_level(audio),
            )
        )

    mixed = mix_stems(tuple(stems), frames)
    if apply_master_gain:
        mixed = apply_shared_gain(mixed, project.output.master_gain_db)

    limiter_report: LimiterReport | None = None
    use_limiter = project.output.limiter_enabled if apply_limiter is None else apply_limiter
    if use_limiter and mixed.size:
        mixed, limiter_report = limit_lookahead(
            mixed, sample_rate, ceiling_dbfs=project.output.limiter_ceiling_dbfs
        )

    report = SceneReport(
        renderer=RENDERER_NAME,
        frames=frames,
        sample_rate_hz=sample_rate,
        block_size=span,
        sources=tuple(rendered),
        master_gain_db=float(project.output.master_gain_db) if apply_master_gain else 0.0,
        peak=peak_level(mixed),
        true_peak_dbfs=true_peak_estimate_db(mixed),
        limiter=limiter_report,
        warnings=render_warnings(project),
    )
    return mixed, report

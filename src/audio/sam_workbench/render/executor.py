"""Execute a :class:`~..plan.CompiledScenePlan` and produce audio.

The plan has been authoritative for compilation, validation, timing, assets,
controls, latency and diagnostics since Phase 1, and nothing rendered from it.
Production reached the renderers through ``compat.render_sam2_voice``, one
voice at a time, which meant the structure a project was validated against and
the structure it was rendered from were two separate readings of the same
document that only testing kept in agreement.

This module closes that gap. It walks the plan's sources, renders each one over
its own window, applies the compiled gain, and mixes through the same
:class:`~.scene_mix.SceneMixer` the scene path already used.

It deliberately does not contain a renderer. Every source is rendered by
``compat.render_voice_channels`` - the same dispatch the per-voice entry point
calls - so the executor cannot drift from the compatibility path it is
replacing. A second implementation would have to be proved equal; one shared
implementation is equal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT
from ..validation import ValidationIssue

__all__ = ["ExecutedScene", "SourceRenderReport", "execute_plan", "render_source_window"]


@dataclass(frozen=True)
class SourceRenderReport:
    """What one source contributed, positioned on the absolute timeline."""

    source_id: str
    renderer_id: str
    start_sample: int
    frames: int
    peak: float

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "renderer": self.renderer_id,
            "startSample": int(self.start_sample),
            "frames": int(self.frames),
            "peak": float(self.peak),
        }


@dataclass(frozen=True)
class ExecutedScene:
    """The mixed result, the stems behind it, and what happened on the way."""

    audio: NDArray[np.float64]
    stems: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    sources: tuple[SourceRenderReport, ...] = ()
    renderer_ids: tuple[str, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.audio.shape[-1])

    def describe(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "renderers": list(self.renderer_ids),
            "sources": [entry.describe() for entry in self.sources],
            "diagnostics": dict(self.diagnostics),
            "warnings": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.warnings
            ],
        }


def _effective_params(source, start_sample: int, frames: int) -> dict[str, Any]:
    """This source's parameters over a window, with its controls resolved.

    A control that never varies is folded in as its constant. One that does
    vary is resolved at the window's own start rather than at the caller's
    block boundary, so what a source is rendered with does not depend on how
    the timeline was cut - the same rule the compiled controls exist to keep.
    """

    params = dict(source.generator)
    for path, control in (source.controls or {}).items():
        # Only the leaf name reaches a voice's parameter dictionary; a dotted
        # path addresses somewhere the per-voice adapter does not look.
        if "." in path:
            continue
        try:
            params[path] = (
                float(control.constant)
                if control.is_constant and control.constant is not None
                else control.value_at(int(start_sample))
            )
        except Exception:  # noqa: BLE001 - a bad control must not lose the render
            continue
    return params


def render_source_window(
    source,
    *,
    sample_rate_hz: float,
    start_sample: int,
    frames: int,
    block_size: int | None = None,
    dataset_override: object | None = None,
) -> NDArray[np.float64]:
    """Render one planned source over an absolute window, gain applied.

    Returns channel-major ``(2, frames)`` positioned within the window: silent
    where the source is not sounding, so a caller can sum stems without
    tracking offsets.
    """

    from ..compat import render_voice_channels

    stem = np.zeros((CHANNEL_COUNT, int(frames)), dtype=np.float64)
    window = source.window(int(start_sample), int(frames))
    if window is None:
        return stem
    placement, source_offset, span = window
    if span <= 0:
        return stem

    params = _effective_params(source, int(start_sample) + placement, span)
    # The plan's bound trajectory is authoritative for modulated path
    # parameters. Renderers re-read ``canonicalTrajectory`` from the parameter
    # dictionary, so hand them the bound positions under the one runtime key
    # they already prefer - otherwise a compiled plan would silently render a
    # frozen path.
    if source.trajectory is not None:
        from ..path_automation import RUNTIME_BOUND_POSITIONS

        positions = getattr(source.trajectory, "positions", None)
        if callable(positions):
            params[RUNTIME_BOUND_POSITIONS] = positions
    audio = render_voice_channels(
        params,
        span,
        float(sample_rate_hz),
        # The source's own elapsed time, not the window's, is what the renderer
        # continues from. Passing the window's start would restart a moving
        # path every time a caller asked for a later block.
        initial_offset=float(source_offset) / float(sample_rate_hz),
        duration=float(span) / float(sample_rate_hz),
        block_size=block_size,
        dataset_override=dataset_override,
    )
    audio = np.asarray(audio, dtype=np.float64)
    if audio.shape[-1] < span:
        padded = np.zeros((CHANNEL_COUNT, span), dtype=np.float64)
        padded[:, : audio.shape[-1]] = audio[:, : audio.shape[-1]]
        audio = padded

    gain = source.gain.at(int(start_sample) + placement, span)
    stem[:, placement : placement + span] = audio[:, :span] * gain[None, :]
    return stem


def execute_plan(
    plan,
    *,
    start_sample: int | None = None,
    frames: int | None = None,
    block_size: int | None = None,
    apply_routing: bool = True,
) -> ExecutedScene:
    """Render the window a plan describes, or the window asked for.

    ``apply_routing`` off renders and returns the stems without buses, mute or
    solo, which is what an inspector wants when it is asking what one source
    sounds like rather than what the mix does with it.
    """

    from .scene_mix import mixer_from_plan

    window_start = plan.start_sample if start_sample is None else int(start_sample)
    span = plan.frames if frames is None else int(frames)
    span = max(0, int(span))
    rate = float(plan.sample_rate_hz)

    stems: dict[str, NDArray[np.float64]] = {}
    reports: list[SourceRenderReport] = []
    renderers: list[str] = []
    failures: list[ValidationIssue] = []

    for source in plan.active_sources(window_start, span):
        try:
            stem = render_source_window(
                source,
                sample_rate_hz=rate,
                start_sample=window_start,
                frames=span,
                block_size=block_size,
            )
        except Exception as error:  # noqa: BLE001 - one bad source is not the scene
            # Reported rather than raised: a scene with six sources and one
            # broken asset should render the five that work and say which one
            # did not, instead of producing nothing.
            failures.append(
                ValidationIssue(
                    f"sources[{source.source_id}]",
                    f"could not be rendered: {error}",
                    "error",
                )
            )
            continue
        stems[source.source_id] = stem
        if source.renderer_id not in renderers:
            renderers.append(source.renderer_id)
        window = source.window(window_start, span)
        reports.append(
            SourceRenderReport(
                source_id=source.source_id,
                renderer_id=source.renderer_id,
                start_sample=source.start_sample,
                frames=0 if window is None else int(window[2]),
                peak=float(np.max(np.abs(stem))) if stem.size else 0.0,
            )
        )

    diagnostics: dict[str, Any] = {
        "sourcesRendered": len(stems),
        "sourcesFailed": len(failures),
    }

    if not apply_routing:
        mixed = np.zeros((CHANNEL_COUNT, span), dtype=np.float64)
        for stem in stems.values():
            mixed += stem
    else:
        mixer = mixer_from_plan(plan, sample_rate_hz=rate)
        routed = mixer.process(stems, frames=span)
        mixed = np.asarray(routed.master, dtype=np.float64)
        diagnostics.update(mixer.diagnostics())

    return ExecutedScene(
        audio=mixed,
        stems=stems,
        sources=tuple(reports),
        renderer_ids=tuple(renderers),
        warnings=tuple(plan.warnings) + tuple(failures),
        diagnostics=diagnostics,
    )

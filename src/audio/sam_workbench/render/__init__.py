"""Renderers built on the DSP primitives.

Phase 1 provides the abstract phase-modulation engine and the scene mixer. The
geometric, HRTF, and hybrid engines arrive in later phases behind the same
:class:`~src.audio.sam_workbench.render.base.SpatialRenderer` contract.
"""

from __future__ import annotations

from .abstract_pm import PRESET_BUILDERS, RENDERER_NAME, AbstractPMRenderer, peak_frequency_deviation_hz
from .base import RenderBlock, RenderContext, RenderDiagnostics, SpatialRenderer
from .geometric import (
    DISTANCE_LAWS,
    GeometricBinauralRenderer,
    GeometricBinauralSpec,
    ear_distances_m,
    render_geometric,
)
from .scene import SceneReport, SourceRender, render_project, render_warnings

__all__ = [
    "PRESET_BUILDERS",
    "RENDERER_NAME",
    "DISTANCE_LAWS",
    "AbstractPMRenderer",
    "GeometricBinauralRenderer",
    "GeometricBinauralSpec",
    "RenderBlock",
    "RenderContext",
    "RenderDiagnostics",
    "SceneReport",
    "SourceRender",
    "SpatialRenderer",
    "ear_distances_m",
    "peak_frequency_deviation_hz",
    "render_geometric",
    "render_project",
    "render_warnings",
]

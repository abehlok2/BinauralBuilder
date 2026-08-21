"""Renderers built on the DSP primitives.

Phase 1 provides the abstract phase-modulation engine and the scene mixer. The
geometric, HRTF, and hybrid engines arrive in later phases behind the same
:class:`~src.audio.sam_workbench.render.base.SpatialRenderer` contract.
"""

from __future__ import annotations

from src.audio.sam_workbench.render.abstract_pm import PRESET_BUILDERS, RENDERER_NAME, AbstractPMRenderer, peak_frequency_deviation_hz
from src.audio.sam_workbench.render.base import RenderBlock, RenderContext, RenderDiagnostics, SpatialRenderer
from src.audio.sam_workbench.render.scene import SceneReport, SourceRender, render_project, render_warnings

__all__ = [
    "PRESET_BUILDERS",
    "RENDERER_NAME",
    "AbstractPMRenderer",
    "RenderBlock",
    "RenderContext",
    "RenderDiagnostics",
    "SceneReport",
    "SourceRender",
    "SpatialRenderer",
    "peak_frequency_deviation_hz",
    "render_project",
    "render_warnings",
]
from .geometric import (
    GeometricBinauralRenderer,
    GeometricSpec,
    cubic_fractional_sample,
    distance_gains,
    geometric_cues,
)
from .hrtf import HRTFRenderer, HRTFRendererSpec, render_hrtf

__all__ += [
    "GeometricBinauralRenderer", "GeometricSpec", "cubic_fractional_sample",
    "distance_gains", "geometric_cues", "HRTFRenderer", "HRTFRendererSpec", "render_hrtf",
]

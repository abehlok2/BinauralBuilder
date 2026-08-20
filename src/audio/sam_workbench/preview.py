"""Real-time-safe preview state exchange and block rendering."""
from __future__ import annotations
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Protocol
import numpy as np
import numpy.typing as npt
from .dsp import ExactSamRenderer
from .model import Project, validate_project

class StereoRenderer(Protocol):
    latency_frames: int
    def process(self, frames: int) -> npt.NDArray[np.float32]: ...

class ProjectPreviewRenderer:
    """Immutable-project mixer made from the same DSP used by export."""
    latency_frames = 0
    def __init__(self, project: Project):
        issues = validate_project(project)
        if issues:
            raise ValueError("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
        if any(source.spatializer.type != "abstract" for source in project.sources):
            raise NotImplementedError("phase two preview supports abstract sources only")
        self._renderers = tuple(ExactSamRenderer(s, project.audio.sample_rate_hz) for s in project.sources)
        self._master_gain = np.float32(10.0 ** (project.output.master_gain_db / 20.0))
    def process(self, frames: int) -> npt.NDArray[np.float32]:
        output = np.zeros((2, frames), dtype=np.float32)
        for renderer in self._renderers:
            output += renderer.process(frames)
        output *= self._master_gain
        return output

@dataclass(frozen=True)
class PreviewTelemetry:
    rendered_blocks: int
    renderer_swaps: int
    peak_linear: float

class PreviewBoundary:
    """Bounded, non-blocking handoff between control and callback threads."""
    def __init__(self, renderer: StereoRenderer | None = None):
        self._renderer = renderer
        self._pending: Queue[StereoRenderer] = Queue(maxsize=1)
        self._rendered_blocks = self._renderer_swaps = 0
        self._peak_linear = 0.0
    def submit(self, renderer: StereoRenderer) -> None:
        try:
            self._pending.put_nowait(renderer)
            return
        except Full:
            pass
        try:
            self._pending.get_nowait()
        except Empty:
            pass
        self._pending.put_nowait(renderer)
    def process_into(self, output: npt.NDArray[np.float32]) -> None:
        if output.ndim != 2 or output.shape[0] != 2 or output.dtype != np.float32:
            raise ValueError("output must be a float32 array shaped (2, frames)")
        try:
            self._renderer = self._pending.get_nowait()
            self._renderer_swaps += 1
        except Empty:
            pass
        if self._renderer is None:
            output.fill(0.0)
        else:
            rendered = self._renderer.process(output.shape[1])
            if rendered.shape != output.shape:
                raise ValueError("renderer returned an unexpected buffer shape")
            np.copyto(output, rendered)
        self._peak_linear = float(np.max(np.abs(output), initial=0.0))
        self._rendered_blocks += 1
    @property
    def telemetry(self) -> PreviewTelemetry:
        return PreviewTelemetry(self._rendered_blocks, self._renderer_swaps, self._peak_linear)

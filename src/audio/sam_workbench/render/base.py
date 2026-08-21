"""The renderer contract shared by every spatializer.

Every engine returns stereo in ``(channels, frames)`` order, declares its
latency, and preserves state across blocks. Later phases add the geometric,
HRTF, and hybrid engines behind this same protocol so they can be compared
honestly against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.dsp.blocks import RenderBlock, RenderContext

__all__ = ["RenderBlock", "RenderContext", "RenderDiagnostics", "SpatialRenderer"]


@dataclass(frozen=True)
class RenderDiagnostics:
    """Numbers a renderer can report about what it just did."""

    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


@runtime_checkable
class SpatialRenderer(Protocol):
    """Common renderer contract.

    ``process`` may be called for consecutive blocks of any size; a renderer
    that cannot be reconstructed from the absolute sample index alone keeps its
    own state and must not reset it at a block boundary.
    """

    def prepare(self, context: RenderContext, source: Any) -> None: ...

    def reset(self, sample_index: int = 0) -> None: ...

    def process(self, mono: NDArray[np.floating] | None, block: RenderBlock) -> NDArray[np.floating]: ...

    def latency_samples(self) -> int: ...

    def diagnostics(self) -> dict[str, Any]: ...

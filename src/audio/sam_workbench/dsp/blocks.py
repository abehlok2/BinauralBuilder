"""Block iteration and the render context.

Every render call carries an **absolute** ``start_sample``. Blocks are a
scheduling detail, never a source of state: rendering ``[0, N)`` in one block
and in many blocks must give the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..conventions import DEFAULT_OFFLINE_BLOCK_SIZE

__all__ = ["RenderBlock", "RenderContext", "iter_blocks"]


@dataclass(frozen=True)
class RenderBlock:
    """One block of a render, positioned on the absolute sample timeline."""

    start_sample: int
    frames: int

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.frames


@dataclass(frozen=True)
class RenderContext:
    """Everything a renderer needs that does not belong to one source."""

    sample_rate_hz: int
    block_size: int = DEFAULT_OFFLINE_BLOCK_SIZE
    speed_of_sound_m_s: float = 343.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


def iter_blocks(frames: int, block_size: int, start_sample: int = 0) -> Iterator[RenderBlock]:
    """Yield absolute-positioned blocks covering ``frames`` samples."""

    if frames < 0:
        raise ValueError(f"frames must not be negative, got {frames}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    position = 0
    while position < frames:
        span = min(block_size, frames - position)
        yield RenderBlock(start_sample=start_sample + position, frames=span)
        position += span

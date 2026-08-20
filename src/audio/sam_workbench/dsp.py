"""Stateful, block-continuous reference spatial-angle modulation DSP."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .conventions import wrap_phase_rad
from .model import Source


class ExactSamRenderer:
    """Render the permanent symmetric SAM reference condition.

    Carrier phase is accumulated per sample, so this implementation remains
    correct when frequency automation is introduced.  The modulator uses the
    absolute sample clock, making output independent of process block size.
    """

    latency_frames = 0

    def __init__(self, source: Source, sample_rate_hz: int):
        if source.spatializer.type != "abstract":
            raise ValueError("ExactSamRenderer requires an abstract spatializer")
        self.source = source
        self.sample_rate_hz = sample_rate_hz
        self.reset()

    def reset(self) -> None:
        self.sample_position = 0
        self.carrier_phase_rad = float(self.source.signal.phase_rad)

    def process(self, frames: int) -> npt.NDArray[np.float32]:
        if frames < 0:
            raise ValueError("frames must be non-negative")
        source = self.source
        increment = 2.0 * math.pi * source.signal.carrier_frequency_hz / self.sample_rate_hz
        carrier = self.carrier_phase_rad + increment * np.arange(frames, dtype=np.float64)
        absolute = self.sample_position + np.arange(frames, dtype=np.float64)
        modulation = source.sam.depth_rad * np.sin(
            source.sam.phase_rad + 2.0 * math.pi * source.sam.rate_hz * absolute / self.sample_rate_hz
        )
        gain = source.amplitude_linear if source.enabled else 0.0
        result = np.vstack((
            gain * np.sin(carrier + modulation + source.sam.left_bias_rad),
            gain * np.sin(carrier - modulation + source.sam.right_bias_rad),
        )).astype(np.float32)
        self.sample_position += frames
        self.carrier_phase_rad = float(wrap_phase_rad(self.carrier_phase_rad + increment * frames))
        return result


def render_source(source: Source, sample_rate_hz: int, frames: int, block_frames: int) -> npt.NDArray[np.float32]:
    """Offline render a source with the same stateful primitive used by preview."""
    if frames < 0 or block_frames <= 0:
        raise ValueError("frames must be non-negative and block_frames positive")
    renderer = ExactSamRenderer(source, sample_rate_hz)
    output = np.empty((2, frames), dtype=np.float32)
    for start in range(0, frames, block_frames):
        output[:, start : start + block_frames] = renderer.process(min(block_frames, frames - start))
    return output

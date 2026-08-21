"""The HRTF renderer.

The chain, per block:

.. code-block:: text

    mono source
      -> trajectory gives a direction for this block
      -> interpolator gives the filter pair for that direction
      -> crossfading convolution, one pair of states per ear
      -> external delay, when the delay policy kept it out of the responses
      -> optional distance gain

Direction is evaluated once per block, at control rate: a filter that changed
every sample would be a different algorithm, and the crossfade is what makes a
per-block change inaudible. Nothing here resets a convolution history when the
direction changes - both the outgoing and incoming filters keep their own
history through the fade, which is precisely what stops a moving source from
clicking.

This renderer never calls `slab` and never loads hidden demonstration data: the
dataset arrives already prepared from an explicitly chosen SOFA asset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, CHANNEL_LEFT
from ..dsp.blocks import RenderBlock, RenderContext, iter_blocks
from ..dsp.convolution import DEFAULT_FILTER_CROSSFADE_MS, CrossfadingConvolver
from ..dsp.delay import FractionalDelayLine
from ..hrtf.dataset import PreparedHrtf
from ..hrtf.interpolation import NEAREST, HrirSelection, HrtfInterpolator
from ..trajectory.spec import TrajectorySpec
from ..trajectory.transforms import ListenerFrame

__all__ = [
    "RENDERER_NAME",
    "HrtfRenderSpec",
    "HrtfRenderer",
    "render_hrtf",
]

RENDERER_NAME = "hrtf"

#: Distance below which the distance gain stops growing.
_MINIMUM_DISTANCE_M = 0.15


@dataclass(frozen=True)
class HrtfRenderSpec:
    """Everything the HRTF engine needs besides the source signal."""

    dataset: PreparedHrtf
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    listener: ListenerFrame = field(default_factory=ListenerFrame)
    interpolation: str = NEAREST
    crossfade_ms: float = DEFAULT_FILTER_CROSSFADE_MS
    neighbor_count: int = 3
    harmonic_order: int = 4
    #: Apply the dataset's external delay when the policy preserved it.
    apply_external_delay: bool = True
    #: Scale level with distance. Off by default: an HRTF encodes direction,
    #: and a measured set is already normalized to its measurement radius.
    apply_distance_gain: bool = False
    reference_distance_m: float = 1.0

    def interpolator(self) -> HrtfInterpolator:
        return HrtfInterpolator(
            self.dataset,
            mode=self.interpolation,
            neighbor_count=self.neighbor_count,
            harmonic_order=self.harmonic_order,
        )


class HrtfRenderer:
    """Convolve a mono source with direction-tracking HRTF filters."""

    def __init__(self, spec: HrtfRenderSpec | None = None) -> None:
        self._spec = spec
        self._context: RenderContext | None = None
        self._interpolator: HrtfInterpolator | None = None
        self._convolvers: list[CrossfadingConvolver] = []
        self._delay_lines: list[FractionalDelayLine] = []
        self._selection: HrirSelection | None = None
        self._switches = 0
        self._extrapolated_blocks = 0
        self._blocks = 0

    # --- SpatialRenderer ----------------------------------------------------

    def prepare(self, context: RenderContext, source: HrtfRenderSpec | None = None) -> None:
        if source is not None:
            self._spec = source
        if self._spec is None:
            raise ValueError("an HRTF renderer needs a specification")

        self._context = context
        self._interpolator = self._spec.interpolator()
        self._convolvers = [
            CrossfadingConvolver(
                crossfade_ms=self._spec.crossfade_ms, sample_rate_hz=context.sample_rate_hz
            )
            for _ in range(CHANNEL_COUNT)
        ]
        maximum_delay = int(max(16, np.ceil(np.max(np.abs(self._spec.dataset.delays_samples)) + 8)))
        self._delay_lines = [
            FractionalDelayLine(max_delay_samples=maximum_delay) for _ in range(CHANNEL_COUNT)
        ]
        self._selection = None
        self._switches = 0
        self._extrapolated_blocks = 0
        self._blocks = 0

    def reset(self, sample_index: int = 0) -> None:
        for convolver in self._convolvers:
            convolver.reset()
        for line in self._delay_lines:
            line.reset()
        self._selection = None

    def process(self, mono: NDArray[np.floating], block: RenderBlock) -> NDArray[np.float64]:
        """Render one block of a mono source into stereo."""

        if self._context is None or self._interpolator is None or self._spec is None:
            raise RuntimeError("prepare() must be called before process()")
        samples = np.asarray(mono, dtype=np.float64).reshape(-1)
        if samples.size != block.frames:
            raise ValueError(f"expected {block.frames} mono samples, got {samples.size}")
        if block.frames == 0:
            return np.zeros((CHANNEL_COUNT, 0), dtype=np.float64)

        sample_rate = self._context.sample_rate_hz
        trajectory = self._spec.trajectory.positions_for_block(
            block.start_sample, 1, sample_rate, listener=self._spec.listener
        )
        position = trajectory.positions_m[:, 0]

        selection = self._interpolator.at(position)
        for ear in range(CHANNEL_COUNT):
            if self._convolvers[ear].set_filter(selection.hrirs[ear]) and ear == CHANNEL_LEFT:
                self._switches += 1
        self._selection = selection
        self._blocks += 1
        if selection.extrapolated:
            self._extrapolated_blocks += 1

        output = np.empty((CHANNEL_COUNT, block.frames), dtype=np.float64)
        for ear in range(CHANNEL_COUNT):
            signal = self._convolvers[ear].process(samples)
            if self._spec.apply_external_delay and self._spec.dataset.has_external_delay:
                signal = self._delay_lines[ear].process(
                    signal, np.full(block.frames, float(selection.delays_samples[ear]))
                )
            output[ear] = signal

        if self._spec.apply_distance_gain:
            distance = max(float(np.linalg.norm(position)), _MINIMUM_DISTANCE_M)
            output *= float(self._spec.reference_distance_m) / distance
        return output

    def latency_samples(self) -> int:
        """Filter latency: where the responses put their energy.

        Reported as the median onset of the dataset, so a scene can align this
        stem against renderers that add none.
        """

        if self._spec is None:
            return 0
        hrirs = self._spec.dataset.hrirs
        peaks = np.argmax(np.abs(hrirs), axis=2)
        return int(np.median(peaks))

    def diagnostics(self) -> dict[str, Any]:
        if self._spec is None or self._selection is None:
            return {}

        dataset = self._spec.dataset
        indices = self._selection.indices
        azimuth = [float(dataset.azimuths_deg[index]) for index in indices]
        elevation = [float(dataset.elevations_deg[index]) for index in indices]
        return {
            "renderer": RENDERER_NAME,
            "interpolation": self._selection.mode,
            "asset": dataset.asset.path,
            "asset_sha256": dataset.asset.sha256,
            "subject": dataset.asset.subject,
            "delay_policy": dataset.delay_policy,
            "measurement_indices": [int(index) for index in indices],
            "measurement_azimuth_deg": azimuth,
            "measurement_elevation_deg": elevation,
            "direction_error_deg": float(self._selection.distance_deg),
            "filter_switches": int(self._switches),
            "blocks": int(self._blocks),
            "extrapolated_blocks": int(self._extrapolated_blocks),
            "crossfade_ms": float(self._spec.crossfade_ms),
            "latency_samples": self.latency_samples(),
            "channels": CHANNEL_COUNT,
        }

    @property
    def selection(self) -> HrirSelection | None:
        """The filter pair chosen for the most recent block."""

        return self._selection


def render_hrtf(
    mono: NDArray[np.floating],
    spec: HrtfRenderSpec,
    sample_rate: float,
    *,
    block_size: int = 512,
    start_sample: int = 0,
) -> NDArray[np.float64]:
    """Render a whole mono signal through the HRTF engine.

    ``block_size`` is a real parameter here, not just a scheduling detail: the
    direction is evaluated once per block, so a larger block tracks motion more
    coarsely. The convolution and its crossfades are unaffected by it.
    """

    samples = np.asarray(mono, dtype=np.float64).reshape(-1)
    context = RenderContext(sample_rate_hz=int(sample_rate), block_size=int(block_size))
    renderer = HrtfRenderer(spec)
    renderer.prepare(context, spec)

    output = np.zeros((CHANNEL_COUNT, samples.size), dtype=np.float64)
    for block in iter_blocks(samples.size, max(1, int(block_size)), start_sample=int(start_sample)):
        offset = block.start_sample - int(start_sample)
        output[:, offset : offset + block.frames] = renderer.process(
            samples[offset : offset + block.frames], block
        )
    return output

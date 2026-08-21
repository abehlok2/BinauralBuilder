"""The geometric binaural renderer.

For a source at ``p_s(t)`` and an ear at ``p_e``:

.. code-block:: text

    r_e(t) = |p_s(t) - p_e|
    tau_e(t) = r_e(t) / c
    y_e(t) = G_e(t) * x(t - tau_e(t))

Each ear reads the same mono source through its own fractional delay line, so
the interaural time difference is a *consequence* of the geometry rather than a
parameter. When the delay is allowed to vary with time, Doppler shift falls out
of the same mechanism; switching it off freezes the delay and keeps the level,
which is the usual creative request.

Declared discontinuities are crossfaded, not smoothed: a jump is meant to be
heard as a jump, just not as a click.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, CHANNEL_LEFT, CHANNEL_RIGHT, linear_to_db
from ..dsp.blocks import RenderBlock, RenderContext
from ..dsp.delay import FractionalDelayLine
from ..dsp.envelopes import equal_power_crossfade, milliseconds_to_frames
from ..dsp.filters import OnePoleLowpass, head_shadow_cutoff_hz
from ..trajectory.spec import TrajectorySpec
from ..trajectory.transforms import ListenerFrame
from ..trajectory.traversal import DEFAULT_JUMP_CROSSFADE_MS

__all__ = [
    "DISTANCE_LAWS",
    "GeometricBinauralRenderer",
    "GeometricBinauralSpec",
    "RENDERER_NAME",
    "ear_distances_m",
    "render_geometric",
]

RENDERER_NAME = "geometric_binaural"

#: How level falls with distance. ``inverse`` is the physical free-field law.
DISTANCE_LAWS = ("inverse", "inverse_square", "none", "exponent")


def ear_distances_m(
    positions_m: NDArray[np.floating], listener: ListenerFrame
) -> NDArray[np.float64]:
    """Distance from each ear to the source, as ``(2, frames)`` metres."""

    source = np.asarray(positions_m, dtype=np.float64)
    ears = listener.ear_positions_m
    return np.vstack(
        [np.linalg.norm(source - ears[:, ear][:, None], axis=0) for ear in range(CHANNEL_COUNT)]
    )


@dataclass(frozen=True)
class GeometricBinauralSpec:
    """Everything the geometric engine needs beyond the source signal."""

    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    listener: ListenerFrame = field(default_factory=ListenerFrame)
    speed_of_sound_m_s: float = 343.0
    distance_law: str = "inverse"
    distance_exponent: float = 1.0
    reference_distance_m: float = 1.0
    #: Level is held constant inside this radius, so a source passing through
    #: the listener's head cannot produce an unbounded gain.
    minimum_distance_m: float = 0.15
    doppler: bool = True
    head_shadow: bool = False
    jump_crossfade_ms: float = DEFAULT_JUMP_CROSSFADE_MS
    maximum_delay_m: float = 30.0

    def __post_init__(self) -> None:
        if self.distance_law not in DISTANCE_LAWS:
            raise ValueError(f"unknown distance law {self.distance_law!r}; expected one of {DISTANCE_LAWS}")
        if self.speed_of_sound_m_s <= 0.0:
            raise ValueError("speed_of_sound_m_s must be positive")
        if self.minimum_distance_m <= 0.0:
            raise ValueError("minimum_distance_m must be positive")

    def gains(self, distances_m: NDArray[np.floating]) -> NDArray[np.float64]:
        """Level for each ear from its distance, under the configured law."""

        distance = np.maximum(np.asarray(distances_m, dtype=np.float64), self.minimum_distance_m)
        if self.distance_law == "none":
            return np.ones_like(distance)
        exponent = {"inverse": 1.0, "inverse_square": 2.0}.get(
            self.distance_law, float(self.distance_exponent)
        )
        return (float(self.reference_distance_m) / distance) ** exponent

    def delays_samples(
        self, distances_m: NDArray[np.floating], sample_rate: float
    ) -> NDArray[np.float64]:
        """Propagation delay per ear in samples."""

        return np.asarray(distances_m, dtype=np.float64) / float(self.speed_of_sound_m_s) * float(
            sample_rate
        )

    def split_delays(
        self, distances_m: NDArray[np.floating], sample_rate: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Split the per-ear delay into a common part and a differential part.

        The **common** delay is the distance to the head, and carries almost
        all of the change when a source approaches - so it is what produces
        Doppler. The **differential** delay is the interaural time difference,
        bounded by the head's size and therefore tiny.

        Separating them is what lets Doppler be switched off without losing the
        spatial cue: freeze the common part, keep the differential part moving.
        """

        delays = self.delays_samples(distances_m, sample_rate)
        common = np.mean(delays, axis=0, keepdims=True)
        return common, delays - common


class GeometricBinauralRenderer:
    """Render a moving source to stereo through per-ear delay and level."""

    def __init__(self, spec: GeometricBinauralSpec | None = None) -> None:
        self._spec = spec or GeometricBinauralSpec()
        self._context: RenderContext | None = None
        self._lines: list[FractionalDelayLine] = []
        self._held_lines: list[FractionalDelayLine] = []
        self._filters: list[OnePoleLowpass] = []
        self._last_distances: NDArray[np.float64] | None = None
        self._last_positions: NDArray[np.float64] | None = None
        self._held_delay_samples: NDArray[np.float64] | None = None
        self._reference_delay: float | None = None

    # --- SpatialRenderer ---------------------------------------------------

    def prepare(self, context: RenderContext, source: GeometricBinauralSpec | None = None) -> None:
        self._context = context
        if source is not None:
            self._spec = source
        maximum = int(
            np.ceil(self._spec.maximum_delay_m / self._spec.speed_of_sound_m_s * context.sample_rate_hz)
        )
        self._lines = [FractionalDelayLine(max_delay_samples=maximum) for _ in range(CHANNEL_COUNT)]
        self._held_lines = [FractionalDelayLine(max_delay_samples=maximum) for _ in range(CHANNEL_COUNT)]
        self._filters = [
            OnePoleLowpass(sample_rate_hz=context.sample_rate_hz) for _ in range(CHANNEL_COUNT)
        ]
        self._held_delay_samples = None
        self._reference_delay = None

    def reset(self, sample_index: int = 0) -> None:
        for line in (*self._lines, *self._held_lines):
            line.reset()
        for filter_ in self._filters:
            filter_.reset()
        self._held_delay_samples = None

    def process(self, mono: NDArray[np.floating], block: RenderBlock) -> NDArray[np.float64]:
        """Render one block of a mono source into stereo."""

        if self._context is None or not self._lines:
            raise RuntimeError("prepare() must be called before process()")
        samples = np.asarray(mono, dtype=np.float64).reshape(-1)
        if samples.size != block.frames:
            raise ValueError(f"expected {block.frames} mono samples, got {samples.size}")
        if block.frames == 0:
            return np.zeros((CHANNEL_COUNT, 0), dtype=np.float64)

        sample_rate = self._context.sample_rate_hz
        trajectory = self._spec.trajectory.positions_for_block(
            block.start_sample, block.frames, sample_rate, listener=self._spec.listener
        )
        positions = trajectory.positions_m
        distances = ear_distances_m(positions, self._spec.listener)
        gains = self._spec.gains(distances)
        delays = self._spec.delays_samples(distances, sample_rate)

        if not self._spec.doppler:
            # Hold the distance-driven common delay at its value at absolute
            # sample zero and keep the interaural difference moving. The level
            # still follows the distance; the pitch does not shift, because the
            # only delay that still changes is bounded by the head's size.
            _, differential = self._spec.split_delays(distances, sample_rate)
            delays = self._reference_common_delay(sample_rate) + differential

        held_delays = self._held_delays(delays, trajectory.jumps)
        weights = self._crossfade_weights(trajectory.jumps, block.frames, trajectory.crossfade_ms, sample_rate)

        output = np.empty((CHANNEL_COUNT, block.frames), dtype=np.float64)
        for ear in (CHANNEL_LEFT, CHANNEL_RIGHT):
            moved = self._lines[ear].process(samples, delays[ear])
            if weights is None:
                signal = moved
            else:
                held = self._held_lines[ear].process(samples, held_delays[ear])
                fade_out, fade_in = weights
                signal = held * fade_out + moved * fade_in
            signal = signal * gains[ear]
            if self._spec.head_shadow:
                signal = self._filters[ear].process(signal, self._shadow_cutoff(positions, ear))
            output[ear] = signal

        self._last_distances = distances
        self._last_positions = positions
        self._held_delay_samples = delays[:, -1:]
        return output

    def latency_samples(self) -> int:
        """Propagation delay of the nearest configured position, in samples.

        The delay is physical rather than algorithmic, but it is reported so a
        scene can align this stem against renderers that add none.
        """

        if self._context is None or self._last_distances is None:
            return 0
        return int(round(float(np.min(self._last_distances)) / self._spec.speed_of_sound_m_s * self._context.sample_rate_hz))

    def diagnostics(self) -> dict[str, Any]:
        if self._last_distances is None or self._context is None:
            return {}
        distances = self._last_distances
        delays = self._spec.delays_samples(distances, self._context.sample_rate_hz)
        gains = self._spec.gains(distances)
        # Positive when the left ear leads, matching `analysis.predicted_itd_s`.
        itd = (delays[CHANNEL_RIGHT] - delays[CHANNEL_LEFT]) / float(self._context.sample_rate_hz)
        ild = linear_to_db(float(np.mean(gains[CHANNEL_LEFT]))) - linear_to_db(
            float(np.mean(gains[CHANNEL_RIGHT]))
        )
        return {
            "renderer": RENDERER_NAME,
            "distance_law": self._spec.distance_law,
            "doppler": bool(self._spec.doppler),
            "head_shadow": bool(self._spec.head_shadow),
            "distance_m_min": float(np.min(distances)),
            "distance_m_max": float(np.max(distances)),
            "itd_s_min": float(np.min(itd)),
            "itd_s_max": float(np.max(itd)),
            "ild_db": float(ild),
            "latency_samples": self.latency_samples(),
            "channels": CHANNEL_COUNT,
        }

    # --- helpers ------------------------------------------------------------

    def _reference_common_delay(self, sample_rate: float) -> NDArray[np.float64]:
        """The frozen common delay used when Doppler is switched off.

        Taken from absolute sample zero rather than from whichever block
        happened to be rendered first, so a chunked render matches a whole one.
        """

        if self._reference_delay is None:
            start = self._spec.trajectory.positions_for_block(0, 1, sample_rate, listener=self._spec.listener)
            distances = ear_distances_m(start.positions_m, self._spec.listener)
            common, _ = self._spec.split_delays(distances, sample_rate)
            self._reference_delay = float(common[0, 0])
        return np.full((CHANNEL_COUNT, 1), self._reference_delay, dtype=np.float64)

    def _held_delays(
        self, delays: NDArray[np.float64], jumps: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Delays as they would have been without the jump, for the crossfade."""

        held = delays.copy()
        indices = np.flatnonzero(jumps)
        if indices.size == 0:
            return held
        first = int(indices[0])
        if first == 0:
            previous = self._held_delay_samples
            if previous is None:
                return held
            held[:] = previous
        else:
            held[:, first:] = delays[:, first - 1 : first]
        return held

    def _crossfade_weights(
        self, jumps: NDArray[np.bool_], frames: int, crossfade_ms: float, sample_rate: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Equal-power weights that fade the old position into the new one."""

        indices = np.flatnonzero(jumps)
        if indices.size == 0:
            return None
        window = max(1, milliseconds_to_frames(crossfade_ms, sample_rate))
        fade_out = np.zeros(frames, dtype=np.float64)
        fade_in = np.ones(frames, dtype=np.float64)
        start = int(indices[0])
        stop = min(frames, start + window)
        out_curve, in_curve = equal_power_crossfade(stop - start)
        fade_out[:start] = 1.0
        fade_in[:start] = 0.0
        fade_out[start:stop] = out_curve
        fade_in[start:stop] = in_curve
        return fade_out, fade_in

    def _shadow_cutoff(self, positions: NDArray[np.float64], ear: int) -> float:
        """Mean incidence cosine for one ear over the block, as a cutoff."""

        ears = self._spec.listener.ear_positions_m
        direction = positions - ears[:, ear][:, None]
        norm = np.linalg.norm(direction, axis=0)
        norm[norm == 0.0] = 1.0
        outward = np.array([0.0, 1.0 if ear == CHANNEL_LEFT else -1.0, 0.0])
        cosine = float(np.mean((direction / norm).T @ outward))
        return float(head_shadow_cutoff_hz(cosine))


def render_geometric(
    mono: NDArray[np.floating],
    spec: GeometricBinauralSpec,
    sample_rate: float,
    *,
    block_size: int | None = None,
    start_sample: int = 0,
) -> NDArray[np.float64]:
    """Render a whole mono signal through the geometric engine.

    ``block_size`` only changes how the work is divided; the delay lines carry
    their history, so the samples are the same either way.
    """

    from ..dsp.blocks import iter_blocks

    samples = np.asarray(mono, dtype=np.float64).reshape(-1)
    context = RenderContext(sample_rate_hz=int(sample_rate), block_size=int(block_size or samples.size or 1))
    renderer = GeometricBinauralRenderer(spec)
    renderer.prepare(context, spec)

    output = np.zeros((CHANNEL_COUNT, samples.size), dtype=np.float64)
    span = int(block_size) if block_size else max(1, samples.size)
    for block in iter_blocks(samples.size, span, start_sample=int(start_sample)):
        offset = block.start_sample - int(start_sample)
        output[:, offset : offset + block.frames] = renderer.process(
            samples[offset : offset + block.frames], block
        )
    return output

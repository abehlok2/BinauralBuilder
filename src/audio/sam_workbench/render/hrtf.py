"""Time-varying explicit-SOFA binaural rendering."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike
from ..conventions import AUDIO_DTYPE
from ..dsp.binaural_convolution import BinauralConvolver
from ..hrtf import DelayPolicy, default_hrtf_cache
from ..trajectory.transforms import ListenerTransform

#: ``logmag_delay`` was this renderer's own name for what the interpolation
#: subsystem calls ``delay_magnitude``. Both are accepted; the canonical name
#: is the subsystem's, so a mode advertised in one place cannot be a mode the
#: other refuses.
INTERPOLATION_ALIASES: dict[str, str] = {"logmag_delay": "delay_magnitude"}


def canonical_interpolation(mode: str) -> str:
    """The subsystem's name for an interpolation mode."""

    return INTERPOLATION_ALIASES.get(str(mode), str(mode))


InterpolationMode = str

@dataclass(frozen=True)
class HRTFRendererSpec:
    """An explicit asset and every policy required for reproduction."""
    sofa_path: str | Path
    trajectory: object
    interpolation: InterpolationMode = "nearest"
    delay_policy: DelayPolicy | str = DelayPolicy.BAKE
    crossfade_ms: float = 10.0
    control_interval_samples: int = 128
    listener: ListenerTransform = ListenerTransform()
    expected_sha256: str | None = None
    project_directory: str | Path | None = None
    #: A dataset to render with instead of loading ``sofa_path``. The hybrid
    #: path passes its cue-modified dataset here, so the creative stage acts on
    #: the filters - where interaural time and level differences actually live -
    #: rather than on an already-mixed stereo signal that would have to be
    #: unpicked first.
    dataset_override: object | None = None

    # --- interpolation -----------------------------------------------------
    #: How many measurements the three-neighbour blend uses.
    neighbor_count: int = 3
    #: Spherical-harmonic order, or None for the highest the dataset supports.
    harmonic_order: int | None = None

    # --- spatial update ----------------------------------------------------
    #: Largest direction change tolerated before the filter is reselected. A
    #: fixed interval updates a slow path far more often than it can hear and a
    #: fast one less often than it needs; bounding the *error* instead adapts to
    #: the path. ``None`` keeps the fixed interval.
    max_angular_error_deg: float | None = 1.0
    #: Bounds on the adaptive interval, in samples.
    min_control_interval_samples: int = 128
    max_control_interval_samples: int = 4096

    # --- distance ----------------------------------------------------------
    #
    # A SOFA measurement encodes a direction. Distance is not in the filter, so
    # a renderer that only looks up a direction reproduces a path's azimuth and
    # elevation and silently discards how near or far it goes. These fields are
    # what make the third dimension of a trajectory audible.
    #
    # The defaults are neutral, so a caller that does not ask for distance
    # handling gets exactly the direction-only render it got before.
    #: ``"none"`` leaves level alone; the others attenuate with distance.
    distance_law: Literal["none", "inverse", "inverse_square"] = "none"
    #: Distance at which the distance law is unity gain.
    reference_distance_m: float = 1.0
    #: Distances are clamped into this range before the law is applied, so a
    #: path through the listener cannot produce unbounded gain.
    minimum_distance_m: float = 0.15
    maximum_distance_m: float = 100.0
    #: Delay the source by the time sound takes to travel the distance. This is
    #: what makes an approaching source arrive early and a receding one late.
    propagation_delay: bool = False
    speed_of_sound_m_s: float = 343.0

    def __post_init__(self):
        from ..hrtf.interpolation import INTERPOLATION_MODES

        if canonical_interpolation(self.interpolation) not in INTERPOLATION_MODES:
            raise ValueError(
                f"unsupported interpolation {self.interpolation!r}; "
                f"expected one of {INTERPOLATION_MODES} "
                f"(or the alias {tuple(INTERPOLATION_ALIASES)})"
            )
        if int(self.neighbor_count) < 1:
            raise ValueError("neighbor_count must be at least 1")
        if self.harmonic_order is not None and int(self.harmonic_order) < 0:
            raise ValueError("harmonic_order must not be negative")
        DelayPolicy(self.delay_policy)
        if not math.isfinite(self.crossfade_ms) or self.crossfade_ms < 0:
            raise ValueError("crossfade_ms must be finite and non-negative")
        if self.control_interval_samples <= 0:
            raise ValueError("control_interval_samples must be positive")
        if self.distance_law not in ("none", "inverse", "inverse_square"):
            raise ValueError(f"unsupported distance law {self.distance_law!r}")
        for name in ("reference_distance_m", "minimum_distance_m", "speed_of_sound_m_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_distance_m < self.minimum_distance_m:
            raise ValueError("maximum_distance_m must not be smaller than minimum_distance_m")

    def distance_gain(self, distance_m: float) -> float:
        """Gain for one distance, under the configured law."""

        if self.distance_law == "none":
            return 1.0
        clamped = min(max(float(distance_m), self.minimum_distance_m), self.maximum_distance_m)
        ratio = self.reference_distance_m / clamped
        return float(ratio * ratio if self.distance_law == "inverse_square" else ratio)

class HRTFRenderer:
    """The canonical binaural engine: block convolution, every interpolation mode.

    This used to convolve sample by sample in Python and offer two of the five
    interpolation modes the rest of the workbench advertises, while a second
    engine served the HRTF Lab with block convolution and all five. Two engines
    meant preview and export could differ from an audition of the same
    settings, and the production one rendered slower than real time.

    One engine now serves preview, export, the lab, A/B, analysis, experiments
    and benchmarks. It keeps everything the production path had - the
    three-dimensional trajectory, the distance law, propagation delay and
    coverage diagnostics - and gains what only the lab had.

    Direction is reselected on an adaptive schedule rather than a fixed one:
    what matters is how far the source has turned since the filter was chosen,
    not how many samples have elapsed, so a slow path costs almost nothing and
    a fast one is still tracked.
    """

    def __init__(self, spec: HRTFRendererSpec):
        self.spec = spec
        self.dataset = None
        self.interpolator = None
        self._sample_rate_hz = 44100
        self._diagnostics: dict = {}
        self._convolver: BinauralConvolver | None = None
        self.reset()

    # --- setup --------------------------------------------------------------

    def prepare(self, context, source=None):
        from ..hrtf.interpolation import HrtfInterpolator

        self._sample_rate_hz = int(context.sample_rate_hz)
        loaded = default_hrtf_cache.get(
            self.spec.sofa_path, self._sample_rate_hz, self.spec.delay_policy,
            self.spec.project_directory,
        )
        if self.spec.expected_sha256 and loaded.content_hash.lower() != self.spec.expected_sha256.lower():
            raise ValueError("SOFA asset hash does not match hrtfAssetHash")
        # The hash is always checked against the asset on disk, even when
        # rendering happens through a modified view of it: what a manifest has
        # to be able to state is which measurements the render came from.
        self.dataset = self.spec.dataset_override or loaded
        self._asset_hash = loaded.content_hash
        self.interpolator = HrtfInterpolator(
            self.dataset,
            mode=canonical_interpolation(self.spec.interpolation),
            neighbor_count=int(self.spec.neighbor_count),
            harmonic_order=self.spec.harmonic_order,
        )
        self._convolver = BinauralConvolver(
            sample_rate_hz=float(self._sample_rate_hz),
            crossfade_ms=float(self.spec.crossfade_ms),
        )
        self.reset(0)

    def reset(self, sample_index: int = 0):
        self._next_sample = int(sample_index)
        self._selection = None
        self._selection_direction = None
        self._selection_sample = None
        self._filter_changes = 0
        self._selections = 0
        self._extrapolated = 0
        if self._convolver is not None:
            self._convolver.reset()
        self._delay_line = np.zeros(self._delay_capacity(), dtype=np.float64)
        self._delay_write = 0
        self._gain = self._gain_step = 0.0
        self._delay_samples = self._delay_step = 0.0
        self._control_remaining = 0
        self._primed_distance = False

    def latency_samples(self):
        return 0

    def diagnostics(self):
        return dict(self._diagnostics)

    # --- distance -----------------------------------------------------------

    def _delay_capacity(self):
        if not self.spec.propagation_delay:
            return 4
        span = self.spec.maximum_distance_m / self.spec.speed_of_sound_m_s
        return int(np.ceil(span * self._sample_rate_hz)) + 8

    def _distance_curve(self, start: int, frames: int):
        """Per-sample gain and propagation delay over an absolute window.

        Evaluated on the same fixed grid the filter uses and interpolated
        between grid points, so both are functions of the absolute sample index
        alone. Ramping from wherever the previous *segment* happened to end
        would make the curve depend on how the caller cut the block, which is
        exactly what a listener must never be able to hear.
        """

        step = max(1, int(self.spec.min_control_interval_samples))
        first = (start // step) * step
        last = ((start + frames + step - 1) // step) * step
        knots = np.arange(first, last + step, step, dtype=np.int64)
        positions = self._positions(knots)
        distances = np.linalg.norm(positions, axis=1)
        gains = np.asarray(
            [self.spec.distance_gain(float(value)) for value in distances], dtype=np.float64
        )
        if self.spec.propagation_delay:
            clamped = np.clip(
                distances, self.spec.minimum_distance_m, self.spec.maximum_distance_m
            )
            delays = clamped / self.spec.speed_of_sound_m_s * self._sample_rate_hz
        else:
            delays = np.zeros(knots.size, dtype=np.float64)
        wanted = np.arange(start, start + frames, dtype=np.float64)
        return (
            np.interp(wanted, knots.astype(np.float64), gains),
            np.interp(wanted, knots.astype(np.float64), delays),
        )

    def _delayed(self, samples: NDArray[np.float64], delays: NDArray[np.float64]):
        """Fractional-delay read for a whole block at once.

        Cubic rather than linear because a linearly interpolated fractional
        delay low-passes by an amount that changes with the fractional part,
        which on a moving source is heard as the timbre wobbling in step with
        the motion. Vectorized because doing it a sample at a time in Python is
        the cost this engine exists to remove.
        """

        size = len(self._delay_line)
        frames = samples.size
        # Write the block into the ring, then read from it: every sample this
        # block needs is either in the ring already or written just now.
        write = (self._delay_write + np.arange(frames)) % size
        self._delay_line[write] = samples
        self._delay_write = int((self._delay_write + frames) % size)

        newest = self._delay_write - 1
        target = (newest - (frames - 1) + np.arange(frames)) - np.maximum(delays, 0.0)
        base = np.floor(target).astype(np.int64)
        fraction = target - base
        p0, p1, p2, p3 = (
            self._delay_line[(base + offset) % size] for offset in (-1, 0, 1, 2)
        )
        return p1 + 0.5 * fraction * (
            p2 - p0
            + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                          + fraction * (3.0 * (p1 - p2) + p3 - p0))
        )

    def _distance_targets(self, position):
        distance = float(np.linalg.norm(position))
        gain = self.spec.distance_gain(distance)
        delay = 0.0
        if self.spec.propagation_delay:
            clamped = min(max(distance, self.spec.minimum_distance_m), self.spec.maximum_distance_m)
            delay = clamped / self.spec.speed_of_sound_m_s * self._sample_rate_hz
        return gain, delay

    def _apply_distance(self, samples: NDArray[np.float64], start: int) -> NDArray[np.float64]:
        """Distance gain and propagation delay over one window."""

        if samples.size == 0:
            return samples
        gains, delays = self._distance_curve(start, samples.size)
        result = samples
        if self.spec.propagation_delay:
            result = self._delayed(samples, delays)
        if self.spec.distance_law != "none":
            result = result * gains
        self._gain = float(gains[-1])
        self._delay_samples = float(delays[-1])
        return result

    # --- direction ----------------------------------------------------------

    def _positions(self, samples):
        """Listener-relative positions at several absolute samples at once.

        Batched because a trajectory query costs about the same for one sample
        as for a hundred, and the renderer needs a whole block's worth of grid
        points before it convolves anything.
        """

        index = np.asarray(samples, dtype=np.float64)
        time = index / self._sample_rate_hz
        trajectory = self.spec.trajectory
        world = trajectory(time) if callable(trajectory) else trajectory.evaluate(time)
        values = np.asarray(world, dtype=np.float64)
        if values.shape != (index.size, 3) or not np.all(np.isfinite(values)):
            raise ValueError("trajectory must return finite (frames, 3) positions")
        return self.spec.listener.world_to_listener(values)

    def _position(self, sample):
        return self._positions([sample])[0]

    @staticmethod
    def _unit(position):
        norm = float(np.linalg.norm(position))
        return None if norm <= 0.0 else np.asarray(position, dtype=np.float64) / norm

    def _needs_new_filter(self, sample: int, direction) -> bool:
        """Whether the direction has moved far enough to be worth reselecting.

        A fixed interval is the wrong question. What a listener can hear is the
        angular error between where the source is and the direction whose
        filter is playing, so that is what is bounded. A path that barely moves
        keeps its filter for the maximum interval; one crossing the head
        quickly is reselected as often as the minimum allows.
        """

        if self._selection is None or direction is None:
            return True
        elapsed = sample - (self._selection_sample or 0)
        if elapsed >= self.spec.max_control_interval_samples:
            return True
        if self.spec.max_angular_error_deg is None:
            return elapsed >= self.spec.control_interval_samples
        if elapsed < self.spec.min_control_interval_samples:
            return False
        cosine = float(np.clip(np.dot(direction, self._selection_direction), -1.0, 1.0))
        return np.degrees(np.arccos(cosine)) >= float(self.spec.max_angular_error_deg)

    def _select(self, sample: int, direction) -> None:
        selection = self.interpolator.at(direction)
        self._selections += 1
        if selection.extrapolated:
            self._extrapolated += 1
        installed = self._convolver.set_filters(selection.hrirs)
        if installed:
            self._filter_changes += 1
        self._selection = selection
        self._selection_direction = direction
        self._selection_sample = sample

    def _grid_points(self, start: int, frames: int):
        """Absolute samples at which a filter change may occur.

        A fixed grid anchored to sample zero, so which points exist does not
        depend on where the caller cut the stream. Whether a change *happens*
        at one of them is the adaptive part.
        """

        step = max(1, int(self.spec.min_control_interval_samples))
        first = ((start + step - 1) // step) * step
        return range(max(first, start), start + frames, step)

    def _select_at(self, sample: int) -> bool:
        """Reselect the filter for ``sample`` if the direction has moved enough."""

        direction = self._unit(self._position(sample))
        if direction is None:
            # A source at the listener's own position has no direction; the
            # filter in force is kept rather than an arbitrary one chosen.
            direction = self._selection_direction
            if direction is None:
                direction = np.array([1.0, 0.0, 0.0])
        if not self._needs_new_filter(sample, direction):
            return False
        self._select(sample, direction)
        return True

    # --- rendering ----------------------------------------------------------

    def _render_segment(self, source, start: int, offset: int, span: int):
        """Convolve one stretch of a block under whatever filter is installed."""

        samples = source[offset : offset + span]
        return self._convolver.process(self._apply_distance(samples, start + offset))

    def process(self, mono: ArrayLike, block):
        if self.dataset is None:
            raise RuntimeError("prepare() must be called before process()")
        source = np.asarray(mono, dtype=np.float64)
        if source.shape != (block.frames,):
            raise ValueError(f"mono input must have shape ({block.frames},), got {source.shape}")
        if int(block.start_sample) != self._next_sample:
            self.reset(int(block.start_sample))
        if source.size == 0:
            return np.zeros((2, 0), dtype=AUDIO_DTYPE)

        start = int(block.start_sample)
        frames = int(block.frames)
        # Filter changes happen at absolute sample positions, never at block
        # boundaries: deciding once per block would make the update rate a
        # function of the caller's block size, so the same render would track a
        # moving source differently depending on how it was chunked.
        #
        # Selection and processing are interleaved rather than planned ahead.
        # Choosing every filter for the block first and only then convolving
        # would leave the last filter installed for the block's opening audio,
        # which sounds like the source jumping ahead of itself.
        if self._selection is None:
            self._select_at(start)

        pieces = []
        cursor = 0
        for point in self._grid_points(start, frames):
            offset = point - start
            if offset > cursor:
                # Audio up to this point belongs to the filter still in force.
                pieces.append(self._render_segment(source, start, cursor, offset - cursor))
                cursor = offset
            self._select_at(point)
        if cursor < frames:
            pieces.append(self._render_segment(source, start, cursor, frames - cursor))
        output = np.concatenate(pieces, axis=1) if pieces else np.zeros((2, 0))

        self._next_sample = start + frames
        selection = self._selection
        self._diagnostics = {
            "asset_sha256": self._asset_hash,
            "interpolation": canonical_interpolation(self.spec.interpolation),
            "delay_policy": DelayPolicy(self.spec.delay_policy).value,
            "distance_law": self.spec.distance_law,
            "distance_gain": float(self._gain),
            "propagation_delay_samples": float(self._delay_samples),
            "hrtf_indices": None if selection is None else selection.indices,
            "hrtf_weights": None if selection is None else selection.weights,
            "angular_distance_deg": 0.0 if selection is None else float(selection.distance_deg),
            "extrapolated": bool(selection.extrapolated) if selection else False,
            "selections": int(self._selections),
            "filter_changes": int(self._filter_changes),
            "extrapolated_selections": int(self._extrapolated),
            "taps": 0 if selection is None else int(selection.taps),
        }
        return output.astype(AUDIO_DTYPE, copy=False)

    @property
    def last_selection(self):
        """The most recent :class:`HrirSelection`, for the inspector."""

        return self._selection


def render_hrtf(mono, spec, sample_rate_hz, *, block_size=4096, start_sample=0):
    from ..dsp.blocks import RenderContext, iter_blocks
    source = np.asarray(mono, dtype=np.float64)
    renderer = HRTFRenderer(spec); renderer.prepare(RenderContext(sample_rate_hz, block_size))
    output = np.empty((2, len(source)), dtype=np.float32); position = 0
    for block in iter_blocks(len(source), block_size, start_sample):
        output[:, position:position+block.frames] = renderer.process(source[position:position+block.frames], block)
        position += block.frames
    return output


# ---------------------------------------------------------------------------
# Direction-indexed renderer
# ---------------------------------------------------------------------------
#
# The signal chain this implements, in order:
#
#     mono source -> trajectory and HRTF selection -> left/right HRIR
#     convolution -> headphone correction -> limiter and output gain
#
# The limiter and output gain are the caller's stage; everything up to and
# including headphone correction happens here.
#
# Direction is chosen once per block, not once per sample. When it changes, the
# outgoing filter keeps convolving alongside the incoming one for the length of
# the fade, so both convolution histories survive the switch and only their
# mixture changes. That is what keeps a moving source from clicking without
# ever averaging raw HRIR samples.

from dataclasses import dataclass as _dataclass

from ..dsp.convolution import DEFAULT_FILTER_CROSSFADE_MS
from ..hrtf.headphones import HeadphoneCorrection, apply_correction
from ..hrtf.interpolation import HrtfInterpolator, INTERPOLATION_MODES, NEAREST


@_dataclass(frozen=True)
class SpatialHrtfSpec:
    """Every policy the direction-indexed renderer needs."""

    interpolation: str = NEAREST
    crossfade_ms: float = DEFAULT_FILTER_CROSSFADE_MS
    #: How many measurements the three-neighbour blend uses.
    neighbor_count: int = 3
    #: Spherical-harmonic order, or None to use the highest the dataset
    #: supports. A fixed order cannot suit every dataset: too high for a sparse
    #: set leaves the fit underdetermined, too low for a dense one throws away
    #: most of what was measured.
    harmonic_order: int | None = None
    #: Applied after binaural rendering, never before it.
    headphone: HeadphoneCorrection | None = None

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATION_MODES:
            raise ValueError(
                f"unsupported interpolation {self.interpolation!r}; "
                f"expected one of {INTERPOLATION_MODES}"
            )
        if not np.isfinite(self.crossfade_ms) or self.crossfade_ms < 0:
            raise ValueError("crossfade_ms must be finite and non-negative")
        if int(self.neighbor_count) < 1:
            raise ValueError(
                f"neighbor_count must be at least 1, got {self.neighbor_count!r}"
            )
        if self.harmonic_order is not None and int(self.harmonic_order) < 0:
            raise ValueError(
                f"harmonic_order must not be negative, got {self.harmonic_order!r}"
            )


class SpatialHrtfRenderer:
    """Block-rate direction selection with per-ear crossfading convolution."""

    def __init__(self, dataset, sample_rate_hz: float, spec: SpatialHrtfSpec | None = None) -> None:
        self.spec = spec or SpatialHrtfSpec()
        self.dataset = dataset
        self.sample_rate_hz = float(sample_rate_hz)
        self.interpolator = HrtfInterpolator(
            dataset,
            mode=self.spec.interpolation,
            neighbor_count=self.spec.neighbor_count,
            harmonic_order=self.spec.harmonic_order,
        )
        # The same convolution core :class:`HRTFRenderer` uses. Two engines that
        # convolve differently is how an audition and an export of identical
        # settings came to differ; sharing this is what makes them agree by
        # construction rather than by testing.
        self._convolver = BinauralConvolver(
            sample_rate_hz=self.sample_rate_hz, crossfade_ms=self.spec.crossfade_ms
        )
        self._last_selection: object = None
        self.reset()

    def reset(self) -> None:
        self._convolver.reset()
        self._last_selection = None
        self._primed = False

    @property
    def last_selection(self):
        """The most recent :class:`HrirSelection`, for the inspector."""

        return self._last_selection

    def process_block(self, mono_block, direction):
        """Render one block from a single direction.

        Returns channel-major ``(2, frames)``. Headphone correction is applied
        by :func:`render_spatial_hrtf` over the whole render rather than per
        block, so a block-boundary does not truncate the correction filter.
        """

        samples = np.asarray(mono_block, dtype=np.float64).reshape(-1)
        selection = self.interpolator.at(direction)
        self._last_selection = selection
        if not self._primed:
            # The first block installs its filter outright. Fading into it
            # would fade up from silence, so a render would open with a ramp
            # nobody asked for.
            self._convolver.reset(selection.hrirs)
            self._primed = True
        else:
            self._convolver.set_filters(selection.hrirs)
        return self._convolver.process(samples)


def render_spatial_hrtf(
    mono,
    directions,
    dataset,
    sample_rate_hz: float,
    *,
    block_size: int = 512,
    spec: SpatialHrtfSpec | None = None,
):
    """Render a mono source along a direction trajectory.

    ``directions`` is either one ``(3,)`` direction held for the whole render
    or a ``(frames, 3)`` array; in the latter case the direction at each
    block's first sample selects that block's filter.
    """

    source = np.asarray(mono, dtype=np.float64).reshape(-1)
    spec = spec or SpatialHrtfSpec()
    renderer = SpatialHrtfRenderer(dataset, sample_rate_hz, spec)

    path = np.asarray(directions, dtype=np.float64)
    if path.ndim == 1:
        path = np.broadcast_to(path.reshape(1, 3), (source.size, 3))
    if path.shape != (source.size, 3):
        raise ValueError(
            f"directions must be (3,) or ({source.size}, 3), got shape {path.shape}"
        )

    block_size = max(1, int(block_size))
    output = np.zeros((2, source.size), dtype=np.float64)
    for start in range(0, source.size, block_size):
        stop = min(source.size, start + block_size)
        output[:, start:stop] = renderer.process_block(source[start:stop], path[start])

    if spec.headphone is not None and spec.headphone.is_active:
        output = apply_correction(output, spec.headphone)
    return output

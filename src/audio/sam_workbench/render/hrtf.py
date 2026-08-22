"""Time-varying explicit-SOFA binaural rendering."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike
from ..conventions import AUDIO_DTYPE
from ..hrtf import DelayPolicy, default_hrtf_cache
from ..hrtf.interpolation import interpolate_log_magnitude_delay, nearest_indices
from ..trajectory.transforms import ListenerTransform

InterpolationMode = Literal["nearest", "logmag_delay"]

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
        if self.interpolation not in ("nearest", "logmag_delay"):
            raise ValueError(f"unsupported interpolation {self.interpolation!r}")
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
    """Streaming FIR renderer with continuous history and equal-power switches."""
    def __init__(self, spec: HRTFRendererSpec):
        self.spec = spec
        self.dataset = None
        self._sample_rate_hz = 44100
        self._fade_frames = 1
        self._diagnostics = {}
        self.reset()

    def prepare(self, context, source=None):
        self._sample_rate_hz = int(context.sample_rate_hz)
        self.dataset = default_hrtf_cache.get(
            self.spec.sofa_path, self._sample_rate_hz, self.spec.delay_policy,
            self.spec.project_directory,
        )
        if self.spec.expected_sha256 and self.dataset.content_hash.lower() != self.spec.expected_sha256.lower():
            raise ValueError("SOFA asset hash does not match hrtfAssetHash")
        self._fade_frames = max(1, round(self.spec.crossfade_ms * self._sample_rate_hz / 1000))
        if self.dataset.has_external_delay:
            from scipy.ndimage import shift
            padding = int(np.ceil(np.max(self.dataset.delay_samples))) + 4
            delayed = np.pad(self.dataset.ir, ((0, 0), (0, 0), (0, padding)))
            for measurement in range(self.dataset.measurements):
                for ear in range(2):
                    delayed[measurement, ear] = shift(delayed[measurement, ear],
                        self.dataset.delay_samples[measurement, ear], order=3,
                        mode="constant", cval=0.0, prefilter=True)
            self._render_ir = delayed
        else:
            self._render_ir = self.dataset.ir
        # After the sample rate is known, so the delay line is sized for it.
        self.reset(0)

    def reset(self, sample_index=0):
        taps = self._render_ir.shape[-1] if hasattr(self, "_render_ir") else 1
        self._history = np.zeros(max(0, taps - 1), dtype=np.float64)
        self._next_sample = int(sample_index)
        self._current_filter = self._previous_filter = None
        self._filter_key = None
        self._fade_position = self._fade_frames
        # Propagation delay needs the input from up to maximum_distance_m ago,
        # held in a ring buffer so the per-sample read stays constant time.
        self._delay_line = np.zeros(self._delay_capacity(), dtype=np.float64)
        self._delay_write = 0
        self._gain = self._gain_step = 0.0
        self._delay_samples = self._delay_step = 0.0
        self._control_remaining = 0
        self._primed_distance = False

    def _delay_capacity(self):
        if not self.spec.propagation_delay:
            return 4
        span = self.spec.maximum_distance_m / self.spec.speed_of_sound_m_s
        return int(np.ceil(span * self._sample_rate_hz)) + 8

    def _read_delayed(self, delay_samples):
        """Four-point interpolated read from the ring buffer.

        Cubic rather than linear because a linearly interpolated fractional
        delay low-passes the signal by an amount that changes with the
        fractional part, which on a moving source is heard as the timbre
        wobbling in step with the motion.
        """

        size = len(self._delay_line)
        newest = self._delay_write - 1
        target = newest - max(0.0, float(delay_samples))
        base = int(np.floor(target))
        fraction = target - base
        p0, p1, p2, p3 = (self._delay_line[(base + offset) % size] for offset in (-1, 0, 1, 2))
        return p1 + 0.5 * fraction * (
            p2 - p0
            + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                          + fraction * (3.0 * (p1 - p2) + p3 - p0))
        )

    def _update_distance(self, position, frames_until_next):
        """Retarget gain and delay, ramping over the coming control interval.

        Stepping either one at a control boundary is audible - gain as a click,
        delay as a tear in the waveform - so both slew across the interval
        instead. The delay slewing is what produces the Doppler shift of an
        approaching or receding source.
        """

        distance = float(np.linalg.norm(position))
        gain = self.spec.distance_gain(distance)
        delay = 0.0
        if self.spec.propagation_delay:
            clamped = min(max(distance, self.spec.minimum_distance_m), self.spec.maximum_distance_m)
            delay = clamped / self.spec.speed_of_sound_m_s * self._sample_rate_hz
        if not self._primed_distance:
            # The first interval starts where it belongs rather than sliding in
            # from silence and zero delay.
            self._gain, self._delay_samples = gain, delay
            self._primed_distance = True
        span = max(int(frames_until_next), 1)
        self._gain_step = (gain - self._gain) / span
        self._delay_step = (delay - self._delay_samples) / span
        self._control_remaining = span

    def latency_samples(self): return 0
    def diagnostics(self): return dict(self._diagnostics)

    def _position(self, sample):
        time = np.array([sample / self._sample_rate_hz])
        trajectory = self.spec.trajectory
        world = trajectory(time) if callable(trajectory) else trajectory.evaluate(time)
        values = np.asarray(world, dtype=np.float64)
        if values.shape != (1, 3) or not np.all(np.isfinite(values)):
            raise ValueError("trajectory must return finite (frames, 3) positions")
        return self.spec.listener.world_to_listener(values)[0]

    def _target_filter(self, position):
        nearest = int(nearest_indices(self.dataset.positions_m, position[None])[0])
        if self.spec.interpolation == "nearest":
            return ("nearest", nearest), self._render_ir[nearest]
        direction = position / np.linalg.norm(position)
        reference = self.dataset.positions_m[nearest]
        reference = reference / np.linalg.norm(reference)
        if np.dot(direction, reference) > 1.0 - 1e-12:
            # Neutral reconstruction at a measured direction is exact rather
            # than an avoidable minimum-phase approximation.
            return ("logmag_delay", *np.round(direction, 6)), self._render_ir[nearest]
        target = interpolate_log_magnitude_delay(self.dataset, position)
        if target.shape[-1] < self._render_ir.shape[-1]:
            target = np.pad(target, ((0, 0), (0, self._render_ir.shape[-1] - target.shape[-1])))
        return ("logmag_delay", *np.round(direction, 6)), target

    @staticmethod
    def _fir(window, filt):
        return np.sum(filt * window[::-1][None, :], axis=1)

    def process(self, mono: ArrayLike, block):
        if self.dataset is None: raise RuntimeError("prepare() must be called before process()")
        source = np.asarray(mono, dtype=np.float64)
        if source.shape != (block.frames,): raise ValueError(f"mono input must have shape ({block.frames},), got {source.shape}")
        if int(block.start_sample) != self._next_sample: self.reset(int(block.start_sample))
        output = np.zeros((2, block.frames), dtype=np.float64)
        selected = np.empty(block.frames, dtype=np.int64)
        for local, value in enumerate(source):
            absolute = int(block.start_sample) + local
            if self._current_filter is None or absolute % self.spec.control_interval_samples == 0:
                position = self._position(absolute)
                # Direction and distance come from the same sampled position:
                # the filter encodes where the source is, these encode how far.
                self._update_distance(position, self.spec.control_interval_samples)
                key, target = self._target_filter(position)
                if self._current_filter is None:
                    self._current_filter, self._filter_key = np.asarray(target), key
                elif key != self._filter_key:
                    if self._previous_filter is not None and self._fade_position < self._fade_frames:
                        progress = self._fade_position / self._fade_frames
                        self._previous_filter = (
                            np.cos(progress*np.pi/2) * self._previous_filter
                            + np.sin(progress*np.pi/2) * self._current_filter
                        )
                    else:
                        self._previous_filter = self._current_filter
                    self._current_filter = np.asarray(target)
                    self._filter_key, self._fade_position = key, 0
            selected[local] = self._filter_key[1] if self._filter_key[0] == "nearest" else -1
            if self._control_remaining > 0:
                self._gain += self._gain_step
                self._delay_samples += self._delay_step
                self._control_remaining -= 1
            if self.spec.propagation_delay:
                self._delay_line[self._delay_write] = value
                self._delay_write = (self._delay_write + 1) % len(self._delay_line)
                value = self._read_delayed(self._delay_samples)
            if self.spec.distance_law != "none":
                value = value * self._gain
            window = np.concatenate((self._history, [value]))
            current = self._fir(window, self._current_filter)
            if self._previous_filter is not None and self._fade_position < self._fade_frames:
                previous = self._fir(window, self._previous_filter)
                progress = (self._fade_position + 1) / self._fade_frames
                output[:, local] = np.cos(progress*np.pi/2)*previous + np.sin(progress*np.pi/2)*current
                self._fade_position += 1
                if self._fade_position >= self._fade_frames: self._previous_filter = None
            else: output[:, local] = current
            self._history = window[1:]
        self._next_sample = int(block.start_sample) + block.frames
        self._diagnostics = {"hrtf_index": selected, "asset_sha256": self.dataset.content_hash,
            "interpolation": self.spec.interpolation, "delay_policy": DelayPolicy(self.spec.delay_policy).value,
            "distance_law": self.spec.distance_law, "distance_gain": float(self._gain),
            "propagation_delay_samples": float(self._delay_samples)}
        return output.astype(AUDIO_DTYPE, copy=False)

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

from ..dsp.convolution import CrossfadingConvolver, DEFAULT_FILTER_CROSSFADE_MS
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
        self._convolvers = [
            CrossfadingConvolver(
                crossfade_ms=self.spec.crossfade_ms, sample_rate_hz=self.sample_rate_hz
            )
            for _ in range(2)
        ]
        self._last_selection: object = None
        self.reset()

    def reset(self) -> None:
        for convolver in self._convolvers:
            convolver.reset()
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
        for ear, convolver in enumerate(self._convolvers):
            if self._primed:
                convolver.set_filter(selection.hrirs[ear])
            else:
                # The first block installs its filter outright. Fading into it
                # would fade up from the convolver's default passthrough, so a
                # render would open with the dry mono source bleeding through.
                convolver.reset(selection.hrirs[ear])
        self._primed = True
        return np.vstack([convolver.process(samples) for convolver in self._convolvers])


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

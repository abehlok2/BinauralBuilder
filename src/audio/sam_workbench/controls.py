"""Declarative controls and their compiled renderers.

Every automatable scalar in the workbench is one `ControlSpec`. A control both
describes itself (so it can be serialized) and renders itself, which keeps the
declarative form and the executed form from drifting apart.

The central guarantee is **block-size invariance**: a control renders from an
absolute sample index, so

.. code-block:: python

    control.render(0, 1000, sr) == concat(control.render(0, 137, sr),
                                          control.render(137, 863, sr))

holds exactly for every stateless control. This is what lets BinauralBuilder
render a step in arbitrary chunks and still get one continuous result.

Stateful controls (smoothing) declare `is_stateful` and must be rendered in
order; they are never used on the legacy compatibility path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.waveforms import TWO_PI, WAVEFORMS, evaluate_waveform

__all__ = [
    "WAVEFORMS",
    "CompiledControl",
    "ControlBase",
    "ConstantControl",
    "KeyframeControl",
    "LfoControl",
    "MapRangeControl",
    "ProductControl",
    "RampControl",
    "SumControl",
    "SmoothedControl",
    "as_control",
    "compile_control",
    "control_from_dict",
    "instantaneous_frequency_deviation_hz",
    "linear_transition_control",
    "sample_times",
]


@runtime_checkable
class CompiledControl(Protocol):
    """The interface every compiled control satisfies."""

    def reset(self, sample_index: int = 0) -> None: ...

    def render(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]: ...


def sample_times(start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
    """Absolute times in seconds for ``frames`` samples starting at ``start_sample``.

    Building the index array from the absolute sample number is what makes
    every stateless control bit-identical across block boundaries.
    """

    if frames < 0:
        raise ValueError(f"frames must not be negative, got {frames}")
    index = np.arange(int(start_sample), int(start_sample) + int(frames), dtype=np.float64)
    return index / float(sample_rate)


@dataclass(frozen=True)
class ControlBase:
    """Fields shared by every control.

    ``minimum``/``maximum`` are declared bounds, applied after the control's own
    computation, so a modulated value can never leave its documented range.
    """

    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    enabled: bool = True

    #: Stateless controls may be rendered from any absolute index in any order.
    is_stateful: bool = field(default=False, init=False, repr=False)

    # --- protocol ----------------------------------------------------------

    def reset(self, sample_index: int = 0) -> None:
        """No-op for stateless controls; stateful subclasses override it."""

    def render(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        values = self._evaluate(int(start_sample), int(frames), float(sample_rate))
        return self._bound(values)

    # --- helpers -----------------------------------------------------------

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        raise NotImplementedError

    def _bound(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.minimum is None and self.maximum is None:
            return values
        return np.clip(values, self.minimum, self.maximum)

    def constant_value(self) -> float | None:
        """Return the control's value when it is constant, otherwise ``None``."""

        return None

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantControl(ControlBase):
    """A fixed value. Renders as a broadcast-filled array."""

    value: float = 0.0

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        return np.full(frames, float(self.value), dtype=np.float64)

    def constant_value(self) -> float | None:
        value = float(self.value)
        if self.minimum is not None:
            value = max(value, self.minimum)
        if self.maximum is not None:
            value = min(value, self.maximum)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "constant", "value": float(self.value), "unit": self.unit}


@dataclass(frozen=True)
class LfoControl(ControlBase):
    """A periodic modulator: ``offset + depth * waveform(2*pi*rate*t + phase)``."""

    rate_hz: float = 1.0
    depth: float = 1.0
    offset: float = 0.0
    phase_rad: float = 0.0
    waveform: str = "sine"

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = sample_times(start_sample, frames, sample_rate)
        phase = TWO_PI * float(self.rate_hz) * times + float(self.phase_rad)
        return float(self.offset) + float(self.depth) * evaluate_waveform(self.waveform, phase)

    def constant_value(self) -> float | None:
        # A zero-depth or zero-rate LFO never moves; it sits at its phase value.
        if float(self.depth) == 0.0 or float(self.rate_hz) == 0.0:
            phase_value = evaluate_waveform(self.waveform, np.array([float(self.phase_rad)]))[0]
            return float(self.offset) + float(self.depth) * float(phase_value)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lfo",
            "rate_hz": float(self.rate_hz),
            "depth": float(self.depth),
            "offset": float(self.offset),
            "phase_rad": float(self.phase_rad),
            "waveform": str(self.waveform),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class RampControl(ControlBase):
    """An unbounded linear ramp: ``intercept + slope_per_s * t``.

    Used where a value must keep growing rather than settle, such as the
    per-ear phase that turns one shared carrier into a binaural beat.
    """

    slope_per_s: float = 0.0
    intercept: float = 0.0

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = sample_times(start_sample, frames, sample_rate)
        return float(self.intercept) + float(self.slope_per_s) * times

    def constant_value(self) -> float | None:
        return float(self.intercept) if float(self.slope_per_s) == 0.0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ramp",
            "slope_per_s": float(self.slope_per_s),
            "intercept": float(self.intercept),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class KeyframeControl(ControlBase):
    """Piecewise automation through ``(time_s, value)`` keyframes.

    Values are held before the first and after the last keyframe, which is what
    a BinauralBuilder transition needs: a start value, a ramp, and an end value
    that persists for the rest of the step.
    """

    keyframes: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("a KeyframeControl needs at least one keyframe")
        if self.interpolation not in ("linear", "step", "smooth"):
            raise ValueError(f"unknown interpolation {self.interpolation!r}")
        ordered = tuple(sorted(((float(time), float(value)) for time, value in self.keyframes)))
        object.__setattr__(self, "keyframes", ordered)

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = sample_times(start_sample, frames, sample_rate)
        knot_times = np.array([time for time, _ in self.keyframes], dtype=np.float64)
        knot_values = np.array([value for _, value in self.keyframes], dtype=np.float64)

        if knot_times.size == 1:
            return np.full(frames, knot_values[0], dtype=np.float64)
        if self.interpolation == "step":
            positions = np.clip(np.searchsorted(knot_times, times, side="right") - 1, 0, knot_values.size - 1)
            return knot_values[positions]

        values = np.interp(times, knot_times, knot_values)
        if self.interpolation == "smooth":
            # Smoothstep the fractional position inside each segment.
            positions = np.clip(np.searchsorted(knot_times, times, side="right") - 1, 0, knot_times.size - 2)
            segment_start = knot_times[positions]
            segment_span = knot_times[positions + 1] - segment_start
            with np.errstate(invalid="ignore", divide="ignore"):
                fraction = np.where(segment_span > 0.0, (times - segment_start) / segment_span, 0.0)
            fraction = np.clip(fraction, 0.0, 1.0)
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            values = knot_values[positions] + eased * (knot_values[positions + 1] - knot_values[positions])
        return values

    def constant_value(self) -> float | None:
        values = {value for _, value in self.keyframes}
        return float(next(iter(values))) if len(values) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "keyframe",
            "keyframes": [[float(time), float(value)] for time, value in self.keyframes],
            "interpolation": str(self.interpolation),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class SumControl(ControlBase):
    """Sum of nested controls."""

    terms: tuple[ControlBase, ...] = ()

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        total = np.zeros(frames, dtype=np.float64)
        for term in self.terms:
            if term.enabled:
                total += term.render(start_sample, frames, sample_rate)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "sum", "terms": [term.to_dict() for term in self.terms], "unit": self.unit}


@dataclass(frozen=True)
class ProductControl(ControlBase):
    """Product of nested controls."""

    factors: tuple[ControlBase, ...] = ()

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        total = np.ones(frames, dtype=np.float64)
        for factor in self.factors:
            if factor.enabled:
                total *= factor.render(start_sample, frames, sample_rate)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "product",
            "factors": [factor.to_dict() for factor in self.factors],
            "unit": self.unit,
        }


@dataclass(frozen=True)
class MapRangeControl(ControlBase):
    """Linear remap of a nested control from one range onto another."""

    source: ControlBase = field(default_factory=ConstantControl)
    from_low: float = -1.0
    from_high: float = 1.0
    to_low: float = 0.0
    to_high: float = 1.0

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        values = self.source.render(start_sample, frames, sample_rate)
        span = float(self.from_high) - float(self.from_low)
        if span == 0.0:
            return np.full(frames, float(self.to_low), dtype=np.float64)
        fraction = (values - float(self.from_low)) / span
        return float(self.to_low) + fraction * (float(self.to_high) - float(self.to_low))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "map_range",
            "source": self.source.to_dict(),
            "from_low": float(self.from_low),
            "from_high": float(self.from_high),
            "to_low": float(self.to_low),
            "to_high": float(self.to_high),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class SmoothedControl(ControlBase):
    """One-pole smoothing applied after control composition.

    Smoothing is stateful by nature, so this control must be rendered in
    order from its reset point. It is offered for GUI-driven parameter changes;
    the offline and legacy compatibility paths use stateless controls so their
    output cannot depend on the block size.
    """

    source: ControlBase = field(default_factory=ConstantControl)
    smoothing_ms: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_stateful", True)
        object.__setattr__(self, "_state", None)

    def reset(self, sample_index: int = 0) -> None:
        object.__setattr__(self, "_state", None)

    def _evaluate(self, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        target = self.source.render(start_sample, frames, sample_rate)
        if frames == 0:
            return target
        time_constant_s = max(0.0, float(self.smoothing_ms)) / 1000.0
        if time_constant_s <= 0.0:
            object.__setattr__(self, "_state", float(target[-1]))
            return target

        coefficient = math.exp(-1.0 / (time_constant_s * float(sample_rate)))
        state = getattr(self, "_state", None)
        if state is None:
            state = float(target[0])
        smoothed = np.empty_like(target)
        for index in range(frames):  # noqa: PLR1702 - one-pole recursion is inherently serial
            state = target[index] + coefficient * (state - target[index])
            smoothed[index] = state
        object.__setattr__(self, "_state", float(state))
        return smoothed

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "smoothed",
            "source": self.source.to_dict(),
            "smoothing_ms": float(self.smoothing_ms),
            "unit": self.unit,
        }


# --- compiler ---------------------------------------------------------------


def as_control(value: ControlBase | float | int) -> ControlBase:
    """Promote a bare number to a :class:`ConstantControl`."""

    if isinstance(value, ControlBase):
        return value
    return ConstantControl(value=float(value))


def compile_control(spec: ControlBase | float | int) -> ControlBase:
    """Compile a control specification into its renderer.

    Controls in this package are their own renderers, so compilation folds
    disabled and constant controls into a `ConstantControl` and otherwise
    returns the specification unchanged.
    """

    control = as_control(spec)
    if not control.enabled:
        return ConstantControl(value=0.0, unit=control.unit)
    constant = control.constant_value()
    if constant is not None:
        return ConstantControl(value=constant, unit=control.unit)
    return control


def linear_transition_control(
    start_value: float,
    end_value: float,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    unit: str = "",
    curve: str = "linear",
) -> ControlBase:
    """Compile a BinauralBuilder start/end parameter pair into keyframes.

    A transition is two keyframes: the start value held until ``start_s`` and
    the end value reached at ``start_s + duration_s`` and held afterwards. A
    zero-length or valueless transition folds into a constant.
    """

    if float(start_value) == float(end_value):
        return ConstantControl(value=float(start_value), unit=unit)
    if duration_s is None or float(duration_s) <= 0.0:
        return ConstantControl(value=float(end_value), unit=unit)
    interpolation = "smooth" if curve == "smooth" else "linear"
    return KeyframeControl(
        keyframes=(
            (float(start_s), float(start_value)),
            (float(start_s) + float(duration_s), float(end_value)),
        ),
        interpolation=interpolation,
        unit=unit,
    )


_CONTROL_KINDS: dict[str, type[ControlBase]] = {
    "constant": ConstantControl,
    "lfo": LfoControl,
    "keyframe": KeyframeControl,
    "ramp": RampControl,
    "sum": SumControl,
    "product": ProductControl,
    "map_range": MapRangeControl,
    "smoothed": SmoothedControl,
}


def control_from_dict(data: Mapping[str, Any]) -> ControlBase:
    """Rebuild a control from its serialized form."""

    if not isinstance(data, Mapping):
        raise TypeError(f"expected a control object, got {type(data).__name__}")
    kind = str(data.get("kind", "constant"))
    control_type = _CONTROL_KINDS.get(kind)
    if control_type is None:
        raise ValueError(f"unknown control kind {kind!r}")

    fields = {key: value for key, value in data.items() if key != "kind"}
    if kind == "keyframe":
        fields["keyframes"] = tuple((float(time), float(value)) for time, value in fields.get("keyframes", ()))
    for nested in ("source",):
        if nested in fields:
            fields[nested] = control_from_dict(fields[nested])
    for nested_sequence in ("terms", "factors"):
        if nested_sequence in fields:
            fields[nested_sequence] = tuple(control_from_dict(item) for item in fields[nested_sequence])
    return control_type(**fields)


def instantaneous_frequency_deviation_hz(depth_rad: float, rate_hz: float) -> float:
    """Peak instantaneous-frequency deviation ``|beta * f_m|`` for sinusoidal PM."""

    return abs(float(depth_rad) * float(rate_hz))

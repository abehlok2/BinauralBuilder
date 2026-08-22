"""Driving the abstract phase-modulation engine from a three-dimensional path.

The abstract renderer is opposed-ear phase modulation.  It is not a
spatializer: phase modulation alone cannot produce a reliable sense of height,
because the cues that carry elevation are spectral - the pinna notches an HRTF
measures - and there is no filter here to put them in.  What it *can* do is
take the path as a control source, so a trajectory authored once drives the
abstract voice and the HRTF voice together and the two stay in step.

Everything in this module is therefore labelled a **creative mapping**.  The
distinction is not cosmetic.  A listener told that elevation is being rendered
will report hearing height that is not there, and an experiment that treats
these mappings as spatialization measures the labelling rather than the audio.
:data:`CREATIVE_MAPPING_NOTICE` is the text the GUI and every export carry.

The mappings offered are the documented ones:

* azimuth to the interaural phase relationship - the one mapping with a real
  perceptual basis, since interaural phase genuinely carries lateral position;
* elevation to carrier frequency, spectral tilt, or modulation depth - a
  deliberate convention, chosen because rising brightness reads as rising
  height, not because it reproduces one;
* distance to amplitude or modulation intensity - a plausible loudness cue
  with no propagation delay or air absorption behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from ..controls import ControlBase, sample_times
from ..trajectory.spherical import cartesian_array_to_spherical

__all__ = [
    "CREATIVE_MAPPING_NOTICE",
    "PATH_QUANTITIES",
    "TrajectoryControl",
    "CreativeMapping",
    "CreativeMappingSpec",
    "MAPPING_TARGETS",
]

#: Shown wherever these mappings are configured or reported. The abstract
#: renderer must never present them as spatialization.
CREATIVE_MAPPING_NOTICE = (
    "Creative mapping, not spatialization. The abstract phase-modulation "
    "renderer uses the path as a control source; it does not reproduce "
    "direction or distance. Elevation in particular is a convention here - "
    "convincing height needs HRTF filtering. Use the HRTF or hybrid renderer "
    "for physical localization."
)

#: Parameters of the abstract renderer a path may drive, and their units.
MAPPING_TARGETS: Mapping[str, str] = {
    "interaural_phase": "rad",
    "carrier_hz": "Hz",
    "spectral_tilt": "",
    "modulation_depth": "rad",
    "modulation_rate_hz": "Hz",
    "amplitude": "",
}

#: Quantities a path can supply, and the range each spans by nature.
PATH_QUANTITIES: Mapping[str, tuple[float, float]] = {
    "azimuth": (-180.0, 180.0),
    "elevation": (-90.0, 90.0),
    "distance": (0.0, 10.0),
    "x": (-10.0, 10.0),
    "y": (-10.0, 10.0),
    "z": (-10.0, 10.0),
}


@dataclass(frozen=True)
class TrajectoryControl(ControlBase):
    """One quantity of a path, mapped onto a parameter's range.

    A control rather than a bespoke renderer input, so a path can drive any
    parameter the workbench already exposes without either side knowing about
    the other.  Stateless: every value comes from the absolute sample index, so
    blocks render identically in any order and at any size, which is what keeps
    a seeked preview identical to a sequential export.
    """

    #: A ``PathModel``, a ``CanonicalTrajectory``, or any callable from times
    #: in seconds to ``(frames, 3)`` metres.
    trajectory: Any = None
    quantity: str = "azimuth"
    #: The parameter range the quantity is mapped onto.
    output_low: float = 0.0
    output_high: float = 1.0
    #: The span of the quantity treated as full scale. Defaults to the natural
    #: range in :data:`PATH_QUANTITIES`.
    input_low: float | None = None
    input_high: float | None = None
    invert: bool = False

    def __post_init__(self) -> None:
        if self.quantity not in PATH_QUANTITIES:
            raise ValueError(
                f"unknown path quantity {self.quantity!r}; "
                f"expected one of {tuple(PATH_QUANTITIES)}"
            )
        if self.trajectory is None:
            raise ValueError("a trajectory control needs a trajectory")
        for name in ("output_low", "output_high"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        low, high = self._input_span()
        if high == low:
            raise ValueError("input_low and input_high must differ")

    def _input_span(self) -> tuple[float, float]:
        natural = PATH_QUANTITIES[self.quantity]
        return (
            natural[0] if self.input_low is None else float(self.input_low),
            natural[1] if self.input_high is None else float(self.input_high),
        )

    def _positions(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        trajectory = self.trajectory
        for name in ("positions", "evaluate"):
            method = getattr(trajectory, name, None)
            if callable(method):
                return np.asarray(method(times), dtype=np.float64)
        return np.asarray(trajectory(times), dtype=np.float64)

    def _evaluate(
        self, start_sample: int, frames: int, sample_rate: float
    ) -> NDArray[np.float64]:
        if frames <= 0:
            return np.zeros(0, dtype=np.float64)
        times = sample_times(start_sample, frames, sample_rate)
        points = self._positions(times)
        if points.shape != (frames, 3):
            raise ValueError(
                f"trajectory must return ({frames}, 3) positions, got {points.shape}"
            )
        if self.quantity in ("x", "y", "z"):
            raw = points[:, "xyz".index(self.quantity)]
        else:
            spherical = cartesian_array_to_spherical(points)
            raw = spherical[:, ("azimuth", "elevation", "distance").index(self.quantity)]

        low, high = self._input_span()
        normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)
        if self.invert:
            normalized = 1.0 - normalized
        return self.output_low + (self.output_high - self.output_low) * normalized

    def constant_value(self) -> float | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        low, high = self._input_span()
        return {
            "kind": "trajectory",
            "quantity": self.quantity,
            "outputLow": float(self.output_low),
            "outputHigh": float(self.output_high),
            "inputLow": float(low),
            "inputHigh": float(high),
            "invert": bool(self.invert),
            "unit": self.unit,
            # Carried into the document so a project cannot present this as
            # spatialization once it leaves the GUI that labelled it.
            "note": CREATIVE_MAPPING_NOTICE,
        }


@dataclass(frozen=True)
class CreativeMapping:
    """One path quantity wired to one abstract-renderer parameter."""

    quantity: str
    #: The parameter being driven; see :data:`CreativeMappingSpec.TARGETS`.
    target: str
    low: float
    high: float
    invert: bool = False

    def control(self, trajectory: Any) -> TrajectoryControl:
        return TrajectoryControl(
            trajectory=trajectory,
            quantity=self.quantity,
            output_low=self.low,
            output_high=self.high,
            invert=self.invert,
            unit=MAPPING_TARGETS.get(self.target, ""),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "target": self.target,
            "low": float(self.low),
            "high": float(self.high),
            "invert": bool(self.invert),
        }


@dataclass(frozen=True)
class CreativeMappingSpec:
    """The set of creative mappings applied to one abstract voice."""

    mappings: tuple[CreativeMapping, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        for mapping in self.mappings:
            if mapping.quantity not in PATH_QUANTITIES:
                raise ValueError(f"unknown path quantity {mapping.quantity!r}")
            if mapping.target not in MAPPING_TARGETS:
                raise ValueError(
                    f"unknown creative mapping target {mapping.target!r}; "
                    f"expected one of {tuple(MAPPING_TARGETS)}"
                )

    @property
    def is_neutral(self) -> bool:
        """True when nothing is being driven from the path."""

        return not self.enabled or not self.mappings

    def controls(self, trajectory: Any) -> dict[str, TrajectoryControl]:
        """Build one control per target, ready to attach to a compiled source."""

        if self.is_neutral:
            return {}
        return {
            mapping.target: mapping.control(trajectory) for mapping in self.mappings
        }

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "physical": False,
            "notice": CREATIVE_MAPPING_NOTICE,
            "mappings": [mapping.describe() for mapping in self.mappings],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CreativeMappingSpec":
        values = dict(data or {})
        entries = values.get("mappings", ())
        return cls(
            mappings=tuple(
                CreativeMapping(
                    quantity=str(entry.get("quantity", "azimuth")),
                    target=str(entry.get("target", "interaural_phase")),
                    low=float(entry.get("low", 0.0)),
                    high=float(entry.get("high", 1.0)),
                    invert=bool(entry.get("invert", False)),
                )
                for entry in entries
            ),
            enabled=bool(values.get("enabled", True)),
        )

    @classmethod
    def documented_default(cls) -> "CreativeMappingSpec":
        """The three mappings the specification names, at usable ranges.

        Azimuth to interaural phase across a full cycle, elevation to carrier
        over an octave, and distance to amplitude falling off with range.
        """

        return cls(
            mappings=(
                CreativeMapping("azimuth", "interaural_phase", -math.pi, math.pi),
                CreativeMapping("elevation", "carrier_hz", 180.0, 360.0),
                CreativeMapping("distance", "amplitude", 1.0, 0.2, invert=False),
            )
        )

"""Estimating what a scene will cost before paying for it.

Multiband and multi-source multiply. Four bands across eight sources is
thirty-two convolutions where there was one, and the difference between that
being fine and that being a stalled render is not obvious from the controls
that caused it. So the cost is stated up front, and a scene too expensive for
real time is routed to an offline render rather than left to underrun.

The model is deliberately simple and deliberately labelled a model. It counts
the multiply-accumulates the dominant stage actually performs - convolution
against an HRIR, per band, per source - and divides by a throughput figure for
this machine. That figure can be measured rather than assumed, and
:func:`measure_throughput` does so from a real convolution, because a guess
about someone else's hardware is worth very little.

What the estimate is for is the decision "real time or offline", and it is
consulted with headroom. Being wrong in the optimistic direction produces
dropouts; being wrong in the pessimistic direction produces an offline render
that was not strictly necessary, which is the cheaper mistake.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_REALTIME_HEADROOM",
    "CostEstimate",
    "CostInputs",
    "REALTIME",
    "OFFLINE",
    "estimate_cost",
    "interpolation_weight",
    "measure_throughput",
]

REALTIME = "realtime"
OFFLINE = "offline"

#: Only commit to real time below this fraction of the audio callback budget.
#: The remainder absorbs the rest of the application, and the operating system.
DEFAULT_REALTIME_HEADROOM = 0.5

#: Multiply-accumulates per second this machine can sustain, until measured.
#: Conservative on purpose: an optimistic default produces dropouts.
FALLBACK_THROUGHPUT_MACS = 4.0e8

#: How much work each interpolation mode does relative to plain nearest.
#: Nearest convolves one filter pair; the blending modes must also build that
#: pair, and the spherical harmonic mode evaluates a fit for every direction.
_INTERPOLATION_WEIGHTS = {
    "nearest": 1.0,
    "three_neighbor": 1.6,
    "spherical_triangular": 1.7,
    "delay_magnitude": 2.4,
    "spherical_harmonic": 3.0,
    # Phase 4's own two names, kept so an older project still estimates.
    "logmag_delay": 2.4,
}

#: Renderers that do not convolve at all cost far less per sample.
def _renderer_weight(identifier: str) -> float:
    """Relative cost per sample, from the renderer's own definition.

    The registry owns this so a renderer added there is costed correctly here
    without a second table needing to be remembered.
    """

    from src.audio.sam_workbench.render.registry import REGISTRY

    if identifier not in REGISTRY:
        return 1.0
    return float(REGISTRY.get(identifier).cost_weight)


def interpolation_weight(mode: str) -> float:
    """Relative cost of one interpolation mode. Unknown modes cost the most."""

    return float(_INTERPOLATION_WEIGHTS.get(str(mode), max(_INTERPOLATION_WEIGHTS.values())))


@dataclass(frozen=True)
class CostInputs:
    """Everything the estimate depends on, per §10.8."""

    sources: int = 1
    bands: int = 1
    hrir_taps: int = 256
    interpolation: str = "nearest"
    sample_rate_hz: float = 48_000.0
    block_size: int = 512
    renderer: str = "hrtf"
    #: Crossovers cost per band per source, on top of the convolution.
    crossover_order: int = 4

    def __post_init__(self) -> None:
        for name in ("sources", "bands", "hrir_taps", "block_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)!r}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def convolutions(self) -> int:
        """One filter pair per band per source, both ears."""

        return int(self.sources) * int(self.bands) * 2

    def describe(self) -> dict[str, Any]:
        return {
            "sources": int(self.sources),
            "bands": int(self.bands),
            "hrirTaps": int(self.hrir_taps),
            "interpolation": self.interpolation,
            "sampleRateHz": float(self.sample_rate_hz),
            "blockSize": int(self.block_size),
            "renderer": self.renderer,
        }


@dataclass(frozen=True)
class CostEstimate:
    """What a scene is predicted to cost, and what to do about it."""

    inputs: CostInputs
    macs_per_second: float
    throughput_macs: float
    #: Fraction of one audio callback budget this scene is predicted to use.
    load: float
    mode: str
    headroom: float
    measured: bool

    @property
    def feasible_realtime(self) -> bool:
        return self.mode == REALTIME

    @property
    def block_budget_s(self) -> float:
        """Wall-clock time one block of audio is allowed to take."""

        return float(self.inputs.block_size) / float(self.inputs.sample_rate_hz)

    def summary(self) -> str:
        """One line a status bar can show."""

        percent = self.load * 100.0
        detail = (
            f"{self.inputs.sources} source(s) x {self.inputs.bands} band(s), "
            f"{self.inputs.hrir_taps} taps, {self.inputs.interpolation}"
        )
        if self.mode == REALTIME:
            return f"Real time: about {percent:.0f}% of the audio budget ({detail})."
        return (
            f"Offline: about {percent:.0f}% of the audio budget, over the "
            f"{self.headroom * 100:.0f}% limit ({detail})."
        )

    def describe(self) -> dict[str, Any]:
        return {
            **self.inputs.describe(),
            "macsPerSecond": float(self.macs_per_second),
            "throughputMacs": float(self.throughput_macs),
            "load": float(self.load),
            "mode": self.mode,
            "headroom": float(self.headroom),
            "throughputMeasured": bool(self.measured),
        }


def estimate_cost(
    inputs: CostInputs | None = None,
    *,
    throughput_macs: float | None = None,
    headroom: float = DEFAULT_REALTIME_HEADROOM,
    **overrides: Any,
) -> CostEstimate:
    """Predict a scene's cost and whether it belongs in real time.

    ``throughput_macs`` is this machine's sustained multiply-accumulate rate;
    :func:`measure_throughput` obtains it honestly. Left unset, a deliberately
    conservative fallback is used and the estimate says it was not measured.
    """

    inputs = inputs or CostInputs()
    if overrides:
        from dataclasses import replace

        inputs = replace(inputs, **overrides)

    measured = throughput_macs is not None
    throughput = float(throughput_macs or FALLBACK_THROUGHPUT_MACS)
    if throughput <= 0:
        raise ValueError("throughput_macs must be positive")
    if not 0.0 < headroom <= 1.0:
        raise ValueError(f"headroom must lie in (0, 1], got {headroom!r}")

    renderer_weight = _renderer_weight(str(inputs.renderer))

    # Overlap-save convolution is a transform pair per block rather than a tap
    # per sample, so the per-sample cost grows with the log of the filter, not
    # with the filter.
    per_sample = math.log2(max(inputs.hrir_taps, 2)) * 4.0
    convolution = (
        per_sample
        * inputs.convolutions
        * interpolation_weight(inputs.interpolation)
        * renderer_weight
    )

    # Splitting into bands costs a biquad cascade per band per source, run
    # twice for Linkwitz-Riley, on top of whatever the bands then do.
    crossover = 0.0
    if inputs.bands > 1:
        sections = max(1, inputs.crossover_order // 2)
        crossover = inputs.sources * inputs.bands * 2 * sections * 2 * 5.0

    macs_per_second = (convolution + crossover) * float(inputs.sample_rate_hz)
    load = macs_per_second / throughput
    mode = REALTIME if load <= headroom else OFFLINE

    return CostEstimate(
        inputs=inputs,
        macs_per_second=macs_per_second,
        throughput_macs=throughput,
        load=load,
        mode=mode,
        headroom=headroom,
        measured=measured,
    )


def measure_throughput(*, taps: int = 256, frames: int = 1 << 15, repeats: int = 3) -> float:
    """Measure this machine's sustained convolution throughput.

    Times the operation the estimate is actually about - an overlap-save
    convolution - and converts to multiply-accumulates per second, so the
    prediction is calibrated against the machine it will run on rather than
    against an assumption about someone else's.
    """

    from .dsp.convolution import OverlapSaveConvolver

    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, int(frames))
    filter_taps = rng.normal(0.0, 1.0, int(taps))

    best = math.inf
    for _ in range(max(1, int(repeats))):
        convolver = OverlapSaveConvolver(filter_taps=filter_taps)
        start = time.perf_counter()
        convolver.process(signal)
        best = min(best, time.perf_counter() - start)

    per_sample = math.log2(max(int(taps), 2)) * 4.0
    if best <= 0.0:  # pragma: no cover - clock resolution
        return FALLBACK_THROUGHPUT_MACS
    return float(frames * per_sample / best)

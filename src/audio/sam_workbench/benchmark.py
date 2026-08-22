"""Measuring what the renderer actually costs, before deciding to make it faster.

The specification asks for a benchmark across a named matrix - sample rates,
source counts, filter lengths, interpolation modes, band counts, block sizes -
against a target of less than half the audio callback budget on a stated
reference machine. This module is that benchmark, and it exists first so that
optimization is aimed at measurements rather than at intuitions about which
loop looks expensive.

Two numbers come out of each case, and they answer different questions. The
*average* load is the specification's criterion and says whether the work fits
at all. The *worst block* says whether it fits every time: an audio callback
that runs over budget once has already produced a dropout, and an average that
hides one slow block in two hundred is describing a stream that clicks.

Feasibility is judged on the average, as the specification asks, because the
worst block on a shared or virtualised machine is mostly a measure of the
operating system's scheduler rather than of this code - it moves by tens of
percent between runs that agree on the mean to a fraction of one. The worst
block and the 95th percentile are still reported, and :attr:`has_overruns`
calls out a case that actually ran past a full budget, which no amount of
scheduler noise excuses.

Timing is measured with :func:`time.perf_counter` around each block, which
includes whatever the interpreter and the allocator were doing at the time.
That is the honest measure for this question: an audio callback is charged for
those too.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .hrtf.interpolation import INTERPOLATION_MODES, NEAREST
from .render.routing import BandRouting

__all__ = [
    "REALTIME_TARGET_LOAD",
    "BenchmarkCase",
    "BenchmarkResult",
    "SyntheticDataset",
    "describe_machine",
    "run_case",
    "run_matrix",
    "standard_matrix",
    "synthetic_dataset",
]

#: The specification's target: under half the callback budget on average, so
#: there is margin left for operating-system jitter.
REALTIME_TARGET_LOAD = 0.5

#: Rates the matrix sweeps.
SAMPLE_RATES_HZ = (44_100.0, 48_000.0)
SOURCE_COUNTS = (1, 4, 8)
HRIR_TAPS = (128, 512, 2048)
BAND_COUNTS = (1, 4, 8)
BLOCK_SIZES = (128, 512, 1024)


# --- material ---------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDataset:
    """Just enough of an HRTF dataset to drive the renderer.

    Real measured sets differ in tap count and direction density, which is
    exactly what the matrix is sweeping, so the benchmark generates its own
    rather than depending on an asset that may or may not be installed. The
    responses are not physically meaningful and are not intended to be: this
    measures cost, and cost does not depend on whether the pinna notch is in
    the right place.
    """

    ir: NDArray[np.float64]
    positions_m: NDArray[np.float64]
    sample_rate_hz: float
    delay_samples: NDArray[np.float64]

    @property
    def measurements(self) -> int:
        return int(self.ir.shape[0])

    @property
    def taps(self) -> int:
        return int(self.ir.shape[2])

    @property
    def has_external_delay(self) -> bool:
        return bool(np.any(np.abs(self.delay_samples) > 1e-12))


def synthetic_dataset(
    *, taps: int = 256, azimuth_step_deg: float = 15.0, elevations_deg: Sequence[float] = (-30.0, 0.0, 30.0),
    sample_rate_hz: float = 48_000.0, seed: int = 0,
) -> SyntheticDataset:
    """A direction grid of decaying-noise responses with a plausible delay."""

    rng = np.random.default_rng(seed)
    azimuths = np.arange(-180.0, 180.0, float(azimuth_step_deg))
    directions = [
        (azimuth, elevation) for elevation in elevations_deg for azimuth in azimuths
    ]

    positions = np.empty((len(directions), 3), dtype=np.float64)
    responses = np.zeros((len(directions), 2, int(taps)), dtype=np.float64)
    envelope = np.exp(-np.arange(int(taps)) / max(1.0, taps / 8.0))

    for index, (azimuth, elevation) in enumerate(directions):
        radians_azimuth = np.radians(azimuth)
        radians_elevation = np.radians(elevation)
        cos_elevation = np.cos(radians_elevation)
        positions[index] = (
            cos_elevation * np.cos(radians_azimuth),
            cos_elevation * np.sin(radians_azimuth),
            np.sin(radians_elevation),
        )
        for ear in range(2):
            # A few samples of onset, so the responses are not all impulses at
            # tap zero and the aligned modes have something to align.
            onset = 4 + int(round(3.0 * np.sin(radians_azimuth) * (1 if ear == 0 else -1)))
            onset = max(0, min(int(taps) - 1, onset))
            tail = rng.normal(0.0, 0.2, int(taps) - onset)
            responses[index, ear, onset:] = tail * envelope[: tail.size]

    return SyntheticDataset(
        ir=responses,
        positions_m=positions,
        sample_rate_hz=float(sample_rate_hz),
        delay_samples=np.zeros((len(directions), 2), dtype=np.float64),
    )


# --- cases ------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """One point in the matrix."""

    sources: int = 1
    hrir_taps: int = 256
    interpolation: str = NEAREST
    bands: int = 1
    anchor: bool = False
    block_size: int = 512
    sample_rate_hz: float = 48_000.0
    moving: bool = True

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATION_MODES:
            raise ValueError(
                f"unsupported interpolation {self.interpolation!r}; "
                f"expected one of {INTERPOLATION_MODES}"
            )
        for name in ("sources", "hrir_taps", "bands", "block_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)!r}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def block_budget_s(self) -> float:
        """Wall-clock time one block of audio is allowed to take."""

        return float(self.block_size) / float(self.sample_rate_hz)

    def label(self) -> str:
        return (
            f"{self.sources}src {self.hrir_taps}tap {self.interpolation} "
            f"{self.bands}band block{self.block_size} @{self.sample_rate_hz / 1000:g}k"
            + (" +anchor" if self.anchor else "")
        )

    def describe(self) -> dict[str, Any]:
        return {
            "sources": int(self.sources),
            "hrirTaps": int(self.hrir_taps),
            "interpolation": self.interpolation,
            "bands": int(self.bands),
            "anchor": bool(self.anchor),
            "blockSize": int(self.block_size),
            "sampleRateHz": float(self.sample_rate_hz),
            "moving": bool(self.moving),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """What one case cost, averaged and at its worst."""

    case: BenchmarkCase
    blocks: int
    block_times_s: tuple[float, ...] = field(repr=False, default=())

    @property
    def total_s(self) -> float:
        return float(sum(self.block_times_s))

    @property
    def audio_s(self) -> float:
        return self.blocks * self.case.block_budget_s

    @property
    def mean_load(self) -> float:
        """Average fraction of the callback budget consumed."""

        if not self.block_times_s:
            return 0.0
        return float(statistics.fmean(self.block_times_s)) / self.case.block_budget_s

    @property
    def worst_load(self) -> float:
        """The slowest single block, as a fraction of its budget."""

        if not self.block_times_s:
            return 0.0
        return max(self.block_times_s) / self.case.block_budget_s

    def load_percentile(self, percentile: float) -> float:
        if not self.block_times_s:
            return 0.0
        ordered = sorted(self.block_times_s)
        position = min(len(ordered) - 1, max(0, int(round(percentile / 100.0 * (len(ordered) - 1)))))
        return ordered[position] / self.case.block_budget_s

    @property
    def feasible_realtime(self) -> bool:
        """The specification's criterion: average load under the target."""

        return self.mean_load <= REALTIME_TARGET_LOAD

    @property
    def has_overruns(self) -> bool:
        """Whether any single block took longer than a block of audio lasts."""

        return self.worst_load > 1.0

    def summary(self) -> str:
        verdict = "real time" if self.feasible_realtime else "OFFLINE"
        if self.has_overruns:
            verdict += " (overran)"
        return (
            f"{self.case.label():<58} mean {self.mean_load * 100:6.1f}%  "
            f"p95 {self.load_percentile(95) * 100:6.1f}%  "
            f"worst {self.worst_load * 100:6.1f}%  {verdict}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "case": self.case.describe(),
            "blocks": int(self.blocks),
            "meanLoad": float(self.mean_load),
            "p95Load": float(self.load_percentile(95)),
            "worstLoad": float(self.worst_load),
            "feasibleRealtime": bool(self.feasible_realtime),
            "hasOverruns": bool(self.has_overruns),
        }


# --- running ----------------------------------------------------------------


def describe_machine() -> dict[str, Any]:
    """What was measured on. A load figure without this means nothing."""

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _trajectory(blocks: int, moving: bool) -> NDArray[np.float64]:
    """One direction per block, either circling the listener or held."""

    if not moving:
        return np.tile(np.array([1.0, 0.0, 0.0]), (blocks, 1))
    angles = np.linspace(0.0, 4.0 * np.pi, blocks, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))


def run_case(
    case: BenchmarkCase,
    *,
    seconds: float = 2.0,
    warmup_blocks: int = 4,
    dataset: SyntheticDataset | None = None,
) -> BenchmarkResult:
    """Time one case block by block, as an audio callback would be charged.

    Warm-up blocks are discarded: the first call through a code path pays for
    interpolator caches and FFT plan setup, which a running stream pays once
    and not on every callback.
    """

    from .render.hrtf import SpatialHrtfRenderer, SpatialHrtfSpec

    dataset = dataset or synthetic_dataset(
        taps=case.hrir_taps, sample_rate_hz=case.sample_rate_hz
    )
    renderers = [
        SpatialHrtfRenderer(
            dataset,
            case.sample_rate_hz,
            SpatialHrtfSpec(interpolation=case.interpolation),
        )
        for _ in range(case.sources)
    ]

    splitters = []
    if case.bands > 1:
        crossovers = tuple(
            float(value)
            for value in np.geomspace(120.0, 8000.0, case.bands - 1)
        )
        routing = BandRouting(crossovers_hz=crossovers)
        # One stateful splitter per source: a stream has to be split with the
        # filter state carried across blocks, which is what the renderer does.
        splitters = [
            routing.bank(case.sample_rate_hz).stream() for _ in range(case.sources)
        ]

    rng = np.random.default_rng(0)
    blocks = max(1, int(round(seconds * case.sample_rate_hz / case.block_size)))
    signal = rng.normal(0.0, 0.1, (case.sources, blocks + warmup_blocks, case.block_size))
    anchor_signal = (
        rng.normal(0.0, 0.02, (blocks + warmup_blocks, case.block_size))
        if case.anchor
        else None
    )
    path = _trajectory(blocks + warmup_blocks, case.moving)

    times: list[float] = []
    for index in range(blocks + warmup_blocks):
        start = time.perf_counter()
        mix = np.zeros((2, case.block_size), dtype=np.float64)
        for source, renderer in enumerate(renderers):
            block = signal[source, index]
            if splitters:
                # Splitting costs a biquad cascade per band before any of the
                # bands reach the convolver.
                bands = splitters[source].process(block)
                block = np.sum(bands, axis=0)
            mix += renderer.process_block(block, path[index])
        if anchor_signal is not None:
            mix += anchor_signal[index]
        elapsed = time.perf_counter() - start
        if index >= warmup_blocks:
            times.append(elapsed)

    return BenchmarkResult(case=case, blocks=blocks, block_times_s=tuple(times))


def standard_matrix() -> tuple[BenchmarkCase, ...]:
    """The specification's sweep, one axis varied at a time.

    A full cross product would be several thousand cases and would take longer
    to run than anyone will wait for, which means it would not be run. Varying
    one axis at a time around a stated centre answers the question the matrix
    is actually asked - which axis costs what - and stays runnable.
    """

    centre = BenchmarkCase()
    cases: list[BenchmarkCase] = [centre]

    def add(**changes: Any) -> None:
        from dataclasses import replace

        candidate = replace(centre, **changes)
        if candidate not in cases:
            cases.append(candidate)

    for rate in SAMPLE_RATES_HZ:
        add(sample_rate_hz=rate)
    for sources in SOURCE_COUNTS:
        add(sources=sources)
    for taps in HRIR_TAPS:
        add(hrir_taps=taps)
    for mode in INTERPOLATION_MODES:
        add(interpolation=mode)
    for bands in BAND_COUNTS:
        add(bands=bands)
    for block in BLOCK_SIZES:
        add(block_size=block)
    add(anchor=True)
    return tuple(cases)


def run_matrix(
    cases: Iterable[BenchmarkCase] | None = None, *, seconds: float = 1.0
) -> tuple[BenchmarkResult, ...]:
    """Run every case, reusing one dataset per tap length."""

    cases = tuple(cases if cases is not None else standard_matrix())
    datasets: dict[tuple[int, float], SyntheticDataset] = {}
    results: list[BenchmarkResult] = []
    for case in cases:
        key = (int(case.hrir_taps), float(case.sample_rate_hz))
        if key not in datasets:
            datasets[key] = synthetic_dataset(
                taps=case.hrir_taps, sample_rate_hz=case.sample_rate_hz
            )
        results.append(run_case(case, seconds=seconds, dataset=datasets[key]))
    return tuple(results)


def report(results: Iterable[BenchmarkResult]) -> str:
    """A table, with the machine it was measured on at the top."""

    machine = describe_machine()
    lines = [
        "SAM workbench render benchmark",
        f"  {machine['platform']}",
        f"  python {machine['python']}, numpy {machine['numpy']}",
        f"  target: average block under {REALTIME_TARGET_LOAD * 100:.0f}% of the callback budget",
        "",
    ]
    lines.extend(result.summary() for result in results)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the SAM workbench renderer.")
    parser.add_argument("--seconds", type=float, default=1.0, help="audio seconds per case")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    arguments = parser.parse_args(argv)

    results = run_matrix(seconds=arguments.seconds)
    if arguments.json:
        import json

        print(
            json.dumps(
                {
                    "machine": describe_machine(),
                    "results": [result.describe() for result in results],
                },
                indent=2,
            )
        )
    else:
        print(report(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())

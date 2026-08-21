"""Buses, stems, and per-band routing for scenes with more than one source.

Three things a multi-source render has to get right, and one it has to avoid.

**Stems stay separate.** Every source and every bus keeps its own audio until
the master sums them, so a mix can be inspected, soloed, or level-matched after
the fact rather than re-rendered to answer a question about one part of it.

**Solo and mute are deterministic.** Muting a source must change that source and
nothing else. Anything seeded - noise, decorrelation, grains - derives its seed
from a stable source identifier rather than from position in a list, so
disabling one source cannot shift the random stream of another.

**Bands reconstruct.** A per-band spatializer is only meaningful if summing the
bands returns the input, which is what :mod:`..dsp.crossover` provides.

The thing to avoid is silent cost. Bands multiply work by the number of bands,
and every source multiplies it again; :mod:`..cost` exists so the multiplication
is stated before it is paid rather than discovered as a stalled render.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, db_to_linear
from ..dsp.crossover import DEFAULT_CROSSOVER_ORDER, CrossoverBank

__all__ = [
    "BandRouting",
    "BusSpec",
    "RoutedScene",
    "SourceRouting",
    "band_stems",
    "deterministic_seed",
    "mix_routed",
]

#: The master bus every source reaches unless routed elsewhere.
MASTER_BUS = "master"


def deterministic_seed(*parts: Any, base: int = 0) -> int:
    """A stable seed derived from identifiers rather than from ordering.

    Two renders of the same scene must produce the same noise, and muting one
    source must not change any other. Deriving the seed from the source's own
    identity gives both: nothing about it depends on how many neighbours it has
    or where it sits in a list.
    """

    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8"))
    return (int.from_bytes(digest.digest()[:8], "big") ^ int(base)) & 0x7FFF_FFFF


@dataclass(frozen=True)
class BusSpec:
    """A named summing point with its own gain and solo/mute state."""

    id: str = MASTER_BUS
    name: str = "Master"
    gain_db: float = 0.0
    muted: bool = False
    soloed: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "gainDb": float(self.gain_db),
            "muted": bool(self.muted),
            "soloed": bool(self.soloed),
        }


@dataclass(frozen=True)
class SourceRouting:
    """Where one source goes, and how loudly."""

    source_id: str
    bus_id: str = MASTER_BUS
    gain_db: float = 0.0
    muted: bool = False
    soloed: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "busId": self.bus_id,
            "gainDb": float(self.gain_db),
            "muted": bool(self.muted),
            "soloed": bool(self.soloed),
        }


@dataclass(frozen=True)
class BandRouting:
    """Crossover frequencies and what happens in each resulting band."""

    crossovers_hz: tuple[float, ...] = ()
    #: One entry per band, low to high. Missing entries default to unity.
    band_gains_db: tuple[float, ...] = ()
    band_enabled: tuple[bool, ...] = ()
    order: int = DEFAULT_CROSSOVER_ORDER

    @property
    def band_count(self) -> int:
        return len(self.crossovers_hz) + 1

    @property
    def is_wideband(self) -> bool:
        """True when there is nothing to split."""

        return not self.crossovers_hz

    def gain_for(self, band: int) -> float:
        if band < len(self.band_gains_db):
            return float(db_to_linear(self.band_gains_db[band]))
        return 1.0

    def enabled_for(self, band: int) -> bool:
        if band < len(self.band_enabled):
            return bool(self.band_enabled[band])
        return True

    def bank(self, sample_rate_hz: float) -> CrossoverBank:
        return CrossoverBank(self.crossovers_hz, float(sample_rate_hz), self.order)

    def describe(self) -> dict[str, Any]:
        return {
            "crossoversHz": [float(value) for value in self.crossovers_hz],
            "bandGainsDb": [float(value) for value in self.band_gains_db],
            "bandEnabled": [bool(value) for value in self.band_enabled],
            "order": int(self.order),
        }


def band_stems(
    audio: NDArray[np.floating],
    routing: BandRouting,
    sample_rate_hz: float,
) -> NDArray[np.float64]:
    """Split channel-major audio into per-band stems, gains applied.

    A wideband routing returns the input as a single band rather than running
    it through a crossover that would only add phase shift for nothing.
    """

    block = np.asarray(audio, dtype=np.float64)
    if routing.is_wideband:
        return block[None, ...].copy()

    bands = routing.bank(sample_rate_hz).split(block)
    scaled = np.empty_like(bands)
    for index in range(bands.shape[0]):
        if not routing.enabled_for(index):
            scaled[index] = 0.0
            continue
        scaled[index] = bands[index] * routing.gain_for(index)
    return scaled


@dataclass(frozen=True)
class RoutedScene:
    """The result of mixing sources through buses, with everything kept."""

    master: NDArray[np.float64]
    #: Post-gain, post-mute audio for each source, by source id.
    source_stems: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    #: Summed audio for each bus, by bus id.
    bus_stems: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    #: Sources silenced by mute or by another source's solo.
    silenced: tuple[str, ...] = ()

    @property
    def frames(self) -> int:
        return int(self.master.shape[1])

    def levels(self) -> dict[str, float]:
        """Root-mean-square of the master and of every stem."""

        def rms(block: NDArray[np.floating]) -> float:
            values = np.asarray(block, dtype=np.float64)
            return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0

        levels = {"master": rms(self.master)}
        levels.update({f"source:{key}": rms(value) for key, value in self.source_stems.items()})
        levels.update({f"bus:{key}": rms(value) for key, value in self.bus_stems.items()})
        return levels


def _active_sources(routings: Sequence[SourceRouting], buses: Mapping[str, BusSpec]) -> set[str]:
    """Which sources survive the solo and mute rules.

    Solo is exclusive within its scope: any soloed source silences every other
    source, and any soloed bus silences every other bus. Mute always wins over
    solo on the same object, because muting something is the more specific
    instruction.
    """

    soloed_sources = {routing.source_id for routing in routings if routing.soloed and not routing.muted}
    soloed_buses = {bus.id for bus in buses.values() if bus.soloed and not bus.muted}

    active: set[str] = set()
    for routing in routings:
        if routing.muted:
            continue
        bus = buses.get(routing.bus_id)
        if bus is not None and bus.muted:
            continue
        if soloed_sources and routing.source_id not in soloed_sources:
            continue
        if soloed_buses and routing.bus_id not in soloed_buses:
            continue
        active.add(routing.source_id)
    return active


def mix_routed(
    stems: Mapping[str, NDArray[np.floating]],
    routings: Sequence[SourceRouting],
    buses: Sequence[BusSpec] = (),
    *,
    frames: int | None = None,
) -> RoutedScene:
    """Sum per-source stems through their buses into a master.

    ``stems`` maps source id to channel-major audio. Sources without a routing
    reach the master at unity, so a scene that has never been routed still
    renders.
    """

    bus_map = {bus.id: bus for bus in buses}
    bus_map.setdefault(MASTER_BUS, BusSpec())

    known = {routing.source_id: routing for routing in routings}
    for source_id in stems:
        known.setdefault(source_id, SourceRouting(source_id))
    ordered = [known[source_id] for source_id in sorted(known)]

    if frames is None:
        frames = max((np.asarray(block).shape[-1] for block in stems.values()), default=0)
    frames = int(frames)

    active = _active_sources(ordered, bus_map)
    source_stems: dict[str, NDArray[np.float64]] = {}
    bus_totals: dict[str, NDArray[np.float64]] = {
        bus_id: np.zeros((CHANNEL_COUNT, frames), dtype=np.float64) for bus_id in bus_map
    }
    silenced: list[str] = []

    for routing in ordered:
        raw = stems.get(routing.source_id)
        if raw is None:
            continue
        block = np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
        source = np.asarray(raw, dtype=np.float64)
        span = min(frames, source.shape[-1])
        if routing.source_id in active:
            block[:, :span] = source[:, :span] * float(db_to_linear(routing.gain_db))
        else:
            silenced.append(routing.source_id)
        source_stems[routing.source_id] = block

        bus_totals.setdefault(
            routing.bus_id, np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
        )
        bus_totals[routing.bus_id] += block

    master = np.zeros((CHANNEL_COUNT, frames), dtype=np.float64)
    for bus_id, total in bus_totals.items():
        bus = bus_map.get(bus_id, BusSpec(id=bus_id))
        bus_totals[bus_id] = total * float(db_to_linear(bus.gain_db))
        master += bus_totals[bus_id]

    return RoutedScene(
        master=master,
        source_stems=source_stems,
        bus_stems=bus_totals,
        silenced=tuple(sorted(silenced)),
    )

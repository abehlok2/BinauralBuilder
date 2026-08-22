"""Mixing a scene's sources through its buses and bands, block by block.

The pieces for this existed and were tested, and nothing used them.
:func:`~.routing.mix_routed` summed sources through buses with solo and mute;
:class:`~..dsp.crossover.CrossoverStream` split a stream into bands while
carrying its filter state. Production did neither. It folded a source's bus
gain and its mute/solo state into a single scalar multiplied onto that source's
own audio, which gets the *level* right for simple cases and is not a mixer:
there is no bus to meter, no bus to process, and nothing a band setting can act
on.

This is the mixer. It is stateful because band splitting is: a biquad's output
depends on the samples before it, so restarting the filters every block steps
every band's output by roughly a quarter of the signal's peak - a click, not a
change of tone. The crossover state therefore lives here and persists across
calls, which is what makes a blocked render identical to a whole one.

Bands are applied per bus. The scene carries one band configuration, so with
the default single master bus that is the master; with several buses each gets
its own filter state, which is what a multiband bus processor is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT, db_to_linear
from ..dsp.crossover import CrossoverStream
from .routing import (
    MASTER_BUS,
    BandRouting,
    BusSpec,
    RoutedScene,
    SourceRouting,
    _active_sources,
)

__all__ = ["SceneMixer", "SceneMixReport", "mixer_from_plan"]


@dataclass(frozen=True)
class SceneMixReport:
    """What one mix did, for metering and diagnostics."""

    frames: int
    #: Peak of each bus after its own gain, by bus id.
    bus_peaks: dict[str, float] = field(default_factory=dict)
    #: Peak of each source after its gain, by source id.
    source_peaks: dict[str, float] = field(default_factory=dict)
    master_peak: float = 0.0
    silenced: tuple[str, ...] = ()
    #: Bands that are configured off, and so contribute nothing.
    disabled_bands: tuple[int, ...] = ()
    band_count: int = 1

    def describe(self) -> dict[str, Any]:
        return {
            "frames": int(self.frames),
            "masterPeak": float(self.master_peak),
            "busPeaks": {name: float(value) for name, value in self.bus_peaks.items()},
            "sourcePeaks": {name: float(value) for name, value in self.source_peaks.items()},
            "silenced": list(self.silenced),
            "bandCount": int(self.band_count),
            "disabledBands": list(self.disabled_bands),
        }


class SceneMixer:
    """Sum sources into buses, band-process each bus, sum buses to master."""

    def __init__(
        self,
        routings: tuple[SourceRouting, ...] = (),
        buses: tuple[BusSpec, ...] = (),
        bands: BandRouting | None = None,
        *,
        sample_rate_hz: float = 44100.0,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.routings = tuple(routings)
        self.buses = tuple(buses) or (BusSpec(),)
        self.bands = bands or BandRouting()
        self._streams: dict[str, CrossoverStream] = {}
        self._report = SceneMixReport(frames=0)

    # --- state --------------------------------------------------------------

    @property
    def bus_map(self) -> dict[str, BusSpec]:
        mapping = {bus.id: bus for bus in self.buses}
        mapping.setdefault(MASTER_BUS, BusSpec())
        return mapping

    def reset(self) -> None:
        """Forget the crossover history, as at the start of a new stream."""

        for stream in self._streams.values():
            stream.reset()

    def diagnostics(self) -> dict[str, Any]:
        return self._report.describe()

    def _stream_for(self, bus_id: str) -> CrossoverStream:
        stream = self._streams.get(bus_id)
        if stream is None:
            stream = CrossoverStream(self.bands.bank(self.sample_rate_hz))
            self._streams[bus_id] = stream
        return stream

    # --- mixing -------------------------------------------------------------

    def _band_process(self, bus_id: str, audio: NDArray[np.float64]) -> NDArray[np.float64]:
        """Split a bus into bands, apply each band's gain, and sum back.

        A wideband configuration returns the input untouched rather than
        running it through a crossover that would only add phase shift for
        nothing.
        """

        if self.bands.is_wideband:
            return audio
        bands = self._stream_for(bus_id).process(audio)
        total = np.zeros_like(audio)
        for index in range(bands.shape[0]):
            if not self.bands.enabled_for(index):
                continue
            total += bands[index] * self.bands.gain_for(index)
        return total

    def process(
        self,
        stems: Mapping[str, NDArray[np.floating]],
        *,
        frames: int | None = None,
    ) -> RoutedScene:
        """Mix one block of per-source stems into a master.

        ``stems`` maps source id to channel-major ``(2, frames)`` audio.
        Sources with no routing reach the master at unity, so a scene that has
        never been routed still renders.
        """

        buses = self.bus_map
        known = {routing.source_id: routing for routing in self.routings}
        for source_id in stems:
            known.setdefault(source_id, SourceRouting(source_id))
        ordered = [known[source_id] for source_id in sorted(known)]

        if frames is None:
            frames = max(
                (np.asarray(block).shape[-1] for block in stems.values()), default=0
            )
        frames = int(frames)

        active = _active_sources(ordered, buses)
        source_stems: dict[str, NDArray[np.float64]] = {}
        bus_totals: dict[str, NDArray[np.float64]] = {
            bus_id: np.zeros((CHANNEL_COUNT, frames), dtype=np.float64) for bus_id in buses
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
        for bus_id in sorted(bus_totals):
            bus = buses.get(bus_id, BusSpec(id=bus_id))
            # Band processing runs on the bus sum, before the bus gain, so a
            # bus fader stays a plain level control rather than interacting
            # with the crossover.
            processed = self._band_process(bus_id, bus_totals[bus_id])
            bus_totals[bus_id] = processed * float(db_to_linear(bus.gain_db))
            master += bus_totals[bus_id]

        def peak(block: NDArray[np.floating]) -> float:
            values = np.asarray(block)
            return float(np.max(np.abs(values))) if values.size else 0.0

        self._report = SceneMixReport(
            frames=frames,
            bus_peaks={name: peak(block) for name, block in bus_totals.items()},
            source_peaks={name: peak(block) for name, block in source_stems.items()},
            master_peak=peak(master),
            silenced=tuple(sorted(silenced)),
            disabled_bands=tuple(
                index
                for index in range(self.bands.band_count)
                if not self.bands.enabled_for(index)
            ),
            band_count=int(self.bands.band_count),
        )
        return RoutedScene(
            master=master,
            source_stems=source_stems,
            bus_stems=bus_totals,
            silenced=tuple(sorted(silenced)),
        )


def mixer_from_plan(plan, *, sample_rate_hz: float | None = None) -> SceneMixer:
    """Build a mixer from a :class:`~..plan.CompiledScenePlan`.

    The plan already carries routing, buses and band configuration - it has
    since they were compiled into it - so this is the step that turns carrying
    them into applying them.
    """

    routings = tuple(
        SourceRouting(
            source_id=str(entry.get("sourceId", source_id)),
            bus_id=str(entry.get("busId", MASTER_BUS) or MASTER_BUS),
            gain_db=float(entry.get("gainDb", 0.0)),
            muted=bool(entry.get("muted", False)),
            soloed=bool(entry.get("soloed", False)),
        )
        for source_id, entry in sorted(plan.routing.items())
    )
    buses = tuple(
        BusSpec(
            id=str(bus.get("id", MASTER_BUS)),
            name=str(bus.get("name", bus.get("id", "Bus"))),
            gain_db=float(bus.get("gainDb", 0.0)),
            muted=bool(bus.get("muted", False)),
            soloed=bool(bus.get("soloed", False)),
        )
        for bus in plan.buses
    )
    band_data = dict(plan.band_routing or {})
    bands = BandRouting(
        crossovers_hz=tuple(float(value) for value in band_data.get("crossoversHz", ())),
        band_gains_db=tuple(float(value) for value in band_data.get("bandGainsDb", ())),
        band_enabled=tuple(bool(value) for value in band_data.get("bandEnabled", ())),
        order=int(band_data.get("order", 4)),
    )
    return SceneMixer(
        routings, buses, bands,
        sample_rate_hz=float(sample_rate_hz or plan.sample_rate_hz),
    )

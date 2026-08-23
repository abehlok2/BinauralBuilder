"""An offline render as a job: a snapshot, a run, and what it cost.

Final export used to run on the GUI thread and keep the interface alive by
calling ``QApplication.processEvents()`` from inside the render. That is not a
background job. It reads and writes the live project while the user can still
edit it, so a render could pick up half of an edit; it delivers no progress a
caller can refuse; and it cannot be cancelled, because nothing is watching.

This module is the Qt-free half. A :class:`RenderSnapshot` is taken once and is
what gets rendered - the user may edit the project freely while it runs, and
the file that appears is the project as it was when they pressed the button.
The Qt worker in :mod:`src.ui.render_job` owns threads and signals and calls
into here; nothing in this file imports Qt or touches a widget.

It lives beside the canonical package rather than inside it. Orchestrating
``sound_creator.generate_audio`` is a legacy-tree concern, and
``src.audio.sam_workbench`` is not allowed to depend on that tree - the
dependency runs the other way, and ``test_clean_import`` enforces it.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "RenderCancelled",
    "RenderMetrics",
    "RenderOutcome",
    "RenderSnapshot",
    "estimate_peak_bytes",
    "estimate_render_seconds",
    "run_render",
]

from src.synth_functions.sound_creator import RenderCancelled  # noqa: E402

#: Rough audio-seconds rendered per wall-clock second, per renderer, used only
#: to estimate before a render starts. Measured throughput replaces it as soon
#: as one render has finished, so this is a starting point rather than a claim.
_THROUGHPUT_GUESS = {
    "abstract_pm": 100.0,
    "geometric": 20.0,
    "hrtf": 5.0,
    "hybrid": 4.0,
}


@dataclass(frozen=True)
class RenderSnapshot:
    """An immutable copy of everything a render reads.

    Deep-copied on the way in, so editing the project while a render is running
    cannot change what that render produces. This is the difference between a
    background job and a long function call that happens to yield.
    """

    track_data: Mapping[str, Any]
    output_path: str
    target_level: float = 0.25
    #: Free-form label for the progress UI ("Full track", "Steps 3-5").
    label: str = "render"
    #: Whether to write a reconstruction manifest beside the audio.
    write_manifest: bool = False

    @staticmethod
    def of(
        track_data: Mapping[str, Any],
        output_path: str,
        *,
        target_level: float = 0.25,
        label: str = "render",
        write_manifest: bool = False,
    ) -> "RenderSnapshot":
        return RenderSnapshot(
            track_data=copy.deepcopy(dict(track_data)),
            output_path=str(output_path),
            target_level=float(target_level),
            label=str(label),
            write_manifest=bool(write_manifest),
        )

    @property
    def sample_rate_hz(self) -> int:
        settings = self.track_data.get("global_settings") or {}
        try:
            return int(settings.get("sample_rate", 44100))
        except (TypeError, ValueError):
            return 44100

    @property
    def duration_s(self) -> float:
        total = 0.0
        for step in self.track_data.get("steps") or ():
            try:
                total += float(step.get("duration", 0.0))
            except (TypeError, ValueError):
                continue
        return total

    @property
    def renderer_ids(self) -> tuple[str, ...]:
        found: list[str] = []
        for step in self.track_data.get("steps") or ():
            for voice in step.get("voices") or ():
                params = voice.get("params") or {}
                identifier = str(params.get("rendererMode", "") or "")
                if identifier and identifier not in found:
                    found.append(identifier)
        return tuple(found)


@dataclass(frozen=True)
class RenderMetrics:
    """What a finished render actually cost."""

    duration_s: float = 0.0
    wall_s: float = 0.0
    peak_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def audio_seconds_per_second(self) -> float:
        """Above 1.0 means the render outran real time."""

        return self.duration_s / self.wall_s if self.wall_s > 0 else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "durationS": float(self.duration_s),
            "wallS": float(self.wall_s),
            "audioSecondsPerSecond": float(self.audio_seconds_per_second),
            "peakBytes": int(self.peak_bytes),
            "cacheHits": int(self.cache_hits),
            "cacheMisses": int(self.cache_misses),
        }

    def summary(self) -> str:
        if self.wall_s <= 0:
            return "not measured"
        return (
            f"{self.duration_s:.0f} s of audio in {self.wall_s:.1f} s "
            f"({self.audio_seconds_per_second:.1f}x realtime), "
            f"peak {self.peak_bytes / (1024 * 1024):.0f} MB"
        )


@dataclass(frozen=True)
class RenderOutcome:
    """How a render ended. Exactly one of succeeded/cancelled/failed is true."""

    succeeded: bool = False
    cancelled: bool = False
    error: str = ""
    output_path: str = ""
    metrics: RenderMetrics = field(default_factory=RenderMetrics)
    #: Where the reconstruction manifest went, when one was asked for.
    manifest_path: str = ""

    @property
    def failed(self) -> bool:
        return not self.succeeded and not self.cancelled


def estimate_peak_bytes(snapshot: RenderSnapshot) -> int:
    """Roughly how much memory a render of this snapshot will want.

    The track is materialized as float32 stereo, and the export spools rather
    than holding scaled copies, so the dominant term is one full-length buffer
    plus a working allowance. Deliberately an over-estimate: a warning that
    never fires is worse than one that fires slightly early.
    """

    frames = max(0, int(snapshot.duration_s * snapshot.sample_rate_hz))
    track_bytes = frames * 2 * 4
    return int(track_bytes * 1.5) + (64 << 20)


def estimate_render_seconds(snapshot: RenderSnapshot, throughput: float | None = None) -> float:
    """How long this render is likely to take, in wall-clock seconds.

    ``throughput`` is audio-seconds per wall-second measured from a previous
    render on this machine. Without one, the slowest renderer in the snapshot
    decides the guess, because the slowest is what the user will wait for.
    """

    if throughput and throughput > 0:
        return snapshot.duration_s / throughput
    rates = [_THROUGHPUT_GUESS.get(name, 10.0) for name in snapshot.renderer_ids]
    rate = min(rates) if rates else _THROUGHPUT_GUESS["abstract_pm"]
    return snapshot.duration_s / rate if rate > 0 else 0.0


def _cache_counts() -> tuple[int, int]:
    """Filter-cache hits and misses, when the HRTF cache exposes them."""

    try:
        from .sam_workbench.hrtf.interpolation import cache_statistics

        stats = cache_statistics()
        return int(stats.get("hits", 0)), int(stats.get("misses", 0))
    except Exception:
        return 0, 0


def run_render(
    snapshot: RenderSnapshot,
    *,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    measure_memory: bool = True,
) -> RenderOutcome:
    """Render ``snapshot`` to its output path and report how it went.

    Cancellation is cooperative: ``should_cancel`` is consulted wherever the
    renderer reports progress, so a cancel takes effect within a chunk of a
    long step rather than instantly.

    A cancelled or failed render leaves no file behind. A half-written export
    is worse than none: it plays, it sounds like the track, and it stops early.
    """

    from src.synth_functions import sound_creator

    destination = Path(snapshot.output_path)
    pre_existing = destination.exists()
    hits_before, misses_before = _cache_counts()

    def relay(fraction: float) -> None:
        if should_cancel is not None and should_cancel():
            raise RenderCancelled()
        if progress is not None:
            progress(max(0.0, min(1.0, float(fraction))))

    tracker = None
    if measure_memory:
        import tracemalloc

        tracemalloc.start()
        tracker = tracemalloc

    started = time.perf_counter()
    try:
        succeeded = bool(
            sound_creator.generate_audio(
                dict(snapshot.track_data),
                output_filename=snapshot.output_path,
                target_level=snapshot.target_level,
                progress_callback=relay,
            )
        )
    except RenderCancelled:
        _discard_partial(destination, pre_existing)
        return RenderOutcome(cancelled=True, output_path=snapshot.output_path)
    except Exception as error:  # noqa: BLE001 - reported to the caller verbatim
        _discard_partial(destination, pre_existing)
        return RenderOutcome(error=str(error), output_path=snapshot.output_path)
    finally:
        peak = 0
        if tracker is not None:
            _current, peak = tracker.get_traced_memory()
            tracker.stop()
        elapsed = time.perf_counter() - started

    hits_after, misses_after = _cache_counts()
    metrics = RenderMetrics(
        duration_s=snapshot.duration_s,
        wall_s=elapsed,
        peak_bytes=int(peak),
        cache_hits=max(0, hits_after - hits_before),
        cache_misses=max(0, misses_after - misses_before),
    )
    if not succeeded:
        _discard_partial(destination, pre_existing)
        return RenderOutcome(
            error="the audio engine reported failure",
            output_path=snapshot.output_path,
            metrics=metrics,
        )
    manifest_path = ""
    if snapshot.write_manifest:
        # Written after the audio, so a manifest never describes a file that
        # does not exist, and a manifest failure never loses a good render.
        try:
            from src.audio.track_manifest import build_track_manifest, write_track_manifest

            manifest = build_track_manifest(
                snapshot.track_data,
                audio_path=snapshot.output_path,
                target_level=snapshot.target_level,
                metrics=metrics.describe(),
            )
            manifest_path = str(write_track_manifest(manifest, snapshot.output_path))
        except Exception as error:  # noqa: BLE001 - reported, render still stands
            print(f"Could not write the export manifest: {error}")
    return RenderOutcome(
        succeeded=True,
        output_path=snapshot.output_path,
        metrics=metrics,
        manifest_path=manifest_path,
    )


def _discard_partial(destination: Path, pre_existing: bool) -> None:
    """Remove a file this render created and did not finish.

    A file that was already there is left alone: the user's previous export is
    not this render's to delete, and replacing it with nothing would be a worse
    outcome than a failed render.
    """

    if pre_existing:
        return
    try:
        if destination.exists():
            os.unlink(destination)
    except OSError:  # pragma: no cover - the report matters more than the tidy-up
        pass

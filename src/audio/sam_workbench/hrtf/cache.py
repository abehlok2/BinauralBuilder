"""Caching prepared HRTF datasets.

Preparing a dataset - converting coordinates, applying a delay policy,
resampling every response - takes long enough to be worth doing once. The
cache key is everything that changes the result: the asset's content hash, the
project sample rate, and the delay policy. Two preparations that differ in any
of them are different filters and never share an entry.

The in-process cache keeps a small number of prepared datasets alive so
switching between two subjects in the GUI does not re-read either. The on-disk
cache stores the prepared arrays in the application data directory, so a
restart does not pay the preparation cost again.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DELAY_POLICIES, HrtfAssetInfo, PreparedHrtf, prepare_from_path
from .sofa_io import file_sha256
from .storage import cache_root

__all__ = ["HrtfCache", "cache_key_for", "default_cache"]

#: Enough to hold an A/B comparison plus the set being auditioned.
DEFAULT_MEMORY_ENTRIES = 4


def cache_key_for(sha256: str, sample_rate_hz: int, delay_policy: str) -> str:
    """The identity of one preparation."""

    if delay_policy not in DELAY_POLICIES:
        raise ValueError(f"unknown delay policy {delay_policy!r}")
    payload = f"{sha256}|{int(sample_rate_hz)}|{delay_policy}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class HrtfCache:
    """Prepared datasets, in memory and optionally on disk."""

    memory_entries: int = DEFAULT_MEMORY_ENTRIES
    directory: Path | None = None
    _memory: "OrderedDict[str, PreparedHrtf]" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._memory is None:
            self._memory = OrderedDict()

    # --- keys ---------------------------------------------------------------

    def key_for_path(self, path: str | Path, sample_rate_hz: int, delay_policy: str) -> str:
        return cache_key_for(file_sha256(path), sample_rate_hz, delay_policy)

    def _disk_path(self, key: str) -> Path:
        return (self.directory or cache_root()) / f"{key}.npz"

    # --- access -------------------------------------------------------------

    def get(
        self,
        path: str | Path,
        *,
        sample_rate_hz: int = 44_100,
        delay_policy: str = DELAY_POLICIES[0],
        subject: str = "",
        use_disk: bool = True,
    ) -> PreparedHrtf:
        """Return a prepared dataset, preparing it only if it is not cached."""

        key = self.key_for_path(path, sample_rate_hz, delay_policy)
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            return cached

        prepared = None
        if use_disk:
            prepared = self._read_disk(key)
        if prepared is None:
            prepared = prepare_from_path(
                path, sample_rate_hz=sample_rate_hz, delay_policy=delay_policy, subject=subject
            )
            if use_disk:
                self._write_disk(key, prepared)

        self._remember(key, prepared)
        return prepared

    def _remember(self, key: str, prepared: PreparedHrtf) -> None:
        self._memory[key] = prepared
        self._memory.move_to_end(key)
        while len(self._memory) > max(1, self.memory_entries):
            self._memory.popitem(last=False)

    def clear_memory(self) -> None:
        self._memory.clear()

    # --- disk ---------------------------------------------------------------

    def _read_disk(self, key: str) -> PreparedHrtf | None:
        location = self._disk_path(key)
        if not location.is_file():
            return None
        try:
            with np.load(location, allow_pickle=False) as data:
                asset = HrtfAssetInfo(
                    path=str(data["asset_path"]),
                    sha256=str(data["asset_sha256"]),
                    convention=str(data["convention"]),
                    convention_version=str(data["convention_version"]),
                    database_name=str(data["database_name"]),
                    subject=str(data["subject"]),
                    license_note=str(data["license_note"]),
                    source_sample_rate_hz=float(data["source_sample_rate_hz"]),
                    backend=str(data["backend"]),
                )
                return PreparedHrtf(
                    directions=np.array(data["directions"]),
                    positions_m=np.array(data["positions_m"]),
                    hrirs=np.array(data["hrirs"]),
                    delays_samples=np.array(data["delays_samples"]),
                    sample_rate_hz=int(data["sample_rate_hz"]),
                    delay_policy=str(data["delay_policy"]),
                    asset=asset,
                )
        except (OSError, ValueError, KeyError):
            # A damaged cache entry is not an error worth surfacing: drop it
            # and prepare the dataset again.
            location.unlink(missing_ok=True)
            return None

    def _write_disk(self, key: str, prepared: PreparedHrtf) -> Path | None:
        location = self._disk_path(key)
        try:
            location.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                location,
                directions=prepared.directions,
                positions_m=prepared.positions_m,
                hrirs=prepared.hrirs,
                delays_samples=prepared.delays_samples,
                sample_rate_hz=np.array(prepared.sample_rate_hz),
                delay_policy=np.array(prepared.delay_policy),
                asset_path=np.array(prepared.asset.path),
                asset_sha256=np.array(prepared.asset.sha256),
                convention=np.array(prepared.asset.convention),
                convention_version=np.array(prepared.asset.convention_version),
                database_name=np.array(prepared.asset.database_name),
                subject=np.array(prepared.asset.subject),
                license_note=np.array(prepared.asset.license_note),
                source_sample_rate_hz=np.array(prepared.asset.source_sample_rate_hz),
                backend=np.array(prepared.asset.backend),
            )
            return location
        except OSError:
            # A cache that cannot be written is a slower application, not a
            # broken one.
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "memory_entries": len(self._memory),
            "memory_limit": int(self.memory_entries),
            "directory": str(self.directory or cache_root()),
        }


_DEFAULT_CACHE = HrtfCache()


def default_cache() -> HrtfCache:
    """The process-wide cache the GUI and the renderers share."""

    return _DEFAULT_CACHE

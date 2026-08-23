"""A content-addressed store for preprocessed HRTF datasets, on disk.

Loading a SOFA asset is not cheap: read the file, convert coordinates, apply
the delay policy, resample to the project's rate. The in-process cache made a
second render in one session fast and did nothing for the next session, so
previewing, closing the application and exporting paid that cost twice - and
the SONICOM files this is built for are large.

The key is what the result depends on and nothing else: the *content* of the
asset, the target sample rate, the canonical delay policy, and a format
version. A path is not part of it, so moving or copying a dataset does not
invalidate anything; an edited dataset gets a different key rather than a stale
hit, because the hash is of the bytes.

Correctness before speed, in three places:

* every entry stores the key it was written under and is refused if it does not
  match, so a truncated or hand-edited file cannot be served as something it is
  not;
* entries are written to a temporary file and renamed into place, so an
  interrupted write leaves the previous entry or nothing, never half of one;
* a read that raises for any reason is a miss. A cache that can fail a render
  is worse than no cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sofa_io import DelayPolicy, HRTFDataset

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DiskCacheStats",
    "cache_key",
    "clear_disk_cache",
    "disk_cache_statistics",
    "prune_disk_cache",
    "read_cached_dataset",
    "set_disk_cache_limit",
    "write_cached_dataset",
]

#: Bumped whenever the stored layout or the preprocessing changes. An entry
#: written by a different version is a miss, not a guess: the alternative is
#: rendering from arrays that mean something slightly different now.
CACHE_FORMAT_VERSION = 1

#: How much the store may hold before least recently used entries go. The
#: datasets this is built for run to tens of megabytes each.
_DEFAULT_LIMIT_BYTES = 2 << 30

_limit_bytes = _DEFAULT_LIMIT_BYTES
_stats = {"hits": 0, "misses": 0, "writes": 0, "rejected": 0, "evicted": 0}

#: Arrays stored for each entry.
_ARRAYS = ("ir", "delay_samples", "positions_m", "receiver_positions_m")


@dataclass(frozen=True)
class DiskCacheStats:
    """What the store has done this session, and what it holds."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    rejected: int = 0
    evicted: int = 0
    entries: int = 0
    bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "rejected": self.rejected,
            "evicted": self.evicted,
            "entries": self.entries,
            "bytes": self.bytes,
            "hitRate": self.hit_rate,
        }


def set_disk_cache_limit(limit_bytes: int | None) -> None:
    """Set how much the store may hold. ``None`` restores the default."""

    global _limit_bytes
    _limit_bytes = _DEFAULT_LIMIT_BYTES if limit_bytes is None else max(0, int(limit_bytes))


def cache_key(
    content_hash: str, sample_rate_hz: int | float | None, delay_policy: DelayPolicy | str
) -> str:
    """The identity of one preprocessed dataset.

    ``delay_policy`` goes through the enum rather than being used as written, so
    a document saying ``preserve_external_delay`` and a registry saying
    ``keep_external_delay`` land on one entry instead of two copies of the same
    arrays.
    """

    rate = "source" if sample_rate_hz is None else str(int(sample_rate_hz))
    material = " ".join(
        (
            str(CACHE_FORMAT_VERSION),
            str(content_hash),
            rate,
            DelayPolicy(delay_policy).value,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _entry_path(root: Path, key: str) -> Path:
    # Two characters of fan-out: a flat directory of thousands of entries is
    # slow to list on some filesystems, and listing is what pruning does.
    return root / key[:2] / (key + ".npz")


def _cache_root() -> Path:
    from .storage import cache_root

    return cache_root() / "datasets"


def read_cached_dataset(
    key: str, path: Path, *, source_sample_rate_hz: float | None = None
) -> HRTFDataset | None:
    """Return the stored dataset for ``key``, or ``None`` for any reason at all.

    ``path`` is the asset this stands in for. It is not part of the key, so it
    is supplied rather than stored: the same dataset reached through two paths
    is one entry.
    """

    entry = _entry_path(_cache_root(), key)
    try:
        if not entry.exists():
            _stats["misses"] += 1
            return None
        with np.load(entry, allow_pickle=False) as stored:
            if str(stored["key"]) != key:
                # Not what it claims to be. Refuse it, and say so in the counts.
                _stats["rejected"] += 1
                _stats["misses"] += 1
                return None
            meta = json.loads(str(stored["meta"]))
            arrays = {name: np.asarray(stored[name], dtype=np.float64) for name in _ARRAYS}
        if arrays["ir"].ndim != 3:
            _stats["rejected"] += 1
            _stats["misses"] += 1
            return None

        dataset = HRTFDataset(
            path=Path(path),
            content_hash=str(meta["content_hash"]),
            convention=str(meta["convention"]),
            convention_version=str(meta["convention_version"]),
            ir=arrays["ir"],
            sample_rate_hz=float(meta["sample_rate_hz"]),
            delay_samples=arrays["delay_samples"],
            positions_m=arrays["positions_m"],
            receiver_positions_m=arrays["receiver_positions_m"],
            coordinate_type=str(meta["coordinate_type"]),
            coordinate_units=str(meta["coordinate_units"]),
            metadata=dict(meta.get("metadata") or {}),
            delay_policy=DelayPolicy(meta["delay_policy"]),
            source_sample_rate_hz=float(
                meta.get("source_sample_rate_hz", source_sample_rate_hz or 0.0)
            ),
        )
    except Exception:  # noqa: BLE001 - a cache must never be able to fail a render
        _stats["rejected"] += 1
        _stats["misses"] += 1
        return None

    # Touch it so pruning sees it as recently used.
    try:
        os.utime(entry, None)
    except OSError:  # pragma: no cover - a read-only store still reads
        pass
    _stats["hits"] += 1
    return dataset


def write_cached_dataset(key: str, dataset: HRTFDataset) -> bool:
    """Store ``dataset`` under ``key``. Returns whether it was written.

    A failure here is reported and otherwise ignored: being unable to cache
    something is not a reason to fail the render that produced it.
    """

    entry = _entry_path(_cache_root(), key)
    meta = {
        "content_hash": dataset.content_hash,
        "convention": dataset.convention,
        "convention_version": dataset.convention_version,
        "sample_rate_hz": float(dataset.sample_rate_hz),
        "coordinate_type": dataset.coordinate_type,
        "coordinate_units": dataset.coordinate_units,
        "metadata": _plain(dataset.metadata),
        "delay_policy": DelayPolicy(dataset.delay_policy).value,
        "source_sample_rate_hz": float(dataset.source_sample_rate_hz),
    }
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        # Written beside its destination and renamed into place: a rename
        # within one filesystem is atomic, so a crash mid-write leaves the
        # previous entry or nothing, rather than a truncated file that would
        # later be read as a dataset.
        handle, temporary = tempfile.mkstemp(dir=str(entry.parent), suffix=".partial")
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            with open(temporary_path, "wb") as stream:
                np.savez(
                    stream,
                    key=np.asarray(key),
                    meta=np.asarray(json.dumps(meta)),
                    ir=np.asarray(dataset.ir, dtype=np.float64),
                    delay_samples=np.asarray(dataset.delay_samples, dtype=np.float64),
                    positions_m=np.asarray(dataset.positions_m, dtype=np.float64),
                    receiver_positions_m=np.asarray(
                        dataset.receiver_positions_m, dtype=np.float64
                    ),
                )
            os.replace(temporary_path, entry)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    except Exception as error:  # noqa: BLE001 - reported, never raised
        print("Could not cache the prepared HRTF dataset: " + str(error))
        return False

    _stats["writes"] += 1
    prune_disk_cache()
    return True


def _plain(value: Any) -> Any:
    """Reduce metadata to what JSON can carry, rather than failing on it."""

    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _entries(root: Path) -> list[tuple[float, int, Path]]:
    found: list[tuple[float, int, Path]] = []
    if not root.exists():
        return found
    for entry in root.rglob("*.npz"):
        try:
            stat = entry.stat()
        except OSError:  # pragma: no cover - vanished between listing and stat
            continue
        found.append((stat.st_mtime, stat.st_size, entry))
    return found


def prune_disk_cache(limit_bytes: int | None = None) -> int:
    """Drop least-recently-used entries until the store is within its limit.

    Returns how many were removed.
    """

    limit = _limit_bytes if limit_bytes is None else max(0, int(limit_bytes))
    found = _entries(_cache_root())
    total = sum(size for _mtime, size, _path in found)
    if total <= limit:
        return 0

    removed = 0
    for _mtime, size, entry in sorted(found):  # oldest touched first
        if total <= limit:
            break
        try:
            entry.unlink()
        except OSError:  # pragma: no cover
            continue
        total -= size
        removed += 1
        _stats["evicted"] += 1
    return removed


def clear_disk_cache() -> None:
    """Remove every entry and reset the counters."""

    for _mtime, _size, entry in _entries(_cache_root()):
        try:
            entry.unlink()
        except OSError:  # pragma: no cover
            continue
    for name in _stats:
        _stats[name] = 0


def disk_cache_statistics() -> DiskCacheStats:
    """What the store has done this session, and what it currently holds."""

    found = _entries(_cache_root())
    return DiskCacheStats(
        entries=len(found),
        bytes=sum(size for _mtime, size, _path in found),
        **{name: int(value) for name, value in _stats.items()},
    )

"""Bounded caches for immutable preprocessed HRTF datasets, in memory and on disk.

Two tiers, because they answer different questions. The in-memory tier keeps a
handful of datasets for the renders happening now. The disk tier keeps them
across restarts, so previewing a project, closing the application and exporting
it does not pay the same preparation twice.

A miss in memory falls through to disk before the file is read and prepared
again; a hit on disk is promoted into memory. A disk failure of any kind is
just a miss.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock, RLock

from .sofa_io import DelayPolicy, HRTFDataset, hash_asset, load_sofa, resolve_sofa_path


class HRTFCache:
    def __init__(self, max_entries: int = 4, *, use_disk: bool = True):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        #: Whether a memory miss may be served from, and written to, the disk
        #: store. Off is what a test wants when it is measuring the loader.
        self.use_disk = bool(use_disk)
        self._items: OrderedDict[tuple, HRTFDataset] = OrderedDict()
        self._lock = RLock()
        #: One lock per key being loaded. Preparing a dataset is seconds of
        #: work, and preview and export now render at the same time, so two
        #: threads missing on one asset would otherwise both read, convert,
        #: resample and write it. The second waits and takes the first's
        #: result instead. Keyed rather than global so a load of one asset
        #: does not hold up a different one.
        self._loading: dict[tuple, Lock] = {}

    def get(self, path: str | Path, sample_rate_hz: int, delay_policy: DelayPolicy | str,
            project_directory: str | Path | None = None) -> HRTFDataset:
        resolved = resolve_sofa_path(path, project_directory)
        stat = resolved.stat()
        key = (str(resolved), stat.st_mtime_ns, stat.st_size, int(sample_rate_hz), DelayPolicy(delay_policy).value)
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                return found
        # Content-addressed rather than path-addressed, so the key needs the
        # asset's bytes. Hashing a large SOFA file is far cheaper than parsing,
        # converting, applying the delay policy and resampling it - and the
        # digest is handed to the loader below rather than recomputed there.
        from .disk_cache import cache_key, read_cached_dataset, write_cached_dataset

        digest = None
        disk_key = None
        if self.use_disk:
            try:
                digest = hash_asset(resolved)
                disk_key = cache_key(digest, sample_rate_hz, delay_policy)
                stored = read_cached_dataset(disk_key, resolved)
            except Exception:  # noqa: BLE001 - the disk tier must not fail a load
                stored = None
            if stored is not None:
                with self._lock:
                    self._items[key] = stored
                    self._items.move_to_end(key)
                    while len(self._items) > self.max_entries:
                        self._items.popitem(last=False)
                return stored

        with self._lock:
            gate = self._loading.get(key)
            if gate is None:
                gate = self._loading[key] = Lock()

        with gate:
            # Another thread may have finished this exact load while this one
            # waited, in which case its result is the answer.
            with self._lock:
                found = self._items.get(key)
                if found is not None:
                    self._items.move_to_end(key)
                    return found

            try:
                loaded = load_sofa(
                    resolved,
                    target_sample_rate_hz=sample_rate_hz,
                    delay_policy=delay_policy,
                    content_hash=digest,
                )
                if self.use_disk and disk_key is not None:
                    write_cached_dataset(disk_key, loaded)
                with self._lock:
                    self._items[key] = loaded
                    self._items.move_to_end(key)
                    while len(self._items) > self.max_entries:
                        self._items.popitem(last=False)
            finally:
                # Dropped whether or not the load worked: a failed one must
                # leave nothing behind that a later attempt has to step over.
                with self._lock:
                    self._loading.pop(key, None)
            return loaded

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


default_hrtf_cache = HRTFCache()

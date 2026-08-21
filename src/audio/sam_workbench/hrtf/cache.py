"""Thread-safe bounded cache for immutable preprocessed HRTF datasets."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock

from .sofa_io import DelayPolicy, HRTFDataset, load_sofa, resolve_sofa_path


class HRTFCache:
    def __init__(self, max_entries: int = 4):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._items: OrderedDict[tuple, HRTFDataset] = OrderedDict()
        self._lock = RLock()

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
        loaded = load_sofa(resolved, target_sample_rate_hz=sample_rate_hz, delay_policy=delay_policy)
        with self._lock:
            self._items[key] = loaded
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
        return loaded

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


default_hrtf_cache = HRTFCache()

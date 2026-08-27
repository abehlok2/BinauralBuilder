"""Preparing a SOFA asset once, not once per session.

The in-process cache made a second render in one session fast and did nothing
for the next one, so previewing a project, closing the application and
exporting it paid the same preparation twice. This store persists it.

A cache is only worth having if it cannot be wrong, so most of what is asserted
here is refusal: a truncated entry, an entry under the wrong key, an
unreadable store - each must be a miss and a re-derivation rather than a crash
or, worse, a dataset that is not the one asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.hrtf import storage
from src.audio.sam_workbench.hrtf.cache import HRTFCache
from src.audio.sam_workbench.hrtf.disk_cache import (
    CACHE_FORMAT_VERSION,
    cache_key,
    clear_disk_cache,
    disk_cache_statistics,
    prune_disk_cache,
    read_cached_dataset,
    set_disk_cache_limit,
    write_cached_dataset,
)
from src.audio.sam_workbench.hrtf.sofa_io import DelayPolicy, load_sofa

FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")
LARGER = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_sonicom_hrir.sofa")


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Keep every test's entries out of the developer's real cache."""

    storage.set_data_root(tmp_path)
    clear_disk_cache()
    set_disk_cache_limit(None)
    yield tmp_path
    set_disk_cache_limit(None)
    storage.set_data_root(None)


def _entries(root):
    return sorted((root / "hrtf" / "cache" / "datasets").rglob("*.npz"))


# --- the key -----------------------------------------------------------------


def test_the_key_is_the_content_not_the_path():
    """Moving or copying a dataset must not throw the preparation away."""

    first = cache_key("abc123", 44100, "bake_delay_into_ir")
    second = cache_key("abc123", 44100, "bake_delay_into_ir")
    assert first == second


def test_different_content_is_a_different_key():
    assert cache_key("abc123", 44100, "bake_delay_into_ir") != cache_key(
        "def456", 44100, "bake_delay_into_ir"
    )


def test_the_sample_rate_is_part_of_the_key():
    assert cache_key("abc", 44100, "bake_delay_into_ir") != cache_key(
        "abc", 48000, "bake_delay_into_ir"
    )


def test_the_delay_policy_is_part_of_the_key():
    assert cache_key("abc", 44100, "bake_delay_into_ir") != cache_key(
        "abc", 44100, "keep_external_delay"
    )


def test_a_renamed_policy_does_not_get_a_second_entry():
    """The alias and the canonical name are one policy, so one key.

    Two members meaning the same thing would store the same arrays twice and
    halve the hit rate for anyone with older documents.
    """

    assert cache_key("abc", 44100, "preserve_external_delay") == cache_key(
        "abc", 44100, "keep_external_delay"
    )


def test_the_format_version_is_part_of_the_key():
    """An entry written by another layout must be a miss, not a guess."""

    import src.audio.sam_workbench.hrtf.disk_cache as module

    original = module.CACHE_FORMAT_VERSION
    key_now = cache_key("abc", 44100, "bake_delay_into_ir")
    module.CACHE_FORMAT_VERSION = original + 1
    try:
        assert cache_key("abc", 44100, "bake_delay_into_ir") != key_now
    finally:
        module.CACHE_FORMAT_VERSION = original


# --- round trip ---------------------------------------------------------------


def test_a_stored_dataset_comes_back_the_same():
    dataset = load_sofa(FIXTURE, target_sample_rate_hz=44100)
    key = cache_key(dataset.content_hash, 44100, dataset.delay_policy)

    assert write_cached_dataset(key, dataset) is True
    restored = read_cached_dataset(key, Path(FIXTURE))

    assert restored is not None
    assert np.array_equal(restored.ir, dataset.ir)
    assert np.array_equal(restored.delay_samples, dataset.delay_samples)
    assert np.array_equal(restored.positions_m, dataset.positions_m)
    assert np.array_equal(restored.receiver_positions_m, dataset.receiver_positions_m)
    assert restored.content_hash == dataset.content_hash
    assert restored.sample_rate_hz == pytest.approx(dataset.sample_rate_hz)
    assert restored.delay_policy is dataset.delay_policy
    assert restored.convention == dataset.convention
    assert restored.coordinate_type == dataset.coordinate_type


def test_the_path_comes_from_the_caller_not_the_entry():
    """One entry serves the same dataset reached through any path."""

    dataset = load_sofa(FIXTURE)
    key = cache_key(dataset.content_hash, None, dataset.delay_policy)
    write_cached_dataset(key, dataset)

    restored = read_cached_dataset(key, Path("/somewhere/else/copy.sofa"))
    assert restored is not None
    assert restored.path == Path("/somewhere/else/copy.sofa")


def test_metadata_that_json_cannot_carry_does_not_stop_a_write():
    dataset = load_sofa(FIXTURE)
    dataset.metadata["an_array"] = np.arange(3)
    dataset.metadata["an_object"] = object()
    key = cache_key("with-odd-metadata", None, dataset.delay_policy)

    assert write_cached_dataset(key, dataset) is True
    assert read_cached_dataset(key, Path(FIXTURE)) is not None


# --- refusal ------------------------------------------------------------------


def test_a_missing_entry_is_a_miss():
    assert read_cached_dataset(cache_key("nothing", 44100, "bake_delay_into_ir"),
                               Path(FIXTURE)) is None
    assert disk_cache_statistics().misses == 1


def test_a_truncated_entry_is_a_miss_rather_than_a_crash(isolated_store):
    dataset = load_sofa(FIXTURE)
    key = cache_key(dataset.content_hash, None, dataset.delay_policy)
    write_cached_dataset(key, dataset)

    entry = _entries(isolated_store)[0]
    entry.write_bytes(entry.read_bytes()[: len(entry.read_bytes()) // 2])

    assert read_cached_dataset(key, Path(FIXTURE)) is None
    assert disk_cache_statistics().rejected >= 1


def test_an_entry_that_is_not_what_it_claims_is_refused(isolated_store):
    """A file put where another key belongs must not be served as that key."""

    dataset = load_sofa(FIXTURE)
    honest = cache_key(dataset.content_hash, None, dataset.delay_policy)
    write_cached_dataset(honest, dataset)

    stored = _entries(isolated_store)[0]
    impostor_key = cache_key("something-else", 44100, "bake_delay_into_ir")
    from src.audio.sam_workbench.hrtf.disk_cache import _entry_path, _cache_root

    target = _entry_path(_cache_root(), impostor_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(stored.read_bytes())

    assert read_cached_dataset(impostor_key, Path(FIXTURE)) is None
    assert disk_cache_statistics().rejected >= 1


def test_garbage_in_place_of_an_entry_is_a_miss(isolated_store):
    key = cache_key("garbage", 44100, "bake_delay_into_ir")
    from src.audio.sam_workbench.hrtf.disk_cache import _entry_path, _cache_root

    entry = _entry_path(_cache_root(), key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("not an npz file at all")

    assert read_cached_dataset(key, Path(FIXTURE)) is None


def test_a_store_that_cannot_be_written_does_not_fail_the_caller(monkeypatch):
    dataset = load_sofa(FIXTURE)
    monkeypatch.setattr(
        "src.audio.sam_workbench.hrtf.disk_cache.tempfile.mkstemp",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")),
    )
    assert write_cached_dataset(cache_key("x", None, "bake_delay_into_ir"), dataset) is False


def test_an_interrupted_write_leaves_nothing_rather_than_half(isolated_store, monkeypatch):
    """A partial file would later be read as a dataset."""

    dataset = load_sofa(FIXTURE)
    monkeypatch.setattr(
        "src.audio.sam_workbench.hrtf.disk_cache.np.savez",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    key = cache_key(dataset.content_hash, None, dataset.delay_policy)

    assert write_cached_dataset(key, dataset) is False
    assert _entries(isolated_store) == []
    # And no leftover temporary either.
    leftovers = list((isolated_store / "hrtf" / "cache").rglob("*.partial"))
    assert leftovers == []


# --- bounds -------------------------------------------------------------------


def test_the_store_is_pruned_to_its_limit(isolated_store):
    dataset = load_sofa(FIXTURE)
    for index in range(5):
        write_cached_dataset(cache_key(f"entry-{index}", 44100, "bake_delay_into_ir"), dataset)
    assert len(_entries(isolated_store)) == 5

    one_entry = _entries(isolated_store)[0].stat().st_size
    removed = prune_disk_cache(limit_bytes=one_entry * 2)

    assert removed >= 3
    assert len(_entries(isolated_store)) <= 2


def test_pruning_keeps_the_most_recently_used(isolated_store):
    import os
    import time

    dataset = load_sofa(FIXTURE)
    keys = [cache_key(f"lru-{index}", 44100, "bake_delay_into_ir") for index in range(3)]
    for key in keys:
        write_cached_dataset(key, dataset)

    entries = {key: None for key in keys}
    from src.audio.sam_workbench.hrtf.disk_cache import _entry_path, _cache_root

    now = time.time()
    for age, key in enumerate(keys):
        path = _entry_path(_cache_root(), key)
        entries[key] = path
        os.utime(path, (now - 1000 * (len(keys) - age), now - 1000 * (len(keys) - age)))

    size = entries[keys[0]].stat().st_size
    prune_disk_cache(limit_bytes=size)

    # The oldest went first; the newest survived.
    assert not entries[keys[0]].exists()
    assert entries[keys[-1]].exists()


def test_clearing_removes_everything_and_resets_the_counts(isolated_store):
    dataset = load_sofa(FIXTURE)
    write_cached_dataset(cache_key("a", None, "bake_delay_into_ir"), dataset)
    assert disk_cache_statistics().entries == 1

    clear_disk_cache()
    stats = disk_cache_statistics()
    assert stats.entries == 0
    assert stats.writes == 0 and stats.hits == 0


def test_the_statistics_describe_themselves():
    described = disk_cache_statistics().describe()
    assert json.loads(json.dumps(described))["hitRate"] == 0.0


# --- through the loader -------------------------------------------------------


def test_a_second_session_does_not_prepare_the_asset_again():
    """The whole point: a fresh cache object is what a new process looks like."""

    first = HRTFCache().get(LARGER, 48000, "keep_external_delay")
    after_cold = disk_cache_statistics()
    assert after_cold.writes == 1

    second = HRTFCache().get(LARGER, 48000, "keep_external_delay")
    after_warm = disk_cache_statistics()

    assert after_warm.hits == 1
    assert np.array_equal(first.ir, second.ir)
    assert np.array_equal(first.delay_samples, second.delay_samples)
    assert second.delay_policy is DelayPolicy.KEEP


def test_the_alias_spelling_hits_the_entry_the_canonical_one_wrote():
    HRTFCache().get(LARGER, 48000, "keep_external_delay")
    entries_before = disk_cache_statistics().entries

    HRTFCache().get(LARGER, 48000, "preserve_external_delay")

    assert disk_cache_statistics().entries == entries_before
    assert disk_cache_statistics().hits >= 1


def test_a_cache_told_not_to_use_the_disk_does_not_touch_it(isolated_store):
    HRTFCache(use_disk=False).get(FIXTURE, 44100, "bake_delay_into_ir")
    assert _entries(isolated_store) == []


def test_an_edited_asset_is_not_served_from_the_old_entry(tmp_path):
    """The hash is of the bytes, so an edit cannot be a stale hit."""

    original = Path(LARGER).read_bytes()
    copy = tmp_path / "asset.sofa"
    copy.write_bytes(original)

    first = HRTFCache().get(str(copy), 48000, "bake_delay_into_ir")

    # Any change at all to the file's bytes.
    copy.write_bytes(original[:-1])
    try:
        second = HRTFCache().get(str(copy), 48000, "bake_delay_into_ir")
    except Exception:
        # A deliberately corrupted file may simply fail to load; what must not
        # happen is the previous entry being returned for it.
        return
    assert second.content_hash != first.content_hash


def test_a_disk_failure_still_loads_the_asset(monkeypatch):
    monkeypatch.setattr(
        "src.audio.sam_workbench.hrtf.disk_cache.read_cached_dataset",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("store unreadable")),
    )
    dataset = HRTFCache().get(FIXTURE, 44100, "bake_delay_into_ir")
    assert dataset.ir.size > 0


# --- hashing the asset without holding it -----------------------------------


def test_the_chunked_hash_is_the_hash_of_the_whole_file():
    import hashlib

    from src.audio.sam_workbench.hrtf.sofa_io import hash_asset

    asset = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"
    assert hash_asset(asset) == hashlib.sha256(asset.read_bytes()).hexdigest()


def test_hashing_an_asset_does_not_hold_it_in_memory(tmp_path):
    """A SONICOM asset is large enough that reading it whole to hash it is a
    transient allocation its own size - on the path whose purpose is bounding
    what a long render holds."""

    import tracemalloc

    from src.audio.sam_workbench.hrtf.sofa_io import hash_asset

    asset = tmp_path / "large.bin"
    size = 32 << 20
    asset.write_bytes(b"\x00" * size)

    tracemalloc.start()
    try:
        hash_asset(asset)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < size // 4


def test_a_supplied_digest_is_used_rather_than_read_again(monkeypatch):
    """The cache has already hashed the bytes by the time it calls the loader."""

    from src.audio.sam_workbench.hrtf import sofa_io

    asset = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"
    calls = []
    real = sofa_io.hash_asset
    monkeypatch.setattr(
        sofa_io, "hash_asset", lambda path: (calls.append(path), real(path))[1]
    )

    dataset = sofa_io.load_sofa(asset, content_hash="supplied-digest")

    assert dataset.content_hash == "supplied-digest"
    assert calls == []


def test_two_threads_missing_at_once_load_the_asset_only_once(monkeypatch, tmp_path):
    """Preview and export render at the same time now, so this is reachable.

    Preparing a dataset is seconds of work. Both threads used to do all of it.
    """

    import threading

    from src.audio.sam_workbench.hrtf import cache as cache_module

    asset = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"
    loads = []
    started = threading.Barrier(2)
    real = cache_module.load_sofa

    def counted(*args, **kwargs):
        loads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache_module, "load_sofa", counted)
    subject = cache_module.HRTFCache(use_disk=False)
    results = {}

    def load(name):
        started.wait(timeout=10)
        results[name] = subject.get(asset, 44100, "bake_delay_into_ir")

    threads = [threading.Thread(target=load, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert len(loads) == 1
    assert results["a"] is results["b"]


def test_a_failed_load_does_not_wedge_later_ones(monkeypatch):
    """A load that raises must not leave the gate held for the session."""

    from src.audio.sam_workbench.hrtf import cache as cache_module

    asset = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"
    subject = cache_module.HRTFCache(use_disk=False)

    def explode(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(cache_module, "load_sofa", explode)
    with pytest.raises(RuntimeError):
        subject.get(asset, 44100, "bake_delay_into_ir")

    monkeypatch.undo()
    assert subject.get(asset, 44100, "bake_delay_into_ir") is not None

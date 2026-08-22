"""Phase 8 exit criteria: recovery, migration, and long renders.

The specification's release gate is short and specific - no known data-loss or
real-time safety defects, and project migration, crash recovery and long
renders tested. Each of those is a way of losing someone's work, so each is
checked here rather than argued about.

The rules this file protects:

* an autosave is never the user's save, and never writes over their project;
* a snapshot older than the saved file is not offered, because accepting it
  would replace good work with stale work;
* a declined snapshot is gone, so the prompt keeps meaning something;
* a document written before the schema carried a version still opens, and
  survives a save and a reload unchanged;
* a long render is the same audio however it was blocked, and does not drift.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.export import project_sha256
from src.audio.sam_workbench.migrations import UNVERSIONED, migrate_project_dict
from src.audio.sam_workbench.model import (
    Project,
    SamModulationSpec,
    SignalSpec,
    Source,
    load_project,
    save_project,
)
from src.audio.sam_workbench.recovery import (
    DEFAULT_AUTOSAVE_SECONDS,
    RecoverySnapshot,
    RecoveryStore,
    snapshot_path_for,
)
from src.audio.sam_workbench.render import render_project

SAMPLE_RATE = 44_100


def _project(name: str = "Release test") -> Project:
    source = Source(
        id="carrier",
        amplitude_linear=0.4,
        signal=SignalSpec(carrier_frequency_hz=220.0),
        sam=SamModulationSpec(rate_hz=4.0, depth_rad=1.0),
    )
    return Project(name=name, sources=(source,))


# --- crash recovery ---------------------------------------------------------


def test_a_snapshot_never_touches_the_project(tmp_path):
    project_path = tmp_path / "session.json"
    project_path.write_text('{"schema_version": "1.0", "name": "saved"}', encoding="utf-8")
    before = project_path.read_text(encoding="utf-8")

    store = RecoveryStore(project_path)
    store.save({"name": "edited but not saved"}, reason="autosave")

    assert project_path.read_text(encoding="utf-8") == before
    assert store.path != project_path
    assert store.path.parent == project_path.parent


def test_a_newer_snapshot_is_offered_and_an_older_one_is_not(tmp_path):
    project_path = tmp_path / "session.json"
    project_path.write_text("{}", encoding="utf-8")

    store = RecoveryStore(project_path)
    store.save({"steps": [1, 2, 3]})
    offered = store.offer()
    assert offered is not None
    assert offered.payload["steps"] == [1, 2, 3]

    # Saving the project makes the snapshot stale: it describes work the user
    # has already superseded.
    time.sleep(0.01)
    os.utime(project_path, None)
    assert store.offer() is None
    # Stale, not deleted - the file is still there to inspect.
    assert store.read() is not None


def test_declining_a_recovery_removes_it(tmp_path):
    store = RecoveryStore(tmp_path / "session.json")
    store.save({"a": 1})
    assert store.exists()
    assert store.discard() is True
    assert store.exists() is False
    assert store.discard() is False
    assert store.offer() is None


def test_saving_under_a_new_name_moves_the_snapshot_with_it(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    store = RecoveryStore(first)
    store.save({"a": 1})
    assert snapshot_path_for(first).exists()

    store.adopt(second)
    assert snapshot_path_for(first).exists() is False
    assert store.project_path == second


def test_work_that_was_never_saved_is_still_recoverable(tmp_path):
    store = RecoveryStore(None, fallback_directory=tmp_path)
    store.save({"steps": []}, reason="autosave")
    # Nothing to compare against, so anything present is worth offering.
    assert store.offer() is not None
    assert store.path.parent == tmp_path


def test_a_damaged_snapshot_is_treated_as_absent(tmp_path):
    """A convenience file must not be able to stop a project from opening."""

    store = RecoveryStore(tmp_path / "session.json")
    store.save({"a": 1})
    store.path.write_text("{ this is not json", encoding="utf-8")
    assert store.read() is None
    assert store.offer() is None

    store.save({"a": 1})
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["schemaVersion"] = 99
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.read() is None


def test_a_snapshot_is_written_whole_or_not_at_all(tmp_path):
    store = RecoveryStore(tmp_path / "session.json")
    store.save({"steps": list(range(500))})
    # No partial file left beside it, and what is there parses.
    leftovers = [path.name for path in tmp_path.iterdir() if path.name.endswith(".partial")]
    assert leftovers == []
    assert json.loads(store.path.read_text(encoding="utf-8"))["payload"]["steps"][-1] == 499


def test_a_snapshot_says_when_and_why_it_was_taken():
    snapshot = RecoverySnapshot(payload={}, saved_utc=0.0, reason="before rendering")
    assert "1970" in snapshot.summary()
    assert "before rendering" in snapshot.summary()
    assert DEFAULT_AUTOSAVE_SECONDS > 0.0


# --- project migration ------------------------------------------------------


def test_a_pre_versioned_document_opens_and_survives_a_round_trip(tmp_path):
    """The oldest documents must keep working, unchanged where it matters."""

    legacy = {"name": "legacy", "sources": []}
    assert "schema_version" not in legacy
    migrated = migrate_project_dict(legacy)
    assert migrated["schema_version"] == "1.0"
    # The original is not mutated: callers keep their own copy.
    assert "schema_version" not in legacy

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    opened = load_project(path)
    save_project(opened, path)
    reopened = load_project(path)
    assert project_sha256(opened) == project_sha256(reopened)


def test_migration_preserves_fields_it_does_not_understand(tmp_path):
    """An unknown key belongs to something; deleting it loses that thing."""

    path = tmp_path / "extended.json"
    payload = {
        "name": "extended",
        "sources": [],
        "somethingFromAnExtension": {"kept": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    opened = load_project(path)
    save_project(opened, path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["somethingFromAnExtension"] == {"kept": True}


def test_the_unversioned_marker_is_what_migration_starts_from():
    assert UNVERSIONED == "0"


# --- long renders -----------------------------------------------------------


@pytest.mark.parametrize("block_size", [128, 512, 4096])
def test_a_long_render_is_the_same_audio_however_it_was_blocked(block_size):
    """Section 14.3: block size must not change the result.

    Sixty seconds is long enough for a phase error of one sample per block to
    show: at 128-sample blocks that is twenty thousand opportunities to drift.
    """

    project = _project()
    reference, _ = render_project(project, 60.0, block_size=SAMPLE_RATE)
    blocked, _ = render_project(project, 60.0, block_size=block_size)

    assert blocked.shape == reference.shape
    assert np.max(np.abs(blocked - reference)) < 1e-9


def test_a_long_render_does_not_drift_towards_its_end():
    """A phase error accumulates, so the end is where it would be visible."""

    project = _project()
    reference, _ = render_project(project, 60.0, block_size=SAMPLE_RATE)
    blocked, _ = render_project(project, 60.0, block_size=257)

    tail = slice(-SAMPLE_RATE, None)
    head = slice(0, SAMPLE_RATE)
    assert np.max(np.abs(blocked[:, tail] - reference[:, tail])) < 1e-9
    # And the error at the end is no larger than at the start, which is what
    # "does not drift" means.
    assert np.max(np.abs(blocked[:, tail] - reference[:, tail])) <= max(
        1e-12, 10.0 * np.max(np.abs(blocked[:, head] - reference[:, head]))
    )


def test_a_long_render_is_reproducible_and_stays_within_the_output_ceiling():
    project = _project()
    first, report = render_project(project, 30.0)
    second, _ = render_project(project, 30.0)

    assert np.array_equal(first, second)
    assert first.shape == (2, int(round(30.0 * SAMPLE_RATE)))
    assert np.all(np.isfinite(first))
    # The limiter's ceiling is the promise the export makes about peak level.
    assert np.max(np.abs(first)) <= 1.0
    assert report is not None


def test_rendering_from_an_absolute_position_matches_the_whole_render():
    """A resumed or chunked long render must agree with the continuous one."""

    project = _project()
    whole, _ = render_project(project, 20.0)
    offset = 10 * SAMPLE_RATE
    later, _ = render_project(
        project, frames=5 * SAMPLE_RATE, start_sample=offset, block_size=512
    )
    assert np.max(np.abs(later - whole[:, offset : offset + 5 * SAMPLE_RATE])) < 1e-9

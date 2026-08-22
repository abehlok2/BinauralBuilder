"""Autosave snapshots, and offering them back after a crash.

The specification is precise about what an autosave is and is not: a recovery
snapshot written on a timer and before long jobs, which is *not* the user's
explicit save, and which is offered back on startup only when it is newer than
the project it belongs to. Every one of those clauses is here because the
obvious implementations get it wrong in a way that loses work.

Writing a snapshot over the project would make autosave into a save, and would
commit edits the user was still deciding about. Offering an older snapshot
would invite someone to overwrite a good file with stale work. Offering a
snapshot the user has already declined, every time they open the project, would
train them to dismiss the prompt without reading it.

So snapshots live beside the project under a separate name, carry the time and
the origin they were taken from, and are compared against the project's own
modification time before they are offered. Discarding one deletes it.

The HRTF cache is deliberately not covered here. It is regenerable, and the
specification is explicit that it must never be the only copy of a derived SOFA
artefact - a derived file is written where the user asked for it, not into a
cache that a cleanup could remove.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .version import PACKAGE_VERSION

__all__ = [
    "DEFAULT_AUTOSAVE_SECONDS",
    "RecoverySnapshot",
    "RecoveryStore",
    "snapshot_path_for",
]

#: How often to take a snapshot while editing. Frequent enough that a crash
#: costs a minute rather than an afternoon, rare enough not to be noticed.
DEFAULT_AUTOSAVE_SECONDS = 60.0

_SUFFIX = ".recovery.json"
_SCHEMA_VERSION = 1


def snapshot_path_for(project_path: str | os.PathLike[str]) -> Path:
    """Where the snapshot for a project lives: beside it, under its own name."""

    path = Path(project_path)
    return path.with_name(f".{path.name}{_SUFFIX}")


@dataclass(frozen=True)
class RecoverySnapshot:
    """A saved editing state, and enough context to judge whether to offer it."""

    payload: Mapping[str, Any]
    saved_utc: float
    origin: str = ""
    reason: str = ""
    application_version: str = PACKAGE_VERSION

    @property
    def saved_at(self) -> time.struct_time:
        return time.gmtime(self.saved_utc)

    def describe(self) -> dict[str, Any]:
        return {
            "schemaVersion": _SCHEMA_VERSION,
            "application": "BinauralBuilder SAM workbench",
            "applicationVersion": self.application_version,
            "savedUtc": float(self.saved_utc),
            "origin": str(self.origin),
            "reason": str(self.reason),
            "payload": dict(self.payload),
        }

    def summary(self) -> str:
        """One line for the prompt, naming the thing that matters: when."""

        stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", self.saved_at)
        detail = f" ({self.reason})" if self.reason else ""
        return f"Unsaved work from {stamp}{detail}"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecoverySnapshot":
        version = int(data.get("schemaVersion", _SCHEMA_VERSION))
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported recovery snapshot version {version}")
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("a recovery snapshot must carry a payload object")
        return cls(
            payload=dict(payload),
            saved_utc=float(data.get("savedUtc", 0.0)),
            origin=str(data.get("origin", "")),
            reason=str(data.get("reason", "")),
            application_version=str(data.get("applicationVersion", PACKAGE_VERSION)),
        )


class RecoveryStore:
    """Writes, offers, and clears the snapshot belonging to one project.

    ``project_path`` may be ``None`` for a project that has never been saved;
    the snapshot then lives in ``fallback_directory`` under a fixed name, so
    unsaved work still survives a crash even though there is nothing to compare
    it against.
    """

    def __init__(
        self,
        project_path: str | os.PathLike[str] | None = None,
        *,
        fallback_directory: str | os.PathLike[str] | None = None,
    ) -> None:
        self.project_path = Path(project_path) if project_path is not None else None
        self.fallback_directory = (
            Path(fallback_directory) if fallback_directory is not None else None
        )

    # --- location -----------------------------------------------------------

    @property
    def path(self) -> Path:
        if self.project_path is not None:
            return snapshot_path_for(self.project_path)
        base = self.fallback_directory or Path(tempfile.gettempdir())
        return base / f"untitled{_SUFFIX}"

    def exists(self) -> bool:
        return self.path.exists()

    # --- writing ------------------------------------------------------------

    def save(self, payload: Mapping[str, Any], *, reason: str = "autosave") -> Path:
        """Write a snapshot atomically. Never touches the project itself."""

        snapshot = RecoverySnapshot(
            payload=dict(payload),
            saved_utc=time.time(),
            origin=str(self.project_path) if self.project_path else "",
            reason=reason,
        )
        destination = self.path
        destination.parent.mkdir(parents=True, exist_ok=True)

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(snapshot.describe(), handle, indent=2, sort_keys=True)
                handle.flush()
                # A snapshot that survives the crash only if the page cache does
                # is not a recovery snapshot.
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    # --- reading ------------------------------------------------------------

    def read(self) -> RecoverySnapshot | None:
        """The stored snapshot, or None if there is none or it is unreadable.

        A corrupt snapshot is treated as absent rather than raised: it is a
        convenience file, and refusing to open the project because the
        convenience file is damaged would turn a small problem into a large one.
        """

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, Mapping):
            return None
        try:
            return RecoverySnapshot.from_mapping(data)
        except (ValueError, TypeError):
            return None

    def offer(self) -> RecoverySnapshot | None:
        """The snapshot to offer on startup, if there is one worth offering.

        Only when it is *newer* than the project it belongs to. A snapshot older
        than the saved file describes work the user has already superseded, and
        offering it invites them to overwrite good work with stale work.
        """

        snapshot = self.read()
        if snapshot is None:
            return None
        if self.project_path is None or not self.project_path.exists():
            return snapshot
        try:
            saved_at = self.project_path.stat().st_mtime
        except OSError:
            return snapshot
        return snapshot if snapshot.saved_utc > saved_at else None

    # --- clearing -----------------------------------------------------------

    def discard(self) -> bool:
        """Delete the snapshot. True if there was one.

        Called when the user saves, and when they decline a recovery: a
        declined snapshot offered again on the next launch teaches people to
        dismiss the prompt without reading it.
        """

        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def adopt(self, project_path: str | os.PathLike[str]) -> None:
        """Point at a project the work has just been saved as.

        The snapshot for the old location is removed, because it now describes
        work that lives somewhere else.
        """

        self.discard()
        self.project_path = Path(project_path)

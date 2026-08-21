"""Where HRTF data lives on this machine.

Datasets, caches, and subject ratings belong in the user's application data
directory, not inside the repository: the dataset is large and licensed, a
checkout is not a good place for it, and a git working tree should not fill
with derived files.

The default follows each platform's convention - ``%LOCALAPPDATA%`` on
Windows, ``~/Library/Application Support`` on macOS, ``$XDG_DATA_HOME`` on
Linux - which is what ``QStandardPaths.AppLocalDataLocation`` reports. The core
stays Qt-free, so the GUI calls :func:`set_data_root` with Qt's own answer and
everything below follows it.

What lands in a project file is only the selected SOFA path and subject
identifier. Nothing here is required to open a project: a missing dataset is a
reported state, not a failure.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "APPLICATION_NAME",
    "data_root",
    "set_data_root",
    "hrtf_root",
    "cache_root",
    "datasets_root",
    "ratings_path",
    "read_json",
    "write_json",
]

APPLICATION_NAME = "BinauralBuilder"

_data_root_override: Path | None = None


def _platform_data_root() -> Path:
    """The platform's per-user application data directory."""

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(base or Path.home() / "AppData" / "Local") / APPLICATION_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APPLICATION_NAME
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base or Path.home() / ".local" / "share") / APPLICATION_NAME


def set_data_root(path: str | os.PathLike[str] | None) -> None:
    """Point storage at a specific directory.

    The GUI passes ``QStandardPaths.AppLocalDataLocation`` here so the Qt-free
    core and the Qt application agree on one location; tests use it to keep
    their files in a temporary directory.
    """

    global _data_root_override
    _data_root_override = None if path is None else Path(path)


def data_root() -> Path:
    """The application data directory currently in use."""

    return _data_root_override or _platform_data_root()


def hrtf_root() -> Path:
    """Where HRTF material for this application lives."""

    return data_root() / "hrtf"


def datasets_root() -> Path:
    """The default place to look for installed HRTF datasets.

    A user may point anywhere; this is only the suggestion, and it is outside
    the repository by construction.
    """

    return hrtf_root() / "datasets"


def cache_root() -> Path:
    """Where prepared datasets are cached."""

    return hrtf_root() / "cache"


def ratings_path() -> Path:
    """The subject-rating store."""

    return hrtf_root() / "subject_ratings.json"


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when it is missing or unreadable.

    Local state should never stop the application from starting, so a corrupt
    file degrades to the default rather than raising.
    """

    location = Path(path)
    try:
        return json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write JSON atomically, creating the directory if needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class StorageReport:
    """What exists on this machine, for the GUI's status line."""

    data_root: str
    datasets_root: str
    datasets_present: bool
    cache_root: str
    ratings_file: str
    ratings_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": self.data_root,
            "datasets_root": self.datasets_root,
            "datasets_present": self.datasets_present,
            "cache_root": self.cache_root,
            "ratings_file": self.ratings_file,
            "ratings_present": self.ratings_present,
        }


def storage_report() -> StorageReport:
    """Describe the local storage state without creating anything."""

    return StorageReport(
        data_root=str(data_root()),
        datasets_root=str(datasets_root()),
        datasets_present=datasets_root().is_dir(),
        cache_root=str(cache_root()),
        ratings_file=str(ratings_path()),
        ratings_present=ratings_path().is_file(),
    )


__all__.extend(["StorageReport", "storage_report"])

"""Version and schema identifiers for the canonical SAM workbench package.

`src.audio.sam_workbench` is the canonical, Qt-free SAM/HRTF implementation for
BinauralBuilder.  Every schema-affecting change must bump `SCHEMA_VERSION` and
gain an explicit migration in :mod:`src.audio.sam_workbench.migrations`; old
project files are never silently reinterpreted.
"""

from __future__ import annotations

from typing import Final

#: Version of the SAM project JSON schema written by this package.
SCHEMA_VERSION: Final[str] = "1.0"

#: Schema versions this package can load.  Older entries must have a migration.
SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("1.0",)

#: Implementation version of the workbench itself (independent of the schema).
PACKAGE_VERSION: Final[str] = "0.1.0"

__all__ = ["SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS", "PACKAGE_VERSION"]

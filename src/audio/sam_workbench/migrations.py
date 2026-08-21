"""Explicit schema migrations for SAM project documents.

Migrations are always explicit and ordered: a document is upgraded step by step
until it reaches :data:`~src.audio.sam_workbench.version.SCHEMA_VERSION`.  An
unknown or newer version is an error rather than a silent reinterpretation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .validation import ProjectValidationError, ValidationIssue
from .version import SCHEMA_VERSION

__all__ = [
    "UNVERSIONED",
    "MIGRATIONS",
    "migrate_project_dict",
]

#: Version assigned to documents written before the schema carried a version.
UNVERSIONED = "0"


def _migrate_unversioned_to_1_0(data: dict[str, Any]) -> dict[str, Any]:
    """Tag a pre-versioned document as schema ``1.0``.

    Pre-versioned documents predate the published schema, so no field is
    reinterpreted here; missing fields fall back to the documented ``1.0``
    dataclass defaults during parsing.
    """

    data["schema_version"] = "1.0"
    return data


#: Ordered ``from_version -> migration`` registry.
MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    UNVERSIONED: _migrate_unversioned_to_1_0,
}


def migrate_project_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` upgraded to the current schema version."""

    if not isinstance(data, Mapping):
        raise ProjectValidationError(
            [ValidationIssue("", f"expected a JSON object, got {type(data).__name__}")]
        )

    migrated = dict(data)
    raw_version = migrated.get("schema_version", UNVERSIONED)
    if raw_version is None:
        raw_version = UNVERSIONED
    if not isinstance(raw_version, str):
        raise ProjectValidationError(
            [
                ValidationIssue(
                    "schema_version",
                    f"expected a string, got {type(raw_version).__name__}",
                )
            ]
        )

    version = raw_version
    seen: set[str] = set()
    while version != SCHEMA_VERSION:
        if version in seen:  # pragma: no cover - defensive against a bad registry
            raise ProjectValidationError(
                [ValidationIssue("schema_version", f"migration cycle detected at {version!r}")]
            )
        seen.add(version)
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise ProjectValidationError(
                [
                    ValidationIssue(
                        "schema_version",
                        f"unsupported schema version {version!r}; "
                        f"this build writes {SCHEMA_VERSION!r}",
                    )
                ]
            )
        migrated = migration(migrated)
        version = str(migrated.get("schema_version", version))

    migrated["schema_version"] = SCHEMA_VERSION
    return migrated

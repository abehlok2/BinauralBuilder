"""Canonical SAM/HRTF workbench core for BinauralBuilder.

``src.audio.sam_workbench`` is the single source of truth for SAM synthesis.
It is Qt-free, ``slab``-free, and independent of the Rust realtime backend; it
must never import :mod:`PyQt5`, :mod:`slab`, ``main``, ``sound_creator``, the
UI dialogs, or ``src.realtime_backend``.  Legacy public entry points delegate
to this package through compatibility adapters instead of keeping a second
implementation (see ``README.md`` next to this file).

Phase 0 provides the conventions, the versioned project document, aggregate
validation, atomic persistence, explicit migrations, and the ``new``/
``validate`` command-line shell.
"""

from __future__ import annotations

from .conventions import (
    AUDIO_DTYPE,
    CHANNEL_LEFT,
    CHANNEL_RIGHT,
    DEFAULT_SAMPLE_RATE_HZ,
    cartesian_to_spherical_deg,
    db_to_linear,
    linear_to_db,
    samples_to_seconds,
    seconds_to_samples,
    spherical_to_cartesian_m,
    to_channel_major,
    to_frame_major,
)
from .migrations import migrate_project_dict
from .model import (
    AudioSettings,
    ListenerSettings,
    OutputSettings,
    Project,
    ProjectValidationError,
    SamModulationSpec,
    SignalSpec,
    Source,
    load_project,
    project_from_dict,
    save_project,
    validate_project,
)
from .validation import ValidationCollector, ValidationIssue
from .version import PACKAGE_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

__all__ = [
    "AUDIO_DTYPE",
    "CHANNEL_LEFT",
    "CHANNEL_RIGHT",
    "DEFAULT_SAMPLE_RATE_HZ",
    "PACKAGE_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AudioSettings",
    "ListenerSettings",
    "OutputSettings",
    "Project",
    "ProjectValidationError",
    "SamModulationSpec",
    "SignalSpec",
    "Source",
    "ValidationCollector",
    "ValidationIssue",
    "cartesian_to_spherical_deg",
    "db_to_linear",
    "linear_to_db",
    "load_project",
    "migrate_project_dict",
    "project_from_dict",
    "samples_to_seconds",
    "save_project",
    "seconds_to_samples",
    "spherical_to_cartesian_m",
    "to_channel_major",
    "to_frame_major",
    "validate_project",
]

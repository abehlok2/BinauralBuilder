"""Canonical SAM/HRTF workbench core for BinauralBuilder.

``src.audio.sam_workbench`` is the single source of truth for SAM synthesis.
It is Qt-free, ``slab``-free, and independent of the Rust realtime backend; it
must never import :mod:`PyQt5`, :mod:`slab`, ``main``, ``sound_creator``, the
UI dialogs, or ``src.realtime_backend``.  Legacy public entry points delegate
to this package through compatibility adapters instead of keeping a second
implementation (see ``README.md`` next to this file).

Phase 0 provides the conventions, the versioned project document, aggregate
validation, atomic persistence, explicit migrations, and the ``new``/
``validate`` command-line shell. Phase 1 adds the control engine, the DSP
primitives, the abstract phase-modulation renderer, WAV export with a
reconstruction manifest, the ``render`` command, and the BinauralBuilder
compatibility adapter that both public synthesis trees delegate to.
"""

from __future__ import annotations

from src.audio.sam_workbench.conventions import (
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
from src.audio.sam_workbench.compat import Sam2Spec, render_sam2, render_sam2_voice, sam2_spec_from_params
from src.audio.sam_workbench.controls import (
    ConstantControl,
    ControlBase,
    KeyframeControl,
    LfoControl,
    ExpressionControl,
    ExternalControl,
    RampControl,
    RandomWalkControl,
    StepSequenceControl,
    compile_control,
    linear_transition_control,
)
from src.audio.sam_workbench.dsp import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
    CompiledSource,
    ModulatorSpec,
    RenderContext,
    compile_source,
    render_source,
)
from src.audio.sam_workbench.export import build_manifest, export_wav, project_sha256
from src.audio.sam_workbench.migrations import migrate_project_dict
from src.audio.sam_workbench.model import (
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
from src.audio.sam_workbench.render import AbstractPMRenderer, SceneReport, render_project
from src.audio.sam_workbench.state import load_parameter_state, save_parameter_state
from src.audio.sam_workbench.validation import ValidationCollector, ValidationIssue
from src.audio.sam_workbench.version import PACKAGE_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

__all__ = [
    "AUDIO_DTYPE",
    "EAR_POLARITY_CANONICAL",
    "EAR_POLARITY_LEGACY",
    "EAR_POLARITY_SAME",
    "AbstractPMRenderer",
    "CompiledSource",
    "ConstantControl",
    "ExpressionControl",
    "ExternalControl",
    "ControlBase",
    "KeyframeControl",
    "LfoControl",
    "ModulatorSpec",
    "RampControl",
    "RandomWalkControl",
    "RenderContext",
    "Sam2Spec",
    "StepSequenceControl",
    "SceneReport",
    "build_manifest",
    "compile_control",
    "compile_source",
    "export_wav",
    "linear_transition_control",
    "project_sha256",
    "render_project",
    "render_sam2",
    "render_sam2_voice",
    "render_source",
    "sam2_spec_from_params",
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
    "load_parameter_state",
    "migrate_project_dict",
    "project_from_dict",
    "samples_to_seconds",
    "save_project",
    "save_parameter_state",
    "seconds_to_samples",
    "spherical_to_cartesian_m",
    "to_channel_major",
    "to_frame_major",
    "validate_project",
]

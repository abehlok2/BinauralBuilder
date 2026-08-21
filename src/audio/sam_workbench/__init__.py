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
compatibility adapter that both public synthesis trees delegate to. Phase 2
adds the parameter registry, analysis, and preview the GUI is built on, and
Phase 3 adds canonical trajectories - geometry, traversal, transforms, the
legacy path bridges - and the geometric binaural renderer.
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
from .compat import Sam2Spec, render_sam2, render_sam2_voice, sam2_spec_from_params
from .controls import (
    ConstantControl,
    ControlBase,
    KeyframeControl,
    LfoControl,
    RampControl,
    compile_control,
    linear_transition_control,
)
from .dsp import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
    CompiledSource,
    ModulatorSpec,
    RenderContext,
    compile_source,
    render_source,
)
from .export import build_manifest, export_wav, project_sha256
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
from .render import (
    AbstractPMRenderer,
    GeometricBinauralRenderer,
    GeometricBinauralSpec,
    SceneReport,
    render_geometric,
    render_project,
)
from .trajectory import (
    ArcGeometry,
    CircleGeometry,
    EllipseGeometry,
    LegacyPathTransform,
    ListenerFrame,
    PolylineGeometry,
    SplineGeometry,
    TrajectorySpec,
    TransformSpec,
    TraversalSpec,
    profile_to_geometry,
    upgrade_profile,
)
from .validation import ValidationCollector, ValidationIssue
from .version import PACKAGE_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

__all__ = [
    "AUDIO_DTYPE",
    "EAR_POLARITY_CANONICAL",
    "EAR_POLARITY_LEGACY",
    "EAR_POLARITY_SAME",
    "AbstractPMRenderer",
    "ArcGeometry",
    "CircleGeometry",
    "CompiledSource",
    "ConstantControl",
    "ControlBase",
    "EllipseGeometry",
    "GeometricBinauralRenderer",
    "GeometricBinauralSpec",
    "KeyframeControl",
    "LegacyPathTransform",
    "LfoControl",
    "ListenerFrame",
    "ModulatorSpec",
    "PolylineGeometry",
    "RampControl",
    "RenderContext",
    "Sam2Spec",
    "SceneReport",
    "SplineGeometry",
    "TrajectorySpec",
    "TransformSpec",
    "TraversalSpec",
    "build_manifest",
    "compile_control",
    "compile_source",
    "export_wav",
    "linear_transition_control",
    "profile_to_geometry",
    "project_sha256",
    "render_geometric",
    "render_project",
    "render_sam2",
    "render_sam2_voice",
    "render_source",
    "sam2_spec_from_params",
    "upgrade_profile",
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

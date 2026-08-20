"""Versioned, serializable phase-zero project model.

Frozen dataclasses keep snapshots safe to hand to workers.  The schema is
intentionally small at this stage, but its tagged signal/spatializer records
and stable IDs leave room for later renderer-specific fields without coupling
the domain to Qt or live DSP state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from uuid import uuid4

from .conventions import (
    DEFAULT_LIMITER_CEILING_DBFS,
    DEFAULT_MASTER_GAIN_DB,
    DEFAULT_OFFLINE_BLOCK_FRAMES,
    DEFAULT_PREVIEW_BLOCK_FRAMES,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SOUND_SPEED_M_S,
)

SCHEMA_VERSION = 1
SUPPORTED_RENDER_MODES = frozenset(("abstract", "geometric", "hrtf", "hybrid"))


def _new_id() -> str:
    return str(uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AudioSettings:
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    preview_block_frames: int = DEFAULT_PREVIEW_BLOCK_FRAMES
    offline_block_frames: int = DEFAULT_OFFLINE_BLOCK_FRAMES


@dataclass(frozen=True)
class Listener:
    id: str = field(default_factory=_new_id)
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    ear_spacing_m: float = 0.18


@dataclass(frozen=True)
class Environment:
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED_M_S


@dataclass(frozen=True)
class SignalSpec:
    type: str = "sine"
    carrier_frequency_hz: float = 440.0
    phase_rad: float = 0.0


@dataclass(frozen=True)
class SpatializerSpec:
    type: str = "abstract"


@dataclass(frozen=True)
class Source:
    id: str = field(default_factory=_new_id)
    name: str = "Source 1"
    enabled: bool = True
    start_s: float = 0.0
    duration_s: float | None = None
    amplitude_linear: float = 0.5
    signal: SignalSpec = field(default_factory=SignalSpec)
    spatializer: SpatializerSpec = field(default_factory=SpatializerSpec)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputSettings:
    master_gain_db: float = DEFAULT_MASTER_GAIN_DB
    limiter_enabled: bool = True
    limiter_ceiling_dbfs: float = DEFAULT_LIMITER_CEILING_DBFS
    format: str = "WAV"
    subtype: str = "PCM_24"


@dataclass(frozen=True)
class Project:
    schema_version: int = SCHEMA_VERSION
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    modified_at: str = field(default_factory=_now_iso)
    name: str = "Untitled SAM Project"
    audio: AudioSettings = field(default_factory=AudioSettings)
    listener: Listener = field(default_factory=Listener)
    environment: Environment = field(default_factory=Environment)
    sources: tuple[Source, ...] = field(default_factory=lambda: (Source(),))
    output: OutputSettings = field(default_factory=OutputSettings)
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class ProjectValidationError(ValueError):
    """Raised with all schema issues rather than only the first one."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.path}: {i.message}" for i in issues))


def validate_project(project: Project) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if project.schema_version != SCHEMA_VERSION:
        issues.append(ValidationIssue("schema_version", f"expected {SCHEMA_VERSION}"))
    if project.audio.sample_rate_hz < 8_000 or project.audio.sample_rate_hz > 384_000:
        issues.append(ValidationIssue("audio.sample_rate_hz", "must be between 8000 and 384000"))
    for key, value in (
        ("preview_block_frames", project.audio.preview_block_frames),
        ("offline_block_frames", project.audio.offline_block_frames),
    ):
        if value <= 0:
            issues.append(ValidationIssue(f"audio.{key}", "must be positive"))
    if not math.isfinite(project.environment.sound_speed_m_s) or project.environment.sound_speed_m_s <= 0:
        issues.append(ValidationIssue("environment.sound_speed_m_s", "must be finite and positive"))
    if project.listener.ear_spacing_m <= 0 or not math.isfinite(project.listener.ear_spacing_m):
        issues.append(ValidationIssue("listener.ear_spacing_m", "must be finite and positive"))
    seen_ids: set[str] = set()
    for index, source in enumerate(project.sources):
        prefix = f"sources[{index}]"
        if source.id in seen_ids:
            issues.append(ValidationIssue(f"{prefix}.id", "must be unique"))
        seen_ids.add(source.id)
        if source.spatializer.type not in SUPPORTED_RENDER_MODES:
            issues.append(ValidationIssue(f"{prefix}.spatializer.type", "unsupported render mode"))
        if not math.isfinite(source.signal.carrier_frequency_hz) or not (
            0 < source.signal.carrier_frequency_hz < project.audio.sample_rate_hz / 2
        ):
            issues.append(ValidationIssue(f"{prefix}.signal.carrier_frequency_hz", "must be below Nyquist"))
        if source.duration_s is not None and source.duration_s <= 0:
            issues.append(ValidationIssue(f"{prefix}.duration_s", "must be positive or null"))
        if not math.isfinite(source.amplitude_linear) or source.amplitude_linear < 0:
            issues.append(ValidationIssue(f"{prefix}.amplitude_linear", "must be finite and non-negative"))
    return tuple(issues)


def project_from_dict(data: Mapping[str, Any]) -> Project:
    """Decode schema version 1, preserving arbitrary project metadata."""
    try:
        version = int(data.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ProjectValidationError([ValidationIssue("schema_version", f"unsupported version {version}")])
        sources = tuple(
            Source(
                id=item.get("id", _new_id()), name=item.get("name", "Source"),
                enabled=item.get("enabled", True), start_s=item.get("start_s", 0.0),
                duration_s=item.get("duration_s"), amplitude_linear=item.get("amplitude_linear", 0.5),
                signal=SignalSpec(**item.get("signal", {})),
                spatializer=SpatializerSpec(**item.get("spatializer", {})),
                tags=tuple(item.get("tags", ())),
            ) for item in data.get("sources", ())
        )
        listener_data = dict(data.get("listener", {}))
        if "position_m" in listener_data:
            listener_data["position_m"] = tuple(listener_data["position_m"])
        project = Project(
            schema_version=version, id=data.get("id", _new_id()),
            created_at=data.get("created_at", _now_iso()), modified_at=data.get("modified_at", _now_iso()),
            name=data.get("name", "Untitled SAM Project"),
            audio=AudioSettings(**data.get("audio", {})), listener=Listener(**listener_data),
            environment=Environment(**data.get("environment", {})), sources=sources,
            output=OutputSettings(**data.get("output", {})), seed=data.get("seed", 0),
            metadata=dict(data.get("metadata", {})),
        )
    except ProjectValidationError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise ProjectValidationError([ValidationIssue("project", str(error))]) from error
    issues = list(validate_project(project))
    if issues:
        raise ProjectValidationError(issues)
    return project


def load_project(path: str | Path) -> Project:
    with Path(path).open("r", encoding="utf-8") as stream:
        try:
            data = json.load(stream)
        except json.JSONDecodeError as error:
            raise ProjectValidationError([ValidationIssue("project", f"invalid JSON: {error.msg}")]) from error
    if not isinstance(data, dict):
        raise ProjectValidationError([ValidationIssue("project", "root must be an object")])
    return project_from_dict(data)


def save_project(project: Project, path: str | Path) -> None:
    """Validate and atomically replace *path* with a formatted JSON snapshot."""
    issues = list(validate_project(project))
    if issues:
        raise ProjectValidationError(issues)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(project.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

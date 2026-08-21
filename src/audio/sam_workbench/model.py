"""Typed, versioned SAM project document.

This is the headless data model behind the command-line shell and the tests.
It is deliberately Qt-free and independent of BinauralBuilder's track/step
dictionaries: the compatibility adapter translates camelCase voice parameters
into these unit-explicit snake_case types rather than the model growing legacy
shapes.

Rules honoured here:

* safe dataclass defaults (quiet master gain, limiter on, conservative source
  amplitude);
* aggregate validation - every problem is reported together with a stable field
  path;
* explicit schema migrations, never a silent reinterpretation of an old field;
* unknown fields are preserved through a load/save round trip;
* atomic persistence via a temporary sibling file and ``os.replace``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.audio.sam_workbench.conventions import (
    DEFAULT_LIMITER_CEILING_DBFS,
    DEFAULT_MASTER_GAIN_DB,
    DEFAULT_OFFLINE_BLOCK_SIZE,
    DEFAULT_PREVIEW_BLOCK_SIZE,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SPEED_OF_SOUND_M_S,
    MAX_PREVIEW_BLOCK_SIZE,
    MAX_SAMPLE_RATE_HZ,
    MIN_PREVIEW_BLOCK_SIZE,
    MIN_SAMPLE_RATE_HZ,
)
from src.audio.sam_workbench.migrations import migrate_project_dict
from src.audio.sam_workbench.validation import ProjectValidationError, ValidationCollector, ValidationIssue
from src.audio.sam_workbench.version import SCHEMA_VERSION

__all__ = [
    "AudioSettings",
    "ListenerSettings",
    "OutputSettings",
    "Project",
    "ProjectValidationError",
    "SamModulationSpec",
    "SignalSpec",
    "Source",
    "ValidationIssue",
    "load_project",
    "project_from_dict",
    "save_project",
    "validate_project",
]

SIGNAL_KINDS: tuple[str, ...] = ("sine",)
MAX_AMPLITUDE_LINEAR = 4.0
MAX_MASTER_GAIN_DB = 12.0
MAX_DEPTH_RAD = 8.0 * math.pi


# --- helpers ---------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _format_datetime(value: datetime) -> str:
    moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any, path: str, collector: ValidationCollector) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        collector.add(path, f"expected an ISO-8601 timestamp, got {type(value).__name__}")
        return _utc_now()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        collector.add(path, f"expected an ISO-8601 timestamp, got {value!r}")
        return _utc_now()
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _read_number(
    raw: Mapping[str, Any],
    key: str,
    default: float,
    collector: ValidationCollector,
    *,
    allow_none: bool = False,
) -> float | None:
    """Read a numeric field, reporting a typed issue instead of silently defaulting."""

    if key not in raw:
        return default
    value = raw[key]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        collector.add(key, f"expected a number, got {type(value).__name__}")
        return default
    return float(value)


def _read_int(raw: Mapping[str, Any], key: str, default: int, collector: ValidationCollector) -> int:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        collector.add(key, f"expected an integer, got {type(value).__name__}")
        return default
    return int(value)


def _read_bool(raw: Mapping[str, Any], key: str, default: bool, collector: ValidationCollector) -> bool:
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        collector.add(key, f"expected a boolean, got {type(value).__name__}")
        return default
    return value


def _read_text(
    raw: Mapping[str, Any],
    key: str,
    default: str | None,
    collector: ValidationCollector,
    *,
    allow_none: bool = False,
) -> str | None:
    if key not in raw:
        return default
    value = raw[key]
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        collector.add(key, f"expected a string, got {type(value).__name__}")
        return default
    return value


def _extras(data: Mapping[str, Any], known: Sequence[str]) -> dict[str, Any]:
    """Return the fields this build does not know about, so they survive a round trip."""

    return {key: value for key, value in data.items() if key not in set(known)}


def _mapping(value: Any, path: str, collector: ValidationCollector) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    collector.add(path, f"expected an object, got {type(value).__name__}")
    return {}


# --- settings --------------------------------------------------------------


@dataclass(frozen=True)
class AudioSettings:
    """Global audio settings shared by preview and offline rendering."""

    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    preview_block_size: int = DEFAULT_PREVIEW_BLOCK_SIZE
    offline_block_size: int = DEFAULT_OFFLINE_BLOCK_SIZE
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S
    random_seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = (
        "sample_rate_hz",
        "preview_block_size",
        "offline_block_size",
        "speed_of_sound_m_s",
        "random_seed",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": int(self.sample_rate_hz),
            "preview_block_size": int(self.preview_block_size),
            "offline_block_size": int(self.offline_block_size),
            "speed_of_sound_m_s": float(self.speed_of_sound_m_s),
            "random_seed": int(self.random_seed),
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "AudioSettings":
        raw = _mapping(data, "", collector)
        defaults = cls()
        return cls(
            sample_rate_hz=_read_int(raw, "sample_rate_hz", defaults.sample_rate_hz, collector),
            preview_block_size=_read_int(raw, "preview_block_size", defaults.preview_block_size, collector),
            offline_block_size=_read_int(raw, "offline_block_size", defaults.offline_block_size, collector),
            speed_of_sound_m_s=_read_number(raw, "speed_of_sound_m_s", defaults.speed_of_sound_m_s, collector),
            random_seed=_read_int(raw, "random_seed", defaults.random_seed, collector),
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector) -> None:
        collector.check_integer(
            "sample_rate_hz",
            self.sample_rate_hz,
            minimum=MIN_SAMPLE_RATE_HZ,
            maximum=MAX_SAMPLE_RATE_HZ,
        )
        collector.check_integer(
            "preview_block_size",
            self.preview_block_size,
            minimum=MIN_PREVIEW_BLOCK_SIZE,
            maximum=MAX_PREVIEW_BLOCK_SIZE,
        )
        collector.check_integer("offline_block_size", self.offline_block_size, minimum=64, maximum=1 << 20)
        collector.check_number("speed_of_sound_m_s", self.speed_of_sound_m_s, minimum=0.0, exclusive_minimum=True)
        collector.check_integer("random_seed", self.random_seed, minimum=0)


@dataclass(frozen=True)
class ListenerSettings:
    """Listener placement in the canonical right-handed listener frame."""

    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    head_radius_m: float = 0.0875
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = ("position_m", "yaw_deg", "pitch_deg", "roll_deg", "head_radius_m")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": [float(value) for value in self.position_m],
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
            "head_radius_m": float(self.head_radius_m),
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "ListenerSettings":
        raw = _mapping(data, "", collector)
        defaults = cls()
        position = raw.get("position_m", list(defaults.position_m))
        if (
            isinstance(position, Sequence)
            and not isinstance(position, (str, bytes))
            and len(position) == 3
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in position)
        ):
            coordinates = tuple(float(value) for value in position)
        else:
            collector.add("position_m", "expected three numeric coordinates in metres")
            coordinates = defaults.position_m
        return cls(
            position_m=coordinates,  # type: ignore[arg-type]
            yaw_deg=_read_number(raw, "yaw_deg", defaults.yaw_deg, collector),
            pitch_deg=_read_number(raw, "pitch_deg", defaults.pitch_deg, collector),
            roll_deg=_read_number(raw, "roll_deg", defaults.roll_deg, collector),
            head_radius_m=_read_number(raw, "head_radius_m", defaults.head_radius_m, collector),
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector) -> None:
        for axis, value in zip("xyz", self.position_m):
            collector.check_number(f"position_m.{axis}", value)
        collector.check_number("yaw_deg", self.yaw_deg)
        collector.check_number("pitch_deg", self.pitch_deg)
        collector.check_number("roll_deg", self.roll_deg)
        collector.check_number("head_radius_m", self.head_radius_m, minimum=0.0, exclusive_minimum=True, maximum=0.5)


@dataclass(frozen=True)
class OutputSettings:
    """Master bus settings.

    Both ears always share one gain: no SAM/HRTF stage may normalize a single
    channel.  Headphone compensation is opt-in and is reported in the render
    manifest.
    """

    master_gain_db: float = DEFAULT_MASTER_GAIN_DB
    limiter_enabled: bool = True
    limiter_ceiling_dbfs: float = DEFAULT_LIMITER_CEILING_DBFS
    headphone_compensation_enabled: bool = False
    headphone_compensation_asset: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = (
        "master_gain_db",
        "limiter_enabled",
        "limiter_ceiling_dbfs",
        "headphone_compensation_enabled",
        "headphone_compensation_asset",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_gain_db": float(self.master_gain_db),
            "limiter_enabled": bool(self.limiter_enabled),
            "limiter_ceiling_dbfs": float(self.limiter_ceiling_dbfs),
            "headphone_compensation_enabled": bool(self.headphone_compensation_enabled),
            "headphone_compensation_asset": self.headphone_compensation_asset,
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "OutputSettings":
        raw = _mapping(data, "", collector)
        defaults = cls()
        return cls(
            master_gain_db=_read_number(raw, "master_gain_db", defaults.master_gain_db, collector),
            limiter_enabled=_read_bool(raw, "limiter_enabled", defaults.limiter_enabled, collector),
            limiter_ceiling_dbfs=_read_number(raw, "limiter_ceiling_dbfs", defaults.limiter_ceiling_dbfs, collector),
            headphone_compensation_enabled=_read_bool(
                raw, "headphone_compensation_enabled", defaults.headphone_compensation_enabled, collector
            ),
            headphone_compensation_asset=_read_text(
                raw,
                "headphone_compensation_asset",
                defaults.headphone_compensation_asset,
                collector,
                allow_none=True,
            ),
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector) -> None:
        collector.check_number("master_gain_db", self.master_gain_db, maximum=MAX_MASTER_GAIN_DB)
        collector.check_boolean("limiter_enabled", self.limiter_enabled)
        collector.check_number("limiter_ceiling_dbfs", self.limiter_ceiling_dbfs, minimum=-60.0, maximum=0.0)
        collector.check_boolean("headphone_compensation_enabled", self.headphone_compensation_enabled)
        if self.headphone_compensation_asset is not None:
            collector.check_text("headphone_compensation_asset", self.headphone_compensation_asset)
        if self.headphone_compensation_enabled and not self.headphone_compensation_asset:
            collector.add(
                "headphone_compensation_asset",
                "headphone compensation is enabled but no asset is selected",
            )


# --- source ----------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """Carrier signal for a source.

    Phase 0 defines the sine carrier only; richer variants (harmonic banks,
    band-limited waves, noise, files) are tagged by ``kind`` in later phases.
    """

    kind: str = "sine"
    carrier_frequency_hz: float = 220.0
    phase_rad: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = ("kind", "carrier_frequency_hz", "phase_rad")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "carrier_frequency_hz": float(self.carrier_frequency_hz),
            "phase_rad": float(self.phase_rad),
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "SignalSpec":
        raw = _mapping(data, "", collector)
        defaults = cls()
        return cls(
            kind=_read_text(raw, "kind", defaults.kind, collector),
            carrier_frequency_hz=_read_number(
                raw, "carrier_frequency_hz", defaults.carrier_frequency_hz, collector
            ),
            phase_rad=_read_number(raw, "phase_rad", defaults.phase_rad, collector),
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector, *, sample_rate_hz: int) -> None:
        collector.check_choice("kind", self.kind, SIGNAL_KINDS)
        nyquist = float(sample_rate_hz) / 2.0 if sample_rate_hz > 0 else None
        collector.check_number(
            "carrier_frequency_hz",
            self.carrier_frequency_hz,
            minimum=0.0,
            exclusive_minimum=True,
            maximum=nyquist,
        )
        collector.check_number("phase_rad", self.phase_rad)


@dataclass(frozen=True)
class SamModulationSpec:
    """Opposed-ear phase modulation applied to a source's carrier.

    The canonical exact mode adds the modulation to the left ear and subtracts
    it from the right.  Legacy unversioned SAM2 voices use the opposite
    polarity and keep it through an explicit legacy-orientation flag at the
    adapter boundary, never by changing this specification.
    """

    rate_hz: float = 4.0
    depth_rad: float = 1.0
    phase_rad: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = ("rate_hz", "depth_rad", "phase_rad")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate_hz": float(self.rate_hz),
            "depth_rad": float(self.depth_rad),
            "phase_rad": float(self.phase_rad),
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "SamModulationSpec":
        raw = _mapping(data, "", collector)
        defaults = cls()
        return cls(
            rate_hz=_read_number(raw, "rate_hz", defaults.rate_hz, collector),
            depth_rad=_read_number(raw, "depth_rad", defaults.depth_rad, collector),
            phase_rad=_read_number(raw, "phase_rad", defaults.phase_rad, collector),
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector, *, sample_rate_hz: int) -> None:
        nyquist = float(sample_rate_hz) / 2.0 if sample_rate_hz > 0 else None
        collector.check_number("rate_hz", self.rate_hz, minimum=0.0, maximum=nyquist)
        collector.check_number("depth_rad", self.depth_rad, minimum=0.0, maximum=MAX_DEPTH_RAD)
        collector.check_number("phase_rad", self.phase_rad)


@dataclass(frozen=True)
class Source:
    """One SAM voice inside a project."""

    id: str = field(default_factory=lambda: _new_id("source"))
    name: str = "Source"
    enabled: bool = True
    start_s: float = 0.0
    duration_s: float | None = None
    amplitude_linear: float = 0.5
    signal: SignalSpec = field(default_factory=SignalSpec)
    sam: SamModulationSpec = field(default_factory=SamModulationSpec)
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = (
        "id",
        "name",
        "enabled",
        "start_s",
        "duration_s",
        "amplitude_linear",
        "signal",
        "sam",
        "tags",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": str(self.name),
            "enabled": bool(self.enabled),
            "start_s": float(self.start_s),
            "duration_s": None if self.duration_s is None else float(self.duration_s),
            "amplitude_linear": float(self.amplitude_linear),
            "signal": self.signal.to_dict(),
            "sam": self.sam.to_dict(),
            "tags": [str(tag) for tag in self.tags],
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: Any, collector: ValidationCollector) -> "Source":
        raw = _mapping(data, "", collector)
        defaults = cls()
        tags = raw.get("tags", [])
        if (
            isinstance(tags, Sequence)
            and not isinstance(tags, (str, bytes))
            and all(isinstance(tag, str) for tag in tags)
        ):
            parsed_tags = tuple(tags)
        else:
            collector.add("tags", "expected a list of strings")
            parsed_tags = ()
        return cls(
            id=_read_text(raw, "id", defaults.id, collector),
            name=_read_text(raw, "name", defaults.name, collector),
            enabled=_read_bool(raw, "enabled", defaults.enabled, collector),
            start_s=_read_number(raw, "start_s", defaults.start_s, collector),
            duration_s=_read_number(raw, "duration_s", defaults.duration_s, collector, allow_none=True),
            amplitude_linear=_read_number(raw, "amplitude_linear", defaults.amplitude_linear, collector),
            signal=SignalSpec.from_dict(raw.get("signal", {}), collector.child("signal")),
            sam=SamModulationSpec.from_dict(raw.get("sam", {}), collector.child("sam")),
            tags=parsed_tags,
            extras=_extras(raw, cls._FIELDS),
        )

    def validate(self, collector: ValidationCollector, *, sample_rate_hz: int) -> None:
        collector.check_text("id", self.id)
        collector.check_text("name", self.name, allow_empty=True)
        collector.check_boolean("enabled", self.enabled)
        collector.check_number("start_s", self.start_s, minimum=0.0)
        collector.check_number(
            "duration_s", self.duration_s, minimum=0.0, exclusive_minimum=True, allow_none=True
        )
        collector.check_number(
            "amplitude_linear", self.amplitude_linear, minimum=0.0, maximum=MAX_AMPLITUDE_LINEAR
        )
        self.signal.validate(collector.child("signal"), sample_rate_hz=sample_rate_hz)
        self.sam.validate(collector.child("sam"), sample_rate_hz=sample_rate_hz)


# --- project ---------------------------------------------------------------


@dataclass(frozen=True)
class Project:
    """Root SAM project document."""

    schema_version: str = SCHEMA_VERSION
    id: str = field(default_factory=lambda: _new_id("project"))
    name: str = "Untitled SAM Project"
    created_utc: datetime = field(default_factory=_utc_now)
    modified_utc: datetime = field(default_factory=_utc_now)
    audio: AudioSettings = field(default_factory=AudioSettings)
    listener: ListenerSettings = field(default_factory=ListenerSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    sources: tuple[Source, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    _FIELDS = (
        "schema_version",
        "id",
        "name",
        "created_utc",
        "modified_utc",
        "audio",
        "listener",
        "output",
        "sources",
        "metadata",
    )

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "id": str(self.id),
            "name": str(self.name),
            "created_utc": _format_datetime(self.created_utc),
            "modified_utc": _format_datetime(self.modified_utc),
            "audio": self.audio.to_dict(),
            "listener": self.listener.to_dict(),
            "output": self.output.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "metadata": dict(self.metadata),
            **self.extras,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, ensure_ascii=False) + "\n"

    # --- convenience -------------------------------------------------------

    def touched(self) -> "Project":
        """Return a copy stamped with the current modification time."""

        return replace(self, modified_utc=_utc_now())

    def with_source(self, source: Source) -> "Project":
        return replace(self, sources=self.sources + (source,))

    def validate(self) -> None:
        """Raise :class:`ProjectValidationError` listing every problem found."""

        collector = ValidationCollector()
        validate_project(self, collector)
        collector.raise_if_any()


def validate_project(project: Project, collector: ValidationCollector) -> None:
    """Collect every validation issue for ``project`` under ``collector``."""

    if project.schema_version != SCHEMA_VERSION:
        collector.add(
            "schema_version",
            f"expected {SCHEMA_VERSION!r}, got {project.schema_version!r}",
        )
    collector.check_text("id", project.id)
    collector.check_text("name", project.name)
    if project.modified_utc < project.created_utc:
        collector.add("modified_utc", "must not precede created_utc")

    project.audio.validate(collector.child("audio"))
    project.listener.validate(collector.child("listener"))
    project.output.validate(collector.child("output"))

    seen_ids: set[str] = set()
    for position, source in enumerate(project.sources):
        nested = collector.index("sources", position)
        source.validate(nested, sample_rate_hz=project.audio.sample_rate_hz)
        if isinstance(source.id, str):
            if source.id in seen_ids:
                nested.add("id", f"duplicate source id {source.id!r}")
            else:
                seen_ids.add(source.id)


def project_from_dict(data: Mapping[str, Any]) -> Project:
    """Parse, migrate, and validate a project document.

    Every problem is reported together through :class:`ProjectValidationError`.
    """

    migrated = migrate_project_dict(data)
    collector = ValidationCollector()

    raw_sources = migrated.get("sources", [])
    sources: list[Source] = []
    if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes)):
        for position, raw_source in enumerate(raw_sources):
            sources.append(Source.from_dict(raw_source, collector.index("sources", position)))
    else:
        collector.add("sources", "expected a list of sources")

    metadata = migrated.get("metadata", {})
    if not isinstance(metadata, Mapping):
        collector.add("metadata", f"expected an object, got {type(metadata).__name__}")
        metadata = {}

    defaults = Project()
    project = Project(
        schema_version=_read_text(migrated, "schema_version", SCHEMA_VERSION, collector),
        id=_read_text(migrated, "id", defaults.id, collector),
        name=_read_text(migrated, "name", defaults.name, collector),
        created_utc=_parse_datetime(
            migrated.get("created_utc", defaults.created_utc), "created_utc", collector
        ),
        modified_utc=_parse_datetime(
            migrated.get("modified_utc", defaults.modified_utc), "modified_utc", collector
        ),
        audio=AudioSettings.from_dict(migrated.get("audio", {}), collector.child("audio")),
        listener=ListenerSettings.from_dict(migrated.get("listener", {}), collector.child("listener")),
        output=OutputSettings.from_dict(migrated.get("output", {}), collector.child("output")),
        sources=tuple(sources),
        metadata=dict(metadata),
        extras=_extras(migrated, Project._FIELDS),
    )

    validate_project(project, collector)
    collector.raise_if_any()
    return project


def project_to_dict(project: Project) -> dict[str, Any]:
    """Return the JSON-ready mapping for ``project``."""

    return project.to_dict()


def save_project(project: Project, path: str | os.PathLike[str]) -> Path:
    """Validate then write ``project`` atomically to ``path``.

    The document is written to a temporary sibling file, flushed to disk, and
    moved into place with :func:`os.replace`, so a reader never observes a
    partially written project and a failed write leaves no stray file behind.
    """

    project.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = project.to_json()

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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_project(path: str | os.PathLike[str]) -> Project:
    """Read, migrate, and validate the project stored at ``path``."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"no SAM project at {source}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProjectValidationError(
            [ValidationIssue("", f"{source} is not valid JSON: {error.msg} (line {error.lineno})")]
        ) from error
    if not isinstance(data, Mapping):
        raise ProjectValidationError(
            [ValidationIssue("", f"expected a JSON object, got {type(data).__name__}")]
        )
    return project_from_dict(data)

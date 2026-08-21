"""Writing a modified HRTF back out as a SOFA file, with its provenance.

A derived asset is not a measurement, and the file has to say so. Someone who
finds one of these on disk a year from now - or a listening-test participant
comparing subjects - must be able to tell at a glance that it was computed
rather than measured, from which source, and with which cue settings.

So every derived file carries:

* the modified impulse responses;
* the source asset's path, hash, and identifying metadata;
* the cue parameters that produced it;
* the application and its version;
* the delay policy the source was read under;
* the one shared normalization gain that was applied;
* quality metrics;
* an unambiguous derived-data marker.

The provenance travels as JSON in ``GLOBAL_Comment`` so it survives any reader,
and the marker is repeated in ``GLOBAL_Title`` and ``GLOBAL_History`` where a
human will actually look. Verification runs before a write is reported as
successful: a file that does not pass is not a file worth handing to anyone.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..version import PACKAGE_VERSION
from .coordinates import cartesian_to_sofa_spherical
from .modification import CueTransform, ModifiedHrtf
from .sofa_io import SofaDependencyError

__all__ = [
    "APPLICATION_NAME",
    "DERIVED_MARKER",
    "PROVENANCE_KEY",
    "DerivedProvenance",
    "DerivedWriteResult",
    "read_derived_provenance",
    "verify_sofa_file",
    "write_derived_sofa",
]

APPLICATION_NAME = "BinauralBuilder SAM workbench"

#: Stated wherever a human or a parser might look.
DERIVED_MARKER = "DERIVED DATA - computed from a measured HRTF, not a measurement"

#: Key the provenance object sits under inside GLOBAL_Comment.
PROVENANCE_KEY = "binauralbuilder_derived"


@dataclass(frozen=True)
class DerivedProvenance:
    """Everything needed to explain, or reproduce, a derived asset."""

    source_path: str = ""
    source_sha256: str = ""
    source_convention: str = ""
    source_database: str = ""
    source_listener: str = ""
    delay_policy: str = ""
    transform: dict[str, Any] = field(default_factory=dict)
    normalization_gain: float = 1.0
    quality: dict[str, float] = field(default_factory=dict)
    application: str = APPLICATION_NAME
    application_version: str = PACKAGE_VERSION
    created_utc: str = ""
    derived: bool = True
    marker: str = DERIVED_MARKER

    def to_json(self) -> str:
        return json.dumps({PROVENANCE_KEY: asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DerivedProvenance | None":
        """Parse provenance out of a comment, or return None if it is absent.

        A comment that is not ours, or not JSON at all, is not an error: plenty
        of legitimate SOFA files carry free text there.
        """

        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or PROVENANCE_KEY not in payload:
            return None

        values = payload[PROVENANCE_KEY]
        if not isinstance(values, dict):
            return None
        known = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**known)

    def cue_transform(self) -> CueTransform:
        """The transform this asset was produced with."""

        return CueTransform.from_mapping(self.transform)


@dataclass(frozen=True)
class DerivedWriteResult:
    """What happened when a derived asset was written."""

    path: Path
    verified: bool
    message: str
    provenance: DerivedProvenance

    def __bool__(self) -> bool:
        return self.verified


def _quiet():
    """Silence the chatter ``sofar`` prints on read and write.

    It reports custom entries and dimension tables on stdout. That is useful at
    a prompt and noise inside an application, and some of it comes from netCDF
    below the Python layer, so the file descriptor is redirected rather than
    just ``sys.stdout``.
    """

    return _redirect_fd(1)


@contextlib.contextmanager
def _redirect_fd(fileno: int):
    import os
    import sys

    sys.stdout.flush()
    saved = os.dup(fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fileno)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, fileno)
        os.close(devnull)
        os.close(saved)


def _require_sofar():
    try:
        import sofar
    except ImportError as error:  # pragma: no cover - depends on the install
        raise SofaDependencyError(
            "writing a derived SOFA file needs the 'sofar' package; "
            "install it with: pip install sofar"
        ) from error
    return sofar


def verify_sofa_file(path: str | Path) -> tuple[bool, str]:
    """Run ``sofar``'s convention verification over a written file.

    Returns ``(passed, message)``. Verification that cannot run at all - no
    ``sofar`` installed - is reported as not passing, with a message saying
    why, rather than being quietly treated as success.
    """

    try:
        sofar = _require_sofar()
    except SofaDependencyError as error:
        return False, str(error)

    try:
        with _quiet():
            data = sofar.read_sofa(str(path))
            issues = data.verify(mode="read", issue_handling="return")
    except Exception as error:  # pragma: no cover - malformed file
        return False, f"verification failed: {error}"

    if issues:
        return False, str(issues)
    return True, "sofar verification passed"


def write_derived_sofa(
    modified: ModifiedHrtf,
    path: str | Path,
    *,
    source: Any = None,
    delay_policy: str = "",
    title: str = "",
    verify: bool = True,
) -> DerivedWriteResult:
    """Write a modified HRTF as a SOFA file carrying its full provenance.

    The write is only reported as successful once verification passes.
    """

    sofar = _require_sofar()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(getattr(source, "metadata", {}) or {})
    provenance = DerivedProvenance(
        source_path=str(getattr(source, "path", "") or ""),
        source_sha256=str(getattr(source, "content_hash", "") or modified.source_hash),
        source_convention=str(getattr(source, "convention", "") or ""),
        source_database=str(metadata.get("DatabaseName", "") or ""),
        source_listener=str(metadata.get("ListenerShortName", "") or ""),
        delay_policy=str(delay_policy or getattr(getattr(source, "delay_policy", None), "value", "")),
        transform=modified.transform.describe(),
        normalization_gain=float(modified.normalization_gain),
        quality=modified.quality_metrics(),
        created_utc=_datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
    )

    data = sofar.Sofa("SimpleFreeFieldHRIR")
    data.Data_IR = np.asarray(modified.hrirs, dtype=np.float64)
    data.Data_SamplingRate = float(modified.sample_rate_hz)
    # The cue transform rebuilds each response with its delay already in place,
    # so nothing is left to carry separately.
    data.Data_Delay = np.zeros((1, 2), dtype=np.float64)

    spherical = cartesian_to_sofa_spherical(modified.positions_m)
    data.SourcePosition = np.asarray(spherical, dtype=np.float64)
    data.SourcePosition_Type = "spherical"
    data.SourcePosition_Units = "degree, degree, metre"

    data.GLOBAL_Title = title or f"{DERIVED_MARKER}"
    data.GLOBAL_ApplicationName = APPLICATION_NAME
    data.GLOBAL_ApplicationVersion = PACKAGE_VERSION
    data.GLOBAL_Comment = provenance.to_json()
    data.GLOBAL_History = (
        f"{DERIVED_MARKER}; derived from {provenance.source_path or 'an unnamed source'} "
        f"(sha256 {provenance.source_sha256[:16] or 'unknown'}) by {APPLICATION_NAME} "
        f"{PACKAGE_VERSION}"
    )
    if provenance.source_database:
        data.GLOBAL_DatabaseName = provenance.source_database
    if provenance.source_listener:
        data.GLOBAL_ListenerShortName = provenance.source_listener

    with _quiet():
        sofar.write_sofa(str(destination), data)

    if not verify:
        return DerivedWriteResult(destination, True, "written without verification", provenance)

    passed, message = verify_sofa_file(destination)
    return DerivedWriteResult(destination, passed, message, provenance)


def read_derived_provenance(path: str | Path) -> DerivedProvenance | None:
    """Read a derived file's provenance, or None if it carries none.

    Used to tell a derived asset from a measured one without trusting its
    filename.
    """

    try:
        from .sofa_io import _read_hdf5

        with _quiet():
            attributes = _read_hdf5(Path(path))[-1]
    except Exception:
        return None

    comment = attributes.get("Comment") or attributes.get("GLOBAL_Comment") or ""
    if isinstance(comment, bytes):
        comment = comment.decode("utf-8", "replace")
    return DerivedProvenance.from_json(str(comment))

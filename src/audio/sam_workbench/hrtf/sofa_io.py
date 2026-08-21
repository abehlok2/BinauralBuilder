"""Reading SOFA HRTF assets.

Two backends, tried in order: `sofar`, which is standards-aware and validates
conventions, then `h5py`, since a SOFA file is a NetCDF4/HDF5 container.
Neither is imported at module load, so BinauralBuilder opens and every
non-HRTF feature works on a machine with no HRTF extras installed. Asking for
an HRTF without them raises :class:`HrtfDependencyError`, which the GUI turns
into an actionable message rather than a crash.

Nothing here interprets the data beyond what the file declares about itself.
Coordinate conversion, delay policy, and resampling belong to
:mod:`~src.audio.sam_workbench.hrtf.dataset`; this module's job is to read the
container faithfully and record what it found, including a content hash so a
render manifest can name the exact asset that produced it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HrtfDependencyError",
    "SofaFile",
    "SofaLoadError",
    "available_backend",
    "file_sha256",
    "load_sofa",
    "missing_dependency_message",
]

#: Read in chunks so hashing a 100 MB dataset does not load it twice.
_HASH_CHUNK = 1 << 20


class SofaLoadError(RuntimeError):
    """A SOFA file exists but cannot be read as one."""


class HrtfDependencyError(RuntimeError):
    """No SOFA backend is installed.

    Carries the message the GUI should show, naming what to install rather
    than reporting a bare import failure.
    """


def missing_dependency_message() -> str:
    """What to tell a user who has no SOFA backend installed."""

    return (
        "Reading SOFA HRTF files needs the HRTF extras. Install 'sofar' for "
        "standards-aware loading (recommended), or 'h5py' for basic reading:\n"
        "    pip install sofar\n"
        "Every non-HRTF feature works without them."
    )


def available_backend() -> str | None:
    """``'sofar'``, ``'h5py'``, or ``None`` when neither is installed."""

    from importlib.util import find_spec

    for candidate in ("sofar", "h5py"):
        try:  # pragma: no cover - depends on the environment
            if find_spec(candidate) is not None:
                return candidate
        except (ImportError, ValueError):
            continue
    return None


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Content hash of an asset, for the manifest and the cache key."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SofaFile:
    """A SOFA file's contents, as declared by the file itself.

    Nothing is converted or reordered here. ``impulse_responses`` keeps SOFA's
    own ``(measurements, receivers, samples)`` layout and ``source_positions``
    its own coordinate type and units, so a validator can report on what is
    actually in the file.
    """

    path: Path
    sha256: str
    convention: str
    convention_version: str
    data_type: str
    impulse_responses: NDArray[np.float64]  # (measurements, receivers, samples)
    sample_rate_hz: float
    source_positions: NDArray[np.float64]  # (measurements, 3)
    source_position_type: str
    source_position_units: str
    receiver_positions: NDArray[np.float64]  # (receivers, 3)
    data_delay: NDArray[np.float64]  # (1 or measurements, receivers)
    attributes: dict[str, Any] = field(default_factory=dict)
    backend: str = "sofar"

    @property
    def measurements(self) -> int:
        return int(self.impulse_responses.shape[0])

    @property
    def receivers(self) -> int:
        return int(self.impulse_responses.shape[1])

    @property
    def taps(self) -> int:
        return int(self.impulse_responses.shape[2])

    @property
    def has_external_delay(self) -> bool:
        """True when ``Data_Delay`` carries a nonzero delay that must be honoured."""

        return bool(np.any(np.abs(self.data_delay) > 1e-9))

    def attribute(self, *names: str, default: str = "") -> str:
        """First present global attribute among ``names``.

        SOFA spells its attributes inconsistently across versions and writers
        (``DatabaseName`` versus ``GLOBAL_DatabaseName``), so lookups are
        forgiving about the prefix.
        """

        for name in names:
            for key in (name, f"GLOBAL_{name}", name.replace("GLOBAL_", "")):
                value = self.attributes.get(key)
                if value not in (None, ""):
                    return str(value)
        return default

    @property
    def database_name(self) -> str:
        return self.attribute("DatabaseName", "Title")

    @property
    def listener_short_name(self) -> str:
        return self.attribute("ListenerShortName", "ListenerDescription")

    @property
    def license_note(self) -> str:
        return self.attribute("License")


def _as_array(value: Any) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


@contextlib.contextmanager
def _quiet_stdout():
    """Silence output from the reader, including writes below Python.

    `sofar` prints a dimension table on every read, and the netCDF layer under
    it writes straight to file descriptor 1, where `redirect_stdout` cannot
    reach. An application console is not the place for either, so the
    descriptor itself is redirected for the duration of the read.
    """

    try:
        saved = os.dup(1)
    except (AttributeError, OSError):  # pragma: no cover - platform without dup
        with contextlib.redirect_stdout(io.StringIO()):
            yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _load_with_sofar(path: Path) -> SofaFile:
    import sofar

    try:
        with _quiet_stdout():
            data = sofar.read_sofa(str(path), verify=False)
    except Exception as error:  # pragma: no cover - malformed file
        raise SofaLoadError(f"{path.name} could not be read as SOFA: {error}") from error

    # Read only the declared global attributes. Walking `dir()` would touch
    # `sofar` properties that print their own report as a side effect of being
    # read, which is how a library ends up spamming an application console.
    attributes: dict[str, Any] = {}
    for name in vars(data):
        if not name.startswith("GLOBAL_"):
            continue
        value = getattr(data, name, None)
        if isinstance(value, (str, int, float)):
            attributes[name] = value

    impulse_responses = np.atleast_3d(np.asarray(data.Data_IR, dtype=np.float64))
    delay = _as_array(getattr(data, "Data_Delay", 0.0))
    if delay.ndim == 1:
        delay = delay.reshape(1, -1)

    return SofaFile(
        path=path,
        sha256=file_sha256(path),
        convention=str(getattr(data, "GLOBAL_SOFAConventions", "")),
        convention_version=str(getattr(data, "GLOBAL_SOFAConventionsVersion", "")),
        data_type=str(getattr(data, "GLOBAL_DataType", "")),
        impulse_responses=impulse_responses,
        sample_rate_hz=float(np.atleast_1d(np.asarray(data.Data_SamplingRate)).ravel()[0]),
        source_positions=np.atleast_2d(np.asarray(data.SourcePosition, dtype=np.float64)),
        source_position_type=str(getattr(data, "SourcePosition_Type", "spherical")),
        source_position_units=str(getattr(data, "SourcePosition_Units", "degree, degree, metre")),
        receiver_positions=np.asarray(data.ReceiverPosition, dtype=np.float64).reshape(-1, 3),
        data_delay=delay,
        attributes=attributes,
        backend="sofar",
    )


def _load_with_h5py(path: Path) -> SofaFile:
    import h5py

    try:
        handle = h5py.File(path, "r")
    except Exception as error:  # pragma: no cover - malformed file
        raise SofaLoadError(f"{path.name} could not be opened as HDF5/SOFA: {error}") from error

    with handle:
        if "Data.IR" not in handle:
            raise SofaLoadError(f"{path.name} has no Data.IR dataset; it is not a SOFA HRTF file")

        def attribute_text(value: Any) -> str:
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

        attributes = {key: attribute_text(value) for key, value in handle.attrs.items()}
        impulse_responses = np.atleast_3d(np.asarray(handle["Data.IR"], dtype=np.float64))
        source = handle["SourcePosition"]
        receiver = np.asarray(handle["ReceiverPosition"], dtype=np.float64).reshape(-1, 3)
        delay = _as_array(handle["Data.Delay"][()] if "Data.Delay" in handle else 0.0)
        if delay.ndim == 1:
            delay = delay.reshape(1, -1)

        return SofaFile(
            path=path,
            sha256=file_sha256(path),
            convention=attributes.get("SOFAConventions", ""),
            convention_version=attributes.get("SOFAConventionsVersion", ""),
            data_type=attributes.get("DataType", ""),
            impulse_responses=impulse_responses,
            sample_rate_hz=float(np.asarray(handle["Data.SamplingRate"]).ravel()[0]),
            source_positions=np.atleast_2d(np.asarray(source, dtype=np.float64)),
            source_position_type=attribute_text(source.attrs.get("Type", "spherical")),
            source_position_units=attribute_text(source.attrs.get("Units", "degree, degree, metre")),
            receiver_positions=receiver,
            data_delay=delay,
            attributes=attributes,
            backend="h5py",
        )


def load_sofa(path: str | os.PathLike[str], *, backend: str | None = None) -> SofaFile:
    """Read a SOFA file with whichever backend is available.

    ``backend`` forces a choice, which the tests use to exercise both readers
    against the same fixture.
    """

    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"no SOFA file at {location}")

    chosen = backend or available_backend()
    if chosen is None:
        raise HrtfDependencyError(missing_dependency_message())
    if chosen == "sofar":
        try:
            return _load_with_sofar(location)
        except SofaLoadError:
            # A file a standards-aware reader rejects may still be readable as
            # the HDF5 container it is. Fall back only when the caller did not
            # pin a backend, and record which reader actually read the file.
            from importlib.util import find_spec

            if backend is not None or find_spec("h5py") is None:
                raise
            return _load_with_h5py(location)
    if chosen == "h5py":
        return _load_with_h5py(location)
    raise ValueError(f"unknown SOFA backend {backend!r}")

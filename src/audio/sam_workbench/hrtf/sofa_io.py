"""Standards-aware, explicit SOFA loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as fractional_shift
from scipy.signal import resample_poly

from .coordinates import sofa_positions_to_cartesian


class SofaDependencyError(ImportError):
    pass


class DelayPolicy(str, Enum):
    BAKE = "bake_delay_into_ir"
    #: ``keep_external_delay`` is the canonical name the renderer registry
    #: advertises: SOFA Data.Delay stays outside the impulse response.
    KEEP = "keep_external_delay"
    #: Historical spelling of :attr:`KEEP`. Older documents still carry it,
    #: and it has always meant exactly this policy - never another behaviour.
    PRESERVE = KEEP

    @classmethod
    def _missing_(cls, value: object) -> DelayPolicy | None:
        """Accept exactly the alias spellings :data:`DELAY_POLICY_ALIASES` lists.

        Read from that table rather than matching a literal, because the
        renderer registry advertises its keys as selectable policies. Deciding
        acceptance here and advertisement there from two different sources is
        how a policy becomes offerable in the interface and unloadable in the
        engine; one table means adding an alias cannot produce that.
        """

        if not isinstance(value, str):
            return None
        canonical = DELAY_POLICY_ALIASES.get(value.strip().lower())
        return None if canonical is None else cls(canonical)


#: Renamed policy values that persisted documents may still contain, mapped to
#: their canonical spellings; mirrors INTERPOLATION_ALIASES for logmag_delay.
#:
#: Kept as data rather than as extra enum members so that iterating the enum
#: yields each policy once - a second member sharing a meaning would make the
#: registry offer the same choice twice and a cache key store it twice. Both
#: :meth:`DelayPolicy._missing_` and the registry's advertised choices read
#: this one table, so the set accepted and the set offered cannot drift apart.
DELAY_POLICY_ALIASES: dict[str, str] = {
    "preserve_external_delay": DelayPolicy.KEEP.value,
}


@dataclass(frozen=True)
class HRTFDataset:
    path: Path
    content_hash: str
    convention: str
    convention_version: str
    ir: NDArray[np.float64]
    sample_rate_hz: float
    delay_samples: NDArray[np.float64]
    positions_m: NDArray[np.float64]
    receiver_positions_m: NDArray[np.float64]
    coordinate_type: str
    coordinate_units: str
    metadata: dict[str, Any]
    delay_policy: DelayPolicy
    source_sample_rate_hz: float

    @property
    def measurements(self) -> int:
        return int(self.ir.shape[0])

    @property
    def taps(self) -> int:
        return int(self.ir.shape[2])

    @property
    def has_external_delay(self) -> bool:
        return bool(np.any(np.abs(self.delay_samples) > 1e-12))

    def metadata_report(self) -> dict[str, Any]:
        peak_taps = np.argmax(np.abs(self.ir), axis=-1)
        tail_start = max(1, int(self.taps * 0.9))
        total_energy = float(np.sum(np.square(self.ir)))
        tail_energy = float(np.sum(np.square(self.ir[..., tail_start:])))
        return {
            "path": str(self.path), "sha256": self.content_hash,
            "convention": self.convention, "convention_version": self.convention_version,
            "measurements": self.measurements, "receivers": int(self.ir.shape[1]),
            "taps": self.taps, "sample_rate_hz": self.sample_rate_hz,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "delay_policy": self.delay_policy.value,
            "external_delay_nonzero": self.has_external_delay,
            "hrir_peak_tap_min": int(np.min(peak_taps)),
            "hrir_peak_tap_max": int(np.max(peak_taps)),
            "tail_energy_ratio": tail_energy / max(total_energy, 1e-30),
            "peak_amplitude": float(np.max(np.abs(self.ir), initial=0.0)),
            "coordinate_type": self.coordinate_type, "coordinate_units": self.coordinate_units,
            **self.metadata,
        }


def resolve_sofa_path(path: str | Path, project_directory: str | Path | None = None) -> Path:
    """Resolve portable project-relative and per-machine HRTF asset paths."""

    requested = Path(path).expanduser()
    candidates = [requested] if requested.is_absolute() else []
    if not requested.is_absolute():
        if project_directory is not None:
            candidates.append(Path(project_directory).expanduser() / requested)
        configured = os.environ.get("SAM_WORKBENCH_HRTF_DIR")
        if configured:
            candidates.append(Path(configured).expanduser() / requested)
        candidates.append(Path.cwd() / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SOFA asset does not exist; searched: {searched}")


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode(value.reshape(-1)[0])
    return value


def _read_hdf5(path: Path):
    try:
        import h5py
    except ImportError as error:
        raise SofaDependencyError(
            "SOFA loading requires h5py. Reinstall the application's standard "
            "dependencies with `python -m pip install -r requirements.txt`."
        ) from error
    with h5py.File(path, "r") as handle:
        def data(*names):
            for name in names:
                if name in handle:
                    return np.asarray(handle[name])
            raise ValueError(f"SOFA file is missing {names[0]}")
        ir = data("Data.IR", "Data/IR")
        rate = float(data("Data.SamplingRate", "Data/SamplingRate").reshape(-1)[0])
        delay_name = "Data.Delay" if "Data.Delay" in handle else "Data/Delay" if "Data/Delay" in handle else None
        delay = np.asarray(handle[delay_name]) if delay_name else np.zeros((1, 2))
        delay_units = str(_decode(handle[delay_name].attrs.get("Units", "second"))) if delay_name else "second"
        position_dataset = handle["SourcePosition"]
        positions = np.asarray(position_dataset)
        coordinate_type = str(_decode(position_dataset.attrs.get("Type", "spherical")))
        coordinate_units = str(_decode(position_dataset.attrs.get("Units", "degree, degree, metre")))
        attrs = {str(key): _decode(value) for key, value in handle.attrs.items()}
        receiver = np.asarray(handle["ReceiverPosition"]) if "ReceiverPosition" in handle else None
        receiver_type = str(_decode(handle["ReceiverPosition"].attrs.get("Type", "cartesian"))) if receiver is not None else "cartesian"
        receiver_units = str(_decode(handle["ReceiverPosition"].attrs.get("Units", "metre"))) if receiver is not None else "metre"
    return (ir, rate, delay, delay_units, positions, coordinate_type,
            coordinate_units, receiver, receiver_type, receiver_units, attrs)


def _sofar_verification(path: Path) -> tuple[bool | None, str]:
    """Run standards verification when the declared HRTF extra is installed."""

    try:
        import sofar
    except ImportError:
        return None, "sofar is not installed; structural validation only"
    try:
        sofar.read_sofa(str(path), verify=True, verbose=False)
    except Exception as error:
        return False, str(error)
    return True, "SOFA convention verification passed"


def _expanded_delay(delay: NDArray[np.float64], measurements: int) -> NDArray[np.float64]:
    values = np.asarray(delay, dtype=np.float64)
    if values.ndim == 1 and values.shape == (2,):
        values = values[None, :]
    if values.shape == (1, 2):
        values = np.repeat(values, measurements, axis=0)
    if values.shape != (measurements, 2):
        raise ValueError(f"Data.Delay must broadcast to ({measurements}, 2), got {values.shape}")
    return values


def _bake_delay(ir: NDArray[np.float64], delays: NDArray[np.float64]) -> NDArray[np.float64]:
    # Positive SOFA delay means the response occurs later. Padding prevents a
    # positive shift from truncating the original tail.
    pad = int(np.ceil(np.max(delays, initial=0.0))) + 4
    result = np.pad(ir, ((0, 0), (0, 0), (0, pad)))
    for measurement in range(ir.shape[0]):
        for ear in range(2):
            result[measurement, ear] = fractional_shift(
                result[measurement, ear], delays[measurement, ear], order=3,
                mode="constant", cval=0.0, prefilter=True,
            )
    return result


#: Read size for hashing. Large enough that the syscall overhead is
#: irrelevant, small enough that it never shows up as memory.
_HASH_CHUNK_BYTES = 1 << 20


def hash_asset(path: str | Path) -> str:
    """The SHA-256 of an asset's bytes, read in pieces.

    The whole file was previously read into memory to hash it. A SONICOM
    dataset is large enough that doing so is a transient allocation the size
    of the asset, for a value that never needs more than a megabyte in hand at
    once - and it happened on a path whose entire purpose is bounding how much
    a long render holds.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sofa(
    path: str | Path, *, target_sample_rate_hz: int | None = None,
    delay_policy: DelayPolicy | str = DelayPolicy.BAKE,
    project_directory: str | Path | None = None,
    content_hash: str | None = None,
) -> HRTFDataset:
    """Load an explicit SOFA asset, apply its delay policy, and resample once.

    ``content_hash`` lets a caller that has already hashed the bytes - the
    cache, deciding whether it holds this dataset - pass the digest in rather
    than have the file read through a second time for the same answer.
    """

    source = resolve_sofa_path(path, project_directory)
    policy = DelayPolicy(delay_policy)
    digest = hash_asset(source) if content_hash is None else str(content_hash)
    (ir, rate, raw_delay, delay_units, positions, kind, units, receiver,
     receiver_type, receiver_units, attrs) = _read_hdf5(source)
    ir = np.asarray(ir, dtype=np.float64)
    if ir.ndim != 3:
        raise ValueError(f"Data.IR must be three-dimensional, got {ir.shape}")
    delay = _expanded_delay(np.asarray(raw_delay, dtype=np.float64), ir.shape[0])
    # SimpleFreeFieldHRIR declares Data.Delay in seconds. Some files use an
    # explicit samples unit; support both rather than guessing silently.
    # "sample" must be tested first: it also starts with "s", so testing the
    # seconds spellings first would scale a samples delay by the sample rate.
    units_text = delay_units.strip().lower()
    if units_text.startswith("sample"):
        pass
    elif units_text.startswith(("second", "sec", "s")):
        delay = delay * rate
    else:
        raise ValueError(f"unsupported Data.Delay units {delay_units!r}")
    if policy is DelayPolicy.BAKE and np.any(np.abs(delay) > 0.0):
        ir = _bake_delay(ir, delay)
        delay = np.zeros_like(delay)
    source_rate = rate
    target_rate = float(target_sample_rate_hz or rate)
    if target_rate <= 0:
        raise ValueError("target_sample_rate_hz must be positive")
    if not np.isclose(target_rate, rate):
        from fractions import Fraction
        ratio = Fraction(target_rate / rate).limit_denominator(10_000)
        ir = resample_poly(ir, ratio.numerator, ratio.denominator, axis=-1)
        delay = delay * target_rate / rate
        rate = target_rate
    positions_m = sofa_positions_to_cartesian(positions, kind, units)
    if receiver is None:
        receiver_positions_m = np.empty((0, 3), dtype=np.float64)
    else:
        receiver = np.asarray(receiver, dtype=np.float64)
        # ReceiverPosition commonly has R,C,I dimensions with singleton I.
        receiver = np.squeeze(receiver)
        receiver_positions_m = sofa_positions_to_cartesian(
            receiver.reshape(-1, 3), receiver_type, receiver_units
        )
    metadata_keys = ("Title", "ListenerShortName", "DatabaseName", "AuthorContact", "Organization", "License")
    sofar_verified, sofar_message = _sofar_verification(source)
    dataset = HRTFDataset(
        source, digest, str(attrs.get("SOFAConventions", "")),
        str(attrs.get("SOFAConventionsVersion", "")), ir, rate, delay,
        positions_m, receiver_positions_m, kind, units,
        {**{key: attrs[key] for key in metadata_keys if key in attrs},
         "sofar_verified": sofar_verified, "sofar_verification": sofar_message},
        policy, source_rate,
    )
    from .validation import validate_dataset
    fatal = [issue for issue in validate_dataset(dataset) if issue.severity == "error"]
    if fatal:
        raise ValueError("invalid SOFA asset: " + "; ".join(f"{x.path}: {x.message}" for x in fatal))
    return dataset

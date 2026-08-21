"""Turning a SOFA file into a render-ready HRTF dataset.

Preparation happens once, not per block:

1. convert the file's declared coordinates into the canonical frame;
2. apply the chosen ``Data_Delay`` policy - never ignore a nonzero delay;
3. resample every HRIR to the project's sample rate;
4. record where the dataset came from, so a manifest can name it exactly.

The result is immutable. Direction lookup, interpolation, and convolution all
read from it and none of them modify it, which is what lets one prepared
dataset be shared by several voices and cached across renders.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from ..conventions import CHANNEL_COUNT
from ..dsp.delay import read_fractional
from .coordinates import canonical_to_spherical_deg, positions_to_canonical, unit_vectors
from .sofa_io import SofaFile, load_sofa

__all__ = [
    "DELAY_POLICIES",
    "BAKE_DELAY",
    "PRESERVE_DELAY",
    "HrtfAssetInfo",
    "PreparedHrtf",
    "prepare_dataset",
    "prepare_from_path",
]

#: Apply ``Data_Delay`` to the impulse responses and store zeros.
BAKE_DELAY = "bake_delay_into_ir"
#: Keep the impulse responses as measured and apply the delay at render time.
PRESERVE_DELAY = "preserve_external_delay"
DELAY_POLICIES = (BAKE_DELAY, PRESERVE_DELAY)

#: Extra taps added when baking a delay, so a shifted response is not truncated.
_BAKE_HEADROOM_TAPS = 8


@dataclass(frozen=True)
class HrtfAssetInfo:
    """Where a dataset came from, for the manifest and for the GUI."""

    path: str
    sha256: str
    convention: str
    convention_version: str
    database_name: str
    subject: str
    license_note: str
    source_sample_rate_hz: float
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "convention": self.convention,
            "convention_version": self.convention_version,
            "database_name": self.database_name,
            "subject": self.subject,
            "license_note": self.license_note,
            "source_sample_rate_hz": float(self.source_sample_rate_hz),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class PreparedHrtf:
    """A dataset ready to render: canonical directions, resampled responses."""

    directions: NDArray[np.float64]  # (3, measurements) unit vectors
    positions_m: NDArray[np.float64]  # (3, measurements) canonical metres
    hrirs: NDArray[np.float64]  # (measurements, 2, taps)
    delays_samples: NDArray[np.float64]  # (measurements, 2), zero once baked
    sample_rate_hz: int
    delay_policy: str
    asset: HrtfAssetInfo
    spherical_deg: NDArray[np.float64] = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if self.spherical_deg is None:
            object.__setattr__(self, "spherical_deg", canonical_to_spherical_deg(self.positions_m))

    @property
    def measurements(self) -> int:
        return int(self.hrirs.shape[0])

    @property
    def taps(self) -> int:
        return int(self.hrirs.shape[2])

    @property
    def has_external_delay(self) -> bool:
        return bool(np.any(np.abs(self.delays_samples) > 1e-9))

    @property
    def azimuths_deg(self) -> NDArray[np.float64]:
        return self.spherical_deg[0]

    @property
    def elevations_deg(self) -> NDArray[np.float64]:
        return self.spherical_deg[1]

    def cache_key(self) -> str:
        """Identity of this preparation: asset, rate, and policy together.

        Two preparations differing in any of those are different filters, so
        they must never share a cache entry.
        """

        payload = f"{self.asset.sha256}|{self.sample_rate_hz}|{self.delay_policy}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        """A manifest-ready description of the prepared dataset."""

        return {
            **self.asset.to_dict(),
            "measurements": self.measurements,
            "taps": self.taps,
            "sample_rate_hz": int(self.sample_rate_hz),
            "delay_policy": self.delay_policy,
            "external_delay_present": self.has_external_delay,
            "azimuth_deg_range": [float(np.min(self.azimuths_deg)), float(np.max(self.azimuths_deg))],
            "elevation_deg_range": [
                float(np.min(self.elevations_deg)),
                float(np.max(self.elevations_deg)),
            ],
        }


def _expand_delay(delay: NDArray[np.floating], measurements: int, receivers: int) -> NDArray[np.float64]:
    """Broadcast ``Data_Delay`` to ``(measurements, receivers)``."""

    values = np.atleast_2d(np.asarray(delay, dtype=np.float64))
    if values.shape == (measurements, receivers):
        return values
    if values.shape[0] == 1:
        return np.repeat(values[:, :receivers], measurements, axis=0)
    raise ValueError(f"Data_Delay shape {values.shape} does not fit {measurements}x{receivers}")


def _bake_delays(hrirs: NDArray[np.float64], delays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shift each response by its own delay, fractionally, and pad for it."""

    measurements, receivers, taps = hrirs.shape
    headroom = int(np.ceil(np.max(np.abs(delays)))) + _BAKE_HEADROOM_TAPS
    baked = np.zeros((measurements, receivers, taps + headroom), dtype=np.float64)
    for measurement in range(measurements):
        for receiver in range(receivers):
            padded = np.concatenate((hrirs[measurement, receiver], np.zeros(headroom)))
            shift = float(delays[measurement, receiver])
            baked[measurement, receiver] = read_fractional(
                padded, np.full(padded.size, shift), outside="zero"
            )
    return baked


def _resample(
    hrirs: NDArray[np.float64], source_rate: float, target_rate: int
) -> NDArray[np.float64]:
    """Resample every response to the project rate, once, at preparation time."""

    if int(round(source_rate)) == int(target_rate):
        return hrirs
    from math import gcd

    source = int(round(source_rate))
    divisor = gcd(source, int(target_rate))
    up, down = int(target_rate) // divisor, source // divisor
    resampled = scipy_signal.resample_poly(hrirs, up, down, axis=2)
    return np.asarray(resampled, dtype=np.float64)


def prepare_dataset(
    sofa_file: SofaFile,
    *,
    sample_rate_hz: int = 44_100,
    delay_policy: str = BAKE_DELAY,
    subject: str = "",
) -> PreparedHrtf:
    """Prepare a loaded SOFA file for rendering at ``sample_rate_hz``."""

    if delay_policy not in DELAY_POLICIES:
        raise ValueError(f"unknown delay policy {delay_policy!r}; expected one of {DELAY_POLICIES}")
    if sofa_file.receivers != CHANNEL_COUNT:
        raise ValueError(
            f"{sofa_file.path.name} has {sofa_file.receivers} receivers; "
            f"a binaural HRTF needs {CHANNEL_COUNT}"
        )

    positions = positions_to_canonical(
        sofa_file.source_positions, sofa_file.source_position_type, sofa_file.source_position_units
    )
    delays = _expand_delay(sofa_file.data_delay, sofa_file.measurements, sofa_file.receivers)
    hrirs = np.asarray(sofa_file.impulse_responses, dtype=np.float64)

    if delay_policy == BAKE_DELAY and np.any(np.abs(delays) > 1e-9):
        hrirs = _bake_delays(hrirs, delays)
        delays = np.zeros_like(delays)

    scale = float(sample_rate_hz) / float(sofa_file.sample_rate_hz)
    hrirs = _resample(hrirs, sofa_file.sample_rate_hz, sample_rate_hz)
    delays = delays * scale

    asset = HrtfAssetInfo(
        path=str(sofa_file.path),
        sha256=sofa_file.sha256,
        convention=sofa_file.convention,
        convention_version=sofa_file.convention_version,
        database_name=sofa_file.database_name,
        subject=subject or sofa_file.listener_short_name,
        license_note=sofa_file.license_note,
        source_sample_rate_hz=float(sofa_file.sample_rate_hz),
        backend=sofa_file.backend,
    )
    return PreparedHrtf(
        directions=unit_vectors(positions),
        positions_m=positions,
        hrirs=hrirs,
        delays_samples=delays,
        sample_rate_hz=int(sample_rate_hz),
        delay_policy=delay_policy,
        asset=asset,
    )


def prepare_from_path(
    path: str | Path,
    *,
    sample_rate_hz: int = 44_100,
    delay_policy: str = BAKE_DELAY,
    subject: str = "",
    backend: str | None = None,
) -> PreparedHrtf:
    """Load and prepare a SOFA asset in one step."""

    return prepare_dataset(
        load_sofa(path, backend=backend),
        sample_rate_hz=sample_rate_hz,
        delay_policy=delay_policy,
        subject=subject,
    )

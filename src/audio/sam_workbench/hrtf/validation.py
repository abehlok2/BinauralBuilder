"""SOFA ingest validation.

An HRTF asset is reported on, not silently trusted. The report separates two
kinds of finding, because they have different consequences:

* **fatal** - the file cannot be rendered with (wrong receiver count, no
  usable directions, non-finite data). Rendering is blocked.
* **warning** - the file is usable but something about it should be known
  (sparse rear coverage, a sample-rate mismatch, clipped or acausal responses,
  no licence statement). Rendering proceeds and the Inspector shows the note.

Only fatal findings block a render; everything else is information a user
should have before trusting what they hear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..conventions import CHANNEL_COUNT
from ..validation import ValidationIssue
from .coordinates import canonical_to_spherical_deg, positions_to_canonical
from .sofa_io import SofaFile

__all__ = ["HrtfReport", "validate_sofa_file"]

#: Beyond this the responses are almost certainly clipped.
_CLIPPING_LEVEL = 0.999
#: Energy before the onset, relative to the total, that suggests an acausal IR.
_ACAUSAL_ENERGY_FRACTION = 0.02
#: Directions this far apart leave an audible hole in the lookup.
_SPARSE_SPACING_DEG = 30.0


@dataclass(frozen=True)
class HrtfReport:
    """What an asset declares, and what is wrong with it."""

    issues: tuple[ValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fatal(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_renderable(self) -> bool:
        """False only for problems that make rendering impossible."""

        return not self.fatal

    @property
    def quality(self) -> str:
        """A one-word status for the GUI's quality field."""

        if self.fatal:
            return "missing resources"
        return "suspected outlier" if self.warnings else "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "renderable": self.is_renderable,
            "metadata": dict(self.metadata),
            "issues": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.issues
            ],
        }


def _direction_spacing_deg(spherical: NDArray[np.float64]) -> float:
    """Median nearest-neighbour spacing between directions, in degrees."""

    from .coordinates import angular_distance_deg, unit_vectors
    from ..conventions import spherical_to_cartesian_m

    directions = unit_vectors(
        np.vstack(
            [spherical_to_cartesian_m(azimuth, elevation, 1.0)
             for azimuth, elevation in zip(spherical[0], spherical[1])]
        ).T
    )
    distances = angular_distance_deg(directions, directions)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))


def validate_sofa_file(
    sofa_file: SofaFile, *, project_sample_rate_hz: int | None = None
) -> HrtfReport:
    """Inspect a loaded SOFA file and report what it contains."""

    issues: list[ValidationIssue] = []
    metadata: dict[str, Any] = {
        "path": str(sofa_file.path),
        "sha256": sofa_file.sha256,
        "backend": sofa_file.backend,
        "convention": sofa_file.convention,
        "convention_version": sofa_file.convention_version,
        "data_type": sofa_file.data_type,
        "measurements": sofa_file.measurements,
        "receivers": sofa_file.receivers,
        "taps": sofa_file.taps,
        "sample_rate_hz": float(sofa_file.sample_rate_hz),
        "database_name": sofa_file.database_name,
        "subject": sofa_file.listener_short_name,
        "license": sofa_file.license_note,
        "source_position_type": sofa_file.source_position_type,
        "source_position_units": sofa_file.source_position_units,
        "data_delay_shape": list(np.shape(sofa_file.data_delay)),
        "data_delay_nonzero": sofa_file.has_external_delay,
    }

    # --- structure ---------------------------------------------------------

    if sofa_file.convention and "HRIR" not in sofa_file.convention.upper():
        issues.append(
            ValidationIssue(
                "convention",
                f"convention is {sofa_file.convention!r}, not an HRIR convention",
                "warning",
            )
        )
    if sofa_file.receivers != CHANNEL_COUNT:
        issues.append(
            ValidationIssue(
                "receivers",
                f"expected {CHANNEL_COUNT} receivers for a binaural HRTF, found {sofa_file.receivers}",
            )
        )
    if sofa_file.measurements == 0 or sofa_file.taps == 0:
        issues.append(ValidationIssue("Data.IR", "the file contains no impulse responses"))
        return HrtfReport(tuple(issues), metadata)

    hrirs = np.asarray(sofa_file.impulse_responses, dtype=np.float64)
    if not np.all(np.isfinite(hrirs)):
        issues.append(ValidationIssue("Data.IR", "impulse responses contain non-finite samples"))

    # --- receiver ordering --------------------------------------------------

    if sofa_file.receiver_positions.shape[0] >= CHANNEL_COUNT:
        lateral = sofa_file.receiver_positions[:CHANNEL_COUNT, 1]
        metadata["receiver_lateral_m"] = [float(value) for value in lateral]
        if lateral[0] < lateral[1]:
            issues.append(
                ValidationIssue(
                    "ReceiverPosition",
                    "receiver 0 is on the right; SOFA orders the left ear first, so the "
                    "channels may be swapped",
                    "warning",
                )
            )

    # --- directions ---------------------------------------------------------

    try:
        positions = positions_to_canonical(
            sofa_file.source_positions,
            sofa_file.source_position_type,
            sofa_file.source_position_units,
        )
    except ValueError as error:
        issues.append(ValidationIssue("SourcePosition", str(error)))
        return HrtfReport(tuple(issues), metadata)

    if not np.all(np.isfinite(positions)):
        issues.append(ValidationIssue("SourcePosition", "some directions are non-finite"))
        positions = positions[:, np.all(np.isfinite(positions), axis=0)]

    spherical = canonical_to_spherical_deg(positions)
    metadata["azimuth_deg_range"] = [float(np.min(spherical[0])), float(np.max(spherical[0]))]
    metadata["elevation_deg_range"] = [float(np.min(spherical[1])), float(np.max(spherical[1]))]
    metadata["radius_m_range"] = [float(np.min(spherical[2])), float(np.max(spherical[2]))]

    rounded = np.round(np.vstack((spherical[0], spherical[1])).T, 3)
    unique = np.unique(rounded, axis=0)
    duplicates = rounded.shape[0] - unique.shape[0]
    metadata["duplicate_directions"] = int(duplicates)
    if duplicates:
        issues.append(
            ValidationIssue(
                "SourcePosition", f"{duplicates} duplicated direction(s) in the grid", "warning"
            )
        )

    if positions.shape[1] >= 2:
        spacing = _direction_spacing_deg(spherical)
        metadata["median_direction_spacing_deg"] = spacing
        if spacing > _SPARSE_SPACING_DEG:
            issues.append(
                ValidationIssue(
                    "SourcePosition",
                    f"directions are {spacing:.1f}° apart on average; interpolation will be coarse",
                    "warning",
                )
            )

    elevation_span = float(np.max(spherical[1]) - np.min(spherical[1]))
    if elevation_span < 1.0:
        issues.append(
            ValidationIssue(
                "SourcePosition",
                "the grid is horizontal only; elevation cannot be rendered from this asset",
                "warning",
            )
        )
    if float(np.max(spherical[2])) - float(np.min(spherical[2])) > 1e-6:
        issues.append(
            ValidationIssue(
                "SourcePosition",
                "measurements are at several radii; the renderer uses direction only",
                "warning",
            )
        )

    # --- sample rate --------------------------------------------------------

    if project_sample_rate_hz and int(project_sample_rate_hz) != int(round(sofa_file.sample_rate_hz)):
        issues.append(
            ValidationIssue(
                "Data.SamplingRate",
                f"asset is {sofa_file.sample_rate_hz:.0f} Hz and the project is "
                f"{project_sample_rate_hz} Hz; responses will be resampled once on load",
                "warning",
            )
        )

    # --- response quality ---------------------------------------------------

    peak = float(np.max(np.abs(hrirs)))
    metadata["peak"] = peak
    if peak >= _CLIPPING_LEVEL:
        issues.append(
            ValidationIssue("Data.IR", f"responses reach {peak:.3f}; they may be clipped", "warning")
        )
    if peak == 0.0:
        issues.append(ValidationIssue("Data.IR", "every impulse response is silent"))

    onsets = np.argmax(np.abs(hrirs) > 0.1 * np.maximum(
        np.max(np.abs(hrirs), axis=2, keepdims=True), 1e-12
    ), axis=2)
    metadata["onset_samples_median"] = float(np.median(onsets))
    metadata["onset_samples_range"] = [int(np.min(onsets)), int(np.max(onsets))]

    energy = np.sum(np.square(hrirs), axis=2)
    leading = np.sum(np.square(hrirs[:, :, : max(1, int(np.min(onsets)))]), axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(energy > 0.0, leading / np.maximum(energy, 1e-30), 0.0)
    if float(np.max(fraction)) > _ACAUSAL_ENERGY_FRACTION:
        issues.append(
            ValidationIssue(
                "Data.IR",
                "some responses carry energy before their onset; they may be acausal or "
                "already time-aligned",
                "warning",
            )
        )

    tail = np.sum(np.square(hrirs[:, :, -max(1, sofa_file.taps // 8) :]), axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        tail_fraction = np.where(energy > 0.0, tail / np.maximum(energy, 1e-30), 0.0)
    metadata["tail_energy_fraction_max"] = float(np.max(tail_fraction))
    if float(np.max(tail_fraction)) > 0.05:
        issues.append(
            ValidationIssue(
                "Data.IR",
                "responses still carry energy in their final eighth; the window may be too short",
                "warning",
            )
        )

    # --- provenance ---------------------------------------------------------

    if not sofa_file.license_note:
        issues.append(
            ValidationIssue(
                "License", "the file states no licence; record its provenance yourself", "warning"
            )
        )
    if not sofa_file.database_name:
        issues.append(
            ValidationIssue("DatabaseName", "the file names no database", "warning")
        )

    return HrtfReport(tuple(issues), metadata)

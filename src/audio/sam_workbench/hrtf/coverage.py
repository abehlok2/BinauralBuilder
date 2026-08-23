"""Does this dataset actually measure where the path goes?

A three-dimensional path can ask for directions a SOFA file never measured.
Many published HRTF sets are dense around the horizontal plane and thin or
empty above and below it, so a floor-to-overhead sweep spends much of its
travel in regions the dataset can only extrapolate.  The renderer will still
produce audio - it always finds a nearest measurement - and that is exactly the
problem: silent extrapolation sounds like a rendering decision rather than like
missing data.

This module answers the question before a render starts, from the dataset's
measurement directions and the path's sampled positions.  It deliberately
takes plain arrays rather than a loaded :class:`HRTFDataset`, so checking
coverage never pulls in ``sofar`` or ``h5py``.

Nothing here refuses to render.  Every finding is a warning attached to a
stable path, in the same form the rest of the workbench reports problems, so a
GUI can show it beside the control that caused it and an export can record it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..validation import ValidationIssue
from .coordinates import unit_vectors

__all__ = [
    "CoverageReport",
    "assess_path_coverage",
    "SPARSE_SPACING_FACTOR",
    "MIN_HEMISPHERE_MEASUREMENTS",
]

#: A region counts as sparse when the nearest measurement is this many times
#: further away than the dataset's own median spacing. Relative rather than
#: absolute, because a 5-degree grid and a 15-degree grid are both usable - what
#: matters is falling into a hole in whichever grid is in use.
SPARSE_SPACING_FACTOR: float = 2.5

#: Below this many measurements in a hemisphere, height in that hemisphere is
#: being invented rather than reproduced.
MIN_HEMISPHERE_MEASUREMENTS: int = 8

#: Warn when more than this fraction of the path sits in sparse regions.
_SPARSE_PATH_FRACTION: float = 0.15

#: Warn above this much direction change between two filter updates. A filter
#: switch is a crossfade between two HRIRs; past roughly this angle the two are
#: different enough that the crossfade is audible as a smear rather than as
#: motion. This is advisory: a path faster than the grid can track is a
#: creative choice and the renderer stays continuous either way, so the user is
#: told rather than overruled.
_MAX_STEP_DEG: float = 10.0


@dataclass(frozen=True)
class CoverageReport:
    """What the dataset covers, what the path asked for, and the gap between."""

    issues: tuple[ValidationIssue, ...] = ()
    #: Elevation range the dataset measures, in degrees.
    measured_elevation_deg: tuple[float, float] = (0.0, 0.0)
    #: Elevation range the path visits, in degrees.
    requested_elevation_deg: tuple[float, float] = (0.0, 0.0)
    #: Median nearest-neighbour spacing of the measurement grid, in degrees.
    median_spacing_deg: float = 0.0
    #: Measurements above and below the horizontal plane.
    upper_measurements: int = 0
    lower_measurements: int = 0
    #: Fraction of path samples whose nearest measurement is unusually far.
    sparse_fraction: float = 0.0
    #: Largest direction change between consecutive filter updates, in degrees.
    max_step_deg: float = 0.0
    #: The same, expressed against the crossfade rather than the control rate.
    max_crossfade_step_deg: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing was worth warning about."""

        return not self.issues

    def summary(self) -> str:
        """One line per warning, for a status area."""

        return "\n".join(f"{issue.path}: {issue.message}" for issue in self.issues)

    def describe(self) -> dict[str, Any]:
        """JSON form, so an export can record what it warned about."""

        return {
            "measuredElevationDeg": list(self.measured_elevation_deg),
            "requestedElevationDeg": list(self.requested_elevation_deg),
            "medianSpacingDeg": float(self.median_spacing_deg),
            "upperMeasurements": int(self.upper_measurements),
            "lowerMeasurements": int(self.lower_measurements),
            "sparseFraction": float(self.sparse_fraction),
            "maxStepDeg": float(self.max_step_deg),
            "maxCrossfadeStepDeg": float(self.max_crossfade_step_deg),
            "warnings": [
                {
                    "path": issue.path,
                    "message": issue.message,
                    "severity": issue.severity,
                }
                for issue in self.issues
            ],
        }


def _elevations_deg(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elevation of each direction, safe at the origin."""

    radius = np.linalg.norm(points, axis=-1)
    safe = np.maximum(radius, np.finfo(np.float64).tiny)
    return np.degrees(np.arcsin(np.clip(points[..., 2] / safe, -1.0, 1.0)))


def _median_spacing_deg(directions: NDArray[np.float64]) -> float:
    """Median nearest-neighbour spacing, a proxy for how dense the grid is."""

    if len(directions) < 2:
        return 180.0
    # Chunked so a large dataset does not build an n-by-n matrix at once.
    nearest = np.empty(len(directions), dtype=np.float64)
    for start in range(0, len(directions), 512):
        stop = min(len(directions), start + 512)
        cosine = np.clip(directions[start:stop] @ directions.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        angles[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.min(angles, axis=1)
    return float(np.median(nearest))


def _nearest_distance_deg(
    directions: NDArray[np.float64], queries: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Angular distance from each query to its nearest measurement."""

    result = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), 512):
        stop = min(len(queries), start + 512)
        cosine = np.clip(queries[start:stop] @ directions.T, -1.0, 1.0)
        result[start:stop] = np.degrees(np.arccos(np.max(cosine, axis=1)))
    return result


def assess_path_coverage(
    dataset_positions_m: ArrayLike,
    path_positions_m: ArrayLike,
    *,
    sample_rate_hz: float = 44100.0,
    control_interval_samples: int = 128,
    crossfade_ms: float = 10.0,
    interpolation: str = "nearest",
    path: str = "canonicalTrajectory",
) -> CoverageReport:
    """Compare a sampled path against a dataset's measurement grid.

    ``path_positions_m`` should be the path sampled at the renderer's own
    control rate, so that the direction-change checks describe the steps the
    renderer will actually take rather than an arbitrary preview resolution.
    """

    measurements = np.asarray(dataset_positions_m, dtype=np.float64)
    if measurements.ndim != 2 or measurements.shape[1] != 3:
        raise ValueError(
            f"dataset positions must be (n, 3), got shape {measurements.shape}"
        )
    points = np.asarray(path_positions_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"path positions must be (n, 3), got shape {points.shape}")
    if len(measurements) == 0 or len(points) == 0:
        raise ValueError("coverage needs at least one measurement and one position")

    issues: list[ValidationIssue] = []

    def warn(message: str, suffix: str = "") -> None:
        issues.append(
            ValidationIssue(f"{path}{suffix}", message, severity="warning")
        )

    # A path that passes exactly through the listener has no direction there;
    # those samples are dropped rather than allowed to produce a wrong angle.
    radii = np.linalg.norm(points, axis=-1)
    usable = radii > np.finfo(np.float64).eps
    if not np.any(usable):
        warn(
            "the path stays at the listener's own position, where no direction "
            "is defined; no HRTF direction can be chosen for it"
        )
        return CoverageReport(issues=tuple(issues))
    if not np.all(usable):
        warn(
            f"{int(np.sum(~usable))} of {len(points)} path samples sit exactly "
            "at the listener's position, where direction is undefined"
        )

    directions = unit_vectors(measurements)
    queries = unit_vectors(points[usable])

    measured_elevation = _elevations_deg(measurements)
    requested_elevation = _elevations_deg(points[usable])
    measured_range = (
        float(np.min(measured_elevation)),
        float(np.max(measured_elevation)),
    )
    requested_range = (
        float(np.min(requested_elevation)),
        float(np.max(requested_elevation)),
    )

    upper = int(np.sum(measured_elevation > 1.0))
    lower = int(np.sum(measured_elevation < -1.0))
    spacing = _median_spacing_deg(directions)
    nearest_deg = _nearest_distance_deg(directions, queries)
    sparse_threshold = max(SPARSE_SPACING_FACTOR * spacing, 1.0)
    sparse_fraction = float(np.mean(nearest_deg > sparse_threshold))

    # --- elevation outside measured coverage -------------------------------
    tolerance = max(spacing, 1.0)
    if requested_range[1] > measured_range[1] + tolerance:
        above = float(np.mean(requested_elevation > measured_range[1] + tolerance))
        warn(
            f"the path reaches {requested_range[1]:.1f} deg elevation but the "
            f"dataset only measures up to {measured_range[1]:.1f} deg; "
            f"{above:.0%} of the path is above measured coverage and will be "
            "extrapolated from the highest measurements available",
            ".elevation",
        )
    if requested_range[0] < measured_range[0] - tolerance:
        below = float(np.mean(requested_elevation < measured_range[0] - tolerance))
        warn(
            f"the path reaches {requested_range[0]:.1f} deg elevation but the "
            f"dataset only measures down to {measured_range[0]:.1f} deg; "
            f"{below:.0%} of the path is below measured coverage and will be "
            "extrapolated from the lowest measurements available",
            ".elevation",
        )

    # --- hemisphere sampling ------------------------------------------------
    if requested_range[1] > 1.0 and upper < MIN_HEMISPHERE_MEASUREMENTS:
        warn(
            f"the path goes above the horizontal plane but the dataset has only "
            f"{upper} measurement(s) there; height above the listener cannot be "
            "reproduced from this dataset and will be approximated",
            ".elevation",
        )
    if requested_range[0] < -1.0 and lower < MIN_HEMISPHERE_MEASUREMENTS:
        warn(
            f"the path goes below the horizontal plane but the dataset has only "
            f"{lower} measurement(s) there; height below the listener cannot be "
            "reproduced from this dataset and will be approximated",
            ".elevation",
        )

    # --- sparse regions -----------------------------------------------------
    if sparse_fraction > _SPARSE_PATH_FRACTION:
        warn(
            f"{sparse_fraction:.0%} of the path passes more than "
            f"{sparse_threshold:.1f} deg from any measurement, against a median "
            f"grid spacing of {spacing:.1f} deg; the path repeatedly crosses "
            "sparse regions of this dataset",
            ".coverage",
        )

    # --- nearest-neighbour fallback ----------------------------------------
    if str(interpolation) == "nearest":
        warn(
            "nearest-neighbour selection is in use, so direction changes step "
            "between measured points instead of interpolating across the "
            "measurement surface; this is audible on slow moving paths",
            ".interpolation",
        )

    # --- speed against the control rate ------------------------------------
    max_step = 0.0
    max_crossfade_step = 0.0
    if len(queries) > 1:
        steps = np.degrees(
            np.arccos(np.clip(np.sum(queries[1:] * queries[:-1], axis=-1), -1.0, 1.0))
        )
        # The supplied samples are one control interval apart by contract; the
        # crossfade covers a different span, so scale rather than re-sample.
        max_step = float(np.max(steps))
        interval_s = max(int(control_interval_samples), 1) / max(
            float(sample_rate_hz), 1.0
        )
        # The renderer caps a transition at the control interval so it always
        # finishes before the next one is due, so that - not ``crossfade_ms``
        # alone - is the span a blend actually covers. Reporting the uncapped
        # figure would describe smear the renderer no longer produces.
        crossfade_s = min(max(float(crossfade_ms), 0.0) / 1000.0, interval_s)
        max_crossfade_step = max_step * (
            crossfade_s / interval_s if interval_s > 0.0 else 0.0
        )
        if max_step > _MAX_STEP_DEG:
            warn(
                f"the path turns up to {max_step:.1f} deg between filter updates "
                f"({control_interval_samples} samples, {interval_s * 1000.0:.1f} ms); "
                "reduce the control interval or slow the path, or the motion will "
                "be heard as a sequence of jumps",
                ".traversal",
            )
        if crossfade_s > 0.0 and max_crossfade_step > _MAX_STEP_DEG:
            warn(
                f"the path turns up to {max_crossfade_step:.1f} deg during a "
                f"{crossfade_s * 1000.0:.1f} ms transition; the two directions "
                "being blended are far enough apart that the transition will "
                "smear rather than track the motion",
                ".traversal",
            )

    return CoverageReport(
        issues=tuple(issues),
        measured_elevation_deg=measured_range,
        requested_elevation_deg=requested_range,
        median_spacing_deg=spacing,
        upper_measurements=upper,
        lower_measurements=lower,
        sparse_fraction=sparse_fraction,
        max_step_deg=max_step,
        max_crossfade_step_deg=max_crossfade_step,
        metadata={
            "measurements": int(len(measurements)),
            "pathSamples": int(len(points)),
            "interpolation": str(interpolation),
        },
    )

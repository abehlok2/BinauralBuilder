"""Importing a simulated HRTF from Mesh2HRTF, and checking it before trusting it.

This is Route C: a head, torso and pinna scan solved numerically rather than
measured acoustically. The specification is explicit that it is an advanced
import workflow and not a dependency of the core application, so nothing here
runs a solver. What it does is receive the SOFA a solver produced and ask the
questions that catch the mistakes this route actually makes.

Those mistakes are boringly consistent, and none of them announce themselves:

* the mesh was scaled in millimetres and the solve believed metres, so the head
  is a metre wide and every interaural delay is ten times too long;
* the coordinate axes did not match the convention, so front and left are
  swapped and the whole set is rotated;
* the receivers were placed as a pair but not left-then-right, so the ears are
  the wrong way round and the image mirrors;
* the evaluation grid was dense at the equator and empty above, so elevation
  cannot work and nothing says why.

Each produces a file that loads cleanly and sounds wrong. So the validation
here compares against physical plausibility and, where a baseline is supplied,
against a known-good generic set: a simulated HRTF whose interaural cues do not
resemble any real head's is reporting a setup error, not a novel anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..validation import ValidationIssue

__all__ = [
    "IMPORT_STEPS",
    "MeshComparison",
    "compare_to_baseline",
    "import_guidance",
    "validate_simulated",
]

#: The workflow, as the specification lays it out. Presented rather than
#: automated: every step happens in the scanner or the solver, not here.
IMPORT_STEPS = (
    "Acquire a detailed head, torso and pinna scan.",
    "Clean and close the mesh without erasing pinna detail.",
    "Set scale and coordinate axes precisely: +x forward, +y left, +z up, in metres.",
    "Place the left and right receiver locations, left first.",
    "Define an evaluation sphere or grid, covering elevation as well as azimuth.",
    "Solve with Mesh2HRTF / NumCalc.",
    "Export SOFA and verify it here before use.",
    "Compare the simulated cues against a generic baseline and any measurements.",
)

#: A human head is around 0.18 m across; ear separation far outside this range
#: means the mesh scale is wrong rather than the anatomy unusual.
PLAUSIBLE_EAR_SEPARATION_M = (0.12, 0.24)

#: Interaural delay at 90 degrees for a real head, give or take.
PLAUSIBLE_MAX_ITD_US = (400.0, 900.0)

_EPSILON = 1e-12


def import_guidance() -> tuple[str, ...]:
    """The numbered workflow, for display next to the import button."""

    return tuple(f"{index + 1}. {step}" for index, step in enumerate(IMPORT_STEPS))


def validate_simulated(dataset: Any, *, strict: bool = False) -> tuple[ValidationIssue, ...]:
    """Check a simulated SOFA for the setup errors this route tends to make.

    ``strict`` promotes the plausibility warnings to errors, for a pipeline
    that would rather refuse a questionable import than carry it forward.
    """

    issues: list[ValidationIssue] = []

    def report(path: str, message: str, severity: str = "warning") -> None:
        issues.append(ValidationIssue(path, message, "error" if strict and severity == "warning" else severity))

    # --- scale, via the receivers ------------------------------------------
    receivers = np.asarray(getattr(dataset, "receiver_positions_m", np.empty((0, 3))), dtype=np.float64)
    if receivers.shape[0] < 2:
        report(
            "ReceiverPosition",
            "fewer than two receivers, so the ears cannot be identified; "
            "place the left and right receivers before solving",
            "error",
        )
    else:
        separation = float(np.linalg.norm(receivers[0] - receivers[1]))
        low, high = PLAUSIBLE_EAR_SEPARATION_M
        if separation <= _EPSILON:
            report("ReceiverPosition", "both receivers are at the same point", "error")
        elif not low <= separation <= high:
            report(
                "ReceiverPosition",
                f"the ears are {separation:.3f} m apart, outside the plausible "
                f"{low}-{high} m; the mesh scale is probably wrong (millimetres "
                "solved as metres is the usual cause)",
            )
        # --- axes, via which receiver is on which side ---------------------
        if receivers[0][1] < receivers[1][1]:
            report(
                "ReceiverPosition",
                "the first receiver is on the right; the convention is left "
                "first, so the ears are swapped and the image will mirror",
            )
        if abs(receivers[0][1]) < abs(receivers[0][0]):
            report(
                "ReceiverPosition",
                "the receivers are separated mostly along the forward axis "
                "rather than side to side; the coordinate axes do not match "
                "the +x forward, +y left convention",
            )

    # --- grid coverage ------------------------------------------------------
    positions = np.asarray(getattr(dataset, "positions_m", np.empty((0, 3))), dtype=np.float64)
    if positions.shape[0] == 0:
        report("SourcePosition", "the file contains no source directions", "error")
    else:
        from .coordinates import cartesian_to_sofa_spherical

        spherical = cartesian_to_sofa_spherical(positions)
        elevations = spherical[:, 1]
        azimuths = ((spherical[:, 0] + 180.0) % 360.0) - 180.0
        radii = spherical[:, 2]

        if float(np.ptp(elevations)) < 30.0:
            report(
                "SourcePosition",
                f"elevation spans only {float(np.ptp(elevations)):.0f} degrees; "
                "an evaluation grid confined to the horizon cannot carry "
                "elevation cues",
            )
        if float(np.ptp(azimuths)) < 180.0:
            report(
                "SourcePosition",
                f"azimuth spans only {float(np.ptp(azimuths)):.0f} degrees; "
                "the evaluation grid does not go all the way round",
            )
        if radii.size and float(np.ptp(radii)) > 0.05 * max(float(np.mean(radii)), _EPSILON):
            report(
                "SourcePosition",
                "the evaluation points are not all at one radius, so the set "
                "mixes distances; solve on a sphere unless that was intended",
                "info",
            )
        if radii.size and not 0.2 <= float(np.mean(radii)) <= 10.0:
            report(
                "SourcePosition",
                f"the evaluation radius averages {float(np.mean(radii)):.3f} m, "
                "which is implausible for an HRTF; check the mesh scale",
            )

    # --- the responses themselves ------------------------------------------
    responses = np.asarray(getattr(dataset, "ir", np.empty((0, 2, 0))), dtype=np.float64)
    if responses.size:
        if not np.all(np.isfinite(responses)):
            report("Data.IR", "the impulse responses contain NaN or infinity", "error")
        if float(np.max(np.abs(responses), initial=0.0)) <= _EPSILON:
            report("Data.IR", "the impulse responses are silent", "error")

        peak_taps = np.argmax(np.abs(responses), axis=-1)
        if int(np.min(peak_taps)) == 0:
            report(
                "Data.IR",
                "a response peaks at its very first sample, so its onset was "
                "cut; the solve output was probably truncated at the front",
            )

    return tuple(issues)


@dataclass(frozen=True)
class MeshComparison:
    """How a simulated set's interaural cues compare with a baseline."""

    max_itd_us: float
    baseline_max_itd_us: float
    itd_ratio: float
    max_ild_db: float
    baseline_max_ild_db: float
    correlation: float
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def plausible(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def describe(self) -> dict[str, Any]:
        return {
            "maxItdUs": float(self.max_itd_us),
            "baselineMaxItdUs": float(self.baseline_max_itd_us),
            "itdRatio": float(self.itd_ratio),
            "maxIldDb": float(self.max_ild_db),
            "baselineMaxIldDb": float(self.baseline_max_ild_db),
            "correlation": float(self.correlation),
            "plausible": bool(self.plausible),
            "issues": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.issues
            ],
        }


def compare_to_baseline(
    simulated: Any,
    baseline: Any | None = None,
    *,
    elevation_deg: float = 0.0,
) -> MeshComparison:
    """Compare a simulated set's interaural cues against a known-good one.

    Step eight of the workflow. A simulation whose ITD is ten times a real
    head's has a scale error, not unusual anatomy, and comparing the shape of
    the cue against azimuth catches an axis error that a peak value alone
    would not.
    """

    from ..analysis.hrtf_curves import sweep_across_azimuth

    azimuths, itd, ild = sweep_across_azimuth(
        simulated.ir, simulated.positions_m, simulated.sample_rate_hz, elevation_deg=elevation_deg
    )
    max_itd = float(np.max(np.abs(itd))) if itd.size else 0.0
    max_ild = float(np.max(np.abs(ild))) if ild.size else 0.0

    issues: list[ValidationIssue] = []
    low, high = PLAUSIBLE_MAX_ITD_US
    if itd.size and not low <= max_itd <= high:
        issues.append(
            ValidationIssue(
                "itd",
                f"the largest interaural delay is {max_itd:.0f} microseconds, outside "
                f"the {low:.0f}-{high:.0f} plausible for a human head; this is a "
                "mesh scale or coordinate error rather than an anatomy",
                "error" if max_itd > 4.0 * high or max_itd < 0.25 * low else "warning",
            )
        )

    baseline_itd_max = 0.0
    baseline_ild_max = 0.0
    correlation = float("nan")
    if baseline is not None:
        baseline_azimuths, baseline_itd, baseline_ild = sweep_across_azimuth(
            baseline.ir, baseline.positions_m, baseline.sample_rate_hz, elevation_deg=elevation_deg
        )
        baseline_itd_max = float(np.max(np.abs(baseline_itd))) if baseline_itd.size else 0.0
        baseline_ild_max = float(np.max(np.abs(baseline_ild))) if baseline_ild.size else 0.0

        if itd.size and baseline_itd.size:
            # Compare the *shape* against azimuth, which is what catches a
            # rotation or a left/right swap that peak values agree on.
            common = np.intersect1d(np.round(azimuths, 3), np.round(baseline_azimuths, 3))
            if common.size >= 4:
                def sample(values, grid, wanted):
                    order = np.argsort(grid)
                    return np.interp(wanted, np.asarray(grid)[order], np.asarray(values)[order])

                left = sample(itd, azimuths, common)
                right = sample(baseline_itd, baseline_azimuths, common)
                if np.std(left) > _EPSILON and np.std(right) > _EPSILON:
                    correlation = float(np.corrcoef(left, right)[0, 1])
                    if correlation < 0.0:
                        issues.append(
                            ValidationIssue(
                                "itd",
                                f"the interaural delay runs opposite to the baseline "
                                f"(correlation {correlation:.2f}); the ears are swapped "
                                "or the azimuth axis is reversed",
                                "error",
                            )
                        )
                    elif correlation < 0.5:
                        issues.append(
                            ValidationIssue(
                                "itd",
                                f"the interaural delay only loosely follows the baseline "
                                f"(correlation {correlation:.2f}); check the coordinate axes",
                            )
                        )

    ratio = max_itd / baseline_itd_max if baseline_itd_max > _EPSILON else float("nan")
    return MeshComparison(
        max_itd_us=max_itd,
        baseline_max_itd_us=baseline_itd_max,
        itd_ratio=ratio,
        max_ild_db=max_ild,
        baseline_max_ild_db=baseline_ild_max,
        correlation=correlation,
        issues=tuple(issues),
    )

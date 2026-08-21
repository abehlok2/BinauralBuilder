"""Structured validation and quality reporting for imported HRTFs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HRTFValidationIssue:
    path: str
    message: str
    severity: str = "error"


def validate_dataset(dataset) -> tuple[HRTFValidationIssue, ...]:
    issues: list[HRTFValidationIssue] = []
    ir = np.asarray(dataset.ir)
    if ir.ndim != 3:
        issues.append(HRTFValidationIssue("Data.IR", "must be measurements × receivers × samples"))
        return tuple(issues)
    if ir.shape[1] != 2:
        issues.append(HRTFValidationIssue("Data.IR", f"requires exactly two receivers, found {ir.shape[1]}"))
    if ir.shape[2] < 2:
        issues.append(HRTFValidationIssue("Data.IR", "HRIRs need at least two taps"))
    if not np.all(np.isfinite(ir)):
        issues.append(HRTFValidationIssue("Data.IR", "contains NaN or infinity"))
    if np.max(np.abs(ir), initial=0.0) > 1.0:
        issues.append(HRTFValidationIssue("Data.IR", "contains samples above 0 dBFS", "warning"))
    peak_taps = np.argmax(np.abs(ir), axis=-1)
    if np.any(peak_taps == 0):
        issues.append(HRTFValidationIssue(
            "Data.IR", "one or more HRIR peaks occur at tap zero; inspect causality/onset", "warning"
        ))
    tail_start = max(1, int(ir.shape[-1] * 0.9))
    tail_ratio = np.sum(np.square(ir[..., tail_start:])) / max(np.sum(np.square(ir)), 1e-30)
    if tail_ratio > 0.05:
        issues.append(HRTFValidationIssue(
            "Data.IR", f"last 10% contains {tail_ratio:.1%} of HRIR energy; the response may be truncated",
            "warning",
        ))
    positions = np.asarray(dataset.positions_m)
    if positions.shape != (ir.shape[0], 3):
        issues.append(HRTFValidationIssue("SourcePosition", "count must match Data.IR measurements"))
    elif not np.all(np.isfinite(positions)):
        issues.append(HRTFValidationIssue("SourcePosition", "contains NaN or infinity"))
    elif len(np.unique(np.round(positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12), 8), axis=0)) < len(positions):
        issues.append(HRTFValidationIssue("SourcePosition", "contains duplicate directions", "warning"))
    receivers = np.asarray(dataset.receiver_positions_m)
    if receivers.shape != (2, 3):
        issues.append(HRTFValidationIssue(
            "ReceiverPosition", "must identify exactly two three-dimensional receiver positions"
        ))
    else:
        # Canonical receiver order is left (+y), then right (-y). Refuse an
        # ambiguous or swapped file instead of silently exchanging ears.
        if not (receivers[0, 1] > receivers[1, 1]):
            issues.append(HRTFValidationIssue(
                "ReceiverPosition", "receiver order must be left (+y), then right (-y)"
            ))
        if np.linalg.norm(receivers[0] - receivers[1]) < 0.05:
            issues.append(HRTFValidationIssue(
                "ReceiverPosition", "receiver separation is implausibly small", "warning"
            ))
    if not np.isfinite(dataset.sample_rate_hz) or dataset.sample_rate_hz <= 0:
        issues.append(HRTFValidationIssue("Data.SamplingRate", "must be finite and positive"))
    delay = np.asarray(dataset.delay_samples)
    if delay.shape not in {(ir.shape[0], 2), (1, 2)}:
        issues.append(HRTFValidationIssue("Data.Delay", f"unsupported shape {delay.shape}"))
    if not np.all(np.isfinite(delay)):
        issues.append(HRTFValidationIssue("Data.Delay", "contains NaN or infinity"))
    if dataset.convention != "SimpleFreeFieldHRIR":
        issues.append(HRTFValidationIssue("SOFAConventions", f"expected SimpleFreeFieldHRIR, found {dataset.convention!r}"))
    if dataset.metadata.get("sofar_verified") is False:
        issues.append(HRTFValidationIssue(
            "SOFA", f"sofar convention verification did not pass: {dataset.metadata.get('sofar_verification')}",
            "warning",
        ))
    return tuple(issues)


# ---------------------------------------------------------------------------
# Quality status
# ---------------------------------------------------------------------------

#: The three states the workbench reports for a selected asset.
NORMAL = "normal"
SUSPECTED_OUTLIER = "suspected outlier"
MISSING_RESOURCES = "missing resources"

QUALITY_STATES = (NORMAL, SUSPECTED_OUTLIER, MISSING_RESOURCES)


def quality_from_issues(
    issues: "tuple[HRTFValidationIssue, ...]", *, resources_available: bool = True
) -> str:
    """Classify an asset for the workbench's quality indicator.

    Missing resources outrank everything: an asset that could not be read at
    all is not an outlier, it is absent, and telling the two apart is the
    difference between "check this subject" and "install the dataset".
    """

    if not resources_available:
        return MISSING_RESOURCES
    if any(issue.severity == "error" for issue in issues):
        return MISSING_RESOURCES
    if issues:
        return SUSPECTED_OUTLIER
    return NORMAL


__all__ = [
    "HRTFValidationIssue",
    "MISSING_RESOURCES",
    "NORMAL",
    "QUALITY_STATES",
    "SUSPECTED_OUTLIER",
    "quality_from_issues",
    "validate_dataset",
]

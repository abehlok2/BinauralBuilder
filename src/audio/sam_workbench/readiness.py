"""Is this configuration ready to render, and if not, what should be done.

Validation answers "is this legal". Readiness answers a different question: a
project can be entirely legal and still not produce what its author expects -
the dataset moved, the path goes where the dataset never measured, a route
names a source that was deleted, a control is set that this renderer ignores.

Every check here returns something the user can act on. A warning that only
says a thing is wrong, without saying which thing or what to do, costs more
attention than it saves, so each carries the path it concerns and a remedy.

Qt-free, so the same checks back the dialog, the export path and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .validation import ValidationIssue

__all__ = ["ReadinessReport", "assess_readiness"]

#: Above this share of the real-time budget, an offline render is honest but a
#: preview will not keep up, and the user should know before they press play.
_COST_WARNING_LOAD = 0.8


@dataclass(frozen=True)
class ReadinessReport:
    """What stands between this configuration and the render it implies."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ready(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if not self.issues:
            return "Ready to render."
        return "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"path": issue.path, "message": issue.message, "severity": issue.severity}
            for issue in self.issues
        ]


def _asset_issues(params: Mapping[str, Any], uses_sofa: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    asset = str(params.get("hrtfAsset", "") or "")
    if uses_sofa and not asset:
        issues.append(
            ValidationIssue(
                "hrtfAsset",
                "this renderer convolves with a SOFA dataset; choose one in the "
                "HRTF Lab before rendering",
                "error",
            )
        )
        return issues
    if not asset:
        return issues

    if not Path(asset).exists():
        issues.append(
            ValidationIssue(
                "hrtfAsset",
                f"the dataset {asset} is no longer at that path; re-select it or "
                "restore the file",
                "error" if uses_sofa else "warning",
            )
        )
        return issues

    expected = str(params.get("hrtfAssetHash", "") or "")
    if expected and uses_sofa:
        try:
            from .hrtf.sofa_io import load_sofa

            actual = load_sofa(asset).content_hash
        except Exception as error:  # noqa: BLE001 - reported as a readiness issue
            issues.append(
                ValidationIssue("hrtfAsset", f"the dataset could not be read: {error}", "error")
            )
            return issues
        if actual.lower() != expected.lower():
            issues.append(
                ValidationIssue(
                    "hrtfAsset",
                    "the dataset at this path is not the one this project was "
                    "authored against; the render will not match what was "
                    "approved. Re-select it to accept the new file.",
                    "error",
                )
            )
    return issues


def _coverage_issues(params: Mapping[str, Any]) -> list[ValidationIssue]:
    """What the chosen dataset cannot support about the chosen path."""

    asset = str(params.get("hrtfAsset", "") or "")
    trajectory = params.get("canonicalTrajectory")
    if not asset or not isinstance(trajectory, Mapping) or not Path(asset).exists():
        return []
    try:
        import numpy as np

        from .hrtf.coverage import assess_path_coverage
        from .hrtf.sofa_io import load_sofa
        from .trajectory import path_model_from_dict

        dataset = load_sofa(asset)
        model = path_model_from_dict(dict(trajectory))
        samples = model.positions(np.linspace(0.0, float(model.duration_s), 256))
        return list(assess_path_coverage(dataset.positions_m, samples).issues)
    except Exception:  # noqa: BLE001 - advice must not block a render
        return []


def _cost_issues(summary) -> list[ValidationIssue]:
    if not summary.cost:
        return []
    if "Offline" in summary.cost:
        return [
            ValidationIssue(
                "rendererMode",
                f"{summary.cost} This configuration is for offline export; a "
                "live preview will not keep up. Widen the angular tolerance or "
                "use a cheaper interpolation to audition it.",
                "warning",
            )
        ]
    return []


def _peak_issues(peak: float | None) -> list[ValidationIssue]:
    if peak is None:
        return []
    if peak >= 1.0:
        return [
            ValidationIssue(
                "output.peak",
                f"the render reaches {peak:.3f} and will clip; lower the "
                "amplitude or enable the limiter",
                "error",
            )
        ]
    if peak > 0.99:
        return [
            ValidationIssue(
                "output.peak",
                f"the render peaks at {peak:.3f}, close enough to full scale "
                "that resampling or encoding may clip it",
                "warning",
            )
        ]
    return []


def assess_readiness(
    params: Mapping[str, Any],
    *,
    scene: Mapping[str, Any] | None = None,
    sample_rate_hz: int = 44_100,
    peak: float | None = None,
) -> ReadinessReport:
    """Everything actionable about this configuration, in one report."""

    from .flow import summarize_flow
    from .render.registry import REGISTRY
    from .scene_state import validate_scene

    identifier = str(params.get("rendererMode", "abstract_pm")).lower()
    definition = REGISTRY.get(identifier) if identifier in REGISTRY else None
    uses_sofa = bool(
        definition is not None and any(a.kind == "sofa" for a in definition.assets)
    )

    issues: list[ValidationIssue] = []
    issues.extend(_asset_issues(params, uses_sofa))
    issues.extend(_coverage_issues(params) if uses_sofa else [])

    summary = summarize_flow(params, scene=scene, sample_rate_hz=sample_rate_hz)

    # A control that holds a value this renderer never reads. Reported once per
    # setting, naming it, because the user cannot tell an ignored setting from
    # a disabled one by looking at its value.
    for name in summary.inactive:
        issues.append(
            ValidationIssue(
                name,
                f"{summary.renderer_label} does not read this, so it will not "
                "affect the render. Change renderer, or clear it to avoid "
                "carrying a setting that does nothing.",
                "warning",
            )
        )

    if uses_sofa and not str(params.get("headphoneAsset", "") or ""):
        issues.append(
            ValidationIssue(
                "headphoneAsset",
                "no headphone correction profile is set; the measured cues will "
                "reach the ears through whatever response the headphones have",
                "info",
            )
        )

    issues.extend(_cost_issues(summary))
    issues.extend(_peak_issues(peak))

    if scene:
        issues.extend(validate_scene(scene))

    return ReadinessReport(issues=tuple(issues))

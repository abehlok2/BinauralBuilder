"""What the current configuration will actually do, in one summary.

The workbench spreads a voice's configuration across tabs: the renderer sits in
the header, the dataset in the HRTF Lab, the trajectory in the path editor, the
buses in the Scene tab. Each is legible on its own and the whole is not, so it
is possible to set an interpolation mode that a non-HRTF renderer will never
read, or to leave a path enabled that nothing consumes, and see nothing amiss.

This module answers "what happens when I press render?" from the same
parameters production reads. It is deliberately Qt-free: the summary is derived
from the voice and scene dictionaries and can be asserted directly, so the
widget that shows it has nothing to decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .validation import ValidationIssue

__all__ = ["FlowStage", "FlowSummary", "summarize_flow", "SIGNAL_CHAIN_LABELS"]

#: The stages named in the specification's signal-flow line, in order.
SIGNAL_CHAIN_LABELS = (
    "Source",
    "Path",
    "Renderer",
    "Cue transform",
    "Headphone correction",
    "Output",
)


@dataclass(frozen=True)
class FlowStage:
    """One stage of the chain, and whether it does anything here."""

    name: str
    active: bool
    detail: str = ""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "active": bool(self.active), "detail": self.detail}


@dataclass(frozen=True)
class FlowSummary:
    """The state of the whole chain, for a status strip or a manifest."""

    renderer_id: str
    renderer_label: str
    stages: tuple[FlowStage, ...] = ()
    asset: str = ""
    asset_hash: str = ""
    interpolation: str = ""
    path_status: str = ""
    scene_status: str = ""
    cost: str = ""
    warnings: tuple[ValidationIssue, ...] = ()
    #: Controls that carry a value the selected renderer will not read.
    inactive: tuple[str, ...] = ()

    @property
    def active_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.active)

    def chain_text(self) -> str:
        """The signal chain, with inactive stages struck through in words.

        An inactive stage is shown rather than omitted: a chain that changes
        length as options are toggled is harder to read than one where each
        stage is always in the same place and says whether it is doing
        anything.
        """

        return " → ".join(
            stage.name if stage.active else f"({stage.name})" for stage in self.stages
        )

    def describe(self) -> dict[str, Any]:
        return {
            "renderer": self.renderer_id,
            "rendererLabel": self.renderer_label,
            "chain": self.chain_text(),
            "stages": [stage.describe() for stage in self.stages],
            "asset": self.asset,
            "assetSha256": self.asset_hash,
            "interpolation": self.interpolation,
            "path": self.path_status,
            "scene": self.scene_status,
            "cost": self.cost,
            "inactive": list(self.inactive),
            "warnings": [
                {"path": issue.path, "message": issue.message, "severity": issue.severity}
                for issue in self.warnings
            ],
        }


def _path_status(params: Mapping[str, Any]) -> tuple[str, bool]:
    trajectory = params.get("canonicalTrajectory")
    if isinstance(trajectory, Mapping) and trajectory:
        geometry = trajectory.get("geometry")
        kind = ""
        if isinstance(geometry, Mapping):
            kind = str(geometry.get("type", ""))
        traversal = trajectory.get("traversal")
        frame = ""
        if isinstance(traversal, Mapping):
            frame = str(traversal.get("frame", "") or "")
        described = kind or "custom"
        if frame:
            described = f"{described} ({frame})"
        return described, True
    if params.get("pathType"):
        return f"legacy {params['pathType']}", True
    return "none", False


def _scene_status(scene: Mapping[str, Any] | None) -> str:
    if not scene:
        return "no scene"
    sources = scene.get("sources") or ()
    routing = scene.get("routing") or {}
    buses = routing.get("buses") or scene.get("buses") or ()
    soloed = [
        entry
        for entry in (routing.get("sources") or ())
        if isinstance(entry, Mapping) and entry.get("solo")
    ]
    parts = [f"{len(sources)} source(s)", f"{len(buses)} bus(es)"]
    if soloed:
        parts.append(f"{len(soloed)} soloed")
    return ", ".join(parts)


def summarize_flow(
    params: Mapping[str, Any],
    *,
    scene: Mapping[str, Any] | None = None,
    sample_rate_hz: int = 44_100,
    issues: tuple[ValidationIssue, ...] = (),
    asset_hash: str = "",
) -> FlowSummary:
    """Describe what this configuration will do when it is rendered.

    ``issues`` is passed in rather than recomputed so the summary shows exactly
    the warnings the dialog is already showing, instead of a second opinion.
    """

    from .render.registry import REGISTRY

    identifier = str(params.get("rendererMode", "abstract_pm")).lower()
    definition = REGISTRY.get(identifier) if identifier in REGISTRY else None
    capabilities = definition.capabilities if definition is not None else None
    label = (capabilities.label if capabilities else identifier) or identifier

    options = params.get("hrtfOptions")
    options = options if isinstance(options, Mapping) else {}
    uses_sofa = bool(
        definition is not None and any(a.kind == "sofa" for a in definition.assets)
    )
    asset = str(params.get("hrtfAsset", "") or "") if uses_sofa else ""
    interpolation = str(options.get("interpolation", "") or "") if uses_sofa else ""

    path_text, has_path = _path_status(params)
    reads_path = bool(capabilities.consumes_trajectory) if capabilities else False

    cue = options.get("cue")
    cue_set = isinstance(cue, Mapping) and not cue.get("neutral", True)
    supports_cue = bool(capabilities.supports_cue_modification) if capabilities else False

    headphone = str(params.get("headphoneAsset", "") or "")

    stages = (
        FlowStage("Source", True, str(params.get("synthFunction", "") or "SAM")),
        FlowStage("Path", has_path and reads_path, path_text),
        FlowStage("Renderer", True, label),
        FlowStage("Cue transform", cue_set and supports_cue, "" if supports_cue else "not available"),
        FlowStage("Headphone correction", bool(headphone) and uses_sofa, headphone),
        FlowStage("Output", True, f"{sample_rate_hz} Hz"),
    )

    # A control holding a value that this renderer will never read. Reporting
    # it is the difference between a setting that is off and one that is
    # ignored - the user cannot tell those apart from the value alone.
    inactive: list[str] = []
    if has_path and not reads_path:
        inactive.append("canonicalTrajectory")
    if not uses_sofa:
        if params.get("hrtfAsset"):
            inactive.append("hrtfAsset")
        if options.get("interpolation"):
            inactive.append("hrtfOptions.interpolation")
    if cue_set and not supports_cue:
        inactive.append("hrtfOptions.cue")
    if headphone and not uses_sofa:
        inactive.append("headphoneAsset")

    cost = ""
    if uses_sofa:
        from .cost import estimate_cost

        try:
            cost = estimate_cost(
                renderer=identifier,
                interpolation=interpolation or "nearest",
                sample_rate_hz=float(sample_rate_hz),
            ).summary()
        except Exception:  # pragma: no cover - an estimate must never block the UI
            cost = ""

    return FlowSummary(
        renderer_id=identifier,
        renderer_label=label,
        stages=stages,
        asset=asset,
        asset_hash=asset_hash,
        interpolation=interpolation,
        path_status=path_text,
        scene_status=_scene_status(scene),
        cost=cost,
        warnings=tuple(issues),
        inactive=tuple(inactive),
    )

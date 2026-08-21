"""A modulation matrix: which modulators drive which parameters, and how much.

Rows are modulators, columns are targets, and each cell carries depth,
polarity, a mapping curve, and whether it is switched on. That is the shape the
expert view draws, so it is the shape the data takes.

The one thing this must not permit is a cycle. A modulator whose depth
eventually feeds back into itself has no defined value - evaluating it requires
its own result - and the failure is not obvious from the matrix, because each
individual cell looks reasonable. So a route is checked *before* it is
accepted, and adding one that would close a loop is refused with the loop
spelled out rather than accepted and left to misbehave at render time.

Depth is applied to a normalised modulator value and added to the target's base
value; polarity flips the direction without needing a negative depth, so the
matrix can show sign separately from amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .stages import CURVES, apply_curve
from .validation import ValidationIssue

__all__ = [
    "ModulationMatrix",
    "ModulationRoute",
    "ModulationCycleError",
    "search_parameters",
]

_EPSILON = 1e-12


class ModulationCycleError(ValueError):
    """Raised when a route would make a modulator depend on itself."""

    def __init__(self, cycle: tuple[str, ...]) -> None:
        self.cycle = tuple(cycle)
        super().__init__("modulation cycle: " + " -> ".join(self.cycle))


@dataclass(frozen=True)
class ModulationRoute:
    """One cell of the matrix."""

    modulator_id: str
    target_id: str
    parameter_path: str
    depth: float = 0.0
    #: +1 or -1. Kept separate from depth so the matrix can show sign apart
    #: from amount, which is how the expert view presents it.
    polarity: int = 1
    curve: str = "linear"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.modulator_id or not self.target_id or not self.parameter_path:
            raise ValueError("a route needs a modulator, a target, and a parameter path")
        if self.polarity not in (1, -1):
            raise ValueError(f"polarity must be +1 or -1, got {self.polarity!r}")
        if self.curve not in CURVES:
            raise ValueError(f"unknown curve {self.curve!r}; expected one of {CURVES}")
        if not math.isfinite(float(self.depth)):
            raise ValueError("depth must be finite")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.modulator_id, self.target_id, self.parameter_path)

    @property
    def is_active(self) -> bool:
        return self.enabled and abs(self.depth) > _EPSILON

    def apply(self, modulator_value, base_value):
        """Offset a base value by this route's contribution.

        The modulator is expected in 0 to 1; the curve shapes it, polarity and
        depth scale it, and the result is added to the base.
        """

        shaped = apply_curve(modulator_value, self.curve)
        return base_value + float(self.depth) * float(self.polarity) * shaped

    def describe(self) -> dict[str, Any]:
        return {
            "modulatorId": self.modulator_id,
            "targetId": self.target_id,
            "parameterPath": self.parameter_path,
            "depth": float(self.depth),
            "polarity": int(self.polarity),
            "curve": self.curve,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModulationRoute":
        return cls(
            modulator_id=str(data["modulatorId"]),
            target_id=str(data["targetId"]),
            parameter_path=str(data["parameterPath"]),
            depth=float(data.get("depth", 0.0)),
            polarity=int(data.get("polarity", 1)),
            curve=str(data.get("curve", "linear")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class ModulationMatrix:
    """Every route, with the guarantee that none of them close a loop."""

    routes: tuple[ModulationRoute, ...] = ()
    schema_version: int = 1

    # --- structure ----------------------------------------------------------

    @property
    def modulators(self) -> tuple[str, ...]:
        return tuple(sorted({route.modulator_id for route in self.routes}))

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(sorted({route.target_id for route in self.routes}))

    def routes_for_target(self, target_id: str) -> tuple[ModulationRoute, ...]:
        return tuple(route for route in self.routes if route.target_id == target_id)

    def routes_from(self, modulator_id: str) -> tuple[ModulationRoute, ...]:
        return tuple(route for route in self.routes if route.modulator_id == modulator_id)

    def route(
        self, modulator_id: str, target_id: str, parameter_path: str
    ) -> ModulationRoute | None:
        for candidate in self.routes:
            if candidate.key == (modulator_id, target_id, parameter_path):
                return candidate
        return None

    # --- cycles -------------------------------------------------------------

    def _edges(self, extra: ModulationRoute | None = None) -> dict[str, set[str]]:
        """Modulator to target, as a graph over object identifiers."""

        edges: dict[str, set[str]] = {}
        for route in (*self.routes, *(() if extra is None else (extra,))):
            if not route.enabled:
                # A disabled route cannot carry a value, so it cannot close a
                # loop; refusing it would block a legitimate parked setting.
                continue
            edges.setdefault(route.modulator_id, set()).add(route.target_id)
        return edges

    def find_cycle(self, extra: ModulationRoute | None = None) -> tuple[str, ...]:
        """The first loop found, or an empty tuple. Depth-first, so it is stable."""

        edges = self._edges(extra)
        visiting: list[str] = []
        state: dict[str, int] = {}

        def walk(node: str) -> tuple[str, ...]:
            state[node] = 1  # on the current path
            visiting.append(node)
            for neighbour in sorted(edges.get(node, ())):
                if state.get(neighbour, 0) == 1:
                    start = visiting.index(neighbour)
                    return (*visiting[start:], neighbour)
                if state.get(neighbour, 0) == 0:
                    found = walk(neighbour)
                    if found:
                        return found
            visiting.pop()
            state[node] = 2  # finished
            return ()

        for node in sorted(edges):
            if state.get(node, 0) == 0:
                found = walk(node)
                if found:
                    return found
        return ()

    def would_cycle(self, route: ModulationRoute) -> bool:
        """Whether adding this route would close a loop. Ask before committing."""

        return bool(self.find_cycle(route))

    def with_route(self, route: ModulationRoute) -> "ModulationMatrix":
        """Add or replace a route, refusing one that would close a loop."""

        cycle = self.find_cycle(route)
        if cycle:
            raise ModulationCycleError(cycle)
        kept = tuple(candidate for candidate in self.routes if candidate.key != route.key)
        return replace(self, routes=(*kept, route))

    def without_route(
        self, modulator_id: str, target_id: str, parameter_path: str
    ) -> "ModulationMatrix":
        key = (modulator_id, target_id, parameter_path)
        return replace(self, routes=tuple(r for r in self.routes if r.key != key))

    # --- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        modulator_values: Mapping[str, float],
        base_values: Mapping[tuple[str, str], float] | None = None,
    ) -> dict[tuple[str, str], float]:
        """Resolve every driven parameter for one set of modulator values.

        Several routes may reach the same parameter; their contributions add,
        which is what a matrix means. A modulator with no value is treated as
        zero rather than raising, so a partially built scene still evaluates.
        """

        resolved = dict(base_values or {})
        for route in self.routes:
            if not route.is_active:
                continue
            value = float(modulator_values.get(route.modulator_id, 0.0))
            key = (route.target_id, route.parameter_path)
            resolved[key] = route.apply(value, resolved.get(key, 0.0))
        return resolved

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        cycle = self.find_cycle()
        if cycle:
            issues.append(
                ValidationIssue("modulation", "modulation cycle: " + " -> ".join(cycle))
            )
        for position, route in enumerate(self.routes):
            if route.modulator_id == route.target_id:
                issues.append(
                    ValidationIssue(
                        f"modulation[{position}]",
                        f"{route.modulator_id!r} modulates itself",
                    )
                )
            if route.enabled and abs(route.depth) <= _EPSILON:
                issues.append(
                    ValidationIssue(
                        f"modulation[{position}].depth",
                        "this route is enabled but its depth is zero, so it does nothing",
                        "warning",
                    )
                )
        return tuple(issues)

    # --- serialization ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "schemaVersion": int(self.schema_version),
            "routes": [route.describe() for route in self.routes],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ModulationMatrix":
        values = dict(data or {})
        return cls(
            routes=tuple(
                ModulationRoute.from_mapping(entry) for entry in values.get("routes", ())
            ),
            schema_version=int(values.get("schemaVersion", 1)),
        )

    def as_grid(self) -> dict[str, dict[str, ModulationRoute]]:
        """Rows by modulator, columns by target, as the expert view draws it."""

        grid: dict[str, dict[str, ModulationRoute]] = {}
        for route in self.routes:
            column = f"{route.target_id}.{route.parameter_path}"
            grid.setdefault(route.modulator_id, {})[column] = route
        return grid


def search_parameters(query: str, *, is_transition: bool = False, limit: int = 20):
    """Find SAM parameters by name, alias, label, or tooltip.

    The registry is large enough that scrolling it is not a way to find
    anything, which is the whole reason the specification asks for search
    alongside the disclosure modes. Matches on the stored name rank above
    matches buried in a tooltip, so typing a parameter's actual name finds it
    first.
    """

    from .parameters import SAM2_FIELDS

    needle = str(query).strip().lower()
    if not needle:
        return ()

    scored: list[tuple[int, str, Any]] = []
    for entry in SAM2_FIELDS:
        names = [name for name in entry.names_for(is_transition) if name]
        haystacks = (
            (0, [name.lower() for name in names]),
            (1, [str(getattr(entry, "label", "")).lower()]),
            (2, [str(alias).lower() for alias in getattr(entry, "aliases", ())]),
            (3, [str(getattr(entry, "tooltip", "")).lower()]),
        )
        best: int | None = None
        for rank, texts in haystacks:
            for text in texts:
                if not text:
                    continue
                if text == needle:
                    best = min(best, rank) if best is not None else rank
                elif needle in text:
                    # A containment match is weaker than an exact one.
                    candidate = rank + 4
                    best = min(best, candidate) if best is not None else candidate
        if best is not None:
            scored.append((best, names[0] if names else entry.name, entry))

    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(entry for _, _, entry in scored[: max(1, int(limit))])

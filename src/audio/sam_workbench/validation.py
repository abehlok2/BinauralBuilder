"""Aggregate validation primitives for SAM workbench documents.

Validation never stops at the first problem: a collector gathers every issue,
each tagged with a stable dotted/indexed path such as ``sources[1].id`` so the
GUI can bind a message to the field that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "ValidationIssue",
    "ProjectValidationError",
    "ValidationCollector",
]


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a project document."""

    path: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.path}: {self.message}"


class ProjectValidationError(ValueError):
    """Raised with every issue found, not just the first one."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        if self.issues:
            summary = "; ".join(str(issue) for issue in self.issues)
        else:  # pragma: no cover - defensive
            summary = "invalid project"
        super().__init__(f"invalid SAM project ({len(self.issues)} issue(s)): {summary}")

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.issues)


class ValidationCollector:
    """Collects :class:`ValidationIssue` values under a nested path prefix."""

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self._issues: list[ValidationIssue] = []

    # --- path handling -----------------------------------------------------

    def child(self, name: str) -> "ValidationCollector":
        """Return a collector that shares this one's issue list under ``name``."""

        nested = ValidationCollector(self._join(name))
        nested._issues = self._issues
        return nested

    def index(self, name: str, position: int) -> "ValidationCollector":
        """Return a nested collector for ``name[position]``."""

        return self.child(f"{name}[{position}]")

    def _join(self, name: str) -> str:
        if not self._prefix:
            return name
        if name.startswith("["):
            return f"{self._prefix}{name}"
        return f"{self._prefix}.{name}"

    # --- reporting ---------------------------------------------------------

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(self._issues)

    def add(self, path: str, message: str, severity: str = "error") -> None:
        self._issues.append(ValidationIssue(self._join(path) if path else self._prefix, message, severity))

    def raise_if_any(self) -> None:
        """Raise :class:`ProjectValidationError` when any issue was collected."""

        if self._issues:
            raise ProjectValidationError(self._issues)

    # --- typed checks ------------------------------------------------------

    def check_text(self, path: str, value: Any, *, allow_empty: bool = False) -> None:
        if not isinstance(value, str):
            self.add(path, f"expected a string, got {type(value).__name__}")
        elif not allow_empty and not value.strip():
            self.add(path, "must not be empty")

    def check_number(
        self,
        path: str,
        value: Any,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
        allow_none: bool = False,
    ) -> None:
        if value is None:
            if not allow_none:
                self.add(path, "must not be null")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.add(path, f"expected a number, got {type(value).__name__}")
            return
        number = float(value)
        if not math.isfinite(number):
            self.add(path, "must be finite")
            return
        if minimum is not None:
            if exclusive_minimum and number <= minimum:
                self.add(path, f"must be greater than {minimum:g}")
            elif not exclusive_minimum and number < minimum:
                self.add(path, f"must be at least {minimum:g}")
        if maximum is not None and number > maximum:
            self.add(path, f"must be at most {maximum:g}")

    def check_integer(
        self,
        path: str,
        value: Any,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        choices: Sequence[int] | None = None,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            self.add(path, f"expected an integer, got {type(value).__name__}")
            return
        if choices is not None and value not in choices:
            allowed = ", ".join(str(choice) for choice in choices)
            self.add(path, f"must be one of {allowed}")
            return
        if minimum is not None and value < minimum:
            self.add(path, f"must be at least {minimum}")
        if maximum is not None and value > maximum:
            self.add(path, f"must be at most {maximum}")

    def check_boolean(self, path: str, value: Any) -> None:
        if not isinstance(value, bool):
            self.add(path, f"expected a boolean, got {type(value).__name__}")

    def check_choice(self, path: str, value: Any, choices: Sequence[str]) -> None:
        if value not in choices:
            allowed = ", ".join(repr(choice) for choice in choices)
            self.add(path, f"must be one of {allowed}")

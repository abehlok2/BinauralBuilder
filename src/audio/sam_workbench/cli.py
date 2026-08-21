"""Headless ``new``/``validate`` command-line shell for SAM projects.

The shell exists so a project can be created and checked without PyQt5, an
audio device, or any Rust component:

.. code-block:: console

    python -m src.audio.sam_workbench new session.sam.json --name "Reference SAM"
    python -m src.audio.sam_workbench validate session.sam.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .conventions import SUPPORTED_SAMPLE_RATES_HZ
from .model import Project, ProjectValidationError, load_project, save_project
from .version import PACKAGE_VERSION, SCHEMA_VERSION

__all__ = ["main", "build_parser"]

PROGRAM_NAME = "sam-workbench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Create and validate SAM workbench projects without a GUI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM_NAME} {PACKAGE_VERSION} (schema {SCHEMA_VERSION})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    new_command = commands.add_parser("new", help="write a new project with safe defaults")
    new_command.add_argument("path", type=Path, help="destination JSON file")
    new_command.add_argument("--name", default=None, help="project name")
    new_command.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        metavar="HZ",
        help=f"project sample rate (validated; common values: {', '.join(str(rate) for rate in SUPPORTED_SAMPLE_RATES_HZ)})",
    )
    new_command.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file",
    )

    validate_command = commands.add_parser("validate", help="validate an existing project")
    validate_command.add_argument("path", type=Path, help="project JSON file")

    return parser


def _report_issues(error: ProjectValidationError, path: Path) -> None:
    print(f"Invalid SAM project: {path}", file=sys.stderr)
    for issue in error.issues:
        location = issue.path or "<project>"
        print(f"  {location}: {issue.message}", file=sys.stderr)


def _command_new(arguments: argparse.Namespace) -> int:
    path: Path = arguments.path
    if path.exists() and not arguments.force:
        print(f"Refusing to overwrite existing file: {path} (use --force)", file=sys.stderr)
        return 1

    project = Project()
    if arguments.name is not None:
        project = replace(project, name=arguments.name)
    if arguments.sample_rate is not None:
        project = replace(project, audio=replace(project.audio, sample_rate_hz=arguments.sample_rate))

    try:
        save_project(project, path)
    except ProjectValidationError as error:
        _report_issues(error, path)
        return 1
    except OSError as error:
        print(f"Could not write {path}: {error}", file=sys.stderr)
        return 1

    print(f"Created SAM project {path} (schema {project.schema_version}, {project.audio.sample_rate_hz} Hz)")
    return 0


def _command_validate(arguments: argparse.Namespace) -> int:
    path: Path = arguments.path
    try:
        project = load_project(path)
    except ProjectValidationError as error:
        _report_issues(error, path)
        return 1
    except (OSError, FileNotFoundError) as error:
        print(f"Could not read {path}: {error}", file=sys.stderr)
        return 1

    print(
        f"Valid SAM project {path}: {project.name!r} "
        f"(schema {project.schema_version}, {len(project.sources)} source(s), "
        f"{project.audio.sample_rate_hz} Hz)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line shell and return a process exit status."""

    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "new":
        return _command_new(arguments)
    if arguments.command == "validate":
        return _command_validate(arguments)
    parser.error(f"unknown command {arguments.command!r}")  # pragma: no cover - argparse guards this
    return 2


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

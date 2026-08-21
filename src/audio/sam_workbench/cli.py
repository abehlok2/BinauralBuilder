"""Headless ``new``/``validate`` command-line shell for SAM projects.

The shell exists so a project can be created and checked without PyQt5, an
audio device, or any Rust component:

.. code-block:: console

    python -m src.audio.sam_workbench new session.sam.json --name "Reference SAM"
    python -m src.audio.sam_workbench validate session.sam.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

# When this file is executed directly (``python cli.py``), Python only adds the
# workbench directory to ``sys.path``.  Add the repository root so the same
# canonical absolute imports work both as a script and as a package module.
if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.audio.sam_workbench.conventions import SUPPORTED_SAMPLE_RATES_HZ
from src.audio.sam_workbench.export import DEFAULT_SUBTYPE, export_wav
from src.audio.sam_workbench.model import Project, ProjectValidationError, Source, load_project, save_project
from src.audio.sam_workbench.version import PACKAGE_VERSION, SCHEMA_VERSION

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
        "--with-source",
        action="store_true",
        help="add one exact-symmetric SAM source so the project renders audibly",
    )
    new_command.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file",
    )

    validate_command = commands.add_parser("validate", help="validate an existing project")
    validate_command.add_argument("path", type=Path, help="project JSON file")

    render_command = commands.add_parser("render", help="render a project to a WAV file plus manifest")
    render_command.add_argument("path", type=Path, help="project JSON file")
    render_command.add_argument("output", type=Path, help="destination WAV file")
    render_command.add_argument(
        "--duration", type=float, required=True, metavar="SECONDS", help="render length in seconds"
    )
    render_command.add_argument(
        "--block-size",
        type=int,
        default=None,
        metavar="FRAMES",
        help="offline block size; the rendered samples do not depend on it",
    )
    render_command.add_argument(
        "--subtype",
        default=DEFAULT_SUBTYPE,
        help=f"soundfile subtype for the WAV data (default: {DEFAULT_SUBTYPE})",
    )
    render_command.add_argument(
        "--snapshot",
        action="store_true",
        help="also write the rendered project document next to the audio",
    )

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
    if arguments.with_source:
        project = project.with_source(Source(id="source-1", name="Exact symmetric SAM"))

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


def _command_render(arguments: argparse.Namespace) -> int:
    path: Path = arguments.path
    try:
        project = load_project(path)
    except ProjectValidationError as error:
        _report_issues(error, path)
        return 1
    except OSError as error:
        print(f"Could not read {path}: {error}", file=sys.stderr)
        return 1

    if arguments.duration <= 0.0:
        print(f"Render duration must be positive, got {arguments.duration}", file=sys.stderr)
        return 1

    try:
        manifest_path = export_wav(
            project,
            arguments.duration,
            arguments.output,
            block_size=arguments.block_size,
            subtype=arguments.subtype,
            write_project_snapshot=arguments.snapshot,
        )
    except ProjectValidationError as error:
        _report_issues(error, path)
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not render {path}: {error}", file=sys.stderr)
        return 1

    if not any(source.enabled for source in project.sources):
        print(f"Warning: {path} has no enabled sources; the render is silent", file=sys.stderr)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for warning in manifest.get("warnings", ()):
        print(f"Warning: {warning['path']}: {warning['message']}", file=sys.stderr)
    print(
        f"Rendered {arguments.output} ({manifest['frames']} frames, "
        f"{manifest['sample_rate_hz']} Hz, renderer {manifest['renderer']}, "
        f"peak {manifest['levels']['peak']:.4f})"
    )
    print(f"Wrote manifest {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line shell and return a process exit status."""

    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "new":
        return _command_new(arguments)
    if arguments.command == "validate":
        return _command_validate(arguments)
    if arguments.command == "render":
        return _command_render(arguments)
    parser.error(f"unknown command {arguments.command!r}")  # pragma: no cover - argparse guards this
    return 2


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

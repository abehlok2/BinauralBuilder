"""Project creation, validation, rendering, and optional GUI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import Project, ProjectValidationError, load_project, save_project
from .export import export_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam-workbench", description="SAM workbench project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    new = subparsers.add_parser("new", help="write a project with safe defaults")
    new.add_argument("path", type=Path)
    new.add_argument("--name", default="Untitled SAM Project")
    validate = subparsers.add_parser("validate", help="validate an existing project")
    validate.add_argument("path", type=Path)
    render = subparsers.add_parser("render", help="render exact symmetric SAM to WAV")
    render.add_argument("project", type=Path)
    render.add_argument("output", type=Path)
    render.add_argument("--duration-s", type=float, required=True)
    gui = subparsers.add_parser("gui", help="open the phase-two PyQt workbench")
    gui.add_argument("project", type=Path, nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "new":
            save_project(Project(name=args.name), args.path)
            print(f"Created {args.path}")
        elif args.command == "validate":
            load_project(args.path)
            print(f"Valid SAM project: {args.path}")
        elif args.command == "render":
            manifest = export_wav(load_project(args.project), args.duration_s, args.output)
            print(f"Rendered {args.output} (manifest: {manifest})")
        else:
            # Keep Qt optional for every headless command and package import.
            from .gui import run_gui

            return run_gui(load_project(args.project) if args.project else None)
    except (OSError, ProjectValidationError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

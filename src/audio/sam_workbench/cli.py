"""Headless project creation and validation commands for phase zero."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import Project, ProjectValidationError, load_project, save_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam-workbench", description="SAM workbench project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    new = subparsers.add_parser("new", help="write a project with safe defaults")
    new.add_argument("path", type=Path)
    new.add_argument("--name", default="Untitled SAM Project")
    validate = subparsers.add_parser("validate", help="validate an existing project")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "new":
            save_project(Project(name=args.name), args.path)
            print(f"Created {args.path}")
        else:
            load_project(args.path)
            print(f"Valid SAM project: {args.path}")
    except (OSError, ProjectValidationError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

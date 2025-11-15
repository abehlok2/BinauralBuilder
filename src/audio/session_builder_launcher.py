"""Command line launcher for the Session Builder UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

from session_model import (
    Session,
    SessionStep,
    build_binaural_preset_catalog,
    build_noise_preset_catalog,
)

if TYPE_CHECKING:  # pragma: no cover - type checking hook
    from PyQt5.QtWidgets import QApplication


def _ensure_app(argv: Optional[Sequence[str]] = None) -> "QApplication":
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        qt_args = list(argv) if argv is not None else [sys.argv[0]]
        app = QApplication(qt_args)
    return app


def _as_paths(values: Iterable[Path | str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        paths.append(path)
    return paths


def _load_session_from_json(path: Path) -> Session:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Session file must contain a JSON object")
    raw_steps = data.get("steps", [])
    steps = []
    if isinstance(raw_steps, Iterable):
        for entry in raw_steps:
            if not isinstance(entry, Mapping):
                raise ValueError("Session steps must be objects")
            step_kwargs = dict(entry)
            steps.append(SessionStep(**step_kwargs))
    payload = dict(data)
    payload["steps"] = steps
    return Session(**payload)


def launch_session_builder(
    session_path: Optional[Path | str] = None,
    *,
    binaural_preset_dirs: Optional[Iterable[Path | str]] = None,
    noise_preset_dirs: Optional[Iterable[Path | str]] = None,
    theme: Optional[str] = None,
    builtin_duration: float = 300.0,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Launch the Session Builder window and enter the Qt event loop."""

    try:
        from src.ui import themes
        from src.ui.session_builder_window import SessionBuilderWindow
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("PyQt5 is required to launch the session builder UI") from exc

    if session_path is not None:
        session_path = Path(session_path).expanduser()
        if not session_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_path}")
        session = _load_session_from_json(session_path)
    else:
        session = Session()

    binaural_dirs = _as_paths(binaural_preset_dirs or [])
    noise_dirs = _as_paths(noise_preset_dirs or [])

    binaural_catalog = build_binaural_preset_catalog(
        duration=builtin_duration, preset_dirs=binaural_dirs
    )
    noise_catalog = build_noise_preset_catalog(preset_dirs=noise_dirs)

    app = _ensure_app(argv)

    if theme and theme in themes.THEMES:
        themes.apply_theme(app, theme)

    window = SessionBuilderWindow(
        session=session,
        binaural_catalog=binaural_catalog,
        noise_catalog=noise_catalog,
        theme_name=theme,
    )
    window.show()
    return app.exec_()


def _build_parser() -> argparse.ArgumentParser:
    theme_choices = _available_themes()
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Session Builder UI without starting the main binaural beat creator."
        )
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Optional JSON session file to load on startup.",
    )
    parser.add_argument(
        "--binaural-preset-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Directory containing .voice presets to include (can be repeated).",
    )
    parser.add_argument(
        "--noise-preset-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Directory containing .noise presets to include (can be repeated).",
    )
    parser.add_argument(
        "--builtin-duration",
        type=float,
        default=300.0,
        help="Duration in seconds used when generating builtin binaural presets.",
    )
    parser.add_argument(
        "--theme",
        choices=theme_choices or None,
        help="Optional theme name to apply to the UI." + (
            " Available themes: " + ", ".join(theme_choices)
            if theme_choices
            else ""
        ),
    )
    return parser


def _available_themes() -> list[str]:
    try:
        from src.ui import themes
    except ImportError:
        return []
    return sorted(themes.THEMES.keys())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return launch_session_builder(
            session_path=args.session,
            binaural_preset_dirs=args.binaural_preset_dir,
            noise_preset_dirs=args.noise_preset_dir,
            theme=args.theme,
            builtin_duration=args.builtin_duration,
            argv=[parser.prog],
        )
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    sys.exit(main())

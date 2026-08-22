"""Portable persistence for the parameter state edited by the SAM workbench.

This is deliberately separate from the full :class:`Project` document.  A
workbench state is a small preset-like snapshot that can be applied to a voice
without also replacing its routing in the surrounding BinauralBuilder track.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

__all__ = ["STATE_FORMAT", "load_parameter_state", "save_parameter_state"]

STATE_FORMAT = "binauralbuilder.sam-workbench-parameters"
STATE_VERSION = 1


def save_parameter_state(
    params: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    is_transition: bool = False,
) -> Path:
    """Atomically save a JSON parameter snapshot and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "is_transition": bool(is_transition),
        "params": dict(params),
    }
    # Serialise before creating a temporary file so unsupported values cannot
    # disturb an existing state file.
    encoded = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def load_parameter_state(
    path: str | os.PathLike[str],
    *,
    is_transition: bool | None = None,
) -> dict[str, Any]:
    """Load and validate a workbench parameter snapshot.

    Static and transition voices use different field sets, so the optional
    ``is_transition`` guard prevents accidentally applying the wrong kind.
    """

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg} (line {error.lineno})") from error
    if not isinstance(document, dict):
        raise ValueError("parameter state must be a JSON object")
    if document.get("format") != STATE_FORMAT:
        raise ValueError("not a SAM workbench parameter-state file")
    if document.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported parameter-state version: {document.get('version')!r}")
    stored_transition = document.get("is_transition")
    if not isinstance(stored_transition, bool):
        raise ValueError("is_transition must be true or false")
    if is_transition is not None and stored_transition != bool(is_transition):
        expected = "transition" if is_transition else "static"
        actual = "transition" if stored_transition else "static"
        raise ValueError(f"cannot load {actual} parameters into a {expected} voice")
    params = document.get("params")
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    return dict(params)

"""The canonical SAM package must import without PyQt5, ``slab``, or Rust.

Phase 0 exit criterion: ``src.audio.sam_workbench`` imports and works in a
clean Python environment.  Two independent checks are used - a static scan of
the package sources for forbidden imports, and a subprocess import with those
modules blocked at the import system level.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPOSITORY_ROOT / "src" / "audio" / "sam_workbench"

#: Top-level modules the canonical core must never depend on.
FORBIDDEN_ROOTS = (
    "PyQt5",
    "PySide2",
    "PySide6",
    "slab",
    "realtime_backend",
    "sounddevice",
)

#: Repository modules that would invert the dependency direction.
FORBIDDEN_REPOSITORY_MODULES = (
    "main",
    "src.realtime_backend",
    "src.audio.rust_stream_player",
    "src.audio.session_stream",
    "src.synth_functions.sound_creator",
    "src.synth_functions.audio_engine",
    "src.synth_functions.spatial_angle_modulation",
    "src.ui",
)


def _package_sources() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _imported_module_names(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_package_sources_exist():
    assert _package_sources(), f"no modules found in {PACKAGE_DIR}"


@pytest.mark.parametrize("source", _package_sources(), ids=lambda path: path.name)
def test_no_forbidden_imports_in_sources(source: Path):
    imported = _imported_module_names(source)
    for name in imported:
        root = name.split(".", 1)[0]
        assert root not in FORBIDDEN_ROOTS, f"{source.name} imports {name}"
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_REPOSITORY_MODULES
        ), f"{source.name} imports {name}"


CLEAN_IMPORT_SCRIPT = """
import sys

BLOCKED = {blocked!r}


class _Blocker:
    def find_module(self, fullname, path=None):
        self.find_spec(fullname, path)
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in BLOCKED:
            raise ImportError(f"{{fullname}} is blocked in this environment")
        return None


sys.meta_path.insert(0, _Blocker())

import src.audio.sam_workbench as workbench
from src.audio.sam_workbench import cli, conventions, migrations, model, validation, version

for blocked in BLOCKED:
    assert blocked not in sys.modules, f"{{blocked}} was imported"

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "clean.sam.json"
    assert cli.main(["new", str(path), "--name", "Clean"]) == 0
    assert cli.main(["validate", str(path)]) == 0
    assert model.load_project(path).name == "Clean"

print("clean-import-ok")
"""


def test_package_imports_and_runs_without_gui_or_rust(tmp_path):
    script = tmp_path / "clean_import.py"
    script.write_text(CLEAN_IMPORT_SCRIPT.format(blocked=list(FORBIDDEN_ROOTS)), encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "clean-import-ok" in completed.stdout

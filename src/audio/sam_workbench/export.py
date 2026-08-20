"""Deterministic phase-one audio export and reconstruction manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import render_source
from .model import Project, validate_project


def render_project(project: Project, duration_s: float) -> np.ndarray:
    issues = validate_project(project)
    if issues:
        raise ValueError("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    frames = round(duration_s * project.audio.sample_rate_hz)
    mix = np.zeros((2, frames), dtype=np.float32)
    for source in project.sources:
        if source.spatializer.type != "abstract":
            raise NotImplementedError("phase one exports abstract sources only")
        mix += render_source(source, project.audio.sample_rate_hz, frames, project.audio.offline_block_frames)
    mix *= np.float32(10.0 ** (project.output.master_gain_db / 20.0))
    return mix


def export_wav(project: Project, duration_s: float, path: str | Path) -> Path:
    destination = Path(path)
    audio = render_project(project, duration_s)
    sf.write(destination, audio.T, project.audio.sample_rate_hz, subtype=project.output.subtype)
    project_json = json.dumps(project.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = {
        "schema_version": project.schema_version,
        "project_id": project.id,
        "project_sha256": hashlib.sha256(project_json.encode()).hexdigest(),
        "sample_rate_hz": project.audio.sample_rate_hz,
        "frames": audio.shape[1],
        "channels": 2,
        "renderer": "exact_symmetric_sam",
        "seed": project.seed,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path

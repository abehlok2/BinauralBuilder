"""Offline export: audio file plus a reconstruction manifest.

A render is only reproducible if the file is accompanied by everything needed
to recreate it. Every export therefore writes a manifest next to the audio
containing the schema and package versions, a hash of the exact project
document that was rendered, the sample rate and bit depth, the renderer and its
policies, the seeds, the master processing chain, and the measured levels.

Fields that cannot be deterministic - the wall-clock timestamp and the elapsed
render time - are grouped under ``nondeterministic`` so a byte-comparison of
two manifests can ignore them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.audio.sam_workbench.conventions import CHANNEL_ORDER, seconds_to_samples
from src.audio.sam_workbench.render.scene import SceneReport, render_project
from src.audio.sam_workbench.version import PACKAGE_VERSION, SCHEMA_VERSION

__all__ = [
    "DEFAULT_SUBTYPE",
    "build_manifest",
    "export_wav",
    "project_sha256",
    "write_manifest",
]

#: 32-bit float keeps the exported samples bit-identical to the render.
DEFAULT_SUBTYPE = "FLOAT"

_SUBTYPE_BIT_DEPTH = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


def project_sha256(project) -> str:
    """Hash the canonical serialization of a project document."""

    payload = json.dumps(project.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    project,
    report: SceneReport,
    *,
    audio_path: Path,
    subtype: str,
    elapsed_s: float,
) -> dict[str, Any]:
    """Assemble the render manifest for one export."""

    limiter = None if report.limiter is None else asdict(report.limiter)
    return {
        "manifest_version": "1.0",
        "application": "binauralbuilder.sam_workbench",
        "application_version": PACKAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "project_id": project.id,
        "project_name": project.name,
        "project_sha256": project_sha256(project),
        "audio_file": audio_path.name,
        "renderer": report.renderer,
        "renderer_options": {
            "ear_polarity": "left_plus_right_minus",
            "phase_accumulation": "absolute_sample_index",
            "block_size": report.block_size,
            "block_size_invariant": True,
        },
        "sample_rate_hz": report.sample_rate_hz,
        "bit_depth": _SUBTYPE_BIT_DEPTH.get(subtype, 32),
        "subtype": subtype,
        "channels": len(CHANNEL_ORDER),
        "channel_order": list(CHANNEL_ORDER),
        "frames": report.frames,
        "duration_s": report.frames / float(report.sample_rate_hz),
        "sources": [
            {
                "id": entry.source_id,
                "start_sample": entry.start_sample,
                "frames": entry.frames,
                "peak": entry.peak,
            }
            for entry in report.sources
        ],
        "seeds": {"project_random_seed": project.audio.random_seed},
        "master_processing": {
            "master_gain_db": report.master_gain_db,
            "shared_gain_both_ears": True,
            "per_channel_normalization": False,
            "limiter": limiter,
            "headphone_compensation_enabled": project.output.headphone_compensation_enabled,
            "headphone_compensation_asset": project.output.headphone_compensation_asset,
        },
        "levels": {
            "peak": report.peak,
            "true_peak_dbfs": report.true_peak_dbfs,
            "clipped_samples": 0 if limiter is None else limiter["clipped_samples"],
        },
        "warnings": [
            {"path": issue.path, "message": issue.message, "severity": issue.severity}
            for issue in report.warnings
        ],
        "determinism": {
            "deterministic_for_same_inputs": True,
            "block_size_independent": True,
            "tolerance": "bit-exact for the same platform and dependency lock",
        },
        "nondeterministic": {
            "rendered_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_s": elapsed_s,
        },
    }


def write_manifest(manifest: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a manifest atomically next to its audio file."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.partial")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def export_wav(
    project,
    duration_s: float,
    path: str | os.PathLike[str],
    *,
    block_size: int | None = None,
    subtype: str = DEFAULT_SUBTYPE,
    manifest_path: str | os.PathLike[str] | None = None,
    write_project_snapshot: bool = False,
) -> Path:
    """Render ``project`` for ``duration_s`` seconds and write audio plus manifest.

    Returns the manifest path; the audio is written to ``path``. The audio is
    frame-major ``(frames, 2)`` because that is what audio files and the rest of
    BinauralBuilder use - the single channel-major to frame-major conversion
    happens here, at the boundary.
    """

    audio_path = Path(path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frames = seconds_to_samples(duration_s, project.audio.sample_rate_hz)
    audio, report = render_project(project, frames=frames, block_size=block_size)
    elapsed = time.perf_counter() - started

    sf.write(audio_path, np.ascontiguousarray(audio.T), project.audio.sample_rate_hz, subtype=subtype)

    manifest = build_manifest(
        project, report, audio_path=audio_path, subtype=subtype, elapsed_s=elapsed
    )
    destination = Path(manifest_path) if manifest_path else audio_path.with_suffix(".manifest.json")
    written = write_manifest(manifest, destination)

    if write_project_snapshot:
        snapshot = audio_path.with_suffix(".project.json")
        snapshot.write_text(project.to_json(), encoding="utf-8")
        manifest["project_snapshot"] = snapshot.name
        write_manifest(manifest, written)
    return written

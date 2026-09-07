"""Worker functions for path coverage. Never access Qt widgets here."""

import numpy as np
from src.audio.sam_workbench.hrtf.coverage import assess_path_coverage


def load_positions(asset, options, expected_hash):
    from src.audio.sam_workbench.hrtf.sofa_io import (
        load_sofa,
        resolve_sofa_path,
        hash_asset,
    )

    path = resolve_sofa_path(asset, options.get("projectDirectory"))
    if expected_hash and hash_asset(path) != expected_hash:
        raise ValueError("Selected SOFA hash does not match the saved asset")
    return load_sofa(
        path, delay_policy=options.get("delayPolicy", "bake_delay_into_ir")
    ).positions_m


def coverage_preview(measurements, model, duration, rate, interval, options, curve):
    # Bounded work even for hour-long authoring previews. Report any sampling
    # reduction; never claim every renderer update was checked when it wasn't.
    frames = max(1, int(round(duration * rate)))
    stride = max(interval, interval * int(np.ceil(frames / interval / 100000)))
    indices = np.arange(0, frames + 1, stride)
    points = np.concatenate(
        [
            model.positions(part / rate)
            for part in np.array_split(
                indices, max(1, int(np.ceil(len(indices) / 4096)))
            )
        ]
    )
    report = assess_path_coverage(
        measurements,
        points,
        sample_rate_hz=rate,
        control_interval_samples=stride,
        crossfade_ms=float(options.get("crossfadeMs", 10)),
        interpolation=options.get("interpolation", "nearest"),
    )
    from scipy.spatial import cKDTree

    radii = np.linalg.norm(measurements, axis=1)
    directions = measurements[radii > 1e-9] / radii[radii > 1e-9, None]
    norms = np.linalg.norm(curve, axis=1)
    queries = curve / np.maximum(norms[:, None], 1e-12)
    tree = cKDTree(directions)
    distance, _ = tree.query(queries)
    angle = np.degrees(2 * np.arcsin(np.clip(distance / 2, 0, 1)))
    neighbors, _ = tree.query(directions, k=min(2, len(directions)))
    spacing = (
        np.median(np.degrees(2 * np.arcsin(np.clip(neighbors[:, -1] / 2, 0, 1))))
        if len(directions) > 1
        else 0
    )
    elevation = np.degrees(np.arcsin(np.clip(queries[:, 2], -1, 1)))
    measured_elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1, 1)))
    bad = (
        (norms < 1e-9)
        | (angle > max(2.5 * spacing, 1))
        | (elevation < measured_elevation.min() - 0.5)
        | (elevation > measured_elevation.max() + 0.5)
    )
    messages = [issue.message for issue in report.issues]
    if not messages:
        messages.append("No coverage warnings for the sampled interval.")
    if stride != interval:
        messages.append(
            f"Long interval sampled every {stride} samples; short excursions may be missed. Narrow the preview interval for control-grid checks."
        )
    else:
        messages.append(
            f"Checked {duration:g} s on a {interval}-sample candidate control grid; adaptive filter updates may be farther apart."
        )
    return messages, bad

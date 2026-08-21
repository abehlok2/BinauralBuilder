"""Faithful port of the legacy SAM2 path/shape evaluation.

The BinauralBuilder SAM2 voices in use today describe motion with a path type
(``open``, ``closed``, ``discontinuous``, ``custom``), a shape (``sinusoidal``,
``triangle``, ``ramp``, ``square``), and - for custom paths - a GUI profile of
scene-space points with Chaikin smoothing or Catmull-Rom splines.

This module is the canonical home of that evaluation. It is a deliberate,
behaviour-preserving port: existing presets must keep rendering exactly as they
did, so nothing here has been "improved". Both public synthesis trees delegate
to these functions instead of keeping their own copies.

Legacy profile coordinates are GUI scene units, not metres. They stay in their
own normalized shape space here; converting them into the canonical metric
listener frame is the job of the trajectory transforms that arrive with the
geometric renderer.
"""

from __future__ import annotations

import json
import math

import numpy as np

__all__ = [
    "SAM2_DEFAULT_SHAPES_BY_TYPE",
    "catmull_rom_eval",
    "chaikin_smooth_points",
    "custom_path_shape_and_scale",
    "progress_closed",
    "progress_discontinuous",
    "progress_open",
    "resolve_custom_path_xy",
    "resolve_sam2_shape",
    "shape_from_progress",
]


def progress_open(phase: np.ndarray) -> np.ndarray:
    """Back-and-forth path traversal (0 → 1 → 0)."""
    wrapped = (phase / (2.0 * math.pi)) % 1.0
    return 1.0 - (2.0 * np.abs(wrapped - 0.5))


def progress_closed(phase: np.ndarray, direction: str = 'cw') -> np.ndarray:
    """Looping path traversal (clockwise/counterclockwise)."""
    wrapped = (phase / (2.0 * math.pi)) % 1.0
    return wrapped if direction != 'ccw' else (1.0 - wrapped)


def progress_discontinuous(phase: np.ndarray, steps: int = 8, direction: str = 'cw') -> np.ndarray:
    """Stepped looping traversal for staccato position jumps."""
    smooth = progress_closed(phase, direction=direction)
    n_steps = max(2, int(steps))
    return np.floor(smooth * n_steps) / float(n_steps - 1)


def shape_from_progress(path_shape: str, progress: np.ndarray) -> np.ndarray:
    """Map normalized path progress [0,1] to SAM modulation shape [-1,1]."""
    shape = str(path_shape or '').lower()
    if shape in ('triangle', 'tri'):
        return (4.0 * np.abs(progress - 0.5)) - 1.0
    if shape in ('saw', 'sawtooth', 'ramp', 'linear'):
        return (2.0 * progress) - 1.0
    if shape in ('square', 'step'):
        return np.where(np.sin(2.0 * math.pi * progress) >= 0.0, 1.0, -1.0)
    # Default smooth sinusoid around center
    return np.sin(2.0 * math.pi * progress)


SAM2_DEFAULT_SHAPES_BY_TYPE = {
    'open': 'sinusoidal',
    'closed': 'ramp',
    'discontinuous': 'square',
}


def catmull_rom_eval(p0, p1, p2, p3, t: np.ndarray):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2.0 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2.0 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return x, y


def chaikin_smooth_points(points, is_closed: bool, passes: int, ratio: float):
    if passes <= 0 or len(points) < 3:
        return points

    ratio = float(np.clip(ratio, 1e-3, 0.499))
    smoothed = list(points)
    for _ in range(passes):
        if len(smoothed) < 3:
            break

        if is_closed:
            refined = []
            count = len(smoothed)
            for i in range(count):
                p0 = smoothed[i]
                p1 = smoothed[(i + 1) % count]
                q = ((1.0 - ratio) * p0[0] + ratio * p1[0], (1.0 - ratio) * p0[1] + ratio * p1[1])
                r = (ratio * p0[0] + (1.0 - ratio) * p1[0], ratio * p0[1] + (1.0 - ratio) * p1[1])
                refined.extend((q, r))
            smoothed = refined
        else:
            refined = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0 = smoothed[i]
                p1 = smoothed[i + 1]
                q = ((1.0 - ratio) * p0[0] + ratio * p1[0], (1.0 - ratio) * p0[1] + ratio * p1[1])
                r = (ratio * p0[0] + (1.0 - ratio) * p1[0], ratio * p0[1] + (1.0 - ratio) * p1[1])
                refined.extend((q, r))
            refined.append(smoothed[-1])
            smoothed = refined

    return smoothed


def resolve_custom_path_xy(phase: np.ndarray, custom_profile):
    if isinstance(custom_profile, str):
        try:
            custom_profile = json.loads(custom_profile)
        except Exception:
            custom_profile = {}

    points = custom_profile.get('points') if isinstance(custom_profile, dict) else None
    if not isinstance(points, list) or len(points) < 2:
        return None, None

    clean_points = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                clean_points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                pass

    if len(clean_points) < 2:
        return None, None

    is_closed = bool(custom_profile.get('closedLoop', False)) if isinstance(custom_profile, dict) else False
    kind = str(custom_profile.get('kind', '')).lower() if isinstance(custom_profile, dict) else ''
    subnodes_per_segment = int(custom_profile.get('subNodesPerSegment', 24)) if isinstance(custom_profile, dict) else 24
    subnodes_per_segment = max(4, min(subnodes_per_segment, 256))
    smoothing_passes = int(custom_profile.get('smoothingPasses', 1)) if isinstance(custom_profile, dict) else 1
    smoothing_passes = max(0, min(smoothing_passes, 6))
    smoothing_ratio = float(custom_profile.get('smoothingRatio', 0.25)) if isinstance(custom_profile, dict) else 0.25

    if is_closed and clean_points[0] != clean_points[-1]:
        clean_points.append(clean_points[0])

    if kind != 'spline':
        clean_points = chaikin_smooth_points(clean_points, is_closed=is_closed, passes=smoothing_passes, ratio=smoothing_ratio)

    sample_x = []
    sample_y = []
    sample_d = [0.0]

    if kind == 'spline' and len(clean_points) >= 3:
        segment_count = len(clean_points) if is_closed else len(clean_points) - 1
        for i in range(segment_count):
            p1 = clean_points[i]
            p2 = clean_points[(i + 1) % len(clean_points)] if is_closed else clean_points[i + 1]
            p0 = clean_points[i - 1] if i > 0 else (clean_points[-2] if is_closed else clean_points[0])
            p3 = clean_points[(i + 2) % len(clean_points)] if (is_closed or i + 2 < len(clean_points)) else clean_points[-1]
            t = np.linspace(0.0, 1.0, subnodes_per_segment, endpoint=False)
            x, y = catmull_rom_eval(p0, p1, p2, p3, t)
            sample_x.extend(x.tolist())
            sample_y.extend(y.tolist())
        sample_x.append(clean_points[-1][0])
        sample_y.append(clean_points[-1][1])
    else:
        for i in range(len(clean_points) - 1):
            p0 = clean_points[i]
            p1 = clean_points[i + 1]
            t = np.linspace(0.0, 1.0, subnodes_per_segment, endpoint=False)
            for ti in t:
                sample_x.append((1.0 - ti) * p0[0] + ti * p1[0])
                sample_y.append((1.0 - ti) * p0[1] + ti * p1[1])
        sample_x.append(clean_points[-1][0])
        sample_y.append(clean_points[-1][1])

    for i in range(1, len(sample_x)):
        sample_d.append(sample_d[-1] + math.hypot(sample_x[i] - sample_x[i - 1], sample_y[i] - sample_y[i - 1]))

    total = sample_d[-1]
    if total <= 1e-6:
        return None, None

    pos = ((phase / (2.0 * math.pi)) % 1.0) * total
    x_interp = np.interp(pos, np.array(sample_d, dtype=np.float64), np.array(sample_x, dtype=np.float64))
    y_interp = np.interp(pos, np.array(sample_d, dtype=np.float64), np.array(sample_y, dtype=np.float64))
    return x_interp, y_interp


def custom_path_shape_and_scale(phase: np.ndarray, custom_profile):
    x_interp, y_interp = resolve_custom_path_xy(phase, custom_profile)
    if x_interp is None or y_interp is None:
        base = shape_from_progress('sinusoidal', progress_open(phase))
        return base, np.ones_like(base)

    angle_deg = np.degrees(np.arctan2(x_interp, y_interp))
    norm_angle = np.clip(angle_deg / 180.0, -1.0, 1.0)

    radial_dist = np.hypot(x_interp, y_interp)
    d_min = float(np.min(radial_dist))
    d_max = float(np.max(radial_dist))
    if d_max - d_min <= 1e-6:
        return norm_angle, np.ones_like(norm_angle)

    dist_norm = (radial_dist - d_min) / (d_max - d_min)
    # closer (smaller distance) => stronger spatial scale
    dynamic_scale = 1.25 - 0.5 * dist_norm
    return norm_angle, np.clip(dynamic_scale, 0.75, 1.25)


def resolve_sam2_shape(path_type: str, phase: np.ndarray, custom_profile=None, path_shape=None, discontinuous_steps=8, rotation_direction='cw'):
    if path_type.lower() == 'custom':
        return custom_path_shape_and_scale(phase, custom_profile)

    path_type_norm = path_type.lower()
    if path_type_norm == 'closed':
        progress = progress_closed(phase, direction=rotation_direction)
    elif path_type_norm == 'discontinuous':
        progress = progress_discontinuous(phase, steps=discontinuous_steps, direction=rotation_direction)
    else:
        progress = progress_open(phase)

    selected_shape = path_shape or SAM2_DEFAULT_SHAPES_BY_TYPE.get(path_type_norm, 'sinusoidal')
    shape = shape_from_progress(selected_shape, progress)
    return shape, np.ones_like(shape)

"""Trajectory geometry, traversal, and legacy path translation.

Phase 1 contains only the behaviour-preserving port of the legacy SAM2 path
evaluation, which the compatibility adapter needs in order to delegate existing
voices to the canonical core. The separable geometry/traversal/transform model
arrives with the geometric renderer.
"""

from __future__ import annotations

from src.audio.sam_workbench.trajectory.legacy_paths import (
    SAM2_DEFAULT_SHAPES_BY_TYPE,
    custom_path_shape_and_scale,
    progress_closed,
    progress_discontinuous,
    progress_open,
    resolve_custom_path_xy,
    resolve_sam2_shape,
    shape_from_progress,
)

__all__ = [
    "SAM2_DEFAULT_SHAPES_BY_TYPE",
    "custom_path_shape_and_scale",
    "progress_closed",
    "progress_discontinuous",
    "progress_open",
    "resolve_custom_path_xy",
    "resolve_sam2_shape",
    "shape_from_progress",
]
from .geometry import (
    Arc,
    Bezier,
    Circle,
    Ellipse,
    Helix,
    Line,
    Lissajous,
    Mathematical,
    PointCloud,
    Polygon,
    Polyline,
    Spiral,
    Spline,
    TransformedGeometry,
    arclength_table,
    evaluate_arclength,
    sample_polyline_arclength,
)
from .transforms import ListenerTransform, Transform, rotation_matrix_ypr
from .traversal import (
    CanonicalTrajectory,
    KeyframedTraversal,
    StochasticTraversal,
    Traversal,
    apply_easing,
    segment_positions,
)
from .serialization import trajectory_from_dict

__all__ = [
    "Arc",
    "Bezier",
    "Circle",
    "Ellipse",
    "Helix",
    "Line",
    "Lissajous",
    "Mathematical",
    "PointCloud",
    "Polygon",
    "Polyline",
    "Spiral",
    "Spline",
    "TransformedGeometry",
    "Transform",
    "ListenerTransform",
    "CanonicalTrajectory",
    "Traversal",
    "KeyframedTraversal",
    "StochasticTraversal",
    "apply_easing",
    "segment_positions",
    "arclength_table",
    "evaluate_arclength",
    "sample_polyline_arclength",
    "rotation_matrix_ypr",
    "trajectory_from_dict",
]

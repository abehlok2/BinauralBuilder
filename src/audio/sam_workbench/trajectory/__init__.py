"""Trajectories: geometry, traversal, transforms, and the legacy bridges.

Three separable ideas, as the specification requires:

* **geometry** - where the path is;
* **traversal** - how the source moves along it;
* **transform** - how the whole path is placed, turned, and scaled.

`legacy_paths` is the behaviour-preserving port of the SAM2 path evaluation
that existing voices still render through; `legacy_profile` translates the GUI
path formats into canonical metres without rewriting what is stored on disk.
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
from .legacy_profile import (
    LEGACY_PROFILE_VERSION,
    PROFILE_VERSION,
    legacy_azimuth_to_canonical_deg,
    profile_points,
    profile_round_trip,
    profile_schema_version,
    profile_to_geometry,
    profile_transform,
    segment_positions,
    segment_to_trajectory,
    segments_duration_s,
    segments_positions,
    upgrade_profile,
)
from .spec import TrajectoryResult, TrajectorySpec, trajectory_from_dict
from .transforms import (
    AXIS_CONVENTIONS,
    LegacyPathTransform,
    ListenerFrame,
    TransformSpec,
    rotation_matrix,
    transform_from_dict,
)
from .traversal import (
    DEFAULT_JUMP_CROSSFADE_MS,
    TRAVERSAL_MODES,
    TraversalResult,
    TraversalSpec,
    constant_speed_rate_hz,
    traversal_from_dict,
)

__all__ = [
    "AXIS_CONVENTIONS",
    "DEFAULT_JUMP_CROSSFADE_MS",
    "LEGACY_PROFILE_VERSION",
    "PROFILE_VERSION",
    "SAM2_DEFAULT_SHAPES_BY_TYPE",
    "TRAVERSAL_MODES",
    "ArcGeometry",
    "CircleGeometry",
    "EllipseGeometry",
    "GeometrySpec",
    "HelixGeometry",
    "LegacyPathTransform",
    "LineGeometry",
    "LissajousGeometry",
    "ListenerFrame",
    "PointCloudGeometry",
    "PolylineGeometry",
    "SpiralGeometry",
    "SplineGeometry",
    "TrajectoryResult",
    "TrajectorySpec",
    "TransformSpec",
    "TraversalResult",
    "TraversalSpec",
    "constant_speed_rate_hz",
    "custom_path_shape_and_scale",
    "geometry_from_dict",
    "legacy_azimuth_to_canonical_deg",
    "profile_points",
    "profile_round_trip",
    "profile_schema_version",
    "profile_to_geometry",
    "profile_transform",
    "progress_closed",
    "progress_discontinuous",
    "progress_open",
    "resolve_custom_path_xy",
    "resolve_sam2_shape",
    "rotation_matrix",
    "segment_positions",
    "segment_to_trajectory",
    "segments_duration_s",
    "segments_positions",
    "shape_from_progress",
    "trajectory_from_dict",
    "transform_from_dict",
    "traversal_from_dict",
    "upgrade_profile",
]
from .geometry import (
    Arc,
    Bezier,
    Circle,
    Ellipse,
    Helix,
    Line,
    Lissajous,
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
    "Arc", "Bezier", "Circle", "Ellipse", "Helix", "Line", "Lissajous",
    "PointCloud", "Polygon", "Polyline", "Spiral", "Spline", "TransformedGeometry",
    "Transform", "ListenerTransform", "CanonicalTrajectory", "Traversal",
    "KeyframedTraversal", "StochasticTraversal",
    "apply_easing", "segment_positions", "arclength_table",
    "evaluate_arclength", "sample_polyline_arclength", "rotation_matrix_ypr",
    "trajectory_from_dict",
]

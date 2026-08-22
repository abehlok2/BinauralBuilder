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
from .keyframes import (
    Keyframe,
    KeyframedPath,
    keyframes_from_arrays,
    keyframes_from_csv,
    keyframes_from_json,
    load_keyframes,
)
from .path_model import (
    AXIS_DEFINITIONS,
    COORDINATE_SYSTEMS,
    HANDEDNESS,
    INTERPOLATION_KINDS,
    SPEED_LAWS,
    UNITS,
    PathModel,
    SourceOrientation,
)
from .primitives import (
    PRIMITIVE_TYPES,
    DomeTraversal,
    ElevationSweep,
    FigureEight3D,
    HorizontalOrbit,
    OverheadSweep,
    Pendulum,
    RandomWalkVolume,
    RisingArc,
    SphericalOrbit,
    TiltedOrbit,
    Torus,
    VerticalOrbit,
)
from .spherical import (
    SphericalPosition,
    cartesian_array_to_spherical,
    cartesian_to_spherical,
    spherical_array_to_cartesian,
    spherical_to_cartesian,
)
from .serialization import (
    GEOMETRY_TYPES,
    geometry_to_dict,
    path_model_from_dict,
    trajectory_from_dict,
)

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
    "path_model_from_dict",
    "geometry_to_dict",
    "GEOMETRY_TYPES",
    # --- the canonical 3-D path model -------------------------------------
    "PathModel",
    "SourceOrientation",
    "COORDINATE_SYSTEMS",
    "INTERPOLATION_KINDS",
    "SPEED_LAWS",
    "AXIS_DEFINITIONS",
    "HANDEDNESS",
    "UNITS",
    # --- spherical entry ---------------------------------------------------
    "SphericalPosition",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "spherical_array_to_cartesian",
    "cartesian_array_to_spherical",
    # --- keyframes ---------------------------------------------------------
    "Keyframe",
    "KeyframedPath",
    "keyframes_from_arrays",
    "keyframes_from_csv",
    "keyframes_from_json",
    "load_keyframes",
    # --- 3-D primitives ----------------------------------------------------
    "PRIMITIVE_TYPES",
    "HorizontalOrbit",
    "VerticalOrbit",
    "TiltedOrbit",
    "SphericalOrbit",
    "RisingArc",
    "OverheadSweep",
    "ElevationSweep",
    "DomeTraversal",
    "FigureEight3D",
    "Pendulum",
    "Torus",
    "RandomWalkVolume",
]

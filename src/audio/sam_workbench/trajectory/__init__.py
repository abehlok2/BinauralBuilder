"""Trajectory geometry, traversal, and legacy path translation.

A source position is a point in metres in the listener frame,
``p(t) = [x(t), y(t), z(t)]`` with ``+x`` forward, ``+y`` left and ``+z`` up.
Azimuth, elevation and distance are *derived* from it rather than stored beside
it, so there is exactly one description of where a source is and no way for two
of them to disagree. Spherical values are an input convenience, converted at the
boundary by :mod:`.spherical`.

The model is separable in two directions, and both matter:

* **geometry against traversal** - where the source can be, against how it moves
  along that shape over time. A circle is one circle whether it is walked at
  constant speed, eased, reversed or driven from an envelope, so those are time
  laws in :mod:`.traversal` rather than new shapes. :mod:`.path_model` composes
  the two and carries the frame, units and playback semantics a saved path needs
  in order to be read correctly somewhere else.
* **shape against parameters** - :mod:`.geometry` holds the point-based curves
  the editor lets you drag, :mod:`.primitives` the three-dimensional shapes
  defined by their own numbers, and :mod:`.keyframes` timestamped positions,
  including imported and recorded motion. All of them compile to the same
  timestamped trajectory, which is what lets a renderer accept any of them
  without knowing which it was given.

:mod:`.legacy_paths` remains the behaviour-preserving port of the legacy SAM2
path evaluation, which the compatibility adapter needs in order to delegate
existing voices to the canonical core.
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

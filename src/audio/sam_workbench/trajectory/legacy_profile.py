"""Translating the existing GUI path formats into canonical trajectories.

Two legacy formats reach the renderer:

* the **custom-path profile** written by `custom_path_creator_dialog` -
  ``{kind, closedLoop, smoothingPasses, smoothingRatio, points}`` in scene
  pixels;
* the **trajectory segments** written by `spatial_trajectory_dialog` -
  ``rotate``, ``oscillate``, and ``rotating_arc`` laws in degrees, with the
  legacy azimuth sign (positive toward the *right*).

Neither is rewritten on disk. A profile keeps its original pixel coordinates
and its original keys; version-2 metadata is only added when a user explicitly
commits a migration through :func:`upgrade_profile`. Everything else here is a
read-time translation into the canonical frame.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import GeometrySpec, PolylineGeometry, SplineGeometry
from .legacy_paths import chaikin_smooth_points
from .transforms import LegacyPathTransform
from .traversal import TraversalSpec

__all__ = [
    "LEGACY_PROFILE_VERSION",
    "PROFILE_VERSION",
    "legacy_azimuth_to_canonical_deg",
    "profile_points",
    "profile_round_trip",
    "profile_schema_version",
    "profile_to_geometry",
    "profile_transform",
    "segment_positions",
    "segment_to_trajectory",
    "segments_duration_s",
    "segments_positions",
    "upgrade_profile",
]

#: Version implied by a profile that carries no version at all.
LEGACY_PROFILE_VERSION = 1
#: Version written by :func:`upgrade_profile`, carrying coordinate metadata.
PROFILE_VERSION = 2


def legacy_azimuth_to_canonical_deg(azimuth_deg):
    """Convert a legacy ``slab``-style azimuth to the canonical convention.

    The legacy engine documents its own frame as ``+x`` right, ``+y`` forward,
    with azimuth positive toward the **right**. The canonical frame is ``+x``
    forward, ``+y`` left, azimuth positive toward the **left**, so the sign
    flips. This is the only place that flip is applied.
    """

    return -np.asarray(azimuth_deg, dtype=np.float64)


# --- custom path profiles ---------------------------------------------------


def _as_mapping(profile: Any) -> dict[str, Any]:
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except (TypeError, ValueError):
            return {}
    return dict(profile) if isinstance(profile, Mapping) else {}


def profile_schema_version(profile: Any) -> int:
    """The declared version of a custom-path profile; 1 when unversioned."""

    data = _as_mapping(profile)
    try:
        return int(data.get("schemaVersion", LEGACY_PROFILE_VERSION))
    except (TypeError, ValueError):
        return LEGACY_PROFILE_VERSION


def profile_points(profile: Any) -> NDArray[np.float64]:
    """The profile's raw ``(n, 2)`` scene points, unmodified."""

    data = _as_mapping(profile)
    points = data.get("points")
    if not isinstance(points, Sequence) or len(points) < 1:
        return np.zeros((0, 2), dtype=np.float64)
    cleaned = [
        (float(point[0]), float(point[1]))
        for point in points
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) == 2
    ]
    return np.array(cleaned, dtype=np.float64) if cleaned else np.zeros((0, 2), dtype=np.float64)


def profile_transform(profile: Any, default: LegacyPathTransform | None = None) -> LegacyPathTransform:
    """The coordinate bridge a profile declares, or the documented default.

    A version-1 profile declares nothing, so the default describes the editor
    that wrote it: pixels, ``+x`` right, ``+y`` down, centred on the scene
    origin, scaled so the editor's ear markers sit at the head radius.
    """

    data = _as_mapping(profile)
    base = default or LegacyPathTransform()
    if profile_schema_version(data) < PROFILE_VERSION:
        return base
    return LegacyPathTransform(
        centre_scene=tuple(data.get("centreScene", base.centre_scene)),
        axis_convention=str(data.get("axisConvention", base.axis_convention)),
        scene_units_per_metre=float(data.get("sceneUnitsPerMetre", base.scene_units_per_metre)),
        normalization_extent=data.get("normalizationExtent", base.normalization_extent),
        elevation_m=float(data.get("elevationM", base.elevation_m)),
    )


def upgrade_profile(profile: Any, transform: LegacyPathTransform | None = None) -> dict[str, Any]:
    """Return a version-2 copy of ``profile`` with explicit coordinate metadata.

    A migration the user commits, never one that happens on load: the returned
    dictionary is a copy, the original keys and points are untouched, and the
    added metadata states exactly what the legacy pixels already meant.
    """

    data = copy.deepcopy(_as_mapping(profile))
    bridge = transform or profile_transform(data)
    data.update(
        {
            "schemaVersion": PROFILE_VERSION,
            "coordinateSpace": "scene_pixels_2d",
            "axisConvention": bridge.axis_convention,
            "sceneUnitsPerMetre": float(bridge.scene_units_per_metre),
            "centreScene": list(bridge.centre_scene),
            "elevationM": float(bridge.elevation_m),
        }
    )
    if bridge.normalization_extent is not None:
        data["normalizationExtent"] = float(bridge.normalization_extent)
    data.setdefault("kind", "linear")
    data.setdefault("closedLoop", False)
    data.setdefault("smoothingPasses", 1)
    data.setdefault("smoothingRatio", 0.25)
    return data


def profile_to_geometry(
    profile: Any, transform: LegacyPathTransform | None = None
) -> GeometrySpec | None:
    """Translate a custom-path profile into canonical geometry, in metres.

    Smoothing follows the legacy rules exactly - Chaikin for every kind except
    ``spline``, which is drawn as a Catmull-Rom curve - so the canonical path
    has the shape the user drew, not an approximation of it.
    """

    data = _as_mapping(profile)
    points = profile_points(data)
    if points.shape[0] < 2:
        return None

    bridge = transform or profile_transform(data)
    kind = str(data.get("kind", "linear")).lower()
    closed = bool(data.get("closedLoop", False))

    scene_points = [tuple(point) for point in points]
    if closed and scene_points[0] != scene_points[-1]:
        scene_points.append(scene_points[0])
    if kind != "spline":
        scene_points = chaikin_smooth_points(
            scene_points,
            is_closed=closed,
            passes=int(data.get("smoothingPasses", 1)),
            ratio=float(data.get("smoothingRatio", 0.25)),
        )

    canonical = bridge.to_canonical(np.array(scene_points, dtype=np.float64))
    metres = tuple(tuple(float(value) for value in point) for point in canonical.T)
    if kind == "spline":
        return SplineGeometry(points_m=metres, closed=closed)
    return PolylineGeometry(points_m=metres, closed=closed)


def profile_round_trip(profile: Any, transform: LegacyPathTransform | None = None) -> NDArray[np.float64]:
    """Send a profile's points through canonical metres and back to pixels.

    Used to prove a legacy path maps into physical space *reproducibly*: the
    returned points must match the originals to floating-point tolerance.
    """

    points = profile_points(profile)
    if points.shape[0] == 0:
        return points
    bridge = transform or profile_transform(profile)
    return bridge.from_canonical(bridge.to_canonical(points))


# --- spatial trajectory segments --------------------------------------------


@dataclass(frozen=True)
class _SegmentLaw:
    """A legacy segment resolved into canonical azimuth and radius over time."""

    azimuth_deg: NDArray[np.float64]
    radius_m: NDArray[np.float64]


def _segment_distance(segment: Mapping[str, Any], progress: NDArray[np.float64]) -> NDArray[np.float64]:
    distance = segment.get("distance_m", 1.0)
    if isinstance(distance, Sequence) and not isinstance(distance, (str, bytes)) and len(distance) == 2:
        start, end = float(distance[0]), float(distance[1])
        return start + (end - start) * progress
    return np.full(progress.shape, float(distance))


def _segment_law(segment: Mapping[str, Any], times: NDArray[np.float64]) -> _SegmentLaw:
    mode = str(segment.get("mode", "rotate")).lower()
    seconds = max(1e-9, float(segment.get("seconds", 0.0) or 0.0))
    progress = np.clip(times / seconds, 0.0, 1.0)
    if str(segment.get("easing", "linear")).lower() == "sine":
        progress = 0.5 - 0.5 * np.cos(np.pi * progress)

    if mode == "rotate":
        legacy_azimuth = float(segment.get("start_deg", 0.0)) + float(
            segment.get("speed_deg_per_s", 0.0)
        ) * times
    elif mode == "rotating_arc":
        period = max(1e-9, float(segment.get("period_s", 1.0) or 1.0))
        base = float(segment.get("start_deg", 0.0)) + 360.0 * float(
            segment.get("rotate_freq_hz", 0.0)
        ) * times
        legacy_azimuth = base + 0.5 * float(segment.get("extent_deg", 0.0)) * np.sin(
            2.0 * math.pi * times / period
        )
    else:  # oscillate
        period = max(1e-9, float(segment.get("period_s", 1.0) or 1.0))
        legacy_azimuth = float(segment.get("center_deg", 0.0)) + 0.5 * float(
            segment.get("extent_deg", 0.0)
        ) * np.sin(2.0 * math.pi * times / period)

    return _SegmentLaw(
        azimuth_deg=legacy_azimuth_to_canonical_deg(legacy_azimuth),
        radius_m=_segment_distance(segment, progress),
    )


def segment_positions(segment: Mapping[str, Any], times: NDArray[np.floating]) -> NDArray[np.float64]:
    """Canonical ``(3, n)`` positions for one legacy trajectory segment."""

    time_array = np.atleast_1d(np.asarray(times, dtype=np.float64))
    law = _segment_law(segment, time_array)
    azimuth = np.radians(law.azimuth_deg)
    return np.vstack(
        (law.radius_m * np.cos(azimuth), law.radius_m * np.sin(azimuth), np.zeros(azimuth.shape))
    )


def segments_duration_s(segments: Sequence[Mapping[str, Any]]) -> float:
    """Total duration of a legacy segment list."""

    return float(sum(max(0.0, float(segment.get("seconds", 0.0) or 0.0)) for segment in segments))


def segments_positions(
    segments: Sequence[Mapping[str, Any]], times: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Canonical positions for a whole segment list on one absolute timeline.

    Each segment owns its own window; times past the end hold the final
    position rather than wrapping, matching how the legacy list plays.
    """

    time_array = np.atleast_1d(np.asarray(times, dtype=np.float64))
    positions = np.zeros((3, time_array.size), dtype=np.float64)
    if not segments:
        return positions

    elapsed = 0.0
    filled = np.zeros(time_array.size, dtype=bool)
    for segment in segments:
        span = max(0.0, float(segment.get("seconds", 0.0) or 0.0))
        inside = (time_array >= elapsed) & (time_array < elapsed + span)
        if np.any(inside):
            positions[:, inside] = segment_positions(segment, time_array[inside] - elapsed)
            filled |= inside
        elapsed += span

    remaining = ~filled
    if np.any(remaining):
        before = time_array < 0.0
        if np.any(before & remaining):
            positions[:, before & remaining] = segment_positions(segments[0], np.zeros(1))
        after = remaining & ~before
        if np.any(after):
            last = segments[-1]
            positions[:, after] = segment_positions(
                last, np.full(1, max(0.0, float(last.get("seconds", 0.0) or 0.0)))
            )
    return positions


def segment_to_trajectory(segment: Mapping[str, Any]) -> tuple[GeometrySpec, TraversalSpec] | None:
    """Express a legacy segment as canonical geometry plus traversal.

    ``rotate`` and ``oscillate`` map cleanly onto a circle or an arc with a
    timing law. ``rotating_arc`` superimposes two rates and has no single
    geometry-plus-traversal form, so it returns ``None`` and is evaluated
    directly by :func:`segment_positions`.
    """

    from .geometry import ArcGeometry, CircleGeometry

    mode = str(segment.get("mode", "rotate")).lower()
    distance = segment.get("distance_m", 1.0)
    radius = float(distance[0]) if isinstance(distance, Sequence) and not isinstance(distance, (str, bytes)) else float(distance)

    if mode == "rotate":
        speed = float(segment.get("speed_deg_per_s", 0.0))
        start = float(legacy_azimuth_to_canonical_deg(float(segment.get("start_deg", 0.0))))
        geometry = CircleGeometry(radius_m=radius, start_deg=start, end_deg=start + 360.0)
        traversal = TraversalSpec(
            rate_hz=-speed / 360.0,  # legacy azimuth grows to the right
            mode="loop",
            arc_length=False,
        )
        return geometry, traversal

    if mode == "oscillate":
        centre = float(legacy_azimuth_to_canonical_deg(float(segment.get("center_deg", 0.0))))
        half = 0.5 * float(segment.get("extent_deg", 0.0))
        period = max(1e-9, float(segment.get("period_s", 1.0) or 1.0))
        geometry = ArcGeometry(radius_m=radius, start_deg=centre - half, end_deg=centre + half)
        traversal = TraversalSpec(
            rate_hz=1.0 / period, mode="ping_pong", easing="sine", phase=0.5, arc_length=False
        )
        return geometry, traversal

    return None

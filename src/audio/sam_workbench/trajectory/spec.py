"""The composed trajectory: geometry + traversal + transform + frame.

Geometry, traversal, and transform are independently serializable, so one path
can be reused with a different timing law, and one timing law with a different
path. This module is where they meet and produce positions in metres in the
canonical listener frame, evaluated from absolute sample indices so a chunked
render matches a whole one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..conventions import cartesian_to_spherical_deg
from .geometry import CircleGeometry, GeometrySpec, geometry_from_dict
from .transforms import ListenerFrame, TransformSpec, transform_from_dict
from .traversal import TraversalResult, TraversalSpec, traversal_from_dict

__all__ = ["TrajectoryResult", "TrajectorySpec", "trajectory_from_dict"]


@dataclass(frozen=True)
class TrajectoryResult:
    """Source positions over time, and where the motion jumped."""

    positions_m: NDArray[np.float64]  # (3, frames), listener frame
    jumps: NDArray[np.bool_]
    crossfade_ms: float

    @property
    def frames(self) -> int:
        return int(self.positions_m.shape[1])

    def spherical_deg(self) -> NDArray[np.float64]:
        """``(3, frames)`` of azimuth, elevation, and radius."""

        return np.array(
            [cartesian_to_spherical_deg(self.positions_m[:, index]) for index in range(self.frames)],
            dtype=np.float64,
        ).T


@dataclass(frozen=True)
class TrajectorySpec:
    """Where a source is at any time, in the canonical listener frame."""

    geometry: GeometrySpec = field(default_factory=CircleGeometry)
    traversal: TraversalSpec = field(default_factory=TraversalSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    #: ``listener`` paths follow the head; ``world`` paths stay put when it turns.
    coordinate_frame: str = "listener"

    def __post_init__(self) -> None:
        if self.coordinate_frame not in ("listener", "world"):
            raise ValueError(f"unknown coordinate frame {self.coordinate_frame!r}")

    # --- evaluation ---------------------------------------------------------

    def positions_at(
        self, times: NDArray[np.floating], *, listener: ListenerFrame | None = None
    ) -> TrajectoryResult:
        """Evaluate the trajectory at absolute times in seconds."""

        traversal: TraversalResult = self.traversal.positions(times)
        parameter = traversal.position
        points = (
            self.geometry.evaluate_by_arc_length(parameter)
            if self.traversal.arc_length
            else self.geometry.evaluate(parameter)
        )
        points = self.transform.apply(points)
        if self.coordinate_frame == "world":
            frame = listener or ListenerFrame()
            points = frame.world_to_listener(points)
        return TrajectoryResult(
            positions_m=points, jumps=traversal.jumps, crossfade_ms=traversal.crossfade_ms
        )

    def positions_for_block(
        self,
        start_sample: int,
        frames: int,
        sample_rate: float,
        *,
        listener: ListenerFrame | None = None,
    ) -> TrajectoryResult:
        """Evaluate one render block from its absolute sample index."""

        times = self.traversal.sample_times(start_sample, frames, sample_rate)
        return self.positions_at(times, listener=listener)

    def sample_path(self, samples: int = 256) -> NDArray[np.float64]:
        """The path's shape alone, for a plot: ``(3, samples)`` in metres."""

        parameter = np.linspace(0.0, 1.0, int(samples))
        points = (
            self.geometry.evaluate_by_arc_length(parameter)
            if self.traversal.arc_length
            else self.geometry.evaluate(parameter)
        )
        return self.transform.apply(points)

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "geometry": self.geometry.to_dict(),
            "traversal": self.traversal.to_dict(),
            "transform": self.transform.to_dict(),
            "coordinate_frame": self.coordinate_frame,
        }


def trajectory_from_dict(data) -> TrajectorySpec:
    """Rebuild a trajectory from its serialized form."""

    return TrajectorySpec(
        geometry=geometry_from_dict(data["geometry"]),
        traversal=traversal_from_dict(data.get("traversal", {})),
        transform=transform_from_dict(data.get("transform", {})),
        coordinate_frame=str(data.get("coordinate_frame", "listener")),
    )

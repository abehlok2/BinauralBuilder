"""Time-varying path parameters driven by compiled controls.

A saved path stores its primitive's numbers as constants, and everything downstream -
renderers, coverage checks, the editor preview - reads positions through one
callable.  Modulation must therefore enter behind that same callable, or it
would become a second path interpretation competing with the first.

:class:`ModulatedPath` wraps a :class:`~.path_model.PathModel` plus a set of
bindings, each pairing a reserved parameter path (``path.radiusM``,
``transform.yawDeg``) with a control that yields the parameter's *effective
value* over time - the same contract every other automatable parameter has,
where scene automation resolves against the stored base and may blend from it
(stages) or add to it (modulation routes).  Positions stay a pure function of
absolute time: every value is read from its control by absolute sample number,
so a window's contents never depend on how the timeline was cut into blocks -
the same rule the compiled controls themselves guarantee.

Two honesty rules are built in rather than left to callers:

* Binding any geometry field invalidates the constant-speed arc-length table,
  which describes one frozen shape.  Such a path evaluates at the curve's own
  parameter speed instead, says so through :attr:`notes`, and keeps the flag on
  :attr:`uses_constant_speed_law` for readiness checks to report.
* Resolved values are held inside the ranges documented in
  :mod:`.parameter_catalog`, and coupled relationships (the torus tube staying
  on its ring) are re-enforced after every resolution - a modulated radius can
  never steer the path through the listener's head.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameter_catalog import (
    COUPLED_CONSTRAINTS,
    GEOMETRY_PARAMETER_SPECS,
    TRANSFORM_PARAMETER_SPECS,
    PathParameterSpec,
    split_parameter_path,
)
from .primitives import (
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
    dome_traversal_points,
    elevation_sweep_points,
    figure_eight_3d_points,
    horizontal_orbit_points,
    overhead_sweep_points,
    pendulum_points,
    random_walk_points,
    rising_arc_points,
    spherical_orbit_points,
    tilted_orbit_points,
    torus_points,
    vertical_orbit_points,
)
from .spherical import cartesian_array_to_spherical
from .transforms import apply_stacked, rotation_matrices_ypr
from .traversal import Traversal

__all__ = ["PathBinding", "ModulatedPath", "bind_path_parameters", "path_parameter_bases"]


#: Geometry classes whose parameters a route may drive, mapped to the pure
#: formula that evaluates them. Point-based kinds are shapes you edit by
#: dragging; there is no named number for a route to hold onto.
_SHAPE_FORMULAS: dict[type, Any] = {
    HorizontalOrbit: horizontal_orbit_points,
    VerticalOrbit: vertical_orbit_points,
    TiltedOrbit: tilted_orbit_points,
    SphericalOrbit: spherical_orbit_points,
    RisingArc: rising_arc_points,
    OverheadSweep: overhead_sweep_points,
    ElevationSweep: elevation_sweep_points,
    DomeTraversal: dome_traversal_points,
    FigureEight3D: figure_eight_3d_points,
    Pendulum: pendulum_points,
    Torus: torus_points,
    RandomWalkVolume: random_walk_points,
}

_TRIPLE_COMPONENTS = {
    "centre_m": ("centre_x_m", "centre_y_m", "centre_z_m"),
    "pivot_m": ("pivot_x_m", "pivot_y_m", "pivot_z_m"),
    "extent_m": ("extent_x_m", "extent_y_m", "extent_z_m"),
}


@dataclass(frozen=True)
class PathBinding:
    """One reserved parameter path paired with its offset source."""

    #: Full reserved spelling, normalized to snake case (``path.radius_m``).
    parameter_path: str
    #: ``"geometry"`` or ``"transform"``.
    section: str
    #: Snake-case field or component key (``radius_m``, ``yaw_deg``).
    field: str
    #: A plan ``CompiledControl`` (``at(start, frames)``) or a core
    #: ``ControlBase`` (``render(start, frames, rate)``). Either way the values
    #: are the parameter's effective value over time, already resolved against
    #: its stored base by whoever compiled the control.
    control: Any


def bind_path_parameters(
    model: Any,
    bindings: Sequence[PathBinding | tuple[str, Any]] | None,
    *,
    sample_rate_hz: float,
    origin_sample: int = 0,
) -> Any:
    """Bind offset controls to a path's parameters.

    Returns the model untouched when nothing binds - the guarantee that a
    project without path modulation renders byte-identically to before. Times
    given to the result are seconds since the source began sounding;
    ``origin_sample`` places that clock on the scene's absolute timeline.

    Two bindings may not claim one parameter: in production each parameter is
    compiled into a single control that already merges every stage and route
    reaching it, so a duplicate here would mean two answers to one question.
    """

    normalized: list[PathBinding] = []
    for entry in bindings or ():
        if isinstance(entry, PathBinding):
            normalized.append(entry)
        else:
            parameter_path, control = entry
            section, field = split_parameter_path(parameter_path)
            normalized.append(
                PathBinding(str(parameter_path), section, field, control)
            )
    if not normalized:
        return model
    return ModulatedPath(
        model,
        normalized,
        sample_rate_hz=float(sample_rate_hz),
        origin_sample=int(origin_sample),
    )


class ModulatedPath:
    """A :class:`PathModel` whose numbers move, behind the usual callables."""

    def __init__(
        self,
        model: Any,
        bindings: Sequence[PathBinding],
        *,
        sample_rate_hz: float,
        origin_sample: int = 0,
    ) -> None:
        self._model = model
        self._rate = float(sample_rate_hz)
        self._origin = int(origin_sample)
        if self._rate <= 0.0:
            raise ValueError("sample_rate_hz must be positive")

        geometry_groups: dict[str, list[PathBinding]] = {}
        transform_groups: dict[str, list[PathBinding]] = {}
        allowed = self._geometry_allowed_fields()
        supported = any(isinstance(model.geometry, kind) for kind in _SHAPE_FORMULAS)
        for binding in bindings:
            spec = self._spec_for(binding.section, binding.field)
            groups = (
                geometry_groups if binding.section == "geometry"
                else transform_groups
            )
            if binding.field in groups or any(
                binding.field in existing for existing in groups.values()
            ):
                raise ValueError(
                    f"{binding.parameter_path!r} is bound more than once; "
                    "compile one merged control per parameter instead"
                )
            if binding.section == "geometry":
                if not supported:
                    raise ValueError(
                        f"geometry {type(model.geometry).__name__} carries no "
                        "named parameters to modulate; routes must target a "
                        f"primitive from {sorted(kind.__name__ for kind in _SHAPE_FORMULAS)}"
                    )
                if binding.field not in allowed:
                    raise ValueError(
                        f"{binding.parameter_path!r} is not a parameter of "
                        f"{type(model.geometry).__name__}; expected one of "
                        f"{sorted(allowed)}"
                    )
            groups.setdefault(binding.field, []).append(binding)

        self._geometry_groups = geometry_groups
        self._transform_groups = transform_groups
        self._bindings = tuple(bindings)

    # --- description --------------------------------------------------------

    @property
    def model(self) -> Any:
        return self._model

    @property
    def bindings(self) -> tuple[PathBinding, ...]:
        return self._bindings

    @property
    def duration_s(self) -> float:
        return float(getattr(self._model.traversal, "duration_s", 0.0) or 0.0)

    @property
    def uses_constant_speed_law(self) -> bool:
        """False once a geometry field moves: the arc-length table describes
        one frozen shape, so a moving shape advances by curve parameter."""

        return not self._geometry_groups

    @property
    def notes(self) -> tuple[str, ...]:
        if self._geometry_groups:
            names = ", ".join(sorted(self._geometry_groups))
            return (
                f"Path parameters {names} are modulated, so constant-speed "
                "traversal falls back to the curve's own parameter speed.",
            )
        return ()

    # --- evaluation ---------------------------------------------------------

    def positions(self, time_s: ArrayLike) -> NDArray[np.float64]:
        """Listener-relative metres at voice-local seconds."""

        times = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        if times.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        traversal = self._model.traversal
        use_blend = (
            bool(self._model.coordinate_smoothing)
            and isinstance(traversal, Traversal)
            and traversal.mode == "discontinuous"
            and traversal.crossfade_s > 0.0
        )
        if use_blend:
            blend = traversal.discontinuity_blend(times)
            previous = self._resolve(blend.previous_progress, times)
            following = self._resolve(blend.next_progress, times)
            total = np.maximum(
                blend.previous_gain + blend.next_gain, np.finfo(float).eps
            )
            resolved = (
                previous * blend.previous_gain[..., None]
                + following * blend.next_gain[..., None]
            ) / total[..., None]
        else:
            resolved = self._resolve(traversal.progress(times), times)
        return self._to_listener(resolved)

    def spherical(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return cartesian_array_to_spherical(self.positions(time_s))

    def audio_branches(self, time_s: ArrayLike):
        """Positions either side of each discontinuity jump plus equal-power
        gains, for renderers that spatialize branches before mixing."""

        traversal = self._model.traversal
        if not isinstance(traversal, Traversal):
            return None
        times = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        if times.size == 0:
            empty = np.zeros((0, 3), dtype=np.float64)
            zeros = np.zeros(0, dtype=np.float64)
            ones = np.ones(0, dtype=np.float64)
            return empty, empty, zeros, ones
        blend = traversal.discontinuity_blend(time_s)
        previous = self._to_listener(self._resolve(blend.previous_progress, times))
        following = self._to_listener(self._resolve(blend.next_progress, times))
        return previous, following, blend.previous_gain, blend.next_gain

    def parameter_paths(self) -> tuple[str, ...]:
        """The reserved paths bound here, normalized spellings."""

        paths = [binding.parameter_path for group in self._geometry_groups.values() for binding in group]
        paths += [binding.parameter_path for group in self._transform_groups.values() for binding in group]
        return tuple(sorted(paths))

    def sample_parameter(self, parameter_path: str, time_s: ArrayLike):
        """One bound parameter's resolved values at these absolute times.

        ``None`` when this model does not bind that path. Readiness uses this
        to notice motion pressed against its documented limits.
        """

        section, field = split_parameter_path(parameter_path)
        group = (
            self._geometry_groups if section == "geometry" else self._transform_groups
        ).get(field)
        if not group:
            return None
        spec = self._spec_for(section, field)
        times = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        return spec.clamp(self._value(group[0], times))

    __call__ = positions

    # --- internals ----------------------------------------------------------

    def _geometry_allowed_fields(self) -> frozenset[str]:
        from .parameter_catalog import primitive_component_fields

        return primitive_component_fields(type(self._model.geometry))

    @staticmethod
    def _spec_for(section: str, field: str) -> PathParameterSpec:
        table = (
            GEOMETRY_PARAMETER_SPECS if section == "geometry"
            else TRANSFORM_PARAMETER_SPECS
        )
        spec = table.get(field)
        if spec is None:
            raise ValueError(f"unknown {section} path parameter {field!r}")
        return spec

    def _value(self, binding: PathBinding, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """The control's values at these absolute times."""

        samples = self._origin + np.rint(times * self._rate).astype(np.int64)
        low = int(samples.min())
        span = int(samples.max()) - low + 1
        window = self._control_window(binding.control, low, span)
        return window[samples - low]

    def _control_window(self, control: Any, start: int, frames: int) -> NDArray[np.float64]:
        at = getattr(control, "at", None)
        if callable(at):
            values = at(int(start), int(frames))
        else:
            render = getattr(control, "render", None)
            if not callable(render):
                raise TypeError(
                    "a path binding needs a CompiledControl (.at) or a "
                    "ControlBase (.render)"
                )
            values = render(int(start), int(frames), self._rate)
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (int(frames),):
            raise ValueError(
                f"path control produced {array.shape}, expected ({frames},)"
            )
        return array

    def _resolved_geometry_params(
        self, u: NDArray[np.float64], times: NDArray[np.float64]
    ) -> dict[str, Any]:
        """Base field values with modulated ones replaced by clamped series."""

        resolved: dict[str, Any] = {}
        count = len(u)
        for entry in dataclass_fields(self._model.geometry):
            value = getattr(self._model.geometry, entry.name)
            group = self._geometry_groups.get(entry.name)
            if group is not None:
                spec = self._spec_for("geometry", entry.name)
                resolved[entry.name] = spec.clamp(self._value(group[0], times))
                continue
            components = _TRIPLE_COMPONENTS.get(entry.name)
            if components and any(name in self._geometry_groups for name in components):
                resolved[entry.name] = self._stack_components(components, times, count)
                continue
            resolved[entry.name] = value
        for constraint in COUPLED_CONSTRAINTS:
            constraint(resolved)
        return resolved

    def _stack_components(
        self, components: Sequence[str], times: NDArray[np.float64], count: int
    ) -> NDArray[np.float64]:
        columns = []
        triple = np.asarray(
            getattr(self._model.geometry, _base_field_of(components[0])),
            dtype=np.float64,
        )
        for index, name in enumerate(components):
            base = float(triple[index])
            group = self._geometry_groups.get(name)
            if group is None:
                columns.append(np.full(count, base))
                continue
            spec = self._spec_for("geometry", name)
            columns.append(spec.clamp(self._value(group[0], times)))
        return np.stack(columns, axis=-1)

    def _shape_positions(
        self, u: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        geometry = self._model.geometry
        formula = _SHAPE_FORMULAS[type(geometry)]
        params = self._resolved_geometry_params(u, times)
        return np.asarray(formula(u, **params), dtype=np.float64)

    def _base_positions(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """The un-transformed geometry at these progresses, honouring the
        model's speed law while it still can."""

        model = self._model
        if not self._geometry_groups and model.speed_law == "constant_speed":
            from .geometry import evaluate_arclength

            return np.asarray(
                evaluate_arclength(
                    model.geometry, u, int(model.arclength_samples)
                ),
                dtype=np.float64,
            )
        return np.asarray(model.geometry.evaluate(u), dtype=np.float64)

    def _resolve(
        self, u: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raw = (
            self._shape_positions(u, times)
            if self._geometry_groups
            else self._base_positions(u)
        )
        return self._apply_transform(raw, times)

    def _apply_transform(
        self, points: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        transform = self._model.transform
        if not self._transform_groups:
            return np.asarray(transform.apply(points), dtype=np.float64)

        count = len(points)
        angles = []
        for name in ("yaw_deg", "pitch_deg", "roll_deg"):
            base_index = ("yaw_deg", "pitch_deg", "roll_deg").index(name)
            base = float(transform.yaw_pitch_roll_deg[base_index])
            group = self._transform_groups.get(name)
            if group:
                angles.append(np.asarray(self._value(group[0], times), dtype=np.float64))
            else:
                angles.append(np.full(count, base))
        matrices = rotation_matrices_ypr(*angles)

        translation = np.zeros((count, 3), dtype=np.float64)
        for axis, name in enumerate(
            ("translation_x_m", "translation_y_m", "translation_z_m")
        ):
            base = float(transform.translation_m[axis])
            group = self._transform_groups.get(name)
            if group:
                translation[:, axis] = self._value(group[0], times)
            else:
                translation[:, axis] = base

        xy, xz, yz = transform.shear
        shear = np.array(((1.0, xy, xz), (0.0, 1.0, yz), (0.0, 0.0, 1.0)))
        static_linear = shear @ np.diag(transform.scale)
        linear = np.einsum("nij,jk->nik", matrices, static_linear)
        return apply_stacked(points, linear) + translation

    def _to_listener(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model.is_listener_relative:
            return np.asarray(points, dtype=np.float64)
        return np.asarray(
            self._model.listener.world_to_listener(points), dtype=np.float64
        )


def _base_field_of(component: str) -> str:
    """``centre_x_m`` -> ``centre_m``: the stored triple a component belongs to."""

    stem = component.rsplit("_", 2)[0]
    return f"{stem}_m"


def path_parameter_bases(model: Any) -> dict[str, float]:
    """Every reserved parameter path this model can bind, with stored values.

    The bases are what scene automation resolves against: a stage blends from
    them and a modulation route adds to them, exactly as it would for any
    voice parameter. Keys are normalized snake-case spellings under their
    prefixes (``path.radius_m``, ``transform.yaw_deg``).
    """

    from .parameter_catalog import (
        PATH_PREFIX,
        TRANSFORM_PREFIX,
        primitive_component_fields,
    )

    bases: dict[str, float] = {}
    geometry = getattr(model, "geometry", None)
    if geometry is not None:
        allowed = primitive_component_fields(type(geometry))
        for field in sorted(allowed):
            spec = GEOMETRY_PARAMETER_SPECS.get(field)
            if spec is None:
                continue
            components = None
            for triple, names in (
                ("centre_m", ("centre_x_m", "centre_y_m", "centre_z_m")),
                ("pivot_m", ("pivot_x_m", "pivot_y_m", "pivot_z_m")),
                ("extent_m", ("extent_x_m", "extent_y_m", "extent_z_m")),
            ):
                if field in names:
                    components = (triple, names.index(field))
                    break
            if components is None:
                value = getattr(geometry, field, None)
                if isinstance(value, (int, float)):
                    bases[f"{PATH_PREFIX}{field}"] = float(value)
                continue
            triple_value = np.asarray(getattr(geometry, components[0]), dtype=np.float64)
            bases[f"{PATH_PREFIX}{field}"] = float(triple_value.reshape(-1)[components[1]])

    transform = getattr(model, "transform", None)
    if transform is not None:
        for index, name in enumerate(("yaw_deg", "pitch_deg", "roll_deg")):
            bases[f"{TRANSFORM_PREFIX}{name}"] = float(transform.yaw_pitch_roll_deg[index])
        for axis, name in enumerate(
            ("translation_x_m", "translation_y_m", "translation_z_m")
        ):
            bases[f"{TRANSFORM_PREFIX}{name}"] = float(transform.translation_m[axis])
    return bases

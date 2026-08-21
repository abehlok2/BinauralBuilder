"""The BinauralBuilder SAM compatibility adapter.

This module owns *every* legacy convention, so the core never has to:

* camelCase voice parameters to typed snake_case fields;
* ``initial_offset`` seconds to an absolute ``start_sample``, and for
  transitions to a transition-relative time origin - never to an angle;
* legacy path names and GUI profiles to the ported SAM2 path evaluation;
* static/transition parameter pairs to one continuous control;
* channel-major ``(2, frames)`` core audio to frame-major ``(frames, 2)``;
* unversioned voices to the legacy left-minus/right-plus ear polarity, while
  new voices can ask for the canonical left-plus/right-minus mode explicitly;
* unknown parameters preserved rather than erased.

Both public synthesis trees call in here, so one tested implementation backs
`src.synth_functions.spatial_angle_modulation` and
`binauralbuilder_core.synth_functions.spatial_angle_modulation`.

Two behaviours deliberately differ from the pre-Phase-1 code, because the old
ones were defects:

1. Static SAM2 now honours ``initial_offset`` as an absolute time. It used to
   ignore it, so every chunk restarted the waveform.
2. Transition SAM2 integrates phase from the transition's own time origin and
   anchors its parameter ramp to absolute step time. It used to restart the
   ramp inside every chunk and to add ``initial_offset`` to the sine argument
   as if seconds were radians.

Whole-step *static* renders are unchanged to the last bit; transition renders
change as described above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .conventions import AUDIO_DTYPE, seconds_to_samples, to_frame_major
from .dsp.source import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
    CompiledSource,
    render_source,
)
from .trajectory.geometry import ArcGeometry, GeometrySpec
from .trajectory.legacy_paths import SAM2_DEFAULT_SHAPES_BY_TYPE, resolve_sam2_shape
from .trajectory.legacy_profile import profile_to_geometry
from .trajectory.spec import TrajectorySpec
from .trajectory.traversal import TraversalSpec
from .waveforms import TWO_PI

__all__ = [
    "SAM2_PARAMETER_DEFAULTS",
    "NOMINAL_PATH_RADIUS_M",
    "sam2_trajectory",
    "SAM2_TRANSITION_PARAMETER_DEFAULTS",
    "SAM_SCHEMA_VERSION",
    "Sam2Spec",
    "compiled_source_from_sam2",
    "render_sam2",
    "render_sam2_voice",
    "sam2_spec_from_params",
]

#: Version stamped into a SAM voice payload once it has been written by this
#: build. Voices without it are unversioned legacy SAM2.
SAM_SCHEMA_VERSION = 1

#: The single authoritative table of static SAM2 parameters and defaults.
#: The GUI's parameter editors must read this rather than redeclaring it.
SAM2_PARAMETER_DEFAULTS: dict[str, Any] = {
    "amp": 0.7,
    "carrierFreq": 440.0,
    "modFreq": 4.0,
    "arcWidthDeg": 90.0,
    "directionOffsetDeg": 0.0,
    "spatialScale": 1.0,
    "pathType": "open",
    "pathShape": None,  # resolved from pathType
    "rotationDirection": "cw",
    "discontinuousSteps": 8,
    "customPathProfile": None,
    "phaseOffsetLRad": 0.0,
    "phaseOffsetRRad": 0.0,
}

#: Transition parameters. Each start/end pair compiles into one control.
SAM2_TRANSITION_PARAMETER_DEFAULTS: dict[str, Any] = {
    "startCarrierFreq": 440.0,
    "endCarrierFreq": 440.0,
    "startModFreq": 4.0,
    "endModFreq": 4.0,
    "startArcWidthDeg": 90.0,
    "endArcWidthDeg": 90.0,
    "startDirectionOffsetDeg": 0.0,
    "endDirectionOffsetDeg": 0.0,
    "startSpatialScale": 1.0,
    "endSpatialScale": 1.0,
}

#: Legacy aliases accepted for the same quantity, oldest name last.
_ALIASES: dict[str, tuple[str, ...]] = {
    "modFreq": ("modFreq", "beatFreq"),
    "arcWidthDeg": ("arcWidthDeg", "arcWidth"),
    "directionOffsetDeg": ("directionOffsetDeg", "directionOffset"),
    "startModFreq": ("startModFreq", "startBeatFreq"),
    "endModFreq": ("endModFreq", "endBeatFreq"),
}

#: Parameters of the duplicate `binauralbuilder_core` SAM2 implementation.
#: They are translated rather than dropped: `peakPhaseDev` scaled the
#: interaural phase exactly as `spatialScale` does, and the per-ear phase
#: offsets are audible, so they map onto the canonical per-ear phase fields.
_CORE_TREE_TRANSLATIONS: dict[str, str] = {
    "peakPhaseDev": "spatialScale",
    "startPeakPhaseDev": "startSpatialScale",
    "endPeakPhaseDev": "endSpatialScale",
    "phaseOffsetL": "phaseOffsetLRad",
    "phaseOffsetR": "phaseOffsetRRad",
    "startPhaseOffsetL": "phaseOffsetLRad",
    "startPhaseOffsetR": "phaseOffsetRRad",
}

_KNOWN_KEYS = (
    set(SAM2_PARAMETER_DEFAULTS)
    | set(SAM2_TRANSITION_PARAMETER_DEFAULTS)
    | set(_CORE_TREE_TRANSLATIONS)
    | {alias for aliases in _ALIASES.values() for alias in aliases}
    | {
        "samSchemaVersion",
        "rendererMode",
        "earPolarity",
        "customPathSmoothingPasses",
        "customPathSmoothingRatio",
        "endPhaseOffsetL",
        "endPhaseOffsetR",
        "transitionCurve",
    }
)


def _lookup(params: Mapping[str, Any], name: str, default: Any) -> Any:
    for key in _ALIASES.get(name, (name,)):
        if key in params and params[key] is not None:
            return params[key]
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class Sam2Spec:
    """A SAM2 voice in unit-explicit, snake_case form.

    Start/end pairs are kept as separate fields plus a transition window rather
    than being flattened, because the phase integral of a ramped frequency has
    a closed form only if the ramp itself is known.
    """

    amplitude: float = 0.7
    carrier_start_hz: float = 440.0
    carrier_end_hz: float = 440.0
    modulation_start_hz: float = 4.0
    modulation_end_hz: float = 4.0
    arc_width_start_deg: float = 90.0
    arc_width_end_deg: float = 90.0
    direction_offset_start_deg: float = 0.0
    direction_offset_end_deg: float = 0.0
    spatial_scale_start: float = 1.0
    spatial_scale_end: float = 1.0
    left_phase_rad: float = 0.0
    right_phase_rad: float = 0.0
    path_type: str = "open"
    path_shape: str = "sinusoidal"
    rotation_direction: str = "cw"
    discontinuous_steps: int = 8
    custom_path_profile: Any = None
    ear_polarity: str = EAR_POLARITY_LEGACY
    is_transition: bool = False
    #: Local time at which the transition starts; negative once it began in an
    #: earlier chunk. Meaningless for static voices.
    transition_start_s: float = 0.0
    transition_duration_s: float | None = None
    schema_version: int | None = None
    unknown_params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ramped(self) -> bool:
        """True when any parameter actually changes over the transition."""

        return self.is_transition and (
            self.carrier_start_hz != self.carrier_end_hz
            or self.modulation_start_hz != self.modulation_end_hz
            or self.arc_width_start_deg != self.arc_width_end_deg
            or self.direction_offset_start_deg != self.direction_offset_end_deg
            or self.spatial_scale_start != self.spatial_scale_end
        )


def sam2_spec_from_params(
    params: Mapping[str, Any],
    *,
    is_transition: bool = False,
    initial_offset: float = 0.0,
    transition_duration: float | None = None,
    duration: float | None = None,
) -> Sam2Spec:
    """Translate a legacy voice ``params`` mapping into a :class:`Sam2Spec`.

    Unknown keys are preserved on the spec so a round trip through the adapter
    cannot erase an older or external extension.
    """

    working = dict(params)
    for legacy_name, canonical_name in _CORE_TREE_TRANSLATIONS.items():
        if legacy_name in working and canonical_name not in working:
            working[canonical_name] = working[legacy_name]

    schema_version = working.get("samSchemaVersion")
    polarity = str(working.get("earPolarity", "")).strip()
    if polarity not in (EAR_POLARITY_CANONICAL, EAR_POLARITY_LEGACY, EAR_POLARITY_SAME):
        # Unversioned voices keep the legacy orientation; a versioned voice
        # that does not name a polarity opts into the canonical exact mode.
        polarity = EAR_POLARITY_LEGACY if schema_version is None else EAR_POLARITY_CANONICAL

    path_type = str(_lookup(working, "pathType", SAM2_PARAMETER_DEFAULTS["pathType"])).lower()
    path_shape = working.get("pathShape")
    if path_shape is None:
        path_shape = SAM2_DEFAULT_SHAPES_BY_TYPE.get(path_type, "sinusoidal")

    profile = working.get("customPathProfile", {})
    if isinstance(profile, dict):
        profile = dict(profile)
        if "customPathSmoothingPasses" in working:
            profile["smoothingPasses"] = int(
                _as_float(working.get("customPathSmoothingPasses"), profile.get("smoothingPasses", 1))
            )
        if "customPathSmoothingRatio" in working:
            profile["smoothingRatio"] = _as_float(
                working.get("customPathSmoothingRatio"), profile.get("smoothingRatio", 0.25)
            )

    def pair(static_name: str, start_name: str, end_name: str) -> tuple[float, float]:
        """Resolve a value that may be static, or a transition start/end pair."""

        static_default = SAM2_PARAMETER_DEFAULTS[static_name]
        static_value = _as_float(_lookup(working, static_name, static_default), static_default)
        start_value = _as_float(_lookup(working, start_name, static_value), static_value)
        end_value = _as_float(_lookup(working, end_name, start_value), start_value)
        if not is_transition:
            return static_value, static_value
        return start_value, end_value

    carrier_start, carrier_end = pair("carrierFreq", "startCarrierFreq", "endCarrierFreq")
    modulation_start, modulation_end = pair("modFreq", "startModFreq", "endModFreq")
    arc_start, arc_end = pair("arcWidthDeg", "startArcWidthDeg", "endArcWidthDeg")
    direction_start, direction_end = pair(
        "directionOffsetDeg", "startDirectionOffsetDeg", "endDirectionOffsetDeg"
    )
    scale_start, scale_end = pair("spatialScale", "startSpatialScale", "endSpatialScale")

    span = transition_duration
    if is_transition and span is None and duration is not None:
        span = max(0.0, float(duration) - max(0.0, float(initial_offset)))

    return Sam2Spec(
        amplitude=_as_float(_lookup(working, "amp", SAM2_PARAMETER_DEFAULTS["amp"]), 0.7),
        carrier_start_hz=carrier_start,
        carrier_end_hz=carrier_end,
        modulation_start_hz=modulation_start,
        modulation_end_hz=modulation_end,
        arc_width_start_deg=arc_start,
        arc_width_end_deg=arc_end,
        direction_offset_start_deg=direction_start,
        direction_offset_end_deg=direction_end,
        spatial_scale_start=scale_start,
        spatial_scale_end=scale_end,
        left_phase_rad=_as_float(working.get("phaseOffsetLRad", 0.0), 0.0),
        right_phase_rad=_as_float(working.get("phaseOffsetRRad", 0.0), 0.0),
        path_type=path_type,
        path_shape=str(path_shape).lower(),
        rotation_direction=str(
            _lookup(working, "rotationDirection", SAM2_PARAMETER_DEFAULTS["rotationDirection"])
        ).lower(),
        discontinuous_steps=int(
            _as_float(_lookup(working, "discontinuousSteps", 8), 8)
        ),
        custom_path_profile=profile,
        ear_polarity=polarity,
        is_transition=bool(is_transition),
        transition_start_s=float(initial_offset) if is_transition else 0.0,
        transition_duration_s=None if span is None else float(span),
        schema_version=None if schema_version is None else int(schema_version),
        unknown_params={key: value for key, value in params.items() if key not in _KNOWN_KEYS},
    )


# --- time bases -------------------------------------------------------------


def _spec_times(spec: Sam2Spec, start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
    """The time axis a spec's parameters are functions of.

    Static voices use absolute time derived from ``start_sample``, so a chunk
    continues the waveform instead of restarting it. Transitions use time
    relative to the transition's own origin, which BinauralBuilder keeps
    anchored to the step even while it renders chunk by chunk.
    """

    index = np.arange(int(start_sample), int(start_sample) + int(frames), dtype=np.float64)
    times = index / float(sample_rate)
    if spec.is_transition:
        return times - float(spec.transition_start_s)
    return times


def _ramp_alpha(spec: Sam2Spec, times: NDArray[np.float64]) -> NDArray[np.float64] | float:
    """Transition progress in ``[0, 1]`` at each relative time."""

    if not spec.is_transition:
        return 0.0
    span = spec.transition_duration_s
    if span is None or span <= 0.0:
        return np.where(times >= 0.0, 1.0, 0.0)
    return np.clip(times / float(span), 0.0, 1.0)


def _ramp_value(
    start_value: float, end_value: float, alpha: NDArray[np.float64] | float
) -> NDArray[np.float64] | float:
    if start_value == end_value:
        return float(start_value)
    return float(start_value) + (float(end_value) - float(start_value)) * alpha


def _ramp_phase(
    start_hz: float, end_hz: float, span_s: float | None, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Phase in radians from integrating a linearly ramped frequency.

    Integrating - rather than evaluating ``2*pi*f(t)*t`` - is what keeps the
    pitch correct while the frequency moves, and the closed form is what makes
    an arbitrary chunk start exact without carrying state.
    """

    if start_hz == end_hz:
        return TWO_PI * float(start_hz) * times
    if span_s is None or span_s <= 0.0:
        return TWO_PI * float(end_hz) * times

    span = float(span_s)
    slope = (float(end_hz) - float(start_hz)) / span
    before = TWO_PI * float(start_hz) * times
    inside = TWO_PI * (float(start_hz) * times + 0.5 * slope * np.square(times))
    at_end = TWO_PI * (float(start_hz) * span + 0.5 * slope * span * span)
    after = at_end + TWO_PI * float(end_hz) * (times - span)
    return np.where(times < 0.0, before, np.where(times <= span, inside, after))


# --- rendering --------------------------------------------------------------


def _modulation_provider(spec: Sam2Spec):
    """Build the callable that turns a SAM2 path into interaural phase."""

    def provider(start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = _spec_times(spec, start_sample, frames, sample_rate)
        alpha = _ramp_alpha(spec, times)
        modulation_phase = _ramp_phase(
            spec.modulation_start_hz, spec.modulation_end_hz, spec.transition_duration_s, times
        )
        shape, dynamic_scale = resolve_sam2_shape(
            spec.path_type,
            modulation_phase,
            spec.custom_path_profile,
            path_shape=spec.path_shape,
            discontinuous_steps=spec.discontinuous_steps,
            rotation_direction=spec.rotation_direction,
        )
        arc_width = _ramp_value(spec.arc_width_start_deg, spec.arc_width_end_deg, alpha)
        direction = _ramp_value(
            spec.direction_offset_start_deg, spec.direction_offset_end_deg, alpha
        )
        spatial_scale = _ramp_value(spec.spatial_scale_start, spec.spatial_scale_end, alpha)
        angle_deg = direction + 0.5 * arc_width * shape
        return (spatial_scale * dynamic_scale) * np.sin(np.radians(angle_deg))

    return provider


def _carrier_phase_provider(spec: Sam2Spec):
    def provider(start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = _spec_times(spec, start_sample, frames, sample_rate)
        return _ramp_phase(
            spec.carrier_start_hz, spec.carrier_end_hz, spec.transition_duration_s, times
        )

    return provider


def compiled_source_from_sam2(spec: Sam2Spec) -> CompiledSource:
    """Compile a legacy SAM2 voice into the canonical renderer's source type."""

    return CompiledSource(
        amplitude=spec.amplitude,
        carrier_frequency_hz=spec.carrier_start_hz,
        modulators=(),
        ear_polarity=spec.ear_polarity,
        left_phase_rad=spec.left_phase_rad,
        right_phase_rad=spec.right_phase_rad,
        modulation_provider=_modulation_provider(spec),
        carrier_phase_provider=_carrier_phase_provider(spec),
    )


def render_sam2(
    spec: Sam2Spec,
    frames: int,
    sample_rate: float,
    *,
    start_sample: int = 0,
    block_size: int | None = None,
) -> NDArray[np.float64]:
    """Render a SAM2 voice as channel-major ``(2, frames)`` float64 audio."""

    return render_source(
        compiled_source_from_sam2(spec),
        sample_rate,
        frames,
        block_size,
        start_sample=start_sample,
        dtype=np.float64,
    )


def render_sam2_voice(
    duration: float,
    sample_rate: float = 44_100,
    *,
    params: Mapping[str, Any] | None = None,
    is_transition: bool = False,
    initial_offset: float = 0.0,
    transition_duration: float | None = None,
    block_size: int | None = None,
) -> NDArray[np.float32]:
    """Render a legacy SAM2 voice and return frame-major ``(frames, 2)`` float32.

    This is the entry point both public synthesis trees call. The channel-major
    to frame-major conversion happens exactly here, once.
    """

    frames = int(float(duration) * float(sample_rate))
    if frames <= 0:
        return np.zeros((0, 2), dtype=AUDIO_DTYPE)

    spec = sam2_spec_from_params(
        params or {},
        is_transition=is_transition,
        initial_offset=float(initial_offset),
        transition_duration=transition_duration,
        duration=float(duration),
    )
    start_sample = 0 if is_transition else seconds_to_samples(initial_offset, sample_rate)
    audio = render_sam2(spec, frames, sample_rate, start_sample=start_sample, block_size=block_size)
    return to_frame_major(audio).astype(AUDIO_DTYPE, copy=False)


# --- canonical view of an abstract path -------------------------------------

#: Distance at which an abstract SAM path is drawn. Abstract phase modulation
#: places no source in metres - it modulates interaural phase directly - so a
#: radius is chosen only to make the swept angles visible.
NOMINAL_PATH_RADIUS_M = 1.0


def sam2_trajectory(
    spec: Sam2Spec, *, radius_m: float = NOMINAL_PATH_RADIUS_M
) -> TrajectorySpec:
    """The canonical trajectory a SAM2 voice's path describes.

    This is a **view**, not the renderer's model: the abstract engine modulates
    interaural phase directly and never computes a distance. The trajectory
    shows the angles that modulation sweeps, so the path a user draws and the
    geometry the geometric renderer would use can be compared side by side.

    A custom path is translated through the legacy coordinate bridge; every
    other path type becomes the arc its width and direction offset describe.
    """

    geometry: GeometrySpec | None = None
    if spec.path_type == "custom":
        geometry = profile_to_geometry(spec.custom_path_profile)
    if geometry is None:
        half_width = 0.5 * float(spec.arc_width_start_deg)
        centre = float(spec.direction_offset_start_deg)
        geometry = ArcGeometry(
            radius_m=float(radius_m), start_deg=centre - half_width, end_deg=centre + half_width
        )

    if spec.path_type == "closed":
        traversal = TraversalSpec(rate_hz=spec.modulation_start_hz, mode="loop", arc_length=False)
    elif spec.path_type == "discontinuous":
        steps = max(2, int(spec.discontinuous_steps))
        period = 1.0 / max(1e-9, float(spec.modulation_start_hz))
        traversal = TraversalSpec(
            rate_hz=0.0,
            mode="loop",
            arc_length=False,
            jump_times_s=tuple(period * index / steps for index in range(1, steps)),
            jump_step=1.0 / steps,
        )
    else:
        traversal = TraversalSpec(
            rate_hz=spec.modulation_start_hz, mode="ping_pong", easing="sine", arc_length=False
        )
    return TrajectorySpec(geometry=geometry, traversal=traversal)

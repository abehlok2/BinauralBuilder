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

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.conventions import AUDIO_DTYPE, db_to_linear, seconds_to_samples, to_frame_major
from src.audio.sam_workbench.dsp.blocks import RenderBlock, RenderContext
from src.audio.sam_workbench.dsp.source import (
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_SAME,
    CompiledSource,
    render_source,
)
from src.audio.sam_workbench.render.geometric import GeometricBinauralRenderer, GeometricSpec
from src.audio.sam_workbench.render.registry import REGISTRY
from src.audio.sam_workbench.trajectory.transforms import ListenerTransform
from src.audio.sam_workbench.trajectory import (
    path_model_from_dict,
    spherical_to_cartesian,
    trajectory_from_dict,
)
from src.audio.sam_workbench.trajectory.legacy_paths import SAM2_DEFAULT_SHAPES_BY_TYPE, resolve_sam2_shape
from src.audio.sam_workbench.waveforms import TWO_PI

__all__ = [
    "SAM2_PARAMETER_DEFAULTS",
    "SAM2_TRANSITION_PARAMETER_DEFAULTS",
    "SAM_SCHEMA_VERSION",
    "Sam2Spec",
    "compiled_source_from_sam2",
    "render_sam2",
    "render_sam2_voice",
    "render_voice_channels",
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
    "rendererMode": "abstract_pm",
    "hrtfAsset": None,
    "hrtfAssetHash": None,
    "hrtfOptions": {},
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
        "rendererMode", "hrtfAsset", "hrtfAssetHash", "hrtfOptions",
        "earPolarity",
        "customPathSmoothingPasses",
        "customPathSmoothingRatio",
        "endPhaseOffsetL",
        "endPhaseOffsetR",
        "transitionCurve",
        "canonicalTrajectory",
        "distanceLaw",
        "referenceDistanceM",
        "minimumDistanceM",
        "maximumDistanceM",
        "dopplerEnabled",
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
    #: Scene automation the abstract renderer follows per sample. ``None`` for
    #: every voice whose scene does not automate these, which is what keeps the
    #: established closed-form phase bit-exact wherever it already applied.
    #: Keys are ``"modulation"``/``"carrier"`` for integrated frequencies, and
    #: ``"arc_width"``/``"direction"``/``"spatial_scale"`` for read-per-sample
    #: shape parameters.
    automation_phase: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    automation_shape: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
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


#: Scene-automated parameters the abstract renderer can follow per sample, and
#: the field of :class:`Sam2Spec` each one drives. Frequencies are integrated;
#: the rest are read.
_INTEGRATED_AUTOMATION = {"modFreq": "modulation", "carrierFreq": "carrier"}
_SHAPE_AUTOMATION = {
    "arcWidthDeg": "arc_width",
    "directionOffsetDeg": "direction",
    "spatialScale": "spatial_scale",
}


def _scene_automation(
    sam_scene: Mapping[str, Any],
    source_id: str,
    base: Mapping[str, Any],
    sample_rate: float,
    origin_offset: int,
):
    """Compile the scene's automation for the parameters this renderer follows.

    Returns a mapping from parameter name to a callable over an absolute
    window. Only parameters the scene actually automates appear, so a voice
    with no scene automation gets an empty mapping and the untouched
    closed-form path.
    """

    from .scene_state import automated_paths, scene_parameter_series

    followed = set(_INTEGRATED_AUTOMATION) | set(_SHAPE_AUTOMATION)
    paths = [path for path in automated_paths(sam_scene, source_id) if path in followed]
    if not paths:
        return {}

    def series_for(path: str):
        def evaluate(start_sample: int, frames: int) -> NDArray[np.float64]:
            # ``start_sample`` is on the renderer's clock; the scene wants the
            # project's, which differ by the offset computed once by the caller.
            values = scene_parameter_series(
                sam_scene,
                source_id,
                int(origin_offset) + int(start_sample),
                int(frames),
                sample_rate,
                base,
            )
            if path in values:
                return np.asarray(values[path], dtype=np.float64)
            return np.full(int(frames), float(base.get(path, 0.0) or 0.0), dtype=np.float64)

        return evaluate

    return {path: series_for(path) for path in paths}


def _with_automation(spec: Sam2Spec, automation: Mapping[str, Any], sample_rate: float) -> Sam2Spec:
    """Replace a spec's constant phase providers with automated ones."""

    from .automation import AutomatedPhase

    replacements: dict[str, Any] = {}
    for name, role in _INTEGRATED_AUTOMATION.items():
        if name in automation:
            replacements[role] = AutomatedPhase(automation[name], sample_rate)
    shape = {role: automation[name] for name, role in _SHAPE_AUTOMATION.items() if name in automation}
    if not replacements and not shape:
        return spec
    return replace(spec, automation_phase=replacements or None, automation_shape=shape or None)


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

    integrated = spec.automation_phase or {}
    shaped = spec.automation_shape or {}

    def provider(start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
        times = _spec_times(spec, start_sample, frames, sample_rate)
        alpha = _ramp_alpha(spec, times)
        if "modulation" in integrated:
            modulation_phase = integrated["modulation"].at(start_sample, frames)
        else:
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
        arc_width = (
            shaped["arc_width"](start_sample, frames)
            if "arc_width" in shaped
            else _ramp_value(spec.arc_width_start_deg, spec.arc_width_end_deg, alpha)
        )
        direction = (
            shaped["direction"](start_sample, frames)
            if "direction" in shaped
            else _ramp_value(
                spec.direction_offset_start_deg, spec.direction_offset_end_deg, alpha
            )
        )
        spatial_scale = (
            shaped["spatial_scale"](start_sample, frames)
            if "spatial_scale" in shaped
            else _ramp_value(spec.spatial_scale_start, spec.spatial_scale_end, alpha)
        )
        angle_deg = direction + 0.5 * arc_width * shape
        return (spatial_scale * dynamic_scale) * np.sin(np.radians(angle_deg))

    return provider


def _carrier_phase_provider(spec: Sam2Spec):
    integrated = spec.automation_phase or {}
    if "carrier" in integrated:

        def automated(start_sample: int, frames: int, sample_rate: float) -> NDArray[np.float64]:
            return integrated["carrier"].at(start_sample, frames)

        return automated

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


def render_voice_channels(
    voice_params: Mapping[str, Any],
    frames: int,
    sample_rate: float,
    *,
    is_transition: bool = False,
    initial_offset: float = 0.0,
    transition_duration: float | None = None,
    duration: float | None = None,
    block_size: int | None = None,
    automation: Mapping[str, Any] | None = None,
    dataset_override: object | None = None,
) -> NDArray[np.float64]:
    """Render one voice's parameters to channel-major ``(2, frames)``.

    The renderer dispatch, on its own. Both the per-voice compatibility entry
    point and the compiled-plan executor call this, so the two cannot disagree
    about what a given set of parameters sounds like - which is the only way to
    migrate production onto the plan without an audible step.

    Scene gain and the frame-major conversion stay with the callers, because
    they belong to how a voice is placed rather than to how it is rendered.
    """

    spec = sam2_spec_from_params(
        voice_params,
        is_transition=is_transition,
        initial_offset=float(initial_offset),
        transition_duration=transition_duration,
        duration=float(duration if duration is not None else frames / float(sample_rate)),
    )
    start_sample = 0 if is_transition else seconds_to_samples(initial_offset, sample_rate)
    renderer_mode = str(voice_params.get("rendererMode", "abstract_pm")).lower()
    # The registry decides which modes exist and which the per-voice adapter can
    # drive; this dispatch only says how. A mode it does not carry is refused
    # with the registry's own message rather than a list repeated here.
    definition = REGISTRY.get(renderer_mode) if renderer_mode in REGISTRY else None
    if definition is None or not definition.voice_renderable:
        raise ValueError(
            f"rendererMode {renderer_mode!r} is not available in this build; "
            f"expected one of "
            f"{', '.join(entry.identifier for entry in REGISTRY.voice_renderable)}"
        )
    if automation and renderer_mode == "abstract_pm":
        spec = _with_automation(spec, automation, sample_rate)
    if renderer_mode == "hybrid":
        return _render_hybrid_voice(
            voice_params, frames, sample_rate, start_sample=start_sample,
            block_size=block_size,
        )
    if renderer_mode == "hrtf":
        return _render_hrtf_voice(
            voice_params, frames, sample_rate, start_sample=start_sample,
            block_size=block_size, dataset_override=dataset_override,
        )
    if renderer_mode == "geometric":
        return _render_geometric_voice(
            spec, voice_params, frames, sample_rate, start_sample=start_sample,
            block_size=block_size,
        )
    return render_sam2(
        spec, frames, sample_rate, start_sample=start_sample, block_size=block_size
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
    sam_scene: Mapping[str, Any] | None = None,
    source_id: str = "source.1",
    scene_start_s: float | None = None,
    apply_routing: bool = True,
) -> NDArray[np.float32]:
    """Render a legacy SAM2 voice and return frame-major ``(frames, 2)`` float32.

    This is the entry point both public synthesis trees call. The channel-major
    to frame-major conversion happens exactly here, once.
    """

    frames = int(float(duration) * float(sample_rate))
    if frames <= 0:
        return np.zeros((0, 2), dtype=AUDIO_DTYPE)

    voice_params = dict(params or {})
    automation: dict[str, Any] = {}
    if sam_scene:
        from .scene_state import automated_paths, scene_parameter_overrides

        # Where the scene's automation lands is what decides whether the render
        # is block-invariant. Parameters the renderer integrates or reads per
        # sample are compiled into functions of the absolute sample index and
        # handed over as such; anything left is resolved once, at the source's
        # own origin rather than at the start of whichever chunk is being
        # rendered, so that value too is the same however the caller cut the
        # timeline.
        scene_origin = float(initial_offset if scene_start_s is None else scene_start_s)
        # The renderer's own clock and the scene's timeline are both absolute
        # but need not share an origin: a transition renders from sample zero
        # while sitting somewhere later on the project timeline. The difference
        # is applied once, here, rather than being rediscovered per parameter.
        voice_origin = 0 if is_transition else seconds_to_samples(initial_offset, sample_rate)
        automation = _scene_automation(
            sam_scene,
            str(source_id),
            voice_params,
            sample_rate,
            seconds_to_samples(scene_origin, sample_rate) - voice_origin,
        )
        remaining = {
            path: value
            for path, value in scene_parameter_overrides(
                sam_scene, str(source_id), scene_origin, voice_params
            ).items()
            if path not in automation
        }
        voice_params.update(remaining)
    audio = render_voice_channels(
        voice_params,
        frames,
        sample_rate,
        is_transition=is_transition,
        initial_offset=float(initial_offset),
        transition_duration=transition_duration,
        duration=float(duration),
        block_size=block_size,
        automation=automation,
    )
    if sam_scene:
        from .scene_state import scene_gain_envelope

        gain_start = seconds_to_samples(
            initial_offset if scene_start_s is None else scene_start_s, sample_rate
        )
        audio *= scene_gain_envelope(
            sam_scene, str(source_id), gain_start, frames, sample_rate,
            include_routing=apply_routing,
        )[None, :]
    return to_frame_major(audio).astype(AUDIO_DTYPE, copy=False)


def _listener_from(options: Mapping[str, Any]) -> ListenerTransform:
    """The listener pose a voice's options describe, or the default one.

    Production ignored this entirely: a project could place and orient the
    listener and be rendered as though the head were at the origin facing
    forward, which silently moves every source.
    """

    data = options.get("listener")
    if not isinstance(data, Mapping):
        return ListenerTransform()
    return ListenerTransform(
        position_m=tuple(float(value) for value in data.get("positionM", (0.0, 0.0, 0.0))),
        yaw_pitch_roll_deg=tuple(
            float(value) for value in data.get("yawPitchRollDegrees", (0.0, 0.0, 0.0))
        ),
        ear_spacing_m=float(data.get("earSpacingM", 0.18)),
    )


def _hrtf_trajectory(params: Mapping[str, Any], options: Mapping[str, Any]):
    """The position function the HRTF renderer samples, in metres.

    A saved ``canonicalTrajectory`` is the whole path: azimuth, elevation and
    distance all vary along it, and all three reach the renderer.  This used to
    be discarded here - the renderer was handed a sinusoidal azimuth at a fixed
    ``elevationDeg`` and ``distanceM``, so a three-dimensional path authored in
    the editor was flattened to a horizontal sweep the moment it was previewed
    or exported.  A path is now only synthesized from those two options when
    there is no trajectory to use, which is the genuinely legacy case.
    """

    payload = params.get("canonicalTrajectory")
    if isinstance(payload, Mapping) and payload.get("geometry"):
        model = path_model_from_dict(payload)
        # ``positions`` resolves a world-frame path against the listener pose,
        # so the renderer receives listener-relative metres either way.
        return model.positions

    rate = float(params.get("modFreq", 4.0))
    width = float(params.get("arcWidthDeg", 90.0))
    direction = float(params.get("directionOffsetDeg", 0.0))
    radius = float(options.get("distanceM", 1.0))
    elevation = float(options.get("elevationDeg", 0.0))

    def legacy(times):
        azimuth = direction + 0.5 * width * np.sin(TWO_PI * rate * np.asarray(times))
        return spherical_to_cartesian(azimuth, elevation, radius)

    return legacy


def hrtf_coverage_report(
    params: Mapping[str, Any],
    dataset_positions_m,
    *,
    sample_rate_hz: float = 44100.0,
    duration_s: float | None = None,
):
    """Check a voice's path against a SOFA dataset's measured directions.

    Sampled at the renderer's own control interval so the direction-change
    warnings describe the steps the renderer will really take.  Returns
    ``None`` when the voice is not an HRTF voice, so a caller can ask about any
    voice without first classifying it.
    """

    from .hrtf.coverage import assess_path_coverage

    if str(params.get("rendererMode", "abstract_pm")).lower() != "hrtf":
        return None
    options = dict(params.get("hrtfOptions") or {})
    trajectory = _hrtf_trajectory(params, options)
    interval = max(int(options.get("controlIntervalSamples", 128)), 1)

    if duration_s is None:
        payload = params.get("canonicalTrajectory")
        traversal = (payload or {}).get("traversal", {}) if isinstance(payload, Mapping) else {}
        duration_s = float(traversal.get("durationS", 5.0))
    count = max(2, int(float(duration_s) * float(sample_rate_hz) / interval))
    # Cap the sample count: a long path checked at the control rate would
    # otherwise build a very large direction array to answer a yes/no question.
    count = min(count, 20_000)
    times = np.arange(count, dtype=np.float64) * (interval / float(sample_rate_hz))
    return assess_path_coverage(
        dataset_positions_m,
        np.asarray(trajectory(times), dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        control_interval_samples=interval,
        crossfade_ms=float(options.get("crossfadeMs", 10.0)),
        interpolation=str(options.get("interpolation", "nearest")),
    )


def _headphone_from(options: Mapping[str, Any]):
    """The headphone correction a voice asks for, if any."""

    from .hrtf.headphones import HeadphoneCorrection

    if not options.get("headphoneAsset") and not options.get("headphoneMode"):
        return None
    try:
        return HeadphoneCorrection.from_mapping(dict(options))
    except (AttributeError, TypeError, ValueError):
        return None


def _render_hybrid_voice(
    params: Mapping[str, Any],
    frames: int,
    sample_rate: float,
    *,
    start_sample: int,
    block_size: int | None,
) -> NDArray[np.float64]:
    """Source -> SAM -> trajectory -> HRTF -> cue -> headphone -> output.

    The order is the point, and it is why each stage sits where it does. Cue
    modification acts on the *filters*: interaural time and level differences
    are properties of the filter pair for a direction, and recovering them from
    an already-mixed stereo signal would mean undoing the convolution first.
    Headphone correction runs once over the finished mix, because applying it
    per stem would apply it twice. Everything past the HRTF stage is a declared
    departure from the measured cues rather than a measurement.
    """

    from .hrtf.headphones import apply_correction
    from .hrtf.modification import transform_dataset
    from .render.anchor import anchor_directions, make_anchor_signal
    from .render.hybrid import HybridSpec, _ModifiedDatasetView

    options = dict(params.get("hrtfOptions") or {})
    spec = HybridSpec.from_options(options, headphone=_headphone_from(options))

    dataset = None
    if not spec.cue.is_neutral:
        from .hrtf import default_hrtf_cache

        loaded = default_hrtf_cache.get(
            params.get("hrtfAsset"), int(sample_rate),
            str(options.get("delayPolicy", "bake_delay_into_ir")),
            options.get("projectDirectory"),
        )
        dataset = _ModifiedDatasetView(loaded, transform_dataset(loaded, spec.cue).hrirs)

    mixed = _render_hrtf_voice(
        params, frames, sample_rate, start_sample=start_sample,
        block_size=block_size, dataset_override=dataset,
    )

    if spec.anchor.enabled:
        # A second source through the same filters, offset from the first. It
        # is rendered rather than faked so it carries the same spatial
        # treatment as the source it anchors.
        trajectory = _hrtf_trajectory(params, options)
        times = (np.arange(frames, dtype=np.float64) + int(start_sample)) / float(sample_rate)
        directions = anchor_directions(
            np.asarray(trajectory(times), dtype=np.float64), spec.anchor
        )
        signal = make_anchor_signal(spec.anchor, frames, float(sample_rate))
        mixed = mixed + _render_directions(
            signal, directions, params, options, sample_rate, block_size, dataset
        )

    if spec.headphone is not None and spec.headphone.is_active:
        mixed = apply_correction(mixed, spec.headphone)
    if spec.output_gain_db != 0.0:
        mixed = mixed * float(db_to_linear(spec.output_gain_db))
    return mixed


def _render_directions(
    mono, directions, params, options, sample_rate, block_size, dataset
) -> NDArray[np.float64]:
    """Render a signal along an explicit direction array rather than a path."""

    from .render.hrtf import HRTFRendererSpec, render_hrtf

    positions = np.asarray(directions, dtype=np.float64)

    def path(times):
        index = np.clip(
            (np.asarray(times) * float(sample_rate)).astype(int), 0, len(positions) - 1
        )
        return positions[index]

    spec = HRTFRendererSpec(
        sofa_path=params.get("hrtfAsset"),
        trajectory=path,
        interpolation=str(options.get("interpolation", "nearest")),
        delay_policy=str(options.get("delayPolicy", "bake_delay_into_ir")),
        crossfade_ms=float(options.get("crossfadeMs", 10.0)),
        neighbor_count=int(options.get("neighborCount", 3)),
        dataset_override=dataset,
        distance_law="none",
        propagation_delay=False,
    )
    return np.asarray(
        render_hrtf(mono, spec, int(sample_rate), block_size=block_size or 4096),
        dtype=np.float64,
    )


def _render_hrtf_voice(
    params: Mapping[str, Any],
    frames: int,
    sample_rate: float,
    *,
    start_sample: int,
    block_size: int | None,
    dataset_override: object | None = None,
) -> NDArray[np.float64]:
    """Translate the versioned voice envelope into the explicit-SOFA renderer."""

    asset = params.get("hrtfAsset")
    if not asset:
        raise ValueError("rendererMode 'hrtf' requires an explicit hrtfAsset SOFA path")
    options = dict(params.get("hrtfOptions") or {})
    if int(options.get("schemaVersion", 1)) != 1:
        raise ValueError("unsupported hrtfOptions schemaVersion")
    from .render.hrtf import HRTFRendererSpec, render_hrtf

    carrier = float(params.get("carrierFreq", 440.0))
    amplitude = float(params.get("amp", 0.7))
    trajectory = _hrtf_trajectory(params, options)

    # Distance handling is opt-in, and stays off for a voice that has no
    # trajectory: those sit at one fixed distance, where attenuation and
    # propagation delay are a constant gain and a constant latency that only
    # change how the voice lines up against the rest of the mix.
    has_trajectory = isinstance(params.get("canonicalTrajectory"), Mapping)
    distance_law = str(
        options.get("distanceLaw", "inverse" if has_trajectory else "none")
    )
    renderer_spec = HRTFRendererSpec(
        sofa_path=asset,
        trajectory=trajectory,
        interpolation=str(options.get("interpolation", "nearest")),
        delay_policy=str(options.get("delayPolicy", "bake_delay_into_ir")),
        crossfade_ms=float(options.get("crossfadeMs", 10.0)),
        control_interval_samples=int(options.get("controlIntervalSamples", 128)),
        expected_sha256=params.get("hrtfAssetHash"),
        project_directory=options.get("projectDirectory"),
        distance_law=distance_law,
        reference_distance_m=_as_float(options.get("referenceDistanceM", 1.0), 1.0),
        minimum_distance_m=_as_float(options.get("minimumDistanceM", 0.15), 0.15),
        maximum_distance_m=_as_float(options.get("maximumDistanceM", 100.0), 100.0),
        propagation_delay=bool(
            options.get("propagationDelay", has_trajectory)
        ),
        # Interpolation settings the production path used to drop on the floor:
        # a project could ask for a three-neighbour blend at a chosen harmonic
        # order, have it validate, and be rendered nearest-neighbour anyway.
        neighbor_count=int(options.get("neighborCount", 3)),
        harmonic_order=(
            None if options.get("harmonicOrder") is None
            else int(options["harmonicOrder"])
        ),
        max_angular_error_deg=(
            None if options.get("maxAngularErrorDeg") is None
            else _as_float(options["maxAngularErrorDeg"], 1.0)
        ),
        min_control_interval_samples=int(
            options.get("minControlIntervalSamples",
                        options.get("controlIntervalSamples", 128))
        ),
        max_control_interval_samples=int(options.get("maxControlIntervalSamples", 4096)),
        listener=_listener_from(options),
        dataset_override=dataset_override,
    )
    # Reconstruct the convolution and crossfade history from the voice origin,
    # since the legacy synth boundary cannot return checkpointed renderer state.
    discard = int(start_sample)
    render_frames = frames + discard
    samples = np.arange(render_frames, dtype=np.float64)
    mono = amplitude * np.sin(TWO_PI * carrier * samples / float(sample_rate))
    rendered = render_hrtf(
        mono, renderer_spec, int(sample_rate), block_size=block_size or 4096,
        start_sample=0,
    )
    return rendered[:, discard:discard + frames].astype(np.float64, copy=False)


def _render_geometric_voice(
    spec: Sam2Spec,
    params: Mapping[str, Any],
    frames: int,
    sample_rate: float,
    *,
    start_sample: int,
    block_size: int | None,
) -> NDArray[np.float64]:
    """Render a saved canonical trajectory, including seek pre-roll.

    The adapter can reconstruct its sine source at any absolute sample.  It
    therefore renders the maximum propagation-history window immediately
    before a requested chunk and discards it.  This makes random scheduling
    match a sequential export rather than resetting to a silent delay line.
    """

    payload = params.get("canonicalTrajectory")
    if not isinstance(payload, Mapping):
        raise ValueError("geometric rendererMode requires canonicalTrajectory")
    trajectory = trajectory_from_dict(payload)
    maximum_distance = _as_float(params.get("maximumDistanceM", 100.0), 100.0)
    geometric_spec = GeometricSpec(
        trajectory=trajectory,
        distance_law=str(params.get("distanceLaw", "inverse")),
        reference_distance_m=_as_float(params.get("referenceDistanceM", 1.0), 1.0),
        minimum_distance_m=_as_float(params.get("minimumDistanceM", 0.05), 0.05),
        maximum_distance_m=maximum_distance,
        doppler_enabled=bool(params.get("dopplerEnabled", True)),
    )
    history_frames = int(np.ceil(maximum_distance / geometric_spec.speed_of_sound_m_s * sample_rate)) + 4
    render_start = max(0, int(start_sample) - history_frames)
    discard = int(start_sample) - render_start
    total_frames = discard + frames
    times = _spec_times(spec, render_start, total_frames, sample_rate)
    # Geometric mode spatializes a mono source; its phase is accumulated on
    # the same absolute clock as abstract PM and is reconstructible on seeks.
    mono = spec.amplitude * np.sin(
        _ramp_phase(spec.carrier_start_hz, spec.carrier_end_hz, spec.transition_duration_s, times)
    )
    size = int(block_size or total_frames or 1)
    context = RenderContext(int(sample_rate), size)
    renderer = GeometricBinauralRenderer(geometric_spec)
    renderer.prepare(context)
    pieces = []
    for offset in range(0, total_frames, size):
        span = min(size, total_frames - offset)
        pieces.append(renderer.process(mono[offset : offset + span], RenderBlock(render_start + offset, span)))
    rendered = np.concatenate(pieces, axis=1)
    return rendered[:, discard:].astype(np.float64, copy=False)

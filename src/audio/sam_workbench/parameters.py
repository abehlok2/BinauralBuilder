"""The one SAM parameter registry.

Before this module the SAM/SAM2 parameter tables were declared twice inside
`voice_editor_dialog.get_default_params_for_function` - literally two entries
with the same key in one dict, where the later silently shadowed the earlier -
and the tooltips, ranges, and path-shape lists lived in three more places. A
field added to one block and not the others produced inconsistent
static/transition or default/preset behaviour.

Every SAM field is now declared exactly once, here, with:

* its serialized camelCase name and any legacy aliases;
* unit, bounds, decimals, and default;
* a tooltip;
* which disclosure mode shows it (basic / advanced / expert);
* its transition policy - paired `start*`/`end*`, shared across a transition,
  or timing-only;
* whether it is automatable.

The GUI generates its rows from this registry; the compatibility adapter
translates the same names into the typed core spec. Serialized keys stay
camelCase at the boundary, so existing track JSON and `.voice` presets are
unaffected.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .compat import SAM2_PARAMETER_DEFAULTS
from .conventions import MAX_SAMPLE_RATE_HZ
from .dsp.source import EAR_POLARITY_CANONICAL, EAR_POLARITY_LEGACY, EAR_POLARITY_SAME
from .hrtf.dataset import BAKE_DELAY, DELAY_POLICIES
from .hrtf.headphones import CORRECTION_MODES, OFF as HEADPHONE_OFF
from .hrtf.interpolation import INTERPOLATION_MODES, NEAREST
from .validation import ValidationIssue

__all__ = [
    "BASIC",
    "ADVANCED",
    "EXPERT",
    "PATH_SHAPES",
    "PATH_TYPES",
    "RENDERER_MODES",
    "ROTATION_DIRECTIONS",
    "EAR_POLARITIES",
    "SAM2_FIELDS",
    "ParameterField",
    "field_for",
    "fields_for_mode",
    "legacy_editor_defaults",
    "sam2_parameter_defaults",
    "transition_names",
    "validate_sam2_params",
]

BASIC = "basic"
ADVANCED = "advanced"
EXPERT = "expert"
_MODE_ORDER = (BASIC, ADVANCED, EXPERT)

PATH_TYPES: tuple[str, ...] = ("open", "closed", "discontinuous", "custom")
#: Shape names accepted by the legacy path evaluation, including its aliases.
PATH_SHAPES: tuple[str, ...] = ("sinusoidal", "triangle", "ramp", "saw", "square")
ROTATION_DIRECTIONS: tuple[str, ...] = ("cw", "ccw")
EAR_POLARITIES: tuple[str, ...] = (
    EAR_POLARITY_LEGACY,
    EAR_POLARITY_CANONICAL,
    EAR_POLARITY_SAME,
)

#: How a voice is rendered. An existing voice names none of these and keeps
#: rendering as abstract phase modulation, which is what it always was.
RENDERER_MODES: tuple[str, ...] = ("abstract_pm", "geometric", "hrtf", "hybrid")

#: The legacy default custom-path profile written by the existing editor.
DEFAULT_CUSTOM_PATH_PROFILE: dict[str, Any] = {
    "kind": "linear",
    "points": [],
    "smoothingPasses": 1,
    "smoothingRatio": 0.25,
}


@dataclass(frozen=True)
class ParameterField:
    """One SAM parameter, declared once."""

    name: str
    label: str
    default: Any
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 2
    tooltip: str = ""
    mode: str = BASIC
    kind: str = "float"  # float | int | choice | bool | json | path | text
    choices: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    #: "pair" - the transition form is `start<Name>`/`end<Name>`;
    #: "shared" - one value spans the transition;
    #: "timing" - transition timing only, never present on a static voice.
    transition: str = "pair"
    automatable: bool = True
    #: False for fields the pre-Phase-2 generic editor never showed. They are
    #: available in the workbench but are not added to existing voices, so an
    #: old preset keeps exactly the keys it had.
    in_legacy_editor: bool = True
    #: Whether the value may safely change while preview is playing.
    live_safe: bool = True

    @property
    def capitalized(self) -> str:
        return self.name[:1].upper() + self.name[1:]

    @property
    def start_name(self) -> str:
        return f"start{self.capitalized}"

    @property
    def end_name(self) -> str:
        return f"end{self.capitalized}"

    def names_for(self, is_transition: bool) -> tuple[str, ...]:
        """The serialized keys this field contributes in the given mode."""

        if self.transition == "timing":
            return (self.name,) if is_transition else ()
        if is_transition and self.transition == "pair":
            return (self.start_name, self.end_name)
        return (self.name,)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


SAM2_FIELDS: tuple[ParameterField, ...] = (
    ParameterField(
        name="amp",
        label="Amplitude",
        default=SAM2_PARAMETER_DEFAULTS["amp"],
        unit="linear",
        minimum=0.0,
        maximum=1.0,
        decimals=3,
        tooltip="Signal amplitude before the master gain (0-1).",
        mode=BASIC,
        transition="shared",
    ),
    ParameterField(
        name="carrierFreq",
        label="Carrier frequency",
        default=SAM2_PARAMETER_DEFAULTS["carrierFreq"],
        unit="Hz",
        minimum=0.0,
        maximum=MAX_SAMPLE_RATE_HZ / 2.0,
        decimals=2,
        tooltip="Carrier frequency in Hz (the audible tone).",
        mode=BASIC,
    ),
    ParameterField(
        name="modFreq",
        label="Motion rate",
        default=SAM2_PARAMETER_DEFAULTS["modFreq"],
        unit="Hz",
        minimum=0.0,
        maximum=200.0,
        decimals=3,
        tooltip="Spatial modulation rate in Hz: how fast the source traverses its path.",
        mode=BASIC,
        aliases=("beatFreq",),
    ),
    ParameterField(
        name="arcWidthDeg",
        label="Arc width",
        default=SAM2_PARAMETER_DEFAULTS["arcWidthDeg"],
        unit="deg",
        minimum=0.0,
        maximum=360.0,
        decimals=1,
        tooltip="Angular width in degrees swept by the virtual source.",
        mode=BASIC,
        aliases=("arcWidth",),
    ),
    ParameterField(
        name="directionOffsetDeg",
        label="Direction offset",
        default=SAM2_PARAMETER_DEFAULTS["directionOffsetDeg"],
        unit="deg",
        minimum=-360.0,
        maximum=360.0,
        decimals=1,
        tooltip="Midpoint direction of the path in degrees; positive is toward the left.",
        mode=BASIC,
        aliases=("directionOffset",),
    ),
    ParameterField(
        name="spatialScale",
        label="Motion depth",
        default=SAM2_PARAMETER_DEFAULTS["spatialScale"],
        unit="rad",
        minimum=0.0,
        maximum=8.0,
        decimals=3,
        tooltip="Multiplier for the interaural phase, in radians: the depth of the effect.",
        mode=BASIC,
        aliases=("peakPhaseDev",),
    ),
    ParameterField(
        name="pathType",
        label="Path type",
        default=SAM2_PARAMETER_DEFAULTS["pathType"],
        tooltip="Path type: open (back and forth), closed (looping), discontinuous (stepped), or custom.",
        mode=BASIC,
        kind="choice",
        choices=PATH_TYPES,
        transition="shared",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="pathShape",
        label="Path shape",
        default="sinusoidal",
        tooltip="Shape used to traverse open/closed/discontinuous paths.",
        mode=BASIC,
        kind="choice",
        choices=PATH_SHAPES,
        transition="shared",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="rotationDirection",
        label="Rotation direction",
        default=SAM2_PARAMETER_DEFAULTS["rotationDirection"],
        tooltip="Traversal direction for closed and discontinuous paths.",
        mode=ADVANCED,
        kind="choice",
        choices=ROTATION_DIRECTIONS,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="discontinuousSteps",
        label="Discontinuous steps",
        default=SAM2_PARAMETER_DEFAULTS["discontinuousSteps"],
        unit="steps",
        minimum=2,
        maximum=64,
        decimals=0,
        tooltip="Number of discrete positions for a discontinuous path.",
        mode=ADVANCED,
        kind="int",
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="phaseOffsetLRad",
        label="Left ear phase offset",
        default=0.0,
        unit="rad",
        minimum=-6.283185307179586,
        maximum=6.283185307179586,
        decimals=4,
        tooltip="Fixed phase added to the left ear, in radians.",
        mode=ADVANCED,
        transition="shared",
        aliases=("phaseOffsetL",),
        in_legacy_editor=False,
    ),
    ParameterField(
        name="phaseOffsetRRad",
        label="Right ear phase offset",
        default=0.0,
        unit="rad",
        minimum=-6.283185307179586,
        maximum=6.283185307179586,
        decimals=4,
        tooltip="Fixed phase added to the right ear, in radians.",
        mode=ADVANCED,
        transition="shared",
        aliases=("phaseOffsetR",),
        in_legacy_editor=False,
    ),
    ParameterField(
        name="earPolarity",
        label="Ear polarity",
        default=EAR_POLARITY_LEGACY,
        tooltip=(
            "Which ear receives the positive modulation. Unversioned legacy voices use "
            "left-minus/right-plus; the canonical exact mode is left-plus/right-minus."
        ),
        mode=EXPERT,
        kind="choice",
        choices=EAR_POLARITIES,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="customPathSmoothingPasses",
        label="Custom path smoothing passes",
        default=1,
        unit="passes",
        minimum=0,
        maximum=6,
        decimals=0,
        tooltip="Chaikin smoothing passes applied to custom-path points (0-6).",
        mode=ADVANCED,
        kind="int",
        transition="shared",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="customPathSmoothingRatio",
        label="Custom path smoothing ratio",
        default=0.25,
        minimum=0.001,
        maximum=0.499,
        decimals=3,
        tooltip="Chaikin smoothing ratio for custom-path refinement (0.001-0.499).",
        mode=ADVANCED,
        transition="shared",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="customPathProfile",
        label="Custom path profile",
        default=DEFAULT_CUSTOM_PATH_PROFILE,
        tooltip="Saved custom path profile, used when the path type is custom.",
        mode=ADVANCED,
        kind="json",
        transition="shared",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="rendererMode",
        label="Renderer",
        default="abstract_pm",
        tooltip=(
            "Which engine renders this voice: abstract phase modulation, geometric "
            "binaural, HRTF convolution, or the hybrid of the last two."
        ),
        mode=ADVANCED,
        kind="choice",
        choices=RENDERER_MODES,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="hrtfAsset",
        label="HRTF asset",
        default="",
        tooltip=(
            "Path to the SOFA file this voice renders through. Stored with the project "
            "so a render can be reproduced; the dataset itself lives outside the project."
        ),
        mode=ADVANCED,
        kind="path",
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="hrtfSubject",
        label="HRTF subject",
        default="",
        tooltip="Subject identifier within the dataset, such as P0001 or KB0065 (KEMAR).",
        mode=ADVANCED,
        kind="text",
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="hrtfInterpolation",
        label="HRTF interpolation",
        default=NEAREST,
        tooltip=(
            "Nearest with crossfade is the safe default: it never blends raw responses, "
            "so it cannot smear a transient. The others blend after delay alignment."
        ),
        mode=ADVANCED,
        kind="choice",
        choices=INTERPOLATION_MODES,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="hrtfDelayPolicy",
        label="Interaural delay",
        default=BAKE_DELAY,
        tooltip=(
            "Whether the asset's Data_Delay is applied to the responses on load or kept "
            "and applied at render time. A nonzero delay is never ignored."
        ),
        mode=ADVANCED,
        kind="choice",
        choices=DELAY_POLICIES,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="hrtfCrossfadeMs",
        label="HRTF crossfade",
        default=12.0,
        unit="ms",
        minimum=5.0,
        maximum=20.0,
        decimals=1,
        tooltip=(
            "How long to fade between filters when the direction changes. Both "
            "convolution states keep running through the fade."
        ),
        mode=ADVANCED,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="headphoneCorrection",
        label="Headphone correction",
        default=HEADPHONE_OFF,
        tooltip=(
            "Applied after binaural rendering, never before. The HD650 measurement is "
            "correct only for HD650 headphones, so nothing is applied unless chosen."
        ),
        mode=ADVANCED,
        kind="choice",
        choices=CORRECTION_MODES,
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="headphoneAsset",
        label="Headphone measurement",
        default="",
        tooltip="Path to a headphone impulse response or equalization curve, when one is used.",
        mode=ADVANCED,
        kind="path",
        transition="shared",
        automatable=False,
        in_legacy_editor=False,
        live_safe=False,
    ),
    ParameterField(
        name="initial_offset",
        label="Transition start",
        default=0.0,
        unit="s",
        minimum=0.0,
        decimals=3,
        tooltip="Seconds before the transition begins, measured from the start of the step.",
        mode=ADVANCED,
        transition="timing",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="duration",
        label="Transition duration",
        default=0.0,
        unit="s",
        minimum=0.0,
        decimals=3,
        tooltip="Length of the transition itself; 0 means the remainder of the step.",
        mode=ADVANCED,
        transition="timing",
        automatable=False,
        live_safe=False,
    ),
    ParameterField(
        name="transition_curve",
        label="Transition curve",
        default="linear",
        tooltip="Shape of the parameter ramp across the transition.",
        mode=ADVANCED,
        kind="choice",
        choices=("linear", "logarithmic", "exponential"),
        transition="timing",
        automatable=False,
        live_safe=False,
    ),
)

_FIELDS_BY_NAME: dict[str, ParameterField] = {}
for _field in SAM2_FIELDS:
    for _key in (_field.name, *_field.aliases, _field.start_name, _field.end_name):
        _FIELDS_BY_NAME.setdefault(_key, _field)


def field_for(name: str) -> ParameterField | None:
    """Look up a field by its canonical name, an alias, or a `start*`/`end*` form."""

    return _FIELDS_BY_NAME.get(name)


def fields_for_mode(mode: str = BASIC, *, include_timing: bool = False) -> tuple[ParameterField, ...]:
    """Fields visible at ``mode`` and below, in declaration order."""

    if mode not in _MODE_ORDER:
        raise ValueError(f"unknown disclosure mode {mode!r}; expected one of {_MODE_ORDER}")
    limit = _MODE_ORDER.index(mode)
    return tuple(
        entry
        for entry in SAM2_FIELDS
        if _MODE_ORDER.index(entry.mode) <= limit
        and (include_timing or entry.transition != "timing")
    )


def sam2_parameter_defaults(
    is_transition: bool = False, *, include_extended: bool = False
) -> list[tuple[str, Any]]:
    """The ordered ``(name, default)`` pairs for a SAM2 voice.

    With ``include_extended=False`` this reproduces exactly the table the
    generic voice editor used before the registry existed, so opening and
    saving an existing voice does not add or drop a single key.
    """

    pairs: list[tuple[str, Any]] = []
    for entry in SAM2_FIELDS:
        if not include_extended and not entry.in_legacy_editor:
            continue
        for name in entry.names_for(is_transition):
            # Deep-copied: a caller editing the returned profile must not be
            # able to reach back into the registry's default.
            pairs.append((name, copy.deepcopy(entry.default)))
    return pairs


def legacy_editor_defaults(is_transition: bool = False) -> list[tuple[str, Any]]:
    """Alias spelling out intent at the call site in `voice_editor_dialog`."""

    return sam2_parameter_defaults(is_transition)


def transition_names(is_transition: bool = True) -> tuple[str, ...]:
    """Every serialized key a SAM2 voice uses in the given mode."""

    return tuple(
        name for entry in SAM2_FIELDS for name in entry.names_for(is_transition)
    )


def validate_sam2_params(
    params: Mapping[str, Any], *, is_transition: bool = False, sample_rate_hz: int | None = None
) -> tuple[ValidationIssue, ...]:
    """Check a voice parameter mapping against the registry.

    Returns structured issues whose ``path`` is the serialized parameter name,
    so the GUI can attach each message to the field that produced it. Unknown
    keys are reported as informational rather than as errors: they may belong
    to an older or external extension and must be preserved, not deleted.
    """

    issues: list[ValidationIssue] = []
    known = set(transition_names(is_transition)) | set(transition_names(not is_transition))
    known |= {alias for entry in SAM2_FIELDS for alias in entry.aliases}
    known |= {"samSchemaVersion", "rendererMode", "hrtfAsset", "hrtfAssetHash", "hrtfOptions"}

    for key, value in params.items():
        entry = field_for(key)
        if entry is None:
            if key not in known:
                issues.append(
                    ValidationIssue(key, "unknown SAM parameter; it will be preserved unchanged", "info")
                )
            continue

        if entry.kind == "choice":
            if not isinstance(value, str) or value.lower() not in entry.choices:
                issues.append(
                    ValidationIssue(
                        key,
                        f"must be one of {', '.join(entry.choices)}",
                    )
                )
            continue
        if entry.kind == "json":
            if value is not None and not isinstance(value, (dict, str)):
                issues.append(ValidationIssue(key, "must be a path profile object"))
            continue
        if entry.kind in ("path", "text"):
            if value is not None and not isinstance(value, str):
                issues.append(ValidationIssue(key, "must be text"))
            continue
        if entry.kind == "bool":
            if not isinstance(value, bool):
                issues.append(ValidationIssue(key, "must be true or false"))
            continue

        number = _numeric(value)
        if number is None:
            issues.append(ValidationIssue(key, f"must be a number in {entry.unit or 'the declared unit'}"))
            continue
        if entry.minimum is not None and number < entry.minimum:
            issues.append(ValidationIssue(key, f"must be at least {entry.minimum:g}"))
        if entry.maximum is not None and number > entry.maximum:
            issues.append(ValidationIssue(key, f"must be at most {entry.maximum:g}"))

    if sample_rate_hz:
        nyquist = sample_rate_hz / 2.0
        for key in ("carrierFreq", "startCarrierFreq", "endCarrierFreq"):
            carrier = _numeric(params.get(key))
            if carrier is not None and carrier >= nyquist:
                issues.append(
                    ValidationIssue(
                        key, f"is at or above Nyquist ({nyquist:.0f} Hz) for this project's sample rate"
                    )
                )
    renderer_mode = params.get("rendererMode", "abstract_pm")
    if renderer_mode not in ("abstract_pm", "geometric", "hrtf", "hybrid"):
        issues.append(ValidationIssue("rendererMode", "must be abstract_pm, geometric, hrtf, or hybrid"))
    if renderer_mode in ("hrtf", "hybrid") and not params.get("hrtfAsset"):
        issues.append(ValidationIssue("hrtfAsset", "an explicit SOFA asset is required for this renderer"))
    if renderer_mode == "hybrid":
        issues.append(ValidationIssue(
            "rendererMode", "hybrid rendering is a Phase 5 feature and is not available yet"
        ))
    options = params.get("hrtfOptions", {})
    if options is not None and not isinstance(options, dict):
        issues.append(ValidationIssue("hrtfOptions", "must be a versioned object"))
    elif isinstance(options, dict) and int(options.get("schemaVersion", 1)) != 1:
        issues.append(ValidationIssue("hrtfOptions.schemaVersion", "unsupported HRTF options schema version"))
    return tuple(issues)

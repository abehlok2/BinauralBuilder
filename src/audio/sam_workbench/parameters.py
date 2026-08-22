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

import math

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .compat import SAM2_PARAMETER_DEFAULTS
from .conventions import MAX_SAMPLE_RATE_HZ
from .dsp.source import EAR_POLARITY_CANONICAL, EAR_POLARITY_LEGACY, EAR_POLARITY_SAME
from .validation import ValidationIssue

__all__ = [
    "WORKBENCH_OWNED_KEYS",
    "BASIC",
    "ADVANCED",
    "EXPERT",
    "PATH_SHAPES",
    "PATH_TYPES",
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
    kind: str = "float"  # float | int | choice | bool | json
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


#: Keys the workbench itself owns: not SAM synthesis parameters, but not
#: unknown extension data either. Kept in one place so the validator and the
#: compatibility view cannot drift apart about what counts as "not edited here".
WORKBENCH_OWNED_KEYS = frozenset({
    "samSchemaVersion",
    # Scene structure, edited by the workbench's stage, modulation and routing
    # views. Listed here so the compatibility view does not describe keys the
    # workbench itself owns as "preserved but not edited here".
    "samStages",
    "samModulation",
    "samRouting",
    "rendererMode",
    "hrtfAsset",
    "hrtfAssetHash",
    "hrtfOptions",
    "hrtfSubject",
})


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
    known |= WORKBENCH_OWNED_KEYS

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
    # Which renderers exist, and what each one needs, comes from the registry
    # rather than from a list repeated here: a fifth copy of that list is a
    # fifth chance for it to disagree with the other four.
    from src.audio.sam_workbench.render.registry import REGISTRY

    renderer_mode = params.get("rendererMode", "abstract_pm")
    if renderer_mode not in REGISTRY:
        issues.append(
            ValidationIssue(
                "rendererMode", f"must be {', '.join(REGISTRY.identifiers)}"
            )
        )
    else:
        for requirement in REGISTRY.get(renderer_mode).assets:
            if requirement.required and not params.get(requirement.key):
                issues.append(
                    ValidationIssue(
                        requirement.key,
                        f"an explicit {requirement.kind.upper()} asset is required "
                        "for this renderer",
                    )
                )
    options = params.get("hrtfOptions", {})
    if options is not None and not isinstance(options, dict):
        issues.append(ValidationIssue("hrtfOptions", "must be a versioned object"))
    elif isinstance(options, dict):
        if int(options.get("schemaVersion", 1)) != 1:
            issues.append(ValidationIssue("hrtfOptions.schemaVersion", "unsupported HRTF options schema version"))
        issues.extend(_validate_cue_options(options))
    return tuple(issues)


def _validate_cue_options(options: Mapping[str, Any]) -> list[ValidationIssue]:
    """Check the cue transform and spatial anchor a project asks for.

    Out-of-range cue scales are reported as warnings rather than errors: the
    expert ranges are interaction limits, not physical laws, and a project that
    deliberately went past one must still open.
    """

    from .hrtf.modification import CUE_PARAMETERS, EXPERT_RANGES
    from .render.anchor import ANCHOR_PATH_MODES, ANCHOR_SOURCE_TYPES

    from .hrtf.interpolation import INTERPOLATION_MODES

    found: list[ValidationIssue] = []

    interpolation = options.get("interpolation")
    if interpolation is not None and interpolation not in INTERPOLATION_MODES:
        found.append(
            ValidationIssue(
                "hrtfOptions.interpolation",
                f"must be one of {', '.join(INTERPOLATION_MODES)}",
            )
        )

    neighbours = options.get("neighborCount")
    if neighbours is not None:
        try:
            count = int(neighbours)
        except (TypeError, ValueError):
            count = 0
        if count < 2:
            found.append(
                ValidationIssue(
                    "hrtfOptions.neighborCount",
                    "must be at least 2; a blend of one is the nearest mode",
                )
            )

    order = options.get("harmonicOrder")
    if order is not None:
        try:
            value = int(order)
        except (TypeError, ValueError):
            value = -1
        if value < 0:
            found.append(
                ValidationIssue(
                    "hrtfOptions.harmonicOrder",
                    "must not be negative; omit it to fit at the highest order "
                    "the dataset supports",
                )
            )

    cue = options.get("cue")
    if cue is not None and not isinstance(cue, dict):
        found.append(ValidationIssue("hrtfOptions.cue", "must be an object of cue scales"))
    elif isinstance(cue, dict):
        for name in CUE_PARAMETERS:
            if name not in cue:
                continue
            try:
                value = float(cue[name])
            except (TypeError, ValueError):
                found.append(ValidationIssue(f"hrtfOptions.cue.{name}", "must be a number"))
                continue
            if not math.isfinite(value):
                found.append(ValidationIssue(f"hrtfOptions.cue.{name}", "must be finite"))
                continue
            if name == "coherence" and not 0.0 <= value <= 1.0:
                # A hard bound, not a soft range: coherence outside [0, 1] has
                # no meaning, so it is an error and needs no range warning too.
                found.append(ValidationIssue(
                    "hrtfOptions.cue.coherence", "must lie between 0 and 1"
                ))
                continue
            low, high = EXPERT_RANGES[name]
            if not low <= value <= high:
                found.append(ValidationIssue(
                    f"hrtfOptions.cue.{name}",
                    f"is outside the expert range {low} to {high}",
                    "warning",
                ))

    anchor = options.get("anchor")
    if anchor is not None and not isinstance(anchor, dict):
        found.append(ValidationIssue("hrtfOptions.anchor", "must be an object"))
    elif isinstance(anchor, dict):
        source_type = anchor.get("sourceType")
        if source_type is not None and source_type not in ANCHOR_SOURCE_TYPES:
            found.append(ValidationIssue(
                "hrtfOptions.anchor.sourceType",
                f"must be one of {', '.join(ANCHOR_SOURCE_TYPES)}",
            ))
        path_mode = anchor.get("pathMode")
        if path_mode is not None and path_mode not in ANCHOR_PATH_MODES:
            found.append(ValidationIssue(
                "hrtfOptions.anchor.pathMode",
                f"must be one of {', '.join(ANCHOR_PATH_MODES)}",
            ))
        if anchor.get("enabled") and float(anchor.get("levelDb", -30.0)) > 0.0:
            found.append(ValidationIssue(
                "hrtfOptions.anchor.levelDb",
                "the anchor is meant to support the source, not overtake it",
                "warning",
            ))
    return found

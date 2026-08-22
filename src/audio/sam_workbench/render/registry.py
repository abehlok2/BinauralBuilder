"""One place that knows what a renderer is.

Which renderer modes exist, which configuration each accepts, what assets it
needs, how much latency it adds and how expensive it is were all answered in
several places at once: a tuple in the parameter validator, a weights table in
the cost estimator, a combo box in the dialog, an if/elif chain in the
compatibility adapter. Four answers to one question is three chances to
disagree, and adding a renderer meant finding all of them.

This module is the single answer. A renderer registers a
:class:`RendererDefinition` describing itself; validation, cost, latency,
assets, GUI capability metadata and the renderer factory all come from that
one object. Consumers are migrated onto it incrementally - each one that moves
deletes a copy of the list rather than adding a fifth.

Nothing here imports Qt, and nothing here imports a SOFA reader at module
scope: asking what the HRTF renderer needs must not require the ability to
read one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable, Iterable, Mapping

from ..validation import ValidationIssue

__all__ = [
    "AssetRequirement",
    "ConfigField",
    "RendererCapabilities",
    "RendererDefinition",
    "RendererRegistry",
    "REGISTRY",
    "renderer_ids",
    "renderer",
    "validate_renderer_config",
]


@dataclass(frozen=True)
class ConfigField:
    """One configuration key a renderer understands.

    Declared rather than inferred so a GUI can build a form, a validator can
    check a document, and a migration can tell a renamed key from an unknown
    one - all without the renderer being imported.
    """

    name: str
    kind: str = "float"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    label: str = ""
    #: False for a key that names an asset or a policy rather than a quantity.
    automatable: bool = False
    description: str = ""

    def validate(self, value: Any, path: str) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if self.choices and value not in self.choices:
            issues.append(
                ValidationIssue(
                    path, f"{value!r} is not one of {', '.join(map(repr, self.choices))}"
                )
            )
            return tuple(issues)
        if self.kind in ("float", "int"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return (ValidationIssue(path, f"{value!r} is not a number"),)
            if not math.isfinite(number):
                return (ValidationIssue(path, "must be finite"),)
            if self.minimum is not None and number < self.minimum:
                issues.append(ValidationIssue(path, f"must be at least {self.minimum}"))
            if self.maximum is not None and number > self.maximum:
                issues.append(ValidationIssue(path, f"must be at most {self.maximum}"))
        return tuple(issues)


@dataclass(frozen=True)
class AssetRequirement:
    """An external file a renderer cannot run without."""

    #: The configuration key naming the file.
    key: str
    kind: str = "sofa"
    required: bool = True
    #: The key carrying the expected content hash, when there is one.
    hash_key: str | None = None
    description: str = ""


@dataclass(frozen=True)
class RendererCapabilities:
    """What this renderer can honestly do, for the GUI to show.

    ``physical_elevation`` is the one worth being careful about. Every renderer
    can be *handed* an elevation; only a filter measured at that elevation
    reproduces it. A GUI that treats the two the same invites a listener to
    report height that is not there.
    """

    label: str = ""
    description: str = ""
    generates_own_source: bool = False
    consumes_trajectory: bool = False
    physical_azimuth: bool = False
    physical_elevation: bool = False
    physical_distance: bool = False
    supports_doppler: bool = False
    supports_cue_modification: bool = False
    #: Free-text note the GUI must show where the renderer's output could be
    #: mistaken for spatialization.
    honesty_note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "generatesOwnSource": self.generates_own_source,
            "consumesTrajectory": self.consumes_trajectory,
            "physicalAzimuth": self.physical_azimuth,
            "physicalElevation": self.physical_elevation,
            "physicalDistance": self.physical_distance,
            "supportsDoppler": self.supports_doppler,
            "supportsCueModification": self.supports_cue_modification,
            "honestyNote": self.honesty_note,
        }


@dataclass(frozen=True)
class RendererDefinition:
    """Everything the rest of the workbench needs to know about one renderer."""

    identifier: str
    version: int = 1
    capabilities: RendererCapabilities = field(default_factory=RendererCapabilities)
    config_fields: tuple[ConfigField, ...] = ()
    assets: tuple[AssetRequirement, ...] = ()
    #: Relative cost per rendered sample, against the HRTF renderer at 1.0.
    cost_weight: float = 1.0
    #: Where the renderer's configuration lives inside a voice's parameters.
    #: ``None`` means the voice parameters themselves.
    config_key: str | None = None
    #: Whether the compatibility adapter can render a single legacy voice in
    #: this mode today. The hybrid renderer is fully defined here - its config,
    #: assets and cost are real - but the per-voice adapter does not yet drive
    #: it, so offering it in a voice's renderer menu would offer a render that
    #: raises. A GUI should filter on this rather than on a list of its own.
    voice_renderable: bool = True

    # --- hooks. Each is a plain callable so a renderer can supply one without
    # subclassing, and the defaults are the honest "nothing to add" answers.
    _factory: Callable[..., Any] | None = field(default=None, repr=False)
    _extra_validate: Callable[[Mapping[str, Any]], Iterable[ValidationIssue]] | None = field(
        default=None, repr=False
    )
    _latency: Callable[[Mapping[str, Any], float], int] | None = field(default=None, repr=False)
    _tail: Callable[[Mapping[str, Any], float], int] | None = field(default=None, repr=False)
    _migrate: Callable[[Mapping[str, Any], int], dict[str, Any]] | None = field(
        default=None, repr=False
    )

    # --- configuration ------------------------------------------------------

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.config_fields)

    def field(self, name: str) -> ConfigField | None:
        for entry in self.config_fields:
            if entry.name == name:
                return entry
        return None

    def defaults(self) -> dict[str, Any]:
        return {
            entry.name: entry.default
            for entry in self.config_fields
            if entry.default is not None
        }

    def config_from(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Pull this renderer's configuration out of a voice's parameters."""

        if self.config_key is None:
            source: Mapping[str, Any] = params
        else:
            nested = params.get(self.config_key)
            source = nested if isinstance(nested, Mapping) else {}
        config = self.defaults()
        for name in self.field_names:
            if name in source and source[name] is not None:
                config[name] = source[name]
        return config

    def compile(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The validated, defaulted configuration this renderer will run with.

        Unknown keys are dropped from the *compiled* form but never from the
        stored document: the plan carries what the renderer uses, and the
        project keeps what the user wrote.
        """

        config = self.config_from(params)
        issues = self.validate(params)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError(
                f"renderer {self.identifier!r} configuration is invalid: "
                + "; ".join(str(issue) for issue in errors)
            )
        return config

    # --- validation ---------------------------------------------------------

    def validate(self, params: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        """Every problem with this configuration, not just the first."""

        prefix = self.config_key or ""
        config = self.config_from(params)
        issues: list[ValidationIssue] = []
        for entry in self.config_fields:
            if entry.name not in config or config[entry.name] is None:
                continue
            path = f"{prefix}.{entry.name}" if prefix else entry.name
            issues.extend(entry.validate(config[entry.name], path))

        for requirement in self.assets:
            if requirement.required and not self._asset_value(params, requirement.key):
                issues.append(
                    ValidationIssue(
                        self._asset_path(params, requirement.key),
                        f"renderer {self.identifier!r} needs an explicit "
                        f"{requirement.kind} asset",
                    )
                )

        if self._extra_validate is not None:
            issues.extend(self._extra_validate(params))
        return tuple(issues)

    def _asset_value(self, params: Mapping[str, Any], key: str) -> str:
        """Look for an asset key beside the voice's parameters and inside them.

        An asset is named at the top of a voice (``hrtfAsset``) while the
        policies for using it live in the nested options, so a lookup that only
        searched one of the two reported every configured asset as missing.
        """

        nested = params.get(self.config_key) if self.config_key else None
        for source in (params, nested if isinstance(nested, Mapping) else {}):
            value = str(source.get(key, "") or "").strip()
            if value:
                return value
        return ""

    def _asset_path(self, params: Mapping[str, Any], key: str) -> str:
        """Where to point a validation message for an asset key."""

        nested = params.get(self.config_key) if self.config_key else None
        if self.config_key and isinstance(nested, Mapping) and key in nested:
            return f"{self.config_key}.{key}"
        return key

    def required_assets(self, params: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """The assets this configuration actually references, with their hashes."""

        found: list[dict[str, Any]] = []
        for requirement in self.assets:
            path = self._asset_value(params, requirement.key)
            if not path:
                continue
            digest = (
                self._asset_value(params, requirement.hash_key)
                if requirement.hash_key
                else ""
            )
            found.append(
                {
                    "key": requirement.key,
                    "kind": requirement.kind,
                    "path": str(path),
                    "sha256": digest or None,
                }
            )
        return tuple(found)

    # --- runtime ------------------------------------------------------------

    def latency_samples(self, params: Mapping[str, Any], sample_rate_hz: float) -> int:
        return 0 if self._latency is None else int(self._latency(self.config_from(params), sample_rate_hz))

    def tail_samples(self, params: Mapping[str, Any], sample_rate_hz: float) -> int:
        """Samples that keep sounding after the source stops.

        A convolution keeps ringing for the length of its filter and a
        propagation delay line keeps a source audible after it has ended, so a
        render window that stops exactly at the source's end truncates both.
        """

        return 0 if self._tail is None else int(self._tail(self.config_from(params), sample_rate_hz))

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if self._factory is None:
            raise NotImplementedError(
                f"renderer {self.identifier!r} has no factory registered"
            )
        return self._factory(*args, **kwargs)

    # --- serialization ------------------------------------------------------

    def migrate(self, params: Mapping[str, Any], from_version: int) -> dict[str, Any]:
        """Bring a stored configuration up to this renderer's current version."""

        result = dict(params)
        if from_version > self.version:
            raise ValueError(
                f"renderer {self.identifier!r} configuration version {from_version} "
                f"was written by a newer build; this one understands up to {self.version}"
            )
        if self._migrate is not None and from_version < self.version:
            result = dict(self._migrate(result, from_version))
        return result

    def describe(self) -> dict[str, Any]:
        """The GUI's view: what this is, what it needs, what it can claim."""

        return {
            "id": self.identifier,
            "version": self.version,
            "costWeight": float(self.cost_weight),
            "configKey": self.config_key,
            "voiceRenderable": bool(self.voice_renderable),
            "capabilities": self.capabilities.describe(),
            "config": [
                {
                    "name": entry.name,
                    "kind": entry.kind,
                    "default": entry.default,
                    "minimum": entry.minimum,
                    "maximum": entry.maximum,
                    "choices": list(entry.choices),
                    "label": entry.label or entry.name,
                    "automatable": entry.automatable,
                    "description": entry.description,
                }
                for entry in self.config_fields
            ],
            "assets": [
                {
                    "key": requirement.key,
                    "kind": requirement.kind,
                    "required": requirement.required,
                    "hashKey": requirement.hash_key,
                    "description": requirement.description,
                }
                for requirement in self.assets
            ],
        }


class RendererRegistry:
    """The set of renderers this build can run."""

    def __init__(self) -> None:
        self._entries: dict[str, RendererDefinition] = {}

    def register(self, definition: RendererDefinition) -> RendererDefinition:
        if definition.identifier in self._entries:
            raise ValueError(f"renderer {definition.identifier!r} is already registered")
        self._entries[definition.identifier] = definition
        return definition

    def __contains__(self, identifier: object) -> bool:
        return str(identifier) in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, identifier: str) -> RendererDefinition:
        try:
            return self._entries[str(identifier)]
        except KeyError:
            raise KeyError(
                f"rendererMode {identifier!r} is not available in this build; "
                f"expected one of {', '.join(self.identifiers)}"
            ) from None

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Registration order, which is the order the GUI should offer them."""

        return tuple(self._entries)

    @property
    def voice_renderable(self) -> tuple[RendererDefinition, ...]:
        """The renderers a single legacy voice can actually be rendered with."""

        return tuple(entry for entry in self._entries.values() if entry.voice_renderable)

    def describe(self) -> list[dict[str, Any]]:
        return [definition.describe() for definition in self._entries.values()]


REGISTRY = RendererRegistry()


# --- the four renderers -----------------------------------------------------


def _hrtf_latency(config: Mapping[str, Any], sample_rate_hz: float) -> int:
    # The streaming HRTF renderer convolves in place and reports no latency; a
    # baked delay is part of the filter rather than added on top of it.
    return 0


def _hrtf_tail(config: Mapping[str, Any], sample_rate_hz: float) -> int:
    """Filter ring-out plus the longest propagation delay the path can reach."""

    # A conservative HRIR length, since the real one is only known once the
    # asset is opened and this must answer without opening it.
    tail = 512
    if config.get("propagationDelay"):
        maximum = float(config.get("maximumDistanceM", 100.0) or 100.0)
        tail += int(math.ceil(maximum / 343.0 * float(sample_rate_hz)))
    return tail


def _geometric_tail(config: Mapping[str, Any], sample_rate_hz: float) -> int:
    maximum = float(config.get("maximumDistanceM", 100.0) or 100.0)
    return int(math.ceil(maximum / 343.0 * float(sample_rate_hz))) + 4


def _hrtf_factory(*args: Any, **kwargs: Any):
    from .hrtf import HRTFRenderer

    return HRTFRenderer(*args, **kwargs)


def _abstract_factory(*args: Any, **kwargs: Any):
    from .abstract_pm import AbstractPMRenderer

    return AbstractPMRenderer(*args, **kwargs)


def _geometric_factory(*args: Any, **kwargs: Any):
    from .geometric import GeometricBinauralRenderer

    return GeometricBinauralRenderer(*args, **kwargs)


def _hybrid_factory(*args: Any, **kwargs: Any):
    from .hybrid import HybridSpec

    return HybridSpec(*args, **kwargs)


_DISTANCE_LAWS = ("none", "inverse", "inverse_square")

_ABSTRACT_PM = REGISTRY.register(
    RendererDefinition(
        identifier="abstract_pm",
        version=1,
        cost_weight=0.02,
        capabilities=RendererCapabilities(
            label="Abstract phase modulation",
            description="The SAM equations, nothing else.",
            generates_own_source=True,
            consumes_trajectory=True,
            honesty_note=(
                "Not a spatializer. A trajectory may drive it as a control source, "
                "but the result is a creative mapping: phase modulation alone "
                "cannot produce reliable direction, distance or height."
            ),
        ),
        config_fields=(
            ConfigField("carrierFreq", "float", 440.0, minimum=0.0, automatable=True, label="Carrier"),
            ConfigField("modFreq", "float", 4.0, minimum=0.0, automatable=True, label="Modulation rate"),
            ConfigField("amp", "float", 0.7, minimum=0.0, maximum=1.0, automatable=True, label="Amplitude"),
            ConfigField("arcWidthDeg", "float", 90.0, automatable=True, label="Arc width"),
            ConfigField("directionOffsetDeg", "float", 0.0, automatable=True, label="Direction offset"),
        ),
        _factory=_abstract_factory,
    )
)

_GEOMETRIC = REGISTRY.register(
    RendererDefinition(
        identifier="geometric",
        version=1,
        cost_weight=0.08,
        capabilities=RendererCapabilities(
            label="Geometric binaural",
            description="Interaural delay and level from a head model.",
            consumes_trajectory=True,
            physical_azimuth=True,
            physical_distance=True,
            supports_doppler=True,
            honesty_note=(
                "Elevation can colour the sound but does not localize it; "
                "convincing height needs HRTF filtering."
            ),
        ),
        config_fields=(
            ConfigField("distanceLaw", "choice", "inverse", choices=_DISTANCE_LAWS, label="Distance law"),
            ConfigField("referenceDistanceM", "float", 1.0, minimum=1e-6, label="Reference distance"),
            ConfigField("minimumDistanceM", "float", 0.05, minimum=1e-6, label="Minimum distance"),
            ConfigField("maximumDistanceM", "float", 100.0, minimum=1e-6, label="Maximum distance"),
            ConfigField("dopplerEnabled", "bool", True, label="Doppler"),
        ),
        _tail=_geometric_tail,
        _factory=_geometric_factory,
    )
)


def _hrtf_extra_validation(params: Mapping[str, Any]) -> Iterable[ValidationIssue]:
    options = params.get("hrtfOptions")
    if isinstance(options, Mapping) and int(options.get("schemaVersion", 1)) != 1:
        yield ValidationIssue("hrtfOptions.schemaVersion", "unsupported hrtfOptions schemaVersion")


_HRTF_FIELDS = (
    ConfigField(
        "interpolation", "choice", "nearest",
        choices=("nearest", "logmag_delay"), label="Interpolation",
        description="Nearest steps between measured directions; logmag_delay interpolates across them.",
    ),
    ConfigField(
        "delayPolicy", "choice", "bake_delay_into_ir",
        choices=("bake_delay_into_ir", "keep_external_delay"), label="Delay policy",
    ),
    ConfigField("crossfadeMs", "float", 10.0, minimum=0.0, label="Filter crossfade"),
    ConfigField("controlIntervalSamples", "int", 128, minimum=1, label="Control interval"),
    ConfigField("distanceLaw", "choice", "inverse", choices=_DISTANCE_LAWS, label="Distance law"),
    ConfigField("referenceDistanceM", "float", 1.0, minimum=1e-6, label="Reference distance"),
    ConfigField("minimumDistanceM", "float", 0.15, minimum=1e-6, label="Minimum distance"),
    ConfigField("maximumDistanceM", "float", 100.0, minimum=1e-6, label="Maximum distance"),
    ConfigField("propagationDelay", "bool", True, label="Propagation delay"),
    ConfigField("elevationDeg", "float", 0.0, minimum=-90.0, maximum=90.0, label="Fixed elevation"),
    ConfigField("distanceM", "float", 1.0, minimum=1e-6, label="Fixed distance"),
)

_HRTF = REGISTRY.register(
    RendererDefinition(
        identifier="hrtf",
        version=1,
        cost_weight=1.0,
        config_key="hrtfOptions",
        capabilities=RendererCapabilities(
            label="HRTF",
            description="Convolution with an explicit SOFA asset.",
            consumes_trajectory=True,
            physical_azimuth=True,
            physical_elevation=True,
            physical_distance=True,
            supports_doppler=True,
            honesty_note=(
                "Height is only reproduced where the dataset measured it; "
                "check the coverage report before trusting an overhead path."
            ),
        ),
        config_fields=_HRTF_FIELDS,
        assets=(
            AssetRequirement(
                key="hrtfAsset", kind="sofa", required=True, hash_key="hrtfAssetHash",
                description="The SOFA dataset this render convolves with.",
            ),
        ),
        _extra_validate=_hrtf_extra_validation,
        _latency=_hrtf_latency,
        _tail=_hrtf_tail,
        _factory=_hrtf_factory,
    )
)

_HYBRID = REGISTRY.register(
    RendererDefinition(
        identifier="hybrid",
        version=1,
        cost_weight=1.15,
        config_key="hrtfOptions",
        voice_renderable=False,
        capabilities=RendererCapabilities(
            label="Hybrid",
            description="HRTF for spectrum, with declared extra cues.",
            consumes_trajectory=True,
            physical_azimuth=True,
            physical_elevation=True,
            physical_distance=True,
            supports_doppler=True,
            supports_cue_modification=True,
            honesty_note=(
                "Source -> SAM -> 3D trajectory -> HRTF interpolation -> "
                "Cue modification -> Output. Anything past the HRTF stage is a "
                "declared departure from the measured cues, not a measurement."
            ),
        ),
        config_fields=_HRTF_FIELDS
        + (
            ConfigField("neighborCount", "int", 3, minimum=1, label="Neighbours"),
            ConfigField("outputGainDb", "float", 0.0, automatable=True, label="Output gain"),
        ),
        assets=(
            AssetRequirement(
                key="hrtfAsset", kind="sofa", required=True, hash_key="hrtfAssetHash",
                description="The SOFA dataset this render convolves with.",
            ),
            AssetRequirement(
                key="headphoneAsset", kind="headphone", required=False,
                description="Optional headphone correction applied after binaural rendering.",
            ),
        ),
        _extra_validate=_hrtf_extra_validation,
        _latency=_hrtf_latency,
        _tail=_hrtf_tail,
        _factory=_hybrid_factory,
    )
)


# --- convenience ------------------------------------------------------------


def renderer_ids() -> tuple[str, ...]:
    """Every renderer mode this build can run, in the order to offer them."""

    return REGISTRY.identifiers


def renderer(identifier: str) -> RendererDefinition:
    return REGISTRY.get(identifier)


def validate_renderer_config(params: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    """Validate a voice's renderer mode and that renderer's configuration."""

    mode = str(params.get("rendererMode", "abstract_pm"))
    if mode not in REGISTRY:
        return (
            ValidationIssue(
                "rendererMode",
                f"must be one of {', '.join(REGISTRY.identifiers)}",
            ),
        )
    return REGISTRY.get(mode).validate(params)

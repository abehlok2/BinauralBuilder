"""The registry is the one answer to "what is a renderer".

Which modes exist, what configuration each takes, which assets it needs, what
it costs and what it can honestly claim used to be answered in four places at
once. These tests assert that the copies are gone: the parameter validator, the
cost estimator, the compatibility dispatch and the renderer menu must all agree
with the registry, because they now read it rather than repeat it.
"""

from __future__ import annotations

import pytest

from src.audio.sam_workbench.cost import CostInputs, estimate_cost
from src.audio.sam_workbench.parameters import validate_sam2_params
from src.audio.sam_workbench.render.registry import (
    REGISTRY,
    ConfigField,
    RendererDefinition,
    RendererRegistry,
    renderer,
    renderer_ids,
    validate_renderer_config,
)

EXPECTED = ("abstract_pm", "geometric", "hrtf", "hybrid")
VALID_HRTF = {
    "rendererMode": "hrtf",
    "hrtfAsset": "asset.sofa",
    "hrtfAssetHash": "abc123",
    "hrtfOptions": {"interpolation": "logmag_delay", "crossfadeMs": 12.0},
}


# --- contents ---------------------------------------------------------------


def test_the_four_renderers_are_registered():
    assert renderer_ids() == EXPECTED


@pytest.mark.parametrize("identifier", EXPECTED)
def test_every_renderer_declares_what_it_is(identifier):
    definition = renderer(identifier)
    assert definition.capabilities.label
    assert definition.capabilities.description
    assert definition.version >= 1
    assert definition.cost_weight > 0.0


@pytest.mark.parametrize("identifier", EXPECTED)
def test_every_renderer_serializes_its_whole_definition(identifier):
    described = renderer(identifier).describe()
    assert set(described) >= {
        "id", "version", "costWeight", "capabilities", "config", "assets", "voiceRenderable",
    }


def test_an_unknown_renderer_is_refused_with_the_list_of_real_ones():
    with pytest.raises(KeyError, match="abstract_pm"):
        renderer("nonsense")


def test_registering_the_same_identifier_twice_is_refused():
    registry = RendererRegistry()
    registry.register(RendererDefinition(identifier="one"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(RendererDefinition(identifier="one"))


# --- honesty metadata -------------------------------------------------------


def test_only_the_hrtf_renderers_claim_physical_elevation():
    """Every renderer can be handed an elevation; two can reproduce one."""

    claiming = {entry.identifier for entry in REGISTRY if entry.capabilities.physical_elevation}
    assert claiming == {"hrtf", "hybrid"}


def test_the_abstract_renderer_says_it_is_not_a_spatializer():
    note = renderer("abstract_pm").capabilities.honesty_note
    assert "not a spatializer" in note.lower()


def test_the_hybrid_renderer_states_its_stage_order():
    assert "Cue modification" in renderer("hybrid").capabilities.honesty_note


def test_the_hybrid_renderer_is_defined_but_not_offered_for_a_single_voice():
    """Its config and cost are real; the per-voice adapter cannot drive it yet."""

    assert renderer("hybrid").voice_renderable is False
    assert [entry.identifier for entry in REGISTRY.voice_renderable] == [
        "abstract_pm", "geometric", "hrtf",
    ]


# --- configuration ----------------------------------------------------------


def test_a_valid_configuration_produces_no_issues():
    assert renderer("hrtf").validate(VALID_HRTF) == ()


def test_a_missing_required_asset_is_reported_against_its_own_key():
    issues = renderer("hrtf").validate({"rendererMode": "hrtf"})
    assert [issue.path for issue in issues] == ["hrtfAsset"]


def test_a_value_outside_a_choice_list_is_reported():
    params = {"rendererMode": "hrtf", "hrtfAsset": "a.sofa", "hrtfOptions": {"interpolation": "bogus"}}
    assert any("is not one of" in issue.message for issue in renderer("hrtf").validate(params))


def test_a_value_outside_its_range_is_reported():
    params = {"rendererMode": "hrtf", "hrtfAsset": "a.sofa", "hrtfOptions": {"crossfadeMs": -3.0}}
    assert any("at least" in issue.message for issue in renderer("hrtf").validate(params))


def test_compiling_fills_in_the_defaults():
    config = renderer("hrtf").compile(VALID_HRTF)
    assert config["interpolation"] == "logmag_delay"   # what was asked for
    assert config["delayPolicy"] == "bake_delay_into_ir"  # what was not


def test_compiling_an_invalid_configuration_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="hrtfAsset"):
        renderer("hrtf").compile({"rendererMode": "hrtf"})


def test_an_asset_is_found_whether_it_sits_beside_or_inside_the_options():
    definition = renderer("hrtf")
    beside = definition.required_assets(VALID_HRTF)
    inside = definition.required_assets(
        {"rendererMode": "hrtf", "hrtfOptions": {"hrtfAsset": "asset.sofa"}}
    )
    assert beside[0]["path"] == inside[0]["path"] == "asset.sofa"
    assert beside[0]["sha256"] == "abc123"


def test_an_optional_asset_is_not_required():
    assert renderer("hybrid").validate(
        {"rendererMode": "hybrid", "hrtfAsset": "a.sofa"}
    ) == ()


# --- latency and tail -------------------------------------------------------


def test_a_renderer_with_no_tail_reports_none():
    assert renderer("abstract_pm").tail_samples({}, 44100) == 0


def test_the_hrtf_tail_covers_the_filter_and_the_propagation_delay():
    definition = renderer("hrtf")
    without = definition.tail_samples(
        {"hrtfOptions": {"propagationDelay": False}}, 44100
    )
    with_delay = definition.tail_samples(
        {"hrtfOptions": {"propagationDelay": True, "maximumDistanceM": 100.0}}, 44100
    )
    assert without > 0
    assert with_delay > without + 44100 // 4


def test_the_geometric_tail_scales_with_the_maximum_distance():
    definition = renderer("geometric")
    near = definition.tail_samples({"maximumDistanceM": 10.0}, 44100)
    far = definition.tail_samples({"maximumDistanceM": 100.0}, 44100)
    assert far > near


# --- migration --------------------------------------------------------------


def test_a_configuration_from_a_newer_renderer_version_is_refused():
    with pytest.raises(ValueError, match="newer build"):
        renderer("hrtf").migrate({}, from_version=99)


def test_migrating_from_the_current_version_changes_nothing():
    assert renderer("hrtf").migrate({"a": 1}, from_version=renderer("hrtf").version) == {"a": 1}


# --- the copies are gone ----------------------------------------------------


def test_the_parameter_validator_accepts_exactly_the_registered_modes():
    for identifier in EXPECTED:
        params = {"rendererMode": identifier, "hrtfAsset": "a.sofa"}
        assert not [
            issue for issue in validate_sam2_params(params) if issue.path == "rendererMode"
        ]
    assert [
        issue for issue in validate_sam2_params({"rendererMode": "nope"})
        if issue.path == "rendererMode"
    ]


def test_the_parameter_validator_requires_assets_the_registry_declares():
    issues = validate_sam2_params({"rendererMode": "hrtf"})
    assert any(issue.path == "hrtfAsset" for issue in issues)


def test_the_cost_estimator_uses_the_registry_weights():
    """A cheaper renderer must cost less; the ordering comes from one table."""

    costs = {
        identifier: estimate_cost(CostInputs(renderer=identifier)).macs_per_second
        for identifier in EXPECTED
    }
    assert costs["abstract_pm"] < costs["geometric"] < costs["hrtf"] <= costs["hybrid"]


def test_an_unregistered_renderer_costs_the_neutral_weight():
    assert estimate_cost(CostInputs(renderer="nope")).macs_per_second > 0.0


def test_validate_renderer_config_routes_to_the_named_renderer():
    assert validate_renderer_config({"rendererMode": "abstract_pm"}) == ()
    assert validate_renderer_config(VALID_HRTF) == ()
    assert any(
        issue.path == "rendererMode"
        for issue in validate_renderer_config({"rendererMode": "nope"})
    )


def test_the_compatibility_adapter_refuses_a_mode_it_cannot_drive():
    from src.audio.sam_workbench.compat import render_sam2_voice

    with pytest.raises(ValueError, match="not available in this build"):
        render_sam2_voice(0.01, 44100, params={"rendererMode": "hybrid"})


def test_a_config_field_validates_its_own_kind():
    field = ConfigField("x", "float", 1.0, minimum=0.0, maximum=2.0)
    assert field.validate(1.0, "x") == ()
    assert field.validate("not a number", "x")
    assert field.validate(-1.0, "x")
    assert field.validate(3.0, "x")

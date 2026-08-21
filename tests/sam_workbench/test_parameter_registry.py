"""The SAM parameter registry: one definition per field, and no drift."""

from __future__ import annotations

import collections

import pytest

from src.audio.sam_workbench.compat import SAM2_PARAMETER_DEFAULTS
from src.audio.sam_workbench.parameters import (
    ADVANCED,
    BASIC,
    EXPERT,
    PATH_SHAPES,
    SAM2_FIELDS,
    field_for,
    fields_for_mode,
    sam2_parameter_defaults,
    transition_names,
    validate_sam2_params,
)

#: The tables that used to live - twice - inside `voice_editor_dialog`.
LEGACY_STANDARD = [
    ("amp", 0.7),
    ("carrierFreq", 440.0),
    ("modFreq", 4.0),
    ("arcWidthDeg", 90.0),
    ("directionOffsetDeg", 0.0),
    ("spatialScale", 1.0),
    ("pathType", "open"),
    ("pathShape", "sinusoidal"),
    ("customPathSmoothingPasses", 1),
    ("customPathSmoothingRatio", 0.25),
    ("customPathProfile", {"kind": "linear", "points": [], "smoothingPasses": 1, "smoothingRatio": 0.25}),
]

LEGACY_TRANSITION = [
    ("amp", 0.7),
    ("startCarrierFreq", 440.0),
    ("endCarrierFreq", 440.0),
    ("startModFreq", 4.0),
    ("endModFreq", 4.0),
    ("startArcWidthDeg", 90.0),
    ("endArcWidthDeg", 90.0),
    ("startDirectionOffsetDeg", 0.0),
    ("endDirectionOffsetDeg", 0.0),
    ("startSpatialScale", 1.0),
    ("endSpatialScale", 1.0),
    ("pathType", "open"),
    ("pathShape", "sinusoidal"),
    ("customPathSmoothingPasses", 1),
    ("customPathSmoothingRatio", 0.25),
    ("customPathProfile", {"kind": "linear", "points": [], "smoothingPasses": 1, "smoothingRatio": 0.25}),
    ("initial_offset", 0.0),
    ("duration", 0.0),
    ("transition_curve", "linear"),
]


def test_every_field_is_declared_exactly_once():
    names = [entry.name for entry in SAM2_FIELDS]
    duplicates = [name for name, count in collections.Counter(names).items() if count > 1]
    assert duplicates == []


def test_registry_reproduces_the_legacy_editor_tables():
    """Existing voices must keep exactly the keys and defaults they had."""

    assert sam2_parameter_defaults(False) == LEGACY_STANDARD
    assert sam2_parameter_defaults(True) == LEGACY_TRANSITION


def test_defaults_agree_with_the_compatibility_adapter():
    for name in ("amp", "carrierFreq", "modFreq", "arcWidthDeg", "directionOffsetDeg", "spatialScale"):
        assert field_for(name).default == SAM2_PARAMETER_DEFAULTS[name]


def test_extended_fields_are_opt_in():
    legacy_names = {name for name, _ in sam2_parameter_defaults(False)}
    extended_names = {name for name, _ in sam2_parameter_defaults(False, include_extended=True)}
    assert "rotationDirection" not in legacy_names
    assert {"rotationDirection", "discontinuousSteps", "earPolarity"} <= extended_names


def test_default_profile_is_copied_not_shared():
    first = dict(sam2_parameter_defaults(False))["customPathProfile"]
    second = dict(sam2_parameter_defaults(False))["customPathProfile"]
    assert first == second
    first["points"].append([1, 2])
    assert second["points"] == []


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("beatFreq", "modFreq"),
        ("arcWidth", "arcWidthDeg"),
        ("directionOffset", "directionOffsetDeg"),
        ("peakPhaseDev", "spatialScale"),
        ("phaseOffsetL", "phaseOffsetLRad"),
        ("startCarrierFreq", "carrierFreq"),
        ("endModFreq", "modFreq"),
    ],
)
def test_aliases_and_transition_forms_resolve_to_their_field(alias, canonical):
    assert field_for(alias) is field_for(canonical)


def test_unknown_names_have_no_field():
    assert field_for("definitelyNotASamParameter") is None


def test_every_field_declares_its_metadata():
    for entry in SAM2_FIELDS:
        assert entry.label
        assert entry.tooltip, f"{entry.name} has no tooltip"
        assert entry.mode in (BASIC, ADVANCED, EXPERT)
        assert entry.transition in ("pair", "shared", "timing")
        if entry.kind in ("float", "int"):
            assert entry.minimum is not None or entry.maximum is not None, entry.name
        if entry.kind == "choice":
            assert entry.choices
            assert entry.default in entry.choices


def test_disclosure_modes_are_nested():
    basic = {entry.name for entry in fields_for_mode(BASIC)}
    advanced = {entry.name for entry in fields_for_mode(ADVANCED)}
    expert = {entry.name for entry in fields_for_mode(EXPERT)}
    assert basic < advanced < expert
    assert all(entry.mode == BASIC for entry in fields_for_mode(BASIC))


def test_timing_fields_appear_only_for_transitions():
    assert "initial_offset" in transition_names(True)
    assert "initial_offset" not in transition_names(False)
    assert all(entry.transition != "timing" for entry in fields_for_mode(EXPERT))
    assert any(entry.transition == "timing" for entry in fields_for_mode(EXPERT, include_timing=True))


def test_path_shapes_include_every_legacy_alias():
    assert set(PATH_SHAPES) == {"sinusoidal", "triangle", "ramp", "saw", "square"}


# --- validation -------------------------------------------------------------


def test_valid_parameters_produce_no_issues():
    assert validate_sam2_params(dict(LEGACY_STANDARD)) == ()


def test_out_of_range_values_are_reported_against_their_field():
    issues = validate_sam2_params({"amp": 5.0, "carrierFreq": -1.0})
    paths = {issue.path: issue for issue in issues}
    assert set(paths) == {"amp", "carrierFreq"}
    assert "at most" in paths["amp"].message
    assert "at least" in paths["carrierFreq"].message
    assert all(issue.severity == "error" for issue in issues)


def test_bad_choices_are_reported():
    issues = validate_sam2_params({"pathType": "spiral"})
    assert [issue.path for issue in issues] == ["pathType"]
    assert "open" in issues[0].message


def test_unknown_parameters_are_informational_not_errors():
    issues = validate_sam2_params({"someVendorField": 3})
    assert [issue.severity for issue in issues] == ["info"]
    assert "preserved" in issues[0].message


def test_carrier_above_nyquist_is_reported_for_the_project_rate():
    issues = validate_sam2_params({"carrierFreq": 30_000.0}, sample_rate_hz=44_100)
    assert any("Nyquist" in issue.message for issue in issues)
    assert validate_sam2_params({"carrierFreq": 440.0}, sample_rate_hz=44_100) == ()


def test_transition_pairs_are_validated_individually():
    issues = validate_sam2_params(
        {"startCarrierFreq": 440.0, "endCarrierFreq": -3.0}, is_transition=True
    )
    assert [issue.path for issue in issues] == ["endCarrierFreq"]

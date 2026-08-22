"""Track-level scene state: identity, migration, validation, invariance.

The rules this file protects:

* a source's identifier is assigned to the *real* track and outlives editing -
  reordering, renaming and adding voices must not reassign it, because every
  automation route and every mute refers to a source by that identifier;
* a scene document survives a load and a save with the parts this build does
  not understand still in it;
* references that lead nowhere are reported rather than silently doing nothing;
* evaluating the scene is a pure function of absolute time, so where a caller
  cuts the timeline into blocks cannot change the result.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.audio.sam_workbench.modulation import ModulationMatrix, ModulationRoute
from src.audio.sam_workbench.scene_state import (
    AMPLITUDE_PATHS,
    SAM_SCENE_SCHEMA_VERSION,
    SOURCE_ID_KEY,
    assign_source_ids,
    empty_sam_scene,
    migrate_scene,
    migrate_voice_scene,
    normalize_sam_scene,
    scene_gain_envelope,
    scene_parameter_overrides,
    scene_parameter_series,
    validate_scene,
)
from src.audio.sam_workbench.stages import ParameterBinding, StageConfig, Timeline

VERSION_2_SECTIONS = ("assets", "environment", "experiment", "extensions")


def _voice(name, source_id=None):
    voice = {
        "synth_function_name": "spatial_angle_modulation_sam2",
        "description": name,
        "params": {},
    }
    if source_id:
        voice[SOURCE_ID_KEY] = source_id
    return voice


def _track(*names):
    return {"steps": [{"name": "Step", "voices": [_voice(name) for name in names]}]}


def _voices(track):
    return track["steps"][0]["voices"]


def _identity(track):
    return {voice["description"]: voice[SOURCE_ID_KEY] for voice in _voices(track)}


# --- identity ---------------------------------------------------------------


def test_identifiers_are_written_to_the_real_track():
    """Identifiers assigned to a render copy are discarded with it."""

    track = _track("Left", "Right")
    assign_source_ids(track)
    assert [voice[SOURCE_ID_KEY] for voice in _voices(track)] == ["source.1", "source.2"]


def test_identity_survives_reordering_the_voices():
    track = _track("Left", "Right")
    assign_source_ids(track)
    before = _identity(track)

    _voices(track).reverse()
    assign_source_ids(track)
    assert _identity(track) == before


def test_identity_survives_reordering_the_steps():
    track = {
        "steps": [
            {"name": "One", "voices": [_voice("Left")]},
            {"name": "Two", "voices": [_voice("Right")]},
        ]
    }
    assign_source_ids(track)
    before = {
        voice["description"]: voice[SOURCE_ID_KEY]
        for step in track["steps"]
        for voice in step["voices"]
    }

    track["steps"].reverse()
    assign_source_ids(track)
    after = {
        voice["description"]: voice[SOURCE_ID_KEY]
        for step in track["steps"]
        for voice in step["voices"]
    }
    assert after == before


def test_renaming_a_voice_keeps_its_identifier_and_updates_its_name():
    """A name is for reading; an identifier is for referring. They differ."""

    track = _track("Left", "Right")
    assign_source_ids(track)
    identifier = _voices(track)[0][SOURCE_ID_KEY]

    _voices(track)[0]["description"] = "Renamed"
    scene = assign_source_ids(track, persist_scene=True)
    assert _voices(track)[0][SOURCE_ID_KEY] == identifier
    assert scene["sources"][0]["name"] == "Renamed"


def test_adding_a_voice_does_not_disturb_the_existing_identifiers():
    track = _track("Left", "Right")
    assign_source_ids(track)
    before = _identity(track)

    _voices(track).append(_voice("Third"))
    assign_source_ids(track)
    after = _identity(track)
    assert {name: after[name] for name in before} == before
    assert after["Third"] not in before.values()


def test_a_removed_source_keeps_its_scene_record_rather_than_losing_its_routing():
    track = _track("Left", "Right")
    scene = assign_source_ids(track, persist_scene=True)
    removed = _voices(track).pop(1)[SOURCE_ID_KEY]

    scene = assign_source_ids(track, scene)
    orphans = [entry for entry in scene["sources"] if entry.get("orphaned")]
    assert [entry["id"] for entry in orphans] == [removed]


def test_scene_sources_are_given_readable_names():
    track = _track("Left drone", "Right drone")
    scene = assign_source_ids(track, persist_scene=True)
    assert [entry["name"] for entry in scene["sources"]] == ["Left drone", "Right drone"]


def test_a_track_without_a_scene_does_not_acquire_one():
    """Identifiers are always useful; a scene section is not always wanted."""

    track = _track("Left")
    assign_source_ids(track)
    assert "sam_scene" not in track
    assert _voices(track)[0][SOURCE_ID_KEY]


def test_a_track_with_a_scene_keeps_it_maintained():
    track = _track("Left")
    track["sam_scene"] = empty_sam_scene()
    assign_source_ids(track)
    assert [entry["id"] for entry in track["sam_scene"]["sources"]] == ["source.1"]


# --- schema and migration ---------------------------------------------------


def test_a_blank_scene_carries_every_section():
    scene = empty_sam_scene()
    assert scene["schemaVersion"] == SAM_SCENE_SCHEMA_VERSION
    for section in VERSION_2_SECTIONS:
        assert section in scene


def test_a_version_one_document_migrates_to_the_current_version():
    legacy = {
        "schemaVersion": 1,
        "sources": [{"id": "source.1", "name": "Left"}],
        "stages": Timeline().describe(),
        "modulators": [],
        "modulation": ModulationMatrix().describe(),
        "buses": [],
        "routing": {"schemaVersion": 1, "buses": [], "sources": [], "bands": {}},
    }
    migrated = migrate_scene(legacy)
    assert migrated["schemaVersion"] == SAM_SCENE_SCHEMA_VERSION
    for section in VERSION_2_SECTIONS:
        assert section in migrated
    assert migrated["sources"] == legacy["sources"]


def test_migration_leaves_existing_values_alone():
    legacy = {"schemaVersion": 1, "environment": {"speedOfSoundMS": 340.0}}
    assert migrate_scene(legacy)["environment"] == {"speedOfSoundMS": 340.0}


def test_a_document_from_a_newer_build_is_refused_rather_than_misread():
    with pytest.raises(ValueError, match="newer build"):
        normalize_sam_scene({"schemaVersion": SAM_SCENE_SCHEMA_VERSION + 1})


def test_unknown_fields_survive_a_load():
    """A newer build's data must not be deleted by an older one opening it."""

    scene = normalize_sam_scene(
        {
            "schemaVersion": 1,
            "somethingNobodyKnows": {"deep": [1, 2, 3]},
            "sources": [{"id": "source.1", "name": "Left", "futureKey": 42}],
        }
    )
    assert scene["somethingNobodyKnows"] == {"deep": [1, 2, 3]}
    assert scene["sources"][0]["futureKey"] == 42


def test_normalizing_is_idempotent():
    once = normalize_sam_scene({"schemaVersion": 1, "extra": True})
    assert normalize_sam_scene(once) == once


def test_normalizing_does_not_mutate_its_argument():
    original = {"schemaVersion": 1, "sources": []}
    snapshot = copy.deepcopy(original)
    normalize_sam_scene(original)
    assert original == snapshot


def test_legacy_voice_local_scene_keys_still_migrate_to_the_track():
    params = {"samStages": Timeline().describe(), "amp": 0.7}
    scene = migrate_voice_scene(params)
    assert scene is not None
    assert scene["schemaVersion"] == SAM_SCENE_SCHEMA_VERSION
    assert "samStages" not in params
    assert params["amp"] == 0.7


# --- validation -------------------------------------------------------------


def test_duplicate_source_identifiers_are_an_error():
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1"}, {"id": "source.1"}]
    issues = validate_scene(scene)
    assert any("duplicate source identifier" in issue.message for issue in issues)
    assert all(issue.severity == "error" for issue in issues if "duplicate" in issue.message)


def test_a_route_pointing_at_no_source_is_an_error():
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1"}]
    scene["modulators"] = [{"id": "lfo"}]
    scene["modulation"] = ModulationMatrix(
        routes=(
            ModulationRoute(
                modulator_id="lfo", target_id="source.9", parameter_path="modFreq", depth=1.0
            ),
        )
    ).describe()
    assert any("not a source in this scene" in issue.message for issue in validate_scene(scene))


def test_routing_to_a_bus_that_does_not_exist_is_an_error():
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1"}]
    scene["routing"]["sources"] = [{"sourceId": "source.1", "busId": "nowhere"}]
    assert any("not a bus in this scene" in issue.message for issue in validate_scene(scene))


def test_the_wildcard_targets_are_not_treated_as_dangling():
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1"}]
    scene["stages"] = Timeline(
        stages=(
            StageConfig(
                id="stage",
                duration_s=1.0,
                parameter_overrides=(
                    ParameterBinding(target_id="voice", parameter_path="amp", value=0.5),
                ),
            ),
        )
    ).describe()
    assert not [issue for issue in validate_scene(scene) if "not a source" in issue.message]


def test_a_route_with_no_modulator_definition_is_warned_about():
    """The undocumented fallback is visible rather than indefinite."""

    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1"}]
    scene["modulation"] = ModulationMatrix(
        routes=(
            ModulationRoute(
                modulator_id="ghost", target_id="source.1", parameter_path="modFreq", depth=1.0
            ),
        )
    ).describe()
    warnings = [issue for issue in validate_scene(scene) if "no modulator definition" in issue.message]
    assert len(warnings) == 1
    assert warnings[0].severity == "warning"


def test_a_track_voice_the_scene_does_not_declare_is_reported():
    scene = empty_sam_scene()
    track = _track("Left")
    _voices(track)[0][SOURCE_ID_KEY] = "source.7"
    issues = validate_scene(scene, track["steps"])
    assert any("does not declare" in issue.message for issue in issues)


def test_a_consistent_scene_validates_clean():
    track = _track("Left", "Right")
    track["sam_scene"] = empty_sam_scene()
    scene = assign_source_ids(track)
    assert validate_scene(scene, track["steps"]) == ()


# --- evaluation is a function of absolute time ------------------------------


def _automated_scene():
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1", "name": "Left"}]
    scene["stages"] = Timeline(
        stages=(
            StageConfig(
                id="stage",
                start_s=0.0,
                duration_s=3.0,
                transition_in_s=1.0,
                transition_out_s=1.0,
                parameter_overrides=(
                    ParameterBinding(target_id="source.1", parameter_path="modFreq", value=9.0),
                    ParameterBinding(target_id="source.1", parameter_path="amp", value=0.4),
                ),
            ),
        )
    ).describe()
    scene["modulators"] = [{"id": "lfo", "rateHz": 0.7, "waveform": "sine"}]
    scene["modulation"] = ModulationMatrix(
        routes=(
            ModulationRoute(
                modulator_id="lfo", target_id="source.1", parameter_path="carrierFreq", depth=50.0
            ),
        )
    ).describe()
    return scene


BASE = {"modFreq": 4.0, "carrierFreq": 440.0, "amp": 0.7}
RATE = 1000.0
FRAMES = 3000


@pytest.mark.parametrize(
    "cuts",
    [[0, FRAMES], [0, FRAMES // 2, FRAMES], [0, 7, 101, 999, FRAMES - 3, FRAMES]],
    ids=["whole", "halves", "awkward"],
)
def test_the_parameter_series_does_not_depend_on_block_partitioning(cuts):
    scene = _automated_scene()
    whole = scene_parameter_series(scene, "source.1", 0, FRAMES, RATE, BASE)
    assert whole, "the scene should automate something"

    for path, reference in whole.items():
        joined = np.concatenate(
            [
                scene_parameter_series(
                    scene, "source.1", cuts[i], cuts[i + 1] - cuts[i], RATE, BASE
                )[path]
                for i in range(len(cuts) - 1)
            ]
        )
        assert np.array_equal(joined, reference), path


@pytest.mark.parametrize(
    "cuts",
    [[0, FRAMES], [0, FRAMES // 2, FRAMES], [0, 7, 101, 999, FRAMES - 3, FRAMES]],
    ids=["whole", "halves", "awkward"],
)
def test_the_gain_envelope_does_not_depend_on_block_partitioning(cuts):
    scene = _automated_scene()
    whole = scene_gain_envelope(scene, "source.1", 0, FRAMES, RATE)
    joined = np.concatenate(
        [
            scene_gain_envelope(scene, "source.1", cuts[i], cuts[i + 1] - cuts[i], RATE)
            for i in range(len(cuts) - 1)
        ]
    )
    assert np.array_equal(joined, whole)


def test_the_scalar_override_is_the_series_taken_at_that_instant():
    """The two views of the scene must not disagree about any moment."""

    scene = _automated_scene()
    for seconds in (0.0, 0.4, 1.5, 2.75, 3.5):
        series = scene_parameter_series(
            scene, "source.1", int(seconds * RATE), 1, RATE, BASE
        )
        scalar = scene_parameter_overrides(scene, "source.1", seconds, BASE)
        assert set(scalar) == set(series)
        for path, value in scalar.items():
            assert value == pytest.approx(float(series[path][0]))


def test_gain_is_left_to_the_gain_envelope_rather_than_returned_twice():
    scene = _automated_scene()
    series = scene_parameter_series(scene, "source.1", 0, 64, RATE, BASE)
    assert not AMPLITUDE_PATHS & set(series)


def test_the_series_reaches_the_stage_value_at_full_weight():
    scene = _automated_scene()
    middle = scene_parameter_series(scene, "source.1", int(1.5 * RATE), 1, RATE, BASE)
    assert float(middle["modFreq"][0]) == pytest.approx(9.0)


def test_a_parameter_no_stage_touches_is_absent_rather_than_restated():
    """An absent path means "unchanged"; the caller merges it over the base."""

    scene = _automated_scene()
    after = scene_parameter_series(scene, "source.1", int(9.0 * RATE), 1, RATE, BASE)
    assert "modFreq" not in after


def _merged(scene, start, frames, path):
    """What the caller actually sees: the series over the base value."""

    series = scene_parameter_series(scene, "source.1", start, frames, RATE, BASE)
    if path in series:
        return np.asarray(series[path], dtype=float)
    return np.full(frames, float(BASE[path]))


def test_partitioning_across_a_stage_boundary_changes_nothing():
    """The hard case: one block is inside the stage and the next is past it.

    The two blocks disagree about whether the parameter is present at all, so
    only the merged values can be compared - and those are what the renderer
    is handed.
    """

    scene = _automated_scene()
    total = int(6.0 * RATE)  # the stage ends halfway through
    whole = _merged(scene, 0, total, "modFreq")
    for cuts in ([0, total // 2, total], [0, 1500, 2999, 3001, 4500, total]):
        joined = np.concatenate(
            [
                _merged(scene, cuts[i], cuts[i + 1] - cuts[i], "modFreq")
                for i in range(len(cuts) - 1)
            ]
        )
        assert np.array_equal(joined, whole)

"""Nothing a project contains is lost by opening and saving it.

An older build must not delete what a newer one wrote, and this build must not
delete what an extension added. The rule applies at every level - the track,
its steps, its voices, their parameters, the scene, a scene source's record and
a stored path - because a document is only as durable as its least careful
layer.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.audio.sam_workbench.scene_state import (
    SAM_SCENE_SCHEMA_VERSION,
    empty_sam_scene,
    normalize_sam_scene,
)
from src.synth_functions.sound_creator import load_track_from_json, save_track_to_json
from src.utils.voice_file import VoicePreset, load_voice_preset, save_voice_preset

TRAJECTORY = {
    "geometry": {"type": "dome_traversal", "parameters": {}},
    "traversal": {"durationS": 2.0},
    "futureSpecKey": 7,
}


def _scene_with_unknowns():
    scene = empty_sam_scene()
    scene["somethingNobodyKnows"] = {"deep": [1, 2, 3]}
    scene["sources"] = [{"id": "source.1", "name": "A", "futureKey": 42}]
    return scene


def _track_with_unknowns():
    return {
        "global_settings": {"sample_rate": 44100, "unknownGlobal": "keep me"},
        "sam_scene": _scene_with_unknowns(),
        "steps": [
            {
                "duration": 2.0,
                "unknownStep": 1,
                "voices": [
                    {
                        "synth_function_name": "spatial_angle_modulation_sam2",
                        "sam_source_id": "source.1",
                        "unknownVoice": "v",
                        "params": {
                            "amp": 0.5,
                            "unknownParam": "p",
                            "canonicalTrajectory": copy.deepcopy(TRAJECTORY),
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def reopened(tmp_path):
    path = tmp_path / "track.json"
    save_track_to_json(copy.deepcopy(_track_with_unknowns()), str(path))
    return load_track_from_json(str(path))


def test_an_unknown_global_setting_survives(reopened):
    assert reopened["global_settings"]["unknownGlobal"] == "keep me"


def test_an_unknown_step_field_survives(reopened):
    assert reopened["steps"][0]["unknownStep"] == 1


def test_an_unknown_voice_field_survives(reopened):
    assert reopened["steps"][0]["voices"][0]["unknownVoice"] == "v"


def test_an_unknown_voice_parameter_survives(reopened):
    assert reopened["steps"][0]["voices"][0]["params"]["unknownParam"] == "p"


def test_a_stable_source_identifier_survives(reopened):
    assert reopened["steps"][0]["voices"][0]["sam_source_id"] == "source.1"


def test_an_unknown_field_inside_a_stored_path_survives(reopened):
    path = reopened["steps"][0]["voices"][0]["params"]["canonicalTrajectory"]
    assert path["futureSpecKey"] == 7


def test_an_unknown_scene_section_survives(reopened):
    assert reopened["sam_scene"]["somethingNobodyKnows"] == {"deep": [1, 2, 3]}


def test_an_unknown_field_on_a_scene_source_survives(reopened):
    assert reopened["sam_scene"]["sources"][0]["futureKey"] == 42


def test_the_scene_is_saved_at_the_current_schema_version(reopened):
    assert reopened["sam_scene"]["schemaVersion"] == SAM_SCENE_SCHEMA_VERSION


def test_a_voice_preset_keeps_an_unknown_parameter(tmp_path):
    path = tmp_path / "voice.voice"
    save_voice_preset(
        VoicePreset(
            synth_function_name="spatial_angle_modulation_sam2",
            params={"amp": 0.5, "unknownParam": "p"},
        ),
        str(path),
    )
    assert load_voice_preset(str(path)).params["unknownParam"] == "p"


def test_a_legacy_version_one_scene_reopens_with_its_data_intact(tmp_path):
    legacy = {
        "schemaVersion": 1,
        "sources": [{"id": "source.1", "name": "Left"}],
        "buses": [{"id": "master", "gainDb": -3.0}],
        "legacyExtra": True,
    }
    migrated = normalize_sam_scene(legacy)
    assert migrated["schemaVersion"] == SAM_SCENE_SCHEMA_VERSION
    assert migrated["sources"] == legacy["sources"]
    assert migrated["legacyExtra"] is True


def test_compiling_a_plan_changes_nothing_in_a_document_that_has_identifiers():
    """A plan is derived. Compiling one is not an edit."""

    from src.audio.sam_workbench.plan import plan_from_track

    track = _track_with_unknowns()
    before = json.dumps(track, sort_keys=True)
    plan_from_track(track)
    assert json.dumps(track, sort_keys=True) == before


def test_compiling_assigns_a_missing_identifier_and_nothing_else():
    """The one documented exception, and the reason for it.

    Identifiers belong in the track. A compile that invented them privately
    would hand every render a different name for the same source, which is the
    defect stable identifiers exist to prevent - so compiling fills in a
    missing one, and touches nothing else.
    """

    from src.audio.sam_workbench.plan import plan_from_track

    track = {
        "global_settings": {"sample_rate": 44100},
        "steps": [
            {
                "duration": 2.0,
                "voices": [
                    {
                        "synth_function_name": "spatial_angle_modulation_sam2",
                        "params": {"amp": 0.5},
                    }
                ],
            }
        ],
    }
    before = copy.deepcopy(track)
    plan_from_track(track)

    voice = track["steps"][0]["voices"][0]
    assert voice["sam_source_id"]
    # Nothing but the identifier was added, and no scene was forced in.
    assert "sam_scene" not in track
    assert {key: value for key, value in voice.items() if key != "sam_source_id"} == (
        before["steps"][0]["voices"][0]
    )


def test_compiling_twice_keeps_the_identifier_it_assigned():
    from src.audio.sam_workbench.plan import plan_from_track

    track = {
        "global_settings": {"sample_rate": 44100},
        "steps": [
            {
                "duration": 2.0,
                "voices": [
                    {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
                ],
            }
        ],
    }
    plan_from_track(track)
    assigned = track["steps"][0]["voices"][0]["sam_source_id"]
    plan_from_track(track)
    assert track["steps"][0]["voices"][0]["sam_source_id"] == assigned

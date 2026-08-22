import json

import numpy as np

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.scene_state import migrate_voice_scene, normalize_sam_scene
from src.synth_functions.sound_creator import load_track_from_json, save_track_to_json


def _scene(*, muted=False, gain_db=0.0):
    return normalize_sam_scene({
        "schemaVersion": 1,
        "sources": [{"id": "tone", "name": "Tone"}],
        "routing": {
            "schemaVersion": 1,
            "buses": [{"id": "master", "name": "Master", "gainDb": gain_db}],
            "sources": [{"sourceId": "tone", "busId": "master", "muted": muted}],
            "bands": {},
        },
    })


def test_track_scene_routing_is_audible():
    params = {"samSchemaVersion": 1, "amp": .5, "carrierFreq": 220.0, "modFreq": 2.0}
    reference = render_sam2_voice(.02, 8_000, params=params)
    quieter = render_sam2_voice(.02, 8_000, params=params, sam_scene=_scene(gain_db=-6), source_id="tone")
    muted = render_sam2_voice(.02, 8_000, params=params, sam_scene=_scene(muted=True), source_id="tone")
    assert np.max(np.abs(quieter)) < np.max(np.abs(reference))
    assert np.count_nonzero(muted) == 0


def test_track_scene_stage_and_modulation_change_rendered_parameters():
    params = {"samSchemaVersion": 1, "amp": .5, "carrierFreq": 220.0, "modFreq": 2.0}
    scene = _scene()
    scene["stages"] = {"schemaVersion": 1, "stages": [{
        "id": "intro", "startS": 0, "durationS": 1,
        "parameterOverrides": [{"targetId": "tone", "parameterPath": "carrierFreq", "value": 330}],
    }]}
    scene["modulation"] = {"schemaVersion": 1, "routes": [{
        "modulatorId": "lfo", "targetId": "tone", "parameterPath": "modFreq", "depth": 1,
    }]}
    reference = render_sam2_voice(.02, 8_000, params=params)
    rendered = render_sam2_voice(.02, 8_000, params=params, sam_scene=scene, source_id="tone")
    assert not np.allclose(rendered, reference)


def test_legacy_voice_scene_migrates_without_leaving_inert_keys():
    params = {"amp": .5, "samStages": {"schemaVersion": 1, "stages": []},
              "samModulation": {"schemaVersion": 1, "routes": []},
              "samRouting": _scene()["routing"]}
    scene = migrate_voice_scene(params)
    assert scene is not None
    assert not ({"samStages", "samModulation", "samRouting"} & params.keys())
    assert params == {"amp": .5}


def test_track_scene_survives_v2_json_round_trip(tmp_path):
    path = tmp_path / "track.json"
    track = {"global_settings": {}, "steps": [], "sam_scene": _scene(gain_db=-3)}
    assert save_track_to_json(track, path)
    assert "sam_scene" in json.loads(path.read_text())
    assert load_track_from_json(path)["sam_scene"] == track["sam_scene"]

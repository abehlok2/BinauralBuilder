from __future__ import annotations

import json

import pytest

from src.audio.sam_workbench.state import load_parameter_state, save_parameter_state


def test_parameter_state_round_trip_preserves_unknown_values(tmp_path):
    path = tmp_path / "voice.samstate.json"
    params = {
        "rendererMode": "abstract_pm",
        "carrierFreq": 432.0,
        "futureExtension": {"enabled": True},
    }

    assert save_parameter_state(params, path) == path
    assert load_parameter_state(path, is_transition=False) == params


def test_parameter_state_rejects_wrong_voice_kind(tmp_path):
    path = tmp_path / "transition.samstate.json"
    save_parameter_state({"startCarrierFreq": 200.0}, path, is_transition=True)

    with pytest.raises(ValueError, match="transition parameters into a static voice"):
        load_parameter_state(path, is_transition=False)


def test_parameter_state_rejects_unrecognised_document(tmp_path):
    path = tmp_path / "ordinary.json"
    path.write_text(json.dumps({"params": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a SAM workbench"):
        load_parameter_state(path)


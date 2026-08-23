"""The Scene panels route the track's sources, not invented ones.

The routing panel used to start with a "source.1" that looked exactly like a
real route and referred to nothing. A project could be routed, muted and soloed
against a source the track does not have, and nothing said so - the row looks
like any other row. These tests hold the roster to coming from the track, and
the faults that follow from it to being reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.scene_state import (
    MODULATOR_WAVEFORMS,
    assign_source_ids,
    modulator_series,
)

STEPS = [
    {
        "duration": 60.0,
        "description": "induction",
        "voices": [
            {
                "synth_function_name": "spatial_angle_modulation_sam2",
                "description": "Lead tone",
                "params": {},
            },
            {
                "synth_function_name": "spatial_angle_modulation_sam2",
                "description": "Pad",
                "params": {},
            },
        ],
    }
]


# --- the roster -------------------------------------------------------------


def test_the_routing_panel_starts_with_nothing_to_route(qtbot):
    """No placeholder. Until a track supplies a roster there are no sources."""

    from src.ui.sam_routing_panel import SamRoutingPanel

    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    assert panel.sources == ()
    assert panel.roster() == {}


def test_a_roster_brings_real_names_and_keeps_identifiers(qtbot):
    from src.ui.sam_routing_panel import SamRoutingPanel

    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    panel.set_roster([{"id": "sam.1", "name": "Lead tone"}])

    assert [source.source_id for source in panel.sources] == ["sam.1"]
    cell = panel.source_table.item(0, 0)
    assert cell.text() == "Lead tone"
    from PyQt5.QtCore import Qt

    assert cell.data(Qt.UserRole) == "sam.1"


def test_routing_survives_a_roster_refresh(qtbot):
    """Reopening a track must not reset every bus assignment."""

    from src.ui.sam_routing_panel import SamRoutingPanel

    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    panel.set_roster([{"id": "sam.1", "name": "Lead"}])
    panel._set_source(0, gain_db=-6.0, muted=True)

    panel.set_roster([{"id": "sam.1", "name": "Lead renamed"}, {"id": "sam.2", "name": "Pad"}])

    kept = panel.sources[0]
    assert kept.source_id == "sam.1"
    assert kept.gain_db == pytest.approx(-6.0)
    assert kept.muted is True
    assert panel.source_table.item(0, 0).text() == "Lead renamed"


def test_a_route_whose_source_left_the_track_is_kept_and_marked(qtbot):
    """Dropping it would discard a mute an undo in the editor would restore."""

    from src.ui.sam_routing_panel import SamRoutingPanel

    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    panel.set_roster([{"id": "sam.1", "name": "Lead"}, {"id": "sam.2", "name": "Pad"}])
    panel._set_source(1, muted=True)

    panel.set_roster([{"id": "sam.1", "name": "Lead"}])

    assert panel.unresolved_sources() == ("sam.2",)
    assert "missing" in panel.source_table.item(1, 0).text()
    assert panel.sources[1].muted is True


def test_the_roster_order_follows_the_track(qtbot):
    from src.ui.sam_routing_panel import SamRoutingPanel

    panel = SamRoutingPanel()
    qtbot.addWidget(panel)
    panel.set_roster([{"id": "a"}, {"id": "b"}])
    panel.set_roster([{"id": "b"}, {"id": "a"}])
    assert [source.source_id for source in panel.sources] == ["b", "a"]


# --- the dialog feeds the panels the real roster ----------------------------


def test_the_dialog_populates_routing_from_the_track_steps(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)
    dialog.set_steps(STEPS)

    names = set(dialog.routing_panel.roster().values())
    assert {"Lead tone", "Pad"} <= names
    assert dialog.modulation_panel.roster() == dialog.routing_panel.roster()


def test_identifiers_are_written_back_onto_the_real_voices(qtbot):
    """An identifier assigned to a copy is thrown away with the copy."""

    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    steps = [
        {
            "duration": 10.0,
            "voices": [
                {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
            ],
        }
    ]
    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)
    dialog.set_steps(steps)

    assigned = steps[0]["voices"][0].get("sam_source_id")
    assert assigned
    assert assigned in dialog.routing_panel.roster()


def test_a_dangling_route_is_reported_where_the_user_is_looking(qtbot):
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}},
        scene_data={
            "sources": [{"id": "sam.1", "name": "Lead"}],
            "routing": {
                "buses": [{"id": "master", "name": "Master"}],
                "sources": [{"sourceId": "ghost", "busId": "master"}],
            },
        },
    )
    qtbot.addWidget(dialog)

    reported = " ".join(f"{issue.path} {issue.message}" for issue in dialog.issues())
    assert "ghost" in reported


# --- modulators are definitions ---------------------------------------------


def test_every_default_modulator_has_a_real_definition(qtbot):
    from src.ui.sam_modulation_panel import SamModulationPanel

    panel = SamModulationPanel()
    qtbot.addWidget(panel)

    definitions = {entry["id"]: entry for entry in panel.modulators()}
    assert definitions["random.walk"]["waveform"] == "random"
    for entry in definitions.values():
        assert entry["waveform"] in MODULATOR_WAVEFORMS
        assert float(entry["rateHz"]) > 0.0


def test_editing_a_definition_changes_what_is_serialized(qtbot):
    from src.ui.sam_modulation_panel import SamModulationPanel

    panel = SamModulationPanel()
    qtbot.addWidget(panel)
    panel.modulator_list.setCurrentRow(panel._modulators.index("lfo.slow"))
    panel.waveform_combo.setCurrentText("square")
    panel.rate_spin.setValue(3.5)

    stored = {entry["id"]: entry for entry in panel.modulators()}["lfo.slow"]
    assert stored["waveform"] == "square"
    assert stored["rateHz"] == pytest.approx(3.5)


def test_definitions_round_trip(qtbot):
    from src.ui.sam_modulation_panel import SamModulationPanel

    panel = SamModulationPanel()
    qtbot.addWidget(panel)
    panel.modulator_list.setCurrentRow(0)
    panel.rate_spin.setValue(2.25)
    saved = panel.modulators()

    restored = SamModulationPanel()
    qtbot.addWidget(restored)
    restored.set_modulators(saved)
    assert restored.modulators() == saved


# --- the engine renders them ------------------------------------------------


@pytest.mark.parametrize("waveform", MODULATOR_WAVEFORMS)
def test_every_waveform_stays_in_range(waveform):
    times = np.linspace(0.0, 10.0, 501)
    values = modulator_series(
        {"modulators": [{"id": "m", "waveform": waveform, "rateHz": 0.7, "seed": 3}]},
        times,
    )["m"]
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_the_random_waveform_is_seeded_and_block_independent():
    """An export must match the preview it was approved from."""

    times = np.linspace(0.0, 8.0, 401)
    scene = {"modulators": [{"id": "m", "waveform": "random", "rateHz": 0.5, "seed": 11}]}

    whole = modulator_series(scene, times)["m"]
    pieces = np.concatenate(
        [modulator_series(scene, times[start : start + 97])["m"] for start in range(0, 401, 97)]
    )
    assert np.array_equal(whole, pieces)

    other = modulator_series(
        {"modulators": [{"id": "m", "waveform": "random", "rateHz": 0.5, "seed": 12}]},
        times,
    )["m"]
    assert not np.array_equal(whole, other)


def test_two_named_modulators_no_longer_render_identically():
    """The defect the definitions fix: a name with no shape behind it."""

    times = np.linspace(0.0, 30.0, 601)
    scene = {
        "modulators": [
            {"id": "lfo.slow", "waveform": "sine", "rateHz": 0.05},
            {"id": "random.walk", "waveform": "random", "rateHz": 0.2, "seed": 1},
        ]
    }
    series = modulator_series(scene, times)
    assert not np.allclose(series["lfo.slow"], series["random.walk"])


# --- identifiers stay stable ------------------------------------------------


def test_reordering_voices_does_not_move_a_route_to_another_source():
    track = {"steps": [step.copy() for step in STEPS]}
    track["steps"][0]["voices"] = [dict(v) for v in STEPS[0]["voices"]]
    scene = assign_source_ids(track, None, persist_scene=True)
    first = [entry["id"] for entry in scene["sources"]]

    track["steps"][0]["voices"].reverse()
    reordered = assign_source_ids(track, scene, persist_scene=True)

    assert set(entry["id"] for entry in reordered["sources"]) == set(first)
    assert [entry["id"] for entry in reordered["sources"]] == list(reversed(first))

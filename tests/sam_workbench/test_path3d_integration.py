"""The 3-D designer, tied to the dataset and the renderer it feeds.

The four-view designer and the canonical PathModel already existed. What was
missing was the connections: the coverage shell was drawn at a radius guessed
from the path itself rather than from the dataset, so a path could sit exactly
on the shell and still be nowhere near a measurement; coverage warnings only
appeared at render time, long after the moment when they would have changed
what the user drew; and the legacy 2-D creator and the 3-D designer looked like
unrelated tools with no way to carry a path from one to the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
from src.audio.sam_workbench.trajectory import (
    path_model_from_dict,
    promote_profile_to_trajectory,
)

SOFA = str(Path(__file__).resolve().parent / "fixtures" / "synthetic_hrir.sofa")

PROFILE = {
    "schemaVersion": 2,
    "coordinateSpace": "normalized_listener_2d",
    "axisConvention": "x_right_y_down",
    "sceneCentre": [0.0, 0.0],
    "sceneUnitsPerMetre": 100.0,
    "closedLoop": True,
    "kind": "Circle",
    "smoothingPasses": 1,
    "smoothingRatio": 0.25,
    "points": [[-100.0, 0.0], [0.0, -80.0], [100.0, 0.0], [0.0, 80.0]],
}


# --- the two editors are related, explicitly --------------------------------


def test_a_two_dimensional_profile_promotes_to_a_canonical_path():
    spec = promote_profile_to_trajectory(PROFILE)
    model = path_model_from_dict(spec)

    assert model.is_listener_relative is True
    positions = model.positions(np.linspace(0.0, model.duration_s, 64))
    # A plane at ear height: the 2-D editor has no third axis to give.
    assert np.allclose(positions[:, 2], 0.0)
    # Metres, not scene units. 100 units per metre against a 100-unit radius.
    assert float(np.max(np.linalg.norm(positions, axis=1))) < 2.0


def test_the_promotion_records_where_the_path_came_from():
    """A promoted path should not be indistinguishable from a drawn one."""

    spec = promote_profile_to_trajectory(PROFILE)
    assert spec["promotedFrom"]["editor"] == "custom_path_creator"
    assert spec["promotedFrom"]["sceneUnitsPerMetre"] == pytest.approx(100.0)


def test_the_promotion_follows_the_curve_the_2d_preview_drew():
    """Not merely the control points: the smoothing is part of the shape."""

    spec = promote_profile_to_trajectory(PROFILE)
    points = spec["geometry"]["controlPointsM"]
    assert len(points) > len(PROFILE["points"])


def test_a_closed_profile_loops_and_an_open_one_does_not():
    looped = promote_profile_to_trajectory(PROFILE)
    assert looped["traversal"]["loop"] == "loop"

    open_profile = dict(PROFILE, closedLoop=False)
    assert promote_profile_to_trajectory(open_profile)["traversal"]["loop"] == "pingpong"


def test_the_traversal_duration_is_the_caller_s_to_choose():
    spec = promote_profile_to_trajectory(PROFILE, duration_s=42.0)
    assert path_model_from_dict(spec).duration_s == pytest.approx(42.0)


# --- coverage comes from the dataset ----------------------------------------


def test_the_shell_radius_comes_from_the_dataset_when_there_is_one(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)

    guessed, measured = dialog._shell_radius()
    assert measured is False

    dialog.set_hrtf_dataset(load_sofa(SOFA), label="synthetic")
    from_dataset, measured = dialog._shell_radius()
    assert measured is True

    expected = float(np.median(np.linalg.norm(load_sofa(SOFA).positions_m, axis=1)))
    assert from_dataset == pytest.approx(expected)


def test_without_a_dataset_the_editor_says_it_is_not_checking(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)
    dialog._refresh_coverage()
    assert "not being checked" in dialog.coverage_label.text()


def test_a_dataset_that_covers_the_path_says_so(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)
    dialog.set_hrtf_dataset(load_sofa(SOFA), label="synthetic")
    text = dialog.coverage_label.text()
    assert text and "could not" not in text


def test_a_path_the_dataset_cannot_support_is_reported_beside_it(qtbot):
    """While it is being dragged, not after the export."""

    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog(
        {
            "geometry": {"type": "dome_traversal", "parameters": {"turns": 40}},
            "traversal": {"durationS": 1.0},
        }
    )
    qtbot.addWidget(dialog)
    dialog.set_hrtf_dataset(load_sofa(SOFA), label="synthetic")

    assert "•" in dialog.coverage_label.text()


def test_an_unreadable_dataset_is_reported_rather_than_ignored(qtbot, tmp_path):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)
    dialog.set_hrtf_dataset(str(tmp_path / "missing.sofa"))
    assert "could not be read" in dialog.coverage_label.text()


def test_clearing_the_dataset_returns_to_the_guessed_shell(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)
    dialog.set_hrtf_dataset(load_sofa(SOFA))
    assert dialog._shell_radius()[1] is True

    dialog.set_hrtf_dataset(None)
    assert dialog._shell_radius()[1] is False
    assert "not being checked" in dialog.coverage_label.text()


# --- the frame is explicit --------------------------------------------------


def test_the_editor_distinguishes_listener_from_world_coordinates(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)

    frames = {
        dialog.frame_combo.itemData(index)
        for index in range(dialog.frame_combo.count())
    }
    assert frames == {"listener_relative_cartesian", "world_cartesian"}
    assert dialog.trajectory_spec()["coordinateSystem"] in frames


def test_a_world_relative_path_round_trips(qtbot):
    from src.ui.sam_path3d_dialog import SamPath3DDialog

    dialog = SamPath3DDialog()
    qtbot.addWidget(dialog)
    dialog.frame_combo.setCurrentIndex(
        dialog.frame_combo.findData("world_cartesian")
    )
    spec = dialog.trajectory_spec()
    assert spec["coordinateSystem"] == "world_cartesian"
    assert path_model_from_dict(spec).is_listener_relative is False

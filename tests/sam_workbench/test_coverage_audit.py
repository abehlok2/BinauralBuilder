"""Nothing declared in the core may be unreachable, or reachable and broken.

A registry of parameters and a library of path geometries are both promises:
they say the application can do these things. This file checks the promises
against the interface, in both directions, because both directions fail
differently and both had failed.

Something declared and not offered is invisible - the user cannot find it and
has no way to know it exists. Something offered and not understood is worse: it
looks available, and produces an error or, quietly, a different shape than the
one on screen. The path editor offered a Line the loader refused outright, and
offered a circle, spiral and helix whose dragged points were thrown away and
replaced with defaults.

Every case below is one of those, kept as a test so the two halves cannot drift
apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from PyQt5.QtWidgets import QComboBox  # noqa: E402

from src.audio.sam_workbench.parameters import (  # noqa: E402
    EXPERT,
    SAM2_FIELDS,
    fields_for_mode,
)
from src.audio.sam_workbench.trajectory.serialization import (  # noqa: E402
    GEOMETRY_TYPES,
    trajectory_from_dict,
)
from src.ui.sam_path_editor_dialog import SamPathEditorDialog  # noqa: E402
from src.ui.sam_workbench_dialog import SamWorkbenchDialog  # noqa: E402

STATIC_VOICE = {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
TRANSITION_VOICE = {
    "synth_function_name": "spatial_angle_modulation_sam2_transition",
    "is_transition": True,
    "params": {"initial_offset": 3.0, "duration": 42.0, "transition_curve": "linear"},
}

#: Fields the workbench deliberately does not own. Transition timing is the
#: voice's envelope, shared with every other synth function and edited in the
#: voice editor; the workbench carries it through and says so.
NOT_OWNED_BY_THE_WORKBENCH = {"initial_offset", "duration", "transition_curve"}


# --- parameters -------------------------------------------------------------


@pytest.mark.parametrize(
    "voice, is_transition",
    [(STATIC_VOICE, False), (TRANSITION_VOICE, True)],
    ids=["static", "transition"],
)
def test_every_registry_field_the_workbench_owns_has_a_control(qtbot, voice, is_transition):
    dialog = SamWorkbenchDialog(dict(voice))
    qtbot.addWidget(dialog)
    dialog.mode_combo.setCurrentText(EXPERT)

    reachable = set(dialog.parameter_panel._controls)
    reachable |= set(dialog.path_panel.params())
    reachable |= set(dialog.hrtf_panel.params())

    missing = []
    for entry in SAM2_FIELDS:
        if entry.name in NOT_OWNED_BY_THE_WORKBENCH:
            continue
        names = [name for name in entry.names_for(is_transition) if name]
        if not names:
            continue
        # A JSON profile has a dedicated editor rather than a generic control,
        # and no key at all until there is a profile to carry - inventing an
        # empty one would rewrite a voice that had none.
        if entry.kind == "json":
            assert dialog.path_panel.designer_button.isEnabled()
            continue
        if not any(name in reachable for name in names):
            missing.append(entry.name)
    assert missing == []


def test_every_choice_offers_every_choice(qtbot):
    """A choice control that omits a value makes that value unreachable."""

    dialog = SamWorkbenchDialog(dict(STATIC_VOICE))
    qtbot.addWidget(dialog)
    dialog.mode_combo.setCurrentText(EXPERT)

    for entry in SAM2_FIELDS:
        if entry.kind != "choice" or entry.name in NOT_OWNED_BY_THE_WORKBENCH:
            continue
        control = dialog.parameter_panel._controls.get(entry.name)
        assert control is not None, entry.name
        combos = control.findChildren(QComboBox)
        assert combos, entry.name
        offered = {combos[0].itemText(index) for index in range(combos[0].count())}
        offered |= {
            str(combos[0].itemData(index))
            for index in range(combos[0].count())
            if combos[0].itemData(index) is not None
        }
        assert set(entry.choices) <= offered, (entry.name, sorted(set(entry.choices) - offered))


def test_transition_timing_is_reported_as_carried_rather_than_edited(qtbot):
    """The compatibility report must not claim to edit what it only preserves."""

    dialog = SamWorkbenchDialog(dict(TRANSITION_VOICE))
    qtbot.addWidget(dialog)
    report = dialog.compatibility_view.toPlainText()
    preserved = report.split("Parameters preserved but not edited here:")[1]

    for name in NOT_OWNED_BY_THE_WORKBENCH:
        assert name in preserved, name
    assert "voice editor" in preserved

    # Carried through untouched, which is the other half of the promise.
    collected = dialog.collect_params()
    assert collected["initial_offset"] == 3.0
    assert collected["duration"] == 42.0
    assert collected["transition_curve"] == "linear"


def test_a_custom_path_profile_round_trips_through_the_panel(qtbot):
    """The JSON profile has no generic control, so check the editor keeps it."""

    profile = {
        "kind": "linear",
        "points": [[0.0, -100.0], [100.0, 0.0]],
        "smoothingPasses": 1,
        "smoothingRatio": 0.25,
    }
    dialog = SamWorkbenchDialog(
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "params": {"pathType": "custom", "customPathProfile": profile},
        }
    )
    qtbot.addWidget(dialog)
    assert dialog.collect_params()["customPathProfile"] == profile


def test_a_static_voice_has_nothing_left_over(qtbot):
    dialog = SamWorkbenchDialog(dict(STATIC_VOICE))
    qtbot.addWidget(dialog)
    preserved = dialog.compatibility_view.toPlainText().split(
        "Parameters preserved but not edited here:"
    )[1]
    assert "(none)" in preserved


def test_the_disclosure_modes_between_them_show_everything_owned():
    """A field in no mode is in the registry and in no interface."""

    shown = {entry.name for entry in fields_for_mode(EXPERT)}
    declared = {entry.name for entry in SAM2_FIELDS}
    assert declared - shown == NOT_OWNED_BY_THE_WORKBENCH


# --- path geometry ----------------------------------------------------------


def _offered(dialog) -> list[str]:
    return [
        dialog.primitive_combo.itemText(index)
        for index in range(dialog.primitive_combo.count())
    ]


def _editor_for(primitive: str, qtbot):
    """The dialog that owns a geometry kind.

    There are two path editors: the original two-dimensional one, and the
    three-dimensional designer that owns the primitives with height in them.
    A kind belongs to exactly one of them, and this picks the right one.
    """

    from src.ui.sam_path3d_dialog import SamPath3DDialog

    for factory in (SamPathEditorDialog, SamPath3DDialog):
        dialog = factory()
        qtbot.addWidget(dialog)
        if primitive in _offered(dialog):
            dialog.primitive_combo.setCurrentText(primitive)
            return dialog
    raise AssertionError(f"no editor offers geometry {primitive!r}")


def test_the_editors_and_the_loader_agree_on_the_geometry_list(qtbot):
    """Between them the editors offer every kind the loader accepts, and no more.

    Neither editor has to offer all of them - a dome traversal has no place in
    a flat top-down canvas - but a kind the loader understands that no editor
    can create would be unreachable, and a kind an editor offers that the
    loader rejects would be unsavable.
    """

    from src.ui.sam_path3d_dialog import SamPath3DDialog

    flat = SamPathEditorDialog()
    spatial = SamPath3DDialog()
    qtbot.addWidget(flat)
    qtbot.addWidget(spatial)
    # The 3-D designer separates its groups with non-selectable headings.
    offered = {
        name
        for name in _offered(flat) + _offered(spatial)
        if name and not name.startswith("\u2014")
    }

    assert sorted(offered - set(GEOMETRY_TYPES)) == []
    assert sorted(set(GEOMETRY_TYPES) - offered) == []


@pytest.mark.parametrize("primitive", sorted(GEOMETRY_TYPES))
def test_every_offered_geometry_produces_a_trajectory_the_core_accepts(qtbot, primitive):
    dialog = _editor_for(primitive, qtbot)

    trajectory = trajectory_from_dict(dialog.trajectory_spec())
    geometry = getattr(trajectory.geometry, "geometry", trajectory.geometry)
    curve = geometry.evaluate(np.linspace(0.0, 1.0, 512))
    assert curve.shape[1] == 3
    assert np.all(np.isfinite(curve))


@pytest.mark.parametrize(
    "primitive",
    ["polyline", "polygon", "spline", "line", "arc", "circle", "ellipse", "spiral", "helix", "lissajous"],
)
def test_a_seeded_shape_survives_the_round_trip(qtbot, primitive):
    """What comes back must be the shape on the canvas, not the seed's defaults.

    This is the failure that was silent: a circle whose points had been dragged
    came back as a default unit circle, and a spiral came back at half its
    radius, because the loader rebuilt the primitive instead of reading the
    points it was given.
    """

    dialog = SamPathEditorDialog()
    qtbot.addWidget(dialog)
    dialog.primitive_combo.setCurrentText(primitive)

    specification = dialog.trajectory_spec()
    control_points = np.asarray(specification["geometry"]["controlPointsM"], dtype=float)
    assert control_points.size

    trajectory = trajectory_from_dict(specification)
    geometry = getattr(trajectory.geometry, "geometry", trajectory.geometry)
    curve = geometry.evaluate(np.linspace(0.0, 1.0, 4001))

    # Every control point lies on the reconstructed curve, to within a
    # millimetre - far below anything audible as a change of direction.
    worst = max(float(np.min(np.linalg.norm(curve - point, axis=1))) for point in control_points)
    assert worst < 0.01


def test_dragging_a_point_changes_the_saved_path(qtbot):
    dialog = SamPathEditorDialog()
    qtbot.addWidget(dialog)
    dialog.primitive_combo.setCurrentText("circle")

    before = trajectory_from_dict(dialog.trajectory_spec())
    moved = list(dialog._points)
    moved[0] = [moved[0][0] + 0.9, moved[0][1], moved[0][2]]
    dialog._points = moved
    after = trajectory_from_dict(dialog.trajectory_spec())

    samples = np.linspace(0.0, 1.0, 256)
    first = getattr(before.geometry, "geometry", before.geometry).evaluate(samples)
    second = getattr(after.geometry, "geometry", after.geometry).evaluate(samples)
    assert np.max(np.linalg.norm(first - second, axis=1)) > 0.1


def test_a_payload_without_points_still_builds_its_primitive():
    """A project written by hand may describe the shape rather than sample it."""

    payload = {
        "schemaVersion": 1,
        "geometry": {"type": "circle", "parameters": {"radius_m": 2.5}},
        "traversal": {"durationS": 4.0},
    }
    trajectory = trajectory_from_dict(payload)
    geometry = getattr(trajectory.geometry, "geometry", trajectory.geometry)
    curve = geometry.evaluate(np.linspace(0.0, 1.0, 64))
    assert np.allclose(np.linalg.norm(curve[:, :2], axis=1), 2.5, atol=1e-6)


def test_an_unknown_geometry_names_what_is_available():
    with pytest.raises(ValueError) as error:
        trajectory_from_dict({"geometry": {"type": "trapezoid"}, "traversal": {}})
    assert "trapezoid" in str(error.value)
    for name in ("polyline", "circle", "mathematical"):
        assert name in str(error.value)

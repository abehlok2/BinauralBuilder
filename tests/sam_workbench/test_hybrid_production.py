"""Hybrid rendering, end to end, in the order the specification names.

The order is the substance, not presentation. Cue modification acts on the
*filters*, because interaural time and level differences are properties of the
filter pair for a direction; recovering them from an already-mixed stereo
signal would mean undoing the convolution first, and doing it badly. Headphone
correction runs once over the finished mix, because applying it per stem
applies it twice.

    Source -> SAM -> 3-D trajectory -> HRTF -> cue -> headphone -> output

A neutral creative stage must leave a hybrid render identical to a physical
one. That is what makes an A/B between them a comparison of exactly one thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.render.hybrid import SIGNAL_CHAIN
from src.audio.sam_workbench.render.registry import REGISTRY, renderer

SOFA = str(Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa")
RATE = 44100
TRAJECTORY = {
    "geometry": {"type": "dome_traversal", "parameters": {"turns": 2}},
    "traversal": {"durationS": 0.5},
}


def _render(mode, **options):
    params = {
        "amp": 0.5,
        "carrierFreq": 300.0,
        "rendererMode": mode,
        "hrtfAsset": SOFA,
        "canonicalTrajectory": TRAJECTORY,
        "hrtfOptions": dict(options),
    }
    return render_sam2_voice(0.5, RATE, params=params)


# --- the mode is reachable --------------------------------------------------


def test_hybrid_is_a_renderer_a_single_voice_can_use():
    assert renderer("hybrid").voice_renderable is True
    assert "hybrid" in [entry.identifier for entry in REGISTRY.voice_renderable]


def test_hybrid_renders_finite_stereo_audio():
    audio = _render("hybrid")
    assert audio.shape == (RATE // 2, 2)
    assert np.all(np.isfinite(audio))


def test_hybrid_appears_in_the_renderer_menu(qtbot):
    pytest.importorskip("pytestqt")
    from src.ui.sam_workbench_dialog import SamWorkbenchDialog

    dialog = SamWorkbenchDialog(
        {"synth_function_name": "spatial_angle_modulation_sam2", "params": {}}
    )
    qtbot.addWidget(dialog)
    offered = [
        dialog.renderer_combo.itemData(index)
        for index in range(dialog.renderer_combo.count())
    ]
    assert "hybrid" in offered


# --- a neutral creative stage is exactly the physical render ----------------


def test_a_neutral_hybrid_render_is_the_physical_render():
    """Otherwise an A/B between them compares more than one thing."""

    assert np.allclose(_render("hybrid"), _render("hrtf"), atol=1e-6)


def test_the_hybrid_spec_reports_itself_as_physical_when_nothing_is_added():
    from src.audio.sam_workbench.render.hybrid import HybridSpec

    assert HybridSpec.from_options({}).is_physical
    assert not HybridSpec.from_options({"cue": {"itdScale": 1.5}}).is_physical


# --- each stage reaches the audio -------------------------------------------


@pytest.mark.parametrize(
    "options, label",
    [
        ({"cue": {"itdScale": 1.6}}, "interaural time difference"),
        ({"cue": {"ildScale": 1.5}}, "interaural level difference"),
        ({"cue": {"pinnaScale": 0.5}}, "pinna colouration"),
        ({"anchor": {"enabled": True}}, "spatial anchor"),
        ({"outputGainDb": -6.0}, "output gain"),
    ],
)
def test_each_hybrid_stage_changes_the_render(options, label):
    """A stage that changes nothing is a stage that is being dropped."""

    assert not np.allclose(_render("hybrid"), _render("hybrid", **options), atol=1e-6), label


def test_cue_settings_are_accepted_in_either_spelling():
    """Stored options are camelCase everywhere; this one read only snake_case."""

    assert np.allclose(
        _render("hybrid", cue={"itdScale": 1.6}),
        _render("hybrid", cue={"itd_scale": 1.6}),
    )


def test_output_gain_scales_by_the_decibels_it_names():
    plain = _render("hybrid").astype(np.float64)
    quieter = _render("hybrid", outputGainDb=-6.0).astype(np.float64)
    ratio = np.sqrt(np.mean(quieter**2)) / np.sqrt(np.mean(plain**2))
    assert ratio == pytest.approx(10.0 ** (-6.0 / 20.0), rel=0.01)


# --- the order is explicit --------------------------------------------------


def test_the_stage_order_is_the_one_the_specification_names():
    assert SIGNAL_CHAIN.index("SAM") < SIGNAL_CHAIN.index("3D trajectory")
    assert SIGNAL_CHAIN.index("3D trajectory") < SIGNAL_CHAIN.index("HRTF interpolation")
    assert SIGNAL_CHAIN.index("HRTF interpolation") < SIGNAL_CHAIN.index("Cue modification")
    assert SIGNAL_CHAIN.index("Cue modification") < SIGNAL_CHAIN.index("Output")


def test_cue_modification_acts_on_the_filters_rather_than_the_mix():
    """Scaling the ITD must change the two channels differently.

    A gain applied to an already-mixed signal could not do that, so this is
    what distinguishes acting on the filter pair from acting on the output.
    """

    plain = _render("hybrid").astype(np.float64)
    modified = _render("hybrid", cue={"itdScale": 1.8}).astype(np.float64)
    # The adapter returns frame-major (frames, 2), so channels are columns.
    per_channel = np.abs(modified - plain).max(axis=0)
    assert per_channel.min() > 0.0
    assert not np.isclose(per_channel[0], per_channel[1], rtol=1e-3)


def test_hybrid_still_claims_no_more_than_hrtf_does():
    """The creative stage must not upgrade what the renderer says it can do."""

    hybrid = renderer("hybrid").capabilities
    assert hybrid.supports_cue_modification
    assert "declared departure" in hybrid.honesty_note


def test_the_abstract_renderer_still_denies_being_a_spatializer():
    note = renderer("abstract_pm").capabilities.honesty_note.lower()
    assert "not a spatializer" in note
    assert not renderer("abstract_pm").capabilities.physical_elevation

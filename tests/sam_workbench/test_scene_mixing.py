"""A scene mixes through real buses and bands, not a folded scalar.

The pieces for this existed and were tested, and nothing used them. Production
folded a source's bus gain and its mute/solo state into a single scalar
multiplied onto that source's own audio. That gets the level right when nothing
downstream sums buses, and it is not a mixer: there is no bus to meter, no bus
to process, and nothing a band setting can act on.

Band splitting is why the mixer is stateful. A biquad's output depends on the
samples before it, so restarting the filters every block steps every band's
output - a click, not a change of tone. These tests hold that state in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.plan import plan_from_track
from src.audio.sam_workbench.render.routing import BandRouting, BusSpec, SourceRouting
from src.audio.sam_workbench.render.scene_mix import SceneMixer, mixer_from_plan
from src.audio.sam_workbench.scene_state import empty_sam_scene
from src.synth_functions.sound_creator import assemble_track_from_data

RATE = 44100.0


def _stem(value, frames=2048):
    return np.full((2, frames), float(value))


def _stems():
    return {"a": _stem(0.5), "b": _stem(0.25)}


# --- summing, gain, mute and solo -------------------------------------------


def test_sources_without_routing_reach_the_master_at_unity():
    """A scene that has never been routed still renders."""

    assert SceneMixer(sample_rate_hz=RATE).process(_stems()).master[0, 0] == pytest.approx(0.75)


def test_a_source_gain_scales_only_that_source():
    mixer = SceneMixer((SourceRouting("a", gain_db=-6.0),), sample_rate_hz=RATE)
    expected = 0.5 * 10.0 ** (-6.0 / 20.0) + 0.25
    assert mixer.process(_stems()).master[0, 0] == pytest.approx(expected)


def test_muting_a_source_removes_it_and_reports_it():
    mixer = SceneMixer((SourceRouting("a", muted=True),), sample_rate_hz=RATE)
    routed = mixer.process(_stems())
    assert routed.master[0, 0] == pytest.approx(0.25)
    assert routed.silenced == ("a",)


def test_soloing_a_source_silences_every_other_one():
    mixer = SceneMixer((SourceRouting("a", soloed=True),), sample_rate_hz=RATE)
    routed = mixer.process(_stems())
    assert routed.master[0, 0] == pytest.approx(0.5)
    assert routed.silenced == ("b",)


def test_muting_wins_over_soloing_the_same_source():
    """Muting something is the more specific instruction."""

    mixer = SceneMixer((SourceRouting("a", soloed=True, muted=True),), sample_rate_hz=RATE)
    assert "a" in mixer.process(_stems()).silenced


def test_soloing_several_sources_keeps_all_of_them():
    mixer = SceneMixer(
        (SourceRouting("a", soloed=True), SourceRouting("b", soloed=True)),
        sample_rate_hz=RATE,
    )
    routed = mixer.process(_stems())
    assert routed.master[0, 0] == pytest.approx(0.75)
    assert routed.silenced == ()


# --- buses are real ---------------------------------------------------------


def test_a_bus_gain_applies_once_to_everything_on_it():
    mixer = SceneMixer(
        (SourceRouting("a", bus_id="fx"), SourceRouting("b", bus_id="fx")),
        (BusSpec(id="fx", gain_db=-12.0), BusSpec()),
        sample_rate_hz=RATE,
    )
    assert mixer.process(_stems()).master[0, 0] == pytest.approx(
        0.75 * 10.0 ** (-12.0 / 20.0)
    )


def test_each_bus_is_kept_separately_so_it_can_be_metered():
    mixer = SceneMixer(
        (SourceRouting("a", bus_id="fx"), SourceRouting("b", bus_id="dry")),
        (BusSpec(id="fx"), BusSpec(id="dry"), BusSpec()),
        sample_rate_hz=RATE,
    )
    routed = mixer.process(_stems())
    assert routed.bus_stems["fx"][0, 0] == pytest.approx(0.5)
    assert routed.bus_stems["dry"][0, 0] == pytest.approx(0.25)
    assert mixer.diagnostics()["busPeaks"]["fx"] == pytest.approx(0.5)


def test_muting_a_bus_silences_every_source_on_it():
    mixer = SceneMixer(
        (SourceRouting("a", bus_id="fx"), SourceRouting("b", bus_id="dry")),
        (BusSpec(id="fx", muted=True), BusSpec(id="dry"), BusSpec()),
        sample_rate_hz=RATE,
    )
    routed = mixer.process(_stems())
    assert routed.master[0, 0] == pytest.approx(0.25)
    assert routed.silenced == ("a",)


def test_soloing_a_bus_silences_the_others():
    mixer = SceneMixer(
        (SourceRouting("a", bus_id="fx"), SourceRouting("b", bus_id="dry")),
        (BusSpec(id="fx", soloed=True), BusSpec(id="dry"), BusSpec()),
        sample_rate_hz=RATE,
    )
    assert mixer.process(_stems()).master[0, 0] == pytest.approx(0.5)


# --- bands ------------------------------------------------------------------


def _noise(frames=8192, seed=0):
    return np.random.default_rng(seed).normal(size=(2, frames)) * 0.3


def test_a_wideband_scene_is_an_exact_bypass():
    """Running audio through a crossover that does nothing still costs phase."""

    signal = _noise()
    mixer = SceneMixer(bands=BandRouting(), sample_rate_hz=RATE)
    assert np.allclose(mixer.process({"a": signal}).master, signal)


def test_disabling_a_band_removes_its_energy():
    signal = _noise()
    crossovers = (300.0, 3000.0)
    full = SceneMixer(
        bands=BandRouting(crossovers_hz=crossovers), sample_rate_hz=RATE
    ).process({"a": signal})
    without = SceneMixer(
        bands=BandRouting(crossovers_hz=crossovers, band_enabled=(True, False, True)),
        sample_rate_hz=RATE,
    ).process({"a": signal})
    assert np.abs(without.master).max() < np.abs(full.master).max()


def test_a_band_gain_changes_the_sum():
    signal = _noise()
    crossovers = (300.0, 3000.0)
    plain = SceneMixer(
        bands=BandRouting(crossovers_hz=crossovers), sample_rate_hz=RATE
    ).process({"a": signal})
    boosted = SceneMixer(
        bands=BandRouting(crossovers_hz=crossovers, band_gains_db=(0.0, 6.0, 0.0)),
        sample_rate_hz=RATE,
    ).process({"a": signal})
    assert not np.allclose(plain.master, boosted.master)


def test_the_disabled_bands_are_reported():
    mixer = SceneMixer(
        bands=BandRouting(crossovers_hz=(300.0,), band_enabled=(False, True)),
        sample_rate_hz=RATE,
    )
    mixer.process({"a": _noise()})
    assert mixer.diagnostics()["disabledBands"] == [0]
    assert mixer.diagnostics()["bandCount"] == 2


@pytest.mark.parametrize("block", [256, 512, 1024, 4096])
def test_band_splitting_carries_its_filter_state_across_blocks(block):
    """Restarting the filters each block would step every band's output."""

    signal = _noise()
    bands = BandRouting(crossovers_hz=(300.0, 3000.0), band_gains_db=(0.0, -6.0, 3.0))

    def run(span):
        mixer = SceneMixer(bands=bands, sample_rate_hz=RATE)
        return np.concatenate(
            [
                mixer.process({"a": signal[:, start : start + span]}).master
                for start in range(0, signal.shape[1], span)
            ],
            axis=1,
        )

    assert np.array_equal(run(block), run(signal.shape[1]))


def test_resetting_forgets_the_filter_history():
    mixer = SceneMixer(bands=BandRouting(crossovers_hz=(300.0,)), sample_rate_hz=RATE)
    signal = _noise(2048)
    first = mixer.process({"a": signal}).master
    mixer.reset()
    assert np.array_equal(mixer.process({"a": signal}).master, first)


# --- built from the plan ----------------------------------------------------


def test_a_mixer_can_be_built_from_a_compiled_plan():
    scene = empty_sam_scene()
    scene["routing"]["buses"] = [{"id": "fx", "gainDb": -6.0}, {"id": "master"}]
    scene["routing"]["sources"] = [{"sourceId": "source.1", "busId": "fx"}]
    scene["routing"]["bands"] = {"crossoversHz": [500.0], "bandEnabled": [True, False]}
    plan = plan_from_track({"global_settings": {"sample_rate": 44100}, "sam_scene": scene, "steps": []})

    mixer = mixer_from_plan(plan)
    assert [bus.id for bus in mixer.buses] == ["fx", "master"]
    assert mixer.routings[0].bus_id == "fx"
    assert mixer.bands.crossovers_hz == (500.0,)
    assert mixer.bands.enabled_for(1) is False


# --- through the production path --------------------------------------------


def _track(**routing):
    scene = empty_sam_scene()
    scene["routing"]["buses"] = routing.get("buses", [{"id": "master", "gainDb": 0.0}])
    scene["routing"]["sources"] = routing.get("sources", [])
    scene["routing"]["bands"] = routing.get("bands", {})
    voices = [
        {
            "synth_function_name": "spatial_angle_modulation_sam2",
            "description": name,
            "params": {"amp": 0.4, "carrierFreq": 200.0, "modFreq": 4.0},
        }
        for name in ("A", "B")
    ]
    return {
        "global_settings": {"sample_rate": 44100},
        "sam_scene": scene,
        "steps": [{"duration": 0.3, "voices": voices}],
    }


def _peak(track):
    return float(np.abs(assemble_track_from_data(track, 44100, 0.0)).max())


def test_two_sources_sum_through_the_production_path():
    assert _peak(_track()) == pytest.approx(0.8, abs=0.02)


def test_a_bus_gain_reaches_the_production_render():
    """The whole point: a bus fader that does nothing is not a bus."""

    routed = _track(
        buses=[{"id": "fx", "gainDb": -12.0}, {"id": "master"}],
        sources=[
            {"sourceId": "source.1", "busId": "fx"},
            {"sourceId": "source.2", "busId": "fx"},
        ],
    )
    assert _peak(routed) == pytest.approx(0.8 * 10.0 ** (-12.0 / 20.0), abs=0.02)


def test_muting_a_bus_silences_the_production_render():
    routed = _track(
        buses=[{"id": "fx", "muted": True}, {"id": "master"}],
        sources=[
            {"sourceId": "source.1", "busId": "fx"},
            {"sourceId": "source.2", "busId": "fx"},
        ],
    )
    assert _peak(routed) == pytest.approx(0.0, abs=1e-6)


def test_solo_isolates_one_source_in_the_production_render():
    assert _peak(_track(sources=[{"sourceId": "source.1", "soloed": True}])) == pytest.approx(
        0.4, abs=0.02
    )


def test_band_settings_reach_the_production_render():
    """A 200 Hz carrier lives in the low band; disabling it should remove it."""

    routed = _track(bands={"crossoversHz": [500.0], "bandEnabled": [False, True]})
    assert _peak(routed) < 0.1


def test_routing_is_applied_exactly_once():
    """Folded into the source envelope *and* applied by the mixer would square it."""

    from src.audio.sam_workbench.scene_state import scene_gain_envelope

    scene = empty_sam_scene()
    scene["routing"]["sources"] = [{"sourceId": "source.1", "gainDb": -6.0}]
    with_routing = scene_gain_envelope(scene, "source.1", 0, 16, 44100.0)
    without = scene_gain_envelope(
        scene, "source.1", 0, 16, 44100.0, include_routing=False
    )
    assert with_routing[0] == pytest.approx(10.0 ** (-6.0 / 20.0))
    assert without[0] == pytest.approx(1.0)

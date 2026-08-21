"""Crossovers, buses, per-band routing, and the render-cost estimate.

The rules this file protects:

* bands sum back to what went in, so a per-band setting is judged on its own
  merits and not on a crossover artefact;
* muting or soloing one source changes that source and nothing else;
* eight sources across four bands is a supported offline scene;
* the cost of a scene is stated before it is paid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.cost import (
    OFFLINE,
    REALTIME,
    CostInputs,
    estimate_cost,
    interpolation_weight,
    measure_throughput,
)
from src.audio.sam_workbench.dsp.crossover import (
    CrossoverBank,
    linkwitz_riley_sections,
    split_bands,
)
from src.audio.sam_workbench.render.routing import (
    MASTER_BUS,
    BandRouting,
    BusSpec,
    SourceRouting,
    band_stems,
    deterministic_seed,
    mix_routed,
)

RATE = 48_000
FRAMES = 1 << 14


def impulse(frames: int = FRAMES) -> np.ndarray:
    """A unit impulse: its spectrum is the transfer function, with no noise."""

    signal = np.zeros(frames)
    signal[0] = 1.0
    return signal


def transfer_db(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / RATE)
    return frequencies, 20.0 * np.log10(np.maximum(spectrum, 1e-30))


# --- crossovers -------------------------------------------------------------


@pytest.mark.parametrize(
    "crossovers",
    [
        (500.0,),
        (300.0, 3000.0),
        (200.0, 1000.0, 5000.0),
        (120.0, 600.0, 2500.0, 9000.0),
    ],
)
def test_bands_sum_back_to_the_input(crossovers):
    """Measured on the transfer function, which is what reconstruction means."""

    bands = split_bands(impulse(), crossovers, RATE)
    frequencies, decibels = transfer_db(bands.sum(axis=0))
    audible = (frequencies > 20.0) & (frequencies < 20_000.0)

    assert bands.shape[0] == len(crossovers) + 1
    assert np.max(np.abs(decibels[audible])) < 1e-6


def test_the_crossover_point_is_six_decibels_down_on_each_side():
    """The Linkwitz-Riley property that makes the halves sum flat."""

    bands = split_bands(impulse(), (1000.0,), RATE)
    frequencies, _ = transfer_db(bands[0])
    at_crossover = int(np.argmin(np.abs(frequencies - 1000.0)))

    for band in (0, 1):
        _, decibels = transfer_db(bands[band])
        assert decibels[at_crossover] == pytest.approx(-6.0, abs=0.2)


def test_each_band_rejects_what_belongs_to_the_others():
    bank = CrossoverBank((200.0, 1000.0, 5000.0), RATE)
    bands = bank.split(impulse())
    frequencies, _ = transfer_db(bands[0])

    def level_at(band: int, hertz: float) -> float:
        _, decibels = transfer_db(bands[band])
        return float(decibels[int(np.argmin(np.abs(frequencies - hertz)))])

    assert level_at(0, 50.0) > -1.0  # the low band passes the low end
    assert level_at(0, 15_000.0) < -60.0  # and rejects the top
    assert level_at(3, 15_000.0) > -1.0  # the high band does the reverse
    assert level_at(3, 50.0) < -60.0


def test_an_odd_order_is_rejected():
    with pytest.raises(ValueError):
        linkwitz_riley_sections(1000.0, RATE, order=3)


def test_crossovers_must_increase_and_stay_below_nyquist():
    with pytest.raises(ValueError):
        CrossoverBank((3000.0, 300.0), RATE)
    with pytest.raises(ValueError):
        CrossoverBank((30_000.0,), RATE)
    with pytest.raises(ValueError):
        CrossoverBank((0.0,), RATE)


def test_band_edges_describe_the_split():
    bank = CrossoverBank((200.0, 1000.0), RATE)
    assert bank.band_count == 3
    assert bank.band_edges_hz() == ((0.0, 200.0), (200.0, 1000.0), (1000.0, RATE / 2))


# --- per-band routing -------------------------------------------------------


def test_a_wideband_routing_does_not_filter_anything():
    """No crossover means no phase shift, not a crossover set to pass."""

    audio = np.random.default_rng(0).normal(0.0, 0.1, (2, 1024))
    stems = band_stems(audio, BandRouting(), RATE)

    assert stems.shape == (1, 2, 1024)
    assert stems[0] == pytest.approx(audio)


def test_band_gains_and_disables_apply():
    routing = BandRouting(
        crossovers_hz=(200.0, 1000.0, 5000.0),
        band_gains_db=(0.0, -6.0, 0.0, 0.0),
        band_enabled=(True, True, False, True),
    )
    stereo = np.zeros((2, FRAMES))
    stereo[:, 0] = 1.0
    stems = band_stems(stereo, routing, RATE)

    assert stems.shape[0] == 4
    assert np.all(stems[2] == 0.0)  # disabled

    plain = band_stems(stereo, BandRouting(crossovers_hz=routing.crossovers_hz), RATE)
    ratio = np.max(np.abs(stems[1])) / np.max(np.abs(plain[1]))
    assert ratio == pytest.approx(10 ** (-6.0 / 20.0), rel=1e-6)


# --- buses and stems --------------------------------------------------------


@pytest.fixture
def stems():
    rng = np.random.default_rng(0)
    return {f"src{index}": rng.normal(0.0, 0.1, (2, 2048)) for index in range(8)}


def test_eight_sources_mix_to_a_master(stems):
    """Exit criterion: eight sources are a supported scene."""

    scene = mix_routed(stems, [])

    assert len(scene.source_stems) == 8
    assert scene.master == pytest.approx(sum(stems.values()))
    assert scene.silenced == ()


def test_muting_one_source_leaves_every_other_untouched(stems):
    reference = mix_routed(stems, [])
    muted = mix_routed(stems, [SourceRouting("src3", muted=True)])

    assert muted.silenced == ("src3",)
    assert np.all(muted.source_stems["src3"] == 0.0)
    for name in stems:
        if name != "src3":
            assert muted.source_stems[name] == pytest.approx(reference.source_stems[name])


def test_solo_silences_everything_else(stems):
    scene = mix_routed(stems, [SourceRouting("src5", soloed=True)])

    assert len(scene.silenced) == 7
    assert scene.master == pytest.approx(stems["src5"])


def test_mute_beats_solo_on_the_same_source(stems):
    """Muting something is the more specific instruction."""

    scene = mix_routed(stems, [SourceRouting("src2", soloed=True, muted=True)])
    assert "src2" in scene.silenced


def test_buses_sum_their_own_sources_and_then_the_master(stems):
    routings = [
        SourceRouting(f"src{index}", bus_id="a" if index < 4 else "b") for index in range(8)
    ]
    buses = [BusSpec(id="a", name="A", gain_db=-6.0), BusSpec(id="b", name="B")]
    scene = mix_routed(stems, routings, buses)

    expected_a = sum(stems[f"src{index}"] for index in range(4)) * 10 ** (-6.0 / 20.0)
    assert scene.bus_stems["a"] == pytest.approx(expected_a)
    assert scene.master == pytest.approx(scene.bus_stems["a"] + scene.bus_stems["b"])


def test_muting_a_bus_silences_the_sources_on_it(stems):
    routings = [SourceRouting(f"src{index}", bus_id="a") for index in range(3)]
    scene = mix_routed(stems, routings, [BusSpec(id="a", muted=True)])

    assert set(scene.silenced) == {"src0", "src1", "src2"}


def test_a_source_without_a_routing_still_reaches_the_master(stems):
    scene = mix_routed(stems, [SourceRouting("src0", gain_db=-6.0)])

    assert scene.source_stems["src7"] == pytest.approx(stems["src7"])
    assert scene.bus_stems[MASTER_BUS].shape == (2, 2048)


def test_levels_are_reported_for_the_master_and_every_stem(stems):
    scene = mix_routed(stems, [])
    levels = scene.levels()

    assert levels["master"] > 0.0
    assert levels["source:src0"] > 0.0


# --- determinism ------------------------------------------------------------


def test_a_seed_depends_on_identity_not_on_position():
    """Disabling one source must not shift another's random stream."""

    assert deterministic_seed("src3", "noise") == deterministic_seed("src3", "noise")
    assert deterministic_seed("src3", "noise") != deterministic_seed("src4", "noise")
    assert deterministic_seed("src3", "noise") != deterministic_seed("src3", "grains")
    assert 0 <= deterministic_seed("anything") <= 0x7FFF_FFFF


def test_the_same_scene_mixes_identically_twice(stems):
    first = mix_routed(stems, [SourceRouting("src1", gain_db=-3.0)])
    second = mix_routed(stems, [SourceRouting("src1", gain_db=-3.0)])

    assert first.master == pytest.approx(second.master, abs=0.0)


# --- cost -------------------------------------------------------------------


def test_cost_grows_with_sources_bands_taps_and_interpolation():
    baseline = estimate_cost(CostInputs(sources=1, bands=1))

    assert estimate_cost(CostInputs(sources=8, bands=1)).load > baseline.load
    assert estimate_cost(CostInputs(sources=1, bands=4)).load > baseline.load
    assert estimate_cost(CostInputs(sources=1, bands=1, hrir_taps=2048)).load > baseline.load
    assert (
        estimate_cost(CostInputs(interpolation="spherical_harmonic")).load
        > estimate_cost(CostInputs(interpolation="nearest")).load
    )


def test_a_renderer_that_does_not_convolve_costs_far_less():
    convolving = estimate_cost(CostInputs(sources=8, renderer="hrtf"))
    abstract = estimate_cost(CostInputs(sources=8, renderer="abstract_pm"))

    assert abstract.load < convolving.load / 10.0


def test_an_expensive_scene_falls_back_to_offline():
    modest = estimate_cost(CostInputs(sources=2, bands=1), throughput_macs=1e9)
    heavy = estimate_cost(
        CostInputs(sources=64, bands=6, interpolation="spherical_harmonic", hrir_taps=4096),
        throughput_macs=1e9,
    )

    assert modest.mode == REALTIME
    assert modest.feasible_realtime
    assert heavy.mode == OFFLINE
    assert not heavy.feasible_realtime
    assert "Offline" in heavy.summary()


def test_eight_sources_and_four_bands_are_supported(stems):
    """Exit criterion, stated as the estimate the workbench would show."""

    estimate = estimate_cost(CostInputs(sources=8, bands=4), throughput_macs=1e9)

    assert estimate.inputs.convolutions == 8 * 4 * 2
    assert estimate.macs_per_second > 0.0
    assert "8 source(s) x 4 band(s)" in estimate.summary()


def test_an_unmeasured_estimate_says_so():
    assert estimate_cost(CostInputs()).measured is False
    assert estimate_cost(CostInputs(), throughput_macs=1e9).measured is True


def test_the_block_budget_follows_the_block_size():
    estimate = estimate_cost(CostInputs(block_size=512, sample_rate_hz=48_000))
    assert estimate.block_budget_s == pytest.approx(512 / 48_000)


def test_throughput_can_be_measured_from_a_real_convolution():
    throughput = measure_throughput(taps=64, frames=1 << 12, repeats=1)
    assert throughput > 0.0
    assert np.isfinite(throughput)


def test_an_unknown_interpolation_is_costed_at_the_maximum():
    """An unrecognised mode must not be assumed cheap."""

    assert interpolation_weight("something-new") >= interpolation_weight("spherical_harmonic")


def test_impossible_inputs_are_rejected():
    with pytest.raises(ValueError):
        CostInputs(sources=0)
    with pytest.raises(ValueError):
        CostInputs(sample_rate_hz=0.0)
    with pytest.raises(ValueError):
        estimate_cost(CostInputs(), headroom=0.0)
    with pytest.raises(ValueError):
        estimate_cost(CostInputs(), throughput_macs=-1.0)


# --- the scene the exit criteria name ---------------------------------------


def test_eight_sources_across_four_bands_render_and_mix_end_to_end():
    """Phase 6's exit criterion, rendered rather than predicted.

    Eight sources, each coupled around a leader, each split into four bands,
    routed through two buses into a master.
    """

    from src.audio.sam_workbench.hrtf.sofa_io import load_sofa
    from src.audio.sam_workbench.render.hybrid import HybridSpec, render_hybrid
    from src.audio.sam_workbench.trajectory.coupling import CouplingSpec, couple_positions

    dataset = load_sofa("tests/sam_workbench/fixtures/synthetic_sonicom_hrir.sofa")
    rate = int(dataset.sample_rate_hz)
    frames = rate // 4
    routing = BandRouting(crossovers_hz=(200.0, 1000.0, 5000.0))
    leader = np.array([1.0, 0.0, 0.0])

    stems = {}
    for index in range(8):
        rng = np.random.default_rng(deterministic_seed(f"src{index}"))
        mono = rng.normal(0.0, 0.05, frames)
        position = couple_positions(
            CouplingSpec("offset", "lead", angle_deg=index * 45.0), leader
        )
        stereo = render_hybrid(mono, position, dataset, rate, HybridSpec()).audio
        stems[f"src{index}"] = band_stems(stereo, routing, rate).sum(axis=0)

    routings = [
        SourceRouting(f"src{index}", bus_id="low" if index < 4 else "high")
        for index in range(8)
    ]
    buses = [BusSpec(id="low"), BusSpec(id="high", gain_db=-3.0)]
    scene = mix_routed(stems, routings, buses)

    assert routing.band_count == 4
    assert len(scene.source_stems) == 8
    assert scene.master.shape == (2, frames)
    assert np.all(np.isfinite(scene.master))
    assert np.max(np.abs(scene.master)) > 0.0
    assert scene.master == pytest.approx(sum(scene.bus_stems.values()))

    # Deterministic: the same scene twice, and muting one source disturbs
    # nothing else in it.
    again = mix_routed(stems, routings, buses)
    assert np.array_equal(scene.master, again.master)

    muted_routings = [
        SourceRouting("src3", bus_id="low", muted=True) if r.source_id == "src3" else r
        for r in routings
    ]
    muted = mix_routed(stems, muted_routings, buses)
    for name in stems:
        if name != "src3":
            assert np.array_equal(muted.source_stems[name], scene.source_stems[name])

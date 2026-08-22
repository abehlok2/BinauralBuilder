"""Scene automation reaches the audio, and does not depend on block boundaries.

The defect this file guards against was audible rather than theoretical. Every
non-gain parameter was resolved once, at the start of whichever chunk was being
rendered, and held for that chunk's length. The same second of audio therefore
differed by more than half of full scale depending on how the caller had
divided the timeline, and a preview could not be trusted to match its export.

Two things have to hold at once, and they pull in opposite directions:

* a voice whose scene automates nothing must be exactly what it always was -
  bit for bit, because the established SAM2 behaviour is not up for
  renegotiation;
* a voice whose scene automates something must sound the same however the
  render was partitioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.automation import AutomatedPhase
from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.modulation import ModulationMatrix, ModulationRoute
from src.audio.sam_workbench.scene_state import empty_sam_scene
from src.audio.sam_workbench.stages import ParameterBinding, StageConfig, Timeline

RATE = 44100.0
PARAMS = {"amp": 0.5, "carrierFreq": 300.0, "modFreq": 4.0}
#: float32 output quantization. Agreement below this is agreement.
AUDIO_EPS = 1.2e-7


def _scene(*paths, modulated=()):
    scene = empty_sam_scene()
    scene["sources"] = [{"id": "source.1", "name": "Voice"}]
    scene["stages"] = Timeline(
        stages=(
            StageConfig(
                id="st",
                start_s=0.0,
                duration_s=2.0,
                transition_in_s=1.5,
                parameter_overrides=tuple(
                    ParameterBinding(target_id="source.1", parameter_path=path, value=value)
                    for path, value in paths
                ),
            ),
        )
    ).describe()
    if modulated:
        scene["modulators"] = [{"id": "lfo", "rateHz": 1.3}]
        scene["modulation"] = ModulationMatrix(
            routes=tuple(
                ModulationRoute(
                    modulator_id="lfo", target_id="source.1", parameter_path=path, depth=depth
                )
                for path, depth in modulated
            )
        ).describe()
    return scene


def _render(chunks, scene=None):
    return np.concatenate(
        [
            render_sam2_voice(
                duration,
                RATE,
                params=PARAMS,
                initial_offset=start,
                sam_scene=scene,
                source_id="source.1",
                scene_start_s=start,
            )
            for start, duration in chunks
        ]
    )


PARTITIONS = {
    "halves": [(0.0, 0.5), (0.5, 0.5)],
    "quarters": [(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)],
    "awkward": [(0.0, 0.13), (0.13, 0.37), (0.5, 0.11), (0.61, 0.39)],
}


# --- the established behaviour is untouched ---------------------------------


def test_a_voice_with_no_scene_is_bit_identical_across_chunks():
    whole = _render([(0.0, 0.5)])
    halves = _render([(0.0, 0.25), (0.25, 0.25)])
    assert np.array_equal(whole, halves)


def test_a_scene_that_automates_nothing_leaves_the_voice_bit_identical():
    """Attaching an empty scene must not perturb the audio at all."""

    assert np.array_equal(_render([(0.0, 0.5)]), _render([(0.0, 0.5)], empty_sam_scene()))


def test_a_scene_that_only_automates_gain_keeps_the_closed_form_phase():
    scene = _scene(("amp", 0.4))
    whole = _render([(0.0, 0.5)], scene)
    halves = _render([(0.0, 0.25), (0.25, 0.25)], scene)
    assert np.array_equal(whole, halves)


# --- automated parameters reach the audio -----------------------------------


@pytest.mark.parametrize(
    "scene_builder, label",
    [
        (lambda: _scene(("modFreq", 12.0)), "modulation rate"),
        (lambda: _scene(("arcWidthDeg", 150.0)), "arc width"),
        (lambda: _scene(("directionOffsetDeg", 60.0)), "direction offset"),
        (lambda: _scene(modulated=(("carrierFreq", 40.0),)), "carrier"),
    ],
)
def test_automating_a_parameter_changes_the_sound(scene_builder, label):
    """A control that reaches nothing is worse than no control at all."""

    plain = _render([(0.0, 0.5)])
    automated = _render([(0.0, 0.5)], scene_builder())
    assert not np.allclose(plain, automated, atol=1e-4), label


# --- and do not depend on where the timeline was cut ------------------------


@pytest.mark.parametrize("partition", sorted(PARTITIONS))
def test_an_automated_render_does_not_depend_on_block_partitioning(partition):
    scene = _scene(
        ("modFreq", 12.0), ("arcWidthDeg", 150.0), modulated=(("carrierFreq", 40.0),)
    )
    whole = _render([(0.0, 1.0)], scene)
    parts = _render(PARTITIONS[partition], scene)
    assert np.abs(whole - parts).max() < AUDIO_EPS


@pytest.mark.parametrize("partition", sorted(PARTITIONS))
def test_the_gain_envelope_does_not_depend_on_block_partitioning(partition):
    scene = _scene(("amp", 0.4))
    whole = _render([(0.0, 1.0)], scene)
    parts = _render(PARTITIONS[partition], scene)
    assert np.abs(whole - parts).max() < AUDIO_EPS


def test_rendering_out_of_order_matches_rendering_forwards():
    """Blocks may be requested in any order; a seek is not a different render."""

    scene = _scene(("modFreq", 12.0))
    whole = _render([(0.0, 1.0)], scene)
    tail = _render([(0.5, 0.5)], scene)
    head = _render([(0.0, 0.5)], scene)
    assert np.abs(np.concatenate([head, tail]) - whole).max() < AUDIO_EPS


# --- the phase integrator ---------------------------------------------------


def test_a_constant_frequency_integrates_to_the_closed_form():
    phase = AutomatedPhase(lambda start, frames: np.full(frames, 7.0), RATE)
    index = np.arange(10000)
    assert phase.at(0, 10000) == pytest.approx(2.0 * np.pi * 7.0 * index / RATE, abs=1e-9)


def test_a_linear_sweep_integrates_to_its_closed_form():
    """Phase is the integral of frequency, not frequency times time."""

    start_hz, slope = 4.0, 3.0

    def frequency(start, frames):
        return start_hz + slope * (np.arange(start, start + frames) / RATE)

    times = np.arange(10000) / RATE
    expected = 2.0 * np.pi * (start_hz * times + 0.5 * slope * times * times)
    assert AutomatedPhase(frequency, RATE).at(0, 10000) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "cuts", [[0, 10000], [0, 5000, 10000], [0, 37, 1234, 9999, 10000]],
    ids=["whole", "halves", "awkward"],
)
def test_the_integrator_agrees_across_partitions(cuts):
    def frequency(start, frames):
        return 4.0 + 3.0 * (np.arange(start, start + frames) / RATE)

    reference = AutomatedPhase(frequency, RATE).at(0, 10000)
    integrator = AutomatedPhase(frequency, RATE)
    joined = np.concatenate(
        [integrator.at(cuts[i], cuts[i + 1] - cuts[i]) for i in range(len(cuts) - 1)]
    )
    # Summing the same increments in a different order rounds differently; the
    # disagreement is far below anything audible.
    assert np.abs(joined - reference).max() < 1e-9


def test_the_integrator_gives_the_same_answer_however_it_is_reached():
    """The cache is memoization, not state: order of access cannot matter."""

    def frequency(start, frames):
        return 5.0 + np.sin(np.arange(start, start + frames) / RATE)

    forwards = AutomatedPhase(frequency, RATE)
    backwards = AutomatedPhase(frequency, RATE)
    late = backwards.at(8000, 2000)
    backwards.at(0, 8000)
    assert np.abs(late - forwards.at(8000, 2000)).max() < 1e-12


def test_the_integrator_refuses_a_negative_start():
    phase = AutomatedPhase(lambda start, frames: np.ones(frames), RATE)
    with pytest.raises(ValueError, match="from sample zero"):
        phase.at(-1, 10)


def test_an_empty_window_integrates_to_nothing():
    phase = AutomatedPhase(lambda start, frames: np.ones(frames), RATE)
    assert phase.at(0, 0).shape == (0,)

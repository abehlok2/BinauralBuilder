"""The frozen example projects, and the metrics their renders must keep hitting.

Section 16.4 is explicit that audio regression should compare *metrics* rather
than whole waveforms, and reserve golden files for version-pinned algorithms.
So the frozen fixture here records what each example sounds like in terms that
survive a numpy upgrade - level, spectrum, interaural correlation, zero
crossings - and each is checked against a tolerance chosen to be far tighter
than any change worth noticing and far looser than the last bit of a float.

The examples themselves are frozen too: their identifiers and timestamps are
fixed in the files, so a render of an example is reproducible by anyone who
checks out the repository rather than dependent on when they ran it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.model import load_project, save_project
from src.audio.sam_workbench.render import render_project

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
FROZEN = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "example_render_metrics.json").read_text(
        encoding="utf-8"
    )
)

#: How far each metric may move before it is a regression rather than round-off.
TOLERANCES = {
    "frames": 0,
    "peak": 1e-6,
    "rmsLeftDb": 1e-4,
    "rmsRightDb": 1e-4,
    "interauralCorrelation": 1e-6,
    "spectralCentroidHz": 1e-3,
    "peakBinHz": 0.0,
    "zeroCrossingsLeft": 0,
}

EXAMPLE_NAMES = sorted(FROZEN["examples"])


def _metrics(audio: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    """The same measurements the fixture was frozen from."""

    left, right = audio[0], audio[1]
    spectrum = np.abs(np.fft.rfft(left * np.hanning(left.size)))
    frequencies = np.fft.rfftfreq(left.size, 1.0 / sample_rate_hz)
    correlation = (
        float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else 1.0
    )
    return {
        "frames": int(audio.shape[1]),
        "peak": float(np.max(np.abs(audio))),
        "rmsLeftDb": float(20.0 * np.log10(max(float(np.sqrt(np.mean(left**2))), 1e-12))),
        "rmsRightDb": float(20.0 * np.log10(max(float(np.sqrt(np.mean(right**2))), 1e-12))),
        "interauralCorrelation": correlation,
        "spectralCentroidHz": float(
            np.sum(frequencies * spectrum) / max(float(np.sum(spectrum)), 1e-30)
        ),
        "peakBinHz": float(frequencies[int(np.argmax(spectrum))]),
        "zeroCrossingsLeft": int(np.count_nonzero(np.diff(np.signbit(left)))),
    }


def test_every_example_is_covered_by_the_frozen_fixture():
    """A new example without a frozen render is an untested example."""

    on_disk = {path.stem for path in EXAMPLES_DIR.glob("*.json")}
    assert on_disk == set(EXAMPLE_NAMES)
    assert on_disk, "the examples directory must not be empty"


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_an_example_loads_and_validates(name):
    project = load_project(EXAMPLES_DIR / f"{name}.json")
    assert project.sources
    assert project.name
    # Examples are the first thing a new user opens; none of them may be a
    # project the application would refuse.
    project.validate()


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_an_example_renders_to_its_frozen_metrics(name):
    project = load_project(EXAMPLES_DIR / f"{name}.json")
    audio, _report = render_project(project, FROZEN["renderSeconds"])
    measured = _metrics(audio, project.audio.sample_rate_hz)

    expected = FROZEN["examples"][name]
    assert set(measured) == set(expected)
    for key, value in expected.items():
        assert measured[key] == pytest.approx(value, abs=TOLERANCES[key]), key


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_an_example_is_frozen_in_time(name):
    """Fixed identifiers and timestamps, so the render is reproducible."""

    data = json.loads((EXAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert data["id"].startswith("example-")
    assert data["created_utc"] == data["modified_utc"]
    assert data["created_utc"].endswith("Z")


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_an_example_survives_being_opened_and_saved(name, tmp_path):
    """Opening an example in the application must not rewrite what it means."""

    source = EXAMPLES_DIR / f"{name}.json"
    destination = tmp_path / f"{name}.json"
    save_project(load_project(source), destination)

    before = json.loads(source.read_text(encoding="utf-8"))
    after = json.loads(destination.read_text(encoding="utf-8"))
    # The modification stamp is expected to move; nothing else may.
    before.pop("modified_utc")
    after.pop("modified_utc")
    assert after == before


def test_examples_cover_more_than_one_sample_rate_and_source_count():
    """A set of examples that are all the same shape tests one thing, not three."""

    projects = [load_project(EXAMPLES_DIR / f"{name}.json") for name in EXAMPLE_NAMES]
    assert len({project.audio.sample_rate_hz for project in projects}) > 1
    assert len({len(project.sources) for project in projects}) > 1
    assert any(source.start_s > 0.0 for project in projects for source in project.sources)

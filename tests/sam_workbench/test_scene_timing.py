"""A source renders within its own absolute interval, and nowhere else.

A render window and a source's lifetime are two independent intervals on the
same absolute timeline. The scene's job is to intersect them. Getting one end
right and not the other is the failure this file exists to prevent: clamping a
source's length to its duration without first subtracting the part that had
already elapsed let a late window run past the source's end, and let a window
opening after the source had finished render it from its own beginning as
though it were only then starting.

Every test states an absolute interval and asserts on where audio actually
appears, because that is the thing that was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.model import Project, Source
from src.audio.sam_workbench.render.scene import render_project

#: The source under test is alive from 1.0 s to 2.0 s absolute.
START_S = 1.0
DURATION_S = 1.0


def _project(start_s=START_S, duration_s=DURATION_S, **extra):
    return Project(sources=(Source(id="s", start_s=start_s, duration_s=duration_s, **extra),))


def _sounding_interval(project, window_start_s, window_duration_s):
    """The absolute seconds over which the render is non-silent, or ``None``."""

    rate = project.audio.sample_rate_hz
    window_start = int(round(window_start_s * rate))
    audio, _ = render_project(
        project,
        frames=int(round(window_duration_s * rate)),
        start_sample=window_start,
        apply_limiter=False,
    )
    sounding = np.nonzero(np.abs(audio).max(axis=0) > 1e-12)[0]
    if not len(sounding):
        return None
    return ((window_start + sounding[0]) / rate, (window_start + sounding[-1] + 1) / rate)


# --- the five window positions ----------------------------------------------


def test_a_window_entirely_before_the_source_is_silent():
    assert _sounding_interval(_project(), 0.0, 0.5) is None


def test_a_window_overlapping_the_beginning_starts_at_the_source():
    begin, end = _sounding_interval(_project(), 0.5, 1.0)
    assert begin == pytest.approx(START_S, abs=1e-4)
    assert end == pytest.approx(1.5, abs=1e-4)


def test_a_window_inside_the_source_is_filled_throughout():
    begin, end = _sounding_interval(_project(), 1.2, 0.3)
    assert begin == pytest.approx(1.2, abs=1e-4)
    assert end == pytest.approx(1.5, abs=1e-4)


def test_a_window_overlapping_the_end_stops_at_the_source_end():
    """The regression: this used to run a further half second past 2.0 s."""

    begin, end = _sounding_interval(_project(), 1.5, 1.0)
    assert begin == pytest.approx(1.5, abs=1e-4)
    assert end == pytest.approx(START_S + DURATION_S, abs=1e-4)


def test_a_window_entirely_after_the_source_is_silent():
    """The regression: this used to render the source from its own beginning."""

    assert _sounding_interval(_project(), 3.0, 0.5) is None


def test_a_window_enclosing_the_source_bounds_it_on_both_sides():
    begin, end = _sounding_interval(_project(), 0.0, 3.0)
    assert begin == pytest.approx(START_S, abs=1e-4)
    assert end == pytest.approx(START_S + DURATION_S, abs=1e-4)


# --- with and without an explicit duration ----------------------------------


def test_a_source_without_a_duration_runs_to_the_end_of_any_window():
    begin, end = _sounding_interval(_project(duration_s=None), 0.0, 3.0)
    assert begin == pytest.approx(START_S, abs=1e-4)
    assert end == pytest.approx(3.0, abs=1e-4)


def test_a_source_without_a_duration_still_respects_its_start():
    assert _sounding_interval(_project(duration_s=None), 0.0, 0.5) is None


def test_a_source_without_a_duration_fills_a_window_that_opens_after_its_start():
    begin, end = _sounding_interval(_project(duration_s=None), 5.0, 0.5)
    assert begin == pytest.approx(5.0, abs=1e-4)
    assert end == pytest.approx(5.5, abs=1e-4)


def test_a_zero_overlap_window_at_the_exact_source_end_is_silent():
    """The source's last sample is the one before its end, not at it."""

    assert _sounding_interval(_project(), START_S + DURATION_S, 0.25) is None


# --- the report describes the same interval ---------------------------------


def test_the_report_counts_only_the_frames_that_were_rendered():
    project = _project()
    rate = project.audio.sample_rate_hz
    _, report = render_project(
        project, frames=int(rate), start_sample=int(1.5 * rate), apply_limiter=False
    )
    # Half a second of overlap, not the source's whole second.
    assert [entry.frames for entry in report.sources] == [int(0.5 * rate)]


def test_a_source_outside_the_window_is_absent_from_the_report():
    project = _project()
    rate = project.audio.sample_rate_hz
    _, report = render_project(
        project, frames=rate // 2, start_sample=3 * rate, apply_limiter=False
    )
    assert report.sources == ()


# --- partition invariance ---------------------------------------------------


def _partitions(total):
    """Whole render, halves, an awkward split, and a regular block grid."""

    return [
        [0, total],
        [0, total // 2, total],
        [0, 1000, 5077, total // 3, total - 13, total],
        [0, *range(4096, total, 4096), total],
    ]


@pytest.mark.parametrize("index", range(4))
def test_arbitrary_window_partitions_reassemble_into_the_whole_render(index):
    """Where the caller cuts the timeline must not change a single sample."""

    project = Project(
        sources=(
            Source(id="a", start_s=0.3, duration_s=0.9),
            Source(id="b", start_s=0.7, duration_s=None),
        )
    )
    total = int(2.0 * project.audio.sample_rate_hz)
    whole, _ = render_project(project, frames=total, start_sample=0, apply_limiter=False)

    cuts = sorted(set(_partitions(total)[index]))
    joined = np.concatenate(
        [
            render_project(
                project,
                frames=cuts[position + 1] - cuts[position],
                start_sample=cuts[position],
                apply_limiter=False,
            )[0]
            for position in range(len(cuts) - 1)
        ],
        axis=1,
    )
    assert np.array_equal(joined, whole)


@pytest.mark.parametrize("block_size", [128, 512, 4096, 88200])
def test_the_block_size_does_not_change_the_render(block_size):
    project = Project(
        sources=(
            Source(id="a", start_s=0.3, duration_s=0.9),
            Source(id="b", start_s=0.7, duration_s=None),
        )
    )
    total = int(2.0 * project.audio.sample_rate_hz)
    reference, _ = render_project(project, frames=total, apply_limiter=False)
    audio, _ = render_project(
        project, frames=total, block_size=block_size, apply_limiter=False
    )
    assert np.array_equal(audio, reference)


def test_a_late_window_continues_the_source_rather_than_restarting_it():
    """Phase must carry on from where the source is, not reset to its opening."""

    project = _project(duration_s=None)
    rate = project.audio.sample_rate_hz
    whole, _ = render_project(project, frames=3 * rate, apply_limiter=False)
    tail, _ = render_project(
        project, frames=rate, start_sample=2 * rate, apply_limiter=False
    )
    assert np.array_equal(tail, whole[:, 2 * rate : 3 * rate])

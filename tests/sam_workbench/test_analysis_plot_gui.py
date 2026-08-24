"""The analysis plots must be readable, not merely drawn.

The rules this file protects:

* ticks land on round numbers inside the plotted range, and their labels
  carry exactly the precision the tick spacing justifies;
* margins are computed from the labels themselves, so text is never
  squeezed against the frame;
* two named series produce a legend; one does not;
* a preview buffer populates all four views and everything still paints.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt", reason="GUI tests need pytest-qt")

from src.audio.sam_workbench.analysis import peak_envelope
from src.ui.sam_analysis_panel import (
    PlotSeries,
    PlotWidget,
    SamAnalysisPanel,
    format_tick_value,
    nice_ticks,
)


# --- ticks -------------------------------------------------------------------


def test_ticks_land_on_round_numbers_inside_the_range():
    ticks = nice_ticks(0.0, 10.0, target=5)
    assert np.allclose(ticks, [0.0, 2.5, 5.0, 7.5, 10.0])
    assert ticks[0] >= 0.0 and ticks[-1] <= 10.0


def test_ticks_stay_inside_the_plotted_range():
    low, high = -96.0, 6.0
    ticks = nice_ticks(low, high)
    assert ticks[0] >= low - 1e-9
    assert ticks[-1] <= high + 1e-9


def test_tick_count_stays_readable_at_any_span():
    for span in (0.01, 1.0, 6.283185307179586, 1000.0, 96_000.0):
        ticks = nice_ticks(-span / 2.0, span / 2.0)
        assert 3 <= len(ticks) <= 9, span
        steps = np.diff(ticks)
        assert np.allclose(steps, steps[0])


def test_degenerate_ranges_do_not_raise():
    assert len(nice_ticks(1.0, 1.0)) == 2
    assert len(nice_ticks(0.0, float("nan"))) == 2


def test_label_precision_matches_the_step():
    assert format_tick_value(2500.0, 1000.0) == "2500"
    assert format_tick_value(0.5, 0.25) == "0.50"
    assert format_tick_value(-3.0, 2.0) == "-3"
    # Tiny and huge values fall back to significant digits.
    assert format_tick_value(0.00002, 0.00001) == "2e-05"
    assert format_tick_value(120_000.0, 50_000.0) == "1.2e+05"


# --- layout ------------------------------------------------------------------


def test_a_y_axis_label_widens_the_left_margin(qtbot):
    plain = PlotWidget("t")
    labelled = PlotWidget("t", y_label="amplitude")
    for widget in (plain, labelled):
        qtbot.addWidget(widget)
        widget.resize(320, 220)
        x = np.linspace(0.0, 1.0, 64)
        widget.set_series([PlotSeries(x, np.sin(2 * np.pi * x))])
        widget.grab()  # forces a paint

    assert labelled.last_plot_rect.left() > plain.last_plot_rect.left()
    # The data area stays inside the widget either way.
    for widget in (plain, labelled):
        rect = widget.last_plot_rect
        assert 0 < rect.left() < rect.right() <= widget.width()
        assert 0 < rect.top() < rect.bottom() <= widget.height()


def test_two_named_series_make_a_legend_and_one_does_not(qtbot):
    widget = PlotWidget("t")
    qtbot.addWidget(widget)
    x = np.linspace(0.0, 1.0, 32)

    widget.set_series([PlotSeries(x, np.sin(x), name="left")])
    assert len(widget._legend_items()) == 1
    widget.set_series(
        [
            PlotSeries(x, np.sin(x), name="left"),
            PlotSeries(x, np.cos(x), name="right"),
        ]
    )
    items = widget._legend_items()
    assert [name for name, _color in items] == ["left", "right"]
    widget.grab()


# --- the panel end to end ----------------------------------------------------


def _tone(frames=8000, rate=44_100):
    t = np.arange(frames) / rate
    left = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 220.0 * t + np.pi * 0.25)
    return np.stack((left, right), axis=1)


def test_a_preview_populates_all_four_views_with_series(qtbot):
    panel = SamAnalysisPanel()
    qtbot.addWidget(panel)

    panel.set_audio(_tone(), 44_100)
    for plot in (
        panel.waveform_plot,
        panel.spectrum_plot,
        panel.frequency_plot,
        panel.ipd_plot,
    ):
        assert plot.series, plot._title
        plot.grab()
    assert "Peak" in panel.summary_label.text()

    panel.clear()
    assert not panel.waveform_plot.series


def test_waveform_envelope_band_is_drawn_from_real_envelope_data(qtbot):
    panel = SamAnalysisPanel()
    qtbot.addWidget(panel.waveform_plot)

    block = _tone()
    panel.set_audio(block, 44_100)
    series = panel.waveform_plot.series
    assert len(series) == 2
    minimum, maximum = peak_envelope(block, panel.PLOT_POINTS)
    np.testing.assert_allclose(series[0].y_low, minimum[0], rtol=1e-9)

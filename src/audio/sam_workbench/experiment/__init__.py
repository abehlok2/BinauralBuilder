"""Listening experiments: choosing an HRTF, and recording why.

Two things live here, and they answer different questions.

:mod:`.localization` asks *where did you hear it* - an objective task with a
right answer, scored by how far the report fell from the truth and how often
front was confused with back. That is the specification's Route A, the
lowest-cost personalisation method, and the one that should precede any
attempt at measurement.

:mod:`.session` asks *which did you prefer*, and is the machinery around any
such comparison: building conditions, randomising their order, capturing
responses, and exporting the result with enough provenance that someone else
could say what was actually played.
"""

from __future__ import annotations

__all__ = ["localization", "session"]

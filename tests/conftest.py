"""Test-suite-wide setup.

Qt needs a platform plugin chosen before any widget is created. Selecting the
offscreen plugin here - and only when the environment has not already chosen
one - lets the GUI tests run on a machine with no display, without changing how
they behave for a developer running them locally with a real one.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

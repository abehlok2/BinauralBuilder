"""Test-suite-wide setup.

Qt needs a platform plugin chosen before any widget is created. Selecting the
offscreen plugin here - and only when the environment has not already chosen
one - lets the GUI tests run on a machine with no display, without changing how
they behave for a developer running them locally with a real one.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def isolate_qsettings(tmp_path_factory, monkeypatch):
    """Keep the suite out of the developer's real preferences.

    The workbench remembers its disclosure mode in QSettings, which resolves to
    a file under the user's config directory. Without this, running the tests
    rewrites whatever the person at this machine had chosen - and, worse, tests
    read that leftover state, so a test asserting a first-use default can pass
    because an earlier test happened to store the value it wanted.

    Pointing the user scope at a per-session temporary directory makes every
    run start from genuinely unset settings.
    """

    from PyQt5.QtCore import QSettings

    root = str(tmp_path_factory.mktemp("qsettings"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    for fmt in (QSettings.IniFormat, QSettings.NativeFormat):
        QSettings.setPath(fmt, QSettings.UserScope, root)
        QSettings.setPath(fmt, QSettings.SystemScope, root)
    yield

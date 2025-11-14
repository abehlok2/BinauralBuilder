import pytest

try:
    from PyQt5.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyQt5 not available: {exc}", allow_module_level=True)

from src.audio.session_model import Session, SessionPresetChoice, SessionStep
from src.ui.session_builder_window import SessionBuilderWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _catalog_entry(identifier: str, label: str, kind: str = "binaural") -> SessionPresetChoice:
    return SessionPresetChoice(id=identifier, label=label, kind=kind)


class DummyStream:
    def __init__(self, track_data):
        self.track_data = track_data
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class DummyAssembler:
    def __init__(self, session, binaural_catalog, noise_catalog, **options):
        self.session = session
        self.binaural_catalog = binaural_catalog
        self.noise_catalog = noise_catalog
        self.options = options
        self.track_data = {"steps": [step.__dict__ for step in session.steps], "global_settings": {}}
        self.render_calls = []

    def render_to_file(self, path):
        self.render_calls.append(path)
        return True


def test_step_duration_updates_model(qapp):
    binaural_catalog = {"alpha": _catalog_entry("alpha", "Alpha")}
    noise_catalog = {"rain": _catalog_entry("rain", "Rain", kind="noise")}
    session = Session(steps=[SessionStep(binaural_preset_id="alpha", duration=60.0)])

    window = SessionBuilderWindow(
        session=session,
        binaural_catalog=binaural_catalog,
        noise_catalog=noise_catalog,
        assembler_factory=lambda s, b, n, **opts: DummyAssembler(s, b, n, **opts),
        stream_player_factory=lambda track: DummyStream(track),
    )
    window.show()
    qapp.processEvents()

    window.step_table.selectRow(0)
    qapp.processEvents()
    window.duration_spin.setValue(120.0)
    qapp.processEvents()

    assert session.steps[0].duration == pytest.approx(120.0)
    index = window.step_model.index(0, 0)
    assert window.step_model.data(index) == "120.00"

    window.description_edit.setPlainText("Deep focus")
    qapp.processEvents()
    assert session.steps[0].description == "Deep focus"

    window.close()


def test_preview_and_export_invoke_services(qapp, monkeypatch, tmp_path):
    binaural_catalog = {"alpha": _catalog_entry("alpha", "Alpha")}
    session = Session(steps=[SessionStep(binaural_preset_id="alpha", duration=30.0)])

    assemblers = []

    def assembler_factory(session, binaural_catalog, noise_catalog, **opts):
        assembler = DummyAssembler(session, binaural_catalog, noise_catalog, **opts)
        assemblers.append(assembler)
        return assembler

    streams = []

    def stream_factory(track_data):
        stream = DummyStream(track_data)
        streams.append(stream)
        return stream

    window = SessionBuilderWindow(
        session=session,
        binaural_catalog=binaural_catalog,
        noise_catalog={},
        assembler_factory=assembler_factory,
        stream_player_factory=stream_factory,
    )
    window.show()
    qapp.processEvents()

    window.preview_btn.click()
    qapp.processEvents()

    assert assemblers, "Assembler should be created for preview"
    assert streams and streams[-1].started is True

    window.stop_btn.click()
    qapp.processEvents()
    assert streams[-1].stopped is True

    export_path = tmp_path / "session.flac"
    monkeypatch.setattr(
        "src.ui.session_builder_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path), ""),
    )

    window.export_btn.click()
    qapp.processEvents()

    assert assemblers[-1].render_calls == [str(export_path)]

    window.close()

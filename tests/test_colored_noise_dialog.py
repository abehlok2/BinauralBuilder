import pytest

try:
    from PyQt5.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyQt5 not available: {exc}", allow_module_level=True)

from src.ui.colored_noise_dialog import ColoredNoiseDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class DummySignal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self, value):
        if self.callback:
            self.callback(value)


class DummyAudioFormat:
    LittleEndian = "little"
    SignedInt = "signed"

    def __init__(self):
        self.params = {}

    def setCodec(self, value):
        self.params["codec"] = value

    def setSampleRate(self, value):
        self.params["sample_rate"] = value

    def setSampleSize(self, value):
        self.params["sample_size"] = value

    def setChannelCount(self, value):
        self.params["channels"] = value

    def setByteOrder(self, value):
        self.params["byte_order"] = value

    def setSampleType(self, value):
        self.params["sample_type"] = value


class DummyAudioOutput:
    def __init__(self, fmt, parent=None):
        self.format = fmt
        self.parent = parent
        self.stateChanged = DummySignal()
        self.buffer_started = None
        self.stopped = False

    def start(self, buffer):
        self.buffer_started = buffer
        return True

    def stop(self):
        self.stopped = True


class DummyAudioDeviceInfo:
    @staticmethod
    def defaultOutputDevice():
        return DummyAudioDeviceInfo()

    def isFormatSupported(self, fmt):
        self.last_format = fmt
        return True


class DummyAudioStates:
    IdleState = "idle"
    StoppedState = "stopped"


def test_test_and_stop_buttons(qapp, monkeypatch):
    from src.ui import colored_noise_dialog as dialog_mod

    dialog = ColoredNoiseDialog()
    dialog.duration_spin.setValue(0.1)

    monkeypatch.setattr(dialog_mod, "QT_MULTIMEDIA_AVAILABLE", True)
    monkeypatch.setattr(dialog_mod, "QAudioOutput", DummyAudioOutput)
    monkeypatch.setattr(dialog_mod, "QAudioFormat", DummyAudioFormat)
    monkeypatch.setattr(dialog_mod, "QAudioDeviceInfo", DummyAudioDeviceInfo)
    monkeypatch.setattr(dialog_mod, "QAudio", DummyAudioStates)

    dialog.test_btn.click()
    qapp.processEvents()

    assert isinstance(dialog.audio_output, DummyAudioOutput)
    assert dialog.stop_btn.isEnabled() is True
    assert dialog.audio_buffer is not None

    dialog.stop_btn.click()
    qapp.processEvents()

    assert dialog.audio_output is None
    assert dialog.audio_buffer is None
    assert dialog.stop_btn.isEnabled() is False

    dialog.close()

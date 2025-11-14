"""Incremental audio streaming utilities for session playback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - import guard mirrors UI dialogs
    from PyQt5.QtCore import QBuffer, QIODevice, QObject
    from PyQt5.QtMultimedia import (
        QAudio,
        QAudioDeviceInfo,
        QAudioFormat,
        QAudioOutput,
    )

    QT_MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover - allow headless operation
    QT_MULTIMEDIA_AVAILABLE = False

    class _DummyQObject:
        def __init__(self, parent: Optional[object] = None) -> None:
            self._parent = parent

    class _DummyQIODevice:
        ReadOnly = 0x0001

        def __init__(self, parent: Optional[object] = None) -> None:
            self._parent = parent
            self._is_open = False

        def open(self, *args, **kwargs) -> bool:
            self._is_open = True
            return True

        def close(self) -> None:
            self._is_open = False

        def bytesAvailable(self) -> int:
            return 0

        def isSequential(self) -> bool:
            return True

    class _DummyAudioFormat:
        LittleEndian = 1
        SignedInt = 1

        def setCodec(self, *_args, **_kwargs) -> None:
            pass

        def setSampleRate(self, *_args, **_kwargs) -> None:
            pass

        def setSampleSize(self, *_args, **_kwargs) -> None:
            pass

        def setChannelCount(self, *_args, **_kwargs) -> None:
            pass

        def setByteOrder(self, *_args, **_kwargs) -> None:
            pass

        def setSampleType(self, *_args, **_kwargs) -> None:
            pass

    class _DummyDeviceInfo:
        @staticmethod
        def defaultOutputDevice() -> "_DummyDeviceInfo":
            return _DummyDeviceInfo()

        def isFormatSupported(self, *_args, **_kwargs) -> bool:
            return True

    class _DummyQBuffer(_DummyQIODevice):
        def __init__(self, parent: Optional[object] = None) -> None:
            super().__init__(parent)
            self._data = b""

        def setData(self, data: bytes) -> None:
            self._data = bytes(data)

        def seek(self, *_args, **_kwargs) -> None:
            pass

    class _DummyQAudio:
        IdleState = 0
        StoppedState = 1
        UnderflowError = 2
        NoError = 0

    QIODevice = _DummyQIODevice  # type: ignore[misc, assignment]
    QObject = _DummyQObject  # type: ignore[misc, assignment]
    QAudio = _DummyQAudio  # type: ignore
    QAudioDeviceInfo = _DummyDeviceInfo  # type: ignore
    QAudioFormat = _DummyAudioFormat  # type: ignore
    QAudioOutput = None  # type: ignore
    QBuffer = _DummyQBuffer  # type: ignore

from src.synth_functions.sound_creator import (
    crossfade_signals,
    generate_single_step_audio_segment,
)


_INT16_MAX = np.int16(32767).item()
_BYTES_PER_FRAME = 4  # 16-bit stereo


@dataclass
class _StreamState:
    index: int = 0
    prev_chunk: Optional[np.ndarray] = None
    prev_crossfade_samples: int = 0
    prev_crossfade_curve: str = "linear"


class _PCMBufferDevice(QIODevice):
    """Sequential ``QIODevice`` backed by a FIFO queue of PCM frames."""

    def __init__(self, parent: Optional[QObject] = None) -> None:  # type: ignore[override]
        super().__init__(parent)
        self._queue: Deque[bytes] = deque()
        self._current: bytes = b""
        self._offset: int = 0

    def isSequential(self) -> bool:  # pragma: no cover - Qt hook
        return True

    def bytesAvailable(self) -> int:  # pragma: no cover - Qt hook
        pending = len(self._current) - self._offset
        pending = max(pending, 0)
        queued = sum(len(chunk) for chunk in self._queue)
        return pending + queued + super().bytesAvailable()

    def readData(self, maxlen: int) -> bytes:  # pragma: no cover - Qt hook
        if maxlen <= 0:
            return bytes()
        result = bytearray()
        while len(result) < maxlen:
            if self._current:
                remaining = len(self._current) - self._offset
                if remaining <= 0:
                    self._current = b""
                    self._offset = 0
                    continue
                to_copy = min(maxlen - len(result), remaining)
                start = self._offset
                end = start + to_copy
                result.extend(self._current[start:end])
                self._offset = end
                if self._offset >= len(self._current):
                    self._current = b""
                    self._offset = 0
                continue
            if not self._queue:
                break
            self._current = self._queue.popleft()
            self._offset = 0
        return bytes(result)

    def writeData(self, data: bytes) -> int:  # pragma: no cover - Qt hook
        return -1  # Read-only device

    def enqueue(self, chunk: bytes) -> None:
        if chunk:
            self._queue.append(bytes(chunk))

    def clear(self) -> None:
        self._queue.clear()
        self._current = b""
        self._offset = 0

    def queued_bytes(self) -> int:
        pending = len(self._current) - self._offset
        pending = max(pending, 0)
        return pending + sum(len(chunk) for chunk in self._queue)

    def take_all(self) -> bytes:
        remainder = bytearray()
        if self._current:
            remainder.extend(self._current[self._offset :])
        for chunk in self._queue:
            remainder.extend(chunk)
        self.clear()
        return bytes(remainder)


class SessionStreamPlayer(QObject):  # type: ignore[misc]
    """Stream a session timeline into a :class:`QAudioOutput` incrementally."""

    def __init__(
        self,
        track_data: Dict[str, object],
        parent: Optional[QObject] = None,  # type: ignore[override]
        *,
        use_prebuffer: bool = False,
        ring_buffer_seconds: float = 3.0,
        audio_output_factory: Optional[Callable[[QAudioFormat, Optional[QObject]], object]] = None,
        validate_format: bool = True,
    ) -> None:
        super().__init__(parent)  # type: ignore[misc]
        self._track_data = dict(track_data or {})
        self._steps: List[Dict[str, object]] = list(
            self._track_data.get("steps", [])  # type: ignore[arg-type]
        )
        global_settings = dict(self._track_data.get("global_settings", {}))
        self._global_settings = global_settings
        self._sample_rate = int(global_settings.get("sample_rate", 44100))
        self._default_crossfade_duration = float(global_settings.get("crossfade_duration", 0.0))
        self._default_crossfade_curve = str(global_settings.get("crossfade_curve", "linear"))

        self._use_prebuffer = bool(use_prebuffer)
        self._ring_buffer_seconds = max(float(ring_buffer_seconds), 0.1)
        self._validate_format = bool(validate_format)

        if audio_output_factory is None:
            if not QT_MULTIMEDIA_AVAILABLE:  # pragma: no cover - guard
                raise RuntimeError("Qt multimedia backend is not available")
            audio_output_factory = QAudioOutput
        self._audio_output_factory = audio_output_factory

        self._audio_output: Optional[QAudioOutput] = None  # type: ignore[assignment]
        self._buffer_device: Optional[_PCMBufferDevice] = None
        self._prebuffer_device: Optional[QBuffer] = None  # type: ignore[type-arg]
        self._stream_state = _StreamState(prev_crossfade_curve=self._default_crossfade_curve)
        self._total_samples_estimate = self._estimate_total_samples()
        self._processed_samples = 0

        self._progress_callback: Optional[Callable[[float], None]] = None
        self._time_remaining_callback: Optional[Callable[[float], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_progress_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._progress_callback = callback

    def set_time_remaining_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._time_remaining_callback = callback

    def start(self, *, use_prebuffer: Optional[bool] = None) -> None:
        if use_prebuffer is not None:
            self._use_prebuffer = bool(use_prebuffer)
        self.stop()

        fmt = self._build_format()
        self._processed_samples = 0
        self._stream_state = _StreamState(prev_crossfade_curve=self._default_crossfade_curve)

        if self._use_prebuffer:
            data = self._render_full_audio()
            self._processed_samples = len(data) // _BYTES_PER_FRAME
            self._emit_progress()
            buffer = QBuffer()
            buffer.setData(data)
            buffer.open(QIODevice.ReadOnly)
            self._prebuffer_device = buffer
            self._audio_output = self._create_audio_output(fmt)
            self._audio_output.start(buffer)
        else:
            device = _PCMBufferDevice()
            device.open(QIODevice.ReadOnly)
            self._buffer_device = device
            self._audio_output = self._create_audio_output(fmt)
            self._prime_ring_buffer()
            self._audio_output.start(device)

    def pause(self) -> None:
        if self._audio_output:
            self._audio_output.suspend()

    def resume(self) -> None:
        if self._audio_output:
            self._prime_ring_buffer()
            self._audio_output.resume()

    def stop(self) -> None:
        if self._audio_output:
            try:
                self._audio_output.stateChanged.disconnect(self._handle_state_change)
            except Exception:  # pragma: no cover - defensive
                pass
            self._audio_output.stop()
            self._audio_output = None
        if self._buffer_device:
            self._buffer_device.close()
            self._buffer_device.clear()
            self._buffer_device = None
        if self._prebuffer_device:
            self._prebuffer_device.close()
            self._prebuffer_device = None
        self._stream_state = _StreamState(prev_crossfade_curve=self._default_crossfade_curve)
        self._processed_samples = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_audio_output(self, fmt: QAudioFormat) -> QAudioOutput:  # type: ignore[override]
        audio_output = self._audio_output_factory(fmt, self)  # type: ignore[misc]
        if hasattr(audio_output, "stateChanged"):
            audio_output.stateChanged.connect(self._handle_state_change)  # type: ignore[call-arg]
        return audio_output  # type: ignore[return-value]

    def _build_format(self) -> QAudioFormat:
        fmt = QAudioFormat()
        if hasattr(fmt, "setCodec"):
            fmt.setCodec("audio/pcm")
        if hasattr(fmt, "setSampleRate"):
            fmt.setSampleRate(self._sample_rate)
        if hasattr(fmt, "setSampleSize"):
            fmt.setSampleSize(16)
        if hasattr(fmt, "setChannelCount"):
            fmt.setChannelCount(2)
        if hasattr(fmt, "setByteOrder"):
            fmt.setByteOrder(getattr(QAudioFormat, "LittleEndian", 0))
        if hasattr(fmt, "setSampleType"):
            fmt.setSampleType(getattr(QAudioFormat, "SignedInt", 0))

        if QT_MULTIMEDIA_AVAILABLE and self._validate_format and QAudioDeviceInfo is not None:
            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(fmt):  # pragma: no cover - hardware dependent
                raise RuntimeError("Default output device does not support 16-bit stereo PCM")
        return fmt

    def _estimate_total_samples(self) -> int:
        total = 0
        prev_crossfade = 0
        first = True
        for step in self._steps:
            duration = float(step.get("duration", 0.0))  # type: ignore[arg-type]
            samples = max(int(duration * self._sample_rate), 0)
            if first:
                total += samples
                first = False
            else:
                total += max(samples - prev_crossfade, 0)
            crossfade = float(step.get("crossfade_duration", self._default_crossfade_duration))
            prev_crossfade = max(int(crossfade * self._sample_rate), 0)
        return max(total, 0)

    def _prime_ring_buffer(self) -> None:
        if not self._buffer_device:
            return
        target_bytes = int(self._ring_buffer_seconds * self._sample_rate * _BYTES_PER_FRAME)
        while self._buffer_device.queued_bytes() < target_bytes:
            chunk = self._next_float_chunk()
            if chunk is None:
                break
            self._enqueue_audio_chunk(chunk)

    def _render_full_audio(self) -> bytes:
        chunks: List[np.ndarray] = []
        state = _StreamState(prev_crossfade_curve=self._default_crossfade_curve)
        while True:
            chunk = self._generate_next_chunk(state)
            if chunk is None:
                break
            chunks.append(chunk)
        if chunks:
            audio = np.concatenate(chunks, axis=0)
        else:
            audio = np.zeros((0, 2), dtype=np.float32)
        return self._float_to_pcm(audio)

    def _next_float_chunk(self) -> Optional[np.ndarray]:
        return self._generate_next_chunk(self._stream_state)

    def _generate_next_chunk(self, state: _StreamState) -> Optional[np.ndarray]:
        while True:
            if state.index >= len(self._steps):
                chunk = state.prev_chunk
                state.prev_chunk = None
                if chunk is not None and chunk.size:
                    return chunk
                return None

            step = self._steps[state.index]
            duration = float(step.get("duration", 0.0))
            if duration <= 0.0:
                state.index += 1
                state.prev_crossfade_samples = int(
                    max(float(step.get("crossfade_duration", self._default_crossfade_duration)), 0.0)
                    * self._sample_rate
                )
                state.prev_crossfade_curve = str(step.get("crossfade_curve", self._default_crossfade_curve))
                continue

            audio = generate_single_step_audio_segment(
                step,
                self._global_settings,
                duration,
            )
            if audio is None or audio.ndim != 2 or audio.shape[1] != 2:
                audio = np.zeros((int(duration * self._sample_rate), 2), dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)

            crossfade_duration = float(step.get("crossfade_duration", self._default_crossfade_duration))
            crossfade_samples = max(int(crossfade_duration * self._sample_rate), 0)
            crossfade_curve = str(step.get("crossfade_curve", self._default_crossfade_curve))

            if state.prev_chunk is None:
                state.prev_chunk = audio
                state.prev_crossfade_samples = crossfade_samples
                state.prev_crossfade_curve = crossfade_curve
                state.index += 1
                continue

            prev_chunk = state.prev_chunk
            incoming_samples = min(
                state.prev_crossfade_samples,
                prev_chunk.shape[0],
                audio.shape[0],
            )
            if incoming_samples > 0:
                transition = crossfade_signals(
                    prev_chunk[-incoming_samples:],
                    audio[:incoming_samples],
                    self._sample_rate,
                    incoming_samples / self._sample_rate,
                    curve=state.prev_crossfade_curve or self._default_crossfade_curve,
                    phase_align=True,
                )
                head = prev_chunk[:-incoming_samples]
                if head.size:
                    chunk_out = np.concatenate([head, transition], axis=0)
                else:
                    chunk_out = transition
                remainder = audio[incoming_samples:]
            else:
                chunk_out = prev_chunk
                remainder = audio

            state.prev_chunk = remainder if remainder.size else None
            state.prev_crossfade_samples = crossfade_samples
            state.prev_crossfade_curve = crossfade_curve
            state.index += 1
            if chunk_out.size:
                return chunk_out.astype(np.float32, copy=False)

    def _enqueue_audio_chunk(self, chunk: np.ndarray) -> None:
        if not self._buffer_device or chunk.size == 0:
            return
        pcm = self._float_to_pcm(chunk)
        self._buffer_device.enqueue(pcm)
        self._processed_samples += chunk.shape[0]
        self._emit_progress()

    def _float_to_pcm(self, audio: np.ndarray) -> bytes:
        if audio.size == 0:
            return b""
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = np.asarray((clipped * _INT16_MAX).round(), dtype=np.int16)
        return pcm.tobytes()

    def _emit_progress(self) -> None:
        if self._progress_callback:
            total = self._total_samples_estimate or 1
            ratio = min(max(self._processed_samples / total, 0.0), 1.0)
            self._progress_callback(ratio)
        if self._time_remaining_callback:
            remaining_samples = max(self._total_samples_estimate - self._processed_samples, 0)
            seconds = remaining_samples / float(self._sample_rate or 1)
            self._time_remaining_callback(seconds)

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------
    def _handle_state_change(self, state: int) -> None:  # pragma: no cover - Qt runtime
        if not self._audio_output:
            return
        if not self._use_prebuffer:
            if state == QAudio.IdleState:
                self._prime_ring_buffer()
                if self._buffer_device and self._buffer_device.queued_bytes() == 0:
                    self.stop()
            elif state == QAudio.StoppedState:
                if self._audio_output.error() == QAudio.UnderflowError:
                    self._prime_ring_buffer()
        else:
            if state == QAudio.IdleState and self._prebuffer_device:
                self._prebuffer_device.seek(0)
                self._audio_output.start(self._prebuffer_device)


__all__ = ["SessionStreamPlayer"]

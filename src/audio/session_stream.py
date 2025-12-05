"""Incremental audio streaming utilities for session playback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - import guard mirrors UI dialogs
    from PyQt5.QtCore import QBuffer, QIODevice, QObject, QTimer, QMutex, QMutexLocker, QThread, pyqtSignal, pyqtSlot, QCoreApplication
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
            
    class _DummySignal:
        def connect(self, slot): pass
        def emit(self, *args): pass

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
        def bytesAvailable(self) -> int: return 0
        def isSequential(self) -> bool: return True
        def read(self, maxlen): return b""

    class _DummyAudioFormat:
        LittleEndian = 1
        SignedInt = 1
        def setCodec(self, *_, **__): pass
        def setSampleRate(self, *_, **__): pass
        def setSampleSize(self, *_, **__): pass
        def setChannelCount(self, *_, **__): pass
        def setByteOrder(self, *_, **__): pass
        def setSampleType(self, *_, **__): pass

    class _DummyDeviceInfo:
        @staticmethod
        def defaultOutputDevice() -> "_DummyDeviceInfo": return _DummyDeviceInfo()
        def isFormatSupported(self, *_, **__) -> bool: return True

    class _DummyQBuffer(_DummyQIODevice):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._data = b""
        def setData(self, data): self._data = data
        def seek(self, pos): pass

    class _DummyQAudio:
        IdleState = 0
        StoppedState = 1
        UnderflowError = 2
        NoError = 0

    class _DummyQTimer:
        def __init__(self, parent=None): pass
        def start(self, ms=None): pass
        def stop(self): pass
        def setInterval(self, ms): pass
        @property
        def timeout(self): return _DummySignal()

    class _DummyQMutex:
        def lock(self): pass
        def unlock(self): pass
        
    class _DummyQMutexLocker:
        def __init__(self, mutex): pass

    class _DummyQThread:
        def start(self): pass
        def quit(self): pass
        def wait(self): pass
        def msleep(self, ms): pass
        def isInterruptionRequested(self): return False
        def requestInterruption(self): pass
        @property
        def started(self): return _DummySignal()

    def pyqtSignal(*types): return _DummySignal()
    def pyqtSlot(*types): 
        def decorator(func): return func
        return decorator

    QIODevice = _DummyQIODevice  # type: ignore
    QObject = _DummyQObject  # type: ignore
    QAudio = _DummyQAudio  # type: ignore
    QAudioDeviceInfo = _DummyDeviceInfo  # type: ignore
    QAudioFormat = _DummyAudioFormat  # type: ignore
    QAudioOutput = None  # type: ignore
    QBuffer = _DummyQBuffer  # type: ignore
    QTimer = _DummyQTimer # type: ignore
    QMutex = _DummyQMutex # type: ignore
    QMutexLocker = _DummyQMutexLocker # type: ignore
    QThread = _DummyQThread # type: ignore
    QCoreApplication = None # type: ignore


from src.synth_functions.sound_creator import (
    crossfade_signals,
    generate_single_step_audio_segment,
)


_INT16_MAX = np.int16(32767).item()
_BYTES_PER_FRAME = 4  # 16-bit stereo


@dataclass
class _StepPlaybackInfo:
    index: int
    start_sample: int
    end_sample: int
    fade_in_samples: int
    fade_in_curve: str
    fade_out_samples: int
    fade_out_curve: str
    data: Dict[str, object]


class _PCMBufferDevice(QIODevice):
    """Sequential ``QIODevice`` backed by a FIFO queue of PCM frames.
    
    This class is thread-safe.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:  # type: ignore[override]
        super().__init__(parent)
        self._queue: Deque[bytes] = deque()
        self._current: bytes = b""
        self._offset: int = 0
        self._mutex = QMutex()

    def isSequential(self) -> bool:  # pragma: no cover - Qt hook
        return True

    def bytesAvailable(self) -> int:  # pragma: no cover - Qt hook
        locker = QMutexLocker(self._mutex)
        pending = len(self._current) - self._offset
        pending = max(pending, 0)
        queued = sum(len(chunk) for chunk in self._queue)
        # Note: calling super().bytesAvailable() might not be thread safe depending on implementation, 
        # but for QIODevice it usually just calls pure virtual unless buffered. 
        # Safest to just return exact count we know.
        return pending + queued + super().bytesAvailable()

    def readData(self, maxlen: int) -> bytes:  # pragma: no cover - Qt hook
        locker = QMutexLocker(self._mutex)
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
            locker = QMutexLocker(self._mutex)
            self._queue.append(bytes(chunk))
            # Critical: Notify readers (QAudioOutput) that data is available!
            # We must emit this signal, but emitting from a background thread (worker) 
            # to a QIODevice living on the main thread (presumably) is safe due to Qt signal/slot queuing,
            # BUT readyRead is a signal of QIODevice. emitting it directly is fine if thread affinity is handled,
            # or if we are careful. QAudioOutput usually connects to this.
            # Ideally we unlock before emitting to avoid deadlock if slot calls back immediately (unlikely for readyRead but good practice).
            locker.unlock()
            self.readyRead.emit()

    def clear(self) -> None:
        locker = QMutexLocker(self._mutex)
        self._queue.clear()
        self._current = b""
        self._offset = 0

    def queued_bytes(self) -> int:
        locker = QMutexLocker(self._mutex)
        pending = len(self._current) - self._offset
        pending = max(pending, 0)
        return pending + sum(len(chunk) for chunk in self._queue)

    def take_all(self) -> bytes:
        locker = QMutexLocker(self._mutex)
        remainder = bytearray()
        if self._current:
            remainder.extend(self._current[self._offset :])
        for chunk in self._queue:
            remainder.extend(chunk)
        self._queue.clear()
        self._current = b""
        self._offset = 0
        return bytes(remainder)


class AudioGeneratorWorker(QObject):
    """Background worker for generating audio chunks."""
    
    chunk_ready = pyqtSignal() # Signal that some data was added (optional, maybe just pull status)
    progress_updated = pyqtSignal(float)
    time_remaining_updated = pyqtSignal(float)
    finished = pyqtSignal()
    
    def __init__(
        self,
        track_data: Dict[str, object],
        buffer_device: _PCMBufferDevice,
        sample_rate: int,
        ring_buffer_seconds: float,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._track_data = track_data
        self._buffer_device = buffer_device
        self._sample_rate = sample_rate
        self._ring_buffer_seconds = ring_buffer_seconds
        
        global_settings = dict(self._track_data.get("global_settings", {}))
        self._global_settings = global_settings
        self._default_crossfade_duration = float(global_settings.get("crossfade_duration", 0.0))
        self._default_crossfade_curve = str(global_settings.get("crossfade_curve", "linear"))
        
        self._steps: List[Dict[str, object]] = list(
            self._track_data.get("steps", [])
        )
        
        self._playback_sample = 0
        self._playback_step_states: Dict[int, List[dict]] = {}
        self._step_infos: List[_StepPlaybackInfo] = []
        self._total_samples_estimate = 0
        
        self._running = False
        self._paused = False
        
        self._recalculate_timeline()
        
    def _recalculate_timeline(self) -> None:
        """Pre-calculate start/end samples for all steps based on crossfades."""
        self._step_infos = []
        current_time_sample = 0
        prev_crossfade_samples = 0
        
        for i, step in enumerate(self._steps):
            duration = float(step.get("duration", 0.0))
            samples = max(int(duration * self._sample_rate), 0)
            
            crossfade = float(step.get("crossfade_duration", self._default_crossfade_duration))
            crossfade_samples = max(int(crossfade * self._sample_rate), 0)
            crossfade_curve = str(step.get("crossfade_curve", self._default_crossfade_curve))
            
            start_sample = current_time_sample
            end_sample = start_sample + samples
            
            fade_in_len = prev_crossfade_samples if i > 0 else 0
            fade_out_len = crossfade_samples
            
            prev_step_curve = self._steps[i-1].get("crossfade_curve", self._default_crossfade_curve) if i > 0 else "linear"
            
            info = _StepPlaybackInfo(
                index=i,
                start_sample=start_sample,
                end_sample=end_sample,
                fade_in_samples=fade_in_len,
                fade_in_curve=str(prev_step_curve),
                fade_out_samples=fade_out_len,
                fade_out_curve=crossfade_curve,
                data=step
            )
            self._step_infos.append(info)
            
            advance = max(0, samples - crossfade_samples)
            current_time_sample += advance
            
            prev_crossfade_samples = crossfade_samples
            
        if self._step_infos:
            self._total_samples_estimate = self._step_infos[-1].end_sample
        else:
            self._total_samples_estimate = 0
            
    @pyqtSlot()
    def start_generation(self):
        self._running = True
        self._paused = False
        self._process_loop()
        
    @pyqtSlot()
    def stop_generation(self):
        self._running = False
        
    @pyqtSlot()
    def pause_generation(self):
        self._paused = True
    
    @pyqtSlot()
    def resume_generation(self):
        self._paused = False
        
    @pyqtSlot(float)
    def seek(self, time_seconds: float):
        target_sample = int(time_seconds * self._sample_rate)
        # Clamp to valid range
        target_sample = max(0, min(target_sample, self._total_samples_estimate))
        
        self._playback_sample = target_sample
        # Reset states on seek (simplification)
        self._playback_step_states = {} 
        
        # Clear buffer to ensure immediate response
        self._buffer_device.clear()

    @property
    def total_samples(self):
        return self._total_samples_estimate
    
    @property
    def current_sample(self):
        return self._playback_sample

    def _process_loop(self):
        while self._running:
            if QThread.currentThread().isInterruptionRequested():
                break
                
            if self._paused:
                QThread.msleep(50)
                continue
                
            # Check buffer level
            target_bytes = int(self._ring_buffer_seconds * self._sample_rate * _BYTES_PER_FRAME)
            current_bytes = self._buffer_device.queued_bytes()
            
            if current_bytes >= target_bytes:
                # Buffer full, sleep a bit
                QThread.msleep(10)
                continue
            
            # Generate a chunk
            chunk_size = 4096 # approx 100ms
            chunk = self._generate_next_chunk(self._playback_sample, chunk_size, self._playback_step_states)
            
            if chunk is not None:
                self._playback_sample += chunk.shape[0]
                self._enqueue_audio_chunk(chunk)
                
                # Check if done
                if self._playback_sample >= self._total_samples_estimate:
                    # End of stream
                    # Don't stop running, just idle until seek or stop
                     QThread.msleep(100)
            else:
                # End of stream or error
                QThread.msleep(100)
                
            QCoreApplication.processEvents()
            
        self.finished.emit()

    def _enqueue_audio_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        pcm = self._float_to_pcm(chunk)
        self._buffer_device.enqueue(pcm)
        self.chunk_ready.emit()
        self._emit_progress()

    def _float_to_pcm(self, audio: np.ndarray) -> bytes:
        if audio.size == 0:
            return b""
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = np.asarray((clipped * _INT16_MAX).round(), dtype=np.int16)
        return pcm.tobytes()

    def _emit_progress(self) -> None:
        total = self._total_samples_estimate or 1
        ratio = min(max(self._playback_sample / total, 0.0), 1.0)
        self.progress_updated.emit(ratio)
        
        remaining_samples = max(self._total_samples_estimate - self._playback_sample, 0)
        seconds = remaining_samples / float(self._sample_rate or 1)
        self.time_remaining_updated.emit(seconds)

    def _generate_next_chunk(self, start_sample: int, max_frames: int, step_states: Dict[int, List[dict]]) -> Optional[np.ndarray]:
        if start_sample >= self._total_samples_estimate:
            return None
            
        end_sample = min(start_sample + max_frames, self._total_samples_estimate)
        num_frames = end_sample - start_sample
        
        if num_frames <= 0:
            return None
            
        mix_buffer = np.zeros((num_frames, 2), dtype=np.float32)
        
        for info in self._step_infos:
            if info.end_sample <= start_sample:
                continue
            if info.start_sample >= end_sample:
                break 
                
            chunk_rel_start = max(0, info.start_sample - start_sample)
            chunk_rel_end = min(num_frames, info.end_sample - start_sample)
            
            step_rel_start = max(0, start_sample - info.start_sample)
            step_rel_end = step_rel_start + (chunk_rel_end - chunk_rel_start)
            
            gen_len = step_rel_end - step_rel_start
            
            if gen_len <= 0:
                continue
                
            chunk_start_time = step_rel_start / self._sample_rate
            duration = gen_len / self._sample_rate
            
            current_states = step_states.get(info.index)
            
            audio, new_states = generate_single_step_audio_segment(
                info.data,
                self._global_settings,
                duration,
                duration_override=duration,
                chunk_start_time=chunk_start_time,
                voice_states=current_states,
                return_state=True
            )
            
            step_states[info.index] = new_states
            
            if audio.shape[0] != gen_len:
                if audio.shape[0] < gen_len:
                    audio = np.pad(audio, ((0, gen_len - audio.shape[0]), (0, 0)))
                else:
                    audio = audio[:gen_len]
            
            # Fade In
            if info.fade_in_samples > 0 and step_rel_start < info.fade_in_samples:
                fade_start_idx = 0 
                fade_end_idx = min(gen_len, info.fade_in_samples - step_rel_start)
                start_p = step_rel_start / info.fade_in_samples
                end_p = (step_rel_start + fade_end_idx) / info.fade_in_samples
                curve = np.linspace(start_p, end_p, fade_end_idx)
                envelope = curve
                audio[:fade_end_idx] *= envelope[:, np.newaxis]
                
            # Fade Out
            step_duration_samples = info.end_sample - info.start_sample
            fade_out_start_sample = step_duration_samples - info.fade_out_samples
            
            if info.fade_out_samples > 0 and step_rel_end > fade_out_start_sample:
                local_start = max(0, fade_out_start_sample - step_rel_start)
                local_end = gen_len
                start_p = (step_rel_start + local_start - fade_out_start_sample) / info.fade_out_samples
                end_p = (step_rel_start + local_end - fade_out_start_sample) / info.fade_out_samples
                curve = np.linspace(start_p, end_p, local_end - local_start)
                envelope = 1.0 - curve
                audio[local_start:local_end] *= envelope[:, np.newaxis]
            
            mix_buffer[chunk_rel_start:chunk_rel_end] += audio
            
        return mix_buffer


class SessionStreamPlayer(QObject):  # type: ignore[misc]
    """Stream a session timeline into a :class:`QAudioOutput` using a threaded generator and ring buffer."""

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
        global_settings = dict(self._track_data.get("global_settings", {}))
        self._sample_rate = int(global_settings.get("sample_rate", 44100))
        
        self._use_prebuffer = bool(use_prebuffer)
        self._ring_buffer_seconds = max(float(ring_buffer_seconds), 0.1)
        self._validate_format = bool(validate_format)
        
        if audio_output_factory is None:
            if not QT_MULTIMEDIA_AVAILABLE:  # pragma: no cover - guard
                raise RuntimeError("Qt multimedia backend is not available")
            audio_output_factory = QAudioOutput
        self._audio_output_factory = audio_output_factory
        
        self._audio_output: Optional[QAudioOutput] = None
        self._buffer_device: Optional[_PCMBufferDevice] = None
        self._prebuffer_device: Optional[QBuffer] = None
        
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[AudioGeneratorWorker] = None
        
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._time_remaining_callback: Optional[Callable[[float], None]] = None

    def set_progress_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._progress_callback = callback

    def set_time_remaining_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._time_remaining_callback = callback

    def start(self, use_prebuffer: bool = False) -> None:
        """Start playback."""
        self.stop()
        self._use_prebuffer = bool(use_prebuffer)
        
        fmt = self._build_format()

        if self._use_prebuffer:
            # For prebuffer we still need to generate, could use worker or just do it inline.
            # Inline for simplicity if prebuffer is requested (usually for export preview or short clips).
            # But let's use the worker logic but blockingly? Or just legacy method.
            # Actually, prebuffer is rarely used for long sessions.
            # Let's keep existing prebuffer logic if possible, BUT we refactored the generation into worker.
            # So we instantiate a worker temporarily ?
            
            # Temporary worker for pre-render
            temp_device = _PCMBufferDevice()
            worker = AudioGeneratorWorker(self._track_data, temp_device, self._sample_rate, 10.0) # Large buffer
            
            # We need to run it synchronously?
            # Or just hack it:
            # The previous _render_full_audio logic is gone.
            # Reuse worker logic manually?
            
            # Re-implement simple full render:
            total = worker.total_samples
            chunk_size = 4096 * 4
            current = 0
            chunks = []
            states = {}
            while current < total:
                c = worker._generate_next_chunk(current, chunk_size, states)
                if c is None: break
                chunks.append(c)
                current += c.shape[0]
            
            if chunks:
                audio = np.concatenate(chunks, axis=0)
            else:
                audio = np.zeros((0, 2), dtype=np.float32)
            data = worker._float_to_pcm(audio)
            
            buffer = QBuffer()
            buffer.setData(data)
            buffer.open(QIODevice.ReadOnly)
            self._prebuffer_device = buffer
            self._audio_output = self._create_audio_output(fmt)
            self._audio_output.start(buffer)
            
        else:
            # Incremental Threaded Playback
            self._buffer_device = _PCMBufferDevice()
            self._buffer_device.open(QIODevice.ReadOnly)
            
            self._worker_thread = QThread()
            self._worker = AudioGeneratorWorker(
                self._track_data, 
                self._buffer_device, 
                self._sample_rate, 
                self._ring_buffer_seconds
            )
            self._worker.moveToThread(self._worker_thread)
            
            self._worker_thread.started.connect(self._worker.start_generation)
            self._worker.finished.connect(self._worker_thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._worker_thread.finished.connect(self._worker_thread.deleteLater)
            
            self._worker.progress_updated.connect(self._on_worker_progress)
            self._worker.time_remaining_updated.connect(self._on_worker_time_remaining)
            
            self._audio_output = self._create_audio_output(fmt)
            
            # Start everything
            self._worker_thread.start()
            self._audio_output.start(self._buffer_device)

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 - 1.0)."""
        if self._audio_output:
            self._audio_output.setVolume(volume)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self._worker:
             return self._worker.total_samples / self._sample_rate
        # Fallback estimate
        return 0.0

    @property
    def position(self) -> float:
        """Current playback position in seconds (approximate)."""
        if self._worker:
            return self._worker.current_sample / self._sample_rate
        return 0.0

    def seek(self, time_seconds: float) -> None:
        """Seek to a specific time in seconds."""
        if self._use_prebuffer and self._prebuffer_device:
            target_sample = int(time_seconds * self._sample_rate)
            byte_offset = target_sample * _BYTES_PER_FRAME
            byte_offset = (byte_offset // _BYTES_PER_FRAME) * _BYTES_PER_FRAME
            self._prebuffer_device.seek(byte_offset)
            return

        if self._worker:
             # This must be thread safe. We call a slot on the worker.
             # But seeking also requires clearing the buffer which the worker does.
             # We invoke method via meta object to ensure it runs on worker thread?
             # Or simply direct call if we used mutexes?
             # QObject calls across threads are safe if slots.
             self._worker.seek(time_seconds)
             
    def pause(self) -> None:
        if self._audio_output:
            self._audio_output.suspend()
        if self._worker:
            self._worker.pause_generation()

    def resume(self) -> None:
        if self._audio_output:
            self._audio_output.resume()
        if self._worker:
            self._worker.resume_generation()

    def stop(self) -> None:
        if self._audio_output:
            self._audio_output.stop()
            self._audio_output = None
            
        if self._worker:
            self._worker.stop_generation()
            if self._worker_thread:
                self._worker_thread.requestInterruption()
                self._worker_thread.quit()
                self._worker_thread.wait()
            self._worker = None
            self._worker_thread = None
            
        if self._buffer_device:
            self._buffer_device.close()
            self._buffer_device.clear()
            self._buffer_device = None
            
        if self._prebuffer_device:
            self._prebuffer_device.close()
            self._prebuffer_device = None

    def _on_worker_progress(self, ratio: float):
        if self._progress_callback:
            self._progress_callback(ratio)
            
    def _on_worker_time_remaining(self, seconds: float):
        if self._time_remaining_callback:
            self._time_remaining_callback(seconds)

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
        
    def _handle_state_change(self, state: int) -> None:  # pragma: no cover - Qt runtime
        if not self._audio_output:
            return
        if state == QAudio.IdleState and not self._use_prebuffer:
             # Buffer underrun or finished?
             # If finished, we could stop.
             pass


__all__ = ["SessionStreamPlayer"]


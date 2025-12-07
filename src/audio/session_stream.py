"""Incremental audio streaming utilities for session playback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import time

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

        def moveToThread(self, *_args, **_kwargs):
            # Thread affinity is irrelevant for the dummy shim
            return None
            
    class _DummySignal:
        def __init__(self):
            self._callbacks: list[Callable[..., None]] = []

        def connect(self, slot):
            self._callbacks.append(slot)

        def emit(self, *args):
            for cb in list(self._callbacks):
                cb(*args)

    class _DummyQIODevice:
        ReadOnly = 0x0001

        readyRead = _DummySignal()
        
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
        @staticmethod
        def singleShot(ms, callback):
            # In headless mode, just call the callback immediately
            if callable(callback):
                callback()

    class _DummyQMutex:
        def lock(self): pass
        def unlock(self): pass
        
    class _DummyQMutexLocker:
        def __init__(self, mutex):
            self._mutex = mutex

        def unlock(self):
            if hasattr(self._mutex, "unlock"):
                self._mutex.unlock()

    class _DummyQThread:
        _current = None

        def __init__(self):
            self._interrupted = False
            self._started = _DummySignal()

        @property
        def started(self):
            return self._started

        def start(self):
            _DummyQThread._current = self
            self._started.emit()

        def quit(self):
            self._interrupted = True

        def wait(self):
            return None

        @staticmethod
        def msleep(ms):
            time.sleep(ms / 1000.0)

        def isInterruptionRequested(self):
            return self._interrupted

        def requestInterruption(self):
            self._interrupted = True

        @staticmethod
        def currentThread():
            return _DummyQThread._current or _DummyQThread()

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
from src.synth_functions.noise_flanger import (
    _generate_swept_notch_arrays,
    _generate_swept_notch_arrays_transition,
)
from src.utils.noise_file import load_noise_params


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
        # Track total bytes consumed by audio output for accurate position tracking
        self._total_bytes_consumed: int = 0

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
        # Track bytes consumed for accurate position reporting
        self._total_bytes_consumed += len(result)
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

    def clear(self, reset_consumed: bool = False) -> None:
        locker = QMutexLocker(self._mutex)
        self._queue.clear()
        self._current = b""
        self._offset = 0
        if reset_consumed:
            self._total_bytes_consumed = 0

    def total_bytes_consumed(self) -> int:
        """Return total bytes consumed by audio output (thread-safe)."""
        locker = QMutexLocker(self._mutex)
        return self._total_bytes_consumed

    def reset_consumed_counter(self, value: int = 0) -> None:
        """Reset the consumed counter, optionally to a specific value."""
        locker = QMutexLocker(self._mutex)
        self._total_bytes_consumed = value

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

        self._background_noise: Optional[np.ndarray] = None

        self._running = False
        self._paused = False

        self._recalculate_timeline()
        self._prepare_background_noise()
        
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

    def _prepare_background_noise(self) -> None:
        """Generate the background noise layer to mirror offline assembly."""

        self._background_noise = None

        bg_cfg = self._track_data.get("background_noise") or {}
        if not isinstance(bg_cfg, dict):
            return

        bg_file = (
            bg_cfg.get("file")
            or bg_cfg.get("params_path")
            or bg_cfg.get("noise_file")
        )
        if not bg_file:
            return

        track_duration_sec = self._total_samples_estimate / float(self._sample_rate or 1)
        if track_duration_sec <= 0:
            return

        try:
            params = load_noise_params(bg_file)
        except Exception:
            return

        params.duration_seconds = track_duration_sec
        params.sample_rate = self._sample_rate

        try:
            if getattr(params, "transition", False):
                start_sweeps = [
                    (sw.get("start_min", 1000), sw.get("start_max", 10000))
                    for sw in params.sweeps
                ]
                end_sweeps = [
                    (sw.get("end_min", 1000), sw.get("end_max", 10000))
                    for sw in params.sweeps
                ]
                start_q = [sw.get("start_q", 30) for sw in params.sweeps]
                end_q = [sw.get("end_q", 30) for sw in params.sweeps]
                start_casc = [sw.get("start_casc", 10) for sw in params.sweeps]
                end_casc = [sw.get("end_casc", 10) for sw in params.sweeps]

                noise_audio, _ = _generate_swept_notch_arrays_transition(
                    track_duration_sec,
                    self._sample_rate,
                    params.start_lfo_freq,
                    params.end_lfo_freq,
                    start_sweeps,
                    end_sweeps,
                    start_q,
                    end_q,
                    start_casc,
                    end_casc,
                    params.start_lfo_phase_offset_deg,
                    params.end_lfo_phase_offset_deg,
                    params.start_intra_phase_offset_deg,
                    params.end_intra_phase_offset_deg,
                    params.input_audio_path or None,
                    params.noise_type,
                    params.lfo_waveform,
                    params.initial_offset,
                    params.duration,
                    "linear",
                    False,
                    2,
                    getattr(params, "static_notches", None),
                )
            else:
                sweeps = [
                    (sw.get("start_min", 1000), sw.get("start_max", 10000))
                    for sw in params.sweeps
                ]
                notch_q = [sw.get("start_q", 30) for sw in params.sweeps]
                casc = [sw.get("start_casc", 10) for sw in params.sweeps]

                noise_audio, _ = _generate_swept_notch_arrays(
                    track_duration_sec,
                    self._sample_rate,
                    params.lfo_freq,
                    sweeps,
                    notch_q,
                    casc,
                    params.start_lfo_phase_offset_deg,
                    params.start_intra_phase_offset_deg,
                    params.input_audio_path or None,
                    params.noise_type,
                    params.lfo_waveform,
                    False,
                    2,
                    getattr(params, "static_notches", None),
                )
        except Exception:
            return

        if noise_audio.ndim == 1:
            noise_audio = np.column_stack((noise_audio, noise_audio))

        start_time = float(bg_cfg.get("start_time", 0.0))
        start_sample = max(0, int(start_time * self._sample_rate))

        if start_sample > 0:
            noise_audio = np.pad(noise_audio, ((start_sample, 0), (0, 0)), "constant")

        gain = float(bg_cfg.get("gain", 1.0))
        fade_in = float(bg_cfg.get("fade_in", 0.0))
        fade_out = float(bg_cfg.get("fade_out", 0.0))
        amp_env = bg_cfg.get("amp_envelope")

        env = np.ones(noise_audio.shape[0], dtype=np.float32) * gain
        if fade_in > 0:
            n = min(int(fade_in * self._sample_rate), env.size)
            env[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        if fade_out > 0:
            n = min(int(fade_out * self._sample_rate), env.size)
            env[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
        if isinstance(amp_env, list) and amp_env:
            times = [max(0.0, float(p[0])) for p in amp_env]
            amps = [float(p[1]) for p in amp_env]
            t_samples = np.array(times) * self._sample_rate
            interp = np.interp(
                np.arange(env.size, dtype=np.float32),
                t_samples,
                amps,
                left=amps[0],
                right=amps[-1],
            )
            env *= interp

        noise_audio = noise_audio[: env.size] * env[:, None]
        self._background_noise = noise_audio.astype(np.float32, copy=False)

        if self._background_noise.shape[0] > self._total_samples_estimate:
            self._total_samples_estimate = self._background_noise.shape[0]
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

        # Clear buffer and reset consumed counter to seek position
        self._buffer_device.clear(reset_consumed=False)
        # Set consumed counter to reflect the new playback position
        seek_bytes = target_sample * _BYTES_PER_FRAME
        self._buffer_device.reset_consumed_counter(seek_bytes)

    @property
    def total_samples(self):
        return self._total_samples_estimate
    
    @property
    def current_sample(self):
        return self._playback_sample

    def _process_loop(self):
        # Use a consistent chunk size that's a power of 2 for audio alignment
        # 2048 samples at 44100 Hz ~= 46.4ms per chunk
        chunk_size = 2048

        # Calculate timing parameters based on sample rate
        # We want to generate audio at a rate that keeps the buffer filled
        # but doesn't overfill it. Target keeping buffer between 50-100% full.
        min_buffer_seconds = self._ring_buffer_seconds * 0.5
        target_buffer_seconds = self._ring_buffer_seconds * 0.75
        chunk_duration_ms = int((chunk_size / self._sample_rate) * 1000)

        while self._running:
            if QThread.currentThread().isInterruptionRequested():
                break

            # Process Qt events for signal delivery (stop, seek, pause)
            if QCoreApplication is not None:
                QCoreApplication.processEvents()

            if self._paused:
                QThread.msleep(50)
                continue

            # Check buffer level and calculate optimal sleep time
            current_bytes = self._buffer_device.queued_bytes()
            current_seconds = current_bytes / (self._sample_rate * _BYTES_PER_FRAME)

            if current_seconds >= self._ring_buffer_seconds:
                # Buffer is full - sleep for approximately the time it takes
                # to consume one chunk, so we wake up right when space is available
                sleep_ms = max(5, min(chunk_duration_ms, 50))
                QThread.msleep(sleep_ms)
                continue

            # Check if we've reached the end of the stream
            if self._playback_sample >= self._total_samples_estimate:
                # End of stream - idle but stay responsive
                QThread.msleep(50)
                continue

            # Calculate how many chunks we should generate to maintain buffer level
            # Generate enough to reach target buffer level
            deficit_seconds = max(0, target_buffer_seconds - current_seconds)
            deficit_samples = int(deficit_seconds * self._sample_rate)

            # Generate at least one chunk, but cap at a reasonable number
            # to avoid blocking the thread for too long
            samples_to_generate = max(chunk_size, min(deficit_samples, chunk_size * 4))

            # Generate audio in chunk_size increments for consistency
            samples_generated = 0
            while samples_generated < samples_to_generate and self._running and not self._paused:
                remaining = self._total_samples_estimate - self._playback_sample
                if remaining <= 0:
                    break

                this_chunk_size = min(chunk_size, remaining)
                chunk = self._generate_next_chunk(
                    self._playback_sample,
                    this_chunk_size,
                    self._playback_step_states
                )

                if chunk is None:
                    break

                self._playback_sample += chunk.shape[0]
                samples_generated += chunk.shape[0]
                self._enqueue_audio_chunk(chunk)

                # Brief yield to allow other operations (seek, stop)
                if QCoreApplication is not None:
                    QCoreApplication.processEvents()

            # If buffer is below minimum, don't sleep - keep generating
            if current_seconds >= min_buffer_seconds and samples_generated > 0:
                # Sleep proportionally to how much audio we just generated
                # This helps maintain consistent generation rate
                sleep_ms = max(2, chunk_duration_ms // 2)
                QThread.msleep(sleep_ms)

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

        if self._background_noise is not None:
            bg_slice = self._background_noise[start_sample:end_sample]
            if bg_slice.shape[0] < num_frames:
                bg_slice = np.pad(
                    bg_slice,
                    ((0, num_frames - bg_slice.shape[0]), (0, 0)),
                    "constant",
                )
            mix_buffer += bg_slice
        
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
            
            try:
                audio, new_states = generate_single_step_audio_segment(
                    info.data,
                    self._global_settings,
                    duration,
                    duration_override=duration,
                    chunk_start_time=chunk_start_time,
                    voice_states=current_states,
                    return_state=True
                )
            except TypeError:
                # Backwards-compatible path for simplified stubs used in tests
                # that do not accept chunk-based parameters.
                audio = generate_single_step_audio_segment(
                    info.data,
                    self._global_settings,
                    duration,
                    duration_override=duration,
                )
                new_states = current_states

            step_states[info.index] = new_states
            
            if audio.shape[0] != gen_len:
                if audio.shape[0] < gen_len:
                    audio = np.pad(audio, ((0, gen_len - audio.shape[0]), (0, 0)))
                else:
                    audio = audio[:gen_len]
            
            def _fade_envelope(start_frac: float, end_frac: float, length: int, curve: str, *, invert: bool = False) -> np.ndarray:
                """Return a fade envelope matching :func:`crossfade_signals` curves.

                ``start_frac``/``end_frac`` represent the fractional progress (0→1)
                through the fade window covered by this chunk. ``invert`` toggles
                between fade-in (False) and fade-out (True).
                """

                positions = np.linspace(start_frac, end_frac, length, endpoint=False, dtype=np.float32)
                positions = np.clip(positions, 0.0, 1.0)

                if curve == "equal_power":
                    theta = positions * (np.pi / 2.0)
                    values = np.cos(theta) if invert else np.sin(theta)
                else:
                    values = 1.0 - positions if invert else positions

                return values

            # Fade In
            if info.fade_in_samples > 0 and step_rel_start < info.fade_in_samples:
                fade_start_idx = 0
                fade_end_idx = min(gen_len, info.fade_in_samples - step_rel_start)
                start_p = step_rel_start / max(info.fade_in_samples, 1)
                end_p = (step_rel_start + fade_end_idx) / max(info.fade_in_samples, 1)
                envelope = _fade_envelope(start_p, end_p, fade_end_idx, info.fade_in_curve, invert=False)
                audio[:fade_end_idx] *= envelope[:, np.newaxis]

            # Fade Out
            step_duration_samples = info.end_sample - info.start_sample
            fade_out_start_sample = step_duration_samples - info.fade_out_samples

            if info.fade_out_samples > 0 and step_rel_end > fade_out_start_sample:
                local_start = max(0, fade_out_start_sample - step_rel_start)
                local_end = gen_len
                start_p = (step_rel_start + local_start - fade_out_start_sample) / max(info.fade_out_samples, 1)
                end_p = (step_rel_start + local_end - fade_out_start_sample) / max(info.fade_out_samples, 1)
                envelope = _fade_envelope(start_p, end_p, local_end - local_start, info.fade_out_curve, invert=True)
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
            self._worker.progress_updated.connect(self._on_worker_progress)
            self._worker.time_remaining_updated.connect(self._on_worker_time_remaining)
            if not QT_MULTIMEDIA_AVAILABLE:
                # Headless fallback: synchronously render into the buffer so tests
                # and non-Qt environments still receive audio data.
                self._render_full_stream_headless()
                self._audio_output = self._create_audio_output(fmt)
                self._audio_output.start(self._buffer_device)
                return
            if hasattr(self._worker, "moveToThread"):
                self._worker.moveToThread(self._worker_thread)

            self._worker_thread.started.connect(self._worker.start_generation)
            self._worker.finished.connect(self._worker_thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._worker_thread.finished.connect(self._worker_thread.deleteLater)

            self._audio_output = self._create_audio_output(fmt)

            # Start everything and prime a small buffer so the audio device
            # does not immediately underflow before the worker thread queues
            # data. This also helps the session builder when launching
            # streaming previews.
            self._worker_thread.start()
            self._wait_for_initial_buffer()
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
        """Current playback position in seconds based on actual audio consumed."""
        if self._buffer_device:
            # Use actual bytes consumed by audio output for accurate position
            consumed_bytes = self._buffer_device.total_bytes_consumed()
            consumed_samples = consumed_bytes // _BYTES_PER_FRAME
            return consumed_samples / self._sample_rate
        if self._worker:
            # Fallback to generation position if buffer not available
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
            self._buffer_device.clear(reset_consumed=True)
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
    def _wait_for_initial_buffer(self, min_seconds: float = 1.0, timeout_seconds: float = 5.0) -> None:
        """Wait for the audio buffer to fill before starting playback.

        On first play, audio generation may be slow due to JIT compilation,
        lazy imports, and memory allocation. We wait longer (5s timeout) to
        ensure enough data is buffered before playback begins, reducing the
        chance of immediate buffer underruns.
        """
        if not self._buffer_device:
            return

        # Target 1.5 seconds of audio (or half the ring buffer, whichever is larger)
        # to provide adequate headroom for initial generation slowness
        target_seconds = max(min_seconds, min(self._ring_buffer_seconds / 2.0, 1.5))
        target_bytes = int(target_seconds * self._sample_rate * _BYTES_PER_FRAME)
        deadline = time.monotonic() + timeout_seconds

        while self._buffer_device.queued_bytes() < target_bytes and time.monotonic() < deadline:
            # Process Qt events to keep UI responsive during initial buffering
            if QCoreApplication is not None:
                QCoreApplication.processEvents()
            try:
                QThread.msleep(10)
            except Exception:
                time.sleep(0.010)

    def _render_full_stream_headless(self) -> None:
        """Render the entire stream synchronously when Qt threads are unavailable."""
        if not self._worker or not self._buffer_device:
            return

        # Use consistent chunk size matching the threaded process loop
        chunk_size = 2048
        while True:
            remaining = self._worker.total_samples - self._worker._playback_sample
            if remaining <= 0:
                break

            this_chunk_size = min(chunk_size, remaining)
            chunk = self._worker._generate_next_chunk(
                self._worker._playback_sample,
                this_chunk_size,
                self._worker._playback_step_states,
            )
            if chunk is None:
                break

            self._worker._playback_sample += chunk.shape[0]
            pcm = self._worker._float_to_pcm(chunk)
            self._buffer_device.enqueue(pcm)

        self._worker._emit_progress()

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
            # Buffer underrun or finished - check if there's more data coming
            # and resume if needed. This handles the case where initial audio
            # generation is slow and causes a temporary underrun.
            if self._worker and self._worker.current_sample < self._worker.total_samples:
                # Worker is still generating - need to wait for sufficient buffer
                # before resuming to prevent immediate re-underrun
                min_resume_bytes = int(0.25 * self._sample_rate * _BYTES_PER_FRAME)  # 250ms
                if self._buffer_device and self._buffer_device.queued_bytes() >= min_resume_bytes:
                    # Enough data available, restart playback
                    self._restart_audio_output()
                else:
                    # Wait for buffer to fill before resuming
                    QTimer.singleShot(25, self._try_resume_after_underrun)

    def _restart_audio_output(self) -> None:  # pragma: no cover - Qt runtime
        """Restart audio output after underrun with fresh buffer connection."""
        if not self._audio_output or not self._buffer_device:
            return

        state = self._audio_output.state() if hasattr(self._audio_output, 'state') else None
        if state == QAudio.IdleState or state == QAudio.StoppedState:
            # Full restart for clean state
            self._audio_output.stop()
            self._audio_output.start(self._buffer_device)
        elif hasattr(self._audio_output, 'resume'):
            self._audio_output.resume()

    def _try_resume_after_underrun(self) -> None:  # pragma: no cover - Qt runtime
        """Attempt to resume audio output after a buffer underrun."""
        if not self._audio_output or not self._buffer_device:
            return

        # Require minimum buffer before resuming to prevent stuttering
        min_resume_bytes = int(0.25 * self._sample_rate * _BYTES_PER_FRAME)  # 250ms
        current_bytes = self._buffer_device.queued_bytes()

        if current_bytes >= min_resume_bytes:
            # Sufficient buffer built up, restart playback
            self._restart_audio_output()
        elif self._worker and self._worker.current_sample < self._worker.total_samples:
            # Still waiting for buffer to fill - use shorter interval for responsiveness
            QTimer.singleShot(20, self._try_resume_after_underrun)


__all__ = ["SessionStreamPlayer"]


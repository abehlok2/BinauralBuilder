"""Incremental audio streaming utilities for session playback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - import guard mirrors UI dialogs
    from PyQt5.QtCore import QBuffer, QIODevice, QObject, QTimer
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
    
    class _DummyQTimer:
        def __init__(self, parent=None): pass
        def start(self, ms=None): pass
        def stop(self): pass
        def setInterval(self, ms): pass
        @property
        def timeout(self): return _DummySignal()
        
    class _DummySignal:
        def connect(self, slot): pass
        
    QTimer = _DummyQTimer # type: ignore

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
        
        self._playback_sample = 0
        self._playback_step_states: Dict[int, List[dict]] = {}
        self._step_infos: List[_StepPlaybackInfo] = []
        self._total_samples_estimate = 0

        self._progress_callback: Optional[Callable[[float], None]] = None
        self._time_remaining_callback: Optional[Callable[[float], None]] = None
        
        self._recalculate_timeline()
        
        self._refill_timer = QTimer(self)
        if hasattr(self._refill_timer, "timeout"):
             self._refill_timer.timeout.connect(self._check_buffer_status)
        self._refill_timer.setInterval(50) # Check every 50ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_progress_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._progress_callback = callback

    def set_time_remaining_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        self._time_remaining_callback = callback

    def start(self, use_prebuffer: bool = False) -> None:
        """Start playback."""
        self.stop()
        self._use_prebuffer = bool(use_prebuffer)
        self._playback_sample = 0
        self._playback_step_states = {}
        
        fmt = self._build_format()

        if self._use_prebuffer:
            data = self._render_full_audio()
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
            self._refill_timer.start()

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 - 1.0)."""
        if self._audio_output:
            self._audio_output.setVolume(volume)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._total_samples_estimate / self._sample_rate

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        return self._playback_sample / self._sample_rate

    def seek(self, time_seconds: float) -> None:
        """Seek to a specific time in seconds."""
        target_sample = int(time_seconds * self._sample_rate)
        target_sample = max(0, min(target_sample, self._total_samples_estimate))
        
        if self._use_prebuffer and self._prebuffer_device:
            # Seek in pre-rendered buffer
            byte_offset = target_sample * _BYTES_PER_FRAME
            # Align to frame boundary
            byte_offset = (byte_offset // _BYTES_PER_FRAME) * _BYTES_PER_FRAME
            self._prebuffer_device.seek(byte_offset)
            self._playback_sample = target_sample
            return

        # Incremental seeking
        self.stop() # Stop current playback to reset buffers
        
        # Reset state for incremental playback
        self._playback_sample = target_sample
        self._playback_step_states = {} # Clear states, they will regenerate from scratch at new position
        # Note: This means LFOs might reset if we seek into the middle of a step.
        # Ideally we would fast-forward generation, but that's expensive.
        # For now, accepting state reset on seek.
        
        # Restart stream
        fmt = self._build_format()
        device = _PCMBufferDevice()
        device.open(QIODevice.ReadOnly)
        self._buffer_device = device
        self._audio_output = self._create_audio_output(fmt)
        
        # Prime buffer - this will trigger _generate_next_chunk
        self._prime_ring_buffer()
        self._audio_output.start(device)
        self._refill_timer.start()

    def pause(self) -> None:
        if self._audio_output:
            self._refill_timer.stop()
            self._audio_output.suspend()

    def resume(self) -> None:
        if self._audio_output:
            self._prime_ring_buffer()
            self._audio_output.resume()
            self._refill_timer.start()

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
        self._playback_sample = 0
        self._playback_step_states = {}
        if hasattr(self, "_refill_timer"):
            self._refill_timer.stop()

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
            
            # Start sample for this step in the global timeline
            # If it's the first step, it starts at 0.
            # If it's not, it starts 'prev_crossfade_samples' earlier than the "end" of the previous step's exclusive region.
            # Wait, standard logic:
            # Step N starts at `current_time_sample`.
            # Step N+1 starts at `current_time_sample + samples - crossfade_samples`.
            
            start_sample = current_time_sample
            end_sample = start_sample + samples
            
            # Fade in from previous step
            fade_in_len = prev_crossfade_samples if i > 0 else 0
            # Fade out to next step
            fade_out_len = crossfade_samples
            
            # Previous step's crossfade curve determines our fade in? 
            # Usually: 
            # Prev Step Fade Out: using Prev Step's curve.
            # This Step Fade In: using Prev Step's curve (complementary).
            # So fade_in_curve comes from previous step.
            
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
            
            # Advance time
            # The next step starts 'crossfade_samples' before this step ends.
            advance = max(0, samples - crossfade_samples)
            current_time_sample += advance
            
            prev_crossfade_samples = crossfade_samples
            
        # Total samples is the end of the last step
        if self._step_infos:
            self._total_samples_estimate = self._step_infos[-1].end_sample
        else:
            self._total_samples_estimate = 0

    def _check_buffer_status(self) -> None:
        """Periodically check buffer level and refill if needed."""
        if not self._buffer_device or self._use_prebuffer:
            return
        
        # If we are near the end, don't buffer too much? 
        # _generate_next_chunk handles end of stream.
        
        # Target: keep buffer at least 50% full
        target_bytes = int(self._ring_buffer_seconds * self._sample_rate * _BYTES_PER_FRAME)
        current_bytes = self._buffer_device.queued_bytes()
        
        if current_bytes < target_bytes * 0.5:
            # Refill up to target
            self._prime_ring_buffer()

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
        # Pre-render using the same chunk generation logic
        # We process in chunks to avoid huge memory spikes during generation, 
        # but we concatenate everything at the end.
        chunk_size = 4096 * 4
        current_sample = 0
        chunks: List[np.ndarray] = []
        step_states: Dict[int, List[dict]] = {}
        
        while current_sample < self._total_samples_estimate:
            # Generate next chunk
            chunk = self._generate_next_chunk(current_sample, chunk_size, step_states)
            if chunk is None or chunk.size == 0:
                break
            chunks.append(chunk)
            current_sample += chunk.shape[0]
            
        if chunks:
            audio = np.concatenate(chunks, axis=0)
        else:
            audio = np.zeros((0, 2), dtype=np.float32)
        return self._float_to_pcm(audio)

    def _next_float_chunk(self) -> Optional[np.ndarray]:
        # Generate chunk for current playback position
        chunk_size = 4096 # 100ms approx at 44.1k
        chunk = self._generate_next_chunk(self._playback_sample, chunk_size, self._playback_step_states)
        if chunk is not None:
            self._playback_sample += chunk.shape[0]
        return chunk

    def _generate_next_chunk(self, start_sample: int, max_frames: int, step_states: Dict[int, List[dict]]) -> Optional[np.ndarray]:
        if start_sample >= self._total_samples_estimate:
            return None
            
        end_sample = min(start_sample + max_frames, self._total_samples_estimate)
        num_frames = end_sample - start_sample
        
        if num_frames <= 0:
            return None
            
        mix_buffer = np.zeros((num_frames, 2), dtype=np.float32)
        
        # Find active steps
        # Optimization: could use binary search or keep track of active indices, 
        # but linear scan is fine for small number of steps.
        
        for info in self._step_infos:
            # Check overlap
            if info.end_sample <= start_sample:
                continue
            if info.start_sample >= end_sample:
                break # Sorted by start time, so we can stop
                
            # Calculate overlap range relative to the chunk
            chunk_rel_start = max(0, info.start_sample - start_sample)
            chunk_rel_end = min(num_frames, info.end_sample - start_sample)
            
            # Calculate overlap range relative to the step
            step_rel_start = max(0, start_sample - info.start_sample)
            step_rel_end = step_rel_start + (chunk_rel_end - chunk_rel_start)
            
            gen_len = step_rel_end - step_rel_start
            
            if gen_len <= 0:
                continue
                
            # Generate audio for this step segment
            chunk_start_time = step_rel_start / self._sample_rate
            duration = gen_len / self._sample_rate
            
            current_states = step_states.get(info.index)
            
            # Call generation
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
            
            # Ensure shape matches
            if audio.shape[0] != gen_len:
                # Pad or truncate
                if audio.shape[0] < gen_len:
                    audio = np.pad(audio, ((0, gen_len - audio.shape[0]), (0, 0)))
                else:
                    audio = audio[:gen_len]
            
            # Apply Fades
            # We need to apply fade in/out based on position in step
            
            # Fade In
            if info.fade_in_samples > 0 and step_rel_start < info.fade_in_samples:
                # We are in fade in region
                # Calculate local fade indices
                fade_start_idx = 0 # relative to audio chunk
                fade_end_idx = min(gen_len, info.fade_in_samples - step_rel_start)
                
                # Calculate progress (0.0 to 1.0)
                # Global progress in fade: step_rel_start / fade_in_samples
                
                start_p = step_rel_start / info.fade_in_samples
                end_p = (step_rel_start + fade_end_idx) / info.fade_in_samples
                
                curve = np.linspace(start_p, end_p, fade_end_idx)
                # Apply curve shape (linear for now, TODO: support others)
                # For crossfade, we usually want 'linear' or 'equal_power'
                # If curve is linear:
                envelope = curve
                
                audio[:fade_end_idx] *= envelope[:, np.newaxis]
                
            # Fade Out
            step_duration_samples = info.end_sample - info.start_sample
            fade_out_start_sample = step_duration_samples - info.fade_out_samples
            
            if info.fade_out_samples > 0 and step_rel_end > fade_out_start_sample:
                # We are in fade out region
                local_start = max(0, fade_out_start_sample - step_rel_start)
                local_end = gen_len
                
                # Progress 0.0 (start of fade out) to 1.0 (end of step)
                # Global pos: step_rel_start + local_i
                # Progress = (global_pos - fade_out_start_sample) / fade_out_samples
                
                start_p = (step_rel_start + local_start - fade_out_start_sample) / info.fade_out_samples
                end_p = (step_rel_start + local_end - fade_out_start_sample) / info.fade_out_samples
                
                curve = np.linspace(start_p, end_p, local_end - local_start)
                envelope = 1.0 - curve # Fade out
                
                audio[local_start:local_end] *= envelope[:, np.newaxis]
            
            # Mix into buffer
            mix_buffer[chunk_rel_start:chunk_rel_end] += audio
            
        return mix_buffer

    def _enqueue_audio_chunk(self, chunk: np.ndarray) -> None:
        if not self._buffer_device or chunk.size == 0:
            return
        pcm = self._float_to_pcm(chunk)
        self._buffer_device.enqueue(pcm)
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
            ratio = min(max(self._playback_sample / total, 0.0), 1.0)
            self._progress_callback(ratio)
        if self._time_remaining_callback:
            remaining_samples = max(self._total_samples_estimate - self._playback_sample, 0)
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

"""Rust-based streaming audio player for session playback.

This module provides a streaming player that uses the Rust realtime_backend
for audio generation and playback. If the Rust backend is not available,
it falls back to the pure Python SessionStreamPlayer.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, Optional, TYPE_CHECKING

# Try to import the Rust backend
try:
    import realtime_backend as _rust_backend
    RUST_BACKEND_AVAILABLE = True
except ImportError:
    _rust_backend = None
    RUST_BACKEND_AVAILABLE = False

try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    QObject = object
    QTimer = None
    pyqtSignal = None


def is_rust_backend_available() -> bool:
    """Check if the Rust realtime backend is available."""
    return RUST_BACKEND_AVAILABLE


class RustStreamPlayer(QObject if QT_AVAILABLE else object):
    """Streaming audio player using the Rust realtime backend.

    This player uses the Rust-based cpal audio backend for low-latency,
    efficient audio streaming. It manages position tracking and provides
    a similar API to SessionStreamPlayer.

    The Rust backend handles all audio generation and device output
    on a separate thread, making the Python side lightweight.
    """

    def __init__(
        self,
        track_data: Dict[str, object],
        parent: Optional[object] = None,
    ) -> None:
        if QT_AVAILABLE and QObject is not object:
            super().__init__(parent)

        self._track_data = self._prepare_track_data(track_data or {})
        self._track_json = json.dumps(self._track_data)

        # Calculate duration from track data
        self._duration = self._calculate_duration()

        # Playback state
        self._is_playing = False
        self._is_paused = False
        self._position = 0.0
        self._start_timestamp: Optional[float] = None
        self._pause_timestamp: Optional[float] = None

        # Volume (stored locally, Rust backend uses native volume)
        self._volume = 1.0

        # Callbacks for progress updates
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._time_remaining_callback: Optional[Callable[[float], None]] = None

        # Position tracking timer
        if QT_AVAILABLE and QTimer is not None:
            self._position_timer = QTimer()
            self._position_timer.setInterval(50)  # 50ms update rate
            self._position_timer.timeout.connect(self._update_position)
        else:
            self._position_timer = None

    def _calculate_duration(self) -> float:
        """Calculate total duration from track data."""
        steps = self._track_data.get("steps", [])
        global_settings = self._track_data.get("global_settings", {})
        crossfade_duration = float(global_settings.get("crossfade_duration", 0.0))

        total_time = 0.0
        prev_crossfade = 0.0

        for i, step in enumerate(steps):
            step_duration = float(step.get("duration", 0.0))
            step_crossfade = float(step.get("crossfade_duration", crossfade_duration))

            if i == 0:
                total_time += step_duration
            else:
                # Account for overlap from previous step's crossfade
                total_time += max(0.0, step_duration - prev_crossfade)

            prev_crossfade = step_crossfade

        return total_time

    def _prepare_track_data(self, track_data: Dict[str, object]) -> Dict[str, object]:
        """Normalize track data for the Rust backend.

        The Rust models treat ``clips`` and ``overlay_clips`` as aliases. If
        both keys are present in the JSON payload, Serde will raise a duplicate
        field error. Merge any clip aliases into a single ``clips`` entry and
        drop the redundant key before serializing.
        """

        normalized: Dict[str, object] = dict(track_data)

        overlay_clips = normalized.pop("overlay_clips", None)
        clips = normalized.get("clips")

        merged_clips: list = []
        if isinstance(clips, list):
            merged_clips.extend(clips)
        if isinstance(overlay_clips, list):
            merged_clips.extend(overlay_clips)

        normalized["clips"] = merged_clips
        return normalized

    def set_progress_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        """Set callback for progress updates (0.0 - 1.0)."""
        self._progress_callback = callback

    def set_time_remaining_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        """Set callback for time remaining updates (seconds)."""
        self._time_remaining_callback = callback

    @property
    def duration(self) -> float:
        """Total playback duration in seconds."""
        return self._duration

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        if not self._is_playing or self._start_timestamp is None:
            return self._position

        if self._is_paused:
            return self._position

        # Calculate current position from wall clock
        elapsed = time.monotonic() - self._start_timestamp
        return min(self._position + elapsed, self._duration)

    def start(self, use_prebuffer: bool = False) -> None:
        """Start playback from the beginning.

        Args:
            use_prebuffer: Ignored for Rust backend (always uses streaming).
        """
        if not RUST_BACKEND_AVAILABLE:
            raise RuntimeError("Rust realtime backend is not available")

        self.stop()

        # Start the Rust audio stream
        _rust_backend.start_stream(self._track_json, 0.0)

        self._is_playing = True
        self._is_paused = False
        self._position = 0.0
        self._start_timestamp = time.monotonic()

        # Start position tracking
        if self._position_timer is not None:
            self._position_timer.start()

    def stop(self) -> None:
        """Stop playback completely."""
        if not RUST_BACKEND_AVAILABLE:
            return

        if self._is_playing:
            _rust_backend.stop_stream()

        self._is_playing = False
        self._is_paused = False
        self._position = 0.0
        self._start_timestamp = None
        self._pause_timestamp = None

        if self._position_timer is not None:
            self._position_timer.stop()

    def pause(self) -> None:
        """Pause playback."""
        if not RUST_BACKEND_AVAILABLE or not self._is_playing:
            return

        if not self._is_paused:
            _rust_backend.pause_stream()
            # Capture current position before pausing
            if self._start_timestamp is not None:
                elapsed = time.monotonic() - self._start_timestamp
                self._position = min(self._position + elapsed, self._duration)
            self._is_paused = True
            self._pause_timestamp = time.monotonic()

    def resume(self) -> None:
        """Resume paused playback."""
        if not RUST_BACKEND_AVAILABLE or not self._is_playing:
            return

        if self._is_paused:
            _rust_backend.resume_stream()
            self._is_paused = False
            self._start_timestamp = time.monotonic()

    def seek(self, time_seconds: float) -> None:
        """Seek to a specific time position.

        Args:
            time_seconds: Target position in seconds.
        """
        if not RUST_BACKEND_AVAILABLE:
            return

        time_seconds = max(0.0, min(time_seconds, self._duration))

        if self._is_playing:
            _rust_backend.start_from(time_seconds)
            self._position = time_seconds
            if not self._is_paused:
                self._start_timestamp = time.monotonic()
        else:
            self._position = time_seconds

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 - 1.0).

        Note: The Rust backend uses the system's native volume control.
        This method stores the volume locally for compatibility but
        doesn't directly control the Rust backend's volume.
        """
        self._volume = max(0.0, min(1.0, volume))

    def update_track(self, track_data: Dict[str, object]) -> None:
        """Update the track data while playing.

        This allows for live updates to the session without stopping playback.

        Args:
            track_data: New track data dictionary.
        """
        if not RUST_BACKEND_AVAILABLE:
            return

        self._track_data = self._prepare_track_data(track_data)
        self._track_json = json.dumps(self._track_data)
        self._duration = self._calculate_duration()

        if self._is_playing:
            _rust_backend.update_track(self._track_json)

    def _update_position(self) -> None:
        """Update position tracking and emit callbacks."""
        if not self._is_playing:
            return

        current_pos = self.position

        # Emit progress callback
        if self._progress_callback is not None and self._duration > 0:
            ratio = min(current_pos / self._duration, 1.0)
            self._progress_callback(ratio)

        # Emit time remaining callback
        if self._time_remaining_callback is not None:
            remaining = max(0.0, self._duration - current_pos)
            self._time_remaining_callback(remaining)

        # Check if playback finished
        if current_pos >= self._duration:
            self.stop()


class HybridStreamPlayer:
    """Factory class that creates the best available stream player.

    This class automatically selects between:
    - RustStreamPlayer (if Rust backend is available) for streaming
    - SessionStreamPlayer (Python fallback) if Rust is not available
    """

    @staticmethod
    def create(
        track_data: Dict[str, object],
        parent: Optional[object] = None,
        prefer_rust: bool = True,
    ) -> object:
        """Create a stream player instance.

        Args:
            track_data: The track data dictionary for playback.
            parent: Optional Qt parent object.
            prefer_rust: If True, prefer Rust backend when available.

        Returns:
            A stream player instance (RustStreamPlayer or SessionStreamPlayer).
        """
        if prefer_rust and RUST_BACKEND_AVAILABLE:
            return RustStreamPlayer(track_data, parent)

        # Fall back to Python SessionStreamPlayer
        from src.audio.session_stream import SessionStreamPlayer
        return SessionStreamPlayer(track_data, parent)


__all__ = [
    "RustStreamPlayer",
    "HybridStreamPlayer",
    "is_rust_backend_available",
    "RUST_BACKEND_AVAILABLE",
]

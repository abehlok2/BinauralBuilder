"""Audio session data structures and helpers."""

from .session_model import (
    Session,
    SessionStep,
    SessionPresetChoice,
    build_binaural_preset_catalog,
    build_noise_preset_catalog,
    session_to_track_data,
)
from .session_engine import SessionAssembler
from .session_builder_launcher import launch_session_builder
from .session_stream import SessionStreamPlayer
from .rust_stream_player import (
    RustStreamPlayer,
    HybridStreamPlayer,
    is_rust_backend_available,
)

__all__ = [
    "Session",
    "SessionStep",
    "SessionPresetChoice",
    "build_binaural_preset_catalog",
    "build_noise_preset_catalog",
    "session_to_track_data",
    "SessionAssembler",
    "SessionStreamPlayer",
    "RustStreamPlayer",
    "HybridStreamPlayer",
    "is_rust_backend_available",
    "launch_session_builder",
]

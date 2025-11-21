"""Standalone-friendly binaural synthesis and session assembly API."""

from .session import (
    Session,
    SessionStep,
    SessionPresetChoice,
    build_binaural_preset_catalog,
    build_noise_preset_catalog,
    session_to_track_data,
)
from .assembly import SessionAssembler
from .synthesis import assemble_track_from_data, generate_audio

__all__ = [
    "Session",
    "SessionStep",
    "SessionPresetChoice",
    "build_binaural_preset_catalog",
    "build_noise_preset_catalog",
    "session_to_track_data",
    "SessionAssembler",
    "assemble_track_from_data",
    "generate_audio",
]

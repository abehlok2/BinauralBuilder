"""Compatibility wrapper for session planning helpers.

All core data structures and conversion utilities now live in the
:mod:`binauralbuilder_core.session` module so they can be imported as a
standalone package.  The exports remain available under
:mod:`src.audio.session_model` for the existing GUI code.
"""

from binauralbuilder_core.session import (
    Session,
    SessionStep,
    SessionPresetChoice,
    build_binaural_preset_catalog,
    build_noise_preset_catalog,
    session_to_track_data,
    MAX_INDIVIDUAL_GAIN,
)

__all__ = [
    "Session",
    "SessionStep",
    "SessionPresetChoice",
    "build_binaural_preset_catalog",
    "build_noise_preset_catalog",
    "session_to_track_data",
    "MAX_INDIVIDUAL_GAIN",
]

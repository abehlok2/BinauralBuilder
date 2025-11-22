"""Wrapper exports for binaural synthesis utilities."""

from .synth_functions.sound_creator import (
    assemble_track_from_data,
    generate_audio,
)

__all__ = ["assemble_track_from_data", "generate_audio"]

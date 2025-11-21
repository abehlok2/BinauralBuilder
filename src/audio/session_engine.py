"""Compatibility wrapper for session assembly helpers.

Session assembly now lives in :mod:`binauralbuilder_core.assembly` to allow
standalone use while keeping the existing GUI imports intact.
"""

from binauralbuilder_core.assembly import SessionAssembler

__all__ = ["SessionAssembler"]

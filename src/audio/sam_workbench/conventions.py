"""Canonical coordinate, audio, and array conventions for the SAM workbench.

Coordinates
-----------
A right-handed Cartesian frame attached to the listener:

* ``+x`` listener forward,
* ``+y`` listener left,
* ``+z`` up,
* azimuth ``0 deg`` front, positive azimuth toward the left,
* elevation positive upward,
* distance in metres.

The left receiver sits at positive ``y`` and the right receiver at negative
``y``.  Legacy ``slab`` coordinates (``+x`` right, ``+y`` forward, azimuth
positive to the right) and GUI scene/pixel coordinates are *not* this frame;
they must be converted in a named legacy converter at the adapter boundary and
never leak into the core.

Audio and time
--------------
* ``float64`` for phase/delay accumulation, ``float32`` for bulk buffers.
* Time is seconds at the domain boundary and integer samples inside render
  loops; every core render call receives an absolute ``start_sample``.
* Phase is radians internally, degrees only at user-facing boundaries.
* Level is linear gain internally, decibels at user-facing boundaries.
* The core returns channel-major ``(channels, frames)`` audio.  The
  BinauralBuilder adapter is the single place that converts to the legacy
  frame-major ``(frames, 2)`` contract.
"""

from __future__ import annotations

import math
from typing import Final, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.audio.sam_workbench.version import SCHEMA_VERSION

# --- audio -----------------------------------------------------------------

DEFAULT_SAMPLE_RATE_HZ: Final[int] = 44_100
SUPPORTED_SAMPLE_RATES_HZ: Final[tuple[int, ...]] = (44_100, 48_000, 88_200, 96_000, 192_000)
MIN_SAMPLE_RATE_HZ: Final[int] = 8_000
MAX_SAMPLE_RATE_HZ: Final[int] = 192_000

DEFAULT_PREVIEW_BLOCK_SIZE: Final[int] = 512
MIN_PREVIEW_BLOCK_SIZE: Final[int] = 128
MAX_PREVIEW_BLOCK_SIZE: Final[int] = 2_048
DEFAULT_OFFLINE_BLOCK_SIZE: Final[int] = 4_096

DEFAULT_MASTER_GAIN_DB: Final[float] = -6.0
DEFAULT_LIMITER_CEILING_DBFS: Final[float] = -1.0
DEFAULT_SPEED_OF_SOUND_M_S: Final[float] = 343.0

#: Decibel value treated as digital silence when converting to linear gain.
SILENCE_DB: Final[float] = -300.0

# --- arrays ----------------------------------------------------------------

CHANNEL_COUNT: Final[int] = 2
CHANNEL_LEFT: Final[int] = 0
CHANNEL_RIGHT: Final[int] = 1
CHANNEL_ORDER: Final[tuple[str, ...]] = ("left", "right")

#: Accumulation dtype for phase, delay, and analysis.
ACCUMULATOR_DTYPE: Final[np.dtype] = np.dtype(np.float64)
#: Bulk audio buffer dtype.
AUDIO_DTYPE: Final[np.dtype] = np.dtype(np.float32)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_SAMPLE_RATE_HZ",
    "SUPPORTED_SAMPLE_RATES_HZ",
    "MIN_SAMPLE_RATE_HZ",
    "MAX_SAMPLE_RATE_HZ",
    "DEFAULT_PREVIEW_BLOCK_SIZE",
    "MIN_PREVIEW_BLOCK_SIZE",
    "MAX_PREVIEW_BLOCK_SIZE",
    "DEFAULT_OFFLINE_BLOCK_SIZE",
    "DEFAULT_MASTER_GAIN_DB",
    "DEFAULT_LIMITER_CEILING_DBFS",
    "DEFAULT_SPEED_OF_SOUND_M_S",
    "SILENCE_DB",
    "CHANNEL_COUNT",
    "CHANNEL_LEFT",
    "CHANNEL_RIGHT",
    "CHANNEL_ORDER",
    "ACCUMULATOR_DTYPE",
    "AUDIO_DTYPE",
    "spherical_to_cartesian_m",
    "cartesian_to_spherical_deg",
    "wrap_degrees",
    "wrap_radians",
    "db_to_linear",
    "linear_to_db",
    "seconds_to_samples",
    "samples_to_seconds",
    "intersect_window",
    "to_channel_major",
    "to_frame_major",
]


# --- coordinates -----------------------------------------------------------


def spherical_to_cartesian_m(
    azimuth_deg: float, elevation_deg: float, radius_m: float = 1.0
) -> NDArray[np.float64]:
    """Convert canonical spherical coordinates to metres in the listener frame.

    Azimuth ``0 deg`` is front and positive azimuth turns toward the listener's
    left (``+y``); elevation is positive upward (``+z``).
    """

    azimuth_rad = math.radians(float(azimuth_deg))
    elevation_rad = math.radians(float(elevation_deg))
    radius = float(radius_m)
    horizontal = radius * math.cos(elevation_rad)
    return np.array(
        (
            horizontal * math.cos(azimuth_rad),
            horizontal * math.sin(azimuth_rad),
            radius * math.sin(elevation_rad),
        ),
        dtype=np.float64,
    )


def cartesian_to_spherical_deg(point: Sequence[float] | NDArray[np.floating]) -> tuple[float, float, float]:
    """Convert a listener-frame Cartesian point to ``(azimuth_deg, elevation_deg, radius_m)``.

    Azimuth is wrapped to ``(-180, 180]``.  The origin maps to
    ``(0.0, 0.0, 0.0)`` rather than raising, so degenerate trajectories stay
    finite.
    """

    coordinates = np.asarray(point, dtype=np.float64)
    if coordinates.shape != (3,):
        raise ValueError(f"expected a 3-element point, got shape {coordinates.shape}")
    x, y, z = (float(value) for value in coordinates)
    radius = math.sqrt(x * x + y * y + z * z)
    if radius == 0.0:
        return (0.0, 0.0, 0.0)
    azimuth_deg = wrap_degrees(math.degrees(math.atan2(y, x)))
    elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, z / radius))))
    return (azimuth_deg, elevation_deg, radius)


def wrap_degrees(angle_deg: float) -> float:
    """Wrap an angle in degrees to the half-open canonical range ``(-180, 180]``."""

    wrapped = math.fmod(float(angle_deg) + 180.0, 360.0)
    if wrapped <= 0.0:
        wrapped += 360.0
    return wrapped - 180.0


def wrap_radians(angle_rad: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]``."""

    wrapped = math.fmod(float(angle_rad) + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


# --- levels ----------------------------------------------------------------


def db_to_linear(level_db: float) -> float:
    """Convert decibels to linear gain; values at or below :data:`SILENCE_DB` are ``0``."""

    value = float(level_db)
    if not math.isfinite(value) or value <= SILENCE_DB:
        return 0.0
    return 10.0 ** (value / 20.0)


def linear_to_db(gain_linear: float, floor_db: float = SILENCE_DB) -> float:
    """Convert linear gain to decibels, clamping non-positive gains to ``floor_db``."""

    value = float(gain_linear)
    if not math.isfinite(value) or value <= 0.0:
        return float(floor_db)
    return max(float(floor_db), 20.0 * math.log10(value))


# --- time ------------------------------------------------------------------


def seconds_to_samples(seconds: float, sample_rate_hz: float) -> int:
    """Convert seconds to an absolute sample index.

    The documented rounding rule is round-half-away-from-zero, so a legacy
    ``chunk_start_time`` always maps to the same absolute ``start_sample``
    regardless of platform float-to-int behaviour.
    """

    value = float(seconds) * float(sample_rate_hz)
    if not math.isfinite(value):
        raise ValueError(f"cannot convert non-finite time {seconds!r} to samples")
    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


def samples_to_seconds(samples: int, sample_rate_hz: float) -> float:
    """Convert an absolute sample index to seconds."""

    return int(samples) / float(sample_rate_hz)


def intersect_window(
    source_start: int, source_end: int | None, window_start: int, window_frames: int
) -> tuple[int, int, int] | None:
    """Intersect a source's absolute interval with a render window.

    Returns ``(placement, source_offset, frames)``: where the overlap sits in
    the window, how far into the source's own timeline it begins, and how long
    it is - or ``None`` when the two do not overlap.  ``source_end`` of ``None``
    means the source runs as long as the window asks.

    One implementation, deliberately.  Intersecting only the near end is how a
    source came to render past its own end, and how a window opening after a
    source had finished came to render it from its beginning.
    """

    window_end = int(window_start) + int(window_frames)
    end = window_end if source_end is None else min(int(source_end), window_end)
    begin = max(int(source_start), int(window_start))
    if end <= begin:
        return None
    return (begin - int(window_start), begin - int(source_start), end - begin)


# --- array layout ----------------------------------------------------------


def to_channel_major(audio: NDArray[np.floating] | Iterable[float]) -> NDArray[np.floating]:
    """Return frame-major ``(frames, channels)`` audio as core ``(channels, frames)``."""

    array = np.asarray(audio)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    if array.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"expected {CHANNEL_COUNT} columns, got shape {array.shape}")
    return np.ascontiguousarray(array.T)


def to_frame_major(audio: NDArray[np.floating] | Iterable[float]) -> NDArray[np.floating]:
    """Return core ``(channels, frames)`` audio as legacy frame-major ``(frames, channels)``."""

    array = np.asarray(audio)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    if array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"expected {CHANNEL_COUNT} rows, got shape {array.shape}")
    return np.ascontiguousarray(array.T)

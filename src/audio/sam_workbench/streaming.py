"""Rendering a long track without holding all of it in memory.

An hour of stereo audio at 44.1 kHz is 1.3 GB as float32. The export path held
that, then a float64 copy of it to normalize, then an int16 copy to encode -
three full-length arrays alive at once, for a render whose output is a file.

Exact normalization is what forces this: the final gain depends on the peak,
and the peak is not known until the last sample has been rendered. So the
render happens twice over the same data rather than once over three copies of
it: pass one streams float32 to temporary storage and measures the peak as it
goes; pass two reads that back in chunks, applies the gain, and encodes.

Peak measurement is exact - it is a running maximum over every sample, not an
estimate from a subset - so the result is the same normalization the in-memory
path produced, reached without ever materializing the track.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_STREAM_CHUNK_FRAMES",
    "PeakMeter",
    "TrackStream",
    "stream_normalized_export",
]

#: How much audio moves through memory at a time. Large enough that per-chunk
#: overhead is irrelevant, small enough that the working set stays a few
#: megabytes however long the track is.
DEFAULT_STREAM_CHUNK_FRAMES = 1 << 16

#: Below this, holding the track in memory is cheaper than a round trip
#: through the filesystem and the streaming path buys nothing.
STREAMING_THRESHOLD_FRAMES = 1 << 21


class PeakMeter:
    """A running absolute peak and sample count over a streamed render."""

    __slots__ = ("peak", "frames", "non_finite")

    def __init__(self) -> None:
        self.peak = 0.0
        self.frames = 0
        self.non_finite = 0

    def update(self, block: NDArray[np.floating]) -> NDArray[np.floating]:
        """Measure a block, replacing anything non-finite with silence.

        A single NaN would otherwise poison the peak and silence the whole
        render, so it is counted and zeroed here rather than propagating into
        the normalization.
        """

        values = np.asarray(block)
        bad = ~np.isfinite(values)
        if bad.any():
            self.non_finite += int(np.count_nonzero(bad))
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.size:
            self.peak = max(self.peak, float(np.max(np.abs(values))))
            self.frames += int(values.shape[0])
        return values

    def gain_for(self, target_level: float) -> float:
        """The scale factor that puts the peak at ``target_level``."""

        if self.peak <= 1e-9:
            return 1.0
        return float(target_level) / self.peak


class TrackStream:
    """Bounded-memory temporary storage for one render, as raw float32.

    A flat binary file rather than a WAV, because this is scratch: nothing
    reads it but the second pass, and the second pass knows the shape.
    """

    def __init__(self, channels: int = 2, directory: str | os.PathLike | None = None) -> None:
        self.channels = int(channels)
        self.meter = PeakMeter()
        handle, path = tempfile.mkstemp(suffix=".f32", dir=directory)
        os.close(handle)
        self.path = Path(path)
        self._file = open(self.path, "wb")

    # --- writing ------------------------------------------------------------

    def write(self, block: NDArray[np.floating]) -> None:
        """Append one frame-major ``(frames, channels)`` block."""

        values = np.asarray(block, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(
                f"expected (frames, {self.channels}) audio, got {values.shape}"
            )
        if self._file is None:
            raise RuntimeError("this stream has been finished")
        self._file.write(self.meter.update(values).astype(np.float32, copy=False).tobytes())

    def finish(self) -> None:
        """Close the writing side; the stream can still be read."""

        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # --- reading ------------------------------------------------------------

    @property
    def frames(self) -> int:
        return self.meter.frames

    def blocks(self, chunk_frames: int = DEFAULT_STREAM_CHUNK_FRAMES) -> Iterator[NDArray[np.float32]]:
        """Read the render back, one bounded block at a time."""

        self.finish()
        width = self.channels * 4
        with open(self.path, "rb") as handle:
            while True:
                raw = handle.read(int(chunk_frames) * width)
                if not raw:
                    return
                yield np.frombuffer(raw, dtype=np.float32).reshape(-1, self.channels)

    def close(self) -> None:
        self.finish()
        with contextlib.suppress(OSError):
            self.path.unlink()

    def __enter__(self) -> "TrackStream":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def stream_normalized_export(
    blocks: Iterable[NDArray[np.floating]],
    output_path: str | os.PathLike,
    sample_rate_hz: int,
    *,
    target_level: float = 0.25,
    channels: int = 2,
    subtype: str = "PCM_16",
    chunk_frames: int = DEFAULT_STREAM_CHUNK_FRAMES,
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Render ``blocks`` to a normalized audio file without holding the track.

    Two passes over temporary storage rather than three copies in memory. The
    peak is measured exactly during the first, so the gain applied during the
    second is the one the in-memory path would have chosen.
    """

    import soundfile as sf

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TrackStream(channels=channels, directory=destination.parent) as stream:
        for block in blocks:
            stream.write(block)
        stream.finish()

        gain = stream.meter.gain_for(target_level)
        written = 0
        with sf.SoundFile(
            str(destination), "w", int(sample_rate_hz), channels, subtype=subtype
        ) as handle:
            for block in stream.blocks(chunk_frames):
                handle.write(_encode(block * gain, subtype))
                written += int(block.shape[0])
                if progress is not None and stream.frames:
                    progress(written / stream.frames)

        return {
            "frames": int(stream.frames),
            "peak": float(stream.meter.peak),
            "gain": float(gain),
            "nonFiniteSamples": int(stream.meter.non_finite),
            "sampleRateHz": int(sample_rate_hz),
            "channels": int(channels),
            "subtype": subtype,
        }


def _encode(block: NDArray[np.floating], subtype: str) -> NDArray:
    """Quantize exactly as the in-memory export path does.

    soundfile rounds a float to the nearest integer and scales by 32768; the
    in-memory path truncates toward zero and scales by 32767. The two differ by
    up to two least-significant bits - about -90 dBFS, inaudible - but choosing
    the streaming path is a memory decision, and a memory decision should not
    also change the samples. Doing the conversion here makes the two byte-identical.
    """

    if subtype != "PCM_16":
        return block
    # Kept in the block's own precision rather than widened: the in-memory path
    # scales a float32 track in float32, and widening here would round
    # differently in the last bit.
    return np.int16(np.clip(block * 32767, -32768, 32767))


def iter_track_blocks(
    track: NDArray[np.floating], chunk_frames: int = DEFAULT_STREAM_CHUNK_FRAMES
) -> Iterator[NDArray[np.floating]]:
    """Cut an in-memory track into blocks, for callers that already have one.

    A bridge rather than a destination: it lets the streaming encoder be used
    by a path that has not been converted to produce blocks of its own yet, so
    two of the three full-length copies go even before the first one does.
    """

    values = np.asarray(track)
    for start in range(0, int(values.shape[0]), int(chunk_frames)):
        yield values[start : start + int(chunk_frames)]

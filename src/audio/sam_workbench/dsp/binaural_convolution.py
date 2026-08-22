"""Binaural block convolution with one input transform per block.

A binaural render convolves the *same* mono signal with four filters at once:
left and right, current and outgoing during a crossfade. Running four
independent convolvers transforms that identical input four times, and the
forward transform is the expensive half of an overlap-save block.

This engine transforms the input once and multiplies the result by each filter
spectrum it needs. Overlap-save is what makes that possible: its state is the
tail of the *input*, not a partially finished output, so every filter sharing
one input history is automatically aligned with every other.

That also removes the need to keep a dormant outgoing convolver running. There
is no warm-up to preserve - an outgoing filter is just another spectrum
multiplied against the input history the engine already holds - so a filter
transition costs two extra inverse transforms while it lasts, and nothing at
all when it is not.

Both paths are exact. The partitioned path reassembles precisely the sum the
single long transform computes, to floating-point round-off, rather than
approximating it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .convolution import (
    MAX_PARTITIONS,
    MIN_PARTITION_SAMPLES,
    PARTITION_THRESHOLD_RATIO,
)
from .envelopes import milliseconds_to_frames

__all__ = ["BinauralConvolver", "BinauralFilterPair"]

#: Channel order, matching the workbench's channel-major convention.
LEFT, RIGHT = 0, 1


class BinauralFilterPair:
    """One direction's left/right impulse responses, and their spectra.

    Spectra are computed once per transform length and kept, because a filter
    is constant between direction changes while the input is not. Transforming
    the filters every block was work spent recomputing a constant.
    """

    __slots__ = ("taps", "_spectra", "_length", "_partitioned", "_partition")

    def __init__(self, taps: ArrayLike) -> None:
        values = np.asarray(taps, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 2:
            raise ValueError(f"a binaural filter pair must be (2, taps), got {values.shape}")
        self.taps = values
        self._spectra: NDArray[np.complex128] | None = None
        self._length = 0
        self._partitioned: NDArray[np.complex128] | None = None
        self._partition = 0

    @property
    def length(self) -> int:
        return int(self.taps.shape[1])

    def spectra(self, length: int) -> NDArray[np.complex128]:
        """``(2, length // 2 + 1)`` spectra at the requested transform length."""

        if self._spectra is None or self._length != length:
            self._spectra = np.fft.rfft(self.taps, length, axis=1)
            self._length = int(length)
        return self._spectra

    def partitioned_spectra(self, partition: int) -> NDArray[np.complex128]:
        """``(2, partitions, partition + 1)`` spectra for the partitioned path."""

        if self._partitioned is None or self._partition != partition:
            count = max(1, int(np.ceil(self.length / partition)))
            padded = np.zeros((2, count * partition), dtype=np.float64)
            padded[:, : self.length] = self.taps
            self._partitioned = np.fft.rfft(
                padded.reshape(2, count, partition), 2 * partition, axis=2
            )
            self._partition = int(partition)
        return self._partitioned

    @property
    def partitions(self) -> int:
        return 0 if self._partitioned is None else int(self._partitioned.shape[1])


class BinauralConvolver:
    """Convolve one mono stream into two channels, with crossfaded filters.

    ``process`` returns channel-major ``(2, frames)``. Filters are installed
    with :meth:`set_filters`; a change begins an equal-power crossfade over
    ``crossfade_ms`` during which both the outgoing and incoming filters are
    evaluated against the shared input transform.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float = 44100.0,
        crossfade_ms: float = 12.0,
        partition: int | None = None,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if crossfade_ms < 0.0:
            raise ValueError("crossfade_ms must not be negative")
        self.sample_rate_hz = float(sample_rate_hz)
        self.crossfade_ms = float(crossfade_ms)
        self._partition_override = partition
        self._current: BinauralFilterPair | None = None
        self._previous: BinauralFilterPair | None = None
        self._history = np.zeros(0, dtype=np.float64)
        self._fade_position = 0
        self._fade_frames = 0
        # Partitioned state: one frequency-domain input delay line, shared by
        # both ears and by both filters during a crossfade.
        self._delay_line: NDArray[np.complex128] | None = None
        self._delay_partition = 0
        self._previous_block = np.zeros(0, dtype=np.float64)

    # --- state --------------------------------------------------------------

    @property
    def is_fading(self) -> bool:
        return self._previous is not None and self._fade_position < self._fade_frames

    @property
    def taps(self) -> int:
        return 0 if self._current is None else self._current.length

    @property
    def history(self) -> NDArray[np.float64]:
        """The raw input tail, which is the whole of this engine's alignment."""

        return self._history

    def reset(self, filters: ArrayLike | None = None) -> None:
        """Clear history, optionally installing a filter pair outright."""

        self._previous = None
        self._fade_position = self._fade_frames = 0
        self._delay_line = None
        self._delay_partition = 0
        self._previous_block = np.zeros(0, dtype=np.float64)
        if filters is not None:
            self._current = BinauralFilterPair(filters)
        self._history = np.zeros(max(0, self.taps - 1), dtype=np.float64)

    def set_filters(self, filters: ArrayLike) -> bool:
        """Install a filter pair. Returns True when a crossfade began.

        The first pair is installed outright: fading into it would fade up from
        silence, so a render would open with a ramp nobody asked for.
        """

        pair = BinauralFilterPair(filters)
        if self._current is None:
            self._current = pair
            self._history = np.zeros(max(0, pair.length - 1), dtype=np.float64)
            return False
        if pair.length == self._current.length and np.array_equal(pair.taps, self._current.taps):
            return False

        fade = milliseconds_to_frames(self.crossfade_ms, self.sample_rate_hz)
        if fade <= 0:
            self._current = pair
            self._grow_history(pair.length)
            return False

        # A change arriving mid-fade collapses the fade that was running: the
        # value being faded from is what is currently audible, which is the
        # mixture, and approximating that by the outgoing filter alone is
        # closer than restarting from a filter two changes old.
        self._previous = self._current
        self._current = pair
        self._fade_frames = fade
        self._fade_position = 0
        self._grow_history(max(pair.length, self._previous.length))
        return True

    def _grow_history(self, taps: int) -> None:
        needed = max(0, taps - 1)
        if self._history.size < needed:
            self._history = np.concatenate(
                (np.zeros(needed - self._history.size), self._history)
            )

    def prime(self, history: ArrayLike) -> None:
        """Seed the input history, so the engine can be entered mid-stream."""

        tail = np.asarray(history, dtype=np.float64).reshape(-1)
        needed = max(0, self.taps - 1)
        if tail.size >= needed:
            self._history = tail[tail.size - needed :].copy()
        else:
            self._history = np.concatenate((np.zeros(needed - tail.size), tail))
        self._delay_line = None

    # --- processing ---------------------------------------------------------

    def _wants_partitioning(self, block_size: int) -> bool:
        if self._partition_override is not None:
            return bool(self._partition_override) and self.taps > self._partition_override
        if block_size < MIN_PARTITION_SAMPLES:
            return False
        if self.taps <= PARTITION_THRESHOLD_RATIO * block_size:
            return False
        return self.taps <= MAX_PARTITIONS * block_size

    def process(self, block: ArrayLike) -> NDArray[np.float64]:
        """Convolve one block into channel-major ``(2, frames)``."""

        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        if self._current is None:
            raise RuntimeError("set_filters() must be called before process()")
        if samples.size == 0:
            return np.zeros((2, 0), dtype=np.float64)

        if self._wants_partitioning(samples.size):
            output = self._process_partitioned(samples)
        else:
            self._delay_line = None
            output = self._process_plain(samples)

        if self._previous is not None:
            self._fade_position = min(self._fade_position + samples.size, self._fade_frames)
            if self._fade_position >= self._fade_frames:
                self._previous = None
        return output

    def _fade_weights(self, frames: int):
        """Equal-power weights for this block, or ``None`` when not fading."""

        if self._previous is None or self._fade_frames <= 0:
            return None
        position = self._fade_position
        remaining = max(0, self._fade_frames - position)
        span = min(frames, remaining)
        outgoing = np.zeros(frames, dtype=np.float64)
        incoming = np.ones(frames, dtype=np.float64)
        if span > 0:
            progress = (np.arange(span, dtype=np.float64) + position) / float(self._fade_frames)
            # The same equal-power curve :func:`equal_power_crossfade` defines,
            # evaluated at arbitrary progress rather than over a whole fade, so
            # a fade that spans several blocks continues where it left off.
            angle = progress * (np.pi / 2.0)
            outgoing[:span] = np.cos(angle)
            incoming[:span] = np.sin(angle)
        return outgoing, incoming

    def _process_plain(self, samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """One forward transform, two or four multiplies, two or four inverses."""

        keep = max(0, self.taps - 1)
        combined = np.concatenate((self._history, samples))
        length = int(2 ** np.ceil(np.log2(combined.size + self.taps - 1)))
        # The one transform this whole class exists to share.
        spectrum = np.fft.rfft(combined, length)

        offset = self._history.size
        current = np.fft.irfft(spectrum * self._current.spectra(length), length, axis=1)
        output = current[:, offset : offset + samples.size]

        weights = self._fade_weights(samples.size)
        if weights is not None:
            outgoing_weight, incoming_weight = weights
            previous = np.fft.irfft(spectrum * self._previous.spectra(length), length, axis=1)
            output = (
                previous[:, offset : offset + samples.size] * outgoing_weight
                + output * incoming_weight
            )

        self._history = combined[-keep:] if keep else np.zeros(0, dtype=np.float64)
        return output

    def _ensure_delay_line(self, partition: int, partitions: int) -> None:
        """Build or rebuild the shared frequency-domain input delay line."""

        shape = (partitions, partition + 1)
        if (
            self._delay_line is not None
            and self._delay_partition == partition
            and self._delay_line.shape == shape
        ):
            return
        length = 2 * partition
        needed = partitions * partition
        tail = self._history
        if tail.size < needed:
            tail = np.concatenate((np.zeros(needed - tail.size), tail))
        else:
            tail = tail[tail.size - needed :]
        self._previous_block = tail[-partition:].copy()
        self._delay_line = np.zeros(shape, dtype=np.complex128)
        # ``process`` rolls before writing entry zero, so entry j here is what
        # entry j + 1 must be next call. The last entry is the one the roll
        # discards, so it stays zero.
        for index in range(partitions - 1):
            stop = tail.size - index * partition
            self._delay_line[index] = np.fft.rfft(tail[stop - length : stop], length)
        self._delay_partition = partition

    def _process_partitioned(self, samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """Uniform-partitioned overlap-save, sharing one input delay line."""

        partition = samples.size
        spectra = self._current.partitioned_spectra(partition)
        partitions = int(spectra.shape[1])
        previous_spectra = (
            self._previous.partitioned_spectra(partition) if self._previous is not None else None
        )
        if previous_spectra is not None:
            partitions = max(partitions, int(previous_spectra.shape[1]))
        self._ensure_delay_line(partition, partitions)

        length = 2 * partition
        segment = np.concatenate((self._previous_block, samples))
        self._previous_block = samples.copy()
        self._delay_line = np.roll(self._delay_line, 1, axis=0)
        # Again, one forward transform for both ears and both filters.
        self._delay_line[0] = np.fft.rfft(segment, length)

        def convolve(filter_spectra: NDArray[np.complex128]) -> NDArray[np.float64]:
            count = int(filter_spectra.shape[1])
            accumulated = np.einsum(
                "pk,cpk->ck", self._delay_line[:count], filter_spectra, optimize=False
            )
            return np.fft.irfft(accumulated, length, axis=1)[:, partition:]

        output = convolve(spectra)
        weights = self._fade_weights(samples.size)
        if weights is not None and previous_spectra is not None:
            outgoing_weight, incoming_weight = weights
            output = convolve(previous_spectra) * outgoing_weight + output * incoming_weight

        keep = max(0, self.taps - 1)
        combined = np.concatenate((self._history, samples))
        self._history = combined[-keep:] if keep else np.zeros(0, dtype=np.float64)
        return output

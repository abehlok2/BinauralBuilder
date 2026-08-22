"""Block convolution for time-varying filters.

Overlap-save, because a moving source changes filters while audio is flowing
and overlap-save keeps exactly the state that makes a block boundary
invisible: the tail of the *input*, not a partially finished output.

Switching filters is the interesting part. Replacing a filter between blocks
steps the output discontinuously - the first sample of the new filter's
response has no history behind it - which is audible as a click on every
direction change. :class:`CrossfadingConvolver` therefore keeps **two**
convolvers, feeds the same input to both, and fades between their outputs over
a few milliseconds. Both histories keep running throughout, so neither output
has a discontinuity of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .envelopes import equal_power_crossfade, milliseconds_to_frames

__all__ = [
    "DEFAULT_FILTER_CROSSFADE_MS",
    "MAX_FILTER_CROSSFADE_MS",
    "MIN_FILTER_CROSSFADE_MS",
    "MAX_PARTITIONS",
    "MIN_PARTITION_SAMPLES",
    "PARTITION_THRESHOLD_RATIO",
    "CrossfadingConvolver",
    "OverlapSaveConvolver",
    "PartitionedEngine",
]

#: The specification's range for a filter-switch crossfade.
MIN_FILTER_CROSSFADE_MS = 5.0
MAX_FILTER_CROSSFADE_MS = 20.0
DEFAULT_FILTER_CROSSFADE_MS = 12.0

#: Partition the filter once it is longer than this many blocks. Measured, not
#: guessed: at 512-sample blocks the partitioned path is 43% *slower* at 128 and
#: 256 taps, breaks even at 512, and is 42%, 67% and 81% faster at 1024, 2048
#: and 4096. The crossover sits where the filter first exceeds one block, which
#: is where the single transform has to grow past twice the block length.
PARTITION_THRESHOLD_RATIO = 1.0

#: Blocks below this are not worth partitioning; see :meth:`_wants_partitioning`.
MIN_PARTITION_SAMPLES = 64

#: An upper bound on partitions, so a very long filter against a small block
#: does not turn into a delay line of hundreds of tiny transforms.
MAX_PARTITIONS = 64


class PartitionedEngine:
    """Uniform-partitioned overlap-save, for filters longer than a block.

    The plain path transforms ``block + taps`` samples every block, so a long
    filter forces a long transform on audio that is mostly zero padding. Cutting
    the filter into block-sized partitions keeps every transform at twice the
    block length: the input is transformed once, held in a frequency-domain
    delay line, and each partition is a complex multiply-accumulate against a
    spectrum computed when the filter was installed.

    The result is the same convolution to floating-point round-off, not an
    approximation - the partitions reassemble exactly the sum the single long
    transform computes.
    """

    def __init__(self, filter_taps: NDArray[np.float64], partition: int) -> None:
        self.partition = int(partition)
        self.length = 2 * self.partition
        self._install(filter_taps)
        self._delay_line = np.zeros(
            (self.partitions, self.length // 2 + 1), dtype=np.complex128
        )
        self._previous = np.zeros(self.partition, dtype=np.float64)

    def _install(self, filter_taps: NDArray[np.float64]) -> None:
        taps = np.asarray(filter_taps, dtype=np.float64).reshape(-1)
        self.partitions = max(1, int(np.ceil(taps.size / self.partition)))
        padded = np.zeros(self.partitions * self.partition, dtype=np.float64)
        padded[: taps.size] = taps
        self._spectra = np.fft.rfft(
            padded.reshape(self.partitions, self.partition), self.length, axis=1
        )

    def set_filter(self, filter_taps: NDArray[np.float64]) -> bool:
        """Install a filter. False when the partition count changed."""

        before = self.partitions
        self._install(filter_taps)
        if self.partitions == before:
            return True
        # A different partition count needs a different delay line; the caller
        # rebuilds this engine rather than carrying a mismatched history.
        return False

    def prime(self, history: NDArray[np.float64]) -> None:
        """Fill the delay line from raw input history, newest sample last.

        This is what lets the engine be entered mid-stream: the delay line is
        derived from the same input the plain path keeps, so switching between
        the two is seamless rather than a reset.
        """

        needed = self.history_needed
        tail = np.asarray(history, dtype=np.float64).reshape(-1)
        if tail.size < needed:
            tail = np.concatenate((np.zeros(needed - tail.size), tail))
        else:
            tail = tail[tail.size - needed :]

        self._previous = tail[-self.partition :].copy()
        self._delay_line[:] = 0.0
        # `process` rolls the line before writing entry zero, so entry j here is
        # what entry j + 1 must be during the next call: the segment spanning
        # the two blocks that end j + 1 blocks back. The last entry is the one
        # the roll discards, so it is left at zero.
        for index in range(self.partitions - 1):
            stop = tail.size - index * self.partition
            segment = tail[stop - self.length : stop]
            self._delay_line[index] = np.fft.rfft(segment, self.length)

    @property
    def history_needed(self) -> int:
        """Raw input samples :meth:`prime` needs to fill the delay line."""

        return self.partitions * self.partition

    def process(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convolve one partition-sized block."""

        segment = np.concatenate((self._previous, block))
        self._previous = block.copy()
        self._delay_line = np.roll(self._delay_line, 1, axis=0)
        self._delay_line[0] = np.fft.rfft(segment, self.length)
        accumulated = np.einsum(
            "pk,pk->k", self._delay_line, self._spectra, optimize=False
        )
        return np.fft.irfft(accumulated, self.length)[self.partition :]


@dataclass
class OverlapSaveConvolver:
    """Convolve a stream with one filter, block by block.

    The filter may be replaced, but doing so mid-stream steps the output;
    :class:`CrossfadingConvolver` is what makes a change inaudible.
    """

    filter_taps: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0]))
    _history: NDArray[np.float64] = field(default=None, repr=False)
    #: The filter's spectrum at the transform length last used. The filter is
    #: constant between :meth:`set_filter` calls while the input is not, so
    #: transforming it again every block was a third of all the transform work
    #: this class did. Install filters through :meth:`set_filter`, which clears
    #: this; assigning ``filter_taps`` directly would leave it stale.
    _spectrum: NDArray[np.complex128] | None = field(default=None, repr=False)
    _spectrum_length: int = field(default=0, repr=False)
    #: The partitioned engine, built on the first block long enough to want one.
    _engine: "PartitionedEngine | None" = field(default=None, repr=False)
    _last_block_size: int = field(default=0, repr=False)
    #: Set when a block bypassed the engine, so its delay line no longer
    #: matches the stream and has to be rebuilt from history before reuse.
    _engine_stale: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.set_filter(self.filter_taps, reset_history=True)

    @property
    def taps(self) -> int:
        return int(self.filter_taps.size)

    @property
    def latency_samples(self) -> int:
        """Overlap-save adds no latency of its own beyond the filter's."""

        return 0

    def set_filter(self, filter_taps: NDArray[np.floating], *, reset_history: bool = False) -> None:
        """Install a filter, keeping the input history unless asked otherwise."""

        taps = np.asarray(filter_taps, dtype=np.float64).reshape(-1)
        if taps.size == 0:
            taps = np.array([0.0])
        self.filter_taps = taps
        self._spectrum = None
        self._spectrum_length = 0
        if self._engine is not None and not self._engine.set_filter(taps):
            # A different partition count needs a differently shaped delay
            # line; rebuild it from the input history rather than carry a
            # mismatched one, which keeps the output continuous either way.
            self._engine = None
            self._engine_stale = False
        required = max(0, taps.size - 1)
        if reset_history or self._history is None:
            self._history = np.zeros(required, dtype=np.float64)
        elif self._history.size < required:
            self._history = np.concatenate((np.zeros(required - self._history.size), self._history))
        elif self._history.size > required:
            self._history = self._history[self._history.size - required :] if required else np.zeros(0)

    def reset(self) -> None:
        self._engine = None
        self._engine_stale = False
        self._last_block_size = 0
        self._history = np.zeros(max(0, self.taps - 1), dtype=np.float64)

    @property
    def history(self) -> NDArray[np.float64]:
        return self._history

    def process(self, block: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convolve one block, carrying the input tail into the next call."""

        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return samples.astype(np.float64)

        partitioned = self._partitioned_output(samples)
        if partitioned is not None:
            return partitioned

        combined = np.concatenate((self._history, samples))
        length = int(2 ** np.ceil(np.log2(combined.size + self.taps - 1)))
        if self._spectrum is None or self._spectrum_length != length:
            self._spectrum = np.fft.rfft(self.filter_taps, length)
            self._spectrum_length = length
        convolved = np.fft.irfft(np.fft.rfft(combined, length) * self._spectrum, length)

        output = convolved[self._history.size : self._history.size + samples.size]
        keep = max(0, self.taps - 1)
        self._history = combined[-keep:] if keep else np.zeros(0)
        return output

    def _primed_engine_history(self, needed: int) -> NDArray[np.float64]:
        """The plain path's history, left-padded to what priming wants.

        Padding rather than keeping more input is not an approximation. The
        oldest partition's filter coefficients past ``taps`` are the zeros that
        padded the filter out to whole partitions, so input older than
        ``taps - 1`` is multiplied by zero either way. Priming from exactly the
        history the plain path keeps is what makes the two paths agree.
        """

        history = self._history
        if history.size >= needed:
            return history[history.size - needed :]
        return np.concatenate((np.zeros(needed - history.size), history))

    def _wants_partitioning(self, block_size: int) -> bool:
        """Only where partitioning measured faster than the single transform.

        That is when the filter is longer than a block. The floor on the block
        size keeps the pathological end out: a tiny block would cut the filter
        into hundreds of partitions of a few samples each, which is all
        bookkeeping and no transform.
        """

        if block_size < MIN_PARTITION_SAMPLES:
            return False
        if self.taps <= PARTITION_THRESHOLD_RATIO * block_size:
            return False
        return self.taps <= MAX_PARTITIONS * block_size

    def _partitioned_output(self, samples: NDArray[np.float64]):
        """Run the partitioned path, or return None to use the plain one.

        The engine is built lazily because the block size is not known until a
        block arrives, and it is bypassed for any block of a different size -
        the final, short block of a render, typically - so the choice never
        costs correctness.
        """

        block_size = int(samples.size)
        previous_size, self._last_block_size = self._last_block_size, block_size

        engine_handles_it = (
            self._wants_partitioning(block_size)
            and self._engine is not None
            and self._engine.partition == block_size
        )
        if not engine_handles_it and self._engine is not None:
            # Any block the engine does not process leaves its delay line a
            # block out of step with the stream, whatever the reason - a short
            # final block, a filter that shrank below the threshold, a
            # different block size. Re-prime before using it again.
            self._engine_stale = True
        if not self._wants_partitioning(block_size):
            return None
        if self._engine is not None and self._engine.partition != block_size:
            return None
        if self._engine is None:
            if previous_size != block_size:
                # Wait for a second block of the same size before committing to
                # a partition size, so a one-off short block never builds an
                # engine the rest of the stream would have to throw away.
                return None
            engine = PartitionedEngine(self.filter_taps, block_size)
            engine.prime(self._primed_engine_history(engine.history_needed))
            self._engine = engine
        elif self._engine_stale:
            self._engine.prime(self._primed_engine_history(self._engine.history_needed))
            self._engine_stale = False

        output = self._engine.process(samples)
        # The plain path's history must stay current too: it is what the next
        # differently sized block, and any re-priming, will work from.
        keep = max(0, self.taps - 1)
        if keep:
            self._history = np.concatenate((self._history, samples))[-keep:]
        return output


@dataclass
class CrossfadingConvolver:
    """Two convolvers and an equal-power fade between them.

    A filter change starts a fade from the outgoing convolver to the incoming
    one. Both keep processing the same input for the whole fade, so each output
    is continuous and only their *mixture* changes.

    A change arriving mid-fade is queued rather than applied: restarting a fade
    from a partially mixed signal is what produces the artefact the fade exists
    to avoid. The newest request wins when the current fade finishes.
    """

    crossfade_ms: float = DEFAULT_FILTER_CROSSFADE_MS
    sample_rate_hz: float = 44_100.0
    _active: OverlapSaveConvolver = field(default_factory=OverlapSaveConvolver, repr=False)
    _outgoing: OverlapSaveConvolver = field(default_factory=OverlapSaveConvolver, repr=False)
    _fade_position: int = field(default=0, repr=False)
    _fade_length: int = field(default=0, repr=False)
    _pending: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def fade_frames(self) -> int:
        """Length of a filter-switch fade, clamped to the documented range."""

        milliseconds = float(
            np.clip(self.crossfade_ms, MIN_FILTER_CROSSFADE_MS, MAX_FILTER_CROSSFADE_MS)
        )
        return max(1, milliseconds_to_frames(milliseconds, self.sample_rate_hz))

    @property
    def is_fading(self) -> bool:
        return self._fade_position < self._fade_length

    @property
    def filter_taps(self) -> NDArray[np.float64]:
        return self._active.filter_taps

    def reset(self, filter_taps: NDArray[np.floating] | None = None) -> None:
        """Clear both convolvers. Only at a genuine transport reset."""

        taps = np.array([1.0]) if filter_taps is None else np.asarray(filter_taps, dtype=np.float64)
        self._active.set_filter(taps, reset_history=True)
        self._outgoing.set_filter(taps, reset_history=True)
        self._fade_position = 0
        self._fade_length = 0
        self._pending = None

    def set_filter(self, filter_taps: NDArray[np.floating]) -> bool:
        """Request a filter change. Returns True when a fade started now."""

        taps = np.asarray(filter_taps, dtype=np.float64).reshape(-1)
        if taps.shape == self._active.filter_taps.shape and np.array_equal(
            taps, self._active.filter_taps
        ):
            return False
        if self.is_fading:
            self._pending = taps
            return False

        # The outgoing convolver takes over the current filter *and* the
        # current history, so its output continues without a step.
        self._outgoing.filter_taps = self._active.filter_taps
        self._outgoing._history = np.array(self._active.history, copy=True)
        self._active.set_filter(taps)
        self._fade_length = self.fade_frames
        self._fade_position = 0
        return True

    def process(self, block: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convolve one block, mixing across a fade when one is in progress."""

        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return samples.astype(np.float64)

        incoming = self._active.process(samples)
        if not self.is_fading:
            # Keep the outgoing line's history current so a future fade starts
            # from a continuous signal rather than from silence.
            self._outgoing.process(samples)
            return incoming

        outgoing = self._outgoing.process(samples)
        fade_out, fade_in = equal_power_crossfade(self._fade_length)
        start = self._fade_position
        stop = min(self._fade_length, start + samples.size)
        span = stop - start

        weights_out = np.ones(samples.size, dtype=np.float64)
        weights_in = np.zeros(samples.size, dtype=np.float64)
        weights_out[:span] = fade_out[start:stop]
        weights_in[:span] = fade_in[start:stop]
        weights_out[span:] = 0.0
        weights_in[span:] = 1.0

        self._fade_position = stop
        mixed = outgoing * weights_out + incoming * weights_in

        if not self.is_fading and self._pending is not None:
            queued, self._pending = self._pending, None
            self.set_filter(queued)
        return mixed

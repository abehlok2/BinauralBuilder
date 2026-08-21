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
    "CrossfadingConvolver",
    "OverlapSaveConvolver",
]

#: The specification's range for a filter-switch crossfade.
MIN_FILTER_CROSSFADE_MS = 5.0
MAX_FILTER_CROSSFADE_MS = 20.0
DEFAULT_FILTER_CROSSFADE_MS = 12.0


@dataclass
class OverlapSaveConvolver:
    """Convolve a stream with one filter, block by block.

    The filter may be replaced, but doing so mid-stream steps the output;
    :class:`CrossfadingConvolver` is what makes a change inaudible.
    """

    filter_taps: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0]))
    _history: NDArray[np.float64] = field(default=None, repr=False)

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
        required = max(0, taps.size - 1)
        if reset_history or self._history is None:
            self._history = np.zeros(required, dtype=np.float64)
        elif self._history.size < required:
            self._history = np.concatenate((np.zeros(required - self._history.size), self._history))
        elif self._history.size > required:
            self._history = self._history[self._history.size - required :] if required else np.zeros(0)

    def reset(self) -> None:
        self._history = np.zeros(max(0, self.taps - 1), dtype=np.float64)

    @property
    def history(self) -> NDArray[np.float64]:
        return self._history

    def process(self, block: NDArray[np.floating]) -> NDArray[np.float64]:
        """Convolve one block, carrying the input tail into the next call."""

        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return samples.astype(np.float64)

        combined = np.concatenate((self._history, samples))
        length = int(2 ** np.ceil(np.log2(combined.size + self.taps - 1)))
        spectrum = np.fft.rfft(combined, length) * np.fft.rfft(self.filter_taps, length)
        convolved = np.fft.irfft(spectrum, length)

        output = convolved[self._history.size : self._history.size + samples.size]
        keep = max(0, self.taps - 1)
        self._history = combined[-keep:] if keep else np.zeros(0)
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

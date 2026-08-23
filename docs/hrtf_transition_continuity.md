# Why a fast path buzzed

A source on a fast path — the reported case was a torus with `major_turns = 21`,
so twenty-one complete orbits of the listener — rendered with a periodic buzz,
a harmonic ladder above it, sidebands around the carrier, and broadband spill.

## The cause

It was a filter-transition state bug, and the path's angular speed only made it
audible.

The renderer's defaults are a 128-sample minimum control interval, a 1° angular
error bound, and a 10 ms crossfade at 44.1 kHz. So:

| quantity | value |
| --- | --- |
| control interval | 128 samples = 2.90 ms |
| crossfade length | 441 samples = 10 ms |
| filter request rate | 344.5 per second |

Twenty-one orbits spread over a 20-second preview still moves about 1.097° per
control interval, which is already past the 1° bound, so the renderer reselects
at the fastest rate the grid allows. That part is the adaptive logic working as
designed.

The failure was what happened next. A transition took 441 samples and the next
request arrived after 128, so **no transition could ever finish**. Each new
request replaced the running one:

```python
self._previous = self._current
self._current = pair
self._fade_frames = fade
self._fade_position = 0
```

At the end of a 128-sample interval a 441-sample equal-power transition has
reached `progress = 0.290`, so the audible sample is about

```
0.898 x old_filter_output + 0.440 x current_filter_output
```

and the next sample becomes `1.0 x current_filter_output`. The mixture vanishes
in one sample, 344.5 times a second. That is the buzz, and everything else in
the spectrum follows from it.

Measured on the reported torus, before and after:

| | before | after |
| --- | --- | --- |
| filter requests | 6891 | 6891 |
| arriving mid-transition | 6889 | **0** |
| transitions abandoned | 6889 | **0** |
| max sample-to-sample step | 5.07e-01 | **4.27e-02** |
| out-of-band energy | 1.27e+09 | **2.28e+05** |

The input's own largest step is 1.42e-02, so the render went from stepping 35x
harder than its source to 3x.

## The fix

**Planned adjacent interpolation.** A transition now spans exactly the gap to
the next point on the control grid, so it ends precisely where the next one
begins. Every interval finishes on the filter the following interval starts
from, which makes the filter trajectory continuous by construction rather than
by hoping the fade fits.

`crossfade_ms` still sets the length but is capped at the control interval. At
the default the cap binds and the morph spans the whole interval; a shorter
setting reaches the new filter sooner and holds it, which is equally
continuous. Leaving it uncapped is what made every transition unfinishable.

**A running transition is never abandoned.** A request arriving mid-transition
is queued rather than applied, and only the newest waiting request survives —
an older one is out of date before it was ever heard. The queued transition
begins at the exact sample the running one ended, not at the next block
boundary, so where a transition starts does not depend on how the caller cut
the stream. In the planned path above this never triggers; it is the safety net
for callers that do not select on a grid.

**A linear fade for continuous motion.** Equal power keeps the sum of squares
constant, which is right for uncorrelated signals. Two HRTF-filtered copies of
one carrier are strongly correlated, so their amplitudes add: at the midpoint
both weights are 0.707 and nearly identical filters sum to 1.414, about 3 dB of
gain. Once, that is inaudible; hundreds of times a second it is amplitude
modulation at the control rate. Equal power is kept for genuinely discontinuous
jumps, where the two signals may be decorrelated.

## Diagnostics

The renderer reports, alongside its existing counters:

```
filter_requests  filter_changes  mid_fade_requests
queued_filter_updates  dropped_filter_updates
fade_restarts  maximum_filter_age_samples
```

`fade_restarts` is the one to watch. It is zero by construction and the tests
assert it.

## Interpolation

`delay_magnitude` is the mode suited to morphing between adjacent directions:
it aligns the impulse responses, interpolates the onset delay separately from
the log magnitude, and restores the fractional delay afterwards. `nearest`
remains as a compatibility and fallback option, and its coverage warning
already says that it steps rather than interpolates.

## Path speed

This is a creative choice, not a defect, and the renderer now degrades
gracefully under it rather than generating discontinuities.

`major_turns` is the number of complete orbits of the listener, not a drawing
resolution: raising it multiplies angular speed for the same traversal
duration. If the intent is one horizontal orbit winding twenty-one times
vertically, the twenty-one belongs on `minor_turns`. If twenty-one orbits are
intended, lengthen the traversal in proportion — twenty-one turns over
twenty-one seconds is the same mean angular velocity as one turn per second.

The coverage report warns when a path turns more than 10° between filter
updates. That warning now uses the *effective* transition span rather than the
raw `crossfade_ms`, since the renderer caps it; reporting the uncapped figure
would describe smear the renderer no longer produces.

## Tests

`tests/sam_workbench/test_hrtf_transition_continuity.py` runs the reported
21-turn torus with a 200 Hz sinusoid, a 128-sample control interval and a 10 ms
crossfade, and asserts: the path really does request a filter every interval;
no transition is abandoned; every transition finishes inside its own interval;
the output has no step larger than 3x the source's own; energy stays near the
carrier; and all of it is independent of the caller's block size. It also pins
the convolver's contract directly — mid-fade requests wait, newest wins, a
queued transition starts where the previous ended, and the linear curve does
not bulge where equal power does.

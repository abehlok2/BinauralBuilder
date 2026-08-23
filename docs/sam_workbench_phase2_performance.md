# Phase 2: one HRTF engine, and what it cost before

## The measurement

`tools/benchmark_export.py` runs the real production export path —
`render_sam2_voice` through the compatibility adapter — so the numbers describe
what happens when someone renders a track, not what a hand-built harness does.

Audio-seconds per wall-clock second; above 1.0 is faster than real time.

| case | before | after | |
|---|---|---|---|
| abstract_pm | 108.99 | 104.61 | 1.0× |
| geometric | 14.02 | 16.13 | 1.2× |
| hrtf / nearest / 128 | 1.06 | 4.58 | 4.3× |
| hrtf / nearest / 512 | 1.36 | 15.43 | 11.3× |
| hrtf / delay_magnitude / 128 | 0.63 | 2.88 | 4.6× |
| hrtf / delay_magnitude / 512 | 0.86 | 10.40 | 12.0× |

HRTF rendering was **slower than real time**. A thirty-minute track with
delay-magnitude interpolation took about forty-eight minutes to export.

## What actually cost the time

Two things, and only one of them was the obvious one.

**Per-sample Python convolution.** The production engine convolved sample by
sample in a Python loop. Replacing it with block overlap-save was the expected
fix and gave about 2×.

**Rebuilding the arc-length table on every position query.** Profiling after
the convolution fix showed 74% of the remaining time in `arclength_table`,
which built a 2049-point lookup table *per query* — thousands of times per
second of audio. Constant-speed traversal was costing more than the convolution
it was feeding. Caching it by geometry value gave another 2×, and batching
position queries gave a further 1.4×.

The lesson is in the ordering: the slow part was not where it looked.

## The convolution engine

`dsp/binaural_convolution.py` transforms the input **once** per block and
multiplies that one spectrum by every filter it needs — left, right, and both
again during a crossfade.

Overlap-save is what makes this possible: its state is the tail of the *input*,
not a partially finished output, so filters sharing one input history are
automatically aligned. That also removes the dormant outgoing convolver
entirely — an outgoing filter is just another spectrum against history the
engine already holds, so a transition costs two extra inverse transforms while
it lasts and nothing when it does not.

Both the plain and partitioned paths match a direct convolution to 1e-11. They
are exact, not approximate.

## Adaptive spatial updates

Direction is reselected on a schedule bounded by **angular error** rather than
elapsed samples. What a listener can hear is how far the source has turned since
the filter was chosen, not how much time has passed.

Over a rotating path, loosening the bound from 1° to 60° takes selections from
345 to 23. That is most of the difference between the 128 and 512 rows above.

Two subtleties the tests caught:

- Selecting once per block made the update rate a function of the caller's
  block size. Selection now happens on a fixed grid anchored to sample zero,
  and a block is cut only where a change actually falls.
- Choosing every filter for a block *before* convolving any of it left the last
  filter installed for the block's opening audio. Selection and processing are
  interleaved.

## Determinism

| case | guarantee |
|---|---|
| Static filter, any block pattern | 1e-15 |
| Moving source, any block size | < 2e-6 |
| Streamed vs in-memory export | **byte-identical** |

## Memory

Exact normalization needs the peak, and the peak is not known until the last
sample exists. The export path answered that with three full-length arrays alive
at once — the track, a scaled float copy, an int16 copy — about four gigabytes
for an hour of stereo.

`streaming.py` spools float32 to temporary storage while measuring the peak
exactly, then reads it back in bounded blocks. On ten minutes of audio, extra
peak allocation falls from **635 MB to 1 MB**, and four times the audio costs no
more memory than one.

This bounds the *export*. The renderer still materializes the track before
handing it over; `iter_track_blocks` bridges from an array that already exists,
and the memory tests measure the encoder against a generator precisely so that
distinction is not blurred.

### `ENABLE_SEQUENTIAL_CHUNKING`, and a wrong reason corrected

This flag is now **on**, and the reason recorded for it previously being off was
wrong. That entry said chunked generation diverged from whole generation because
"the synth functions do not carry their oscillator phase across a boundary —
`binaural_beat`, for one, has no state to return at all". `binaural_beat`
returns `oscillator_phases`, `startPhaseL` and `startPhaseR`, and when those are
threaded it reproduces a whole render with zero differing samples. So does
`isochronic_tone`, and so do the SAM implementations.

The actual cause was in the harness, not the synths. A 10 ms declick fade was
applied to **every generated buffer**. That is right at the edges of a voice,
where steps are concatenated without a crossfade — and wrong at every internal
chunk boundary, because a step rendered in chunks calls the generator once per
chunk. The audio either side of each boundary was correct and bit-identical; only
the 440-sample seam between them was not. The fade now applies only when the
buffer really is at the voice's start or end.

Which functions are safe is decided at runtime rather than by a list. After the
first chunk, `_states_carry_continuity` asks the states that came back whether
they contain anything to resume from; a step whose voices report nothing is
rendered whole instead. An allowlist would quietly rot as functions are added,
and the failure mode of a stale one is audio that is silently wrong. Dormant
functions such as `qam_beat` fall back automatically under this gate.

Measured peak allocation for a single-step `binaural_beat` render, 30 s chunks:

| Render length | Whole | Chunked | Saving |
| --- | --- | --- | --- |
| 2 min | 437 MB | 182 MB | 2.40× |
| 10 min | 2019 MB | 505 MB | 4.00× |

The step buffer itself is still allocated once — `assemble_track_from_data`
returns an array, so its caller holds the whole render either way. What chunking
bounds is the transient on top of it: the synth's working arrays are one chunk
wide rather than one step wide, which is where the multi-GB peaks came from.

`test_long_render_memory.py` pins all of this: that the flag is on, that chunked
and whole renders are equal sample-for-sample for each live synth, that the fade
survives at the voice edges and not at the seams, and that the gate falls back
for a synth reporting no state.

## Scene mixing

The pieces for real bus mixing existed and were tested, and nothing used them.
Production folded a source's bus gain and its mute/solo state into a single
scalar on that source's own audio — right level, no mixer: nothing to meter,
nothing to process, nothing a band setting could act on.

`render/scene_mix.py` sums sources into buses, band-processes and gains each
bus, and sums buses to master, keeping every stem so it can be metered. It is
stateful because band splitting is: restarting the crossover filters each block
steps every band's output by roughly a quarter of the signal's peak — a click,
not a change of tone. Blocked and whole renders are bit-identical at every block
size tested.

## The filter cache, and a bug it found

Caching interpolated filters per interpolator bought nothing — 1.00 to 1.02×,
measured. The adaptive control interval already declines to reselect until the
direction has moved past its tolerance, so within one render consecutive lookups
are genuinely different directions.

What repeats is *renders*: preview then export, the two halves of an A/B,
several sources on one trajectory. Moved to module scope, a second render of the
same voice is **1.25–1.59×** faster and byte-identical.

Writing the key-correctness tests found a real bug in the first version. The key
used the dataset's SOFA content hash — and a cue-modified dataset is a *view* of
the file it came from and forwards that hash, so a hybrid render with cue
modification would have been served the unmodified filters. Silently, and only
in hybrid mode. The fingerprint now hashes the impulse responses themselves.

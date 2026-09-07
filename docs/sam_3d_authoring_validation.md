# SAM 3D authoring validation

Base: `54f313338d3924d0c5724e3bb2fd1d9fdf00b5fb`.

The implementation fixes the vertical projection, preserves editor metadata and
listener transforms, adds whole-path constraints and spherical interpolation,
and aligns previews and coverage checks with bound trajectory evaluation. The
editor now supports transactional motion edits, undo/redo, elapsed-time playback,
and background HRTF coverage analysis. The user guide documents the opt-in
trajectory schema 3 additions and the vertical audition workflow.

## Checks

- 32 new regression tests passed, including production geometric/HRTF block
  equivalence, coordinate signs, constrained intermediate samples, authored
  timing, metadata preservation, undo/Cancel, selected-asset context, modulated
  coverage, and explicit audition material.
- The focused path/GUI run passed 108 tests before the final legacy-promotion
  regression was added; the complete new regression file was then rerun.
- Wider workbench validation completed 1,677 passing tests across two runs:
  1,585 completed before interrupting a slow memory benchmark, followed by the
  remaining 92 tests. The three memory benchmarks below were excluded, along
  with `test_long_render_memory.py`.
- The long-render file was checked separately. Its isochronic-tone exact chunk
  comparison failed with maximum difference `1.137e-13`. The same test failed
  on an unchanged checkout of the base commit with the same difference.
- Offscreen GUI rendering was inspected for projection and clipped controls.
  `git diff --check` passed. An optional SOFA dependency emitted a NumPy binary
  compatibility warning; the SOFA functional checks passed after installing
  `sofar`.

Equivalent wider-suite selection:

```sh
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python -m pytest -q \
  tests/sam_workbench \
  --ignore=tests/sam_workbench/test_long_render_memory.py \
  -k 'not test_chunking_bounds_what_a_long_render_costs_on_top_of_its_output and not test_peak_memory_does_not_follow_the_track_length and not test_the_whole_track_path_still_grows_with_length'
```

## Limits

These results do not establish a fully passing unfiltered suite or new memory
and throughput guarantees. No hardware listening study was performed. Constant
angular speed is limited to static paths; animated paths report their parameter
speed fallback. Long coverage intervals may be downsampled with an explicit
warning, and coverage sampling does not guarantee perceptual localization.

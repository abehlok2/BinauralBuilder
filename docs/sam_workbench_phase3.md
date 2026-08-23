# Phase 3: the workbench around the finished engine

Phases 1 and 2 built the canonical scene plan, the renderer registry, one HRTF
engine and bounded-memory rendering. Phase 3 is about what a user can reach:
whether every renderer is offered, whether every control that is offered does
something, whether the interface tells the truth about what will happen, and
whether a finished export can say what produced it.

## The baseline failures, verified rather than classified

Four tests were failing at the start of this phase and had previously been
reported — by me — as pre-existing and unrelated. That was too quick. Each had
a different cause and three were fixable:

| test | cause |
| --- | --- |
| `test_hrtf_modify_gui` | `sofar` is declared in `pyproject.toml` and was not installed. The panel worked and said so. |
| `test_colored_noise_dialog` | `PyQt5.QtMultimedia` could not import without `libpulse-mainloop-glib0`, so the module never defined the names the test patches; and the test drove a `duration_spin` that never existed in that file's history. |
| `test_session_builder_window` | `DummyStream` implemented `start`/`stop` while the window also calls `set_volume`, `update_track`, `pause`, `resume`, `seek`. |
| `test_scene_and_experiment_gui` | Genuinely superseded: `collect_params` strips the legacy scene keys deliberately and `scene_data` owns them. |

## Lists that could drift, and no longer can

The renderer combo already came from the registry. The HRTF Lab and the routing
panel each repeated the engine's interpolation modes, and the Lab repeated the
delay policies. They agreed with the engine and nothing held them there. Both
now derive from `INTERPOLATION_MODES` and `DelayPolicy`; the GUI keeps only the
human labels, and a mode without a label still appears under a derived one
rather than disappearing.

The validator the dialog displayed checked the registry for the renderer and
its assets but never ran the renderer's own `validate()` — the one the compiled
plan runs before rendering. A renderer-specific misconfiguration was therefore
enforced at render time and never shown. It runs both now.

Tab relevance comes from the registry too: a renderer declaring a SOFA asset
needs the HRTF Lab, one consuming a trajectory needs the path editor.
Irrelevant tabs are disabled with a tooltip rather than removed, because
removing them renumbers the rest and makes the Lab unreachable as a tool for
auditioning a dataset. What matters for correctness is that nothing *enabled*
is ignored, and disabling satisfies that.

## Opening in Basic, and the defect that exposed

The workbench opened in Expert every time. Expert is the whole parameter space
at once — where to return to, not where to arrive. It now opens in Basic on
first use and in the last-chosen mode afterwards.

Making that change exposed a latent defect. `SamBasicPanel` built its parameter
groups from the mode it was *constructed* in, while `set_mode` only toggled the
visibility of groups that already existed. A panel opened in Basic had no
Expert controls at all — not hidden, absent — so switching to Expert could not
reveal them and `params()` would not return them, quietly dropping those keys
from saved voices. It was invisible only because the dialog always constructed
in Expert. Every group is now built and `set_mode` decides what is shown.

The test suite was also reading and writing the developer's real `QSettings`
file. Beyond rewriting somebody's preferences, it meant the old
default-to-Expert test kept passing after the default changed, because an
earlier test had stored `expert`. Settings are now isolated per test.

## The signal-flow summary

`flow.py` answers "what happens when I press render?" from the same parameters
production reads, and the dialog keeps that answer visible whichever tab is
open:

```
Source → Path → Renderer → (Cue transform) → (Headphone correction) → Output
```

Inactive stages are shown in parentheses rather than omitted, so each stage
stays in the same place as options are toggled. Alongside it: renderer, asset,
interpolation, path status, scene roster, a cost estimate, and the warnings the
dialog is already showing rather than a second opinion.

It also **names** any control holding a value the selected renderer will never
read — a SOFA asset under abstract phase modulation, a cue transform under
plain HRTF. From the value alone a user cannot tell a setting that is off from
one that is ignored.

## Real sources, and modulators with a shape

The routing panel started with a `source.1` that looked exactly like a real
route and referred to nothing, and the dialog invented the same identifier when
a voice arrived without one. A project could be routed, muted and soloed
against a source the track does not have, and nothing said so.

The roster now comes from the track, with `assign_source_ids` running against
the real steps rather than a copy — an identifier assigned to a throwaway is
thrown away with it. Rows show the voice's name and carry the identifier as
data. Adding a route picks from the roster instead of accepting free text.
Refreshing keeps existing routing; a route whose source is gone is kept and
marked rather than dropped, because dropping it discards a mute that an undo
would restore.

Modulators were names and nothing else, so every row compiled to the documented
1 Hz sine fallback: `random.walk` modulated identically to `lfo.slow`. Rows now
carry waveform, rate, phase and seed. The engine gained triangle and random
waveforms to make that true. The random one is a seeded function of absolute
time, hashed per step index rather than drawn as the render advances, so it is
independent of how the timeline was cut into blocks — which is what makes an
export match the preview it was approved from.

## Export as a background job

Final export ran on the GUI thread and kept the window alive by calling
`QApplication.processEvents()` from inside the render. That is not a background
job: it read the live project while the user could still edit it, it could not
be cancelled because nothing was watching, and a failure left whatever had been
written.

A render now starts from an immutable `RenderSnapshot`, deep-copied at the
button press, and runs on its own thread. The user may keep editing; the file
that appears is the project as it was when they asked for it.

Cancellation needed a route through the engine. Every progress callback site in
`sound_creator` swallows exceptions deliberately, so a buggy callback does not
lose a render that is otherwise fine. Cancellation wants the opposite, so it
has its own type, `RenderCancelled`, re-raised where the others are still
swallowed. It is checked where progress is reported, so a cancel lands within a
chunk of a long step rather than instantly.

Neither a cancelled nor a failed render leaves a file behind — a half-written
export plays, sounds like the track, and stops early — but a file that already
existed is left alone, since the previous export is not this render's to delete.

The window reports audio-seconds per wall-clock second, peak memory and HRTF
cache hits, and estimates the next render from the last one's measured
throughput. Closing with a render running asks first, then cancels and waits.

## The manifest

The typed-project exporter has written manifests since Phase 1. Normal track
export wrote a WAV and nothing else, so a finished file carried no record of
which dataset, which interpolation, which path or which seeds produced it.

Track export now writes one beside the audio. "Sufficient to reconstruct the
acoustic condition" is not provable by listing fields, so `reconstruct_track`
sits beside `build_track_manifest` and rebuilds a renderable track from the
manifest alone; the tests render both and compare. A field that mattered and
was not recorded would show up as audio that differs.

Parameters are stored whole rather than filtered to a known list, because a
filtered copy drops exactly the keys a newer build added. A renderer this build
does not know is recorded as unknown rather than reinterpreted. A dataset that
cannot be read is recorded as unreadable rather than omitted.

One subtlety the acceptance tests caught: reconstruction attached an empty
normalized scene to every track, and a SAM2 voice *with* a scene renders
through the scene path. A project that never used scene features would have
come back sounding slightly different from the export its manifest described.
A scene is now attached only when it says something.

## Readiness

Validation answers "is this legal". A project can be legal and still not
produce what its author expects. Readiness reports the rest: a dataset whose
hash no longer matches what the project was authored against (an error — the
render would succeed and not be the one that was approved), a path that leaves
the measured region, a route naming a deleted source, a control this renderer
ignores, a peak that will clip, a configuration too expensive to preview.

A missing headphone profile is advice rather than an error: rendering without
one is a legitimate choice.

## The 3-D editor

The coverage shell was drawn at a radius averaged from the path's own points,
which says nothing about the dataset — a path can sit exactly on a guessed
shell and be nowhere near a measurement. It now takes its radius from the
median measurement distance of the selected dataset. Coverage warnings sit
beside the path and update with it, because the moment to learn that a dome
leaves the measured region is while it is being dragged.

`promote_profile_to_trajectory` writes down the relationship between the legacy
2-D creator and the 3-D designer. It converts through the evaluator both
already share, so a promoted path follows the curve the 2-D preview drew rather
than merely connecting its control points, and lands as metres in the canonical
listener frame on the plane at ear height — all a 2-D editor can honestly
supply. The result records where it came from.

## Measurements

Export throughput, current build:

| case | audio-s per wall-s |
| --- | --- |
| abstract_pm | 156.2 |
| geometric | 19.2 |
| hrtf / nearest / 128 | 6.5 |
| hrtf / nearest / 512 | 28.9 |
| hrtf / delay_magnitude / 128 | 4.2 |
| hrtf / delay_magnitude / 512 | 25.0 |

Peak memory for a single-step 300 s `binaural_beat` render: 350 MB chunked
against a 106 MB output buffer, and chunking saves more than one output buffer
against the unchunked path. Chunking does not stop peak memory growing with
length — `assemble_track_from_data` holds a step buffer and a track buffer, both
of which scale — it bounds the synth's working set on top of them, which is
where the multi-gigabyte peaks came from.

## Tests

1493 passing, 0 failing. New in this phase: `test_gui_registry_driven.py` (11),
`test_flow_summary.py` (13), `test_scene_real_sources.py` (18),
`test_render_job.py` (18), `test_export_manifest.py` (17),
`test_readiness.py` (19), `test_path3d_integration.py` (13),
`test_phase3_acceptance.py` (25).

This repository has no CI configured, so local runs are the only evidence.

## Remaining limitations

- The step tester still renders synchronously. It previews one step rather than
  exporting, so it is outside what the background-job work covers, but a long
  step still blocks the window while it loads.
- `main.py` retains one `QApplication.processEvents()` call, in that step
  tester, to repaint a label before loading.
- Renderers still reach production through `compat.render_sam2_voice` rather
  than through `CompiledScenePlan`. The plan is authoritative for structure and
  validation; the per-voice render path is not yet built on it. This is Phase 1
  item 22, carried forward.
- Parallelism, content-addressed caching across runs, and seek checkpoints
  (Phase 2 item 10) remain unimplemented. The render job runs one render at a
  time by design; the manager can hold several but the window refuses a second.

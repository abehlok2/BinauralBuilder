# SAM workbench phases 0–6 implementation audit

**Audit baseline:** branch `work` at `733e4b5` (2026-08-21)  
**Specification:** repository-root `AGENTS.md`, especially sections 5–17  
**Scope:** Python offline/rendering and PyQt5 SAM/HRTF work only. The Rust backend
was not inspected or tested.

## Executive finding

The repository contains a substantial and well-tested **DSP/core implementation**
of the phase 0–6 roadmap. Phases 0, 1, 4, and 5 are the closest to their roadmap
exit criteria. Phase 2 and phase 3 have useful editor integration. Phase 6 is a
strong collection of independently tested domain primitives, but is **not yet an
end-to-end application feature**: stages, routing, the modulation matrix, source
coupling, and the cost decision are not connected to the canonical `Project`, the
scene renderer/export path, or a PyQt workbench panel.

Accordingly, “implemented through phase 6” is accurate if it means that the
planned core building blocks exist. It is not accurate if it means that every
phase 0–6 task and exit criterion in `AGENTS.md` is complete. The most important
remaining work is integration rather than another DSP rewrite.

| Phase | Assessment | Confidence |
| --- | --- | --- |
| 0 — canonical package | Substantially complete | High |
| 1 — exact SAM/adapters | Substantially complete | High |
| 2 — existing GUI integration | Substantially complete | High |
| 3 — geometry/path migration | Mostly complete, with integration gaps | High |
| 4 — SOFA/HRTF rendering | Substantially complete | High |
| 5 — HRTF modification/hybrid | Substantially complete | High |
| 6 — timeline/multiband/multi-source | Core primitives complete; product integration incomplete | High |

## Method

This comparison used four forms of evidence:

1. mapped every task and exit criterion in roadmap phases 0–6 to production
   modules, adapters, UI modules, and tests;
2. searched production call sites to distinguish a reusable primitive from a
   feature actually connected to `Project`, preview, export, and the existing
   editor;
3. ran the phase 0/1 contracts and the complete `tests/sam_workbench` suite;
4. checked the repository for scope violations and for dependencies that the
   plan says must remain optional.

The passing test suite demonstrates the behavior it covers, but it does not by
itself satisfy an exit criterion whose product-level wiring or performance
measurement is absent.

## Phase-by-phase comparison

### Phase 0 — establish the canonical package

**Implemented**

- `src/audio/sam_workbench` exists as a Qt-free canonical namespace, with
  conventions, versioning, validation, migrations, persistence, CLI, render,
  trajectory, HRTF, and analysis subpackages.
- The project schema is versioned `1.0`; dataclass factories avoid shared mutable
  defaults; validation aggregates issues; project and manifest writes use sibling
  temporary files and replacement.
- The `new` and `validate` CLI commands exist, and the package has a module entry
  point.
- Legacy behavior and synthetic SOFA/impulse assets have committed fixtures and
  fixture-generation scripts.
- Clean-import tests guard against PyQt5, `slab`, Rust, UI, and legacy synthesis
  imports in the canonical core.
- The package README explicitly identifies the old public entry points and their
  delegation status.

**Differences or caveats**

- No third-party generic HRTF is bundled, which is consistent with the licensing
  rule. `HRTF_ASSETS.md` documents external acquisition while tests use
  purpose-built synthetic assets.
- Running the `pytest` console script in this environment did not put the
  repository root on `sys.path`; `python -m pytest` is the reliable invocation.
  This is a test-launch/environment issue, not a canonical-package import defect.

**Verdict:** the roadmap work and exit criteria are substantially met.

### Phase 1 — exact SAM core and BinauralBuilder adapters

**Implemented**

- The exact opposed-ear equation lives once in `dsp/source.py`; carrier and
  modulator phases are accumulated from absolute sample positions.
- Controls, oscillators, modulators, envelopes, source compilation, block
  rendering, mixing, limiting, WAV export, reconstruction manifests, and CLI
  rendering are present.
- The compatibility layer translates legacy camelCase dictionaries and performs
  the one channel-major/frame-major transpose.
- Static and transition SAM2 entry points in both Python synthesis trees delegate
  to the canonical implementation. The legacy polarity convention is explicit.
- Tests cover the mathematical reference, deterministic output, whole-versus-
  chunked rendering, transition continuity, manifests, adapter parity, and legacy
  fixtures.

**Differences or caveats**

- The Phase 1 control compiler now covers constant, ramp, keyframe, LFO, step
  sequence, deterministic random walk, restricted expression, captured external
  input, sum, product, map-range, and smoothing controls. Restricted expressions
  are interpreted from a validated arithmetic AST and never use Python `eval`.
- `SignalSpec` serializes the periodic carrier waveform and the canonical renderer
  honors it. Tagged harmonic-bank, noise, and file variants remain later source-
  signal inventory work rather than requirements of the exact Phase 1 SAM exit.
- The CLI/export surface is WAV-specific. FLAC and stem export called for by the
  broader plan are not exposed.

**Verdict:** the phase 1 roadmap and executable exit criteria are complete. The
broader source-signal inventory remains future work.

### Phase 2 — existing PyQt5 editor integration

**Implemented**

- `TrackEditorApp`, the existing voice editor, track dictionaries, and preset
  workflow remain in place; no second `QMainWindow` was introduced.
- `parameters.py` centralizes SAM metadata, defaults, bounds, units, tooltips,
  disclosure level, transition policy, and validation paths.
- `SamWorkbenchDialog` edits a copied voice dictionary. Apply publishes the copy;
  Cancel leaves the live voice unchanged; unknown parameters survive.
- Basic, path, HRTF Lab, HRTF modification, and analysis panels are integrated
  into the dialog.
- Preview rendering runs in a worker and uses the canonical render path. Tests
  compare preview audio with export-renderer audio and exercise error reporting,
  edit coalescing, validation-to-field binding, and plot painting.
- The preview UI accepts an absolute start time, allowing a user to audition a
  later automated/transition state without synchronously rendering the preceding
  material.

**Differences or caveats**

- The workbench implements preview-buffer generation and emits PCM to its host,
  but the audit found no phase-6 timeline/modulation/routing editor within it.
- Existing transition start/end fields provide the Phase 2 automation workflow.
  A general automation-lane editor remains intentionally deferred to Phase 6.
- Background preview is tested. A product-level test that a long existing-app
  export remains responsive, can be cancelled, and reports progress is absent.
- Save/reopen behavior is strongly tested at dictionary/preset level, but there
  is no single GUI integration test that constructs a scene and drives the full
  main-window save, reopen, preview, and export workflow.

**Verdict:** the Phase 2 tasks are substantially complete: the existing editor can
construct, audition at an absolute automation time, preserve/save/reopen voices,
and reach the canonical Python export path. A single main-window workflow test and
progress/cancellation coverage would strengthen, rather than define, completion.

### Phase 3 — canonical geometry and path migration

**Implemented**

- Geometry includes line, arc, circle, ellipse, spiral, helix, Lissajous,
  Bézier, spline, polyline, polygon, point cloud, and transformed geometry.
- Traversal includes loop, one-shot, ping-pong, discontinuous, keyframed, and
  stochastic modes, plus discontinuity crossfades and arc-length tables.
- Translation, rotation, scale/reflection, shear, listener transforms, canonical
  serialization, and legacy pixel-coordinate conversion are explicit.
- The geometric binaural renderer includes receiver geometry, distance law,
  fractional propagation delay, and optional Doppler.
- A workbench path panel and visual path editor write canonical path metadata
  while retaining a compatibility profile. Tests cover analytic ITD/level,
  canonical coordinates, migration round trips, discontinuities, and GUI edits.

**Differences or caveats**

- The workbench uses a new SAM path editor rather than extending the legacy
  `custom_path_creator_dialog.py` in place. It deliberately avoids importing the
  legacy editors. This preserves data compatibility but differs from the roadmap
  instruction to reuse/extend the existing interaction model.
- The repository contains a translator for legacy segment mappings, but the
  audit found no production call from `spatial_trajectory_dialog.py` into that
  translator.
- No basic 3-D trajectory viewer is present. The plan allows this only after 2-D
  compatibility, so this is a deferred enhancement rather than a blocker.
- Environment features from the broader plan—air absorption, early reflections,
  late reverb, and occlusion—are not part of the geometric renderer.

**Verdict:** the numerical core is close to the phase roadmap; legacy-editor and
main-application integration is incomplete.

### Phase 4 — SOFA and HRTF rendering

**Implemented**

- SOFA loading handles metadata, coordinate conversion, sample-rate conversion,
  delay units/policies, validation, hashing, storage, and caching.
- HRTF direction selection and interpolation include nearest, neighbor/triangle,
  aligned delay/log-magnitude, and spherical-harmonic-related modes.
- Renderers maintain convolution state and crossfade changed filters rather than
  switching them discontinuously.
- The HRTF Lab provides explicit asset/subject selection, metadata and quality
  reporting, spherical coverage, audition directions/signals, ratings, and
  optional headphone correction.
- Compatibility parameters include `rendererMode`, `hrtfAsset`, and versioned
  `hrtfOptions`. Missing optional HRTF capabilities degrade with actionable
  messages; canonical HRTF code does not import `slab`.
- Tests cover synthetic SOFA ingest, coordinates, delay policies, resampling,
  interpolation, render continuity, optional dependencies, asset discovery, and
  the HRTF Lab GUI.

**Differences or caveats**

- The exact roadmap label `sh_logmag_delay` is not the sole public spelling; the
  implementation supports a larger interpolation vocabulary and compatibility
  aliases. Schema/UI documentation should remain authoritative to avoid drift.
- The legacy `audio_engine.SAMVoice` path remains frozen as required. A dedicated
  GUI migration-preview workflow comparing that renderer with explicit SOFA was
  not identified; HRTF Lab audition/comparison covers related comparisons but
  does not silently migrate legacy presets.
- `sofar` and `pyfar` are not base dependencies. They are optional in
  `requirements-hrtf.txt`, while `h5py` provides standard SOFA reading. This is a
  deliberate capability-layer interpretation of the technology section, not a
  functional failure.

**Verdict:** the principal phase 4 rendering and GUI capabilities are present;
the explicit legacy-migration comparison remains a gap.

### Phase 5 — HRTF modification and hybrid mode

**Implemented**

- Decomposition separates delays, common/differential response, smooth spectral
  shape, and pinna residuals.
- Cue transforms cover ITD, ILD, pinna, distance, coherence, and extra
  differential delay with neutral defaults and range warnings.
- Derived SOFA output records provenance, verifies the result, hashes the source,
  and uses shared-gain normalization.
- Hybrid rendering keeps physical and creative stages distinct. The broadband
  anchor is opt-in, off by default, and supports controlled comparisons.
- The HRTF modification panel provides progressive ranges, reset, response
  plots, throttled updates, level-matched blinded A/B/X trials, derived-file
  output, and unknown-option preservation.
- Tests cover neutral transforms, intended cue changes, common response, delay
  relationships, derived verification/provenance, hybrid separation, anchor
  level control, GUI comparison, and track/preset option round trips.

**Differences or caveats**

- Portable path resolution exists for absolute, project-relative, and configured
  library paths, but packed-project asset management is not a complete workflow.
- The broader project model still does not expose the full typed
  `HybridSpatializerSpec` union described in the plan; compatibility dictionaries
  remain the main integrated representation.

**Verdict:** the phase 5 roadmap is substantially complete.

### Phase 6 — timeline, multiband, and multi-source

**Implemented as core primitives**

- `stages.py` maps existing step mappings to named stages, supports grouped
  contiguous steps, transition envelopes, stable-ID/path bindings, overlap
  blending, validation, and serialization.
- `modulation.py` provides a modulation matrix, route validation and cycle
  detection, additive route evaluation, serialization, and bounded parameter
  search.
- `dsp/crossover.py` provides reconstructing Linkwitz–Riley band splitting.
- `render/routing.py` provides per-band gains/enables, stable source seeds,
  source/bus mute and solo, source and bus stems, and deterministic master mix.
- `trajectory/coupling.py` implements shared, offset, mirrored, orbiting,
  attraction, repulsion, and phase-locked relationships as absolute-time pure
  functions.
- `cost.py` estimates source × band × HRIR/interpolation cost, measures local
  convolution throughput, reserves 50% callback headroom, and returns a
  real-time/offline decision.
- Tests explicitly exercise eight sources, four bands, deterministic re-render,
  chunked coupling, route cycles, stage transitions, reconstructing crossovers,
  stems/buses, and fallback decisions.

**Missing product integration**

- The canonical `Project` contains audio, listener, output, sources, metadata,
  and extras only. It has no typed stages, automation, buses, routing, bands, or
  coupling fields, and therefore does not validate or serialize phase-6 objects
  as part of a normal project document.
- Production searches find phase-6 classes only in their defining modules. They
  are not called by `render/scene.py`, `render_project`, `export_wav`, the
  compatibility adapter, `sound_creator`, or BinauralBuilder session assembly.
- The end-to-end eight-source/four-band test creates synthetic stems and invokes
  the routing primitives directly. It does not render eight canonical sources
  through per-band spatializers from a project/track and export their master and
  stems.
- “Per-band routing” currently applies gain/enable after splitting. There is no
  integrated per-band choice of path or spatializer, although that is required
  by the broader degrees-of-freedom and renderer design.
- No phase-6 PyQt panels expose named stages, automation bindings, the modulation
  matrix, parameter search, buses/stems, coupling, or the cost/offline decision.
- The cost estimator returns a decision, but neither preview nor export consults
  it. Therefore an expensive scene is not actually routed to an offline job by
  the application.
- The `<50%` callback-load exit criterion is encoded as a threshold and unit
  tested mathematically, but supported real-time presets have not been benchmarked
  on a declared reference machine. The current SAM preview is a background
  offline-buffer job rather than a block-callback streaming engine.
- Stage transitions are deterministic as isolated value resolution, but the
  resolved bindings are not applied to renderer parameters in the scene/export
  path. Product-level staged renders therefore are not yet demonstrated.

**Verdict:** phase 6's algorithms and data structures are well designed and
tested, but most phase 6 user-facing and render-pipeline acceptance criteria are
not complete.

## Cross-cutting differences from the detailed plan

### Data model breadth

The canonical dataclasses are intentionally smaller than the section 8 model.
`Project` lacks environment, stages, automation, buses, experiment, and asset
collections. `Source` lacks typed amplitude controls, trajectory, spatializer,
modulators, and routing. Compatibility dictionaries and standalone phase modules
carry many of these concepts, but that prevents aggregate validation and a
single reconstructable project schema.

### Control and signal breadth

The exact SAM control vocabulary is complete, but “every useful numeric parameter
is automatable” is not yet true. Most HRTF, path, hybrid, routing, and phase-6
parameters are scalar configuration values rather than `ControlSpec` instances.
The project signal model supports periodic carrier waveforms but not yet the full
harmonic/noise/file inventory.

### Rendering/output workflow

WAV plus a detailed manifest is implemented. FLAC, selected stem export, packed
assets, loudness targeting/dither, and a phase-6 routed scene manifest are not.
One shared gain and channel-major conventions are consistently respected.

### Test strategy

The focused suite is large and fast and gives unusually good regression coverage.
Remaining differences from section 16 include:

- no Hypothesis dependency or property-based suite despite the technology/test
  strategy requesting it;
- no formal golden spectral/delay tolerance report against a licensed real HRTF;
- no declared-reference-machine real-time benchmark or callback-load gate;
- limited full-main-window workflow coverage for preview/export/project reopen;
- no phase-6 project-to-render-to-stems integration test.

## Recommended completion order

1. **Integrate phase-6 types into the versioned project schema.** Add explicit
   migrations and aggregate validation for stages, modulation routes, buses,
   band routing, and coupling while preserving unknown fields.
2. **Compile one immutable render plan.** Resolve stages and modulation at
   absolute sample positions, construct coupled trajectories, render each source
   and band through its selected spatializer, and feed `mix_routed` without
   duplicating DSP.
3. **Connect the cost decision.** Have preview measure/cache throughput, show the
   estimate, and dispatch expensive renders to the existing background offline
   path with progress/cancel behavior.
4. **Add progressive phase-6 GUI panels.** Begin with stage labels over existing
   steps and a searchable modulation matrix; add routing/coupling after their
   project round trips are stable.
5. **Complete control compilation.** Add composite, step, stochastic, restricted
   expression, and external controls, then use them for currently scalar spatial
   and HRTF parameters.
6. **Add acceptance-level integration tests.** Cover project save/reopen, eight
   rendered sources × four independently spatialized bands, deterministic staged
   chunking, bus/stem export, expensive-preview fallback, and responsive GUI
   export.
7. **Benchmark named presets.** Record interpreter, CPU, sample rate, block size,
   assets, and callback load on a reference machine; treat the 50% threshold as
   measured acceptance rather than only an estimator constant.

## Bottom line

The implementation is not a façade: phases 0–5 contain real canonical DSP,
compatibility, asset handling, and GUI functionality backed by extensive tests,
and phase 6 contains credible, reusable algorithms. The discrepancy is that the
phase-6 commit added a **library layer beside the application pipeline**, not the
final application wiring described by the roadmap. Closing that integration gap,
then expanding the typed controls/project schema, is the shortest path to an
honest “phase 6 complete” designation.

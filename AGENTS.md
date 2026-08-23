# BinauralBuilder SAM/HRTF Workbench

## Current implementation guide for coding agents

**Repository:** <https://github.com/abehlok2/BinauralBuilder>  
**Default branch:** `main`  
**Implementation audit baseline:** `0168dcfee3c1f6d334e6b47902358351d0365154`, inspected 2026-08-23  
**Document status:** Current-state guide. It replaces the earlier greenfield roadmap that described `src.audio.sam_workbench` and most of its renderer, trajectory, scene, export, and GUI layers as missing.

Reinspect the current branch before every implementation task. The baseline above records what this document was checked against; it is not permission to assume the repository has stopped changing.

## 1. Purpose and implementation stance

BinauralBuilder now contains a substantial Python/PyQt5 SAM and HRTF workbench. An agent working here should extend and consolidate the implementation that exists, not recreate the original roadmap or introduce another application.

The acoustic mechanisms may be implemented and tested precisely. Claims about altered consciousness, neural targeting, microtubules, nonlocal effects, healing, or therapeutic outcomes remain hypotheses and subjective intentions, not validated product behavior. Use acoustic and perceptual terminology in code, controls, documentation, manifests, and presets. Preserve experiment tooling that supports controlled comparison without presenting a desired outcome as guaranteed.

The source basis remains:

- `US20130010967A1.pdf`, for the spatial-angle-modulation mechanism and path concepts;
- `2012-1 (TMIJ) Induction of Expanded States of Consciousness Using Spatial Angle Modulation Audio Support Technology - F. Holmes Atwater.pdf`, for historical and experimental context;
- the current repository code and tests, which are authoritative for implemented behavior;
- SOFA specifications and the metadata of the selected HRTF asset, which are authoritative at the HRTF boundary.

## 2. Non-negotiable boundaries

1. **Python and PyQt5 only.** Rust is outside this workbench scope. Do not modify, port, test, document, or require changes in `src/realtime_backend/`, `src/audio/rust_stream_player.py`, Rust streaming, foreign-function interfaces, WebAssembly, or GPU backends unless a user explicitly opens a separate Rust task.
2. **Keep the existing application.** Preserve `main.py`, `TrackEditorApp`, `VoiceEditorDialog`, the session builder, existing track JSON, `.voice` presets, public synth-function names, and the current step/voice workflow. Do not create another `QMainWindow`, application shell, or competing track format.
3. **Keep the canonical core headless.** `src/audio/sam_workbench/` must not import PyQt5, `main.py`, `src.ui`, `sound_creator`, `audio_engine`, `slab`, or the Rust backend. `tests/sam_workbench/test_clean_import.py` enforces this dependency direction.
4. **One acoustic implementation.** New SAM, trajectory, geometric, HRTF, hybrid, routing, and control behavior belongs in `src.audio.sam_workbench`. Compatibility surfaces delegate to it. Do not copy DSP into both `src/synth_functions` and `binauralbuilder_core/synth_functions`.
5. **Preserve serialized data.** Existing fields, unknown fields, unrecognized extension data, asset references, and legacy path payloads must survive load/edit/save. Add explicit migrations for schema changes. Never silently reinterpret a document from a newer schema.
6. **Preserve legacy SAM2 behavior.** Unversioned SAM2 voices keep their established polarity and defaults. New versioned voices may use canonical polarity, but migration must be explicit and tested.
7. **Preserve the array boundary.** The canonical audio shape is channel-major `(2, frames)`. BinauralBuilder's legacy synthesis/mixing boundary is frame-major `(frames, 2)`. Transpose exactly once in the named compatibility adapter and assert both layouts with asymmetric fixtures.
8. **Use absolute sample time.** Chunking, preview windows, seeking, transitions, automation, seeded randomness, path traversal, and filter selection must not restart at block boundaries.
9. **Use explicit HRTF assets.** Canonical HRTF and hybrid rendering require a selected SOFA file and retain its content hash and policies. Do not introduce an implicit `slab.HRTF.kemar()` dependency into the new core.
10. **Preserve binaural relationships.** Never normalize ears or HRTF directions independently. Apply one shared gain when normalization is needed. Record it.
11. **Do not optimize away compatibility or evidence.** Profile before optimizing, retain correctness tests for the replaced path, and report measured changes rather than assumed gains.
12. **Treat warnings honestly.** Validation answers whether a configuration is legal. Readiness answers whether it is likely to produce the intended render. Do not suppress missing assets, hash mismatches, uncovered trajectories, ignored controls, clipping risk, or unsupported capabilities.

## 3. What is implemented now

The last reported full local suite at the audited baseline was **1493 passed, 0 failed**. The repository does not currently have continuous integration, so this is historical local evidence, not a live status badge. Run the relevant tests again before editing.

| Area | Current state | Principal code |
|---|---|---|
| Canonical package | Implemented, Qt-free, import-tested | `src/audio/sam_workbench/` |
| Standalone project model | Versioned `Project`, validation, migrations, atomic persistence, unknown-field preservation | `model.py`, `migrations.py`, `validation.py`, `version.py` |
| Track-level scene | Versioned `track_data["sam_scene"]`, stable source identifiers, stages, modulators, routing, assets, environment, experiment state | `scene_state.py` |
| Compiled scene contract | Immutable, sample-accurate `CompiledScenePlan` from a track or standalone project | `plan.py` |
| Renderer registry | One source of renderer identifiers, schemas, assets, validation, capabilities, cost, latency, tail, factories, and migration hooks | `render/registry.py` |
| Abstract SAM | Exact/generalized opposed-ear phase modulation, transition and block continuity, legacy polarity adapter | `dsp/source.py`, `render/abstract_pm.py`, `compat.py` |
| Geometric binaural | Per-ear fractional delay, distance law, listener transform, head shadow, optional Doppler | `render/geometric.py` |
| Three-dimensional paths | Canonical geometry/traversal/transform separation, full-height primitives, keyframes, legacy conversion, four-view editor | `trajectory/`, `src/ui/sam_path3d_dialog.py`, `sam_path3d_views.py` |
| Explicit-SOFA HRTF | Loading, validation, coordinate conversion, delay policies, resampling, coverage, interpolation, convolution, cache | `hrtf/`, `render/hrtf.py` |
| HRTF cue modification | ITD, ILD, pinna residual, distance, coherence, extra differential delay, derived SOFA provenance | `hrtf/modification.py`, `hrtf/derived.py` |
| Hybrid and spatial anchor | Physical HRTF stage followed by declared creative cue modification; optional broadband companion | `render/hybrid.py`, `render/anchor.py` |
| Scene mixing | Stable source routing, real buses and stems, multiband processing, mute/solo, stateful crossover continuity | `render/scene_mix.py`, `render/routing.py` |
| Timeline and modulation | Named stages, sample-accurate parameter series, defined modulators, seeded block-independent randomness, cycle detection | `stages.py`, `modulation.py`, `scene_state.py` |
| Analysis and experiments | Waveform/spectrum/cues/trajectory analysis, localization tests, condition sessions, measurement and Mesh2HRTF validation routes | `analysis/`, `experiment/`, `hrtf/measurement.py`, `hrtf/mesh2hrtf.py` |
| GUI integration | Existing editor entry point, Basic/Advanced/Expert disclosure, registry-driven controls, HRTF Lab, routing/stage/modulation panels, readiness and signal-flow summary | `src/ui/sam_*`, `voice_editor_dialog.py` |
| Background export | Immutable snapshots, worker thread, progress, cancellation, failure cleanup, render metrics, safe close | `src/audio/render_job.py`, `src/ui/render_job.py`, `main.py` |
| Reconstruction manifests | Ordinary WAV/FLAC track export writes a sidecar capable of reconstructing a renderable track | `src/audio/track_manifest.py` |
| Long-render optimization | Block convolution, cached/batched paths, adaptive spatial updates, cross-render filter cache, sequential chunking, bounded encoder allocation | `dsp/binaural_convolution.py`, `streaming.py`, `sound_creator.py` |

Several advanced components exist both as headless capabilities and as GUI workflows. Their presence does not mean every future roadmap idea is complete. The remaining production debts are listed in section 12.

## 4. Sources of truth and dependency direction

There are three related scene representations. Their ownership must remain explicit:

| Representation | Meaning | Edited by | Persisted by |
|---|---|---|---|
| `track_data["sam_scene"]` | The authoritative scene inside a normal BinauralBuilder track | Existing GUI panels and track commands | Track JSON |
| `model.Project` | The typed standalone command-line/test document | CLI and headless callers | Its own JSON document |
| `CompiledScenePlan` | Immutable, validated, sample-accurate render description derived from either front door | Nobody | Never |

The rule is: **edit a persisted document, compile a plan, render from the plan or its compatibility bridge**. Do not make a Qt model, renderer instance, preview buffer, HRTF Lab state, manifest, or cache into another authority.

Current application flow:

```text
track JSON / .voice preset
    -> TrackEditorApp and VoiceEditorDialog
    -> track_data["sam_scene"] plus source-specific voice parameters
    -> scene validation / readiness / CompiledScenePlan
    -> compat.render_sam2_voice (current production voice bridge)
    -> renderer registry and canonical renderer
    -> channel-major stereo stem (2, frames)
    -> compatibility transpose to (frames, 2)
    -> scene buses / step mix / clips / background noise
    -> shared output processing
    -> WAV or FLAC plus reconstruction manifest
```

The GUI obtains renderer names and capabilities from `render.registry.REGISTRY`, interpolation modes from `hrtf.interpolation.INTERPOLATION_MODES`, and delay policies from the HRTF subsystem. Do not reintroduce local lists that can drift.

## 5. Canonical conventions

### 5.1 Coordinates and binaural signs

The implemented canonical frame in `conventions.py` is right-handed and listener-relative:

- `+x`: forward;
- `+y`: left;
- `+z`: up;
- azimuth `0 degrees`: front;
- positive azimuth: toward the left;
- positive elevation: upward;
- distance: metres;
- left receiver: positive `y`;
- right receiver: negative `y`;
- positive ITD: the left ear leads.

This differs from the legacy `slab` convention described in the old renderer (`+x` right, `+y` forward, positive azimuth toward the right) and from GUI scene/pixel coordinates. Convert at named boundaries. Do not change the canonical axes to match a UI view or external library without a schema migration and exhaustive conversion tests.

SOFA metadata is authoritative for each asset's stored coordinate type and units. Convert it to canonical Cartesian metres at ingest. Spherical entry and display are supported, but the internal trajectory position remains Cartesian.

### 5.2 Time, phase, level, and arrays

- Domain time is seconds; render time is an absolute integer sample index.
- `seconds_to_samples` uses the repository's documented half-away-from-zero rule.
- Accumulate time-varying oscillator phase. Never use `sin(2*pi*f[n]*t[n])` for a changing frequency.
- Phase is radians internally and degrees at user-facing boundaries.
- Gain is linear internally and decibels at user-facing boundaries.
- Use `float64` for phase, delay, interpolation, and sensitive analysis; use `float32` for bulk audio when appropriate.
- Core renderers return `(2, frames)`. Existing BinauralBuilder mixers consume `(frames, 2)`.
- Apply one common normalization/master gain to both ears.

### 5.3 Path evaluation

Canonical paths separate:

1. geometry: where the curve exists;
2. traversal: how time maps onto the curve;
3. transform/listener frame: how the curve is placed and oriented.

`PathModel` schema version 2 supports listener-relative and world Cartesian coordinates, listener transforms, optional source orientation, coordinate smoothing, and traversal metadata. Do not materialize a full sample-rate `(frames, 3)` array in a compiled plan. Evaluate paths in blocks or on the renderer's control grid.

Implemented three-dimensional primitives include horizontal, vertical, tilted and spherical orbits; rising arcs; overhead and elevation sweeps; dome traversal; three-dimensional figure-eights; pendulum motion; toroidal paths; seeded random walks; and the general geometry, keyframe, spline, Bézier/polyline, transformation, and traversal types under `trajectory/`.

Discontinuous motion is represented as separate old/new audio branches with an explicit equal-power crossfade. Do not smooth a declared spatial jump by inventing intermediate coordinates unless coordinate smoothing was explicitly selected.

## 6. Serialization and migration contracts

### 6.1 Track-level scene

The persisted scene is `track_data["sam_scene"]`, currently schema version `2`:

```text
sam_scene:
    schemaVersion
    sources
    stages
    modulators
    modulation
    buses
    routing:
        schemaVersion
        buses
        sources
        bands
    assets
    environment
    experiment
    extensions
    ...unknown fields preserved...
```

Version 1 migrates explicitly to version 2 by adding `assets`, `environment`, `experiment`, and `extensions`. Voice-local legacy keys `samStages`, `samModulation`, and `samRouting` migrate to the track-level scene. Shared scene data must not be copied back into every voice.

Every SAM scene source has a stable `sam_source_id` stored on the real track voice. Never derive identity from a list index or display name. Renaming and reordering must not redirect automation or routing. Orphaned scene records are retained and marked rather than destroyed so undo can restore their state.

Do not attach a normalized but otherwise empty scene to a track that never had scene content. An empty normalized mapping is truthy because it carries versions and containers; attaching it can select a different compatibility render path and alter legacy output.

### 6.2 Voice parameters

SAM voice parameters retain their camelCase boundary and accept unversioned legacy data. Important versioned keys include:

```text
samSchemaVersion: 1
rendererMode: abstract_pm | geometric | hrtf | hybrid
canonicalTrajectory: {...schemaVersion: 2...}
hrtfAsset
hrtfAssetHash
hrtfOptions: {...schemaVersion: 1...}
```

Keep source-specific generator, path, renderer, cue, and asset overrides in the voice payload where they belong. Keep shared stages, modulators, buses, routing, environment, and experiment state at track level.

The GUI must preserve unknown voice keys, including fields introduced by a newer build or external extension. Compiled renderer configuration may drop keys it does not consume; persisted documents may not.

### 6.3 Standalone project and manifests

The typed standalone `Project` schema remains `1.0`; package version is `0.1.0`. Schema changes require a migration and version bump.

Normal track exports use `src/audio/track_manifest.py`, track manifest schema version `1`, and SAM scene schema version `2`. A manifest stores complete source parameters, three-dimensional paths, listener pose, renderer capabilities and versions, asset references and hashes, interpolation/delay/distance policies, cue modification, anchor and headphone correction, stage/modulation/routing state, seeds, gain/limiter/dither/quality settings, coverage warnings, and measured render cost. `reconstruct_track` must continue to reproduce a renderable acoustic condition from the sidecar.

## 7. Renderer architecture

`render/registry.py` is the only renderer catalog. A renderer definition owns:

- identifier and configuration version;
- typed/defaulted configuration fields;
- aggregate validation;
- required assets and hashes;
- factory hook;
- latency and tail estimates;
- relative cost;
- migration hook;
- capability and honesty metadata used by the GUI.

The four registered modes are:

| Mode | Physical claims | Important constraints |
|---|---|---|
| `abstract_pm` | Creative opposed-ear phase modulation; not reliable physical direction, distance, or height | Generates its own stereo source; trajectory may control the modulation mapping |
| `geometric` | Physical interaural timing/level and distance from a simplified head model | Does not provide pinna-derived physical elevation; height may change geometry without convincing height localization |
| `hrtf` | Direct binaural convolution with azimuth/elevation/distance cues from an explicit SOFA dataset | Height is only as valid as dataset coverage, subject match, and interpolation |
| `hybrid` | HRTF stage plus declared creative cue modification | Anything after the HRTF stage is a deliberate departure from measured cues and must remain visible in the signal chain and manifest |

Do not add a mode to a GUI combo, validator, cost table, or compatibility `if` chain without registering it. Prefer eliminating such local branching in favor of the registry.

### 7.1 Direct binaural and Ambisonics status

The canonical SAM/HRTF workbench is **object-based direct binaural**, not an Ambisonic scene renderer. Each mono source is spatialized to a stereo ear signal before bus mixing. The `spherical_harmonic` HRTF interpolation mode fits aligned HRTF log magnitude and delay over measurement directions; it does not encode program audio into spherical-harmonic soundfield channels.

The wider repository does contain `src/synth_functions/spatial_ambi2d.py` and a duplicate compatibility copy under `binauralbuilder_core`. That is a separate, legacy-style two-dimensional first-order Ambisonic voice: W/X/Y, SN3D, elevation fixed to zero, cardioid stereo decoding, followed by time-varying ITD/ILD. It is not part of `CompiledScenePlan`, the renderer registry, the three-dimensional HRTF path, or scene-level higher-order Ambisonics. Do not describe the entire repository as having no Ambisonic code, and do not describe the canonical workbench as Ambisonic.

If a future task adds Ambisonics, prefer an optional full-sphere higher-order scene/environment bus using explicit ACN channel order and SN3D normalization. Keep direct HRTF as the default for precise discrete sources and height; use the Ambisonic bus for diffuse ambience, spatial reverb, source swarms, head-pose rotation, or reusable multichannel scene export. Headphone output still needs a binaural HRTF decoder. Do not replace direct HRTF with first-order Ambisonics and claim improved elevation.

## 8. HRTF processing and continuity

### 8.1 Assets and interpolation

Basic SOFA loading is part of the standard dependency set through `h5py`. The optional HRTF dependency group adds `sofar` and `pyfar` for standards verification and advanced workflows. The application must remain usable in non-HRTF modes when those optional packages are unavailable.

SOFA assets may resolve from an absolute path, project-relative path, or `SAM_WORKBENCH_HRTF_DIR`. Preserve the selected file, hash, subject/database metadata, sample rate, coverage, receiver order, coordinate metadata, `Data_Delay`, and processing policy.

Production interpolation modes currently include:

1. `nearest`;
2. `three_neighbor`;
3. `spherical_triangular`;
4. `delay_magnitude`, with legacy alias `logmag_delay`;
5. `spherical_harmonic`.

Every interpolated mode aligns delay before blending. Do not average raw unaligned HRIR samples. Delay policies are explicit: `bake_delay_into_ir` or `keep_external_delay` (the historical spelling `preserve_external_delay` is still read and validates as an alias). Never ignore nonzero `Data_Delay`, and never add a second geometric ITD on top of an HRTF unless hybrid configuration explicitly requests an extra cue.

### 8.2 Time-varying convolution

The production HRTF path uses block overlap-save through `dsp/binaural_convolution.py`. It transforms an input block once and applies left/right and transition filters against the same input history. Static and partitioned paths are tested against direct convolution.

Filter transitions must obey the continuity fix documented in `docs/hrtf_transition_continuity.md`:

- selection and processing are interleaved on a control grid anchored to absolute sample zero;
- a transition is planned between adjacent grid points;
- `crossfadeMs` remains user-controlled but is capped at the active control interval;
- a running transition is allowed to finish; a new request must not abandon and restart it;
- continuous correlated material uses a continuity-preserving linear filter interpolation rather than repeated equal-power loudness bulges;
- diagnostics retain filter requests, transition starts/completions, queued requests, and fade restarts.

The reported toroidal-path buzzing was caused by restarting unfinished HRIR transitions, not by the path primitive or by a need for Ambisonics. Do not hide a discontinuity with heavier smoothing; preserve the invariant that every interval finishes on the filter from which the next interval starts.

### 8.3 Coverage and perceptual honesty

The HRTF editor and readiness checks compare a path against the selected dataset's measured coverage and median measurement distance. No selected dataset means “not checked,” not “covered.” An unreadable dataset and a hash mismatch are explicit failures.

Low-frequency carriers contain little pinna-cue energy. The optional broadband spatial anchor exists to add localizable high-frequency content and is disabled by default, nominally around `-30 dB`. Never enable it silently or mistake an anchor-on comparison for the original signal.

## 9. Scene, automation, routing, and timing

- `scene_state.py` evaluates stage and modulation controls as functions of absolute time.
- Modulators carry waveform, rate, phase, and seed. Triangle and seeded random forms are implemented; random values are derived from absolute time so chunk boundaries do not alter them.
- Automation binds to stable source identifiers plus registered parameter paths. Non-automatable paths generate a warning rather than being applied silently.
- Gain automation is sample-accurate and separate from bus routing so routing is applied exactly once.
- `render/scene_mix.py` creates real source and bus stems, applies bus gain/mute/solo and optional multiband processing, and preserves crossover state across blocks.
- Per-source seeds derive from the project seed and stable source identifier, not source order.

Render-window timing must use an interval intersection. A source contributes only between its absolute start and end samples, including correct offsets for windows entirely before, overlapping either edge, entirely inside, and entirely after the source. One large window and multiple adjacent windows must agree.

Renderer latency and tail are part of the plan. Do not truncate convolution ring-out or propagation delay merely because the source generator has ended.

## 10. GUI and usability contracts

The current workbench is hosted by `VoiceEditorDialog` as `SamWorkbenchDialog`; it is not another application. It edits a copy and commits through the existing save path.

Current usability behavior to preserve:

- first use opens in Basic disclosure mode; the last selected Basic/Advanced/Expert mode is remembered afterward;
- every parameter group is constructed regardless of disclosure mode so switching modes cannot drop hidden values;
- renderer selection, tab relevance, option lists, and renderer-specific validation derive from canonical registries;
- irrelevant tabs are disabled with an explanatory tooltip rather than silently disappearing;
- a persistent signal-flow summary states the active source, path, renderer, cue, headphone, and output stages and names controls the selected renderer ignores;
- routing rows come from actual track voices and show human names while storing stable identifiers;
- the four-view three-dimensional editor uses the canonical `PathModel`, shows dataset-based coverage, and can promote a legacy two-dimensional profile into the ear-height canonical plane while preserving provenance;
- HRTF preprocessing, audition, analysis, and final export run outside the GUI thread;
- worker threads do not touch Qt widgets directly;
- GUI tests isolate `QSettings` from the developer's real preferences.

Readiness findings should be actionable and tied to a field or source. Distinguish errors, warnings, and advice. A missing optional headphone profile is advice; a changed required SOFA hash is an error.

## 11. Export and performance

### 11.1 Background export

Final export begins from an immutable deep-copied `RenderSnapshot`. It runs on a worker, reports progress, checks cancellation at chunk progress points, propagates `RenderCancelled`, cleans newly created partial output after cancellation/failure, preserves any file that existed before the job, and asks before closing a window with active work.

The step tester is an exception and remains synchronous; see section 12.

### 11.2 Implemented optimizations

Do not reimplement these without profiling evidence:

- block overlap-save and partitioned binaural convolution instead of per-sample Python convolution;
- one input transform reused across both ears and transition filters;
- cached geometry arc-length tables and batched trajectory queries;
- adaptive HRTF updates bounded by angular error on an absolute control grid;
- module-level, cross-render interpolated-filter cache keyed by impulse-response fingerprint and all processing parameters;
- streaming normalization/encoding with bounded temporary storage;
- sequential synthesis chunking when returned state proves a voice can resume continuously;
- declick fades only at true voice edges, never every internal chunk boundary;
- stateful bus crossover processing across blocks.

The current measured export throughput in `docs/sam_workbench_phase3.md`, in audio-seconds per wall-clock second, is:

| Case | Throughput |
|---|---:|
| `abstract_pm` | 156.2 |
| `geometric` | 19.2 |
| HRTF nearest, 128-sample minimum interval | 6.5 |
| HRTF nearest, 512-sample minimum interval | 28.9 |
| HRTF delay-magnitude, 128-sample minimum interval | 4.2 |
| HRTF delay-magnitude, 512-sample minimum interval | 25.0 |

These are measurements from one environment, not universal promises. Preserve benchmark configuration and compare before/after on the same machine.

Sequential chunking reduced measured peak allocation for a single-step binaural beat from 437 MB to 182 MB at two minutes and from 2019 MB to 505 MB at ten minutes. This bounds the synthesizer's transient working set, not total render memory: the existing assembly path still holds duration-scaled step and track arrays.

### 11.3 Benchmark commands

```console
python -m src.audio.sam_workbench.benchmark --help
python tools/benchmark_export.py --help
```

When changing performance-sensitive code, report throughput, peak memory, output equivalence, block/window invariance, and cache behavior. Faster output that changes phase, path timing, HRTF cues, source duration, or ear relationships is a regression.

## 12. Known limitations and architectural debt

These are the highest-confidence remaining gaps at the audited baseline:

1. **Production rendering does not yet execute `CompiledScenePlan` end to end.** The plan is authoritative for compilation, validation, timing, assets, controls, latency, and diagnostics, but normal voices still reach renderers through `compat.render_sam2_voice`. The next consolidation should make preview, export, HRTF Lab, analysis, and benchmarks execute the same plan without creating another renderer implementation.
2. **The step tester remains synchronous.** `main.py` retains one `QApplication.processEvents()` call to repaint before loading a test step. Long step previews can still block the window. Move this onto the established render-job machinery.
3. **Total long-render memory still grows with duration.** Streaming encoding and sequential synthesis chunking bound important temporary allocations, but `assemble_track_from_data` still materializes step and track buffers. A true source-to-bus-to-encoder streaming pipeline remains future work.
4. **No cross-run content-addressed disk cache.** The in-process filter cache accelerates repeated renders in one run. It does not persist verified preprocessed HRTFs or render fragments across application restarts.
5. **No seek checkpoints.** Arbitrary-window HRTF/geometric compatibility rendering may reconstruct state with preroll. Persistent checkpoints could make distant seeks and repeated partial exports cheaper.
6. **No production render parallelism.** The window intentionally permits one final render at a time. Safe source/bus parallelism, process-level scheduling, and protection against oversubscribed numerical libraries remain unimplemented.
7. **Ambisonics is not integrated into the canonical scene.** The separate two-dimensional first-order voice is not a three-dimensional higher-order bus and should not be treated as one.
8. **No continuous-integration workflow is configured.** Local results must be reported honestly, including dependency or platform failures.

When addressing these, preserve current acoustic output unless the task explicitly changes an algorithm and supplies new acceptance criteria.

## 13. Recommended next implementation order

### Priority 1 — Make the compiled plan the executable production contract

1. Add a plan executor that consumes `CompiledScenePlan` and uses existing registered renderer factories.
2. Route preview, ordinary export, HRTF Lab audition, analysis, and benchmarking through that executor.
3. Keep `compat.render_sam2_voice` as a compatibility facade that compiles or delegates; do not remove public function names.
4. Prove equivalence for all four renderer modes, scene automation, buses, arbitrary windows, latency/tails, legacy polarity, and manifests.
5. Remove duplicated interpretation only after parity tests pass.

### Priority 2 — Finish responsiveness and true bounded-memory rendering

1. Move step testing to the existing snapshot/job/cancellation model.
2. Stream source blocks into persistent scene buses and then into the two-pass encoder without full step/track arrays.
3. Add deterministic seek/checkpoint state for oscillators, delay lines, crossover filters, convolution, filter transitions, and seeded controls.
4. Add content-addressed preprocessed-HRTF and optional render-fragment caching with complete keys and atomic writes.
5. Evaluate source/bus parallelism only after measuring process/thread overhead and numerical-library oversubscription.

### Priority 3 — Add optional spatial extensibility without weakening direct binaural rendering

Potential extensions include head tracking, early reflections/late reverb, and a full-sphere higher-order Ambisonic environment bus. Add them behind registry capabilities and versioned scene schemas. For Ambisonics, specify order, ACN/SN3D conventions, distance/near-field policy, listener rotation, SOFA-derived binaural decoder filters, latency/tail/cost, manifest fields, and block-invariant tests. Retain direct HRTF for discrete moving sources unless measured comparisons justify a different route.

## 14. Test and development workflow

Before editing:

1. inspect the current branch, latest commits, and working tree;
2. read this file completely;
3. inspect the relevant implementation and its tests;
4. run a focused baseline and record failures before classifying them;
5. distinguish missing optional dependencies or system libraries from code defects, but investigate each failure rather than dismissing it as unrelated.

Suggested setup and test commands:

```console
python -m pip install -r requirements-hrtf.txt
python -m pip install pytest pytest-qt
QT_QPA_PLATFORM=offscreen python -m pytest -q

python -m pytest -q tests/test_sam_phase_zero.py tests/test_sam_phase_one.py
python -m pytest -q tests/sam_workbench/test_scene_plan.py
python -m pytest -q tests/sam_workbench/test_scene_timing.py
python -m pytest -q tests/sam_workbench/test_renderer_registry.py
python -m pytest -q tests/sam_workbench/test_path_3d.py tests/sam_workbench/test_path_3d_render.py
python -m pytest -q tests/sam_workbench/test_canonical_hrtf_engine.py
python -m pytest -q tests/sam_workbench/test_hrtf_transition_continuity.py
python -m pytest -q tests/sam_workbench/test_long_render_memory.py
python -m pytest -q tests/sam_workbench/test_phase3_acceptance.py
```

The command-line front door remains:

```console
python -m src.audio.sam_workbench new session.sam.json --name "Reference SAM" --with-source
python -m src.audio.sam_workbench validate session.sam.json
python -m src.audio.sam_workbench render session.sam.json render.wav --duration 30
```

For every implementation task:

- use small pure functions for phase, delay, coordinates, path evaluation, HRTF decomposition, interpolation, and migration;
- keep units explicit in API names;
- use immutable or defensively copied render inputs;
- use deterministic seeds and absolute sample origins;
- preserve unrelated working-tree changes;
- add focused regression tests and then run the relevant wider suite;
- update the appropriate user guide, phase note, or manifest schema documentation;
- report files changed, schema/migration impact, tests and results, performance evidence, compatibility behavior, and remaining limitations.

## 15. Acceptance rules for future work

A change is not complete merely because a control exists or a unit test calls an internal helper. Depending on scope, demonstrate through production entry points that:

- preview and export use the same algorithm and settings;
- whole, blocked, chunked, and arbitrary-window renders agree within declared tolerance;
- source start/end timing and renderer tails are correct;
- audio shapes, ear order, SAM2 polarity, and coordinate signs are explicit;
- all enabled GUI controls affect rendering and ignored controls are named;
- all four current renderers remain previewable/exportable unless the task explicitly changes support;
- SOFA policies, hashes, coverage, and cue modifications reach the manifest;
- cancellation and failure leave no new partial artifact;
- reconstruction manifests reproduce the acoustic condition;
- long renders remain bounded at the layer the change claims to bound;
- old track JSON, `.voice` presets, unknown fields, and public Python entry points still work;
- the canonical package imports without Qt, `slab`, or Rust.

Do not weaken an existing test to accommodate a regression. If a contract must change, explain why, add a migration where serialized meaning changes, and replace the test with one that captures the intentional new behavior.

## 16. Current reference documents

- `src/audio/sam_workbench/README.md` — module inventory and canonical package rules;
- `src/audio/sam_workbench/HRTF_ASSETS.md` — development asset policy;
- `docs/sam_workbench_scene_plan.md` — persisted versus standalone versus compiled scene ownership;
- `docs/sam_workbench_phase2_performance.md` — convolution, trajectory, caching, chunking, and memory measurements;
- `docs/hrtf_transition_continuity.md` — the fast-path buzzing cause, continuity invariant, fix, and measurements;
- `docs/sam_workbench_phase3.md` — registry-driven GUI, real scene sources, three-dimensional editor integration, background export, manifests, readiness, tests, and current limitations;
- `docs/sam_workbench_user_guide.md` and `src/ui/sam_workbench_manual.py` — user-facing behavior;
- `tests/sam_workbench/` — executable contracts.

External resources define formats and tools rather than validating speculative consciousness claims:

- SOFA conventions: <https://www.sofaconventions.org/>
- SOFA HRTF databases: <https://sofacoustics.org/data/database/>
- `sofar`: <https://sofar.readthedocs.io/>
- `pyfar`: <https://pyfar.readthedocs.io/>
- Mesh2HRTF: <https://github.com/Any2HRTF/Mesh2HRTF>

# SAM/HRTF Spatial-Audio Workbench

## Comprehensive Python + PyQt5 implementation plan

**Document purpose:** Give a coding agent enough technical, architectural, and product detail to implement an extensible spatial-angle-modulation (SAM) audio workbench without having to reinterpret the source documents.

**Source basis:**

- `US20130010967A1.pdf` — patent description of spatial angle modulation, path variants, parameterization, multiple sources, and staged sessions.
- `2012-1 (TMIJ) Induction of Expanded States of Consciousness Using Spatial Angle Modulation Audio Support Technology - F. Holmes Atwater.pdf` — explanatory article and experimental/theoretical context.
- Follow-up design discussion covering HRTF creation, modification, interpolation, cue scaling, and personalized measurement.

**Implementation stance:** Reproduce the acoustic mechanisms faithfully, while treating claims about altered consciousness, neural targeting, microtubules, nonlocal effects, or therapeutic outcomes as hypotheses rather than validated product behavior. Describe controls in acoustic terms and provide experiment tools for testing subjective effects.

---

## 1. Outcome and scope

Build a desktop application for constructing, auditioning, analyzing, automating, and exporting stereo spatial-audio scenes in four modes:

1. **Abstract phase modulation** — patent-style opposed-ear phase modulation.
2. **Geometric binaural rendering** — moving sources with time-varying distance, delay, level, and optional Doppler.
3. **HRTF rendering** — dynamic convolution with a SOFA HRTF dataset.
4. **Hybrid rendering** — physical HRTF motion plus intentionally nonphysical cue modulation.

Keep DSP independent of the GUI. Every useful numeric parameter should be optionally time-varying, modulatable, reproducible, and serializable. Prioritize a reliable offline renderer with responsive preview; head tracking, sensor feedback, personal HRTF measurement/import, swarms, and multimodal synchronization are later work.

In scope are exact SAM, multiple sources, arbitrary 2D/3D trajectories, separate geometry/traversal/listener geometry, SOFA HRTFs, time-varying binaural cues, multiband spatialization, a broadband spatial anchor, timeline stages, presets, A/B comparison, deterministic rendering, WAV/FLAC plus manifests, analysis, subjective experiments, and a PyQt5 model/view GUI.

First-release non-goals are medical diagnosis/treatment or guaranteed mental-state induction; claims that HRTFs target brain structures or hemispheres; full room simulation, wave-field synthesis, loudspeaker arrays, or a production DAW; live personal-HRTF measurement; and mobile deployment.

## 2. Core acoustic requirements

For a sinusoidal carrier implement:

\[
s_L(t)=A(t)\sin(\theta_c(t)+\beta_L(t)q_L(\psi_L(t))+\phi_L(t))
\]

\[
s_R(t)=A(t)\sin(\theta_c(t)-\beta_R(t)q_R(\psi_R(t))+\phi_R(t))
\]

The exact symmetric preset uses `theta_c(t) = 2*pi*f_c*t`, `q = sin(2*pi*f_m*t)`, and equal depths. Its instantaneous frequencies are `f_c +/- beta*f_m*cos(2*pi*f_m*t)` and IPD is `2*beta*sin(2*pi*f_m*t) + phi_L - phi_R`. Preserve this reference mode while allowing independent ear depth, rate, waveform, phase, envelope, and bias.

Generalize the model to multiple modulators and time-varying amplitude, carrier phase, frequency, depth, and bias. Carrier phase must be accumulated, not computed as `2*pi*instantaneous_frequency*t`. Signal generation may use sine, a band-limited waveform, wavetable, or source-signal phase transform.

Represent motion as three independent concepts: geometry `C(u)`, traversal `u(t)`, and transform `T(t)`. Support line, arc, circle, ellipse, spiral, helix, Lissajous, spline, Bézier, polygon, and point-cloud paths; open/closed paths; forward/reverse/ping-pong/loop/one-shot/stochastic/discontinuous traversal; constant, eased, keyframed, arc-length-compensated, and external speed; 2D/3D coordinates; coupled paths; and configurable crossfades at jumps.

Allow unlimited sources in the model and optimize 1–8 active sources initially. Sources may share, offset, mirror, repel, attract, orbit, phase-lock, or move independently. Provide user-editable, nonmedical macro stages controlling parameter groups through envelopes or preset transitions.

## 3. Degrees of freedom

Expose controls through Basic, Advanced, and Expert/Matrix views:

- **Signal:** waveform/file, carrier frequency/phase/amplitude, harmonics, noise color/bandwidth, transients.
- **Modulation:** waveform, rate, phase, depth, bias, duty cycle, symmetry, slew, jitter, and per-ear polarity/depth/rate.
- **Geometry/traversal:** Cartesian or spherical position, primitives/control points/transforms/deformation, closure, velocity/acceleration/easing/direction/loops/jumps/arc-length mode.
- **Listener/cues:** position and yaw/pitch/roll, ear spacing, reference frame, ITD, ILD, IPD, common delay, pinna spectrum, coherence, width, externalization, and distance.
- **HRTF/environment:** asset/subject/interpolation/delay policy/cue scales/head tracking; sound speed, distance and absorption, reflections, reverb, occlusion.
- **Structure:** bands/crossovers, per-band or per-partial paths, anchor level/bandwidth, source clocks/relations/formations/density.
- **Timeline/feedback/output:** stages, curves, loop/markers/seeds; pose/EEG-derived scalar/heart/respiration/MIDI/OSC mappings and failsafes; rate, depth, format, limiter, dither, loudness, and stems.

Every control declares units, bounds, default, automation capability, smoothing policy, and playback-change safety.

## 4. Architecture and conventions

Rules:

1. DSP and project models never import PyQt5.
2. GUI edits a serializable model through commands, never renderer state.
3. Preview and offline render share DSP primitives and use immutable snapshots.
4. All time-varying behavior uses a common control interface.
5. Declare and test coordinate, phase, delay, loudness, and rate conventions once.
6. HRTFs are immutable inputs; derived data records provenance.
7. Randomness uses explicit seeds and is block-size invariant.
8. Queue GUI changes for block-boundary application; prioritize real-time safety.
9. Separate experimental labels/claims from acoustic implementation.

Dependency direction is presentation → application/jobs → serializable model → DSP/spatial engines → HRTF and I/O. Analysis consumes DSP. The domain must remain usable through a headless CLI and tests.

Target package areas are `domain`, `dsp`, `render`, `hrtf`, `analysis`, `io`, `gui` (models/widgets/controllers), and `workers`, with unit, integration, GUI, and audio-regression tests plus examples, presets, and schemas.

Use Python 3.11+, PyQt5, NumPy, SciPy, soundfile, sounddevice, sofar, pyfar, Pydantic v2 (preferred) or frozen dataclasses/JSON Schema, pytest, hypothesis, and pytest-qt. Optional dependencies include pyqtgraph/vispy, numba only for measured hot loops, MIDI/OSC packages, and pyloudnorm. Projects must still open without optional packages.

Internal coordinates are right-handed: `+x` forward, `+y` left, `+z` up; azimuth 0° front and positive left; elevation positive up; metres. Left ear is positive `y`, right ear negative. Convert explicitly at SOFA boundaries and test cardinal directions.

Use float64 for phase/delay and sensitive analysis, float32 buffers where beneficial. Default rate is 44,100 Hz (support 48/96 kHz and validated alternatives), preview block 512 (128–2048), offline block 4096/8192. Domain time is seconds; render loops use sample indices. Internal phase is radians and gain is linear; UI may show degrees/dB. Assume headphones. Headphone compensation is optional, explicit, and recorded in manifests.

## 5. Data and control model

Use versioned models, stable UUIDs, and tagged unions. A project includes schema version, IDs/timestamps, global audio, listener, environment, sources, stages, automation, buses, output, optional experiment, and metadata. Defaults: 44.1 kHz, preview 512, offline 4096, master -6 dB, limiter enabled at -1 dBFS, sound speed 343 m/s, seed 0.

Each source has stable ID/name/enabled state/start/duration, a signal, amplitude control, trajectory, spatializer, modulators, routing, and tags. Signals include sine, harmonic bank, band-limited wave, noise, audio file, multiband, and composite variants.

All automatable scalars use a tagged `ControlSpec`: constant, keyframes, LFO, step sequence, random walk, restricted expression, external, sum, product, and range mapping. Controls include ID, unit, min/max, smoothing, optional seed, enabled state, and rate domain (audio, interpolated control/block, or event). Expressions use a restricted AST—never `eval`—with documented math, named parameters, time/beat, and seeded random primitives; detect dependency cycles.

Compiled controls implement reset and block render by absolute sample. Apply linear-ramp or one-pole smoothing after composition and before DSP. Intentional control jumps may bypass smoothing, but audio still crossfades unless click generation is explicitly enabled for research. Promote controls to audio rate for oscillator phase or fractional delay when necessary.

Trajectory serializes geometry, traversal, transform, and listener/world frame separately. Spatializer is a tagged abstract/geometric/HRTF/hybrid union. Automation binds stable object IDs plus schema parameter paths, never names or list indices.

Serialize JSON with a schema version. Use relative packed-asset references and hashes, preserve unknown metadata where possible, report all validation errors together, use explicit migrations without reinterpretation, and save atomically through a sibling temporary file and rename.

## 6. DSP renderers

All renderers prepare/reset/process mono to `(2, frames)`, declare latency, preserve block state, and expose diagnostics. Accumulate varying frequency as `phase[n+1] = wrap(phase[n] + 2*pi*frequency[n]/sample_rate)` and wrap periodically for precision.

The abstract renderer supports exact symmetric SAM, asymmetry, multiple modulators, discontinuous spatial phase, binaural-beat comparison, and static diotic comparison. Warn when peak sinusoidal PM deviation `abs(beta*fm)` approaches Nyquist or a configured range.

The geometric renderer computes each ear distance, delay `tau_e = r_e/c`, and delayed/gained signal. Include distance law, head transforms, preview cubic/Farrow fractional delay, higher-quality offline alternatives, optional natural Doppler or bypass, head shadow, early reflections, and reverb send. Never add geometric ITD to an HRTF containing ITD unless hybrid mode explicitly requests it.

For HRTF rendering: load and verify SOFA once; convert coordinates; explicitly bake or preserve `Data_Delay`; resample once; decompose common/differential delay, minimum phase, log magnitude, and optional directional residual; build spatial lookup; evaluate direction at control rate; interpolate aligned delay/log magnitude rather than unaligned HRIR samples; reconstruct/select filters; and crossfade/partition convolution without resetting signal history. MVP is nearest neighbor plus equal-power crossfade; production uses validated spherical/aligned interpolation.

Hybrid mode applies physical HRTF first, then explicit creative cue transforms: differential delay/IPD, ILD/pinna scales, coherence/decorrelation, width, warped paths, and per-band/partial offsets. Keep physical direction/HRTF, creative transform, and output limiter distinct.

Provide an optional broadband spatial anchor because low carriers poorly express pinna cues. Sources include shaped/pink noise, sparse grains, harmonic bank, or ambience. It is disabled by default, suggested near -30 dB relative, and controls band limits, coherence, path relation, fade, and masking.

Render sources to stereo stems and named buses. Use reconstructing crossovers (Linkwitz–Riley default), deterministic IDs/seeds unaffected by solo/mute, and warn/estimate cost from source/band count, HRIR length, interpolation, and rate.

Default output safety includes -6 dB master gain, click-free changes, look-ahead export limiter, lightweight preview limiter, peak/true-peak warnings, optional loudness measurement, visible preview level, and sweep warnings.

## 7. HRTF subsystem

Use an explicitly selected/licensed generic SOFA asset; never rely on hidden demo data. On import report convention/version, `Data_IR` shape, receivers/order, sample rate, `Data_Delay`, coordinates/units/coverage/radius, missing/duplicate/nonfinite directions, onset/tail/clipping/causality, and author/license/database metadata. Only fatal problems block rendering.

Delay policy is either `bake_delay_into_ir` (fractional shifts, derived delays zero) or `preserve_external_delay` (minimum-phase filters and runtime delay). Never ignore nonzero `Data_Delay`; record policy.

Modify cues by estimating delays; separating common/differential delay; extracting minimum-phase/log magnitude; separating common/differential magnitude; optionally splitting smooth response and high-frequency residual; scaling ITD, ILD, and pinna components; and reconstructing causal padded filters. Use one shared normalization only—never normalize per ear/direction.

Soft Basic/Expert scales: ITD and ILD 0.5–1.5 / 0–2.5, pinna 0.5–1.5 / 0–2, coherence 0.7–1 / 0–1, and extra differential delay ±0.25 ms / ±2 ms. Neutral values are 1 except extra delay 0. Warn about physically inconsistent combinations.

Interpolation order: nearest crossfade; aligned nearest-three spherical/barycentric; log-magnitude plus delay; optional spherical harmonics. Test exact directions, midpoints, poles, azimuth wrapping, sparse regions, and label extrapolation.

Personalization routes are: blinded generic-set localization selection; future synchronized two-ear acoustic sweep measurement with reference deconvolution/windowing/SNR and SOFA metadata; advanced scan/Mesh2HRTF import; and explicitly research-grade anthropometric morphing. Measurement must emphasize safe calibration, pose, repeatability, and quality checks.

Derived SOFA output includes modified HRIRs, source hash/metadata, parameters, app version, delay policy, shared gain, quality metrics, and derived marker. `sofar` verification is required before Save reports success.

## 8. PyQt5 GUI

Use `QMainWindow` with persistent docks: top menus/project/transport/time/preview/CPU; left scene tree; center path/timeline/mixer/HRTF Lab/experiments; right metadata-generated Basic/Advanced/Expert inspector; bottom analysis/render queue/issues/log; status rate/HRTF/renderer/latency/dirty state. Store layout in `QSettings`, outside projects.

The scene tree is a `QAbstractItemModel` over stable IDs and supports add/duplicate/rename/toggle/solo/mute/delete/reorder, drag/drop of modulators and paths, asset/validation state, compatible multi-edit, and ID-based selection preservation. Every edit is a `QUndoCommand`.

Reusable numeric controls contain label/unit, suitable spinbox and slider, reset, automation binding, evaluated value, warning badge, and copy/paste/control-learn/macro actions. Coalesce slider dragging into one undoable command.

Path views show listener axes, ears, sources/trails, control points/tangents, arrows/velocity, transforms, view presets, snap/grid/numeric units, playhead scrubbing, and optional physical versus apparent hybrid direction. Views never own canonical geometry.

Timeline tracks stages, source state/gain, automation, markers/loop, sensors, and render selection. Support zoom/pan/snap/copy/curves, hold/linear/cubic/valid exponential interpolation, folding/search/units, and control-rate warnings. Store seconds/ticks, not pixels. Expert modulation matrix cells expose depth, polarity, mapping, and enabled state with cycle checks.

HRTF Lab shows metadata/validation, spherical coverage, head-linked direction, HRIRs, magnitude/phase/group delay/ITD/ILD, original/modified overlays, cue controls, low-carrier versus anchor audition, A/B/X and bypass, and verified derive/export. Throttle all plots globally.

Background-cached analysis includes waveform/envelope, spectrum/spectrogram, instantaneous frequencies, IPD/ITD/ILD, trajectory/velocity, interpolation indices, loudness/true peak/clipping, and performance/xruns. Cache keys combine snapshot hash and analysis settings.

Presets are transparent parameter collections with a pre-apply diff, never hidden DSP modes. A/B comparisons are level- and latency-matched, safely crossfaded, and may blind labels.

## 9. Threading, jobs, and project safety

The GUI thread owns Qt and the editable facade. The audio callback owns no Qt, allocates nothing, performs no I/O/logging/lock waits, and consumes preallocated state. Workers handle HRTF preprocessing, waveform generation, analysis, and export. Jobs consume immutable compiled snapshots.

Queue edits and apply smoothed/crossfaded state at block boundaries. Compile expensive states in workers and atomically swap ready renderers. Use bounded SPSC queues and low-frequency telemetry. Offer higher-latency fallback blocks.

Use `QThreadPool`/`QRunnable` or a worker service for renders, HRTF import/derivation, analysis, thumbnails, and packing. Cancellation occurs at block boundaries; progress signals return to GUI; clearly mark and normally remove/recover partial output.

Use `QUndoStack` command merging. Autosave recovery snapshots on a timer and before jobs without treating them as explicit saves. Offer recovery only when newer. Keep regenerable HRTF caches separate from durable derived SOFA assets.

## 10. Validation, export, evidence, and tests

Before preview/render validate finite/ranged values, Nyquist/rate, assets/hashes, HRTF receivers/coverage/delay policy, automation cycles, durations, real-time support, peak/CPU/memory estimates, and experiment completeness. Return structured severity, object ID, parameter path, message, and suggested fix.

Exports may contain WAV/FLAC, stems, project snapshot, manifest, cue data, and reports. Manifests record app/schema/time/project/condition IDs, hashes/licenses, audio format, renderer/delay/interpolation, seeds, processing/EQ, peak/true peak/loudness/clipping, elapsed time, and determinism. Fixed snapshots/assets/version/seeds must be block-size stable within documented tolerance and preferably bit-identical on the same locked platform.

Evidence mode uses hidden labels, randomized/counterbalanced conditions, exact capture, matched levels, ratings/notes/tasks, privacy-preserving export, and neutral controls including silence, diotic/static, conventional beat, SAM, physical HRTF, hybrid, and anchor. Never encode speculative mechanisms as facts or medical preset/tooltips. Sensor data stays separate from conclusions and requires dropout/bounds/smoothing/fallback.

Tests must cover phase continuity/accuracy/reference SAM, frequency bounds, block-invariant controls, path math/transforms/jumps, coordinates/SOFA boundaries, fractional delay/distance, HRTF decomposition/scaling/normalization/round trips, schemas/migrations, finite properties, semantic save/load, all render modes, click-free transitions, moving directions, HRTF crossfades/resampling/delay policies, stems, cancellation/recovery, missing assets, audio metrics and clicks/latency, GUI project/undo/bindings/timeline/jobs/thread affinity/settings, and performance matrices for sources/HRIR lengths/interpolation/anchors/bands/block sizes.

## 11. Roadmap and working order

Implement in phases: repository/conventions/schema/CLI; exact SAM/control engine/export; PyQt5 shell/preview; geometry and geometric rendering; SOFA/HRTF import and dynamic render; modification/hybrid/anchor; timeline/multiband/multi-source; personalization/experiments; optimization/accessibility/docs/packaging.

Dependency order for tickets:

1. Conventions and typed schema.
2. `ControlSpec` and block-invariant phase.
3. Headless exact SAM and tests.
4. Export, manifest, validation, CLI.
5. PyQt5 shell and commands.
6. Safe preview boundary.
7. Geometry/path domain and editor.
8. Geometric renderer.
9. SOFA validation/preprocessing.
10. Nearest/crossfaded HRTF.
11. Aligned interpolation and HRTF Lab.
12. Cue modification and derived SOFA.
13. Hybrid and anchor.
14. Timeline, multiband, multi-source.
15. Experiments and personalization.
16. Profile, optimize, package, document.

Each ticket states API/domain changes, schema impact, unit/integration tests, thread-safety, GUI behavior, compatibility/migration, and acceptance evidence.

Initial defaults: Python 3.11+, PyQt5, 44.1 kHz, preview 512, offline 4096, stereo float32 internal and PCM-24/float32 WAV export, `SimpleFreeFieldHRIR`, nearest equal-power HRTF MVP then aligned delay/log magnitude, 1–8 optimized sources, listener-relative paths, cue scales 1, creative cues and anchor disabled, master -6 dB, export limiter -1 dBFS, sensors disabled.

## 12. Acceptance criteria and risks

Completion requires reference-exact block-continuous SAM; four click-safe modes; common automation; separated 3D path concepts; rigorous SOFA validation; stateful HRTF transitions; neutral HRTF round-trip; isolated cue controls; verified/provenanced SOFA derivation; matched anchor tests; nonblocking GUI; tested undo/recovery/migrations; deterministic export; reconstructable manifests; neutral scientific language; and hearing-safety controls.

Mitigate raw-HRIR interpolation artifacts with alignment/log magnitude/stateful crossfade; double ITD through renderer ownership; ILD loss through shared gain; ignored delays through mandatory policy; weak low-frequency pinna cues through optional anchor; complexity through progressive disclosure; unstable audio through immutable snapshots; random block dependence through absolute time/seeds; license uncertainty through provenance; overstated science through neutral controls; sweep discomfort through calibration/limiting/warnings/stop; CPU growth through estimates/caches/quality/offline modes; and schema changes through explicit migrations.

## 13. Instructions for implementing agents

- Start with headless domain/DSP; do not hide behavior in Qt widgets.
- Preserve exact SAM as a permanent reference condition.
- Treat parameters as time-varying unless documented otherwise.
- Use `apply_patch`, preserve unrelated changes, and add tests with behavior.
- Prefer small pure functions for phase, delay, coordinates, decomposition, and interpolation.
- Put units in API names (`_hz`, `_s`, `_ms`, `_db`, `_m`, `_deg`).
- Record latency and align comparisons/stems.
- Never use `eval` or execute project-supplied code.
- Never touch Qt from callbacks or workers.
- Never silently alter HRTF cues; create explicit derived assets.
- Document perceptual/heuristic assumptions and expose diagnostics.
- Add deterministic CLI examples for every major renderer.
- Profile before optimizing and retain correctness tests.
- Describe focus, sleep, meditation, or consciousness presets as subjective intent, never assured outcomes.

## 14. References

- [Patent US20130010967A1](https://patents.google.com/patent/US20130010967A1/en)
- [SOFA convention overview](https://www.sofaconventions.org/mediawiki/index.php/SOFA_%28Spatially_Oriented_Format_for_Acoustics%29)
- [SOFA HRTF databases](https://sofacoustics.org/data/database/)
- [LISTEN HRTF database](https://sofacoustics.org/data/database/listen%20%28hrtf%29/)
- [sofar documentation](https://sofar.readthedocs.io/en/v1.1.3/working_with_sofa_files.html)
- [pyfar DSP documentation](https://pyfar.readthedocs.io/en/latest/modules/pyfar.dsp.html)
- [Mesh2HRTF](https://github.com/Any2HRTF/Mesh2HRTF)

The two source PDFs remain the primary project-specific sources. External resources define formats, APIs, and techniques rather than validating speculative consciousness claims.

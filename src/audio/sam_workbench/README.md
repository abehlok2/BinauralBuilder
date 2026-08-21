# `src.audio.sam_workbench` — canonical SAM/HRTF implementation

This package is **the** implementation of SAM synthesis for BinauralBuilder.
Everything else that exposes SAM behaviour is a compatibility surface that must
delegate here rather than keep its own algorithm.

It is deliberately headless:

* no `PyQt5`, no `slab`, no Rust realtime backend, no audio device;
* no imports of `main.py`, `sound_creator`, `audio_engine`, or `src.ui`;
* no dependency on bundled third-party demo assets (`slab.HRTF.kemar()` and
  friends are legacy-only).

`tests/sam_workbench/test_clean_import.py` enforces both the static import rule
and a clean-environment import in a subprocess with those modules blocked.

## Contents

| Module | Responsibility | Phase |
| --- | --- | --- |
| `conventions.py` | Coordinate frame, audio/time rules, array layout, level conversions | 0 |
| `version.py` | Package version and project schema version (`1.0`) | 0 |
| `validation.py` | `ValidationIssue`, `ValidationCollector`, `ProjectValidationError` | 0 |
| `model.py` | Versioned `Project` document, safe defaults, aggregate validation, atomic JSON persistence | 0 |
| `migrations.py` | Explicit, ordered schema migrations | 0 |
| `cli.py` | Headless `new` / `validate` / `render` shell (`python -m src.audio.sam_workbench`) | 0-1 |
| `waveforms.py` | Dependency-free periodic shapes shared by controls and oscillators | 1 |
| `controls.py` | Declarative controls that render themselves, block-size invariantly | 1 |
| `dsp/` | Phase accumulation, oscillators, modulators, envelopes, blocks, mixing, limiting, and the SAM equations | 1 |
| `render/` | The `SpatialRenderer` contract, the abstract PM engine and its presets, and the scene mixer | 1 |
| `export.py` | WAV export plus the reconstruction manifest | 1 |
| `compat.py` | The BinauralBuilder voice adapter both public trees delegate to | 1 |
| `trajectory/legacy_paths.py` | Behaviour-preserving port of the legacy SAM2 path evaluation | 1 |
| `parameters.py` | The one SAM parameter registry: names, aliases, units, bounds, defaults, tooltips, disclosure mode | 2 |
| `analysis/` | Waveform, spectrum, instantaneous frequency, IPD/ILD/ITD measured from rendered audio | 2 |
| `preview.py` | Preview rendering and 16-bit PCM conversion for the existing QtMultimedia path | 2 |
| `trajectory/` | Geometry, traversal, transforms, and serialization for canonical paths | 3 |
| `render/geometric.py` | Geometric binaural rendering with common/differential delay | 3 |
| `hrtf/` | SOFA ingest, validation, coordinates, direction lookup, interpolation, decomposition, headphone correction, asset discovery, local storage, subject testing | 4 |
| `render/hrtf.py` | Explicit-SOFA rendering, direction chosen per block with a crossfaded filter switch | 4 |
| `hrtf/modification.py` | Cue transforms: ITD, ILD, pinna, distance, coherence, extra differential delay | 5 |
| `hrtf/derived.py` | Writing a modified HRTF as SOFA with provenance, and verifying it | 5 |
| `render/hybrid.py` | Physical HRTF, then creative cue transform, then output - kept separate | 5 |
| `render/anchor.py` | The optional broadband spatial anchor | 5 |
| `analysis/hrtf_curves.py` | Impulse, magnitude, phase, group delay, ITD and ILD curves for the HRTF Lab | 5 |
| `stages.py` | Named stages over the existing step timeline, with bindings by stable id | 6 |
| `modulation.py` | Modulation matrix with cycle detection, and parameter search | 6 |
| `cost.py` | Render-cost estimate and the real-time or offline decision | 6 |
| `dsp/crossover.py` | Linkwitz-Riley band splitting that reconstructs | 6 |
| `render/routing.py` | Buses, stems, per-band routing, deterministic seeds | 6 |
| `trajectory/coupling.py` | How one source's path follows another's | 6 |
| `experiment/localization.py` | Blinded localization test for choosing among generic HRTFs | 7 |
| `experiment/session.py` | Condition builder, randomisation, rating capture, export with provenance | 7 |
| `hrtf/measurement.py` | Swept-sine measurement import, behind an advanced flag | 7 |
| `hrtf/mesh2hrtf.py` | Import guidance and validation for simulated HRTFs | 7 |

Phase 4 adds explicit-SOFA loading, validation, coordinate conversion,
resampling, delay policies, caching, nearest/crossfaded rendering, and aligned
log-magnitude/delay interpolation under `hrtf/` and `render/hrtf.py`. Install
the standard `requirements.txt` for SOFA loading. The optional
`requirements-hrtf.txt` adds standards verification and advanced HRTF tooling;
non-HRTF modes still do not import those optional packages.
Phase 5 adds cue modification on top of that. A measured HRTF can have its
interaural and spectral cues scaled - ITD, ILD, pinna residual, distance,
coherence, and an extra differential delay - through the decomposition in
section 11.4 of `AGENTS.md`: common and differential delay, common and
differential log magnitude, and the smooth spectral shape separated from the
high-frequency directional residual. Every control is neutral at its default,
so a transform left alone returns the dataset untouched, and normalization is
one gain shared by every direction and both ears. Never per ear, and never per
direction: those are the relationships the controls exist to adjust.

A modified dataset can be written back out as a derived SOFA file carrying its
source hash, cue parameters, delay policy, shared gain, quality metrics and an
explicit derived-data marker; the write is only reported as successful once
`sofar` verification passes. The hybrid renderer keeps the physical stage, the
creative stage and the output stage apart, so a comparison between them differs
in exactly one thing, and the optional broadband anchor gives a low-frequency
carrier the high-frequency energy it needs to be localised at all. The anchor
is off by default at -30 dB and is never enabled silently.

Phase 6 covers scenes with more than one source. Stages are named spans over
the existing steps rather than a second scheduler - the specification is
explicit that a competing lane system added too early would fight the step
model it duplicates - and automation binds to a stable object identifier plus a
parameter path, never to a display name or a list index. Stage weights, coupled
paths and resolved bindings are all pure functions of absolute time, so a
staged multi-source render can be chunked and still match a whole one.

Sources split into bands through Linkwitz-Riley crossovers that reconstruct
flat, mix through named buses keeping every stem, and may follow one another -
shared, offset, mirrored, orbiting, repelled, attracted or phase-locked. Seeds
come from source identity rather than list position, so muting one source
cannot shift another's random stream. A modulation matrix connects modulators
to parameters and refuses any route that would close a loop. Because bands
multiply by sources, `cost.py` states what a scene will cost before it is paid
and routes an expensive one to an offline render.

Phase 7 is about choosing an HRTF for a particular listener, and recording how
that choice was made. The cheapest route comes first: a blinded, seeded
localization test across candidate sets, scored on angular error and on
front/back reversals rather than on preference, because the set someone likes
on first listen is not reliably the one they localise best with. Around it sits
the machinery of any listening comparison - conditions, randomised order,
captured responses - and an export that carries the full acoustic description
of every condition, so a result can be read later by someone who was not there.

The measurement and simulation routes are deliberately harder to reach.
`hrtf/measurement.py` implements the swept-sine chain - deconvolution,
reference division, windowing before the first reflection - but is off unless
`SAM_WORKBENCH_ADVANCED` is set, and refuses any measurement that fails its
signal-to-noise, clipping, repeatability or onset checks rather than returning
it with a warning. `hrtf/mesh2hrtf.py` runs no solver; it validates what one
produced, against the setup errors this route reliably makes - a mesh scaled in
millimetres, swapped ears, rotated axes, a grid with no elevation.

SOFA assets may be absolute, project-relative, or resolved through
`SAM_WORKBENCH_HRTF_DIR`; their hashes are checked when supplied. Static HRTF
voice chunks reconstruct state from the absolute voice origin so Python export
does not reset FIR history or an in-progress filter crossfade.
Later phases add cue modification and hybrid rendering, as
described in `AGENTS.md`.

The dependency graph is one-directional and enforced by the tests:

```text
conventions -> waveforms -> controls -> dsp -> render -> export -> cli
                                          \-> compat (with trajectory/legacy_paths)
                                          \-> analysis, preview, parameters
```

The GUI sits above all of it and is the only Qt-aware layer:

```text
src/ui/voice_editor_dialog.py  (existing editor; keeps the entry point)
  -> src/ui/sam_workbench_dialog.py   (QDialog over a copied voice)
       -> src/ui/sam_basic_panel.py     (controls generated from parameters.py)
       -> src/ui/sam_analysis_panel.py  (plots computed by analysis/)
```

## The SAM equations

```text
s_L(t) = a_L(t) * g[ theta_c(t) + sum_k beta_{L,k}(t) q_k(psi_k(t)) + phi_L(t) ]
s_R(t) = a_R(t) * g[ theta_c(t) - sum_k beta_{R,k}(t) q_k(psi_k(t)) + phi_R(t) ]
```

They exist once, in `dsp/source.py`. `theta_c` is *accumulated* phase, never
`2*pi*f(t)*t`, so a moving frequency stays correct. The abstract renderer, the
scene, the export, and the legacy adapter all go through that one function;
the adapter supplies its path-derived interaural phase and its ramped carrier
as providers rather than as a second copy of the equations.

Presets required of the engine - exact symmetric, asymmetric depth/rate,
multi-modulator, discontinuous, binaural-beat comparison, and static diotic -
live in `render/abstract_pm.py` under `PRESET_BUILDERS`.

## Conventions in one screen

* Right-handed listener frame: `+x` forward, `+y` left, `+z` up; azimuth `0°`
  front and positive toward the left; elevation positive up; metres.
* Left receiver at `+y`, right receiver at `-y`.
* `float64` for phase/delay accumulation, `float32` for bulk buffers.
* Seconds at the domain boundary, absolute integer sample indices inside render
  loops (`conventions.seconds_to_samples` rounds half away from zero).
* Radians internally, degrees only at user-facing boundaries; linear gain
  internally, decibels at user-facing boundaries.
* The core is channel-major `(channels, frames)`. The BinauralBuilder adapter is
  the single place that converts to legacy frame-major `(frames, 2)`.
* One shared gain for both ears — no component normalizes a single channel.

## Legacy public SAM entry points that must delegate here

These are the surfaces that currently implement or expose SAM behaviour. They
keep their public names and their serialized parameter shapes; their internals
migrate to this package in the phase noted.

| Entry point | Current behaviour | Required treatment |
| --- | --- | --- |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_sam2` | **Delegates** to `compat.render_sam2_voice`; legacy polarity preserved, `initial_offset` honoured as absolute time | Done (Phase 1) |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_sam2_transition` | **Delegates**; phase integrated from the transition origin, ramp anchored to step time | Done (Phase 1) |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation` / `..._transition` | Legacy `slab` HRTF SAM via `audio_engine` | Keep selecting the legacy renderer until the explicit-SOFA path is validated (Phase 4) |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_monaural_beat` / `..._transition` | SAM plus monaural beat | Delegate the SAM stage once the core covers it (Phase 1) |
| `binauralbuilder_core.synth_functions.spatial_angle_modulation.*` | **Delegates** to the same implementation; its `peakPhaseDev`/`phaseOffsetL`/`phaseOffsetR` are translated onto canonical fields | Done (Phase 1) — its previous independent algorithm is gone |
| `src.synth_functions.audio_engine.SAMVoice` | `slab.HRTF.kemar()` framewise nearest-HRTF overlap-add | Frozen legacy; label as legacy in the GUI, do not extend (Phase 4) |
| `binauralbuilder_core.synth_functions.audio_engine` | Duplicate of the above | Frozen legacy; no new DSP |
| `src.synth_functions.sound_creator.generate_voice_audio` | Chunking, `initial_offset=chunk_start_time`, `(frames, 2)` mixing | Unchanged contract; the adapter converts to absolute `start_sample` and channel-major audio (Phase 1) |
| `binauralbuilder_core.session` / `binauralbuilder_core.assembly` | Session-to-track conversion and Python export | Public API preserved; their SAM voices reach the core through delegation (Phase 1) |
| `src.ui.voice_editor_dialog` SAM/SAM2 parameter tables | Duplicated parameter/default metadata | Consolidate onto `compat.SAM2_PARAMETER_DEFAULTS`, the authoritative registry (Phase 2) |
| `src.ui.custom_path_creator_dialog`, `src.ui.spatial_trajectory_dialog` | GUI pixel/segment paths | Translate through the canonical trajectory types; legacy profiles round-trip unchanged (Phase 3) |

The behaviour of the two SAM2 trees before delegation is captured in
`tests/sam_workbench/fixtures/legacy_sam2_reference.npz`. Measured against it,
Phase 1 kept static `src` voices bit-identical and changed two things on
purpose:

1. transition voices integrate phase from the transition's own origin and treat
   `initial_offset` as time rather than as a phase angle, which is what makes a
   chunked render match a whole-step render;
2. `binauralbuilder_core` voices now render the canonical algorithm instead of
   that tree's own variant.

## Command-line shell

```console
python -m src.audio.sam_workbench new session.sam.json --name "Reference SAM" --with-source
python -m src.audio.sam_workbench validate session.sam.json
python -m src.audio.sam_workbench render session.sam.json render.wav --duration 30
```

`render` writes the audio and a `*.manifest.json` beside it holding everything
needed to reconstruct the render: schema and package versions, a hash of the
project document, sample rate and bit depth, renderer and its policies, seeds,
the shared master gain and limiter report, and the measured levels.

`validate` reports every problem at once, each tagged with a stable field path
such as `sources[1].amplitude_linear`, and exits non-zero on failure.

## The SAM parameter registry

`parameters.py` is the single declaration of every SAM field: serialized
camelCase name, legacy aliases, unit, bounds, decimals, default, tooltip,
disclosure mode (basic / advanced / expert), transition policy, and whether the
value is automatable or safe to change during playback.

Before it, the SAM/SAM2 parameter tables were declared *twice* in one dict
literal inside `voice_editor_dialog.get_default_params_for_function` - the
later copy silently shadowing the earlier - with tooltips and path-shape lists
in three more places. `sam2_parameter_defaults()` reproduces the old table
exactly, so an existing voice opens and saves with precisely the keys it had;
extended fields (`rotationDirection`, `discontinuousSteps`, `earPolarity`, the
per-ear phase offsets) are opt-in and are only written when a user edits them
in the workbench.

`validate_sam2_params()` returns structured issues whose `path` is the
serialized parameter name, which is what lets the GUI badge the field that
produced each message.

## Development assets

No asset is bundled. See `HRTF_ASSETS.md` for the generic SOFA set chosen for
development and for the synthetic fixtures generated by the test suite.

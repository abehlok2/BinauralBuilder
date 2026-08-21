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

Later phases add the rest of `trajectory/`, plus `hrtf/` and `analysis/`, as
described in `AGENTS.md`.

The dependency graph is one-directional and enforced by the tests:

```text
conventions -> waveforms -> controls -> dsp -> render -> export -> cli
                                          \-> compat (with trajectory/legacy_paths)
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

## Development assets

No asset is bundled. See `HRTF_ASSETS.md` for the generic SOFA set chosen for
development and for the synthetic fixtures generated by the test suite.

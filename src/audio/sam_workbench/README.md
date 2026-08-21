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

## Phase 0 contents

| Module | Responsibility |
| --- | --- |
| `conventions.py` | Coordinate frame, audio/time rules, array layout, level conversions |
| `version.py` | Package version and project schema version (`1.0`) |
| `validation.py` | `ValidationIssue`, `ValidationCollector`, `ProjectValidationError` |
| `model.py` | Versioned `Project` document, safe defaults, aggregate validation, atomic JSON persistence |
| `migrations.py` | Explicit, ordered schema migrations |
| `cli.py` | Headless `new` / `validate` shell (`python -m src.audio.sam_workbench`) |

Later phases add `dsp/`, `trajectory/`, `render/`, `hrtf/`, and `analysis/` as
described in `AGENTS.md`.

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
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_sam2` | Opposed-phase SAM2, left-minus/right-plus polarity, ignores `initial_offset` | Delegate to the core (Phase 1) behind a legacy-orientation flag |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_sam2_transition` | Linear parameter ramps with re-accumulated phase | Delegate; transition fields compile into control keyframes (Phase 1) |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation` / `..._transition` | Legacy `slab` HRTF SAM via `audio_engine` | Keep selecting the legacy renderer until the explicit-SOFA path is validated (Phase 4) |
| `src.synth_functions.spatial_angle_modulation.spatial_angle_modulation_monaural_beat` / `..._transition` | SAM plus monaural beat | Delegate the SAM stage once the core covers it (Phase 1) |
| `binauralbuilder_core.synth_functions.spatial_angle_modulation.*` | A second, mathematically different SAM2 copy | Compatibility delegation only — never an independent algorithm (Phase 1) |
| `src.synth_functions.audio_engine.SAMVoice` | `slab.HRTF.kemar()` framewise nearest-HRTF overlap-add | Frozen legacy; label as legacy in the GUI, do not extend (Phase 4) |
| `binauralbuilder_core.synth_functions.audio_engine` | Duplicate of the above | Frozen legacy; no new DSP |
| `src.synth_functions.sound_creator.generate_voice_audio` | Chunking, `initial_offset=chunk_start_time`, `(frames, 2)` mixing | Unchanged contract; the adapter converts to absolute `start_sample` and channel-major audio (Phase 1) |
| `binauralbuilder_core.session` / `binauralbuilder_core.assembly` | Session-to-track conversion and Python export | Public API preserved; their SAM voices reach the core through delegation (Phase 1) |
| `src.ui.voice_editor_dialog` SAM/SAM2 parameter tables | Duplicated parameter/default metadata | Consolidate into one registry before extending (Phase 2) |
| `src.ui.custom_path_creator_dialog`, `src.ui.spatial_trajectory_dialog` | GUI pixel/segment paths | Translate through the canonical trajectory types; legacy profiles round-trip unchanged (Phase 3) |

The behaviour of the two SAM2 trees as of Phase 0 is captured in
`tests/sam_workbench/fixtures/legacy_sam2_reference.npz`, so the delegation work
can prove what it preserves and what it intentionally changes. The capture shows
that the two trees **do not** agree today.

## Command-line shell

```console
python -m src.audio.sam_workbench new session.sam.json --name "Reference SAM"
python -m src.audio.sam_workbench validate session.sam.json
```

`validate` reports every problem at once, each tagged with a stable field path
such as `sources[1].amplitude_linear`, and exits non-zero on failure.

## Development assets

No asset is bundled. See `HRTF_ASSETS.md` for the generic SOFA set chosen for
development and for the synthetic fixtures generated by the test suite.

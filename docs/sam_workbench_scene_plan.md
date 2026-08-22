# The compiled scene plan

This note describes how a project becomes audio, and which module owns which
part of that. It exists because the same question — "what is actually being
rendered?" — used to have several answers.

## Three things, one of which is authoritative

| | What it is | Who edits it | Who saves it |
|---|---|---|---|
| `track_data["sam_scene"]` | The **persisted** scene | The GUI panels | The track file |
| `model.Project` | The **standalone** SAM API | Command line, tests | Its own document |
| `plan.CompiledScenePlan` | The **compiled** scene | Nobody | Nobody |

The track dictionary is the document a user's file contains. `Project` is a
different front door for the same ideas, used where there is no BinauralBuilder
track. The plan is derived from either, is immutable, and can always be thrown
away and recompiled.

`plan_from_track` and `plan_from_project` are the two adapters, and they
produce the same type. Nothing downstream needs to know which door a project
came through.

## What a plan contains

Everything a render needs, in absolute samples:

- the project sample rate and the absolute render range;
- stable source identifiers, and each source's start and end sample;
- the generator's parameters, less anything that is automated;
- the renderer mode and its **validated** configuration, from the registry;
- the source's `PathModel` trajectory, if it has one;
- the listener transform;
- SOFA and correction assets with their content hashes;
- the stage timeline, modulators, routes, buses, source routing, band routing;
- compiled automation controls and the scene gain envelope;
- per-source seeds derived from the project seed and the source identifier;
- latency and tail requirements;
- diagnostics and validation warnings.

A trajectory is carried as its `PathModel`, never as a materialized
`(frames, 3)` array. A path is a function of time; sampling it at the audio
rate would cost megabytes per minute per source to answer a question every
renderer prefers to ask at its own control rate.

## Identity

A source's identifier lives on the voice, in the track, as `sam_source_id`. It
is never derived from list position, because reordering steps or voices would
then reassign every automation route and every mute to a different source.

`assign_source_ids` writes identifiers into the real track. It used to be
called on a render-time deep copy, so what it assigned was discarded with the
copy and every render invented new ones.

Compiling a plan fills in a missing identifier and changes nothing else. That
is the one documented exception to "a plan is derived": inventing identifiers
privately per compile is the defect stable identifiers exist to prevent.

## Automation

Automation is compiled into functions of the absolute sample index, not
resolved per block.

The set of automated parameters is determined from the scene's **structure** —
its stage bindings and active routes — rather than by sampling it at one
instant. A stage that has not begun yet automates its parameter just as much as
one that has.

Parameters then divide by how the renderer uses them:

- **Shape parameters** (arc width, direction offset, spatial scale) are read
  once per sample. Handing the renderer an array instead of a number suffices.
- **Frequency parameters** (modulation rate, carrier) are *integrated*. Phase
  at a sample is the integral of frequency up to that sample, so it cannot be
  recovered from the frequency at that sample alone. `automation.AutomatedPhase`
  accumulates from sample zero as a pure function of the absolute index.
- **Anything else** resolves at the source's own origin rather than at the
  chunk's, which is block-invariant by construction.

Automating a parameter the registry marks non-automatable produces a warning
rather than being silently ignored.

### Determinism

| Case | Guarantee |
|---|---|
| No scene, or a scene automating nothing | **Bit-exact** across any partition |
| Scene automating gain only | **Bit-exact** |
| Scene automating a frequency | Agrees to ~3e-8, the float32 output quantization floor |

The last row is a documented tolerance rather than bit equality: summing the
same increments in a different order rounds differently. It is far below
anything audible, and matches the determinism rule for varying block sizes.

## The renderer registry

`render/registry.py` is the single answer to what a renderer is: identifier and
version, configuration schema with per-field validation, required assets and
their hash keys, a compiler, a lazily imported factory, latency and tail
calculation, a cost weight, migration hooks, and capability metadata.

Capability metadata is where the honesty lives. Only `hrtf` and `hybrid` claim
physical elevation; `abstract_pm` carries a note saying it is not a
spatializer. A GUI reading the registry cannot present a creative mapping as a
measurement by omission.

`hybrid` is fully defined but marked `voice_renderable=False`, because the
per-voice compatibility adapter cannot drive it yet. That is recorded on the
definition rather than left as a gap between a menu that offers it and a
renderer that raises.

## Timing

A render window and a source's lifetime are two independent intervals on the
same absolute timeline. `conventions.intersect_window` intersects them, and is
the only implementation — the plan and the scene renderer share it, so they
cannot drift apart about where a source sounds.

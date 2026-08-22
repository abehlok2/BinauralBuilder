# The SAM/HRTF workbench

This is a guide to what the workbench does, written to be precise about which
of its claims are acoustic measurements and which are not. That distinction is
the point of this document, so it comes first.

## What this software claims, and what it does not

**Acoustic claims.** Everything in this section is a property of the signal,
measurable from the rendered audio with the analysis views in the application
or with any other tool:

- **Spatial Angle Modulation** renders `s_L = A·sin(θc + β·q)` and
  `s_R = A·sin(θc − β·q)` exactly. The two ears receive a symmetric phase
  difference that varies at the modulation rate. This is a phase relationship
  between channels; it is not a claim about where anything is heard.
- **Interaural time difference (ITD)** and **interaural level difference
  (ILD)** are measured from the rendered signal in microseconds and decibels.
  The Analysis and HRTF Lab views plot them directly.
- **HRTF rendering** convolves the source with head-related impulse responses
  from a SOFA file you choose. The result is the measured response of the head
  the file describes, at the directions the file contains, interpolated by the
  method you select.
- **Reconstruction** properties are exact and testable: multiband splits sum to
  an allpass, block size does not change the render, and offline output is
  reproducible for a fixed project, assets and seed.
- **Peak level and limiting** are numeric guarantees about the output file.

**Not claims of this software.** The application deliberately does not assert
any of the following, and no part of its interface is worded as though it does:

- that a particular modulation rate produces a particular mental state,
  brainwave pattern, or physiological response;
- that a rendered direction will be *perceived* at that direction. Localization
  is listener-specific; the localization test exists precisely because a
  generic HRTF works better for some listeners than others, and the only way to
  find out which is to listen;
- that any setting is therapeutic, medical, or safe for a specific person.

If you are using this tool for research, the export manifest records everything
needed to reconstruct a condition, and the experiment tooling keeps comparisons
blinded. If you are using it for listening, treat what you hear as the evidence
and the numbers as a description of the signal that produced it.

## Hearing safety

- The master gain defaults to −6 dB and the export limiter to a −1 dBFS
  ceiling. Neither is a substitute for setting a comfortable listening level
  before you start.
- Preview and export report peak level; a project that would clip is reported
  rather than silently limited into something else.
- Long sessions at any level carry the usual risks of extended headphone
  listening. The application does not monitor your exposure.

## The render modes

| Mode | What it does | When it is the right one |
|---|---|---|
| Abstract phase modulation | The SAM equations, nothing else | The exact, reproducible baseline; no HRTF needed |
| Geometric binaural | Interaural delay and level from a head model | A physical placement without a measured HRTF |
| HRTF | Convolution with an explicit SOFA asset | When you want a measured head's response |
| Hybrid | HRTF for spectrum, with declared extra cues | Deliberate departures from the measured cues |

The renderer is chosen per voice and is used by both the preview and the
export, so what you hear during editing is what gets written.

## Choosing an HRTF

Start with the localization test in **HRTF Lab → Tests & import**. It presents
directions through each candidate dataset with the names hidden, asks you to
point at where you heard each one, and scores the candidates on angular error
and front/back reversals. It costs nothing but your time, needs no equipment,
and settles most of the question of which generic HRTF suits you.

Only after that is it worth considering the two advanced routes:

- **Measuring your own** needs a quiet room, a calibrated rig and patience. It
  is disabled by default; set `SAM_WORKBENCH_ADVANCED=1` to enable it. A
  measurement that fails its quality checks is refused rather than imported,
  because a file written from a bad take outlives the warning attached to it.
- **Simulating from a scan** (Mesh2HRTF) happens outside this application. The
  workbench receives the resulting SOFA and checks it for the errors this route
  actually makes: a mesh scaled in millimetres, swapped ears, an evaluation
  grid that never leaves the horizon.

## Modifying HRTF cues

The Modify tab scales the decomposed cues — ITD, ILD, pinna detail, distance —
independently. Two things are worth knowing:

- The controls act on a *decomposition*, so scaling ITD does not change the
  magnitude spectrum and scaling ILD does not change the delay, as closely as
  the decomposition permits. The plots show measured against modified so you
  can see what actually moved.
- Neutral settings are measurably equivalent to the source. If you have moved
  something and want back, **Reset to measured** returns exactly there.

A derived SOFA file records where it came from and what was done to it, and is
verified before it is reported as saved.

## Autosave and recovery

The editor writes a recovery snapshot on a timer and before a render. A
snapshot is **not** a save: it lives beside your project under its own name and
never overwrites it. When you open a project and a snapshot is newer than the
file, you are offered the choice; declining deletes the snapshot, so the prompt
keeps meaning something.

## Performance

`python -m src.audio.sam_workbench.benchmark` measures this machine against the
matrix the specification defines and reports load as a fraction of the audio
callback budget. Use it before committing to a real-time configuration; the
Routing & cost tab gives the same estimate for a scene you are editing, and can
measure this machine's throughput rather than assuming it.

Roughly, on a modern desktop: a single source with a 256-tap HRIR is a few
percent of the budget; eight sources with 2048-tap HRIRs are around thirty
percent; multiband splitting is the most expensive thing you can turn on, and
eight bands across eight sources is an offline configuration.

## Reproducing a render

Offline output is deterministic for a fixed project, asset set, package version
and seed. The export manifest records all four. If you need a render to be
reproducible by someone else, ship the manifest with it; if you need it to be
reproducible by you later, keep the SOFA assets — the manifest records their
hashes, and a different file with the same name will be detected rather than
silently used.

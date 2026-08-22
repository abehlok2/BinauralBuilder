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

## Moving a source in three dimensions

A source position is a point in metres in the listener's own frame:

```
p(t) = [x(t), y(t), z(t)]
```

The frame is right-handed: **+x** is forward, **+y** is to your left, **+z** is
up. Every renderer derives what it needs from that one point — azimuth is
`atan2(y, x)`, elevation is `atan2(z, hypot(x, y))`, and distance is the length
of the vector. Nothing stores a direction and a distance separately, which is
what stops two descriptions of the same path from drifting apart.

Open **Path & Geometry → Open 3D path designer** to edit one.

### Two ways to type a position, one way to store it

You can enter a point in metres, or in the terms spatial audio is usually
discussed in:

```json
{ "azimuthDegrees": 45.0, "elevationDegrees": 30.0, "distanceMetres": 1.5 }
```

Spherical entry is converted as you type it. Both editors in the dialog show the
same point at the same time, and what is written to the project is Cartesian.

### Keyframes

The most direct way to describe a path is to say where the source is at each
moment:

```json
{
  "coordinateSystem": "listener_relative_cartesian",
  "units": "metres",
  "interpolation": "cubic",
  "keyframes": [
    { "timeSeconds": 0.0,  "position": [0.0,  1.0, 0.0] },
    { "timeSeconds": 5.0,  "position": [1.0,  0.0, 1.5] },
    { "timeSeconds": 10.0, "position": [0.0, -1.0, 0.5] }
  ]
}
```

Interpolation may be `hold`, `linear`, `cubic` or `catmull_rom`. CSV and JSON
import land here too, in either Cartesian or spherical columns, which is how
recorded motion gets in.

### Geometry is not traversal

The dialog keeps two things apart, and it is worth keeping them apart in your
head as well:

* **Path geometry** — where the virtual source can be.
* **Path traversal** — how it moves along that shape over time.

The same circle can be walked at constant speed, accelerated, eased, reversed,
or driven from the stage timeline. Those are all traversal; none of them changes
the circle. Choosing the speed law is a real decision: *constant linear speed*
covers metres evenly, while *curve parameter speed* on a circle means constant
angular speed instead.

### Four views, because depth is ambiguous

A path cannot be edited precisely in a single perspective view — dragging a
point moves it along a ray, and where on that ray it lands is a guess. So the
designer has a perspective view for seeing the shape, and top, front and side
views that each edit exactly the two axes they show and leave the third
untouched. A point selected anywhere is selected everywhere.

The listener is drawn with a nose, ears, a head-height plane and a vertical
axis, so which way you are facing is never in doubt. The optional **HRTF
coverage shell** draws the sphere your dataset measured on.

During preview the moving marker is evaluated through the same path model the
renderer uses, so it follows the trajectory actually being sent — with easing or
reverse in play that is not the same as the drawn curve.

### Paths worth trying

Beyond flat circles and arcs, the primitive list includes vertical and tilted
orbits, helices, rising and falling arcs, overhead sweeps, front-to-back
elevation sweeps, dome traversal, three-dimensional figure-eights, pendulums,
toroidal paths around the head and a seeded random walk in a volume. For this
application the useful ones tend to be:

* a floor-to-overhead sweep (**rising arc**, low start elevation to high);
* front-centre to directly overhead (**overhead sweep**);
* a spiral rising around you (**dome traversal**);
* alternating above-left and below-right (**figure-eight** at 45° tilt);
* an expanding and contracting orbit (**spherical orbit** with two distances);
* a toroidal path around the head (**torus**);
* a slow three-dimensional figure-eight (**figure-eight**, long duration).

### What each renderer can honestly do with height

Height is the part where the renderers genuinely differ, so it is the part worth
being careful about:

* **HRTF** looks the dataset up by azimuth *and* elevation at every control
  interval, and uses distance for gain and propagation delay. This is the only
  mode that can really place a source above or below you.
* **Geometric** derives interaural time and level differences, propagation delay
  and distance attenuation from the path. Elevation can colour the sound, but
  convincing height localization needs HRTF filtering.
* **Abstract phase modulation** can use the path as a control source — azimuth
  to interaural phase, elevation to carrier or modulation depth, distance to
  amplitude. These are labelled **creative mappings, not spatialization**.
  Phase modulation alone cannot produce reliable height, and the label travels
  with the setting into the saved project so it cannot quietly become a claim.
* **Hybrid** runs `Source → SAM → 3D trajectory → HRTF interpolation → Cue
  modification → Output`. The order is stated because it matters: phase
  manipulation applied before binaural filtering and after it are meaningfully
  different renders.

### When the dataset cannot follow the path

Many published HRTF datasets are dense near the horizontal plane and thin or
empty above and below it. An overhead or below-head path can therefore ask for
directions nobody measured, and the renderer will still produce audio — it
always finds a nearest measurement — which is exactly the problem.

The workbench warns when:

* the requested elevation lies outside measured coverage;
* the path repeatedly crosses sparse measurement regions;
* the dataset has too few upper- or lower-hemisphere samples;
* nearest-neighbour fallback is in use;
* the path moves too fast for the control interval or crossfade to track.

None of these refuses a render. They tell you which parts of it are being
extrapolated rather than reproduced, which is a distinction worth having before
you draw a conclusion from what you heard.

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

# SAM workbench phases zero through two

This package is the headless foundation for the spatial-angle-modulation
workbench. Phase zero establishes:

- one documented set of coordinate, phase, level, rate, and block-size
  conventions;
- an immutable, versioned project model with stable UUIDs and tagged signal
  and spatializer records;
- aggregate validation and atomic JSON persistence; and
- a minimal CLI that future render, analysis, and GUI layers can share.

It intentionally contains no Qt or renderer state. Internal coordinates are
right-handed (`+x` forward, `+y` left, `+z` up), angles use degrees only where
named `_deg`, phase uses radians, distance uses metres, time uses seconds,
frequency uses hertz, and gains are linear unless named `_db`/`_dbfs`.

Phase one adds absolute-sample controls, a stateful exact symmetric SAM
renderer, and deterministic WAV export with a reconstruction manifest. The
reference renderer implements opposed-ear sinusoidal phase modulation and
accumulates carrier phase across blocks rather than deriving phase from block
time. Phase one deliberately accepts only the `abstract` spatializer; later
render modes must use their own explicit engines.

Phase two adds a PyQt5 workbench shell with persistent workflow docks,
undoable immutable-project edits, and a bounded preview handoff. Preview
compiles snapshots outside the callback and swaps renderers only at block
boundaries. Its callback-facing API imports no Qt, performs no I/O, and shares
the exact SAM DSP used by offline export.

Create and validate a deterministic project skeleton from the repository root:

```bash
python -m src.audio.sam_workbench new examples/reference.sam.json --name "Reference SAM"
python -m src.audio.sam_workbench validate examples/reference.sam.json
python -m src.audio.sam_workbench render examples/reference.sam.json reference.wav --duration-s 10
python -m src.audio.sam_workbench gui examples/reference.sam.json
```

Schema version 1 is decoded explicitly. Unsupported versions fail rather than
being silently reinterpreted; later versions must add named migration steps.

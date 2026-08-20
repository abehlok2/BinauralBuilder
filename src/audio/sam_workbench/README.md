# SAM workbench phase zero

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

Create and validate a deterministic project skeleton from the repository root:

```bash
python -m src.audio.sam_workbench new examples/reference.sam.json --name "Reference SAM"
python -m src.audio.sam_workbench validate examples/reference.sam.json
```

Schema version 1 is decoded explicitly. Unsupported versions fail rather than
being silently reinterpreted; later versions must add named migration steps.

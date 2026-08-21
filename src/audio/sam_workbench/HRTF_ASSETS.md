# HRTF assets for SAM workbench development

## Rule

The repository ships **no** third-party HRTF data, and no production code path
may depend on bundled demo assets. The legacy `slab.HRTF.kemar()` call in
`audio_engine` is the exact failure mode this rule exists to avoid: it loads
hidden package data implicitly and breaks when that data is unavailable.

Every new HRTF path takes an **explicit** asset path plus a content hash, with
project-relative resolution for portable presets.

## Chosen generic development set

**MIT KEMAR** (Gardner & Martin, MIT Media Lab, 1994) — the "normal pinna"
horizontal-and-elevation set, used in its SOFA form
(`MIT_KEMAR_normal_pinna.sofa`, as published in the SOFA general-purpose
database at <https://sofacoustics.org/data/database/>).

* Licence: explicitly free to use with citation. The distribution states that
  the data is Copyright 1994 by the MIT Media Laboratory and is provided free
  with no restrictions on use, provided the authors are cited in any research
  or commercial application. Keep the licence/README file that ships with the
  download next to the asset and cite Gardner and Martin wherever it is used.
* Why: an unambiguous, quotable licence; a dummy-head measurement suitable as a
  neutral reference; and it is the same dataset the legacy `slab` renderer
  loads implicitly, which makes it the natural A/B baseline when the explicit
  SOFA renderer is compared against the legacy path in Phase 4.

Optional second set, once a denser spherical coverage is needed for
interpolation work: the **SADIE II Binaural Measurement Database** (University
of York, KU100 subjects `D1`/`D2`, native SOFA,
<https://www.york.ac.uk/sadie-project/database.html>). Confirm and record its
Creative Commons terms from the licence file in the download before relying on
it, especially before publishing any derived/modified dataset.

### How developers point at it

Download the set once, outside the repository, and reference it explicitly:

* per-project: store the asset path plus content hash in the project/voice
  parameters, resolved relative to the project file when packed;
* per-machine: set `SAM_WORKBENCH_HRTF_DIR` to the directory holding the SOFA
  files for local development and manual testing.

Tests that need a real measured set must skip when the asset is absent. No test
in this repository requires a downloaded asset.

## Synthetic fixtures (in-repository)

Generated from closed-form formulas by
`tests/sam_workbench/fixtures/make_synthetic_assets.py`, so they carry no
third-party licence:

| Fixture | Contents |
| --- | --- |
| `synthetic_impulse.wav` | 64-frame stereo impulse pair with a deliberate 4-sample ITD and a level difference |
| `synthetic_hrir.npz` | Eight-direction horizontal HRIR set (48 taps/ear) in the canonical frame, readable without any binary-format dependency |
| `synthetic_hrir.sofa` | The same set as a minimal `SimpleFreeFieldHRIR` SOFA file |

The synthetic set models interaural delay and head shadow only; it is
front/back symmetric by construction and is not a substitute for measured data.

Regenerating `synthetic_hrir.sofa` needs `h5py`, which is an optional
development dependency only — nothing in the shipped application requires it,
and the tests that read the SOFA fixture skip when it is missing.

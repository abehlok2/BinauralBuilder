"""Physical HRTF rendering, then controlled cue transformation, then output.

The specification asks for a strict separation between three things, and this
module exists to keep them apart rather than to blend them:

* ``physical`` - the direction and the measured HRTF that encodes it;
* ``creative`` - the cue transform applied on top, which is deliberately not
  physical and must never be mistaken for a measurement;
* ``output`` - headphone correction and gain, the last stage before a limiter.

Keeping the boundary sharp is what makes an honest A/B possible. A listener
comparing "physical" against "hybrid" is comparing exactly one difference, and
the stems come back separately so the comparison can be level-matched instead
of argued about.

The cue transform acts on the HRIRs rather than on the rendered stereo signal.
Interaural time and level differences are properties of the filter pair for a
direction; recovering them from a mixed stereo output in order to rescale them
would mean undoing the convolution first, and doing it badly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..conventions import db_to_linear
from ..hrtf.headphones import HeadphoneCorrection, apply_correction
from ..hrtf.modification import CueTransform, transform_dataset
from .anchor import AnchorSpec, anchor_directions, make_anchor_signal
from .hrtf import SpatialHrtfSpec, render_spatial_hrtf

__all__ = ["HybridResult", "HybridSpec", "render_hybrid"]


@dataclass(frozen=True)
class HybridSpec:
    """One physical stage, one creative stage, one output stage."""

    # --- physical -----------------------------------------------------------
    interpolation: str = "nearest"
    crossfade_ms: float = 12.0

    # --- creative -----------------------------------------------------------
    #: Neutral by default, so a hybrid render with no cue settings is exactly
    #: the physical render.
    cue: CueTransform = field(default_factory=CueTransform)
    anchor: AnchorSpec = field(default_factory=AnchorSpec)

    # --- output -------------------------------------------------------------
    headphone: HeadphoneCorrection | None = None
    output_gain_db: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.output_gain_db):
            raise ValueError("output_gain_db must be finite")

    @property
    def is_physical(self) -> bool:
        """True when nothing creative is being done on top of the HRTF."""

        return self.cue.is_neutral and not self.anchor.enabled

    def as_physical(self) -> "HybridSpec":
        """The same render with the creative stage removed, for A/B."""

        return replace(self, cue=CueTransform(), anchor=replace(self.anchor, enabled=False))

    def describe(self) -> dict[str, Any]:
        return {
            "interpolation": self.interpolation,
            "crossfadeMs": float(self.crossfade_ms),
            "cue": self.cue.describe(),
            "anchor": self.anchor.describe(),
            "outputGainDb": float(self.output_gain_db),
            "physical": self.is_physical,
        }


@dataclass(frozen=True)
class HybridResult:
    """The mixed render and the stems it was mixed from."""

    audio: NDArray[np.float64]  # (2, frames)
    source_stem: NDArray[np.float64]
    anchor_stem: NDArray[np.float64]
    normalization_gain: float = 1.0
    #: True when the cue stage was neutral and the anchor was off.
    physical: bool = True

    @property
    def frames(self) -> int:
        return int(self.audio.shape[1])

    def levels(self) -> dict[str, float]:
        """Root-mean-square of the mix and of each stem, for level matching."""

        def rms(block: NDArray[np.floating]) -> float:
            values = np.asarray(block, dtype=np.float64)
            return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0

        return {
            "mix": rms(self.audio),
            "source": rms(self.source_stem),
            "anchor": rms(self.anchor_stem),
        }


def render_hybrid(
    mono,
    directions,
    dataset,
    sample_rate_hz: float,
    spec: HybridSpec | None = None,
    *,
    block_size: int = 512,
    anchor_material=None,
) -> HybridResult:
    """Render one source through the physical, creative, and output stages.

    ``directions`` is a single ``(3,)`` direction or a ``(frames, 3)`` path, as
    for :func:`~.hrtf.render_spatial_hrtf`.
    """

    spec = spec or HybridSpec()
    source = np.asarray(mono, dtype=np.float64).reshape(-1)
    rate = float(sample_rate_hz)

    # --- creative stage, applied to the filters ----------------------------
    # A neutral transform returns the dataset untouched, so a physical render
    # costs nothing extra here.
    if spec.cue.is_neutral:
        rendering_dataset: Any = dataset
        normalization_gain = 1.0
    else:
        modified = transform_dataset(dataset, spec.cue)
        rendering_dataset = _ModifiedDatasetView(dataset, modified.hrirs)
        normalization_gain = modified.normalization_gain

    # --- physical stage ----------------------------------------------------
    render_spec = SpatialHrtfSpec(
        interpolation=spec.interpolation,
        crossfade_ms=spec.crossfade_ms,
        # Headphone correction belongs to the output stage, not to this one:
        # applying it per stem would apply it twice to the mix.
        headphone=None,
    )
    source_stem = render_spatial_hrtf(
        source, directions, rendering_dataset, rate, block_size=block_size, spec=render_spec
    )

    anchor_stem = np.zeros_like(source_stem)
    if spec.anchor.enabled:
        anchor_signal = make_anchor_signal(
            spec.anchor, source.size, rate, material=anchor_material
        )
        anchor_stem = render_spatial_hrtf(
            anchor_signal,
            anchor_directions(np.asarray(directions, dtype=np.float64), spec.anchor),
            rendering_dataset,
            rate,
            block_size=block_size,
            spec=render_spec,
        )

    # --- output stage ------------------------------------------------------
    mixed = source_stem + anchor_stem
    if spec.headphone is not None and spec.headphone.is_active:
        mixed = apply_correction(mixed, spec.headphone)
    if spec.output_gain_db != 0.0:
        mixed = mixed * float(db_to_linear(spec.output_gain_db))

    return HybridResult(
        audio=mixed,
        source_stem=source_stem,
        anchor_stem=anchor_stem,
        normalization_gain=normalization_gain,
        physical=spec.is_physical,
    )


class _ModifiedDatasetView:
    """A loaded dataset with its impulse responses swapped for modified ones.

    The renderer only reads geometry, rate and responses, so presenting a view
    avoids copying the rest of a dataset - and avoids writing to a frozen one.
    """

    def __init__(self, source: Any, hrirs: NDArray[np.floating]) -> None:
        self._source = source
        self.ir = np.asarray(hrirs, dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @property
    def taps(self) -> int:
        return int(self.ir.shape[2])

    @property
    def measurements(self) -> int:
        return int(self.ir.shape[0])

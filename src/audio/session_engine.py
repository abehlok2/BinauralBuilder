"""High level helpers for assembling session audio previews and renders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from src.audio.session_model import (
    Session,
    SessionPresetChoice,
    session_to_track_data,
)
from src.synth_functions.sound_creator import (
    assemble_track_from_data,
    generate_audio,
)


@dataclass
class _AssemblerOptions:
    sample_rate: int
    crossfade_curve: str
    normalization_ceiling: float


class SessionAssembler:
    """Prepare audio data for a :class:`~src.audio.session_model.Session`.

    The assembler converts the higher level session description into the
    ``track_data`` structure consumed by :mod:`src.synth_functions.sound_creator`.
    It provides convenience helpers for rendering the session to an in-memory
    numpy array as well as exporting directly to disk via the existing audio
    generator.
    """

    def __init__(
        self,
        session: Session,
        binaural_catalog: Mapping[str, SessionPresetChoice],
        noise_catalog: Mapping[str, SessionPresetChoice],
        *,
        sample_rate: Optional[int] = None,
        crossfade_curve: Optional[str] = None,
        normalization_ceiling: float = 0.25,
    ) -> None:
        options = _AssemblerOptions(
            sample_rate=int(sample_rate or session.sample_rate),
            crossfade_curve=str(crossfade_curve or session.crossfade_curve or "linear"),
            normalization_ceiling=float(normalization_ceiling),
        )

        options.normalization_ceiling = float(
            np.clip(options.normalization_ceiling, 0.0, 0.75)
        )

        self._session = session
        self._options = options
        self._track_data = session_to_track_data(session, binaural_catalog, noise_catalog)

        global_settings = self._track_data.setdefault("global_settings", {})
        global_settings["sample_rate"] = options.sample_rate
        global_settings["crossfade_curve"] = options.crossfade_curve
        global_settings.setdefault("crossfade_duration", session.crossfade_duration)
        global_settings.setdefault("output_filename", session.output_filename)

        clips = list(self._track_data.get("clips", []))
        self._track_data["clips"] = clips
        self._track_data["overlay_clips"] = list(clips)

        self._render_cache: Optional[np.ndarray] = None

    @property
    def track_data(self) -> dict:
        """Return the assembled ``track_data`` structure."""

        return self._track_data

    @property
    def sample_rate(self) -> int:
        return self._options.sample_rate

    @property
    def normalization_target(self) -> float:
        return self._options.normalization_ceiling

    def render_to_array(self, force: bool = False) -> np.ndarray:
        """Render the session to a stereo float32 numpy array."""

        if self._render_cache is not None and not force:
            return self._render_cache

        global_settings = self._track_data.get("global_settings", {})
        crossfade_duration = float(global_settings.get("crossfade_duration", 0.0))
        crossfade_curve = str(global_settings.get("crossfade_curve", self._options.crossfade_curve))

        audio = assemble_track_from_data(
            self._track_data,
            self.sample_rate,
            crossfade_duration,
            crossfade_curve,
        )

        if audio is None or audio.size == 0:
            self._render_cache = np.zeros((0, 2), dtype=np.float32)
            return self._render_cache

        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        target = self.normalization_target
        if peak > 1e-9 and target > 0.0:
            audio = audio * (target / peak)

        self._render_cache = audio.astype(np.float32, copy=False)
        return self._render_cache

    def render_to_file(self, path: Path | str) -> bool:
        """Render the session directly to ``path`` using ``generate_audio``."""

        resolved = str(path)
        self._track_data.setdefault("global_settings", {})["output_filename"] = resolved
        self._render_cache = None
        return bool(
            generate_audio(
                self._track_data,
                output_filename=resolved,
                target_level=self.normalization_target,
            )
        )


__all__ = ["SessionAssembler"]


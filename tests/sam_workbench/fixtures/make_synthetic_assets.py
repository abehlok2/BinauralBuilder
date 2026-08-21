"""Generate the small synthetic test assets used by the SAM workbench tests.

Every asset produced here is synthesised from closed-form formulas in this
file, so the repository never ships third-party demo data and no production
code path depends on bundled assets.  Regenerate with::

    python tests/sam_workbench/fixtures/make_synthetic_assets.py

Assets
------
``synthetic_impulse.wav``
    64-frame stereo impulse pair with a deliberate interaural delay and level
    difference.  Useful for fractional-delay, ITD, and ILD checks.
``synthetic_hrir.npz``
    Eight-direction horizontal HRIR set in the canonical listener frame,
    stored without any binary-format dependency.  It models interaural delay
    and head shadow only, so it is front/back symmetric by construction and is
    not a stand-in for measured data.
``synthetic_hrir.sofa``
    The same HRIR set as a ``SimpleFreeFieldHRIR`` SOFA file.
``synthetic_sonicom_hrir.sofa``
    A denser, SONICOM-shaped set: a full azimuth circle at six elevations,
    with a nonzero ``Data_Delay`` so the delay policies have something to act
    on. Small enough to keep in the repository, structured enough to exercise
    direction lookup, interpolation, and delay handling.

Writing SOFA prefers ``sofar`` so the result is convention-valid; ``h5py`` is
used as a fallback. Reading the fixtures in tests needs one of the two, and
those tests skip when neither is installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE_HZ = 44_100
IMPULSE_FRAMES = 64
LEFT_IMPULSE_INDEX = 8
RIGHT_IMPULSE_INDEX = 12
LEFT_IMPULSE_GAIN = 1.0
RIGHT_IMPULSE_GAIN = 0.7

HRIR_TAPS = 48
#: Canonical azimuths in degrees: 0 is front and positive turns toward the left.
HRIR_AZIMUTHS_DEG = tuple(float(index * 45) for index in range(8))
HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND_M_S = 343.0
SOURCE_DISTANCE_M = 1.0


def synthetic_impulse() -> np.ndarray:
    """Return a ``(frames, 2)`` stereo impulse pair with a synthetic ITD/ILD."""

    audio = np.zeros((IMPULSE_FRAMES, 2), dtype=np.float32)
    audio[LEFT_IMPULSE_INDEX, 0] = LEFT_IMPULSE_GAIN
    audio[RIGHT_IMPULSE_INDEX, 1] = RIGHT_IMPULSE_GAIN
    return audio


def _ear_incidence_cosine(azimuth_deg: float, ear_sign: int) -> float:
    """Cosine of the angle between the source and one ear's outward direction.

    ``ear_sign`` is ``+1`` for the left ear (outward ``+y``) and ``-1`` for the
    right ear (outward ``-y``), matching the canonical listener frame.
    """

    return math.sin(math.radians(azimuth_deg)) * ear_sign


def _woodworth_delay_s(azimuth_deg: float, ear_sign: int) -> float:
    """Spherical-head arrival delay for one ear, in seconds.

    The ear facing the source hears it early (the path is shortened by up to
    the head radius); the shadowed ear hears a creeping wave around the head.
    """

    cosine = max(-1.0, min(1.0, _ear_incidence_cosine(azimuth_deg, ear_sign)))
    incidence = math.acos(cosine)
    if incidence <= math.pi / 2.0:
        extra_path = -HEAD_RADIUS_M * cosine
    else:
        extra_path = HEAD_RADIUS_M * (incidence - math.pi / 2.0)
    return (SOURCE_DISTANCE_M + extra_path) / SPEED_OF_SOUND_M_S


def synthetic_hrir_set() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions_deg, hrirs)`` for a synthetic horizontal HRIR set.

    ``positions_deg`` is ``(measurements, 3)`` as ``azimuth, elevation,
    distance``; ``hrirs`` is ``(measurements, 2, taps)`` with the left ear
    first, matching SOFA receiver ordering used at the import boundary.
    """

    positions = np.array(
        [[azimuth, 0.0, SOURCE_DISTANCE_M] for azimuth in HRIR_AZIMUTHS_DEG], dtype=np.float64
    )
    hrirs = np.zeros((len(HRIR_AZIMUTHS_DEG), 2, HRIR_TAPS), dtype=np.float64)
    taps = np.arange(HRIR_TAPS, dtype=np.float64)

    reference_delay_s = SOURCE_DISTANCE_M / SPEED_OF_SOUND_M_S
    for index, azimuth in enumerate(HRIR_AZIMUTHS_DEG):
        for ear, sign in ((0, 1), (1, -1)):
            delay_samples = (_woodworth_delay_s(azimuth, sign) - reference_delay_s) * SAMPLE_RATE_HZ + 14.0
            # A raised-cosine pulse gives a smooth, band-limited fractional delay.
            envelope = np.clip(1.0 - np.abs(taps - delay_samples) / 3.0, 0.0, None)
            pulse = 0.5 * (1.0 - np.cos(math.pi * envelope)) * np.sign(envelope)
            # Simple head-shadow gain: the contralateral ear is attenuated.
            shadow = 0.6 + 0.4 * _ear_incidence_cosine(azimuth, sign)
            hrirs[index, ear] = shadow * pulse
    return positions, hrirs


def write_impulse_wav(path: Path | None = None) -> Path:
    import soundfile as sf

    destination = Path(path) if path is not None else FIXTURE_DIR / "synthetic_impulse.wav"
    sf.write(destination, synthetic_impulse(), SAMPLE_RATE_HZ, subtype="FLOAT")
    return destination


def write_hrir_npz(path: Path | None = None) -> Path:
    destination = Path(path) if path is not None else FIXTURE_DIR / "synthetic_hrir.npz"
    positions, hrirs = synthetic_hrir_set()
    np.savez_compressed(
        destination,
        positions_deg=positions,
        hrirs=hrirs,
        sample_rate_hz=np.array(SAMPLE_RATE_HZ),
    )
    return destination


SONICOM_AZIMUTHS_DEG = tuple(float(value) for value in range(0, 360, 15))
#: The elevations the subject-selection audition grid uses.
SONICOM_ELEVATIONS_DEG = (-45.0, -30.0, 0.0, 30.0, 45.0, 60.0)
SONICOM_TAPS = 128
#: A nonzero Data_Delay, so a loader that ignores it is caught by a test.
SONICOM_DATA_DELAY_SAMPLES = (3.0, 5.0)


def sonicom_shaped_set() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions_deg, hrirs)`` for a SONICOM-shaped synthetic set.

    ``positions_deg`` is ``(measurements, 3)`` azimuth/elevation/distance and
    ``hrirs`` is ``(measurements, 2, taps)``. The responses are the same
    closed-form shadow-and-delay model as the small set, sampled over a full
    sphere-like grid so direction lookup has somewhere to go.
    """

    positions = np.array(
        [[azimuth, elevation, SOURCE_DISTANCE_M]
         for elevation in SONICOM_ELEVATIONS_DEG
         for azimuth in SONICOM_AZIMUTHS_DEG],
        dtype=np.float64,
    )
    hrirs = np.zeros((positions.shape[0], 2, SONICOM_TAPS), dtype=np.float64)
    taps = np.arange(SONICOM_TAPS, dtype=np.float64)
    reference_delay_s = SOURCE_DISTANCE_M / SPEED_OF_SOUND_M_S

    for index, (azimuth, elevation, _) in enumerate(positions):
        elevation_scale = math.cos(math.radians(elevation))
        for ear, sign in ((0, 1), (1, -1)):
            cosine = _ear_incidence_cosine(azimuth, sign) * elevation_scale
            delay_samples = (
                (_woodworth_delay_s(azimuth, sign) - reference_delay_s) * SAMPLE_RATE_HZ + 24.0
            )
            envelope = np.clip(1.0 - np.abs(taps - delay_samples) / 3.0, 0.0, None)
            pulse = 0.5 * (1.0 - np.cos(math.pi * envelope)) * np.sign(envelope)
            shadow = 0.6 + 0.4 * cosine
            # A gentle elevation-dependent notch stands in for a pinna cue.
            notch = 1.0 - 0.25 * math.cos(math.radians(2.0 * elevation))
            hrirs[index, ear] = shadow * notch * pulse
    return positions, hrirs


def _write_sofa_with_sofar(
    path: Path,
    positions_deg: np.ndarray,
    hrirs: np.ndarray,
    *,
    data_delay: np.ndarray,
    database_name: str,
    listener_short_name: str,
    comment: str,
) -> Path:
    import sofar

    sofa = sofar.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = hrirs
    sofa.Data_SamplingRate = float(SAMPLE_RATE_HZ)
    sofa.Data_Delay = data_delay
    sofa.SourcePosition = positions_deg
    sofa.SourcePosition_Type = "spherical"
    sofa.SourcePosition_Units = "degree, degree, metre"
    sofa.ReceiverPosition = np.array(
        [[0.0, HEAD_RADIUS_M, 0.0], [0.0, -HEAD_RADIUS_M, 0.0]], dtype=np.float64
    )
    sofa.GLOBAL_DatabaseName = database_name
    sofa.GLOBAL_ListenerShortName = listener_short_name
    sofa.GLOBAL_Title = "Synthetic BinauralBuilder SAM workbench test HRIR"
    sofa.GLOBAL_AuthorContact = "BinauralBuilder repository"
    sofa.GLOBAL_Organization = "BinauralBuilder"
    sofa.GLOBAL_License = "Synthetic fixture created for this repository's tests"
    sofa.GLOBAL_Comment = comment
    sofar.write_sofa(str(path), sofa)
    return path


def write_sonicom_shaped_sofa(path: Path | None = None) -> Path:
    """Write the denser SONICOM-shaped fixture, with a nonzero ``Data_Delay``."""

    destination = Path(path) if path is not None else FIXTURE_DIR / "synthetic_sonicom_hrir.sofa"
    positions, hrirs = sonicom_shaped_set()
    return _write_sofa_with_sofar(
        destination,
        positions,
        hrirs,
        data_delay=np.array([list(SONICOM_DATA_DELAY_SAMPLES)], dtype=np.float64),
        database_name="Synthetic SONICOM-shaped fixture",
        listener_short_name="P0001",
        comment="windowed",
    )


def write_synthetic_sofa(path: Path | None = None) -> Path:
    """Write the small horizontal set as a ``SimpleFreeFieldHRIR`` SOFA file."""

    destination = Path(path) if path is not None else FIXTURE_DIR / "synthetic_hrir.sofa"
    positions, hrirs = synthetic_hrir_set()
    try:
        return _write_sofa_with_sofar(
            destination,
            positions,
            hrirs,
            data_delay=np.zeros((1, 2)),
            database_name="Synthetic BinauralBuilder fixture",
            listener_short_name="synthetic",
            comment="windowed",
        )
    except ImportError:
        pass
    return _write_synthetic_sofa_with_h5py(destination, positions, hrirs)


def _write_synthetic_sofa_with_h5py(
    path: Path, positions: np.ndarray, hrirs: np.ndarray
) -> Path:
    """Fallback writer for environments with ``h5py`` but no ``sofar``."""

    import h5py

    destination = Path(path)
    measurements = hrirs.shape[0]

    with h5py.File(destination, "w") as handle:
        handle.attrs["Conventions"] = "SOFA"
        handle.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        handle.attrs["SOFAConventionsVersion"] = "1.0"
        handle.attrs["Version"] = "2.1"
        handle.attrs["DataType"] = "FIR"
        handle.attrs["RoomType"] = "free field"
        handle.attrs["Title"] = "Synthetic BinauralBuilder SAM workbench test HRIR"
        handle.attrs["DateCreated"] = "2024-01-01 00:00:00"
        handle.attrs["DateModified"] = "2024-01-01 00:00:00"
        handle.attrs["APIName"] = "binauralbuilder-sam-workbench"
        handle.attrs["APIVersion"] = "0.1.0"
        handle.attrs["AuthorContact"] = "BinauralBuilder repository"
        handle.attrs["Organization"] = "BinauralBuilder"
        handle.attrs["License"] = "Synthetic fixture created for this repository's tests"
        handle.attrs["ListenerShortName"] = "synthetic"

        handle.create_dataset("Data.IR", data=hrirs)
        handle.create_dataset("Data.SamplingRate", data=np.array([float(SAMPLE_RATE_HZ)]))
        handle["Data.SamplingRate"].attrs["Units"] = "hertz"
        handle.create_dataset("Data.Delay", data=np.zeros((1, 2)))

        source_position = handle.create_dataset("SourcePosition", data=positions)
        source_position.attrs["Type"] = "spherical"
        source_position.attrs["Units"] = "degree, degree, metre"

        listener_position = handle.create_dataset("ListenerPosition", data=np.zeros((1, 3)))
        listener_position.attrs["Type"] = "cartesian"
        listener_position.attrs["Units"] = "metre"

        listener_view = handle.create_dataset("ListenerView", data=np.array([[1.0, 0.0, 0.0]]))
        listener_view.attrs["Type"] = "cartesian"
        listener_view.attrs["Units"] = "metre"
        listener_up = handle.create_dataset("ListenerUp", data=np.array([[0.0, 0.0, 1.0]]))
        listener_up.attrs["Type"] = "cartesian"
        listener_up.attrs["Units"] = "metre"

        # Canonical frame: left ear at +y, right ear at -y.
        receiver_position = handle.create_dataset(
            "ReceiverPosition",
            data=np.array([[0.0, HEAD_RADIUS_M, 0.0], [0.0, -HEAD_RADIUS_M, 0.0]]).reshape(2, 3, 1),
        )
        receiver_position.attrs["Type"] = "cartesian"
        receiver_position.attrs["Units"] = "metre"

        emitter_position = handle.create_dataset("EmitterPosition", data=np.zeros((1, 3, 1)))
        emitter_position.attrs["Type"] = "cartesian"
        emitter_position.attrs["Units"] = "metre"

        handle.create_dataset("M", data=np.arange(measurements, dtype=np.float64))
    return destination


def main() -> int:
    print(f"wrote {write_impulse_wav()}")
    print(f"wrote {write_hrir_npz()}")
    for writer, name in (
        (write_synthetic_sofa, "synthetic_hrir.sofa"),
        (write_sonicom_shaped_sofa, "synthetic_sonicom_hrir.sofa"),
    ):
        try:
            print(f"wrote {writer()}")
        except ImportError:
            print(f"no SOFA backend installed; skipped {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

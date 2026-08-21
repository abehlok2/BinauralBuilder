"""The synthetic development fixtures are self-generated and free of demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.sam_workbench.conventions import cartesian_to_spherical_deg

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
IMPULSE_PATH = FIXTURE_DIR / "synthetic_impulse.wav"
HRIR_NPZ_PATH = FIXTURE_DIR / "synthetic_hrir.npz"
SOFA_PATH = FIXTURE_DIR / "synthetic_hrir.sofa"


def test_synthetic_fixtures_are_small_and_present():
    for path in (IMPULSE_PATH, HRIR_NPZ_PATH, SOFA_PATH):
        assert path.exists(), f"missing fixture {path.name}"
        assert path.stat().st_size < 256 * 1024, f"{path.name} is too large for a test fixture"


def test_impulse_fixture_carries_the_intended_itd_and_ild():
    soundfile = pytest.importorskip("soundfile")
    audio, sample_rate = soundfile.read(IMPULSE_PATH, always_2d=True)
    assert sample_rate == 44_100
    assert audio.shape == (64, 2)
    assert int(np.argmax(np.abs(audio[:, 0]))) == 8
    assert int(np.argmax(np.abs(audio[:, 1]))) == 12
    assert audio[8, 0] == pytest.approx(1.0)
    assert audio[12, 1] == pytest.approx(0.7)


def test_hrir_fixture_is_finite_and_uses_canonical_positions():
    with np.load(HRIR_NPZ_PATH) as data:
        positions = data["positions_deg"]
        hrirs = data["hrirs"]
        assert int(data["sample_rate_hz"]) == 44_100
    assert positions.shape == (8, 3)
    assert hrirs.shape == (8, 2, 48)
    assert np.all(np.isfinite(hrirs))
    # Azimuth 90 degrees is the listener's left, so the left ear leads there.
    left_index = int(np.argmin(np.abs(positions[:, 0] - 90.0)))
    left_peak = int(np.argmax(np.abs(hrirs[left_index, 0])))
    right_peak = int(np.argmax(np.abs(hrirs[left_index, 1])))
    assert left_peak < right_peak


def test_regenerating_the_fixtures_is_deterministic(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_synthetic_assets", FIXTURE_DIR / "make_synthetic_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    regenerated = tmp_path / "synthetic_hrir.npz"
    module.write_hrir_npz(regenerated)
    with np.load(regenerated) as fresh, np.load(HRIR_NPZ_PATH) as stored:
        np.testing.assert_array_equal(fresh["hrirs"], stored["hrirs"])
        np.testing.assert_array_equal(fresh["positions_deg"], stored["positions_deg"])


def test_synthetic_sofa_declares_canonical_receiver_geometry():
    h5py = pytest.importorskip("h5py")
    with h5py.File(SOFA_PATH, "r") as handle:
        assert handle.attrs["SOFAConventions"] == "SimpleFreeFieldHRIR"
        assert handle["Data.IR"].shape == (8, 2, 48)
        assert float(handle["Data.SamplingRate"][0]) == 44_100.0
        receivers = np.asarray(handle["ReceiverPosition"]).reshape(2, 3)
    # Left receiver at +y and right receiver at -y in the canonical frame.
    assert cartesian_to_spherical_deg(receivers[0])[0] == pytest.approx(90.0)
    assert cartesian_to_spherical_deg(receivers[1])[0] == pytest.approx(-90.0)

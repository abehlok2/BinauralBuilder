from pathlib import Path
import numpy as np
import pytest

import h5py

from src.audio.sam_workbench.compat import render_sam2_voice
from src.audio.sam_workbench.hrtf import DelayPolicy, load_sofa
from src.audio.sam_workbench.render.hrtf import HRTFRendererSpec, render_hrtf
from src.audio.sam_workbench.hrtf.interpolation import interpolate_log_magnitude_delay
from src.audio.sam_workbench.analysis.comparison import render_ab_comparison

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_hrir.sofa"

def test_sofa_load_validate_resample_and_hash():
    dataset = load_sofa(FIXTURE, target_sample_rate_hz=48_000)
    assert dataset.ir.shape[:2] == (8, 2)
    assert dataset.sample_rate_hz == 48_000
    assert len(dataset.content_hash) == 64
    assert dataset.metadata_report()["delay_policy"] == "bake_delay_into_ir"

def test_standard_requirements_include_sofa_reader():
    requirements = (Path(__file__).parents[2] / "requirements.txt").read_text()
    assert any(line.strip().lower().startswith("h5py") for line in requirements.splitlines())

def test_nonzero_data_delay_is_baked_or_preserved(tmp_path):
    path = tmp_path / "delayed.sofa"
    path.write_bytes(FIXTURE.read_bytes())
    with h5py.File(path, "r+") as handle:
        handle["Data.Delay"][...] = [[0.0001, 0.0002]]
        handle["Data.Delay"].attrs["Units"] = "second"
    baked = load_sofa(path, delay_policy=DelayPolicy.BAKE)
    preserved = load_sofa(path, delay_policy=DelayPolicy.PRESERVE)
    assert not baked.has_external_delay
    assert preserved.delay_samples[0] == pytest.approx([4.41, 8.82])
    assert baked.taps > preserved.taps

def test_aligned_interpolation_is_neutral_on_a_measured_direction():
    dataset = load_sofa(FIXTURE)
    reconstructed = interpolate_log_magnitude_delay(dataset, dataset.positions_m[3])
    np.testing.assert_allclose(reconstructed, dataset.ir[3], atol=1e-12)

def test_renderer_is_block_invariant_and_switches_without_impulses():
    sample_rate = 44_100
    mono = np.sin(2*np.pi*220*np.arange(3000)/sample_rate) * 0.1
    def trajectory(t):
        az = 2*np.pi*3*np.asarray(t)
        return np.column_stack((np.cos(az), np.sin(az), np.zeros_like(az)))
    spec = HRTFRendererSpec(FIXTURE, trajectory, crossfade_ms=8, control_interval_samples=32)
    one = render_hrtf(mono, spec, sample_rate, block_size=len(mono))
    many = render_hrtf(mono, spec, sample_rate, block_size=73)
    np.testing.assert_allclose(one, many, atol=2e-7)
    assert np.max(np.abs(np.diff(many, axis=1))) < 0.25

def test_voice_envelope_selects_explicit_sofa_renderer():
    audio = render_sam2_voice(.02, 44_100, params={"samSchemaVersion": 1,
        "rendererMode": "hrtf", "hrtfAsset": str(FIXTURE), "carrierFreq": 220,
        "hrtfOptions": {"schemaVersion": 1, "interpolation": "logmag_delay"}})
    assert audio.shape == (882, 2)
    assert np.all(np.isfinite(audio))

def test_voice_hrtf_render_matches_arbitrary_binauralbuilder_chunks():
    params = {"samSchemaVersion": 1, "rendererMode": "hrtf",
        "hrtfAsset": str(FIXTURE), "amp": .1, "carrierFreq": 220,
        "modFreq": 7, "arcWidthDeg": 270,
        "hrtfOptions": {"schemaVersion": 1, "interpolation": "nearest",
            "crossfadeMs": 12, "controlIntervalSamples": 37}}
    sample_rate = 44_100
    boundaries = (0, 1367, 3484, 5292)
    whole = render_sam2_voice(boundaries[-1]/sample_rate, sample_rate, params=params)
    parts = []
    for start, end in zip(boundaries, boundaries[1:]):
        parts.append(render_sam2_voice((end-start)/sample_rate, sample_rate,
                                      params=params, initial_offset=start/sample_rate))
    chunked = np.concatenate(parts)
    # Duration-to-frame rounding at each public call can differ by a sample;
    # compare the common timeline, which must contain identical FIR state.
    common = min(len(whole), len(chunked))
    np.testing.assert_allclose(chunked[:common], whole[:common], atol=2e-6)

def test_missing_hrtf_extras_do_not_import_with_core():
    root = Path(__file__).parents[2] / "src" / "audio" / "sam_workbench"
    sources = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "import slab" not in sources

def test_ab_comparison_uses_one_matched_gain_for_both_ears():
    params = {"hrtfAsset": str(FIXTURE), "amp": .05, "carrierFreq": 220,
              "modFreq": 2, "hrtfOptions": {"schemaVersion": 1}}
    compared = render_ab_comparison(params, .03, 44_100, "abstract_pm", "hrtf")
    rms_a = np.sqrt(np.mean(np.square(compared.audio_a.astype(np.float64))))
    rms_b = np.sqrt(np.mean(np.square(compared.audio_b.astype(np.float64))))
    assert rms_b == pytest.approx(rms_a, rel=1e-5)

def test_hybrid_mode_is_not_silently_replaced_by_abstract_pm():
    """Hybrid renders now, but still refuses to run without its SOFA asset.

    The point this test has always made is that an unusable configuration must
    fail rather than quietly render as something else. What changed is why it
    is unusable: the mode is implemented, so a missing asset is the reason.
    """

    with pytest.raises((ValueError, TypeError, OSError)):
        render_sam2_voice(.01, 44_100, params={"rendererMode": "hybrid"})


def test_hybrid_mode_renders_when_it_has_what_it_needs():
    audio = render_sam2_voice(
        .05, 44_100,
        params={"rendererMode": "hybrid", "amp": 0.5, "hrtfAsset": str(FIXTURE)},
    )
    assert audio.shape[1] == 2
    assert np.all(np.isfinite(audio))

import json
import math

import numpy as np
import soundfile as sf

from src.audio.sam_workbench.controls import LfoControl
from src.audio.sam_workbench.dsp import render_source
from src.audio.sam_workbench.export import export_wav
from src.audio.sam_workbench.model import Project, SamModulationSpec, SignalSpec, Source


def test_control_is_block_size_invariant():
    control = LfoControl(rate_hz=3.7, depth=2.0, offset=-0.5, phase_rad=0.2)
    whole = control.render(0, 1000, 48_000)
    split = np.concatenate((control.render(0, 137, 48_000), control.render(137, 863, 48_000)))
    np.testing.assert_array_equal(whole, split)


def test_exact_sam_matches_reference_equations_and_blocks():
    source = Source(amplitude_linear=0.4, signal=SignalSpec(carrier_frequency_hz=440.0, phase_rad=0.3),
                    sam=SamModulationSpec(rate_hz=4.0, depth_rad=1.2, phase_rad=0.1))
    sample_rate = 44_100
    rendered = render_source(source, sample_rate, 5000, 127)
    one_block = render_source(source, sample_rate, 5000, 5000)
    np.testing.assert_allclose(rendered, one_block, atol=2e-7, rtol=0)
    t = np.arange(5000) / sample_rate
    carrier = 2 * math.pi * 440.0 * t + 0.3
    modulation = 1.2 * np.sin(2 * math.pi * 4.0 * t + 0.1)
    expected = np.vstack((0.4 * np.sin(carrier + modulation), 0.4 * np.sin(carrier - modulation)))
    np.testing.assert_allclose(rendered, expected, atol=2e-7, rtol=0)


def test_wav_export_writes_reconstruction_manifest(tmp_path):
    project = Project(sources=(Source(amplitude_linear=0.1),))
    wav_path = tmp_path / "reference.wav"
    manifest_path = export_wav(project, 0.05, wav_path)
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    manifest = json.loads(manifest_path.read_text())
    assert sample_rate == 44_100
    assert audio.shape == (2205, 2)
    assert manifest["renderer"] == "exact_symmetric_sam"
    assert manifest["frames"] == 2205
    assert len(manifest["project_sha256"]) == 64

import copy
import numpy as np

from src.audio.sam_workbench.analysis.trajectory_metrics import trajectory_metrics
from src.audio.sam_workbench.dsp.blocks import RenderBlock, RenderContext
from src.audio.sam_workbench.render.geometric import (
    GeometricBinauralRenderer,
    GeometricSpec,
    geometric_cues,
)
from src.audio.sam_workbench.trajectory import (
    Bezier,
    CanonicalTrajectory,
    Line,
    KeyframedTraversal,
    Mathematical,
    Polyline,
    Spline,
    StochasticTraversal,
    Transform,
    Traversal,
    evaluate_arclength,
    segment_positions,
)
from src.audio.sam_workbench.trajectory.legacy_paths import (
    LegacyPathTransform,
    canonical_profile_points,
    legacy_profile_geometry,
    migrate_legacy_profile,
)
from src.audio.sam_workbench.trajectory.transforms import ListenerTransform


def test_mathematical_geometry_uses_safe_parametric_expressions():
    geometry = Mathematical("cos(2*pi*u)", "sin(2*pi*u)", "u - 0.5")
    points = geometry.evaluate(np.array([0.0, 0.25, 0.5]))
    np.testing.assert_allclose(
        points,
        [[1.0, 0.0, -0.5], [0.0, 1.0, -0.25], [-1.0, 0.0, 0.0]],
        atol=1e-12,
    )


def test_mathematical_geometry_rejects_arbitrary_python():
    geometry = Mathematical("__import__('os').getcwd()", "0", "0")
    with np.testing.assert_raises(ValueError):
        geometry.evaluate([0.0])


def test_legacy_pixel_conversion_round_trips_without_mutating_profile():
    profile = {"points": [[100.0, 0.0], [0.0, -100.0]], "closedLoop": False}
    original = copy.deepcopy(profile)
    transform = LegacyPathTransform(100.0)
    canonical = canonical_profile_points(profile)
    assert np.allclose(canonical, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(transform.from_canonical(canonical), profile["points"])
    assert profile == original and "schemaVersion" not in profile
    assert migrate_legacy_profile(profile)["schemaVersion"] == 2


def test_arc_length_parameterization_is_distance_uniform():
    path = Polyline(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 3.0, 0.0)))
    assert np.allclose(
        evaluate_arclength(path, [0.25, 0.5, 0.75]),
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
        atol=2e-3,
    )


def test_traversal_jump_crossfade_is_equal_power():
    traversal = Traversal(
        mode="discontinuous", steps=4, crossfade_s=0.1, duration_s=1.0
    )
    old, new = traversal.transition_weights(np.linspace(0, 1, 1001))
    active = (old > 0) & (new < 1)
    assert active.any()
    assert np.allclose(old[active] ** 2 + new[active] ** 2, 1.0)


def test_geometric_itd_and_distance_levels_match_analytic_values():
    listener = ListenerTransform(ear_spacing_m=0.2)
    position = np.array([[0.0, 1.0, 0.0]])
    cues = geometric_cues(position, listener)
    assert np.allclose(cues["distance_m"], [[0.9, 1.1]])
    assert np.allclose(cues["itd_s"], [-0.2 / 343.0])
    metrics = trajectory_metrics(position, 44100, listener)
    assert np.allclose(metrics["ild_db"], [20 * np.log10(1.1 / 0.9)])


def test_legacy_segments_translate_without_modification():
    segments = [
        {
            "mode": "rotate",
            "start_deg": 0.0,
            "speed_deg_per_s": 90.0,
            "distance_m": 2.0,
            "seconds": 1.0,
            "easing": "linear",
        }
    ]
    original = copy.deepcopy(segments)
    assert np.allclose(
        segment_positions(segments, [0.0, 1.0])[:, :2],
        [[2.0, 0.0], [0.0, 2.0]],
        atol=1e-12,
    )
    assert segments == original


def test_geometry_and_affine_transform_are_vectorized_and_invertible():
    transform = Transform(
        translation_m=(1.0, 2.0, 3.0),
        yaw_pitch_roll_deg=(30.0, 10.0, -5.0),
        scale=(-2.0, 1.5, 0.5),
        shear=(0.1, -0.2, 0.3),
    )
    points = Bezier(((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 0.0, 1.0))).evaluate(
        np.linspace(0, 1, 9)
    )
    assert points.shape == (9, 3)
    assert np.allclose(transform.inverse(transform.apply(points)), points)
    assert Spline(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 1.0))).evaluate(
        [0.5]
    ).shape == (1, 3)


def test_discontinuous_trajectory_crossfade_moves_between_adjacent_steps():
    trajectory = CanonicalTrajectory(
        Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        Traversal(duration_s=1.0, mode="discontinuous", steps=4, crossfade_s=0.1),
        arc_length=False,
        coordinate_smoothing=True,
    )
    positions = trajectory.evaluate(np.array([0.15, 0.225, 0.249]))
    assert positions[0, 0] == 0.0
    assert 0.0 < positions[1, 0] < 1.0 / 3.0
    assert positions[2, 0] < 1.0 / 3.0


def test_legacy_smoothed_profile_compiles_without_changing_payload():
    profile = {
        "kind": "spline",
        "closedLoop": False,
        "points": [[-100.0, 0.0], [0.0, -100.0], [100.0, 0.0]],
        "subNodesPerSegment": 8,
    }
    original = copy.deepcopy(profile)
    geometry = legacy_profile_geometry(profile)
    assert geometry.evaluate(np.linspace(0.0, 1.0, 16)).shape == (16, 3)
    assert profile == original


def test_geometric_renderer_is_block_invariant_and_reports_cues():
    sample_rate = 8_000
    trajectory = lambda times: np.broadcast_to((1.0, 0.4, 0.0), (len(times), 3))
    spec = GeometricSpec(trajectory=trajectory, doppler_enabled=True)
    source = np.zeros(256, dtype=np.float64)
    source[0] = 1.0
    context = RenderContext(sample_rate_hz=sample_rate, block_size=64)

    one_block = GeometricBinauralRenderer(spec)
    one_block.prepare(context)
    expected = one_block.process(source, RenderBlock(0, len(source)))

    split = GeometricBinauralRenderer(spec)
    split.prepare(context)
    actual = np.concatenate(
        [
            split.process(
                source[start : start + 64],
                RenderBlock(start, min(64, len(source) - start)),
            )
            for start in range(0, len(source), 64)
        ],
        axis=1,
    )
    assert np.array_equal(actual, expected)
    diagnostics = split.diagnostics()
    assert diagnostics["distance_m"].shape == (64, 2)
    assert diagnostics["itd_s"].shape == (64,)


def test_keyframed_and_stochastic_traversals_are_absolute_time_deterministic():
    keyframed = KeyframedTraversal((0.0, 1.0, 2.0), (0.0, 1.0, 0.5), "smooth")
    assert np.allclose(keyframed.progress([0.0, 0.5, 1.0, 2.0]), [0.0, 0.5, 1.0, 0.5])
    stochastic = StochasticTraversal(interval_s=0.25, seed=42, smoothing="linear")
    times = np.arange(100) / 100.0
    whole = stochastic.progress(times)
    blocked = np.concatenate(
        [stochastic.progress(times[:37]), stochastic.progress(times[37:])]
    )
    assert np.array_equal(whole, blocked)


def test_discontinuous_geometric_crossfade_spatializes_two_audio_branches():
    sample_rate = 8_000
    trajectory = CanonicalTrajectory(
        Line((0.3, -1.0, 0.0), (2.0, 1.0, 0.0)),
        Traversal(duration_s=1.0, mode="discontinuous", steps=2, crossfade_s=0.2),
        arc_length=False,
    )
    renderer = GeometricBinauralRenderer(GeometricSpec(trajectory, distance_law="none"))
    renderer.prepare(RenderContext(sample_rate, 256))
    source = np.sin(2 * np.pi * 440 * np.arange(2048) / sample_rate)
    audio = renderer.process(source, RenderBlock(0, len(source)))
    # Coordinates remain stepped; audio is finite and both delayed branches
    # contribute during the fade before the half-cycle jump.
    assert np.allclose(trajectory.evaluate([0.45, 0.49])[:, 1], -1.0)
    assert np.isfinite(audio).all()
    old, new, old_gain, new_gain = trajectory.audio_branches(np.array([0.45, 0.49]))
    assert not np.allclose(old, new)
    assert np.all((old_gain > 0) & (new_gain > 0))

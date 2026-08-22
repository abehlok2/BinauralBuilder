"""Phase 8 optimization: the same audio, measurably cheaper.

Every optimization here is only allowed if it changes cost and not output, so
that is what these tests check first. The partitioned convolver and the plain
one must agree to floating-point round-off across block patterns and filter
changes; the streaming crossover must reproduce, block by block, exactly what
the whole-signal split produces.

The one behaviour that *does* change is the streaming crossover's existence.
Splitting a stream by calling the stateless splitter once per block restarts
every filter at each boundary, which is a defect rather than an approximation -
the error is a step, at roughly a quarter of the signal's peak. That is
measured here too, so the reason the streaming path exists stays on the record.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.sam_workbench.benchmark import (
    REALTIME_TARGET_LOAD,
    BenchmarkCase,
    BenchmarkResult,
    describe_machine,
    run_case,
    standard_matrix,
    synthetic_dataset,
)
from src.audio.sam_workbench.dsp.convolution import (
    MAX_PARTITIONS,
    MIN_PARTITION_SAMPLES,
    CrossfadingConvolver,
    OverlapSaveConvolver,
    PartitionedEngine,
)
from src.audio.sam_workbench.dsp.crossover import CrossoverBank

# Round-off on signals whose peak is in the tens; anything above this is a
# difference in the arithmetic, not in the last bit of it.
TOLERANCE = 1e-9


def _reference(taps: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """The convolution every path here must reproduce."""

    from scipy.signal import fftconvolve

    return fftconvolve(signal, taps)[: signal.size]


def _stream(convolver: OverlapSaveConvolver, signal: np.ndarray, sizes) -> np.ndarray:
    output = []
    position = 0
    for size in sizes:
        size = min(int(size), signal.size - position)
        if size <= 0:
            break
        output.append(convolver.process(signal[position : position + size]))
        position += size
    return np.concatenate(output)


# --- partitioned convolution ------------------------------------------------


@pytest.mark.parametrize("taps", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param([512] * 10 + [137], id="uniform-512-short-tail"),
        pytest.param([1024] * 5 + [137], id="uniform-1024"),
        pytest.param([128] * 41 + [9], id="uniform-128"),
        pytest.param([512, 300, 512, 512, 300, 512, 512, 512, 512, 512, 512, 561], id="mixed"),
    ],
)
def test_convolution_is_the_same_however_the_blocks_arrive(taps, sizes):
    rng = np.random.default_rng(3)
    filter_taps = rng.normal(0.0, 1.0, taps)
    signal = rng.normal(0.0, 0.1, 512 * 10 + 137)

    streamed = _stream(OverlapSaveConvolver(filter_taps), signal, sizes)
    assert streamed.size == signal.size
    assert np.max(np.abs(streamed - _reference(filter_taps, signal))) < TOLERANCE


def test_partitioned_engine_reproduces_the_state_it_would_have_reached():
    """Priming exists so the engine can be entered mid-stream, not reset into."""

    rng = np.random.default_rng(5)
    taps = rng.normal(0.0, 1.0, 2048)
    signal = rng.normal(0.0, 0.1, 512 * 12)

    running = PartitionedEngine(taps, 512)
    for index in range(8):
        running.process(signal[index * 512 : (index + 1) * 512])

    primed = PartitionedEngine(taps, 512)
    primed.prime(signal[: 8 * 512])

    for index in (8, 9, 10):
        block = signal[index * 512 : (index + 1) * 512]
        assert np.array_equal(running.process(block), primed.process(block))


def test_partitioning_engages_only_where_it_measured_faster():
    """Below the threshold the single transform wins, so it must stay."""

    rng = np.random.default_rng(0)
    short = OverlapSaveConvolver(rng.normal(0.0, 1.0, 256))
    for _ in range(3):
        short.process(rng.normal(0.0, 0.1, 512))
    assert short._engine is None

    long_filter = OverlapSaveConvolver(rng.normal(0.0, 1.0, 2048))
    for _ in range(3):
        long_filter.process(rng.normal(0.0, 0.1, 512))
    assert long_filter._engine is not None
    assert long_filter._engine.partition == 512

    # A block too small to partition sensibly is left to the plain path.
    tiny = OverlapSaveConvolver(rng.normal(0.0, 1.0, 2048))
    for _ in range(3):
        tiny.process(rng.normal(0.0, 0.1, MIN_PARTITION_SAMPLES - 1))
    assert tiny._engine is None

    # And so is a filter that would need more partitions than the cap allows.
    absurd = OverlapSaveConvolver(rng.normal(0.0, 1.0, MAX_PARTITIONS * 64 + 1))
    for _ in range(3):
        absurd.process(rng.normal(0.0, 0.1, 64))
    assert absurd._engine is None


def test_changing_the_filter_invalidates_the_cached_spectrum():
    """The filter spectrum is cached; a stale one would convolve with the old filter."""

    rng = np.random.default_rng(11)
    first = rng.normal(0.0, 1.0, 256)
    second = rng.normal(0.0, 1.0, 256)
    signal = rng.normal(0.0, 0.1, 512 * 6)

    convolver = OverlapSaveConvolver(first)
    convolver.process(signal[:512])
    convolver.set_filter(second)
    after = convolver.process(signal[512:1024])

    expected = OverlapSaveConvolver(second)
    expected.set_filter(second, reset_history=False)
    expected._history = np.concatenate(
        (np.zeros(max(0, second.size - 1 - 512)), signal[:512])
    )[-(second.size - 1) :]
    assert np.max(np.abs(after - expected.process(signal[512:1024]))) < TOLERANCE


@pytest.mark.parametrize("taps", [256, 2048])
def test_filters_may_change_length_mid_stream(taps):
    """A moving source switches filters; a hybrid render can change their length."""

    rng = np.random.default_rng(13)
    signal = rng.normal(0.0, 0.1, 512 * 12)
    lengths = [taps, 300, 4096, taps]
    filters = [rng.normal(0.0, 1.0, length) for length in lengths]

    convolver = OverlapSaveConvolver(filters[0])
    plain = OverlapSaveConvolver(filters[0])
    plain._wants_partitioning = lambda block_size: False  # force the reference path

    output, expected = [], []
    for index in range(12):
        if index % 3 == 0:
            convolver.set_filter(filters[index // 3])
            plain.set_filter(filters[index // 3])
        block = signal[index * 512 : (index + 1) * 512]
        output.append(convolver.process(block))
        expected.append(plain.process(block))
    assert np.max(np.abs(np.concatenate(output) - np.concatenate(expected))) < TOLERANCE


def test_crossfading_convolver_still_hides_a_filter_switch():
    """Partitioning must not disturb what the crossfade is there to do."""

    rng = np.random.default_rng(17)
    quiet = np.zeros(2048)
    quiet[0] = 1.0
    loud = np.zeros(2048)
    loud[0] = -1.0

    convolver = CrossfadingConvolver(crossfade_ms=12.0, sample_rate_hz=48_000.0)
    convolver.reset(quiet)
    signal = rng.normal(0.0, 0.1, 512 * 8)
    before = convolver.process(signal[:512])

    convolver.set_filter(loud)
    during = np.concatenate(
        [convolver.process(signal[start : start + 512]) for start in range(512, 512 * 4, 512)]
    )
    # A switch without a crossfade steps by twice the sample; the fade keeps
    # every step within what the signal itself does between samples.
    joined = np.concatenate((before, during))
    steps = np.abs(np.diff(joined))
    assert steps.max() < 4.0 * np.abs(np.diff(signal[: joined.size])).max()


# --- the streaming crossover ------------------------------------------------


@pytest.mark.parametrize("crossovers", [(200.0,), (200.0, 2000.0), (120.0, 800.0, 6000.0)])
def test_streamed_split_matches_the_whole_signal_split(crossovers):
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 0.2, 4096)
    bank = CrossoverBank(crossovers, 48_000.0)

    stream = bank.stream()
    streamed = np.concatenate(
        [stream.process(signal[start : start + 512]) for start in range(0, 4096, 512)],
        axis=1,
    )
    assert np.max(np.abs(bank.split(signal) - streamed)) < TOLERANCE


def test_streamed_split_handles_channel_major_audio():
    rng = np.random.default_rng(1)
    signal = rng.normal(0.0, 0.2, (2, 4096))
    bank = CrossoverBank((200.0, 2000.0), 48_000.0)

    stream = bank.stream()
    streamed = np.concatenate(
        [stream.process(signal[:, start : start + 512]) for start in range(0, 4096, 512)],
        axis=2,
    )
    assert streamed.shape == (3, 2, 4096)
    assert np.max(np.abs(bank.split(signal) - streamed)) < TOLERANCE


def test_splitting_a_stream_statelessly_is_the_defect_the_stream_fixes():
    """Why CrossoverStream exists, kept as a measurement rather than a claim."""

    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 0.2, 4096)
    bank = CrossoverBank((200.0, 2000.0), 48_000.0)

    whole = bank.split(signal)
    restarted = np.concatenate(
        [bank.split(signal[start : start + 512]) for start in range(0, 4096, 512)], axis=1
    )
    error = np.max(np.abs(whole - restarted))
    # Not a rounding difference: a large fraction of the signal's own peak.
    assert error > 0.1 * np.max(np.abs(signal))


def test_streamed_bands_still_sum_to_an_allpass():
    """The reconstruction guarantee has to survive the state, not just the maths."""

    impulse = np.zeros(8192)
    impulse[0] = 1.0
    bank = CrossoverBank((200.0, 2000.0), 48_000.0)
    stream = bank.stream()
    summed = np.concatenate(
        [stream.process(impulse[start : start + 512]) for start in range(0, 8192, 512)],
        axis=1,
    ).sum(axis=0)

    spectrum = np.fft.rfft(summed)
    frequencies = np.fft.rfftfreq(summed.size, 1.0 / 48_000.0)
    audible = (frequencies > 30.0) & (frequencies < 18_000.0)
    magnitude_db = 20.0 * np.log10(np.abs(spectrum[audible]))
    assert magnitude_db.max() - magnitude_db.min() < 0.2


def test_a_reset_stream_starts_over():
    rng = np.random.default_rng(2)
    signal = rng.normal(0.0, 0.2, 2048)
    bank = CrossoverBank((500.0,), 48_000.0)
    stream = bank.stream()

    first = stream.process(signal[:512])
    stream.process(signal[512:1024])
    stream.reset()
    assert np.array_equal(stream.process(signal[:512]), first)


# --- the benchmark itself ---------------------------------------------------


def test_synthetic_dataset_covers_the_sphere_at_the_requested_length():
    dataset = synthetic_dataset(taps=333, azimuth_step_deg=30.0, elevations_deg=(0.0, 30.0))
    assert dataset.taps == 333
    assert dataset.measurements == 24
    assert dataset.ir.shape == (24, 2, 333)
    assert np.allclose(np.linalg.norm(dataset.positions_m, axis=1), 1.0)
    assert dataset.has_external_delay is False


def test_benchmark_case_rejects_what_it_cannot_measure():
    with pytest.raises(ValueError):
        BenchmarkCase(sources=0)
    with pytest.raises(ValueError):
        BenchmarkCase(interpolation="telepathy")
    with pytest.raises(ValueError):
        BenchmarkCase(sample_rate_hz=0.0)


def test_load_is_measured_against_the_callback_budget():
    case = BenchmarkCase(block_size=512, sample_rate_hz=48_000.0)
    budget = case.block_budget_s
    result = BenchmarkResult(case=case, blocks=4, block_times_s=(
        0.1 * budget, 0.2 * budget, 0.3 * budget, 0.4 * budget
    ))
    assert result.mean_load == pytest.approx(0.25)
    assert result.worst_load == pytest.approx(0.4)
    assert result.feasible_realtime is True
    assert result.has_overruns is False

    overrunning = BenchmarkResult(case=case, blocks=2, block_times_s=(0.1 * budget, 1.5 * budget))
    assert overrunning.has_overruns is True


def test_standard_matrix_sweeps_every_axis_the_specification_names():
    cases = standard_matrix()
    assert {case.sources for case in cases} >= {1, 4, 8}
    assert {case.hrir_taps for case in cases} >= {128, 512, 2048}
    assert {case.bands for case in cases} >= {1, 4, 8}
    assert {case.block_size for case in cases} >= {128, 512, 1024}
    assert {case.sample_rate_hz for case in cases} >= {44_100.0, 48_000.0}
    assert any(case.anchor for case in cases)
    assert len(set(cases)) == len(cases)


def test_a_benchmark_run_reports_a_plausible_load():
    case = BenchmarkCase(block_size=512, sample_rate_hz=48_000.0)
    result = run_case(case, seconds=0.2)
    assert result.blocks > 0
    assert len(result.block_times_s) == result.blocks
    assert 0.0 < result.mean_load < 100.0
    assert result.worst_load >= result.mean_load
    assert "real time" in result.summary() or "OFFLINE" in result.summary()
    assert describe_machine()["numpy"]


def test_a_single_nearest_source_stays_well_inside_the_realtime_target():
    """The centre of the matrix is the case that must never regress.

    Deliberately loose: this runs on whatever machine the suite runs on, and a
    contended one measures its scheduler as much as this code. A regression
    worth catching here is an order of magnitude, not a percent.
    """

    result = run_case(BenchmarkCase(), seconds=0.5)
    assert result.mean_load < REALTIME_TARGET_LOAD


def test_a_long_filter_is_not_slower_than_a_short_one_by_its_length_ratio():
    """Partitioning is what keeps an eight-times-longer filter from costing that."""

    short = run_case(BenchmarkCase(hrir_taps=256), seconds=0.5)
    long_filter = run_case(BenchmarkCase(hrir_taps=2048), seconds=0.5)
    assert long_filter.mean_load < 4.0 * short.mean_load

"""Numba availability and configuration helpers.

This module centralises Numba checks so that the synth functions can still
import even when Numba is unavailable. When the real library cannot be
imported we install a lightweight stub in ``sys.modules`` that provides the
minimal decorators and helpers used across the codebase.
"""
from __future__ import annotations
"""Numba availability and benchmarking helpers."""

import os
import sys
import time
import types
from typing import Callable, Iterable

import numpy as np

_NUMBA_LOGGED = False
NUMBA_AVAILABLE = False


class _NumbaStub(types.SimpleNamespace):
    """Lightweight stand-in exposing the Numba API we rely on.

    Decorators simply return the wrapped function unchanged so code continues
    to work (albeit without JIT acceleration). Parallel ranges fall back to
    Python's ``range``.
    """

    __binauralbuilder_stub__ = True

    def __init__(self) -> None:
        super().__init__(
            njit=self._identity_decorator,
            jit=self._identity_decorator,
            vectorize=self._identity_decorator,
            guvectorize=self._identity_decorator,
            prange=self._prange,
            get_num_threads=lambda: os.cpu_count() or 1,
            set_num_threads=lambda *_args, **_kwargs: None,
            config=types.SimpleNamespace(NUMBA_DEFAULT_NUM_THREADS=os.cpu_count() or 1),
        )

    @staticmethod
    def _identity_decorator(*dargs, **dkwargs):
        def _wrap(func: Callable):
            return func

        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]
        return _wrap

    @staticmethod
    def _prange(*args, **kwargs) -> Iterable[int]:
        return range(*args)



def _install_stub() -> types.SimpleNamespace:
    stub = _NumbaStub()
    sys.modules.setdefault("numba", stub)
    return stub



def configure_numba(log: bool = True) -> bool:
    """Ensure a usable ``numba`` module is importable and log status once."""

    global NUMBA_AVAILABLE, _NUMBA_LOGGED

    if "numba" in sys.modules and getattr(sys.modules["numba"], "__binauralbuilder_stub__", False):
        numba_mod = sys.modules["numba"]
        NUMBA_AVAILABLE = False
    else:
        try:
            import numba as numba_mod  # type: ignore

            NUMBA_AVAILABLE = True
        except Exception:  # pragma: no cover - only hit when numba isn't installed
            numba_mod = _install_stub()
            NUMBA_AVAILABLE = False

    if log and not _NUMBA_LOGGED:
        cpu_name = os.environ.get("NUMBA_CPU_NAME", "generic")
        cpu_features = os.environ.get("NUMBA_CPU_FEATURES", "auto")
        if NUMBA_AVAILABLE:
            threads = getattr(numba_mod, "get_num_threads", lambda: os.cpu_count() or 1)()
            print(
                f"INFO: Numba detected (threads={threads}, target={cpu_name}, features={cpu_features})."
            )
        else:
            print(
                "WARNING: Numba is not installed. Falling back to pure Python implementations, "
                "which will be significantly slower. Set NUMBA_CPU_NAME=host before installation "
                "to enable CPU-specific optimisations."
            )
        _NUMBA_LOGGED = True

    return NUMBA_AVAILABLE



def run_parallel_benchmark(array_len: int = 1_000_000) -> bool:
    """Run a minimal parallel benchmark to confirm multi-core execution.

    Returns ``True`` if the benchmark completed, ``False`` when Numba was
    unavailable.
    """

    configure_numba(log=False)
    import numba  # type: ignore  # noqa: F401

    if not NUMBA_AVAILABLE:
        print("Numba is unavailable; skipping parallel benchmark.")
        return False

    data = np.ones(array_len, dtype=np.float32)

    @numba.njit(parallel=True)
    def _sum_squares(values: np.ndarray) -> float:  # type: ignore[name-defined]
        total = 0.0
        for i in numba.prange(values.shape[0]):
            total += values[i] * values[i]
        return total

    _sum_squares(data[:1024])
    start = time.perf_counter()
    total = _sum_squares(data)
    duration = time.perf_counter() - start
    threads_used = numba.get_num_threads()

    print(
        f"Benchmark complete (threads={threads_used}): sum={total:.2f}, "
        f"elapsed={duration:.3f}s, throughput={array_len / duration / 1e6:.1f} Msamples/s"
    )
    return True



def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Numba status and benchmark helper")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a short parallel benchmark to verify multi-core execution",
    )
    parser.add_argument(
        "--array-len",
        type=int,
        default=1_000_000,
        help="Array length to use for the benchmark (default: 1,000,000)",
    )

    args = parser.parse_args(argv)

    configure_numba(log=True)

    if args.benchmark:
        return 0 if run_parallel_benchmark(array_len=args.array_len) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

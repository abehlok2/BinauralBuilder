"""Measure the production export path, not a proxy for it.

Runs the same entry point a user's export runs - ``render_sam2_voice`` through
the compatibility adapter - so the numbers describe what actually happens when
someone renders a track, rather than what a hand-built harness does.

Reports wall time, audio-seconds per wall-clock second (above 1.0 is faster
than real time), peak resident memory, and the configuration each figure
belongs to. Run it before and after a change and compare like for like.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.sam_workbench.compat import render_sam2_voice  # noqa: E402

SOFA = Path(__file__).resolve().parent.parent / "tests" / "sam_workbench" / "fixtures" / "synthetic_hrir.sofa"


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    return usage / (1024.0 if platform.system() != "Darwin" else 1024.0 * 1024.0)


def run_case(name: str, *, duration_s: float, params: dict, sample_rate: int, block_size: int | None):
    start = time.perf_counter()
    audio = render_sam2_voice(
        duration_s, sample_rate, params=params, block_size=block_size
    )
    elapsed = time.perf_counter() - start
    return {
        "case": name,
        "durationS": duration_s,
        "sampleRateHz": sample_rate,
        "blockSize": block_size,
        "renderer": params.get("rendererMode", "abstract_pm"),
        "interpolation": (params.get("hrtfOptions") or {}).get("interpolation"),
        "controlIntervalSamples": (params.get("hrtfOptions") or {}).get("controlIntervalSamples"),
        "wallSeconds": round(elapsed, 4),
        "audioSecondsPerWallSecond": round(duration_s / elapsed, 3) if elapsed else None,
        "frames": int(audio.shape[0]),
        "peakMemoryMb": round(peak_memory_mb(), 1),
    }


def trajectory(turns: float = 3.0, duration_s: float = 4.0):
    return {
        "geometry": {"type": "dome_traversal", "parameters": {"turns": turns}},
        "traversal": {"durationS": duration_s},
    }


def cases(duration_s: float, sample_rate: int):
    base = {"amp": 0.5, "carrierFreq": 300.0, "modFreq": 4.0}
    hrtf = dict(base, rendererMode="hrtf", hrtfAsset=str(SOFA),
                canonicalTrajectory=trajectory(duration_s=duration_s))
    yield "abstract_pm", dict(base), None
    yield "geometric", dict(base, rendererMode="geometric",
                            canonicalTrajectory=trajectory(duration_s=duration_s)), None
    for interpolation in ("nearest", "logmag_delay"):
        for interval in (128, 512):
            params = dict(hrtf)
            params["hrtfOptions"] = {
                "interpolation": interpolation,
                "controlIntervalSamples": interval,
                "crossfadeMs": 10.0,
            }
            yield f"hrtf/{interpolation}/interval{interval}", params, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=4.0, help="audio seconds per case")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--label", default="", help="what this run is measuring")
    parser.add_argument("--out", default="", help="write JSON results here")
    options = parser.parse_args()

    results = []
    for name, params, block_size in cases(options.duration, options.sample_rate):
        try:
            results.append(run_case(name, duration_s=options.duration, params=params,
                                    sample_rate=options.sample_rate, block_size=block_size))
        except Exception as error:  # a case that cannot run is a result too
            results.append({"case": name, "error": f"{type(error).__name__}: {error}"})

    report = {
        "label": options.label,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "durationS": options.duration,
        "sampleRateHz": options.sample_rate,
        "results": results,
    }
    width = max(len(str(entry["case"])) for entry in results)
    print(f"{'case'.ljust(width)}  {'wall s':>8}  {'x realtime':>10}  {'peak MB':>8}")
    print("-" * (width + 32))
    for entry in results:
        if "error" in entry:
            print(f"{entry['case'].ljust(width)}  {entry['error']}")
            continue
        print(
            f"{entry['case'].ljust(width)}  {entry['wallSeconds']:>8.3f}  "
            f"{entry['audioSecondsPerWallSecond']:>10.2f}  {entry['peakMemoryMb']:>8.1f}"
        )
    if options.out:
        Path(options.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

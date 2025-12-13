"""Colored noise generation and spectrogram visualization utilities."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import json
import numpy as np
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt


@dataclass
class ColoredNoiseGenerator:
    """Generate colored noise with configurable spectrum and filters.

    Parameters
    ----------
    sample_rate : int
        Sampling rate of the noise in Hz.
    duration : float
        Duration of the generated noise in seconds.
    exponent : float
        Power-law exponent for the low-frequency end of the spectrum
        (0=white, 1=pink, 2=brown, -1=blue, etc.).
    high_exponent : float, optional
        Target power-law exponent for the highest frequencies. When set,
        the generator interpolates between ``exponent`` and
        ``high_exponent`` across the spectrum, allowing smoother or more
        aggressive coloration changes.
    distribution_curve : float
        Shapes how quickly the interpolation between ``exponent`` and
        ``high_exponent`` occurs (values <1 bias towards the start,
        values >1 bias towards the end).
    lowcut : float, optional
        Low cut-off frequency in Hz for optional filtering.
    highcut : float, optional
        High cut-off frequency in Hz for optional filtering.
    amplitude : float
        Output gain applied to the noise.
    seed : int, optional
        Random seed for reproducibility.
    """

    sample_rate: int = 44100
    duration: float = 1.0
    exponent: float = 1.0
    high_exponent: Optional[float] = None
    distribution_curve: float = 1.0
    lowcut: Optional[float] = None
    highcut: Optional[float] = None
    amplitude: float = 1.0
    seed: Optional[int] = 1

    def generate(self) -> np.ndarray:
        """Return generated noise as a NumPy array."""
        n = int(self.duration * self.sample_rate)
        if self.seed is not None:
            np.random.seed(self.seed)
        noise = self._generate_colored_noise(n)
        if self.lowcut is not None or self.highcut is not None:
            nyq = 0.5 * self.sample_rate
            low = self.lowcut / nyq if self.lowcut else None
            high = self.highcut / nyq if self.highcut else None
            if low and high:
                b, a = butter(4, [low, high], btype="band")
            elif low:
                b, a = butter(4, low, btype="high")
            else:
                b, a = butter(4, high, btype="low")
            noise = lfilter(b, a, noise)
        return (noise * self.amplitude).astype(np.float32)

    def _generate_colored_noise(self, n: int) -> np.ndarray:
        """Create noise with an interpolated spectral slope profile."""

        n = int(n)
        if n <= 0:
            return np.array([])

        white = np.random.randn(n)
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        scale = np.ones_like(freqs)
        nz = freqs > 0

        if np.any(nz):
            start_exp = self.exponent
            target_exp = self.high_exponent if self.high_exponent is not None else self.exponent
            min_f = freqs[nz].min()
            max_f = freqs[nz].max()

            if max_f == min_f:
                interp = np.zeros_like(freqs[nz])
            else:
                log_norm = np.log(freqs[nz] / min_f) / np.log(max_f / min_f)
                curve = max(self.distribution_curve, 1e-6)
                interp = log_norm ** curve

            exp_profile = start_exp + (target_exp - start_exp) * interp
            scale[nz] = freqs[nz] ** (-exp_profile / 2.0)

        scale[0] = 0
        noise = np.fft.irfft(fft_white * scale, n=n)
        max_abs = np.max(np.abs(noise))
        if max_abs > 1e-9:
            noise = noise / max_abs
        return noise


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

COLOR_PRESET_FILE = Path.home() / ".binauralbuilder" / "colored_noise_presets.json"


PRESET_FIELDS = {
    "exponent",
    "high_exponent",
    "distribution_curve",
    "lowcut",
    "highcut",
    "amplitude",
    "seed",
}

COLOR_PARAM_DEFAULTS: Dict[str, Any] = {
    "exponent": 1.0,
    "high_exponent": None,
    "distribution_curve": 1.0,
    "lowcut": None,
    "highcut": None,
    "amplitude": 1.0,
    "seed": 1,
}


def normalized_color_params(noise_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``params`` with all required colour fields populated."""

    merged = {**COLOR_PARAM_DEFAULTS, **(params or {})}
    exponent = merged.get("exponent", COLOR_PARAM_DEFAULTS["exponent"])
    if merged.get("high_exponent") is None:
        merged["high_exponent"] = exponent
    if merged.get("distribution_curve") is None:
        merged["distribution_curve"] = COLOR_PARAM_DEFAULTS["distribution_curve"]
    if merged.get("lowcut") is None:
        merged["lowcut"] = COLOR_PARAM_DEFAULTS["lowcut"]
    if merged.get("highcut") is None:
        merged["highcut"] = COLOR_PARAM_DEFAULTS["highcut"]
    if merged.get("amplitude") is None:
        merged["amplitude"] = COLOR_PARAM_DEFAULTS["amplitude"]
    if merged.get("seed") is None:
        merged["seed"] = COLOR_PARAM_DEFAULTS["seed"]
    if noise_type and not merged.get("name"):
        merged["name"] = noise_type
    return merged


def _color_preset(name: str, **params) -> Dict[str, object]:
    return {"name": name, "params": params}


DEFAULT_COLOR_PRESETS = {
    preset["name"]: normalized_color_params(preset["name"], preset["params"])
    for preset in (
        _color_preset("Pink", exponent=1.0, high_exponent=1.0),
        _color_preset("Brown", exponent=2.0, high_exponent=2.0),
        _color_preset("Green", exponent=0.0, high_exponent=0.0, lowcut=100.0, highcut=8000.0),
        _color_preset("Blue", exponent=-1.0, high_exponent=-1.0),
        _color_preset("Purple", exponent=-2.0, high_exponent=-2.0),
        _color_preset("Red", exponent=2.0, high_exponent=1.5),
        _color_preset("Deep Brown", exponent=2.5, high_exponent=2.0),
        _color_preset("White", exponent=0.0, high_exponent=0.0),
    )
}


def save_custom_color_presets(presets: Dict[str, Dict[str, object]]) -> None:
    """Persist user-defined color presets to disk."""

    COLOR_PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COLOR_PRESET_FILE, "w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=2)


def load_custom_color_presets() -> Dict[str, Dict[str, object]]:
    """Load user-defined color presets from disk if available."""

    if not COLOR_PRESET_FILE.is_file():
        return {}
    with open(COLOR_PRESET_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}


def generator_to_preset(generator: ColoredNoiseGenerator) -> Dict[str, object]:
    """Convert a :class:`ColoredNoiseGenerator` into a serializable dict."""

    return {key: getattr(generator, key) for key in PRESET_FIELDS}


def apply_preset_to_generator(generator: ColoredNoiseGenerator, preset: Dict[str, object]) -> ColoredNoiseGenerator:
    """Return a new generator with attributes populated from ``preset``."""

    kwargs = {**asdict(generator)}
    for key, value in preset.items():
        if key in PRESET_FIELDS:
            kwargs[key] = value
    return ColoredNoiseGenerator(**kwargs)


def plot_spectrogram(noise: np.ndarray, sample_rate: int, cmap: str = "viridis") -> None:
    """Display an interactive heatmap spectrogram of ``noise``.

    Scrolling the mouse wheel over the plot zooms the frequency axis,
    enabling inspection of different bands.
    """
    f, t, Sxx = spectrogram(noise, fs=sample_rate)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap=cmap)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    fig.colorbar(im, ax=ax, label="dB")
    ax.set_ylim(0, sample_rate / 2)

    def _on_scroll(event):
        if event.inaxes != ax or event.ydata is None:
            return
        cur_bottom, cur_top = ax.get_ylim()
        center = event.ydata
        scale = 1.2 if event.button == "up" else 1 / 1.2
        new_range = (cur_top - cur_bottom) * scale
        bottom = max(0, center - new_range / 2)
        top = min(sample_rate / 2, center + new_range / 2)
        ax.set_ylim(bottom, top)
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    plt.show(block=False)
    plt.pause(0.001)


__all__ = [
    "apply_preset_to_generator",
    "COLOR_PRESET_FILE",
    "ColoredNoiseGenerator",
    "DEFAULT_COLOR_PRESETS",
    "generator_to_preset",
    "load_custom_color_presets",
    "normalized_color_params",
    "plot_spectrogram",
    "save_custom_color_presets",
]

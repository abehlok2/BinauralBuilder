"""Colored noise generation and spectrogram visualization utilities."""

from dataclasses import dataclass
from typing import Optional

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
    seed: Optional[int] = None

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


__all__ = ["ColoredNoiseGenerator", "plot_spectrogram"]

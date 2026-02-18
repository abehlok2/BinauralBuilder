"""
Common utilities and helper functions used by synth functions.
"""
import numpy as np
import math
from scipy.signal import butter, lfilter, sosfiltfilt

try:  # numba is optional and may be unavailable on some platforms (e.g. Windows)
    import numba
except Exception:  # pragma: no cover
    numba = None

def sine_wave(freq, t, phase: float=0):
    """Generate a sine wave at a constant frequency using an absolute time base."""
    freq = float(freq)
    phase = float(phase)
    # Ensure freq is not zero or negative if used in log etc later, though not here.
    freq = np.maximum(freq, 1e-9) # Avoid potential issues with zero freq
    return np.sin(2 * np.pi * freq * t + phase)

def sine_wave_varying(freq_array, t, sample_rate=44100):
    """
    Generate a sine wave whose instantaneous frequency is defined by freq_array.
    The phase is computed as the cumulative sum (integral) of the frequency
    over the absolute time vector t. An initial phase is added so that for constant
    frequency the result is sin(2π f t).
    """
    # Ensure freq_array does not contain non-positive values before cumsum
    freq_array = np.maximum(np.asarray(freq_array, dtype=float), 1e-9)
    t = np.asarray(t, dtype=float)
    sample_rate = float(sample_rate)

    if len(t) <= 1:
        return np.zeros_like(t) # Handle edge case of single point or empty array

    # Calculate dt, ensuring it has the same length as t and freq_array
    dt = np.diff(t, prepend=t[0]) # Use first element to calculate first dt assuming t[0] is start time
    # Alternative if t doesn't start at 0: dt = np.diff(t, prepend=t[0] - (t[1]-t[0]) if len(t)>1 else 0)

    # Add an initial phase corresponding to integration from t=0 to t[0]
    initial_phase = 2 * np.pi * freq_array[0] * t[0] if len(t) > 0 else 0
    # Calculate phase using cumulative sum
    phase = initial_phase + 2 * np.pi * np.cumsum(freq_array * dt)
    return np.sin(phase)


if numba is not None:
    @numba.njit(fastmath=True, inline="always")
    def _frac(x):
        """Return the fractional part of x."""
        return x - np.floor(x)


    @numba.njit(fastmath=True, inline="always")
    def skewed_sine_phase(phase_fraction, skew):
        """Sine wave with adjustable up/down symmetry.

        ``phase_fraction`` should be in the range ``[0, 1)`` representing the
        position within one cycle. ``skew`` ranges from -1.0 (long upswing) to
        1.0 (short upswing). A value of 0.0 yields a standard sine shape.
        """
        frac = 0.5 + 0.5 * skew
        if frac <= 0.0:
            frac = 1e-9
        if frac >= 1.0:
            frac = 1.0 - 1e-9

        if phase_fraction < frac:
            local = phase_fraction / frac
            return math.sin(math.pi * local)
        else:
            local = (phase_fraction - frac) / (1.0 - frac)
            return math.sin(math.pi * (1.0 + local))

    @numba.njit(fastmath=True, inline="always")
    def skewed_triangle_phase(phase_fraction, skew):
        """Triangle wave with adjustable up/down symmetry."""
        frac = 0.5 + 0.5 * skew
        if frac <= 0.0:
            frac = 1e-9
        if frac >= 1.0:
            frac = 1.0 - 1e-9

        if phase_fraction < frac:
            local = phase_fraction / frac
            return -1.0 + 2.0 * local
        else:
            local = (phase_fraction - frac) / (1.0 - frac)
            return 1.0 - 2.0 * local
else:
    def _frac(x):
        """Return the fractional part of x."""
        return x - math.floor(x)


    def skewed_sine_phase(phase_fraction, skew):
        """Sine wave with adjustable up/down symmetry.

        ``phase_fraction`` should be in the range ``[0, 1)`` representing the
        position within one cycle. ``skew`` ranges from -1.0 (long upswing) to
        1.0 (short upswing). A value of 0.0 yields a standard sine shape.
        """
        frac = 0.5 + 0.5 * skew
        if frac <= 0.0:
            frac = 1e-9
        if frac >= 1.0:
            frac = 1.0 - 1e-9

        if phase_fraction < frac:
            local = phase_fraction / frac
            return math.sin(math.pi * local)
        else:
            local = (phase_fraction - frac) / (1.0 - frac)
            return math.sin(math.pi * (1.0 + local))

    def skewed_triangle_phase(phase_fraction, skew):
        """Triangle wave with adjustable up/down symmetry."""
        frac = 0.5 + 0.5 * skew
        if frac <= 0.0:
            frac = 1e-9
        if frac >= 1.0:
            frac = 1.0 - 1e-9

        if phase_fraction < frac:
            local = phase_fraction / frac
            return -1.0 + 2.0 * local
        else:
            local = (phase_fraction - frac) / (1.0 - frac)
            return 1.0 - 2.0 * local

def adsr_envelope(t, attack=0.01, decay=0.1, sustain_level=0.8, release=0.1):
    """
    Generate an ADSR envelope over t. Assumes that t starts at 0.
    """
    t = np.asarray(t, dtype=float)
    if len(t) <= 1: return np.zeros_like(t) # Handle edge case

    duration = t[-1] - t[0] + (t[1]-t[0]) if len(t) > 1 else 0
    sr = len(t) / duration if duration > 0 else 44100
    total_samples = len(t)

    # Ensure non-negative durations and valid sustain level
    attack = max(0, float(attack))
    decay = max(0, float(decay))
    release = max(0, float(release))
    sustain_level = np.clip(float(sustain_level), 0, 1)

    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)

    # Ensure samples are not negative
    attack_samples = max(0, attack_samples)
    decay_samples = max(0, decay_samples)
    release_samples = max(0, release_samples)

    # Adjust if total duration is less than ADSR times
    total_adsr_samples = attack_samples + decay_samples + release_samples
    if total_adsr_samples > total_samples and total_adsr_samples > 0:
        scale_factor = total_samples / total_adsr_samples
        attack_samples = int(attack_samples * scale_factor)
        decay_samples = int(decay_samples * scale_factor)
        # Ensure release always has at least one sample if requested > 0 and space allows
        release_samples = total_samples - attack_samples - decay_samples
        release_samples = max(0, release_samples)

    sustain_samples = total_samples - (attack_samples + decay_samples + release_samples)
    sustain_samples = max(0, sustain_samples) # Ensure sustain is not negative

    # Generate curves
    attack_curve = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
    decay_curve = np.linspace(1, sustain_level, decay_samples, endpoint=False) if decay_samples > 0 else np.array([])
    sustain_curve = np.full(sustain_samples, sustain_level) if sustain_samples > 0 else np.array([])
    # Ensure release ends exactly at 0
    release_curve = np.linspace(sustain_level, 0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

    # Concatenate, ensuring correct length
    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])

    # Pad or truncate to ensure exact length
    if len(env) < total_samples:
        # Pad with the end value of the release curve (should be 0)
        pad_value = 0.0 if release_samples > 0 else sustain_level # Pad with sustain if no release
        env = np.pad(env, (0, total_samples - len(env)), mode='constant', constant_values=pad_value)
    elif len(env) > total_samples:
        env = env[:total_samples]

    return env

def create_linear_fade_envelope(total_duration, sample_rate, fade_duration, start_amp, end_amp, fade_type='in'):
    """
    Create a linear fade envelope over a total duration.
    Handles fade_type 'in' (start) and 'out' (end).
    """
    total_duration = float(total_duration)
    sample_rate = int(sample_rate)
    fade_duration = float(fade_duration)
    start_amp = float(start_amp)
    end_amp = float(end_amp)

    total_samples = int(total_duration * sample_rate)
    fade_duration_samples = int(fade_duration * sample_rate)

    if total_samples <= 0: return np.array([])

    # Ensure fade duration isn't longer than total duration
    fade_samples = min(fade_duration_samples, total_samples)
    fade_samples = max(0, fade_samples) # Ensure non-negative

    if fade_samples <= 0: # No fade needed or possible
        # If fade in, hold end_amp; if fade out, hold start_amp
        hold_value = end_amp if fade_type == 'in' else start_amp
        return np.full(total_samples, hold_value)

    envelope = np.ones(total_samples) # Initialize

    if fade_type == 'in':
        # Generate ramp from start_amp to end_amp over fade_samples
        ramp = np.linspace(start_amp, end_amp, fade_samples, endpoint=True)
        envelope[:fade_samples] = ramp
        # Hold the end amplitude for the rest of the duration
        envelope[fade_samples:] = end_amp
    elif fade_type == 'out':
        sustain_samples = total_samples - fade_samples
        # Hold the start amplitude until the fade begins
        envelope[:sustain_samples] = start_amp
        # Generate ramp from start_amp to end_amp over fade_samples at the end
        ramp = np.linspace(start_amp, end_amp, fade_samples, endpoint=True)
        envelope[sustain_samples:] = ramp
    else:
        raise ValueError("fade_type must be 'in' or 'out'")

    return envelope

def linen_envelope(t, attack=0.1, release=0.1):
    """
    Generate a linear envelope that ramps up then down across t.
    Assumes t is relative (starts at 0).
    """
    t = np.asarray(t, dtype=float)
    if len(t) <= 1: return np.zeros_like(t) # Handle edge case

    duration = t[-1] - t[0] + (t[1]-t[0]) if len(t) > 1 else 0
    sr = len(t) / duration if duration > 0 else 44100
    total_samples = len(t)

    # Ensure non-negative durations
    attack = max(0, float(attack))
    release = max(0, float(release))

    attack_samples = int(attack * sr)
    release_samples = int(release * sr)

    # Ensure samples are not negative
    attack_samples = max(0, attack_samples)
    release_samples = max(0, release_samples)

    # Adjust if total duration is less than attack + release
    total_ar_samples = attack_samples + release_samples
    if total_ar_samples > total_samples and total_ar_samples > 0:
        scale_factor = total_samples / total_ar_samples
        attack_samples = int(attack_samples * scale_factor)
        # Ensure release gets remaining samples, minimum 0
        release_samples = max(0, total_samples - attack_samples)

    sustain_samples = total_samples - (attack_samples + release_samples)
    sustain_samples = max(0, sustain_samples)

    # Generate curves
    attack_curve = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
    sustain_curve = np.ones(sustain_samples) if sustain_samples > 0 else np.array([])
    # Ensure release ends exactly at 0
    release_curve = np.linspace(1, 0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

    # Concatenate
    env = np.concatenate([attack_curve, sustain_curve, release_curve])

    # Pad or truncate
    if len(env) < total_samples:
        # Pad with end value (should be 0 if release exists)
        pad_value = 0.0 if release_samples > 0 else 1.0
        env = np.pad(env, (0, total_samples - len(env)), mode='constant', constant_values=pad_value)
    elif len(env) > total_samples:
        env = env[:total_samples]

    return env

def pan2(signal, pan=0):
    """
    Pan a mono signal into stereo.
    pan: -1 => full left, 0 => center, 1 => full right.
    Uses simple sine/cosine panning (equal power).
    """
    pan = float(pan)
    pan = np.clip(pan, -1, 1) # Ensure pan is within range
    # Map pan [-1, 1] to angle [0, pi/2]
    angle = (pan + 1) * np.pi / 4
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    left = left_gain * signal
    right = right_gain * signal
    # Ensure output is 2D array (N, 2)
    if signal.ndim == 1:
        return np.vstack([left, right]).T
    elif signal.ndim == 2 and signal.shape[1] == 1: # If input is already (N, 1)
        return np.hstack([left, right])
    else: # If stereo input is supplied, fall back to using the first channel.
        print("Warning: pan2 received unexpected input shape. Assuming mono.")
        if signal.ndim == 2 and signal.shape[1] > 1:
            signal = signal[:, 0]
            left = left_gain * signal
            right = right_gain * signal
        return np.vstack([left, right]).T


def _ensure_1d_length(value, length, dtype=np.float64):
    """Return a 1-D array of ``length`` from ``value``.

    Scalars are broadcast while 1-D arrays must already match the requested
    length.  This helper keeps the pan utilities resilient to scalar or
    per-sample modulation inputs.
    """
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(length, float(arr), dtype=dtype)
    arr = np.ravel(arr).astype(dtype, copy=False)
    if arr.size != length:
        raise ValueError(f"Expected array of length {length}, received shape {arr.shape}.")
    return arr


def generate_pan_envelope(
    num_samples,
    sample_rate,
    pan_base,
    pan_range_min,
    pan_range_max,
    pan_freq,
    pan_type="linear",
    initial_phase=0.0,
):
    """Create a per-sample pan curve in the range [-1, 1].

    ``pan_base`` provides a static offset (used primarily when ``pan_freq`` is
    zero) while ``pan_range_min``/``pan_range_max`` describe the dynamic
    excursion limits. ``pan_type`` selects how the motion traverses the range:

    ``linear``
        Sinusoidal motion that smoothly oscillates between the configured
        minimum and maximum.
    ``inward``
        Alternating ramps that start at either extremum and move toward the
        center (zero).
    ``outward``
        Alternating ramps that begin at the center and move toward the
        configured extrema.

    Each of these parameters can be a scalar or an array of ``num_samples``
    values to support transition voices.
    """
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    base_arr = _ensure_1d_length(pan_base, num_samples, dtype=np.float32)
    range_min_arr = _ensure_1d_length(pan_range_min, num_samples, dtype=np.float32)
    range_max_arr = _ensure_1d_length(pan_range_max, num_samples, dtype=np.float32)
    freq_arr = _ensure_1d_length(pan_freq, num_samples, dtype=np.float32)
    phase_arr = _ensure_1d_length(initial_phase, num_samples, dtype=np.float32)

    range_low = np.minimum(range_min_arr, range_max_arr)
    range_high = np.maximum(range_min_arr, range_max_arr)

    if sample_rate <= 0.0:
        return np.clip(base_arr, -1.0, 1.0)

    if np.all(freq_arr <= 0.0):
        # No modulation requested – prefer the configured range (if any),
        # otherwise fall back to the static pan control.
        if np.any(range_low != range_high):
            center = 0.5 * (range_high + range_low)
            return np.clip(center, -1.0, 1.0)
        if np.any(range_low != base_arr):
            return np.clip(range_low, -1.0, 1.0)
        return np.clip(base_arr, -1.0, 1.0)

    # Ensure positive frequencies (negative values would simply invert the
    # traversal direction; treat them as their absolute magnitude).
    freq_arr = np.abs(freq_arr)
    dt = np.float32(1.0 / float(sample_rate))
    freq_prefix = np.concatenate((np.asarray([0.0], dtype=np.float32), freq_arr[:-1]))
    cycles = np.cumsum(freq_prefix, dtype=np.float32)
    cycles *= dt

    pan_type_key = str(pan_type).strip().lower()
    if pan_type_key not in {"linear", "inward", "outward"}:
        pan_type_key = "linear"

    if pan_type_key == "linear":
        two_pi = np.float32(2.0 * np.pi)
        phase = phase_arr + two_pi * cycles
        lfo = np.sin(phase, dtype=np.float32)
        center = np.float32(0.5) * (range_high + range_low)
        half_span = np.float32(0.5) * (range_high - range_low)
        pan_values = center + half_span * lfo
    else:
        two_pi = np.float32(2.0 * np.pi)
        phase_cycles = (phase_arr / two_pi) + cycles
        cycle_index = np.floor(phase_cycles).astype(np.int64)
        cycle_pos = phase_cycles - cycle_index

        even_mask = (cycle_index & 1) == 0
        pan_values = np.empty(num_samples, dtype=np.float32)

        if pan_type_key == "inward":
            start_even = range_high
            start_odd = range_low
            pan_values[even_mask] = start_even[even_mask] + (
                0.0 - start_even[even_mask]
            ) * cycle_pos[even_mask]
            pan_values[~even_mask] = start_odd[~even_mask] + (
                0.0 - start_odd[~even_mask]
            ) * cycle_pos[~even_mask]
        else:  # outward
            target_even = range_high
            target_odd = range_low
            pan_values[even_mask] = 0.0 + (
                target_even[even_mask] - 0.0
            ) * cycle_pos[even_mask]
            pan_values[~even_mask] = 0.0 + (
                target_odd[~even_mask] - 0.0
            ) * cycle_pos[~even_mask]

    return np.clip(pan_values, -1.0, 1.0).astype(np.float32, copy=False)


def apply_pan_envelope(audio, pan_values):
    """Apply an equal-power pan envelope to a stereo signal.

    ``pan_values`` can be a scalar or a vector of length ``len(audio)``.
    ``audio`` is modified in-place and also returned for convenience.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("apply_pan_envelope expects an array of shape (N, 2).")

    pan_arr = np.asarray(pan_values, dtype=audio.dtype)
    if pan_arr.ndim == 0:
        pan_arr = np.full(audio.shape[0], float(pan_arr), dtype=audio.dtype)
    elif pan_arr.ndim == 1 and pan_arr.shape[0] == audio.shape[0]:
        pan_arr = pan_arr.astype(audio.dtype, copy=False)
    else:
        raise ValueError("Pan envelope length must match the number of samples.")

    dtype = audio.dtype
    one = dtype.type(1.0)
    half_pi = dtype.type(np.pi / 2.0)
    pan_clipped = np.clip(pan_arr, -one, one)
    angle = (pan_clipped + one) * (half_pi * dtype.type(0.5))

    left_gain = np.cos(angle).astype(audio.dtype, copy=False)
    right_gain = np.sin(angle).astype(audio.dtype, copy=False)
    audio[:, 0] *= left_gain
    audio[:, 1] *= right_gain

    return audio

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Ensure lowcut < highcut and both are positive and less than Nyquist
    lowcut = max(1e-6, float(lowcut)) # Avoid zero/negative freq, ensure float
    highcut = max(lowcut + 1e-6, float(highcut)) # Ensure high > low, ensure float
    # Clip frequencies to be slightly below Nyquist
    lowcut = min(lowcut, nyq * 0.9999)
    highcut = min(highcut, nyq * 0.9999)

    low = lowcut / nyq
    high = highcut / nyq
    # Ensure low < high after normalization, prevent edge cases
    if low >= high:
        low = high * 0.999 # Adjust if rounding caused issues
    low = max(1e-9, low) # Ensure > 0
    high = min(1.0 - 1e-9, high) # Ensure < 1

    # Check for invalid frequency range after clipping
    if low >= high:
         raise ValueError(f"Bandpass filter frequency range is invalid after Nyquist clipping: low={low}, high={high} (normalized)")

    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, center, Q, fs):
    """
    Apply a bandpass filter to the data. Q = center/bandwidth.
    """
    center = float(center)
    Q = float(Q)
    fs = float(fs)
    if Q <= 0: Q = 0.1 # Avoid zero or negative Q
    if center <= 0: center = 1.0 # Avoid zero or negative center frequency

    bandwidth = center / Q
    lowcut = center - bandwidth / 2
    highcut = center + bandwidth / 2
    # Ensure lowcut is positive
    lowcut = max(1e-6, lowcut)
    if lowcut >= highcut: # Handle cases where bandwidth is too large or center too low
        # Recalculate highcut based on positive lowcut and Q factor
        highcut = lowcut * (1 + 1/Q) # Simple approximation
        highcut = max(highcut, lowcut + 1e-6) # Ensure distinct

    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=2) # Lower order often more stable
        return lfilter(b, a, data)
    except ValueError as e:
        print(f"Error in bandpass filter calculation: {e}")
        print(f"  center={center}, Q={Q}, fs={fs} -> lowcut={lowcut}, highcut={highcut}")
        # Return original data or zeros? Returning original for now.
        return data

def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Input validation similar to bandpass
    lowcut = max(1e-6, float(lowcut))
    highcut = max(lowcut + 1e-6, float(highcut))
    lowcut = min(lowcut, nyq * 0.9999)
    highcut = min(highcut, nyq * 0.9999)

    low = lowcut / nyq
    high = highcut / nyq
    if low >= high: low = high * 0.999
    low = max(1e-9, low)
    high = min(1.0 - 1e-9, high)

    if low >= high:
         raise ValueError(f"Bandstop filter frequency range is invalid after Nyquist clipping: low={low}, high={high} (normalized)")

    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def bandreject_filter(data, center, Q, fs):
    """
    Apply a band-reject (notch) filter.
    """
    center = float(center)
    Q = float(Q)
    fs = float(fs)
    if Q <= 0: Q = 0.1
    if center <= 0: center = 1.0

    bandwidth = center / Q
    lowcut = center - bandwidth / 2
    highcut = center + bandwidth / 2
    lowcut = max(1e-6, lowcut)
    if lowcut >= highcut:
        highcut = lowcut * (1 + 1/Q)
        highcut = max(highcut, lowcut + 1e-6)

    try:
        b, a = butter_bandstop(lowcut, highcut, fs, order=2)
        return lfilter(b, a, data)
    except ValueError as e:
        print(f"Error in bandreject filter calculation: {e}")
        print(f"  center={center}, Q={Q}, fs={fs} -> lowcut={lowcut}, highcut={highcut}")
        return data

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    # Ensure cutoff is positive and below Nyquist
    cutoff = max(1e-6, float(cutoff))
    cutoff = min(cutoff, nyq * 0.9999)
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def pink_noise(n):
    """
    Generate approximate pink noise using filtering method (more common).
    Filters white noise with a filter whose magnitude response is proportional to 1/sqrt(f).
    """
    n = int(n)
    if n <= 0:
        return np.array([])

    # Generate white noise and shape it in the frequency domain so that the
    # magnitude response is proportional to 1/sqrt(f).
    white_noise = np.random.randn(n)
    fft_white = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(n, d=1.0 / 44100)  # Use sample rate if known, else d=1.0

    # Create filter: amplitude proportional to 1/sqrt(f)
    # Handle f=0 case (DC component)
    scale = np.ones_like(frequencies)
    non_zero_freq_indices = frequencies > 0
    # Avoid division by zero or sqrt of zero
    valid_freqs = frequencies[non_zero_freq_indices]
    scale[non_zero_freq_indices] = 1.0 / np.sqrt(np.maximum(valid_freqs, 1e-9))
    # Remove DC component
    scale[0] = 0

    # Apply filter in frequency domain
    pink_spectrum = fft_white * scale

    # Inverse FFT to get time domain signal
    pink = np.fft.irfft(pink_spectrum, n=n)

    # Normalize to approx [-1, 1]
    max_abs = np.max(np.abs(pink))
    if max_abs > 1e-9:
        pink = pink / max_abs
    return pink

def brown_noise(n):
    """
    Generate brown (red) noise by cumulatively summing white noise.
    """
    n = int(n)
    if n <= 0: return np.array([])
    wn = np.random.randn(n)
    brown = np.cumsum(wn)
    # Normalize to prevent excessive drift/amplitude
    max_abs = np.max(np.abs(brown))
    return brown / max_abs if max_abs > 1e-9 else brown


def _powerlaw_noise(beta, n):
    """Generate noise with power spectral density proportional to 1/f**beta."""
    n = int(n)
    if n <= 0:
        return np.array([])
    white = np.random.randn(n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / 44100)
    scale = np.ones_like(freqs)
    nz = freqs > 0
    scale[nz] = freqs[nz] ** (-beta / 2.0)
    scale[0] = 0
    noise = np.fft.irfft(fft_white * scale, n=n)
    max_abs = np.max(np.abs(noise))
    if max_abs > 1e-9:
        noise = noise / max_abs
    return noise


def red_noise(n):
    """Alias for brown noise (1/f^2 spectrum)."""
    return _powerlaw_noise(2, n)


def deep_brown_noise(n):
    """Deeper variant of brown noise with 1/f^3 spectrum."""
    return _powerlaw_noise(3, n)


def blue_noise(n):
    """Generate blue noise with rising spectrum (1/f^-1)."""
    return _powerlaw_noise(-1, n)


def purple_noise(n):
    """Generate purple (violet) noise with 1/f^-2 spectrum."""
    return _powerlaw_noise(-2, n)


def green_noise(n, fs=44100):
    """Generate green noise by emphasizing the mid-frequency band around 500 Hz."""
    noise = _powerlaw_noise(1, n)
    low = 100 / (fs * 0.5)
    high = 1000 / (fs * 0.5)
    b, a = butter(4, [low, high], btype="band")
    filtered = lfilter(b, a, noise)
    max_abs = np.max(np.abs(filtered))
    if max_abs > 1e-9:
        filtered = filtered / max_abs
    return filtered

# --- NEW: Helper for Isochronic Tones (Copied from audio_engine.py) ---
def trapezoid_envelope_vectorized(t_in_cycle, cycle_len, ramp_percent, gap_percent):
    """Vectorized trapezoidal envelope generation."""
    env = np.zeros_like(t_in_cycle, dtype=np.float64)
    valid_mask = cycle_len > 0
    if not np.any(valid_mask): return env

    audible_len = (1.0 - gap_percent) * cycle_len
    ramp_total = np.clip(audible_len * ramp_percent * 2.0, 0.0, audible_len) # Ensure ramp doesn't exceed audible
    stable_len = audible_len - ramp_total
    ramp_up_len = ramp_total / 2.0
    stable_end = ramp_up_len + stable_len

    # Masks
    in_gap_mask = (t_in_cycle >= audible_len) & valid_mask
    ramp_up_mask = (t_in_cycle < ramp_up_len) & (~in_gap_mask) & valid_mask
    ramp_down_mask = (t_in_cycle >= stable_end) & (t_in_cycle < audible_len) & (~in_gap_mask) & valid_mask
    stable_mask = (t_in_cycle >= ramp_up_len) & (t_in_cycle < stable_end) & (~in_gap_mask) & valid_mask

    # Calculations with division checks
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ramp up
        div_ramp_up = ramp_up_len[ramp_up_mask]
        ramp_up_vals = np.full_like(div_ramp_up, 0.0)
        valid_div_up = div_ramp_up > 0
        ramp_up_vals[valid_div_up] = t_in_cycle[ramp_up_mask][valid_div_up] / div_ramp_up[valid_div_up]
        env[ramp_up_mask] = np.nan_to_num(ramp_up_vals)

        # Stable
        env[stable_mask] = 1.0

        # Ramp down
        time_into_down = (t_in_cycle[ramp_down_mask] - stable_end[ramp_down_mask])
        down_len = ramp_up_len[ramp_down_mask] # Should be same as ramp up length
        ramp_down_vals = np.full_like(down_len, 0.0)
        valid_div_down = down_len > 0
        ramp_down_vals[valid_div_down] = 1.0 - (time_into_down[valid_div_down] / down_len[valid_div_down])
        env[ramp_down_mask] = np.nan_to_num(ramp_down_vals)

    return np.clip(env, 0.0, 1.0) # Ensure output is [0, 1]


# --- Filter design and application (from your qam.py, not Numba compatible) ---
def design_filter(filter_type, cutoff, fs, order=4):
    """
    Designs a Butterworth filter.
    Args:
        filter_type (str): 'highpass', 'lowpass', 'bandpass', 'bandstop'.
        cutoff (float or list): Cutoff frequency or frequencies.
        fs (float): Sampling frequency.
        order (int): Filter order.
    Returns:
        ndarray: Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs
    norm_cutoff = np.array(cutoff) / nyq if isinstance(cutoff, (list, tuple)) else cutoff / nyq
    return butter(order, norm_cutoff, btype=filter_type, output='sos')

def apply_filters(signal_segment, fs):
    """
    Applies pre-defined high-pass and low-pass filters to a signal segment.
    Args:
        signal_segment (np.ndarray): Input signal (1D).
        fs (float): Sampling frequency.
    Returns:
        np.ndarray: Filtered signal.
    """
    if signal_segment.ndim == 0 or signal_segment.size == 0:
        return signal_segment # Return empty or scalar if no data to filter
    if fs <= 0: # Invalid sampling rate
        return signal_segment
    
    # Define cutoff frequencies ensuring they are valid
    hp_cutoff = 30.0
    lp_cutoff = fs * 0.5 * 0.9 

    if hp_cutoff <= 0 or hp_cutoff >= fs / 2: # Invalid high-pass cutoff
        pass # Skip high-pass if invalid
    else:
        sos_hp = design_filter('highpass', hp_cutoff, fs)
        signal_segment = sosfiltfilt(sos_hp, signal_segment)

    if lp_cutoff <= 0 or lp_cutoff >= fs / 2: # Invalid low-pass cutoff
        pass # Skip low-pass if invalid
    else:
        sos_lp = design_filter('lowpass', lp_cutoff, fs)
        signal_segment = sosfiltfilt(sos_lp, signal_segment)
        
    return signal_segment


def calculate_transition_alpha(
    total_duration,
    sample_rate,
    initial_offset=0.0,
    duration=None,
    curve="linear",
):
    """Create an interpolation factor array taking start/end offsets into account.

    Args:
        total_duration (float): Length of the transition in seconds.
        sample_rate (float): Sampling rate of the generated audio.
        initial_offset (float): Time before the transition begins.
        duration (float, optional): Length of the transition itself. If ``None``
            the transition lasts for the remainder of ``total_duration`` after
            ``initial_offset``.
        curve (str): Name of the transition curve to apply. Supported values are
            ``"linear"`` (default), ``"logarithmic"``, and ``"exponential"``.
            Any other string is interpreted as a Python expression using
            ``alpha`` as the linear ramp (0-1). ``numpy`` can be accessed via
            ``np`` and ``math`` for more advanced shapes.

    Returns:
        np.ndarray: Array of interpolation factors in the range [0, 1].
    """
    total_duration = float(total_duration)
    sample_rate = float(sample_rate)
    initial_offset = min(float(initial_offset), total_duration)

    if duration is None:
        duration = total_duration - initial_offset
    else:
        duration = max(0.0, float(duration))

    max_duration = max(0.0, total_duration - initial_offset)
    duration = min(duration, max_duration)

    N = int(total_duration * sample_rate)
    if N <= 0:
        return np.zeros(0, dtype=np.float64)

    t = np.linspace(0.0, total_duration, N, endpoint=False)

    start_t = initial_offset
    end_t = min(total_duration, start_t + duration)
    trans_time = end_t - start_t

    if trans_time <= 0.0:
        alpha = np.zeros(N, dtype=np.float64)
        if start_t < total_duration:
            alpha[t >= start_t] = 1.0
        return alpha

    alpha = (t - start_t) / trans_time
    alpha = np.clip(alpha, 0.0, 1.0)

    if curve == "linear":
        pass  # already linear
    elif curve == "logarithmic":
        alpha = 1.0 - np.power(1.0 - alpha, 2.0)
    elif curve == "exponential":
        alpha = np.power(alpha, 2.0)
    else:
        try:
            alpha = eval(str(curve), {"np": np, "math": math}, {"alpha": alpha})
        except Exception as exc:
            raise ValueError(
                f"Invalid transition curve expression '{curve}': {exc}"
            ) from exc

    return alpha


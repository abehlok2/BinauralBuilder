import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.io.wavfile import write
import soundfile as sf
import json
import copy

import inspect # Needed to inspect function parameters for GUI
import os # Needed for path checks in main example
import traceback # For detailed error printing
import time
import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from src.utils.noise_file import load_noise_params

# Maximum individual gain for binaural/noise to prevent clipping when combined.
# With both at max (0.48 + 0.48 = 0.96), the combined output stays under 1.0.
MAX_INDIVIDUAL_GAIN = 0.48
from src.synth_functions.noise_flanger import (
    _generate_swept_notch_arrays,
    _generate_swept_notch_arrays_transition,
    noise_swept_notch,
    noise_swept_notch_transition,
)
from src.synth_functions.fx_flanger import flanger_stereo


# Import all synth functions from the synth_functions package
from src.synth_functions import *

# Placeholder for the missing audio_engine module
# If you have the 'audio_engine.py' file, place it in the same directory.
# Otherwise, the SAM functions will not work.
try:
    # Attempt to import the real audio_engine if available
    from .audio_engine import Node, SAMVoice, VALID_SAM_PATHS
    AUDIO_ENGINE_AVAILABLE = True
    print("INFO: audio_engine module loaded successfully.")
except Exception:
    AUDIO_ENGINE_AVAILABLE = False
    print("WARNING: audio_engine module not found. Spatial Angle Modulation (SAM) functions will not be available.")
    # Define dummy classes/variables if audio_engine is missing
    class Node:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy Node class. SAM functionality disabled.")
            # Store args needed for generate_samples duration calculation
            # Simplified: Just store duration if provided
            self.duration = args[0] if args else kwargs.get('duration', 0)
            pass
    class SAMVoice:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy SAMVoice class. SAM functionality disabled.")
            # Store args needed for generate_samples duration calculation
            self._nodes = kwargs.get('nodes', [])
            self._sample_rate = kwargs.get('sample_rate', 44100)
            pass
        def generate_samples(self):
            print("WARNING: SAM generate_samples called on dummy class. Returning silence.")
            # Calculate duration from stored nodes
            duration = 0
            if hasattr(self, '_nodes'):
                # Access duration attribute correctly from dummy Node
                duration = sum(node.duration for node in self._nodes if hasattr(node, 'duration'))
            sample_rate = getattr(self, '_sample_rate', 44100)
            N = int(duration * sample_rate) if duration > 0 else int(1.0 * sample_rate) # Default 1 sec if no duration found
            return np.zeros((N, 2))

    VALID_SAM_PATHS = ['circle', 'line', 'lissajous', 'figure_eight', 'arc'] # Example paths

# -----------------------------------------------------------------------------
# Crossfade and Assembly Logic
# -----------------------------------------------------------------------------

def crossfade_signals(signal_a, signal_b, sample_rate, transition_duration,
                      curve="linear", *, phase_align=False):
    """
    Crossfades two stereo signals. ``signal_a`` fades out and ``signal_b`` fades
    in over ``transition_duration``.  The ``curve`` argument selects the fade
    shape:

    ``linear``
        Linear ramps for both signals.
    ``equal_power``
        Uses a sine/cosine law to maintain approximately constant perceived
        loudness.

    If ``phase_align`` is True, ``signal_b`` is first aligned to the phase of
    ``signal_a`` using :func:`phase_align_signal` before the crossfade is
    applied.

    Returns the blended stereo segment.
    """
    n_samples = int(transition_duration * sample_rate)
    if n_samples <= 0:
        # No crossfade duration, return silence or handle appropriately
        return np.zeros((0, 2))

    # Determine the actual number of samples available for crossfade
    len_a = signal_a.shape[0]
    len_b = signal_b.shape[0]
    actual_crossfade_samples = min(n_samples, len_a, len_b)

    if actual_crossfade_samples <= 0:
        print(f"Warning: Crossfade not possible or zero length. Samples: {n_samples}, SigA: {len_a}, SigB: {len_b}")
        # Return an empty array matching the expected dimensions if no crossfade happens
        return np.zeros((0, 2))

    # Ensure signals are 2D stereo (N, 2) before slicing
    def ensure_stereo(sig):
        if sig.ndim == 1: sig = np.column_stack((sig, sig)) # Mono to Stereo
        elif sig.shape[1] == 1: sig = np.column_stack((sig[:,0], sig[:,0])) # (N, 1) to (N, 2)
        if sig.shape[1] != 2: raise ValueError("Signal must be stereo (N, 2) for crossfade.")
        return sig

    try:
        signal_a = ensure_stereo(signal_a)
        signal_b = ensure_stereo(signal_b)
    except ValueError as e:
        print(f"Error in crossfade_signals: {e}")
        return np.zeros((0, 2)) # Return empty on error

    # Take only the required number of samples for crossfade
    signal_a_seg = signal_a[:actual_crossfade_samples]
    signal_b_seg = signal_b[:actual_crossfade_samples]

    if phase_align:
        signal_b_seg = phase_align_signal(signal_a_seg, signal_b_seg,
                                          actual_crossfade_samples)

    if curve == "equal_power":
        theta = np.linspace(0.0, np.pi / 2, actual_crossfade_samples)[:, None]
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)
    else:  # default to linear
        fade_out = np.linspace(1, 0, actual_crossfade_samples)[:, None]
        fade_in = np.linspace(0, 1, actual_crossfade_samples)[:, None]

    # Apply fades and sum
    blended_segment = signal_a_seg * fade_out + signal_b_seg * fade_in
    return blended_segment


def phase_align_signal(prev_tail, next_audio, max_search_samples=2048):
    """Shift ``next_audio`` so its start aligns in phase with ``prev_tail``.

    The function uses cross-correlation on a small window from the end of
    ``prev_tail`` and the beginning of ``next_audio`` to estimate the best
    alignment.  ``next_audio`` is circularly shifted by the detected offset and
    returned.  If the overlap is too short for analysis the audio is returned
    unchanged.
    """

    n = min(len(prev_tail), len(next_audio), max_search_samples)
    if n <= 1:
        return next_audio

    tail = prev_tail[-n:]
    head = next_audio[:n]

    # Calculate cross-correlation for each stereo channel and sum
    corr_l = np.correlate(tail[:, 0], head[:, 0], mode="full")
    corr_r = np.correlate(tail[:, 1], head[:, 1], mode="full")
    corr = corr_l + corr_r

    offset = int(np.argmax(corr) - (n - 1))
    if offset == 0:
        return next_audio

    # Circular shift the entire segment so the first sample continues the phase
    return np.roll(next_audio, offset, axis=0)


def create_linear_fade_envelope(duration, sample_rate, start_amp=1.0, end_amp=1.0, fade_duration=0.05):
    """
    Creates a linear fade envelope.
    """
    N = int(duration * sample_rate)
    if N <= 0:
        return np.array([])
        
    env = np.full(N, float(end_amp))
    
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples > 0:
        fade_samples = min(fade_samples, N)
        # Fade in from start_amp to end_amp
        fade_curve = np.linspace(float(start_amp), float(end_amp), fade_samples)
        env[:fade_samples] = fade_curve
        
    return env


def _flanger_effect_stereo_continuous(
    audio: np.ndarray,
    sample_rate: float,
    duration: float,
    initial_offset: float,
    transition_duration,
    curve: str,
    start_params: dict,
    end_params: dict,
    enable_start: bool,
    enable_end: bool,
):
    """Apply a stereo flanger where parameters transition from ``start_params``
    to ``end_params`` over the voice's duration.

    Parameters
    ----------
    audio : np.ndarray
        Stereo audio array to process.
    sample_rate : float
        Sample rate of the audio.
    duration : float
        Duration of the voice in seconds.
    initial_offset : float
        Silence duration before the active transition portion begins.
    transition_duration : float or None
        Length of the active transition portion. ``None`` means the transition
        spans the remaining voice duration after ``initial_offset``.
    curve : str
        Name of the interpolation curve (e.g. ``"linear"``).
    start_params, end_params : dict
        Dictionaries containing flanger parameters compatible with
        :func:`flanger_stereo`.
    enable_start, enable_end : bool
        Whether the flanger is enabled at the start and end of the transition.
    """

    from .common import calculate_transition_alpha

    N = audio.shape[0]
    alpha = calculate_transition_alpha(
        duration,
        sample_rate,
        initial_offset,
        transition_duration,
        curve,
    )
    if len(alpha) != N:
        alpha = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(alpha)), alpha)

    dry_audio = audio.astype(np.float32)
    if enable_start:
        start_audio = flanger_stereo(dry_audio, sample_rate, **start_params)
    else:
        start_audio = dry_audio

    if enable_end:
        end_audio = flanger_stereo(dry_audio, sample_rate, **end_params)
    else:
        end_audio = dry_audio

    alpha = alpha[:, None]
    return (start_audio * (1.0 - alpha) + end_audio * alpha).astype(np.float32)


def load_audio_clip(file_path, sample_rate):
    """Load an audio clip as stereo ``float32`` at ``sample_rate``.

    Supports WAV/FLAC using :mod:`soundfile` and MP3 via :mod:`pydub` if
    available.  If ``librosa`` is installed the audio is resampled using it,
    otherwise a simple interpolation fallback is used.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return np.zeros((0, 2), dtype=np.float32)

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".mp3":
            try:
                from pydub import AudioSegment
            except Exception:
                print("Error: pydub not installed. Cannot load MP3 files.")
                return np.zeros((0, 2), dtype=np.float32)

            seg = AudioSegment.from_file(file_path)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples())
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels))
            else:
                samples = samples.reshape((-1, 1))
            data = samples.astype(np.float32) / float(1 << (8 * seg.sample_width - 1))
        else:
            data, sr = sf.read(file_path, always_2d=True, dtype="float32")
    except Exception as e:
        print(f"Error loading audio file '{file_path}': {e}")
        traceback.print_exc()
        return np.zeros((0, 2), dtype=np.float32)

    # Convert to stereo
    if data.ndim == 1:
        data = np.column_stack((data, data))
    elif data.shape[1] == 1:
        data = np.column_stack((data[:, 0], data[:, 0]))
    elif data.shape[1] > 2:
        data = data[:, :2]

    # Resample if needed
    if sr != sample_rate:
        try:
            if librosa is not None:
                data = librosa.resample(data.T, orig_sr=sr, target_sr=sample_rate).T
            else:
                old_n = data.shape[0]
                new_n = int(old_n * sample_rate / sr)
                x_old = np.linspace(0, 1, old_n, endpoint=False)
                x_new = np.linspace(0, 1, new_n, endpoint=False)
                data = np.vstack([
                    np.interp(x_new, x_old, data[:, ch]) for ch in range(2)
                ]).T
        except Exception as e:
            print(f"Error resampling '{file_path}': {e}")
            traceback.print_exc()
            return np.zeros((0, 2), dtype=np.float32)

    return data.astype(np.float32)


# Dictionary mapping function names (strings) to actual functions
# --- UPDATED SYNTH_FUNCTIONS DICTIONARY ---
# Exclude helper/internal functions explicitly
_EXCLUDED_FUNCTION_NAMES = [
    'validate_float', 'validate_int', 'butter_bandpass', 'bandpass_filter',
    'butter_bandstop', 'bandreject_filter', 'lowpass_filter', 'pink_noise',
    'brown_noise', 'sine_wave', 'sine_wave_varying', 'adsr_envelope',
    'create_linear_fade_envelope', 'linen_envelope', 'pan2', 'safety_limiter',
    'crossfade_signals', 'phase_align_signal', 'steps_have_continuous_voices',
    'load_audio_clip',
    'assemble_track_from_data', 'generate_voice_audio',
    'load_track_from_json', 'save_track_to_json', 'generate_audio', 'generate_wav', 'get_synth_params',
    'trapezoid_envelope_vectorized', '_flanger_effect_stereo_continuous',
    'butter', 'lfilter', 'write', 'ensure_stereo', 'apply_filters', 'design_filter', 
    'calculate_transition_alpha',
    # Standard library functions that might be imported
    'json', 'inspect', 'os', 'traceback', 'math', 'copy', 'pkgutil', 'importlib'
]

SYNTH_FUNCTIONS = {}


def _import_module(module_name):
    """Import ``module_name`` if it is available."""

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: PIE786 - we want to surface the module name that failed
        print(f"Warning: Failed to import synth module '{module_name}': {exc}")
        return None


def _discover_synth_modules():
    """Yield imported synth modules within ``src.synth_functions``."""

    base_package_names = ["src.synth_functions"]
    for base_name in base_package_names:
        package = _import_module(base_name)
        if package is None:
            continue

        yield package

        package_path = getattr(package, "__path__", None)
        if not package_path:
            continue

        prefix = package.__name__ + "."
        for module_info in pkgutil.walk_packages(package_path, prefix):
            # Ignore private/dunder modules
            if module_info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            module = _import_module(module_info.name)
            if module is not None:
                yield module


try:
    for module in _discover_synth_modules():
        for name, obj in inspect.getmembers(module):
            if not inspect.isfunction(obj):
                continue
            if name in _EXCLUDED_FUNCTION_NAMES or name.startswith("_"):
                continue
            if not obj.__module__.startswith(("src.synth_functions", "synth_functions")):
                continue

            try:
                parameters = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                # Unable to introspect signature; skip to avoid surfacing helpers.
                continue

            if "duration" not in parameters or "sample_rate" not in parameters:
                continue

            SYNTH_FUNCTIONS[name] = obj
except Exception as e:
    print(f"Error inspecting functions: {e}")

print(f"Detected Synth Functions: {list(SYNTH_FUNCTIONS.keys())}")

# Explicitly register noise functions if missed by discovery
if "noise_swept_notch" not in SYNTH_FUNCTIONS:
    SYNTH_FUNCTIONS["noise_swept_notch"] = noise_swept_notch
if "noise_swept_notch_transition" not in SYNTH_FUNCTIONS:
    SYNTH_FUNCTIONS["noise_swept_notch_transition"] = noise_swept_notch_transition


def get_synth_params(func_name):
    """Gets parameter names and default values for a synth function by inspecting its signature."""
    if func_name not in SYNTH_FUNCTIONS:
        print(f"Warning: Function '{func_name}' not found in SYNTH_FUNCTIONS.")
        return {}

    func = SYNTH_FUNCTIONS[func_name]
    params = {}
    try:
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            # Skip standard args and the catch-all kwargs
            if name in ['duration', 'sample_rate'] or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Store the default value if it exists, otherwise store inspect._empty
            params[name] = param.default # Keep _empty to distinguish from None default

    except Exception as e:
        print(f"Error inspecting signature for '{func_name}': {e}")
        # Fallback to trying source code parsing if signature fails? Or just return empty?
        # For now, return empty on inspection error. Source parsing is done in GUI.
        return {}

    return params


def steps_have_continuous_voices(prev_step, next_step):
    """Return True if ``next_step`` appears to continue the same voices as
    ``prev_step``.  The heuristic simply compares synth function names and
    parameter dictionaries for voices with the same index."""

    voices_a = prev_step.get("voices", []) if prev_step else []
    voices_b = next_step.get("voices", []) if next_step else []
    if len(voices_a) != len(voices_b):
        return False

    for v_a, v_b in zip(voices_a, voices_b):
        if v_a.get("synth_function_name") != v_b.get("synth_function_name"):
            return False
        if v_a.get("params") != v_b.get("params"):
            return False
        if v_a.get("is_transition") != v_b.get("is_transition"):
            return False
    return True


def generate_voice_audio(voice_data, duration, sample_rate, global_start_time, chunk_start_time=0.0, previous_state=None, return_state=False):
    """Generates audio for a single voice based on its definition."""
    func_name = voice_data.get("synth_function_name")
    params = dict(voice_data.get("params", {}))
    if not params.get("flangeEnable"):
        for k in list(params.keys()):
            if "flange" in k.lower():
                params[k] = False if k.lower().endswith("enable") else 0
    flange_params = {k: params.get(k) for k in list(params.keys()) if "flange" in k.lower()}
    if not any(bool(flange_params.get(k)) for k in (
        "flangeEnable", "startFlangeEnable", "endFlangeEnable"
    )):
        flange_params = {}
    core_params = {k: v for k, v in params.items() if "flange" not in k.lower()}
    if "duration" not in core_params and "post_offset" in core_params:
        core_params["duration"] = core_params["post_offset"]
    if "post_offset" in core_params:
        core_params.pop("post_offset", None)
    is_transition = voice_data.get("is_transition", False) # Check if this step IS a transition

    # --- Select the correct function (static or transition) ---
    actual_func_name = func_name
    selected_func_is_transition_type = func_name and func_name.endswith("_transition")

    # Determine the function to actually call based on 'is_transition' flag
    if is_transition:
        if not selected_func_is_transition_type:
            transition_func_name = func_name + "_transition"
            if transition_func_name in SYNTH_FUNCTIONS:
                actual_func_name = transition_func_name
                # print(f"Note: Step marked as transition, using '{actual_func_name}' instead of base '{func_name}'.")
            else:
                print(f"Warning: Step marked as transition, but transition function '{transition_func_name}' not found for base '{func_name}'. Using static version '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the static one)
    else: # Not a transition step
        if selected_func_is_transition_type:
            base_func_name = func_name.replace("_transition", "")
            if base_func_name in SYNTH_FUNCTIONS:
                actual_func_name = base_func_name
                # print(f"Note: Step not marked as transition, using base function '{actual_func_name}' instead of selected '{func_name}'.")
            else:
                print(f"Warning: Step not marked as transition, selected '{func_name}', but base function '{base_func_name}' not found. Using selected '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the transition one user selected)

    if not actual_func_name or actual_func_name not in SYNTH_FUNCTIONS:
        print(f"Error: Synth function '{actual_func_name}' (derived from '{func_name}') not found or invalid.")
        N = int(duration * sample_rate)
        empty = np.zeros((N, 2), dtype=np.float32)
        return (empty, {}) if return_state else empty

    synth_func = SYNTH_FUNCTIONS[actual_func_name]

    # Clean params: remove None values before passing to function, as functions use .get() with defaults
    cleaned_params = {k: v for k, v in core_params.items() if v is not None}

    # duration (and sometimes sample_rate) are supplied explicitly below. If they
    # are still present in the cleaned params dictionary we will pass them twice
    # which results in a "multiple values for keyword argument" TypeError for
    # transition synth functions. Strip them here so the explicit arguments take
    # precedence without causing an error.
    cleaned_params.pop("duration", None)
    cleaned_params.pop("sample_rate", None)

    # --- Generate base audio ---
    try:
        # Prepare params for stateful generation
        call_params = cleaned_params.copy()
        
        # Check if function accepts initial_offset
        # We can inspect or just try passing it if we are sure. 
        # For now, let's assume if it's binaural_beat it accepts it.
        # Or we can rely on **kwargs in the function definition.
        # Most functions accept **kwargs, so passing extra params is safe-ish, 
        # but some might not use it.
        # However, binaural_beat definitely uses it now.
        
        call_params['initial_offset'] = float(chunk_start_time)
        
        if previous_state:
            call_params.update(previous_state)

        # print(f"  Calling: {actual_func_name}(duration={duration}, sample_rate={sample_rate}, initial_offset={chunk_start_time}, ...)")
        result = synth_func(duration=duration, sample_rate=sample_rate, **call_params)
        
        voice_state = {}
        if isinstance(result, tuple) and len(result) == 2:
            audio, voice_state = result
        else:
            audio = result
            
    except Exception as e:
        print(f"Error calling synth function '{actual_func_name}' with params {cleaned_params}:")
        traceback.print_exc()
        N = int(duration * sample_rate)
        empty = np.zeros((N, 2), dtype=np.float32)
        return (empty, {}) if return_state else empty

    if audio is None:
        print(f"Error: Synth function '{actual_func_name}' returned None.")
        N = int(duration * sample_rate)
        empty = np.zeros((N, 2), dtype=np.float32)
        return (empty, {}) if return_state else empty

    # --- Apply volume envelope if defined ---
    envelope_data = voice_data.get("volume_envelope")
    N_audio = audio.shape[0]
    # Ensure t_rel matches audio length, especially if N calculation differs slightly
    t_rel = np.linspace(0, duration, N_audio, endpoint=False) if N_audio > 0 else np.array([])
    env = np.ones(N_audio) # Default flat envelope

    if envelope_data and isinstance(envelope_data, dict) and N_audio > 0:
        env_type = envelope_data.get("type")
        env_params = envelope_data.get("params", {})
        cleaned_env_params = {k: v for k, v in env_params.items() if v is not None}

        try:
            # Pass duration and sample_rate if needed by envelope func
            if 'duration' not in cleaned_env_params: cleaned_env_params['duration'] = duration
            if 'sample_rate' not in cleaned_env_params: cleaned_env_params['sample_rate'] = sample_rate

            if env_type == "adsr":
                env = adsr_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linen":
                 env = linen_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linear_fade":
                 # This function uses duration/sr internally, ensure they are passed if needed
                 required = ['fade_duration', 'start_amp', 'end_amp']
                 # Check params specific to linear_fade
                 specific_env_params = {k: v for k, v in cleaned_env_params.items() if k in required}
                 if all(p in specific_env_params for p in required):
                      # Pass the main duration and sample rate, not t_rel
                      env = create_linear_fade_envelope(duration, sample_rate, **specific_env_params)
                      # Resample envelope if its length doesn't match audio
                      if len(env) != N_audio:
                            print(f"Warning: Resampling '{env_type}' envelope from {len(env)} to {N_audio} samples.")
                            if len(env) > 0:
                                 env = np.interp(np.linspace(0, 1, N_audio), np.linspace(0, 1, len(env)), env)
                            else:
                                 env = np.ones(N_audio) # Fallback
                 else:
                      print(f"Warning: Missing parameters for 'linear_fade' envelope. Using flat envelope. Got: {specific_env_params}")
            # Add other envelope types here
            # elif env_type == "other_env":
            #    env = other_env_function(t_rel, **cleaned_env_params)
            else:
                print(f"Warning: Unknown envelope type '{env_type}'. Using flat envelope.")

            # Ensure envelope is broadcastable (N,)
            if env.shape != (N_audio,):
                 print(f"Warning: Envelope shape mismatch ({env.shape} vs {(N_audio,)}). Attempting reshape.")
                 if len(env) == N_audio: env = env.reshape(N_audio)
                 else:
                      print("Error: Cannot reshape envelope. Using flat envelope.")
                      env = np.ones(N_audio) # Fallback

        except Exception as e:
            print(f"Error creating envelope type '{env_type}':")
            traceback.print_exc()
            env = np.ones(N_audio) # Fallback

    # Apply the calculated envelope
    try:
        if audio.ndim == 2 and audio.shape[1] == 2 and len(env) == audio.shape[0]:
             audio = audio * env[:, np.newaxis] # Apply envelope element-wise to stereo
        elif audio.ndim == 1 and len(env) == len(audio): # Handle potential mono output from synth
             audio = audio * env
        elif N_audio == 0:
             pass # No audio to apply envelope to
        else:
             print(f"Error: Envelope length ({len(env)}) or audio shape ({audio.shape}) mismatch. Skipping envelope application.")
    except Exception as e:
        print(f"Error applying envelope to audio:")
        traceback.print_exc()


    # --- Ensure output is stereo ---
    if audio.ndim == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio. Panning.")
        pan_val = cleaned_params.get('pan', 0.0) # Assume pan param exists or default to center
        audio = pan2(audio, pan_val)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio (N, 1). Panning.")
        pan_val = cleaned_params.get('pan', 0.0)
        audio = pan2(audio[:,0], pan_val) # Extract the single column before panning

    # Final check for shape (N, 2)
    if not (audio.ndim == 2 and audio.shape[1] == 2):
          if N_audio == 0: 
              empty = np.zeros((0, 2))
              return (empty, {}) if return_state else empty
          else:
                print(f"Error: Final audio shape for voice is incorrect ({audio.shape}). Returning silence.")
                N_expected = int(duration * sample_rate)
                empty = np.zeros((N_expected, 2))
                return (empty, {}) if return_state else empty
    # Apply flanger effect if requested
    if flange_params:
        def _collect(prefix: str) -> dict:
            pfx_len = len(prefix)
            return {'flange' + k[pfx_len:]: v for k, v in flange_params.items() if k.startswith(prefix)}

        def _parse(raw: dict) -> dict:
            mapping = {
                'flangeDelayMs': ('delay_ms', float),
                'flangeDepthMs': ('depth_ms', float),
                'flangeRateHz': ('rate_hz', float),
                'flangeShape': ('lfo_shape', lambda x: 0 if str(x) == 'sine' else 1),
                'flangeFeedback': ('feedback', float),
                'flangeMix': ('wet', float),
                'flangeLoopLpfHz': ('loop_lpf_hz', float),
                'flangeLoopHpfHz': ('loop_hpf_hz', float),
                'flangeStereoMode': ('stereo_mode', int),
                'flangeSpreadDeg': ('spread_deg', float),
                'flangeDelayLaw': ('law', int),
                'flangeInterp': ('interp_mode', int),
                'flangeMinDelayMs': ('min_delay_ms', float),
                'flangeMaxDelayMs': ('max_delay_ms', float),
                'flangeDezipperDelayMs': ('dz_delay_ms', float),
                'flangeDezipperDepthMs': ('dz_depth_ms', float),
                'flangeDezipperRateMs': ('dz_rate_ms', float),
                'flangeDezipperFeedbackMs': ('dz_feedback_ms', float),
                'flangeDezipperWetMs': ('dz_wet_ms', float),
                'flangeDezipperFilterMs': ('dz_filter_ms', float),
                'flangeLoudnessMode': ('loud_mode', int),
                'flangeLoudnessTcMs': ('loud_tc_ms', float),
                'flangeLoudnessMinGain': ('loud_min_gain', float),
                'flangeLoudnessMaxGain': ('loud_max_gain', float),
                'flangeEnable': ('enable', bool),
            }
            out = {}
            for k, v in raw.items():
                if k in mapping:
                    tgt, cast = mapping[k]
                    out[tgt] = cast(v)
            return out

        base_raw = {k: v for k, v in flange_params.items() if not (k.startswith('startFlange') or k.startswith('endFlange'))}
        start_raw = {**base_raw, **_collect('startFlange')}
        end_raw = {**base_raw, **_collect('endFlange')}

        start_conf = _parse(start_raw)
        end_conf = _parse(end_raw)
        enable_start = start_conf.pop('enable', False)
        enable_end = end_conf.pop('enable', False)

        if is_transition and (start_conf != end_conf or enable_start != enable_end):
            transition_duration = core_params.get('duration')
            if transition_duration is None:
                transition_duration = core_params.get('post_offset')
            if transition_duration is not None:
                transition_duration = float(transition_duration)
            audio = _flanger_effect_stereo_continuous(
                audio, float(sample_rate), duration,
                float(core_params.get('initial_offset', 0.0)),
                transition_duration,
                str(core_params.get('transition_curve', 'linear')),
                start_conf, end_conf, enable_start, enable_end,
            )
        elif enable_start or enable_end:
            params_to_use = start_conf
            audio = flanger_stereo(audio.astype(np.float32), float(sample_rate), **params_to_use).astype(np.float32)

    # Add a small default fade if no specific envelope was requested and audio exists
    # This helps prevent clicks when steps are concatenated without crossfade
    if not envelope_data and N_audio > 0:
        fade_len = min(N_audio, int(0.01 * sample_rate)) # 10ms fade or audio length
        if fade_len > 1:
             fade_in = np.linspace(0, 1, fade_len)
             fade_out = np.linspace(1, 0, fade_len)
             # Apply fade using broadcasting
             audio[:fade_len] *= fade_in[:, np.newaxis]
             audio[-fade_len:] *= fade_out[:, np.newaxis]

    if return_state:
        return audio.astype(np.float32), voice_state
    return audio.astype(np.float32) # Ensure float32 output


def assemble_track_from_data(track_data, sample_rate, crossfade_duration, crossfade_curve="linear", progress_callback=None):
    """
    Assembles a track from a track_data dictionary.
    Uses crossfading between steps by overlapping their placement. The
    ``crossfade_curve`` parameter controls the shape of the fade between steps
    and is forwarded to :func:`crossfade_signals`.
    Includes per-step normalization to prevent excessive peaks before final mix.
    """
    steps_data = track_data.get("steps", [])
    if not steps_data:
        print("Warning: No steps found in track data.")
        return np.zeros((sample_rate, 2)) # Return 1 second silence

    # --- Calculate Track Length Estimation ---
    if not steps_data:
        estimated_total_duration = 0.0
    else:
        current_est = 0.0
        end_times = []
        for step in steps_data:
            dur = float(step.get("duration", 0))
            start_t = float(step.get("start", step.get("start_time", current_est)))
            if "start" not in step and "start_time" not in step:
                start_t = current_est
                current_est += dur
            end_times.append(start_t + dur)
        estimated_total_duration = max(end_times)
    if estimated_total_duration <= 0:
        print("Warning: Track has zero or negative estimated total duration.")
        return np.zeros((sample_rate, 2))

    # Add buffer for potential rounding errors and final sample
    estimated_total_samples = int(estimated_total_duration * sample_rate) + sample_rate
    track = np.zeros((estimated_total_samples, 2), dtype=np.float32) # Use float32

    # --- Time and Sample Tracking ---
    current_time = 0.0 # Start time for the *next* step to be placed
    last_step_end_sample_in_track = 0 # Tracks the actual last sample index used

    crossfade_samples = int(crossfade_duration * sample_rate)
    if crossfade_samples < 0: crossfade_samples = 0

    print(
        f"Assembling track: {len(steps_data)} steps, Est. Max Duration: {estimated_total_duration:.2f}s, "
        f"Crossfade: {crossfade_duration:.2f}s ({crossfade_samples} samples), Curve: {crossfade_curve}"
    )

    total_steps = len(steps_data)
    if progress_callback:
        try:
            progress_callback(0.0)
        except Exception as e:
            print(f"Progress callback error: {e}")

    global_settings = track_data.get("global_settings", {})
    prev_crossfade_samples = 0
    prev_crossfade_curve = crossfade_curve

    for i, step_data in enumerate(steps_data):
        step_duration = float(step_data.get("duration", 0))
        if step_duration <= 0:
            print(f"Skipping step {i+1} with zero or negative duration.")
            continue

        step_start_time = float(step_data.get("start", step_data.get("start_time", current_time)))
        if step_start_time < 0:
            step_start_time = 0.0

        # --- Calculate Placement Indices ---
        step_start_sample_abs = int(step_start_time * sample_rate)
        N_step = int(step_duration * sample_rate)
        step_end_sample_abs = step_start_sample_abs + N_step

        print(
            f"  Processing Step {i+1}: Place Start: {step_start_time:.2f}s ({step_start_sample_abs}), "
            f"Duration: {step_duration:.2f}s, Samples: {N_step}"
        )

        step_audio_mix = generate_single_step_audio_segment(
            step_data,
            global_settings,
            step_duration,
        )
        # --- Placement and Crossfading ---
        # Clip placement indices to the allocated track buffer boundaries
        safe_place_start = max(0, step_start_sample_abs)
        safe_place_end = min(estimated_total_samples, step_end_sample_abs)
        segment_len_in_track = safe_place_end - safe_place_start

        if segment_len_in_track <= 0:
            print(f"        Skipping Step {i+1} placement (no valid range in track buffer).")
            continue

        # Determine the portion of step_audio_mix to use
        audio_to_use = step_audio_mix[:segment_len_in_track]

        # Double check length (should normally match)
        if audio_to_use.shape[0] != segment_len_in_track:
            print(f"        Warning: Step {i+1} audio length adjustment needed ({audio_to_use.shape[0]} vs {segment_len_in_track}). Padding/Truncating.")
            if audio_to_use.shape[0] < segment_len_in_track:
                audio_to_use = np.pad(audio_to_use, ((0, segment_len_in_track - audio_to_use.shape[0]), (0,0)), 'constant')
            else:
                audio_to_use = audio_to_use[:segment_len_in_track]

        step_crossfade_duration = float(
            step_data.get("crossfade_duration", crossfade_duration)
        )
        step_crossfade_curve = str(
            step_data.get("crossfade_curve", crossfade_curve)
        )
        incoming_crossfade_samples = prev_crossfade_samples
        incoming_crossfade_curve = prev_crossfade_curve or crossfade_curve

        # --- Always use crossfade when overlap exists ---
        overlap_start_sample_in_track = safe_place_start
        overlap_end_sample_in_track = min(safe_place_end, last_step_end_sample_in_track)
        overlap_samples = overlap_end_sample_in_track - overlap_start_sample_in_track

        if i > 0 and overlap_samples > 0 and incoming_crossfade_samples > 0:
            actual_crossfade_samples = min(overlap_samples, incoming_crossfade_samples)
            print(
                f"        Crossfading Step {i+1} with previous. Overlap: {overlap_samples / sample_rate:.3f}s, "
                f"Actual CF: {actual_crossfade_samples / sample_rate:.3f}s, Curve: {incoming_crossfade_curve}"
            )

            prev_segment = track[
                overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples
            ]
            new_segment = audio_to_use[:actual_crossfade_samples]

            blended_segment = crossfade_signals(
                prev_segment,
                new_segment,
                sample_rate,
                actual_crossfade_samples / sample_rate,
                curve=incoming_crossfade_curve,
                phase_align=True,
            )

            track[
                overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples
            ] = blended_segment

            remaining_start_index_in_step_audio = actual_crossfade_samples
            remaining_start_index_in_track = overlap_start_sample_in_track + actual_crossfade_samples
            remaining_end_index_in_track = safe_place_end

            if remaining_start_index_in_track < remaining_end_index_in_track:
                num_remaining_samples_to_add = (
                    remaining_end_index_in_track - remaining_start_index_in_track
                )
                if remaining_start_index_in_step_audio < audio_to_use.shape[0]:
                    remaining_audio_from_step = audio_to_use[
                        remaining_start_index_in_step_audio : remaining_start_index_in_step_audio
                        + num_remaining_samples_to_add
                    ]
                    track[
                        remaining_start_index_in_track : remaining_start_index_in_track
                        + remaining_audio_from_step.shape[0]
                    ] += remaining_audio_from_step

        else:
            if i > 0 and incoming_crossfade_samples > 0:
                # Force crossfade even when no natural overlap exists
                actual_crossfade_samples = min(
                    incoming_crossfade_samples,
                    audio_to_use.shape[0],
                    last_step_end_sample_in_track,
                )
                if actual_crossfade_samples > 0:
                    print(
                        f"        Force crossfading Step {i+1} (no overlap). "
                        f"Actual CF: {actual_crossfade_samples / sample_rate:.3f}s, Curve: {incoming_crossfade_curve}"
                    )
                    prev_segment = track[
                        last_step_end_sample_in_track - actual_crossfade_samples : last_step_end_sample_in_track
                    ]
                    new_segment = audio_to_use[:actual_crossfade_samples]

                    blended_segment = crossfade_signals(
                        prev_segment,
                        new_segment,
                        sample_rate,
                        actual_crossfade_samples / sample_rate,
                        curve=incoming_crossfade_curve,
                        phase_align=True,
                    )

                    track[
                        last_step_end_sample_in_track - actual_crossfade_samples : last_step_end_sample_in_track
                    ] = blended_segment

                    remaining_audio = audio_to_use[actual_crossfade_samples:]
                    end_idx = last_step_end_sample_in_track + remaining_audio.shape[0]
                    track[last_step_end_sample_in_track:end_idx] += remaining_audio
                    safe_place_end = end_idx
                else:
                    track[safe_place_start:safe_place_end] += audio_to_use
            else:
                print(f"        Placing Step {i+1} without crossfade. Adding.")
                track[safe_place_start:safe_place_end] += audio_to_use

        # --- Update Markers for Next Loop ---
        last_step_end_sample_in_track = max(last_step_end_sample_in_track, safe_place_end)
        if "start" in step_data or "start_time" in step_data:
            current_time = step_start_time + step_duration
        else:
            effective_advance_duration = (
                max(0.0, step_duration - max(0.0, step_crossfade_duration))
                if incoming_crossfade_samples > 0 or step_crossfade_duration > 0
                else step_duration
            )
            current_time += effective_advance_duration

        prev_crossfade_samples = int(max(0.0, step_crossfade_duration) * sample_rate)
        prev_crossfade_curve = step_crossfade_curve

        if progress_callback:
            try:
                progress_callback((i + 1) / total_steps)
            except Exception as e:
                print(f"Progress callback error: {e}")


    # --- Final Trimming ---
    final_track_samples = last_step_end_sample_in_track
    if final_track_samples <= 0:
        print("Warning: Final track assembly resulted in zero length.")
        return np.zeros((sample_rate, 2))

    if final_track_samples < track.shape[0]:
        track = track[:final_track_samples]
    elif final_track_samples > estimated_total_samples:
         print(f"Warning: Final track samples ({final_track_samples}) exceeded initial estimate ({estimated_total_samples}).")

    track_duration_sec = final_track_samples / sample_rate

    # --- Optional Background Noise Layer ---
    bg_cfg = track_data.get("background_noise", {})
    bg_file = None
    if isinstance(bg_cfg, dict):
        bg_file = bg_cfg.get("file") or bg_cfg.get("params_path") or bg_cfg.get("noise_file")
    if bg_file:
        try:
            params = load_noise_params(bg_file)
            params.duration_seconds = track_duration_sec
            params.sample_rate = sample_rate
            if getattr(params, "transition", False):
                start_sweeps = [(sw.get("start_min", 1000), sw.get("start_max", 10000)) for sw in params.sweeps]
                end_sweeps = [(sw.get("end_min", 1000), sw.get("end_max", 10000)) for sw in params.sweeps]
                start_q = [sw.get("start_q", 30) for sw in params.sweeps]
                end_q = [sw.get("end_q", 30) for sw in params.sweeps]
                start_casc = [sw.get("start_casc", 10) for sw in params.sweeps]
                end_casc = [sw.get("end_casc", 10) for sw in params.sweeps]
                noise_audio, _ = _generate_swept_notch_arrays_transition(
                    track_duration_sec,
                    sample_rate,
                    params.start_lfo_freq,
                    params.end_lfo_freq,
                    start_sweeps,
                    end_sweeps,
                    start_q,
                    end_q,
                    start_casc,
                    end_casc,
                    params.start_lfo_phase_offset_deg,
                    params.end_lfo_phase_offset_deg,
                    params.start_intra_phase_offset_deg,
                    params.end_intra_phase_offset_deg,
                    params.input_audio_path or None,
                    params.noise_parameters,
                    params.lfo_waveform,
                    params.initial_offset,
                    params.duration,
                    "linear",
                    False,
                    2,
                    getattr(params, "static_notches", None),
                )
            else:
                sweeps = [(sw.get("start_min", 1000), sw.get("start_max", 10000)) for sw in params.sweeps]
                notch_q = [sw.get("start_q", 30) for sw in params.sweeps]
                casc = [sw.get("start_casc", 10) for sw in params.sweeps]
                noise_audio, _ = _generate_swept_notch_arrays(
                    track_duration_sec,
                    sample_rate,
                    params.lfo_freq,
                    sweeps,
                    notch_q,
                    casc,
                    params.start_lfo_phase_offset_deg,
                    params.start_intra_phase_offset_deg,
                    params.input_audio_path or None,
                    params.noise_parameters,
                    params.lfo_waveform,
                    False,
                    2,
                    getattr(params, "static_notches", None),
                )

            if noise_audio.shape[0] < final_track_samples:
                noise_audio = np.pad(
                    noise_audio,
                    ((0, final_track_samples - noise_audio.shape[0]), (0, 0)),
                    "constant",
                )

            gain = float(bg_cfg.get("gain", 1.0))
            start_time = float(bg_cfg.get("start_time", 0.0))
            fade_in = float(bg_cfg.get("fade_in", 0.0))
            fade_out = float(bg_cfg.get("fade_out", 0.0))
            amp_env = bg_cfg.get("amp_envelope")

            start_sample = int(start_time * sample_rate)
            if start_sample > 0:
                noise_audio = np.pad(noise_audio, ((start_sample, 0), (0, 0)), "constant")
                if noise_audio.shape[0] > final_track_samples:
                    noise_audio = noise_audio[:final_track_samples]

            env = np.ones(noise_audio.shape[0]) * gain
            if fade_in > 0:
                n = min(int(fade_in * sample_rate), env.size)
                env[:n] *= np.linspace(0, 1, n)
            if fade_out > 0:
                n = min(int(fade_out * sample_rate), env.size)
                env[-n:] *= np.linspace(1, 0, n)
            if isinstance(amp_env, list) and amp_env:
                times = [max(0.0, float(p[0])) for p in amp_env]
                amps = [float(p[1]) for p in amp_env]
                t_samples = np.array(times) * sample_rate
                interp = np.interp(np.arange(env.size), t_samples, amps, left=amps[0], right=amps[-1])
                env *= interp

            noise_audio = noise_audio[:env.size] * env[:, None]
            if noise_audio.shape[0] < track.shape[0]:
                noise_audio = np.pad(noise_audio, ((0, track.shape[0] - noise_audio.shape[0]), (0, 0)), "constant")

            track += noise_audio[: track.shape[0]]
        except Exception as e:
            print(f"Error generating background noise: {e}")

    # --- Overlay Clips ---
    clips = track_data.get("clips", [])
    if isinstance(clips, list):
        for clip in clips:
            try:
                path = clip.get("path")
                if not path:
                    continue
                clip_audio, clip_sr = sf.read(path)
                if clip_sr != sample_rate:
                    from scipy.signal import resample

                    n_target = int(len(clip_audio) * sample_rate / clip_sr)
                    if clip_audio.ndim == 1:
                        clip_audio = resample(clip_audio, n_target)
                    else:
                        clip_audio = np.stack(
                            [resample(clip_audio[:, i], n_target) for i in range(clip_audio.shape[1])],
                            axis=-1,
                        )
                if clip_audio.ndim == 1:
                    clip_audio = np.column_stack((clip_audio, clip_audio))
                start_sample = int(float(clip.get("start_time", 0)) * sample_rate)
                end_sample = start_sample + clip_audio.shape[0]
                if end_sample > track.shape[0]:
                    track = np.pad(track, ((0, end_sample - track.shape[0]), (0, 0)), "constant")
                    final_track_samples = max(final_track_samples, end_sample)
                gain = float(clip.get("gain", 1.0))
                track[start_sample:end_sample] += clip_audio * gain
            except Exception as e:
                print(f"Error overlaying clip {clip}: {e}")

    print(f"Track assembly finished. Final Duration: {final_track_samples / sample_rate:.2f}s")

    if progress_callback:
        try:
            progress_callback(1.0)
        except Exception as e:
            print(f"Progress callback error: {e}")
    return track


# -----------------------------------------------------------------------------
# JSON Loading/Saving
# -----------------------------------------------------------------------------

# Custom JSON encoder to handle numpy types (if needed)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_track_from_json(filepath):
    """Loads track definition from a JSON file.

    The loader accepts both the legacy structure (``global_settings``/``steps``)
    and the new v2 structure (``global``/``progression``). The returned dict
    always uses ``global_settings``/``steps`` keys so existing code can operate
    on the result without modification.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        print(f"Track data loaded successfully from {filepath}")

        if "progression" in raw or "global" in raw:
            new_data = {
                "global_settings": raw.get("global", {}),
                "steps": raw.get("progression", []),
                "background_noise": raw.get("background_noise", raw.get("noise", {})),
                "clips": raw.get("overlay_clips", raw.get("clips", [])),
            }
            raw = new_data

        if not isinstance(raw, dict) or "steps" not in raw or "global_settings" not in raw:
            print("Error: Invalid JSON structure. Missing required keys.")
            return None

        if not isinstance(raw["steps"], list) or not isinstance(raw["global_settings"], dict):
            print("Error: Invalid JSON structure types for 'steps' or 'global_settings'.")
            return None

        raw.setdefault("background_noise", {})
        raw.setdefault("clips", [])

        # Fill in missing start times so the GUI knows when each step begins
        crossfade = float(raw["global_settings"].get("crossfade_duration", 0.0))
        current_time = 0.0
        for step in raw["steps"]:
            if "start" not in step and "start_time" not in step:
                step["start"] = current_time
                advance = float(step.get("duration", 0))
                if crossfade > 0:
                    advance = max(0.0, advance - crossfade)
                current_time += advance
            else:
                step["start"] = float(step.get("start", step.get("start_time", 0)))
                current_time = max(current_time, step["start"] + float(step.get("duration", 0)))
            step.pop("start_time", None)

        return raw
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}:")
        traceback.print_exc()
        return None

def save_track_to_json(track_data, filepath):
    """Saves track definition to a JSON file using the new v2 structure."""
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        track_data.setdefault("background_noise", {})
        track_data.setdefault("clips", [])

        v2_data = {
            "global": track_data.get("global_settings", {}),
            "progression": [],
            "background_noise": track_data.get("background_noise", {}),
            "overlay_clips": track_data.get("clips", []),
        }

        crossfade = float(track_data.get("global_settings", {}).get("crossfade_duration", 0.0))
        current_time = 0.0
        for step in track_data.get("steps", []):
            step_copy = copy.deepcopy(step)
            if "start" not in step_copy and "start_time" not in step_copy:
                step_copy["start"] = current_time
                advance = float(step_copy.get("duration", 0))
                if crossfade > 0:
                    advance = max(0.0, advance - crossfade)
                current_time += advance
            else:
                step_copy["start"] = float(step_copy.get("start", step_copy.get("start_time", 0)))
                current_time = max(current_time, step_copy["start"] + float(step_copy.get("duration", 0)))
            step_copy.pop("start_time", None)
            v2_data["progression"].append(step_copy)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(v2_data, f, indent=4, cls=NumpyEncoder)
        print(f"Track data saved successfully to {filepath}")
        return True
    except IOError as e:
        print(f"Error writing file to {filepath}: {e}")
        return False
    except TypeError as e:
        print(f"Error serializing track data to JSON: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"An unexpected error occurred saving to {filepath}:")
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Main Generation Function
# -----------------------------------------------------------------------------

def _write_audio_file(audio_int16, sample_rate, filename):
    """Write audio data to a file based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == '.wav':
            write(filename, sample_rate, audio_int16)
        elif ext == '.flac':
            sf.write(filename, audio_int16, sample_rate, subtype='PCM_16')
        elif ext == '.mp3':
            try:
                from pydub import AudioSegment
            except Exception:
                print("Error: pydub not installed. Cannot export MP3.")
                return False
            seg = AudioSegment(audio_int16.tobytes(), frame_rate=sample_rate,
                               sample_width=2, channels=2)
            seg.export(filename, format='mp3')
        else:
            print(f"Unsupported output format: {ext}")
            return False
        print(f"Track successfully written to {filename}")
        return True
    except Exception as e:
        print(f"Error writing audio file {filename}: {e}")
        traceback.print_exc()
        return False


def generate_audio(track_data, output_filename=None, target_level=0.25, progress_callback=None):
    """Generate and export an audio file (WAV/FLAC/MP3) based on track_data."""
    if not track_data:
        print("Error: Cannot generate audio, track data is missing.")
        return False

    global_settings = track_data.get("global_settings", {})
    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        crossfade_duration = float(global_settings.get("crossfade_duration", 1.0))
        crossfade_curve = global_settings.get("crossfade_curve", "linear")
    except (ValueError, TypeError) as e:
         print(f"Error: Invalid global settings (sample_rate or crossfade_duration): {e}")
         return False

    output_filename = output_filename or global_settings.get("output_filename", "generated_track.wav")
    if not output_filename or not isinstance(output_filename, str):
         print(f"Error: Invalid output filename: {output_filename}")
         return False

    # Ensure output directory exists before assembly
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory for WAV: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return False


    start_time = time.time()

    print(f"\n--- Starting WAV Generation ---")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Crossfade Duration: {crossfade_duration} s (curve: {crossfade_curve})")
    print(f"Output File: {output_filename}")

    # Assemble the track (includes per-step normalization now)
    track_audio = assemble_track_from_data(track_data, sample_rate, crossfade_duration, crossfade_curve, progress_callback)

    if track_audio is not None and track_audio.size > 0 and not np.isfinite(track_audio).all():
        bad_count = np.count_nonzero(~np.isfinite(track_audio))
        print(
            f"Warning: Track assembly produced {bad_count} non-finite samples; replacing with zeros before normalization."
        )
        track_audio = np.nan_to_num(track_audio, nan=0.0, posinf=0.0, neginf=0.0)

    if track_audio is None or track_audio.size == 0:
        print("Error: Track assembly failed or resulted in empty audio.")
        return False

    # --- Final Normalization ---
    max_abs_val = np.max(np.abs(track_audio))

    if max_abs_val > 1e-9:  # Avoid division by zero for silent tracks
        target_level = float(target_level)
        scaling_factor = target_level / max_abs_val
        print(f"Normalizing final track (peak value: {max_abs_val:.4f}) to target level: {target_level}")
        normalized_track = track_audio * scaling_factor
        # Optional: Apply a limiter after normalization as a final safety net
        # normalized_track = np.clip(normalized_track, -target_level, target_level)
    else:
        print("Track is silent or near-silent. Skipping final normalization.")
        normalized_track = track_audio # Already silent or zero

    # Convert normalized float audio to 16-bit PCM
    if not np.issubdtype(normalized_track.dtype, np.floating):
         print(f"Warning: Normalized track data type is not float ({normalized_track.dtype}). Attempting conversion.")
         try: normalized_track = normalized_track.astype(np.float64) # Use float64 for precision before scaling
         except Exception as e:
              print(f"Error converting normalized track to float: {e}")
              return False

    # Scale to 16-bit integer range and clip just in case
    track_int16 = np.int16(np.clip(normalized_track * 32767, -32768, 32767))

    success = _write_audio_file(track_int16, sample_rate, output_filename)
    if success:
        print(f"--- Audio Generation Complete ---")
        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.2f} seconds")
    return success


def generate_wav(track_data, output_filename=None, target_level=0.25, progress_callback=None):
    """Backward compatible wrapper for generate_audio."""
    return generate_audio(track_data, output_filename, target_level, progress_callback)


def _synthesize_voice_task(args):
    """Worker helper for per-voice synthesis.

    Returns a tuple of (index, voice_type, func_name_short, voice_audio, new_state).
    """
    (
        i,
        voice_data,
        sample_rate,
        step_generation_duration,
        step_generation_samples,
        chunk_start_time,
        prev_state,
        return_state,
        full_step_duration,
    ) = args

    func_name_short = voice_data.get('synth_function_name', 'UnknownFunc')
    voice_type = voice_data.get('voice_type', 'binaural')

    cached_full_audio = None
    if isinstance(prev_state, dict):
        cached_full_audio = prev_state.get("full_audio")

    new_state = None
    voice_audio = None

    try:
        if voice_type == 'noise' and cached_full_audio is not None:
            start_sample = int(chunk_start_time * sample_rate)
            cached_len = cached_full_audio.shape[0]
            end_sample = min(start_sample + step_generation_samples, cached_len)
            voice_audio = cached_full_audio[start_sample:end_sample]
            if voice_audio.shape[0] < step_generation_samples:
                pad_len = step_generation_samples - voice_audio.shape[0]
                voice_audio = np.pad(
                    voice_audio,
                    ((0, pad_len), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            new_state = prev_state
        elif voice_type == 'noise' and return_state:
            full_duration = float(full_step_duration if full_step_duration is not None else step_generation_duration)
            full_samples = int(full_duration * sample_rate)
            full_audio, full_state = generate_voice_audio(
                voice_data,
                full_duration,
                sample_rate,
                0.0,
                chunk_start_time=0.0,
                previous_state=None,
                return_state=True,
            )
            full_state = full_state or {}
            full_state["full_audio"] = full_audio
            full_peak = np.max(np.abs(full_audio))
            full_state["noise_peak"] = full_peak if full_peak > 1e-9 else 1.0

            start_sample = int(chunk_start_time * sample_rate)
            end_sample = min(start_sample + step_generation_samples, full_samples)
            voice_audio = full_audio[start_sample:end_sample]
            if voice_audio.shape[0] < step_generation_samples:
                pad_len = step_generation_samples - voice_audio.shape[0]
                voice_audio = np.pad(
                    voice_audio,
                    ((0, pad_len), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            new_state = full_state
        else:
            voice_audio, v_state = generate_voice_audio(
                voice_data,
                step_generation_duration,
                sample_rate,
                0.0,
                chunk_start_time=chunk_start_time,
                previous_state=prev_state,
                return_state=True
            )
            new_state = v_state
    except Exception as exc:
        print(f"    Error generating Voice {i+1} ({func_name_short}): {exc}")
        traceback.print_exc()

    return (i, voice_type, func_name_short, voice_audio, new_state)

def generate_single_step_audio_segment(step_data, global_settings, target_duration_seconds, duration_override=None, chunk_start_time=0.0, voice_states=None, return_state=False):
    """
    Generates a raw audio segment for a single step, looping or truncating 
    it to fill a target duration.
    
    Args:
        step_data: Dictionary containing step configuration
        global_settings: Dictionary containing global audio settings
        target_duration_seconds: Target duration for the output segment
        duration_override: Optional override for the step's natural duration when generating audio
        chunk_start_time: Start time of this chunk relative to the step start (for LFO continuity)
        voice_states: List of state dictionaries for each voice from previous chunk
        return_state: If True, returns (audio, new_voice_states)
    """
    if not step_data or not global_settings:
        print("Error: Invalid step_data or global_settings provided.")
        empty = np.zeros((0, 2), dtype=np.float32)
        return (empty, []) if return_state else empty

    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
    except (ValueError, TypeError):
        print("Error: Invalid sample rate in global_settings.")
        empty = np.zeros((0, 2), dtype=np.float32)
        return (empty, []) if return_state else empty

    target_total_samples = int(target_duration_seconds * sample_rate)
    output_audio_segment = np.zeros((target_total_samples, 2), dtype=np.float32)

    voices_data = step_data.get("voices", [])
    if not voices_data:
        print("Warning: Step has no voices. Returning silence.")
        return (output_audio_segment, []) if return_state else output_audio_segment

    # Use duration override if provided, otherwise use step's natural duration
    if duration_override is not None:
        step_generation_duration = float(duration_override)
        # print(f"  Using duration override: {step_generation_duration:.2f}s (natural: {step_data.get('duration', 0):.2f}s)")
    else:
        step_generation_duration = float(step_data.get("duration", 0))
    
    if step_generation_duration <= 0:
        print("Warning: Step generation duration is zero or negative. Returning silence.")
        return (output_audio_segment, []) if return_state else output_audio_segment
    step_generation_samples = int(step_generation_duration * sample_rate)
    if step_generation_samples <= 0:
        print("Warning: Step has zero samples. Returning silence.")
        return (output_audio_segment, []) if return_state else output_audio_segment

    # Generate one iteration of the step's audio
    binaural_mix = np.zeros((step_generation_samples, 2), dtype=np.float32)
    noise_mix = np.zeros((step_generation_samples, 2), dtype=np.float32)
    
    # print(f"  Generating single iteration for step (Duration: {step_generation_duration:.2f}s, Samples: {step_generation_samples})")
    
    has_binaural = False
    has_noise = False

    new_voice_states = [None] * len(voices_data)
    current_voice_states = voice_states if voice_states and len(voice_states) == len(voices_data) else [None] * len(voices_data)

    parallel_requested = bool(step_data.get("parallel_voices", global_settings.get("parallel_voices", False)))
    max_workers_setting = step_data.get("parallel_voices_max_workers", global_settings.get("parallel_voices_max_workers"))
    parallel_backend = step_data.get("parallel_backend", global_settings.get("parallel_backend", "thread"))

    can_parallelize = parallel_requested and return_state and len(voices_data) > 1
    if parallel_requested and not return_state:
        print("    Parallel voice synthesis requested but disabled because return_state=False (streaming continuity not needed).")
    if len(voices_data) <= 1 and parallel_requested:
        print("    Parallel voice synthesis requested but only one voice present; running sequentially.")

    tasks = [
        (
            i,
            voice_data,
            sample_rate,
            step_generation_duration,
            step_generation_samples,
            chunk_start_time,
            current_voice_states[i],
            return_state,
            step_data.get("duration", step_generation_duration),
        )
        for i, voice_data in enumerate(voices_data)
    ]

    voice_results = [None] * len(voices_data)

    if can_parallelize:
        cpu_count = os.cpu_count() or len(voices_data)
        requested_workers = None
        if isinstance(max_workers_setting, (int, float)) and max_workers_setting > 0:
            requested_workers = int(max_workers_setting)
        worker_count = min(requested_workers or cpu_count, len(voices_data))
        Executor = ProcessPoolExecutor if str(parallel_backend).lower() == "process" else ThreadPoolExecutor
        if worker_count <= 1:
            can_parallelize = False
        else:
            print(f"    Parallel voice synthesis enabled using {Executor.__name__} with {worker_count} workers.")
            with Executor(max_workers=worker_count) as executor:
                futures = {executor.submit(_synthesize_voice_task, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        voice_results[idx] = future.result()
                    except Exception as exc:
                        func_name_short = voices_data[idx].get('synth_function_name', 'UnknownFunc')
                        print(f"    Error generating Voice {idx+1} ({func_name_short}) in parallel: {exc}")
                        traceback.print_exc()

    if not can_parallelize:
        for task in tasks:
            result = _synthesize_voice_task(task)
            voice_results[result[0]] = result

    for result in voice_results:
        if result is None:
            continue
        i, voice_type, func_name_short, voice_audio, v_state = result
        new_voice_states[i] = v_state

        if voice_audio is not None and voice_audio.shape[0] == step_generation_samples and voice_audio.ndim == 2 and voice_audio.shape[1] == 2:
            if voice_type == 'noise':
                noise_mix += voice_audio
                has_noise = True
            else:
                binaural_mix += voice_audio
                has_binaural = True
        elif voice_audio is not None:
            print(f"    Warning: Voice {i+1} ({func_name_short}) generated audio shape mismatch ({voice_audio.shape} vs {(step_generation_samples, 2)}). Skipping voice.")

    # --- Independent Normalization & Volume Application ---

    # Get settings and clamp volumes to MAX_INDIVIDUAL_GAIN to prevent clipping
    norm_level = float(step_data.get("normalization_level", global_settings.get("normalization_level", 0.95)))
    binaural_vol = min(float(step_data.get("binaural_volume", MAX_INDIVIDUAL_GAIN)), MAX_INDIVIDUAL_GAIN)
    noise_vol = min(float(step_data.get("noise_volume", MAX_INDIVIDUAL_GAIN)), MAX_INDIVIDUAL_GAIN)
    
    # 1. Normalize Binaural Mix
    if has_binaural:
        bin_peak = np.max(np.abs(binaural_mix))
        if bin_peak > 1e-9:
            # Normalize to target level
            binaural_mix *= (norm_level / bin_peak)
            # Apply volume
            binaural_mix *= binaural_vol
            # print(f"    Binaural mix normalized (peak {bin_peak:.3f} -> {norm_level:.2f}) and scaled by {binaural_vol:.2f}")
        else:
            print("    Warning: Binaural mix is silent.")

    # 2. Normalize Noise Mix
    # For streaming mode, use the cached peak from the full audio to ensure
    # consistent volume across all chunks (avoids volume pumping between chunks)
    if has_noise:
        # Collect cached peak values from noise voice states
        cached_noise_peak = None
        for i, voice_data in enumerate(voices_data):
            if voice_data.get('voice_type', 'binaural') == 'noise':
                state = new_voice_states[i] if i < len(new_voice_states) else None
                if isinstance(state, dict) and "noise_peak" in state:
                    peak_val = state["noise_peak"]
                    if cached_noise_peak is None or peak_val > cached_noise_peak:
                        cached_noise_peak = peak_val

        # Use cached peak if available (streaming mode), otherwise use chunk peak
        if cached_noise_peak is not None and cached_noise_peak > 1e-9:
            noise_peak = cached_noise_peak
        else:
            noise_peak = np.max(np.abs(noise_mix))

        if noise_peak > 1e-9:
            # Normalize to target level using consistent peak value
            noise_mix *= (norm_level / noise_peak)
            # Apply volume
            noise_mix *= noise_vol
            # print(f"    Noise mix normalized (peak {noise_peak:.3f} -> {norm_level:.2f}) and scaled by {noise_vol:.2f}")
        else:
            print("    Warning: Noise mix is silent.")

    # 3. Combine
    single_iteration_audio_mix = binaural_mix + noise_mix

    # 4. Post-mix normalization to prevent clipping
    # When both binaural and noise are at high volumes, their sum can exceed 1.0.
    # Apply a safety ceiling to ensure the final mix never clips.
    SAFETY_CEILING = 0.99  # Leave minimal headroom for float->PCM conversion
    combined_peak = np.max(np.abs(single_iteration_audio_mix))
    if combined_peak > SAFETY_CEILING:
        # Scale down to prevent clipping while preserving the binaural/noise ratio
        single_iteration_audio_mix *= (SAFETY_CEILING / combined_peak)

    # Fill the output_audio_segment by looping/truncating the single_iteration_audio_mix
    if step_generation_samples == 0:
        print("    Error: Step has zero generation samples, cannot loop.")
        return (output_audio_segment, new_voice_states) if return_state else output_audio_segment

    current_pos_samples = 0
    while current_pos_samples < target_total_samples:
        remaining_samples = target_total_samples - current_pos_samples
        samples_to_copy = min(remaining_samples, step_generation_samples)
        
        output_audio_segment[current_pos_samples:current_pos_samples + samples_to_copy] = single_iteration_audio_mix[:samples_to_copy]
        current_pos_samples += samples_to_copy
        
    # print(f"  Generated single step audio segment: {target_duration_seconds:.2f}s ({output_audio_segment.shape[0]} samples)")
    result = output_audio_segment.astype(np.float32)
    return (result, new_voice_states) if return_state else result


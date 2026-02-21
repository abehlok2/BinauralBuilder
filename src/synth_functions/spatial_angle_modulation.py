"""Spatial angle modulation synthesis functions."""

import numpy as np
import math
import numba
import traceback
import json
from .monaural_beat_stereo_amps import monaural_beat_stereo_amps, monaural_beat_stereo_amps_transition

# Placeholder for the missing audio_engine module
# If you have the 'audio_engine.py' file, place it in the same directory.
# Otherwise, the SAM functions will not work.
try:
    # Attempt to import the real audio_engine if available
    from .audio_engine import Node, SAMVoice, VALID_SAM_PATHS
    AUDIO_ENGINE_AVAILABLE = True
    print("INFO: audio_engine module loaded successfully in spatial_angle_modulation.")
except Exception:
    AUDIO_ENGINE_AVAILABLE = False
    print("WARNING: audio_engine module not found in spatial_angle_modulation. SAM functions will not be available.")
    # Define dummy classes/variables if audio_engine is missing
    class Node:
        def __init__(self, *args, **kwargs):
            # Store args needed for generate_samples duration calculation
            # Simplified: Just store duration if provided
            self.duration = args[0] if args else kwargs.get('duration', 0)
            pass
    class SAMVoice:
        def __init__(self, *args, **kwargs):
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
# Spatial Angle Modulation - Helper Functions
# -----------------------------------------------------------------------------

def _sam2_path_open_sinusoidal(phase: np.ndarray) -> np.ndarray:
    return np.sin(phase)


def _sam2_path_closed_circular(phase: np.ndarray) -> np.ndarray:
    return ((phase / (2.0 * math.pi)) % 1.0) * 2.0 - 1.0


def _sam2_path_discontinuous(phase: np.ndarray) -> np.ndarray:
    return np.where(np.sin(phase) >= 0.0, 1.0, -1.0)


SAM2_PATH_GENERATORS = {
    'open': _sam2_path_open_sinusoidal,
    'sinusoidal': _sam2_path_open_sinusoidal,
    'closed': _sam2_path_closed_circular,
    'circular': _sam2_path_closed_circular,
    'discontinuous': _sam2_path_discontinuous,
}


def _resolve_sam2_path_generator(path_type: str):
    return SAM2_PATH_GENERATORS.get(path_type.lower(), _sam2_path_open_sinusoidal)


def _catmull_rom_eval(p0, p1, p2, p3, t: np.ndarray):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2.0 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2.0 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return x, y


def _chaikin_smooth_points(points, is_closed: bool, passes: int, ratio: float):
    if passes <= 0 or len(points) < 3:
        return points

    ratio = float(np.clip(ratio, 1e-3, 0.499))
    smoothed = list(points)
    for _ in range(passes):
        if len(smoothed) < 3:
            break

        if is_closed:
            refined = []
            count = len(smoothed)
            for i in range(count):
                p0 = smoothed[i]
                p1 = smoothed[(i + 1) % count]
                q = ((1.0 - ratio) * p0[0] + ratio * p1[0], (1.0 - ratio) * p0[1] + ratio * p1[1])
                r = (ratio * p0[0] + (1.0 - ratio) * p1[0], ratio * p0[1] + (1.0 - ratio) * p1[1])
                refined.extend((q, r))
            smoothed = refined
        else:
            refined = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0 = smoothed[i]
                p1 = smoothed[i + 1]
                q = ((1.0 - ratio) * p0[0] + ratio * p1[0], (1.0 - ratio) * p0[1] + ratio * p1[1])
                r = (ratio * p0[0] + (1.0 - ratio) * p1[0], ratio * p0[1] + (1.0 - ratio) * p1[1])
                refined.extend((q, r))
            refined.append(smoothed[-1])
            smoothed = refined

    return smoothed


def _resolve_custom_path_xy(phase: np.ndarray, custom_profile):
    if isinstance(custom_profile, str):
        try:
            custom_profile = json.loads(custom_profile)
        except Exception:
            custom_profile = {}

    points = custom_profile.get('points') if isinstance(custom_profile, dict) else None
    if not isinstance(points, list) or len(points) < 2:
        return None, None

    clean_points = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                clean_points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                pass

    if len(clean_points) < 2:
        return None, None

    is_closed = bool(custom_profile.get('closedLoop', False)) if isinstance(custom_profile, dict) else False
    kind = str(custom_profile.get('kind', '')).lower() if isinstance(custom_profile, dict) else ''
    subnodes_per_segment = int(custom_profile.get('subNodesPerSegment', 24)) if isinstance(custom_profile, dict) else 24
    subnodes_per_segment = max(4, min(subnodes_per_segment, 256))
    smoothing_passes = int(custom_profile.get('smoothingPasses', 1)) if isinstance(custom_profile, dict) else 1
    smoothing_passes = max(0, min(smoothing_passes, 6))
    smoothing_ratio = float(custom_profile.get('smoothingRatio', 0.25)) if isinstance(custom_profile, dict) else 0.25

    if is_closed and clean_points[0] != clean_points[-1]:
        clean_points.append(clean_points[0])

    if kind != 'spline':
        clean_points = _chaikin_smooth_points(clean_points, is_closed=is_closed, passes=smoothing_passes, ratio=smoothing_ratio)

    sample_x = []
    sample_y = []
    sample_d = [0.0]

    if kind == 'spline' and len(clean_points) >= 3:
        segment_count = len(clean_points) if is_closed else len(clean_points) - 1
        for i in range(segment_count):
            p1 = clean_points[i]
            p2 = clean_points[(i + 1) % len(clean_points)] if is_closed else clean_points[i + 1]
            p0 = clean_points[i - 1] if i > 0 else (clean_points[-2] if is_closed else clean_points[0])
            p3 = clean_points[(i + 2) % len(clean_points)] if (is_closed or i + 2 < len(clean_points)) else clean_points[-1]
            t = np.linspace(0.0, 1.0, subnodes_per_segment, endpoint=False)
            x, y = _catmull_rom_eval(p0, p1, p2, p3, t)
            sample_x.extend(x.tolist())
            sample_y.extend(y.tolist())
        sample_x.append(clean_points[-1][0])
        sample_y.append(clean_points[-1][1])
    else:
        for i in range(len(clean_points) - 1):
            p0 = clean_points[i]
            p1 = clean_points[i + 1]
            t = np.linspace(0.0, 1.0, subnodes_per_segment, endpoint=False)
            for ti in t:
                sample_x.append((1.0 - ti) * p0[0] + ti * p1[0])
                sample_y.append((1.0 - ti) * p0[1] + ti * p1[1])
        sample_x.append(clean_points[-1][0])
        sample_y.append(clean_points[-1][1])

    for i in range(1, len(sample_x)):
        sample_d.append(sample_d[-1] + math.hypot(sample_x[i] - sample_x[i - 1], sample_y[i] - sample_y[i - 1]))

    total = sample_d[-1]
    if total <= 1e-6:
        return None, None

    pos = ((phase / (2.0 * math.pi)) % 1.0) * total
    x_interp = np.interp(pos, np.array(sample_d, dtype=np.float64), np.array(sample_x, dtype=np.float64))
    y_interp = np.interp(pos, np.array(sample_d, dtype=np.float64), np.array(sample_y, dtype=np.float64))
    return x_interp, y_interp


def _sam2_custom_path_shape_and_scale(phase: np.ndarray, custom_profile):
    x_interp, y_interp = _resolve_custom_path_xy(phase, custom_profile)
    if x_interp is None or y_interp is None:
        base = _sam2_path_open_sinusoidal(phase)
        return base, np.ones_like(base)

    angle_deg = np.degrees(np.arctan2(x_interp, y_interp))
    norm_angle = np.clip(angle_deg / 180.0, -1.0, 1.0)

    radial_dist = np.hypot(x_interp, y_interp)
    d_min = float(np.min(radial_dist))
    d_max = float(np.max(radial_dist))
    if d_max - d_min <= 1e-6:
        return norm_angle, np.ones_like(norm_angle)

    dist_norm = (radial_dist - d_min) / (d_max - d_min)
    # closer (smaller distance) => stronger spatial scale
    dynamic_scale = 1.25 - 0.5 * dist_norm
    return norm_angle, np.clip(dynamic_scale, 0.75, 1.25)


def _resolve_sam2_shape(path_type: str, phase: np.ndarray, custom_profile=None):
    if path_type.lower() == 'custom':
        return _sam2_custom_path_shape_and_scale(phase, custom_profile)
    generator = _resolve_sam2_path_generator(path_type)
    shape = generator(phase)
    return shape, np.ones_like(shape)

@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles(
    mono: np.ndarray,
    sample_rate: float,
    aOD: float, aOF: float, aOP: float,      # AM for this stage
    spatial_freq: float,
    path_radius: float,
    spatial_phase_off: float,                # Initial phase offset for spatial rotation
    clockwise: bool = True                  # Direction of rotation
):
    N = mono.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat  = np.empty(N, dtype=np.float32)
    azimuth   = np.empty(N, dtype=np.float32)
    elevation = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    dt = 1.0 / sample_rate

    # 1) Apply custom AM envelope + build mod_beat
    for i in numba.prange(N):
        t = i * dt
        if aOD > 0.0 and aOF > 0.0:
            clamped_aOD = min(max(aOD, 0.0), 2.0)
            env = (1.0 - clamped_aOD/2.0) + (clamped_aOD/2.0) * math.sin(2*math.pi*aOF*t + aOP)
        else:
            env = 1.0
        mod_beat[i] = mono[i] * env # mono is already float32

    # 2) Compute circular path (radius from mod_beat) → azimuth
    # Corrected phase calculation for parallel loop (spatial_freq is constant here)
    for i in numba.prange(N):
        t_i = i * dt
        # Calculate phase at time t_i directly
        direction = 1.0 if clockwise else -1.0
        current_spatial_phase_at_t = spatial_phase_off + direction * (2 * math.pi * spatial_freq * t_i)
        
        r = path_radius * (0.5 * (mod_beat[i] + 1.0)) # mod_beat is [-1, 1], so (mod_beat+1)/2 is [0,1]
        
        # Cartesian coordinates for HRTF lookup (y is often 'forward' in HRTF)
        # X = R * sin(angle) (side)
        # Y = R * cos(angle) (front/back)
        # atan2(x,y) means angle relative to positive Y axis, clockwise if X is positive.
        x_coord = r * math.sin(current_spatial_phase_at_t)
        y_coord = r * math.cos(current_spatial_phase_at_t)
        
        deg = math.degrees(math.atan2(x_coord, y_coord))
        azimuth[i] = (deg + 360.0) % 360.0

    return mod_beat, azimuth, elevation


@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles_transition_core(
    mono_input: np.ndarray, # Already transitional mono beat
    sample_rate: float,
    sAOD: float, eAOD: float,       # Start/End SAM Amplitude Osc Depth
    sAOF: float, eAOF: float,       # Start/End SAM Amplitude Osc Freq
    sAOP: float, eAOP: float,       # Start/End SAM Amplitude Osc Phase Offset
    sSpatialFreq: float, eSpatialFreq: float,
    sPathRadius: float, ePathRadius: float,
    sSpatialPhaseOff: float, eSpatialPhaseOff: float, # Start/End for initial spatial phase
    clockwise: bool = True
):
    N = mono_input.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat    = np.empty(N, dtype=np.float32)
    azimuth_deg = np.empty(N, dtype=np.float32)
    elevation_deg = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    actual_spatial_phase = np.empty(N, dtype=np.float64) # For storing accumulated phase

    dt = 1.0 / sample_rate

    # Loop 1 (parallel): Interpolate AM params and calculate mod_beat
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        t_i = i * dt

        current_aOD = sAOD + (eAOD - sAOD) * alpha
        current_aOF = sAOF + (eAOF - sAOF) * alpha
        current_aOP = sAOP + (eAOP - sAOP) * alpha
        
        env_factor = 1.0
        if current_aOD > 0.0 and current_aOF > 0.0: # Assuming depth 0-2
            clamped_aOD_i = min(max(current_aOD, 0.0), 2.0)
            env_factor = (1.0 - clamped_aOD_i/2.0) + \
                         (clamped_aOD_i/2.0) * math.sin(2*math.pi*current_aOF*t_i + current_aOP)
        mod_beat[i] = mono_input[i] * env_factor

    # Loop 2 (sequential): Interpolate spatial freq and accumulate spatial phase
    # The 'spatial_phase_off' transition is for the initial phase offset value.
    initial_phase_offset_val = sSpatialPhaseOff + (eSpatialPhaseOff - sSpatialPhaseOff) * 0.0 # Value at alpha=0
    
    direction = 1.0 if clockwise else -1.0
    current_phase_val = initial_phase_offset_val
    if N > 0:
      actual_spatial_phase[0] = current_phase_val

    for i in range(N): # Must be sequential due to phase accumulation
        alpha = i / (N - 1) if N > 1 else 0.0
        current_sf_i = sSpatialFreq + (eSpatialFreq - sSpatialFreq) * alpha
        
        if i > 0: # Accumulate phase
            current_phase_val += direction * (2 * math.pi * current_sf_i * dt)
            actual_spatial_phase[i] = current_phase_val
        elif i == 0: # Already set for i=0 if N>0
             actual_spatial_phase[i] = current_phase_val

    # Loop 3 (parallel): Interpolate path radius and calculate azimuth
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        current_path_r = sPathRadius + (ePathRadius - sPathRadius) * alpha
        
        r_factor = current_path_r * (0.5 * (mod_beat[i] + 1.0))
        
        x_coord = r_factor * math.sin(actual_spatial_phase[i])
        y_coord = r_factor * math.cos(actual_spatial_phase[i])
        deg = math.degrees(math.atan2(x_coord, y_coord))
        azimuth_deg[i] = (deg + 360.0) % 360.0
        
    return mod_beat, azimuth_deg, elevation_deg


# -----------------------------------------------------------------------------
# Spatial Angle Modulation Functions
# -----------------------------------------------------------------------------

def spatial_angle_modulation(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation using external audio_engine module."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    carrierFreq = float(params.get('carrierFreq', 440.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    pathShape = str(params.get('pathShape', 'circle'))
    pathRadius = float(params.get('pathRadius', 1.0))
    arcStartDeg = float(params.get('arcStartDeg', 0.0))
    arcEndDeg = float(params.get('arcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if pathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid pathShape '{pathShape}'. Defaulting to 'circle'. Valid: {VALID_SAM_PATHS}")
        pathShape = 'circle'

    try:
        node = Node(duration, carrierFreq, beatFreq, 1.0, 1.0)
    except Exception as e:
        print(f"Error creating Node for SAM: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_dict = {
        'path_shape': pathShape,
        'path_radius': pathRadius,
        'arc_start_deg': arcStartDeg,
        'arc_end_deg': arcEndDeg
    }

    try:
        voice = SAMVoice(
            nodes=[node],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp,
            sam_node_params=[sam_params_dict]
        )
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def spatial_angle_modulation_transition(
    duration, sample_rate=44100, initial_offset=0.0, transition_duration=None, **params
):
    """Spatial Angle Modulation with parameter transitions."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM transition function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    startCarrierFreq = float(params.get('startCarrierFreq', 440.0))
    endCarrierFreq = float(params.get('endCarrierFreq', 440.0))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', 4.0))
    startPathShape = str(params.get('startPathShape', 'circle'))
    endPathShape = str(params.get('endPathShape', 'circle'))
    startPathRadius = float(params.get('startPathRadius', 1.0))
    endPathRadius = float(params.get('endPathRadius', 1.0))
    startArcStartDeg = float(params.get('startArcStartDeg', 0.0))
    endArcStartDeg = float(params.get('endArcStartDeg', 0.0))
    startArcEndDeg = float(params.get('startArcEndDeg', 360.0))
    endArcEndDeg = float(params.get('endArcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if startPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid startPathShape '{startPathShape}'. Defaulting to 'circle'.")
        startPathShape = 'circle'
    if endPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid endPathShape '{endPathShape}'. Defaulting to 'circle'.")
        endPathShape = 'circle'

    try:
        node_start = Node(duration, startCarrierFreq, startBeatFreq, 1.0, 1.0)
        node_end = Node(0.0, endCarrierFreq, endBeatFreq, 1.0, 1.0)
    except Exception as e:
        print(f"Error creating Nodes for SAM transition: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_list = [
        {
            'path_shape': startPathShape,
            'path_radius': startPathRadius,
            'arc_start_deg': startArcStartDeg,
            'arc_end_deg': startArcEndDeg
        },
        {
            'path_shape': endPathShape,
            'path_radius': endPathRadius,
            'arc_start_deg': endArcStartDeg,
            'arc_end_deg': endArcEndDeg
        }
    ]

    try:
        voice = SAMVoice(
            nodes=[node_start, node_end],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp,
            sam_node_params=sam_params_list
        )
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice transition generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def spatial_angle_modulation_sam2(duration, sample_rate=44100, **params):
    """SAM2 spatialized stereo tone with modular path generators.

    Angular controls are specified in degrees.
    """
    n_samples = int(duration * sample_rate)
    if n_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    amp = float(params.get('amp', 0.7))
    carrier_freq = float(params.get('carrierFreq', 440.0))
    mod_freq = float(params.get('modFreq', params.get('beatFreq', 4.0)))
    arc_width_deg = float(params.get('arcWidthDeg', params.get('arcWidth', 90.0)))
    direction_offset_deg = float(params.get('directionOffsetDeg', params.get('directionOffset', 0.0)))
    spatial_scale = float(params.get('spatialScale', 1.0))
    path_type = str(params.get('pathType', 'open')).lower()
    custom_path_profile = params.get('customPathProfile', {})

    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    mod_phase = 2.0 * math.pi * mod_freq * t

    shape, dynamic_scale = _resolve_sam2_shape(path_type, mod_phase, custom_path_profile)
    spatial_angle_deg = direction_offset_deg + 0.5 * arc_width_deg * shape
    interaural_phase = (spatial_scale * dynamic_scale) * np.sin(np.radians(spatial_angle_deg))
    carrier_phase = 2.0 * math.pi * carrier_freq * t

    left = amp * np.sin(carrier_phase - interaural_phase)
    right = amp * np.sin(carrier_phase + interaural_phase)

    return np.column_stack((left, right)).astype(np.float32)


def spatial_angle_modulation_sam2_transition(
    duration, sample_rate=44100, initial_offset=0.0, transition_duration=None, **params
):
    """SAM2 transition with linear parameter interpolation (angles in degrees)."""
    n_samples = int(duration * sample_rate)
    if n_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    amp = float(params.get('amp', 0.7))
    start_carrier = float(params.get('startCarrierFreq', 440.0))
    end_carrier = float(params.get('endCarrierFreq', 440.0))
    start_mod = float(params.get('startModFreq', params.get('startBeatFreq', 4.0)))
    end_mod = float(params.get('endModFreq', params.get('endBeatFreq', 4.0)))
    start_arc_width = float(params.get('startArcWidthDeg', params.get('arcWidthDeg', 90.0)))
    end_arc_width = float(params.get('endArcWidthDeg', params.get('arcWidthDeg', start_arc_width)))
    start_direction = float(params.get('startDirectionOffsetDeg', params.get('directionOffsetDeg', 0.0)))
    end_direction = float(params.get('endDirectionOffsetDeg', params.get('directionOffsetDeg', start_direction)))
    start_spatial_scale = float(params.get('startSpatialScale', params.get('spatialScale', 1.0)))
    end_spatial_scale = float(params.get('endSpatialScale', params.get('spatialScale', start_spatial_scale)))
    path_type = str(params.get('pathType', 'open')).lower()
    custom_path_profile = params.get('customPathProfile', {})

    alpha = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    carrier_freq = start_carrier + (end_carrier - start_carrier) * alpha
    mod_freq = start_mod + (end_mod - start_mod) * alpha
    arc_width_deg = start_arc_width + (end_arc_width - start_arc_width) * alpha
    direction_offset_deg = start_direction + (end_direction - start_direction) * alpha
    spatial_scale = start_spatial_scale + (end_spatial_scale - start_spatial_scale) * alpha

    mod_phase = np.cumsum((2.0 * math.pi * mod_freq) / float(sample_rate))
    carrier_phase = np.cumsum((2.0 * math.pi * carrier_freq) / float(sample_rate))

    shape, dynamic_scale = _resolve_sam2_shape(path_type, mod_phase, custom_path_profile)
    spatial_angle_deg = direction_offset_deg + 0.5 * arc_width_deg * shape
    interaural_phase = (spatial_scale * dynamic_scale) * np.sin(np.radians(spatial_angle_deg))

    left = amp * np.sin(carrier_phase - interaural_phase + initial_offset)
    right = amp * np.sin(carrier_phase + interaural_phase + initial_offset)

    return np.column_stack((left, right)).astype(np.float32)


def spatial_angle_modulation_monaural_beat(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation combined with monaural beats."""
    # --- unpack AM params for this specific stage ---
    # These are applied *after* the monaural beat generation's own AM
    sam_aOD = float(params.get('sam_ampOscDepth', params.get('ampOscDepth', 0.0))) # prefix with sam_ to avoid conflict
    sam_aOF = float(params.get('sam_ampOscFreq', params.get('ampOscFreq', 0.0)))
    sam_aOP = float(params.get('sam_ampOscPhaseOffset', params.get('ampOscPhaseOffset', 0.0)))

    # --- prepare core beat args (can have its own AM) ---
    beat_params = {
        'amp_lower_L':   float(params.get('amp_lower_L',   0.5)),
        'amp_upper_L':   float(params.get('amp_upper_L',   0.5)),
        'amp_lower_R':   float(params.get('amp_lower_R',   0.5)),
        'amp_upper_R':   float(params.get('amp_upper_R',   0.5)),
        'baseFreq':      float(params.get('baseFreq',      200.0)),
        'beatFreq':      float(params.get('beatFreq',        4.0)),
        'startPhaseL':   float(params.get('startPhaseL',    0.0)),
        'startPhaseR':   float(params.get('startPhaseR',    0.0)), # for upper component
        'phaseOscFreq':  float(params.get('phaseOscFreq',  0.0)),
        'phaseOscRange': float(params.get('phaseOscRange', 0.0)),
        'ampOscDepth':   float(params.get('monaural_ampOscDepth', 0.0)), # AM for monaural beat
        'ampOscFreq':    float(params.get('monaural_ampOscFreq', 0.0)),
        'ampOscPhaseOffset': float(params.get('monaural_ampOscPhaseOffset', 0.0)) # AM Phase for monaural
    }
    beat_freq         = beat_params['beatFreq']
    spatial_freq      = float(params.get('spatialBeatFreq', beat_freq)) # Default to beatFreq
    spatial_phase_off = float(params.get('spatialPhaseOffset', 0.0)) # Radians
    rotation_dir      = str(params.get('rotationDirection', 'cw')).lower()
    clockwise         = rotation_dir != 'ccw'

    # --- SAM controls ---
    amp               = float(params.get('amp', 0.7)) # Overall amplitude for HRTF input
    path_radius       = float(params.get('pathRadius', 1.0)) # Normalized radius factor
    frame_dur_ms      = float(params.get('frame_dur_ms', 46.4))
    overlap_fac       = int(params.get('overlap_factor',   8))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # Generate core stereo beat & collapse to mono
    beat_stereo = monaural_beat_stereo_amps(duration, sample_rate, **beat_params)
    mono_beat   = np.mean(beat_stereo, axis=1).astype(np.float32)

    # Call Numba helper to get mod_beat + az/el arrays
    mod_beat, azimuth_deg, elevation_deg = _prepare_beats_and_angles(
        mono_beat, float(sample_rate),
        sam_aOD, sam_aOF, sam_aOP, # SAM-specific AM params
        spatial_freq, path_radius,
        spatial_phase_off,
        clockwise
    )

    # --- OLA + HRTF (assuming slab and HRTF are available and configured) ---
    stereo_out = np.zeros((N, 2), dtype=np.float32)
    stereo_out[:, 0] = mod_beat * amp
    stereo_out[:, 1] = mod_beat * amp # Simple mono duplicate
    max_val = np.max(np.abs(stereo_out))
    if max_val > 1.0:
       stereo_out /= (max_val / 0.98)
    return stereo_out


def spatial_angle_modulation_monaural_beat_transition(
    duration, sample_rate=44100, initial_offset=0.0, transition_duration=None, **params
):
    """Spatial Angle Modulation monaural beat with transitions."""
    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # --- Parameters for the underlying monaural_beat_stereo_amps_transition ---
    s_ll = float(params.get('start_amp_lower_L', params.get('amp_lower_L', 0.5)))
    e_ll = float(params.get('end_amp_lower_L',   s_ll))
    s_ul = float(params.get('start_amp_upper_L', params.get('amp_upper_L', 0.5)))
    e_ul = float(params.get('end_amp_upper_L',   s_ul))
    s_lr = float(params.get('start_amp_lower_R', params.get('amp_lower_R', 0.5)))
    e_lr = float(params.get('end_amp_lower_R',   s_lr))
    s_ur = float(params.get('start_amp_upper_R', params.get('amp_upper_R', 0.5)))
    e_ur = float(params.get('end_amp_upper_R',   s_ur))
    sBF  = float(params.get('startBaseFreq',     params.get('baseFreq',    200.0)))
    eBF  = float(params.get('endBaseFreq',       sBF))
    sBt  = float(params.get('startBeatFreq',     params.get('beatFreq',      4.0)))
    eBt  = float(params.get('endBeatFreq',       sBt))
    sSPL_mono = float(params.get('startStartPhaseL_monaural', params.get('startPhaseL', 0.0)))
    eSPL_mono = float(params.get('endStartPhaseL_monaural', sSPL_mono))
    sSPU_mono = float(params.get('startStartPhaseU_monaural', params.get('startPhaseR', 0.0)))
    eSPU_mono = float(params.get('endStartPhaseU_monaural', sSPU_mono))
    sPhiF_mono = float(params.get('startPhaseOscFreq_monaural', params.get('phaseOscFreq', 0.0)))
    ePhiF_mono = float(params.get('endPhaseOscFreq_monaural', sPhiF_mono))
    sPhiR_mono = float(params.get('startPhaseOscRange_monaural', params.get('phaseOscRange', 0.0)))
    ePhiR_mono = float(params.get('endPhaseOscRange_monaural', sPhiR_mono))
    sAOD_mono = float(params.get('startAmpOscDepth_monaural', params.get('monaural_ampOscDepth', 0.0)))
    eAOD_mono = float(params.get('endAmpOscDepth_monaural', sAOD_mono))
    sAOF_mono = float(params.get('startAmpOscFreq_monaural', params.get('monaural_ampOscFreq', 0.0)))
    eAOF_mono = float(params.get('endAmpOscFreq_monaural', sAOF_mono))
    sAOP_mono = float(params.get('startAmpOscPhaseOffset_monaural', params.get('monaural_ampOscPhaseOffset', 0.0)))
    eAOP_mono = float(params.get('endAmpOscPhaseOffset_monaural', sAOP_mono))

    monaural_trans_params = {
        'start_amp_lower_L': s_ll, 'end_amp_lower_L': e_ll,
        'start_amp_upper_L': s_ul, 'end_amp_upper_L': e_ul,
        'start_amp_lower_R': s_lr, 'end_amp_lower_R': e_lr,
        'start_amp_upper_R': s_ur, 'end_amp_upper_R': e_ur,
        'startBaseFreq': sBF, 'endBaseFreq': eBF,
        'startBeatFreq': sBt, 'endBeatFreq': eBt,
        'startStartPhaseL': sSPL_mono, 'endStartPhaseL': eSPL_mono,
        'startStartPhaseU': sSPU_mono, 'endStartPhaseU': eSPU_mono,
        'startPhaseOscFreq': sPhiF_mono, 'endPhaseOscFreq': ePhiF_mono,
        'startPhaseOscRange': sPhiR_mono, 'endPhaseOscRange': ePhiR_mono,
        'startAmpOscDepth': sAOD_mono, 'endAmpOscDepth': eAOD_mono,
        'startAmpOscFreq': sAOF_mono, 'endAmpOscFreq': eAOF_mono,
        'startAmpOscPhaseOffset': sAOP_mono, 'endAmpOscPhaseOffset': eAOP_mono,
    }

    # --- Parameters for the SAM stage AM and spatialization (transitional) ---
    sSamAOD = float(params.get('start_sam_ampOscDepth', params.get('sam_ampOscDepth', 0.0)))
    eSamAOD = float(params.get('end_sam_ampOscDepth', sSamAOD))
    sSamAOF = float(params.get('start_sam_ampOscFreq', params.get('sam_ampOscFreq', 0.0)))
    eSamAOF = float(params.get('end_sam_ampOscFreq', sSamAOF))
    sSamAOP = float(params.get('start_sam_ampOscPhaseOffset', params.get('sam_ampOscPhaseOffset', 0.0)))
    eSamAOP = float(params.get('end_sam_ampOscPhaseOffset', sSamAOP))

    default_spatial_freq = (sBt + eBt) / 2.0 # Default to average beatFreq
    sSpatialFreq = float(params.get('startSpatialBeatFreq', params.get('spatialBeatFreq', default_spatial_freq)))
    eSpatialFreq = float(params.get('endSpatialBeatFreq', sSpatialFreq))
    
    sSpatialPhaseOff = float(params.get('startSpatialPhaseOffset', params.get('spatialPhaseOffset', 0.0)))
    eSpatialPhaseOff = float(params.get('endSpatialPhaseOffset', sSpatialPhaseOff))

    rotation_dir = str(params.get('rotationDirection', 'cw')).lower()
    clockwise     = rotation_dir != 'ccw'

    sPathRadius = float(params.get('startPathRadius', params.get('pathRadius', 1.0)))
    ePathRadius = float(params.get('endPathRadius', sPathRadius))

    # --- SAM controls (non-transitional for OLA) ---
    sAmp = float(params.get('startAmp', params.get('amp', 0.7))) # Overall amplitude for HRTF input
    eAmp = float(params.get('endAmp', sAmp))
    # For OLA, amp is applied per frame. We can interpolate it if needed, or use an average.
    # For now, let's use interpolated amp for mono_src
    
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_fac  = int(params.get('overlap_factor',   8))

    # 1. Generate transitional monaural beat
    trans_beat_stereo = monaural_beat_stereo_amps_transition(duration, sample_rate, **monaural_trans_params)
    trans_mono_beat   = np.mean(trans_beat_stereo, axis=1).astype(np.float32)

    # 2. Call the new transitional _prepare_beats_and_angles_transition_core
    trans_mod_beat, trans_azimuth_deg, trans_elevation_deg = \
        _prepare_beats_and_angles_transition_core(
            trans_mono_beat, float(sample_rate),
            sSamAOD, eSamAOD, sSamAOF, eSamAOF, sSamAOP, eSamAOP,
            sSpatialFreq, eSpatialFreq,
            sPathRadius, ePathRadius,
            sSpatialPhaseOff, eSpatialPhaseOff,
            clockwise
        )
    
    # 3. OLA + HRTF processing (using transitional mod_beat and azimuth)
    # This part remains Python-based.
    # Placeholder for slab integration - replace with actual calls.
    print("spatial_angle_modulation_monaural_beat_transition: HRTF processing part is illustrative.")
    # Fallback: return a simple stereo mix of trans_mod_beat with interpolated amp
    final_amp_coeffs = np.linspace(sAmp, eAmp, N, dtype=np.float32) if N > 0 else np.array([], dtype=np.float32)
    
    temp_out = np.zeros((N, 2), dtype=np.float32)
    if N > 0:
       temp_out[:, 0] = trans_mod_beat * final_amp_coeffs
       temp_out[:, 1] = trans_mod_beat * final_amp_coeffs
    
    max_v = np.max(np.abs(temp_out)) if N > 0 and temp_out.size > 0 else 0.0
    if max_v > 1.0: temp_out /= (max_v / 0.98)
    return temp_out

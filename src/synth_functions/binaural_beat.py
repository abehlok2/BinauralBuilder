"""Binaural beat synthesis functions."""

import numpy as np
import numba
from .common import (
    skewed_sine_phase,
    skewed_triangle_phase,
    _frac,
    generate_pan_envelope,
    apply_pan_envelope,
)
from .spatial_ambi2d import spatialize_binaural_mid_only, generate_azimuth_trajectory


def binaural_beat(duration, sample_rate=44100, initial_offset=0.0, **params):
    # --- Unpack synthesis parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    baseF = float(params.get('baseFreq', 200.0))
    beatF = float(params.get('beatFreq', 4.0))
    leftHigh = bool(params.get('leftHigh', False))
    phase_state = params.get('oscillator_phases', {}) if isinstance(params.get('oscillator_phases'), dict) else {}
    startL = float(phase_state.get('left', params.get('startPhaseL', 0.0))) # in radians
    startR = float(phase_state.get('right', params.get('startPhaseR', 0.0))) # in radians
    aODL = float(params.get('ampOscDepthL', 0.0))
    aOFL = float(params.get('ampOscFreqL', 0.0))
    aODR = float(params.get('ampOscDepthR', 0.0))
    aOFR = float(params.get('ampOscFreqR', 0.0))
    fORL = float(params.get('freqOscRangeL', 0.0))
    fOFL = float(params.get('freqOscFreqL', 0.0))
    fORR = float(params.get('freqOscRangeR', 0.0))
    fOFR = float(params.get('freqOscFreqR', 0.0))
    freqOscSkewL = float(params.get('freqOscSkewL', 0.0))
    freqOscSkewR = float(params.get('freqOscSkewR', 0.0))
    freqOscPhaseOffsetL = float(params.get('freqOscPhaseOffsetL', 0.0))
    freqOscPhaseOffsetR = float(params.get('freqOscPhaseOffsetR', 0.0))
    ampOscPhaseOffsetL = float(params.get('ampOscPhaseOffsetL', 0.0))
    ampOscPhaseOffsetR = float(params.get('ampOscPhaseOffsetR', 0.0))
    ampOscSkewL = float(params.get('ampOscSkewL', 0.0))
    ampOscSkewR = float(params.get('ampOscSkewR', 0.0))
    freqOscShape = str(params.get('freqOscShape', 'sine')).lower()
    shape_int = 1 if freqOscShape == 'triangle' else 0
    pOF = float(params.get('phaseOscFreq', 0.0))
    pOR = float(params.get('phaseOscRange', 0.0)) # in radians

    pan_default = float(params.get('pan', 0.0))
    pan_range_min = np.clip(float(params.get('panRangeMin', pan_default)), -1.0, 1.0)
    pan_range_max = np.clip(float(params.get('panRangeMax', pan_default)), -1.0, 1.0)
    if pan_range_min > pan_range_max:
        pan_range_min, pan_range_max = pan_range_max, pan_range_min
    pan_type = str(params.get('panType', 'linear')).strip().lower()
    pan_freq = float(params.get('panFreq', 0.0))
    pan_phase = float(params.get('panPhase', 0.0))

    # --- UNCOMMENT IF "GLITCH/CHIRP" EFFECT NEEDED OR DESIRED
    # --- Unpack glitch parameters --- 
    # glitchInterval   = float(params.get('glitchInterval',   0.0))
    # glitchDur        = float(params.get('glitchDur',        0.0))
    # glitchNoiseLevel = float(params.get('glitchNoiseLevel', 0.0))
    # glitchFocusWidth = float(params.get('glitchFocusWidth', 0.0))
    # glitchFocusExp   = float(params.get('glitchFocusExp',   0.0))

    N = int(duration * sample_rate)
    #
    # # --- Precompute glitch bursts in NumPy ---
    # positions = []
    # bursts = []
    # if glitchInterval > 0 and glitchDur > 0 and glitchNoiseLevel > 0 and N > 0:
    #     full_n = int(glitchDur * sample_rate)
    #     if full_n > 0:
    #         repeats = int(duration / glitchInterval)
    #         for k in range(1, repeats + 1):
    #             t_end   = k * glitchInterval
    #             t_start = max(0.0, t_end - glitchDur)
    #             i0 = int(t_start * sample_rate)
    #             i1 = i0 + full_n
    #             if i1 > N:
    #                 continue
    #
    #             white = np.random.standard_normal(full_n)
    #             S = np.fft.rfft(white)
    #             freqs_denominator = full_n if full_n != 0 else 1
    #             freqs = np.arange(S.size) * (sample_rate / freqs_denominator)
    #
    #             if glitchFocusWidth == 0:
    #                 gauss = np.ones_like(freqs)
    #             else:
    #                 gauss = np.exp(-0.5 * ((freqs - baseF) / glitchFocusWidth) ** glitchFocusExp)
    #
    #             shaped = np.fft.irfft(S * gauss, n=full_n)
    #             max_abs_shaped = np.max(np.abs(shaped))
    #             if max_abs_shaped < 1e-16:
    #                 shaped_normalized = np.zeros_like(shaped)
    #             else:
    #                 shaped_normalized = shaped / max_abs_shaped
    #             ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
    #             burst_samples = (shaped_normalized * ramp * glitchNoiseLevel).astype(np.float32)
    #
    #             positions.append(i0)
    #             bursts.append(burst_samples)
    #
    #             positions.append(i0)
    #             bursts.append(burst_samples)
    #
    # if bursts:
    #     pos_arr   = np.array(positions, dtype=np.int32)
    #     # Ensure all burst_samples have the same length if concatenating or handle individually
    #     # For simplicity, assuming all burst_samples are concatenated correctly if generated
    #     if any(b.ndim == 0 or b.size == 0 for b in bursts): # check for empty or scalar arrays
    #         burst_arr = np.empty(0, dtype=np.float32)
    #         pos_arr = np.empty(0, dtype=np.int32) # Clear positions if bursts are problematic
    #     else:
    #         try:
    #             burst_arr = np.concatenate(bursts)
    #         except ValueError: # Handle cases where concatenation might fail (e.g. varying dimensions beyond the first)
    #              # Fallback: if concatenation fails, create empty burst array or handle error appropriately
    #             burst_arr = np.empty(0, dtype=np.float32)
    #             pos_arr = np.empty(0, dtype=np.int32)
    #
    #

    pos_arr   = np.empty(0, dtype=np.int32)
    burst_arr = np.empty(0, dtype=np.float32)

    audio, finalL, finalR = _binaural_beat_core(
        N,
        float(duration),
        float(sample_rate),
        float(initial_offset),
        leftHigh,
        ampL, ampR, baseF, beatF,
        startL, startR, pOF, pOR,
        aODL, aOFL, aODR, aOFR,
        ampOscPhaseOffsetL, ampOscPhaseOffsetR,
        ampOscSkewL, ampOscSkewR,
        fORL, fOFL, fORR, fOFR,
        freqOscSkewL, freqOscSkewR,
        freqOscPhaseOffsetL, freqOscPhaseOffsetR,
        shape_int,
        pos_arr, burst_arr
    )

    pan_min = min(pan_range_min, pan_range_max)
    pan_max = max(pan_range_min, pan_range_max)

    if audio.size and (
        not np.isclose(pan_min, 0.0)
        or not np.isclose(pan_max, 0.0)
        or not np.isclose(pan_freq, 0.0)
    ):
        pan_curve = generate_pan_envelope(
            audio.shape[0],
            sample_rate,
            0.0,
            pan_min,
            pan_max,
            pan_freq,
            pan_type=pan_type,
            initial_phase=pan_phase,
        )
        apply_pan_envelope(audio, pan_curve)

    if bool(params.get("spatialEnable", False)):
        theta_deg, distance_m = generate_azimuth_trajectory(
            duration, sample_rate,
            segments=params.get(
                "spatialTrajectory",
                [{
                    "mode": "oscillate",
                    "center_deg": 0,
                    "extent_deg": 75,
                    "period_s": 20.0,
                    "distance_m": [1.0, 1.4],
                    "seconds": duration,
                }],
            ),
        )
        audio = spatialize_binaural_mid_only(
            audio.astype(np.float32), float(sample_rate),
            theta_deg, distance_m,
            ild_enable=int(params.get("spatialUseIld", 1)),
            ear_angle_deg=float(params.get("spatialEarAngleDeg", 30.0)),
            head_radius_m=float(params.get("spatialHeadRadiusM", 0.0875)),
            itd_scale=float(params.get("spatialItdScale", 1.0)),
            ild_max_db=float(params.get("spatialIldMaxDb", 1.5)),
            ild_xover_hz=float(params.get("spatialIldXoverHz", 700.0)),
            ref_distance_m=float(params.get("spatialRefDistanceM", 1.0)),
            rolloff=float(params.get("spatialRolloff", 1.0)),
            hf_roll_db_per_m=float(params.get("spatialHfRollDbPerM", 0.0)),
            dz_theta_ms=float(params.get("spatialDezipperThetaMs", 60.0)),
            dz_dist_ms=float(params.get("spatialDezipperDistMs", 80.0)),
            decoder=0 if str(params.get("spatialDecoder", "itd_head")).lower() != "foa_cardioid" else 1,
            min_distance_m=float(params.get("spatialMinDistanceM", 0.1)),
            max_deg_per_s=float(params.get("spatialMaxDegPerS", 90.0)),
            max_delay_step_samples=float(params.get("spatialMaxDelayStepSamples", 0.02)),
            interp_mode=int(params.get("spatialInterp", 1)),
        )

    state = {
        'startPhaseL': finalL,
        'startPhaseR': finalR,
        'oscillator_phases': {
            'left': finalL,
            'right': finalR,
        },
    }
    return audio, state


@numba.njit(parallel=True, fastmath=True)
def _binaural_beat_core(
    N, duration, sample_rate, initial_offset,
    leftHigh,
    ampL, ampR, baseF, beatF,
    startL, startR, pOF, pOR,
    aODL, aOFL, aODR, aOFR,
    ampOscPhaseOffsetL, ampOscPhaseOffsetR,
    ampOscSkewL, ampOscSkewR,
    fORL, fOFL, fORR, fOFR,
    freqOscSkewL, freqOscSkewR,
    freqOscPhaseOffsetL, freqOscPhaseOffsetR,
    freqOscShape,
    pos,   # int32[:] start indices
    burst  # float32[:] concatenated glitch samples
):
    if N <= 0 :
        return np.zeros((0,2), dtype=np.float32), 0.0, 0.0
        
    t = np.empty(N, dtype=np.float64)
    dt = duration / N if N > 0 else 0.0
    for i in numba.prange(N): # Use prange for parallel
        t[i] = i * dt + initial_offset

    halfB = beatF / 2.0
    if leftHigh:
        fL_base = baseF + halfB
        fR_base = baseF - halfB
    else:
        fL_base = baseF - halfB
        fR_base = baseF + halfB
    instL = np.empty(N, dtype=np.float64)
    instR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        phaseL = fOFL * t[i] + freqOscPhaseOffsetL/(2*np.pi)
        phaseR = fOFR * t[i] + freqOscPhaseOffsetR/(2*np.pi)
        if freqOscShape == 1:
            vibL = (fORL/2.0) * skewed_triangle_phase(_frac(phaseL), freqOscSkewL)
            vibR = (fORR/2.0) * skewed_triangle_phase(_frac(phaseR), freqOscSkewR)
        else:
            vibL = (fORL/2.0) * skewed_sine_phase(_frac(phaseL), freqOscSkewL)
            vibR = (fORR/2.0) * skewed_sine_phase(_frac(phaseR), freqOscSkewR)
        instL[i] = max(0.0, fL_base + vibL)
        instR[i] = max(0.0, fR_base + vibR)
    
    # If beat frequency is zero we still want any frequency oscillations to
    # apply.  The original code overwrote the instantaneous frequency with the
    # base value, ignoring modulation.  Dropping that behaviour keeps both
    # channels identical while respecting `freqOsc*` parameters.

    phL = np.empty(N, dtype=np.float64)
    phR = np.empty(N, dtype=np.float64)
    curL = startL
    curR = startR
    for i in range(N): # Sequential loop
        curL += 2 * np.pi * instL[i] * dt
        curR += 2 * np.pi * instR[i] * dt
        phL[i] = curL
        phR[i] = curR

    if pOF != 0.0 or pOR != 0.0:
        for i in numba.prange(N): # Use prange
            dphi = (pOR/2.0) * np.sin(2*np.pi*pOF*t[i])
            phL[i] -= dphi
            phR[i] += dphi

    envL = np.empty(N, dtype=np.float64)
    envR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        phaseL = aOFL*t[i] + ampOscPhaseOffsetL/(2*np.pi)
        phaseR = aOFR*t[i] + ampOscPhaseOffsetR/(2*np.pi)
        envL[i] = 1.0 - aODL * (0.5*(1.0 + skewed_sine_phase(_frac(phaseL), ampOscSkewL)))
        envR[i] = 1.0 - aODR * (0.5*(1.0 + skewed_sine_phase(_frac(phaseR), ampOscSkewR)))

    out = np.empty((N,2), dtype=np.float32)
    for i in numba.prange(N): # Use prange
        out[i,0] = np.float32(np.sin(phL[i]) * envL[i] * ampL)
        out[i,1] = np.float32(np.sin(phR[i]) * envR[i] * ampR)

    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0: # ensure burst is not empty
        # Ensure burst.size is divisible by num_bursts.
        # This check implies burst is not empty and pos is not empty.
        if burst.size % num_bursts == 0 :
            L = burst.size // num_bursts
            if L > 0: # Ensure segment length L is positive
                idx = 0
                # This loop should be sequential if bursts can overlap or write to the same output indices.
                # If pos guarantees no overlap, then outer loop b can be prange.
                # Assuming pos are sorted and bursts don't overlap for safety in parallel context.
                # However, a sequential loop is safer unless overlap is explicitly managed.
                # For now, using sequential for the outer loop over bursts.
                for b in range(num_bursts): # Changed from prange to range for safety with additions
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L]
                    for j in range(L):
                        p = start_idx + j
                        if p < N: # Boundary check
                            # Numba does not automatically make += atomic in all contexts.
                            # For safety, this part is tricky with prange on `b` if `pos` allows overlaps.
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L
    return out, curL, curR


from .common import calculate_transition_alpha


def binaural_beat_transition(
    duration, sample_rate=44100, initial_offset=0.0, transition_duration=None, **params
):
    # --- Unpack start/end parameters ---
    startAmpL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    endAmpR = float(params.get('endAmpR', startAmpR))
    startBaseF = float(params.get('startBaseFreq', params.get('baseFreq', 200.0)))
    endBaseF = float(params.get('endBaseFreq', startBaseF))
    startBeatF = float(params.get('startBeatFreq', params.get('beatFreq', 4.0)))
    endBeatF = float(params.get('endBeatFreq', startBeatF))
    leftHigh = bool(params.get('leftHigh', False))

    phase_state = params.get('oscillator_phases', {}) if isinstance(params.get('oscillator_phases'), dict) else {}
    startStartPhaseL = float(params.get('startStartPhaseL', phase_state.get('left', params.get('startPhaseL', 0.0))))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', phase_state.get('right', params.get('startPhaseR', 0.0))))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))
    
    startPOF = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    endPOF = float(params.get('endPhaseOscFreq', startPOF))
    startPOR = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    endPOR = float(params.get('endPhaseOscRange', startPOR))

    startAODL = float(params.get('startAmpOscDepthL', params.get('ampOscDepthL', 0.0)))
    endAODL = float(params.get('endAmpOscDepthL', startAODL))
    startAOFL = float(params.get('startAmpOscFreqL', params.get('ampOscFreqL', 0.0)))
    endAOFL = float(params.get('endAmpOscFreqL', startAOFL))
    startAODR = float(params.get('startAmpOscDepthR', params.get('ampOscDepthR', 0.0)))
    endAODR = float(params.get('endAmpOscDepthR', startAODR))
    startAOFR = float(params.get('startAmpOscFreqR', params.get('ampOscFreqR', 0.0)))
    endAOFR = float(params.get('endAmpOscFreqR', startAOFR))
    startAmpOscSkewL = float(params.get('startAmpOscSkewL', params.get('ampOscSkewL', 0.0)))
    endAmpOscSkewL = float(params.get('endAmpOscSkewL', startAmpOscSkewL))
    startAmpOscSkewR = float(params.get('startAmpOscSkewR', params.get('ampOscSkewR', 0.0)))
    endAmpOscSkewR = float(params.get('endAmpOscSkewR', startAmpOscSkewR))

    startAmpOscPhaseOffsetL = float(params.get('startAmpOscPhaseOffsetL', params.get('ampOscPhaseOffsetL', 0.0)))
    endAmpOscPhaseOffsetL = float(params.get('endAmpOscPhaseOffsetL', startAmpOscPhaseOffsetL))
    startAmpOscPhaseOffsetR = float(params.get('startAmpOscPhaseOffsetR', params.get('ampOscPhaseOffsetR', 0.0)))
    endAmpOscPhaseOffsetR = float(params.get('endAmpOscPhaseOffsetR', startAmpOscPhaseOffsetR))

    startFORL = float(params.get('startFreqOscRangeL', params.get('freqOscRangeL', 0.0)))
    endFORL = float(params.get('endFreqOscRangeL', startFORL))
    startFOFL = float(params.get('startFreqOscFreqL', params.get('freqOscFreqL', 0.0)))
    endFOFL = float(params.get('endFreqOscFreqL', startFOFL))
    startFORR = float(params.get('startFreqOscRangeR', params.get('freqOscRangeR', 0.0)))
    endFORR = float(params.get('endFreqOscRangeR', startFORR))
    startFOFR = float(params.get('startFreqOscFreqR', params.get('freqOscFreqR', 0.0)))
    endFOFR = float(params.get('endFreqOscFreqR', startFOFR))
    startFreqOscSkewL = float(params.get('startFreqOscSkewL', params.get('freqOscSkewL', 0.0)))
    endFreqOscSkewL = float(params.get('endFreqOscSkewL', startFreqOscSkewL))
    startFreqOscSkewR = float(params.get('startFreqOscSkewR', params.get('freqOscSkewR', 0.0)))
    endFreqOscSkewR = float(params.get('endFreqOscSkewR', startFreqOscSkewR))
    startFreqOscPhaseOffsetL = float(params.get('startFreqOscPhaseOffsetL', params.get('freqOscPhaseOffsetL', 0.0)))
    endFreqOscPhaseOffsetL = float(params.get('endFreqOscPhaseOffsetL', startFreqOscPhaseOffsetL))
    startFreqOscPhaseOffsetR = float(params.get('startFreqOscPhaseOffsetR', params.get('freqOscPhaseOffsetR', 0.0)))
    endFreqOscPhaseOffsetR = float(params.get('endFreqOscPhaseOffsetR', startFreqOscPhaseOffsetR))

    start_pan_value = float(params.get('startPan', params.get('pan', 0.0)))
    end_pan_value = float(params.get('endPan', start_pan_value))
    startPanRangeMin = np.clip(
        float(params.get('startPanRangeMin', params.get('panRangeMin', start_pan_value))),
        -1.0,
        1.0,
    )
    endPanRangeMin = np.clip(
        float(params.get('endPanRangeMin', params.get('panRangeMin', end_pan_value))),
        -1.0,
        1.0,
    )
    startPanRangeMax = np.clip(
        float(params.get('startPanRangeMax', params.get('panRangeMax', start_pan_value))),
        -1.0,
        1.0,
    )
    endPanRangeMax = np.clip(
        float(params.get('endPanRangeMax', params.get('panRangeMax', end_pan_value))),
        -1.0,
        1.0,
    )
    if startPanRangeMin > startPanRangeMax:
        startPanRangeMin, startPanRangeMax = startPanRangeMax, startPanRangeMin
    if endPanRangeMin > endPanRangeMax:
        endPanRangeMin, endPanRangeMax = endPanRangeMax, endPanRangeMin
    startPanType = str(params.get('startPanType', params.get('panType', 'linear'))).strip().lower()
    endPanType = str(params.get('endPanType', startPanType)).strip().lower()
    if startPanType != endPanType:
        raise ValueError('Transition pan types must match for binaural_beat voices.')
    startPanFreq = float(params.get('startPanFreq', params.get('panFreq', 0.0)))
    endPanFreq = float(params.get('endPanFreq', startPanFreq))
    startPanPhase = float(params.get('startPanPhase', params.get('panPhase', 0.0)))
    endPanPhase = float(params.get('endPanPhase', startPanPhase))

    freqOscShape = str(params.get('freqOscShape', 'sine')).lower()
    shape_int = 1 if freqOscShape == 'triangle' else 0

    # Glitch parameters (using average for pre-computation)
    s_glitchInterval = float(params.get('startGlitchInterval', params.get('glitchInterval', 0.0)))
    e_glitchInterval = float(params.get('endGlitchInterval', s_glitchInterval))
    avg_glitchInterval = (s_glitchInterval + e_glitchInterval) / 2.0

    s_glitchDur = float(params.get('startGlitchDur', params.get('glitchDur', 0.0)))
    e_glitchDur = float(params.get('endGlitchDur', s_glitchDur))
    avg_glitchDur = (s_glitchDur + e_glitchDur) / 2.0
    
    s_glitchNoiseLevel = float(params.get('startGlitchNoiseLevel', params.get('glitchNoiseLevel', 0.0)))
    e_glitchNoiseLevel = float(params.get('endGlitchNoiseLevel', s_glitchNoiseLevel))
    avg_glitchNoiseLevel = (s_glitchNoiseLevel + e_glitchNoiseLevel) / 2.0

    s_glitchFocusWidth = float(params.get('startGlitchFocusWidth', params.get('glitchFocusWidth', 0.0)))
    e_glitchFocusWidth = float(params.get('endGlitchFocusWidth', s_glitchFocusWidth))
    avg_glitchFocusWidth = (s_glitchFocusWidth + e_glitchFocusWidth) / 2.0
    
    s_glitchFocusExp = float(params.get('startGlitchFocusExp', params.get('glitchFocusExp', 0.0)))
    e_glitchFocusExp = float(params.get('endGlitchFocusExp', s_glitchFocusExp))
    avg_glitchFocusExp = (s_glitchFocusExp + e_glitchFocusExp) / 2.0

    N = int(duration * sample_rate)
    
    # --- Precompute glitch bursts using average parameters ---
    positions = []
    bursts = []
    # Use the average base frequency for glitch shaping if it's also transitional
    # For simplicity, using startBaseF here, or could use (startBaseF + endBaseF) / 2
    glitch_shaping_base_freq = (startBaseF + endBaseF) / 2.0

    if avg_glitchInterval > 0 and avg_glitchDur > 0 and avg_glitchNoiseLevel > 0 and N > 0:
        full_n = int(avg_glitchDur * sample_rate)
        if full_n > 0:
            repeats = int(duration / avg_glitchInterval) if avg_glitchInterval > 0 else 0
            for k in range(1, repeats + 1):
                t_end   = k * avg_glitchInterval
                t_start = max(0.0, t_end - avg_glitchDur)
                i0 = int(t_start * sample_rate)
                i1 = i0 + full_n
                if i1 > N:
                    continue

                white = np.random.standard_normal(full_n)
                S = np.fft.rfft(white)
                freqs_denominator = full_n if full_n != 0 else 1
                freqs = np.arange(S.size) * (sample_rate / freqs_denominator)

                if avg_glitchFocusWidth == 0:
                    gauss = np.ones_like(freqs)
                else:
                    gauss = np.exp(-0.5 * ((freqs - glitch_shaping_base_freq) / avg_glitchFocusWidth) ** avg_glitchFocusExp)
                
                shaped = np.fft.irfft(S * gauss, n=full_n)
                max_abs_shaped = np.max(np.abs(shaped))
                if max_abs_shaped < 1e-16:
                    shaped_normalized = np.zeros_like(shaped)
                else:
                    shaped_normalized = shaped / max_abs_shaped
                ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
                burst_samples = (shaped_normalized * ramp * avg_glitchNoiseLevel).astype(np.float32)
                
                positions.append(i0)
                bursts.append(burst_samples)
    
    if bursts:
        pos_arr   = np.array(positions, dtype=np.int32)
        if any(b.ndim == 0 or b.size == 0 for b in bursts):
            burst_arr = np.empty(0, dtype=np.float32)
            pos_arr = np.empty(0, dtype=np.int32)
        else:
            try:
                burst_arr = np.concatenate(bursts)
            except ValueError:
                burst_arr = np.empty(0, dtype=np.float32)
                pos_arr = np.empty(0, dtype=np.int32)
    else:
        pos_arr   = np.empty(0, dtype=np.int32)
        burst_arr = np.empty(0, dtype=np.float32)

    curve = str(params.get('transition_curve', 'linear')).strip().lower()
    curve_mode = 0
    if curve == 'linear':
        curve_mode = 0
    elif curve == 'logarithmic':
        curve_mode = 1
    elif curve == 'exponential':
        curve_mode = 2
    else:
        curve_mode = -1

    alpha_arr = np.empty(0, dtype=np.float64)
    if curve_mode == -1:
        alpha_arr = calculate_transition_alpha(
            duration, sample_rate, initial_offset, transition_duration, curve
        )

    audio, finalL, finalR = _binaural_beat_transition_core_streaming(
        N, float(duration), float(sample_rate),
        startAmpL, endAmpL, startAmpR, endAmpR,
        startBaseF, endBaseF, startBeatF, endBeatF, leftHigh,
        startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
        startPOF, endPOF, startPOR, endPOR,
        startAODL, endAODL, startAOFL, endAOFL,
        startAODR, endAODR, startAOFR, endAOFR,
        startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
        startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
        startAmpOscSkewL, endAmpOscSkewL,
        startAmpOscSkewR, endAmpOscSkewR,
        startFORL, endFORL, startFOFL, endFOFL,
        startFORR, endFORR, startFOFR, endFOFR,
        startFreqOscSkewL, endFreqOscSkewL,
        startFreqOscSkewR, endFreqOscSkewR,
        startFreqOscPhaseOffsetL, endFreqOscPhaseOffsetL,
        startFreqOscPhaseOffsetR, endFreqOscPhaseOffsetR,
        shape_int,
        pos_arr, burst_arr, # Pass pre-calculated glitches
        float(initial_offset),
        float(transition_duration if transition_duration is not None else -1.0),
        curve_mode,
        alpha_arr
    )

    if audio.size:
        pan_uses_modulation = (
            abs(startPanRangeMin) > 1e-9
            or abs(endPanRangeMin) > 1e-9
            or abs(startPanRangeMax) > 1e-9
            or abs(endPanRangeMax) > 1e-9
            or abs(startPanFreq) > 1e-9
            or abs(endPanFreq) > 1e-9
        )

        if pan_uses_modulation:
            if alpha_arr.size != N:
                alpha_arr = calculate_transition_alpha(
                    duration, sample_rate, initial_offset, transition_duration, curve
                )
            pan_range_min_arr = startPanRangeMin + (endPanRangeMin - startPanRangeMin) * alpha_arr
            pan_range_max_arr = startPanRangeMax + (endPanRangeMax - startPanRangeMax) * alpha_arr
            pan_freq_arr = startPanFreq + (endPanFreq - startPanFreq) * alpha_arr
            pan_phase_arr = startPanPhase + (endPanPhase - startPanPhase) * alpha_arr
            pan_curve = generate_pan_envelope(
                audio.shape[0],
                sample_rate,
                0.0,
                np.minimum(pan_range_min_arr, pan_range_max_arr),
                np.maximum(pan_range_min_arr, pan_range_max_arr),
                pan_freq_arr,
                pan_type=startPanType,
                initial_phase=pan_phase_arr,
            )
            apply_pan_envelope(audio, pan_curve)

    if bool(params.get("spatialEnable", False)):
        theta_deg, distance_m = generate_azimuth_trajectory(
            duration, sample_rate,
            segments=params.get(
                "spatialTrajectory",
                [{
                    "mode": "oscillate",
                    "center_deg": 0,
                    "extent_deg": 75,
                    "period_s": 20.0,
                    "distance_m": [1.0, 1.4],
                    "seconds": duration,
                }],
            ),
        )
        audio = spatialize_binaural_mid_only(
            audio.astype(np.float32), float(sample_rate),
            theta_deg, distance_m,
            ild_enable=int(params.get("spatialUseIld", 1)),
            ear_angle_deg=float(params.get("spatialEarAngleDeg", 30.0)),
            head_radius_m=float(params.get("spatialHeadRadiusM", 0.0875)),
            itd_scale=float(params.get("spatialItdScale", 1.0)),
            ild_max_db=float(params.get("spatialIldMaxDb", 1.5)),
            ild_xover_hz=float(params.get("spatialIldXoverHz", 700.0)),
            ref_distance_m=float(params.get("spatialRefDistanceM", 1.0)),
            rolloff=float(params.get("spatialRolloff", 1.0)),
            hf_roll_db_per_m=float(params.get("spatialHfRollDbPerM", 0.0)),
            dz_theta_ms=float(params.get("spatialDezipperThetaMs", 60.0)),
            dz_dist_ms=float(params.get("spatialDezipperDistMs", 80.0)),
            decoder=0 if str(params.get("spatialDecoder", "itd_head")).lower() != "foa_cardioid" else 1,
            min_distance_m=float(params.get("spatialMinDistanceM", 0.1)),
            max_deg_per_s=float(params.get("spatialMaxDegPerS", 90.0)),
            max_delay_step_samples=float(params.get("spatialMaxDelayStepSamples", 0.02)),
            interp_mode=int(params.get("spatialInterp", 1)),
        )

    state = {
        'startPhaseL': finalL,
        'startPhaseR': finalR,
        'oscillator_phases': {
            'left': finalL,
            'right': finalR,
        },
    }
    return audio, state


@numba.njit(fastmath=True)
def _binaural_beat_transition_core_streaming(
    N, duration, sample_rate,
    startAmpL, endAmpL, startAmpR, endAmpR,
    startBaseF, endBaseF, startBeatF, endBeatF, leftHigh,
    startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
    startPOF, endPOF, startPOR, endPOR,
    startAODL, endAODL, startAOFL, endAOFL,
    startAODR, endAODR, startAOFR, endAOFR,
    startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
    startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
    startAmpOscSkewL, endAmpOscSkewL,
    startAmpOscSkewR, endAmpOscSkewR,
    startFORL, endFORL, startFOFL, endFOFL,
    startFORR, endFORR, startFOFR, endFOFR,
    startFreqOscSkewL, endFreqOscSkewL,
    startFreqOscSkewR, endFreqOscSkewR,
    startFreqOscPhaseOffsetL, endFreqOscPhaseOffsetL,
    startFreqOscPhaseOffsetR, endFreqOscPhaseOffsetR,
    freqOscShape,
    pos, burst, # Glitch arrays (static for this core run)
    initial_offset,
    transition_duration,
    curve_mode,
    alpha_arr
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32), startStartPhaseL, startStartPhaseR

    dt = duration / N

    start_t = initial_offset
    end_t = duration
    if transition_duration >= 0.0:
        end_t = start_t + transition_duration
        if end_t > duration:
            end_t = duration
    trans_time = end_t - start_t
    inv_trans_time = 0.0
    if trans_time > 0.0:
        inv_trans_time = 1.0 / trans_time

    out = np.empty((N, 2), dtype=np.float32)
    phaseL_fo = 0.0
    phaseR_fo = 0.0
    curL = startStartPhaseL
    curR = startStartPhaseR

    for i in range(N):
        if alpha_arr.size == N:
            alpha = alpha_arr[i]
        else:
            t = i * dt
            if trans_time <= 0.0:
                alpha = 1.0 if t >= start_t else 0.0
            else:
                raw = (t - start_t) * inv_trans_time
                if raw < 0.0:
                    raw = 0.0
                elif raw > 1.0:
                    raw = 1.0
                if curve_mode == 1:
                    alpha = 1.0 - (1.0 - raw) * (1.0 - raw)
                elif curve_mode == 2:
                    alpha = raw * raw
                else:
                    alpha = raw

        t = i * dt
        ampL_i = startAmpL + (endAmpL - startAmpL) * alpha
        ampR_i = startAmpR + (endAmpR - startAmpR) * alpha
        baseF_i = startBaseF + (endBaseF - startBaseF) * alpha
        beatF_i = startBeatF + (endBeatF - startBeatF) * alpha
        pOF_i = startPOF + (endPOF - startPOF) * alpha
        pOR_i = startPOR + (endPOR - startPOR) * alpha
        aODL_i = startAODL + (endAODL - startAODL) * alpha
        aOFL_i = startAOFL + (endAOFL - startAOFL) * alpha
        aODR_i = startAODR + (endAODR - startAODR) * alpha
        aOFR_i = startAOFR + (endAOFR - startAOFR) * alpha
        ampOscPhaseOffsetL_i = startAmpOscPhaseOffsetL + (endAmpOscPhaseOffsetL - startAmpOscPhaseOffsetL) * alpha
        ampOscPhaseOffsetR_i = startAmpOscPhaseOffsetR + (endAmpOscPhaseOffsetR - startAmpOscPhaseOffsetR) * alpha
        ampOscSkewL_i = startAmpOscSkewL + (endAmpOscSkewL - startAmpOscSkewL) * alpha
        ampOscSkewR_i = startAmpOscSkewR + (endAmpOscSkewR - startAmpOscSkewR) * alpha
        fORL_i = startFORL + (endFORL - startFORL) * alpha
        fOFL_i = startFOFL + (endFOFL - startFOFL) * alpha
        fORR_i = startFORR + (endFORR - startFORR) * alpha
        fOFR_i = startFOFR + (endFOFR - startFOFR) * alpha
        freqOscSkewL_i = startFreqOscSkewL + (endFreqOscSkewL - startFreqOscSkewL) * alpha
        freqOscSkewR_i = startFreqOscSkewR + (endFreqOscSkewR - startFreqOscSkewR) * alpha
        freqOscPhaseOffsetL_i = startFreqOscPhaseOffsetL + (endFreqOscPhaseOffsetL - startFreqOscPhaseOffsetL) * alpha
        freqOscPhaseOffsetR_i = startFreqOscPhaseOffsetR + (endFreqOscPhaseOffsetR - startFreqOscPhaseOffsetR) * alpha

        halfB_i = beatF_i * 0.5
        if leftHigh:
            fL_base_i = baseF_i + halfB_i
            fR_base_i = baseF_i - halfB_i
        else:
            fL_base_i = baseF_i - halfB_i
            fR_base_i = baseF_i + halfB_i

        phaseL_fo += fOFL_i * dt
        phaseR_fo += fOFR_i * dt
        phL_frac = _frac(phaseL_fo + freqOscPhaseOffsetL_i / (2 * np.pi))
        phR_frac = _frac(phaseR_fo + freqOscPhaseOffsetR_i / (2 * np.pi))
        if freqOscShape == 1:
            vibL_i = (fORL_i / 2.0) * skewed_triangle_phase(phL_frac, freqOscSkewL_i)
            vibR_i = (fORR_i / 2.0) * skewed_triangle_phase(phR_frac, freqOscSkewR_i)
        else:
            vibL_i = (fORL_i / 2.0) * skewed_sine_phase(phL_frac, freqOscSkewL_i)
            vibR_i = (fORR_i / 2.0) * skewed_sine_phase(phR_frac, freqOscSkewR_i)

        instL_candidate = fL_base_i + vibL_i
        instR_candidate = fR_base_i + vibR_i

        instL_i = instL_candidate if instL_candidate > 0.0 else 0.0
        instR_i = instR_candidate if instR_candidate > 0.0 else 0.0

        curL += 2.0 * np.pi * instL_i * dt
        curR += 2.0 * np.pi * instR_i * dt
        phL_i = curL
        phR_i = curR

        if pOF_i != 0.0 or pOR_i != 0.0:
            dphi = (pOR_i / 2.0) * np.sin(2 * np.pi * pOF_i * t)
            phL_i -= dphi
            phR_i += dphi

        phL_env = aOFL_i * t + ampOscPhaseOffsetL_i / (2 * np.pi)
        phR_env = aOFR_i * t + ampOscPhaseOffsetR_i / (2 * np.pi)
        envL_i = 1.0 - aODL_i * (0.5 * (1.0 + skewed_sine_phase(_frac(phL_env), ampOscSkewL_i)))
        envR_i = 1.0 - aODR_i * (0.5 * (1.0 + skewed_sine_phase(_frac(phR_env), ampOscSkewR_i)))

        out[i, 0] = np.float32(np.sin(phL_i) * envL_i * ampL_i)
        out[i, 1] = np.float32(np.sin(phR_i) * envR_i * ampR_i)

    # Add glitch bursts (using the static pos and burst arrays)
    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0:
        if burst.size % num_bursts == 0:
            L_glitch = burst.size // num_bursts
            if L_glitch > 0:
                idx = 0
                for b in range(num_bursts): # Sequential for safety
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L_glitch]
                    for j in range(L_glitch):
                        p = start_idx + j
                        if p < N:
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L_glitch
    return out, curL, curR

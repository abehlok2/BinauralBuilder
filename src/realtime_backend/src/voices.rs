use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use crate::dsp::trig::{cos_lut, sin_lut};
use crate::dsp::{
    build_volume_envelope, pan2, skewed_sine_phase, skewed_triangle_phase, trapezoid_envelope,
};
use crate::models::{StepData, VoiceData};
use crate::noise_params::{NoiseParams, NoiseSweep};
use crate::scheduler::Voice;
use crate::scheduler::{StepVoice, VoiceType};
use crate::streaming_noise::StreamingNoise;

/// Strongly typed wrapper for all available voice implementations.
pub enum VoiceKind {
    BinauralBeat(BinauralBeatVoice),
    BinauralBeatTransition(BinauralBeatTransitionVoice),
    IsochronicTone(IsochronicToneVoice),
    IsochronicToneTransition(IsochronicToneTransitionVoice),
    QamBeat(QamBeatVoice),
    QamBeatTransition(QamBeatTransitionVoice),
    StereoAmIndependent(StereoAmIndependentVoice),
    StereoAmIndependentTransition(StereoAmIndependentTransitionVoice),
    WaveShapeStereoAm(WaveShapeStereoAmVoice),
    WaveShapeStereoAmTransition(WaveShapeStereoAmTransitionVoice),
    SpatialAngleModulation(SpatialAngleModulationVoice),
    SpatialAngleModulationTransition(SpatialAngleModulationTransitionVoice),
    RhythmicWaveshaping(RhythmicWaveshapingVoice),
    RhythmicWaveshapingTransition(RhythmicWaveshapingTransitionVoice),
    SubliminalEncode(SubliminalEncodeVoice),
    VolumeEnvelope(Box<VolumeEnvelopeVoice>),
    NoiseSweptNotch(NoiseSweptNotchVoice),
    NoiseSweptNotchTransition(NoiseSweptNotchTransitionVoice),
}

fn get_f32(params: &HashMap<String, Value>, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn get_bool(params: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

#[derive(Clone, Copy)]
enum TransitionCurve {
    Linear,
    Logarithmic,
    Exponential,
}

#[derive(Clone, Copy)]
enum LfoShape {
    Sine,
    Triangle,
}

impl LfoShape {
    fn from_str(s: &str) -> Self {
        match s {
            "triangle" => LfoShape::Triangle,
            _ => LfoShape::Sine,
        }
    }
}

impl TransitionCurve {
    fn from_str(s: &str) -> Self {
        match s {
            "logarithmic" => TransitionCurve::Logarithmic,
            "exponential" => TransitionCurve::Exponential,
            _ => TransitionCurve::Linear,
        }
    }

    fn apply(self, alpha: f32) -> f32 {
        match self {
            TransitionCurve::Linear => alpha,
            TransitionCurve::Logarithmic => 1.0 - (1.0 - alpha).powi(2),
            TransitionCurve::Exponential => alpha.powi(2),
        }
    }
}

impl VoiceKind {
    fn normalization_peak(&self) -> f32 {
        match self {
            VoiceKind::NoiseSweptNotch(v) => v.cached_peak(),
            VoiceKind::NoiseSweptNotchTransition(v) => v.cached_peak(),
            VoiceKind::VolumeEnvelope(v) => v.normalization_peak(),
            _ => 1.0,
        }
    }
}

/// Wrapper voice that applies a precomputed volume envelope to another voice.
pub struct VolumeEnvelopeVoice {
    inner: Box<VoiceKind>,
    envelope: Vec<f32>,
    idx: usize,
    temp_buf: Vec<f32>,
}

impl VolumeEnvelopeVoice {
    pub fn new(inner: Box<VoiceKind>, envelope: Vec<f32>) -> Self {
        Self {
            inner,
            envelope,
            idx: 0,
            temp_buf: Vec::new(),
        }
    }

    pub fn normalization_peak(&self) -> f32 {
        let inner_peak = self.inner.normalization_peak();
        let env_peak = self.envelope.iter().cloned().fold(1.0f32, f32::max);
        inner_peak * env_peak
    }
}

impl Voice for VolumeEnvelopeVoice {
    fn process(&mut self, output: &mut [f32]) {
        if self.temp_buf.len() != output.len() {
            self.temp_buf.resize(output.len(), 0.0);
        }
        self.temp_buf.fill(0.0);

        self.inner.process(&mut self.temp_buf);
        let frames = output.len() / 2;
        for i in 0..frames {
            let env = if self.idx < self.envelope.len() {
                self.envelope[self.idx]
            } else {
                *self.envelope.last().unwrap_or(&1.0)
            };
            output[i * 2] += self.temp_buf[i * 2] * env;
            output[i * 2 + 1] += self.temp_buf[i * 2 + 1] * env;
            if self.idx < self.envelope.len() {
                self.idx += 1;
            }
        }
    }

    fn is_finished(&self) -> bool {
        self.inner.is_finished() && self.idx >= self.envelope.len()
    }
}

pub struct BinauralBeatVoice {
    amp_l: f32,
    amp_r: f32,
    base_freq: f32,
    beat_freq: f32,
    force_mono: bool,
    left_high: bool,
    start_phase_l: f32,
    start_phase_r: f32,
    amp_osc_depth_l: f32,
    amp_osc_freq_l: f32,
    amp_osc_depth_r: f32,
    amp_osc_freq_r: f32,
    freq_osc_range_l: f32,
    freq_osc_freq_l: f32,
    freq_osc_range_r: f32,
    freq_osc_freq_r: f32,
    freq_osc_skew_l: f32,
    freq_osc_skew_r: f32,
    freq_osc_phase_offset_l: f32,
    freq_osc_phase_offset_r: f32,
    freq_osc_shape: LfoShape,
    amp_osc_phase_offset_l: f32,
    amp_osc_phase_offset_r: f32,
    amp_osc_skew_l: f32,
    amp_osc_skew_r: f32,
    phase_osc_freq: f32,
    phase_osc_range: f32,
    phase_l: f32,
    phase_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct BinauralBeatTransitionVoice {
    start_amp_l: f32,
    end_amp_l: f32,
    start_amp_r: f32,
    end_amp_r: f32,
    start_base_freq: f32,
    end_base_freq: f32,
    start_beat_freq: f32,
    end_beat_freq: f32,
    start_force_mono: bool,
    end_force_mono: bool,
    left_high: bool,
    start_start_phase_l: f32,
    end_start_phase_l: f32,
    start_start_phase_r: f32,
    end_start_phase_r: f32,
    start_phase_osc_freq: f32,
    end_phase_osc_freq: f32,
    start_phase_osc_range: f32,
    end_phase_osc_range: f32,
    start_amp_osc_depth_l: f32,
    end_amp_osc_depth_l: f32,
    start_amp_osc_freq_l: f32,
    end_amp_osc_freq_l: f32,
    start_amp_osc_depth_r: f32,
    end_amp_osc_depth_r: f32,
    start_amp_osc_freq_r: f32,
    end_amp_osc_freq_r: f32,
    start_amp_osc_phase_offset_l: f32,
    end_amp_osc_phase_offset_l: f32,
    start_amp_osc_phase_offset_r: f32,
    end_amp_osc_phase_offset_r: f32,
    start_freq_osc_range_l: f32,
    end_freq_osc_range_l: f32,
    start_freq_osc_freq_l: f32,
    end_freq_osc_freq_l: f32,
    start_freq_osc_range_r: f32,
    end_freq_osc_range_r: f32,
    start_freq_osc_freq_r: f32,
    end_freq_osc_freq_r: f32,
    start_freq_osc_skew_l: f32,
    end_freq_osc_skew_l: f32,
    start_freq_osc_skew_r: f32,
    end_freq_osc_skew_r: f32,
    start_freq_osc_phase_offset_l: f32,
    end_freq_osc_phase_offset_l: f32,
    start_freq_osc_phase_offset_r: f32,
    end_freq_osc_phase_offset_r: f32,
    start_amp_osc_skew_l: f32,
    end_amp_osc_skew_l: f32,
    start_amp_osc_skew_r: f32,
    end_amp_osc_skew_r: f32,
    freq_osc_shape: LfoShape,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    sample_rate: f32,
    remaining_samples: usize,
    phase_l: f32,
    phase_r: f32,
    sample_idx: usize,
    duration: f32,
}

pub struct IsochronicToneVoice {
    amp_l: f32,
    amp_r: f32,
    base_freq: f32,
    beat_freq: f32,
    force_mono: bool,
    start_phase_l: f32,
    start_phase_r: f32,
    amp_osc_depth_l: f32,
    amp_osc_freq_l: f32,
    amp_osc_depth_r: f32,
    amp_osc_freq_r: f32,
    freq_osc_range_l: f32,
    freq_osc_freq_l: f32,
    freq_osc_range_r: f32,
    freq_osc_freq_r: f32,
    freq_osc_skew_l: f32,
    freq_osc_skew_r: f32,
    freq_osc_phase_offset_l: f32,
    freq_osc_phase_offset_r: f32,
    amp_osc_phase_offset_l: f32,
    amp_osc_phase_offset_r: f32,
    amp_osc_skew_l: f32,
    amp_osc_skew_r: f32,
    phase_osc_freq: f32,
    phase_osc_range: f32,
    ramp_percent: f32,
    gap_percent: f32,
    pan: f32,
    pan_range_min: f32,
    pan_range_max: f32,
    pan_freq: f32,
    pan_phase: f32,
    phase_l: f32,
    phase_r: f32,
    beat_phase: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct IsochronicToneTransitionVoice {
    start_amp_l: f32,
    end_amp_l: f32,
    start_amp_r: f32,
    end_amp_r: f32,
    start_base_freq: f32,
    end_base_freq: f32,
    start_beat_freq: f32,
    end_beat_freq: f32,
    start_force_mono: bool,
    end_force_mono: bool,
    start_start_phase_l: f32,
    end_start_phase_l: f32,
    start_start_phase_r: f32,
    end_start_phase_r: f32,
    start_phase_osc_freq: f32,
    end_phase_osc_freq: f32,
    start_phase_osc_range: f32,
    end_phase_osc_range: f32,
    start_amp_osc_depth_l: f32,
    end_amp_osc_depth_l: f32,
    start_amp_osc_freq_l: f32,
    end_amp_osc_freq_l: f32,
    start_amp_osc_depth_r: f32,
    end_amp_osc_depth_r: f32,
    start_amp_osc_freq_r: f32,
    end_amp_osc_freq_r: f32,
    start_amp_osc_phase_offset_l: f32,
    end_amp_osc_phase_offset_l: f32,
    start_amp_osc_phase_offset_r: f32,
    end_amp_osc_phase_offset_r: f32,
    start_freq_osc_range_l: f32,
    end_freq_osc_range_l: f32,
    start_freq_osc_freq_l: f32,
    end_freq_osc_freq_l: f32,
    start_freq_osc_range_r: f32,
    end_freq_osc_range_r: f32,
    start_freq_osc_freq_r: f32,
    end_freq_osc_freq_r: f32,
    start_freq_osc_skew_l: f32,
    end_freq_osc_skew_l: f32,
    start_freq_osc_skew_r: f32,
    end_freq_osc_skew_r: f32,
    start_freq_osc_phase_offset_l: f32,
    end_freq_osc_phase_offset_l: f32,
    start_freq_osc_phase_offset_r: f32,
    end_freq_osc_phase_offset_r: f32,
    start_amp_osc_skew_l: f32,
    end_amp_osc_skew_l: f32,
    start_amp_osc_skew_r: f32,
    end_amp_osc_skew_r: f32,
    start_ramp_percent: f32,
    end_ramp_percent: f32,
    start_gap_percent: f32,
    end_gap_percent: f32,
    pan: f32,
    start_pan_range_min: f32,
    end_pan_range_min: f32,
    start_pan_range_max: f32,
    end_pan_range_max: f32,
    start_pan_freq: f32,
    end_pan_freq: f32,
    start_pan_phase: f32,
    end_pan_phase: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    sample_rate: f32,
    remaining_samples: usize,
    phase_l: f32,
    phase_r: f32,
    beat_phase: f32,
    sample_idx: usize,
    duration: f32,
}

pub struct QamBeatVoice {
    amp_l: f32,
    amp_r: f32,
    base_freq_l: f32,
    base_freq_r: f32,
    qam_am_freq_l: f32,
    qam_am_depth_l: f32,
    qam_am_phase_offset_l: f32,
    qam_am_freq_r: f32,
    qam_am_depth_r: f32,
    qam_am_phase_offset_r: f32,
    qam_am2_freq_l: f32,
    qam_am2_depth_l: f32,
    qam_am2_phase_offset_l: f32,
    qam_am2_freq_r: f32,
    qam_am2_depth_r: f32,
    qam_am2_phase_offset_r: f32,
    mod_shape_l: f32,
    mod_shape_r: f32,
    cross_mod_depth: f32,
    cross_mod_delay_samples: usize,
    harmonic_depth: f32,
    harmonic_ratio: f32,
    sub_harmonic_freq: f32,
    sub_harmonic_depth: f32,
    phase_osc_freq: f32,
    phase_osc_range: f32,
    phase_osc_phase_offset: f32,
    start_phase_l: f32,
    start_phase_r: f32,
    beating_sidebands: bool,
    sideband_offset: f32,
    sideband_depth: f32,
    attack_time: f32,
    release_time: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
    phase_l: f32,
    phase_r: f32,
    cross_env_l: Vec<f32>,
    cross_env_r: Vec<f32>,
    cross_idx: usize,
}

pub struct QamBeatTransitionVoice {
    start_amp_l: f32,
    end_amp_l: f32,
    start_amp_r: f32,
    end_amp_r: f32,
    start_base_freq_l: f32,
    end_base_freq_l: f32,
    start_base_freq_r: f32,
    end_base_freq_r: f32,
    start_qam_am_freq_l: f32,
    end_qam_am_freq_l: f32,
    start_qam_am_depth_l: f32,
    end_qam_am_depth_l: f32,
    start_qam_am_freq_r: f32,
    end_qam_am_freq_r: f32,
    start_qam_am_depth_r: f32,
    end_qam_am_depth_r: f32,
    start_qam_am_phase_offset_l: f32,
    end_qam_am_phase_offset_l: f32,
    start_qam_am_phase_offset_r: f32,
    end_qam_am_phase_offset_r: f32,
    start_qam_am2_freq_l: f32,
    end_qam_am2_freq_l: f32,
    start_qam_am2_depth_l: f32,
    end_qam_am2_depth_l: f32,
    start_qam_am2_freq_r: f32,
    end_qam_am2_freq_r: f32,
    start_qam_am2_depth_r: f32,
    end_qam_am2_depth_r: f32,
    start_qam_am2_phase_offset_l: f32,
    end_qam_am2_phase_offset_l: f32,
    start_qam_am2_phase_offset_r: f32,
    end_qam_am2_phase_offset_r: f32,
    start_mod_shape_l: f32,
    end_mod_shape_l: f32,
    start_mod_shape_r: f32,
    end_mod_shape_r: f32,
    start_cross_mod_depth: f32,
    end_cross_mod_depth: f32,
    cross_mod_delay_samples: usize,
    harmonic_ratio: f32,
    start_harmonic_depth: f32,
    end_harmonic_depth: f32,
    start_sub_harmonic_freq: f32,
    end_sub_harmonic_freq: f32,
    start_sub_harmonic_depth: f32,
    end_sub_harmonic_depth: f32,
    start_phase_osc_freq: f32,
    end_phase_osc_freq: f32,
    start_phase_osc_range: f32,
    end_phase_osc_range: f32,
    start_start_phase_l: f32,
    end_start_phase_l: f32,
    start_start_phase_r: f32,
    end_start_phase_r: f32,
    phase_osc_phase_offset: f32,
    beating_sidebands: bool,
    sideband_offset: f32,
    sideband_depth: f32,
    attack_time: f32,
    release_time: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
    phase_l: f32,
    phase_r: f32,
    cross_env_l: Vec<f32>,
    cross_env_r: Vec<f32>,
    cross_idx: usize,
}

pub struct StereoAmIndependentVoice {
    amp: f32,
    carrier_freq: f32,
    stereo_width_hz: f32,
    mod_freq_l: f32,
    mod_depth_l: f32,
    mod_phase_l: f32,
    mod_freq_r: f32,
    mod_depth_r: f32,
    mod_phase_r: f32,
    phase_carrier_l: f32,
    phase_carrier_r: f32,
    phase_mod_l: f32,
    phase_mod_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct StereoAmIndependentTransitionVoice {
    amp: f32,
    start_carrier_freq: f32,
    end_carrier_freq: f32,
    start_stereo_width_hz: f32,
    end_stereo_width_hz: f32,
    start_mod_freq_l: f32,
    end_mod_freq_l: f32,
    start_mod_depth_l: f32,
    end_mod_depth_l: f32,
    mod_phase_l: f32,
    start_mod_freq_r: f32,
    end_mod_freq_r: f32,
    start_mod_depth_r: f32,
    end_mod_depth_r: f32,
    mod_phase_r: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    phase_carrier_l: f32,
    phase_carrier_r: f32,
    phase_mod_l: f32,
    phase_mod_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
}

pub struct WaveShapeStereoAmVoice {
    amp: f32,
    carrier_freq: f32,
    shape_mod_freq: f32,
    shape_mod_depth: f32,
    shape_amount: f32,
    stereo_mod_freq_l: f32,
    stereo_mod_depth_l: f32,
    stereo_mod_phase_l: f32,
    stereo_mod_freq_r: f32,
    stereo_mod_depth_r: f32,
    stereo_mod_phase_r: f32,
    phase_carrier: f32,
    phase_shape: f32,
    phase_stereo_l: f32,
    phase_stereo_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct WaveShapeStereoAmTransitionVoice {
    amp: f32,
    start_carrier_freq: f32,
    end_carrier_freq: f32,
    start_shape_mod_freq: f32,
    end_shape_mod_freq: f32,
    start_shape_mod_depth: f32,
    end_shape_mod_depth: f32,
    start_shape_amount: f32,
    end_shape_amount: f32,
    start_stereo_mod_freq_l: f32,
    end_stereo_mod_freq_l: f32,
    start_stereo_mod_depth_l: f32,
    end_stereo_mod_depth_l: f32,
    stereo_mod_phase_l: f32,
    start_stereo_mod_freq_r: f32,
    end_stereo_mod_freq_r: f32,
    start_stereo_mod_depth_r: f32,
    end_stereo_mod_depth_r: f32,
    stereo_mod_phase_r: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    phase_carrier: f32,
    phase_shape: f32,
    phase_stereo_l: f32,
    phase_stereo_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
}

pub struct SpatialAngleModulationVoice {
    amp: f32,
    carrier_freq: f32,
    beat_freq: f32,
    path_radius: f32,
    carrier_phase: f32,
    spatial_phase: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct SpatialAngleModulationTransitionVoice {
    amp: f32,
    start_carrier_freq: f32,
    end_carrier_freq: f32,
    start_beat_freq: f32,
    end_beat_freq: f32,
    start_path_radius: f32,
    end_path_radius: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    carrier_phase: f32,
    spatial_phase: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
}

pub struct RhythmicWaveshapingVoice {
    amp: f32,
    carrier_freq: f32,
    mod_freq: f32,
    mod_depth: f32,
    shape_amount: f32,
    pan: f32,
    carrier_phase: f32,
    lfo_phase: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
}

pub struct RhythmicWaveshapingTransitionVoice {
    amp: f32,
    start_carrier_freq: f32,
    end_carrier_freq: f32,
    start_mod_freq: f32,
    end_mod_freq: f32,
    start_mod_depth: f32,
    end_mod_depth: f32,
    start_shape_amount: f32,
    end_shape_amount: f32,
    pan: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    carrier_phase: f32,
    lfo_phase: f32,
    sample_rate: f32,
    remaining_samples: usize,
    sample_idx: usize,
    duration: f32,
}

pub struct SubliminalEncodeVoice {
    samples: Vec<f32>,
    position: usize,
    remaining_samples: usize,
}

/// Voice wrapper for noise generation with swept notch filtering (static mode).
pub struct NoiseSweptNotchVoice {
    generator: StreamingNoise,
    amp: f32,
    remaining_samples: usize,
    cached_peak: f32,
}

/// Voice wrapper for noise generation with swept notch filtering (transition mode).
pub struct NoiseSweptNotchTransitionVoice {
    generator: StreamingNoise,
    amp: f32,
    remaining_samples: usize,
    cached_peak: f32,
    // Transition parameters for interpolation
    start_lfo_freq: f32,
    end_lfo_freq: f32,
    start_lfo_phase_offset_deg: f32,
    end_lfo_phase_offset_deg: f32,
    start_intra_phase_offset_deg: f32,
    end_intra_phase_offset_deg: f32,
    start_sweeps: Vec<(f32, f32)>,
    end_sweeps: Vec<(f32, f32)>,
    start_qs: Vec<f32>,
    end_qs: Vec<f32>,
    duration_samples: usize,
    sample_idx: usize,
}

impl SubliminalEncodeVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let carrier = get_f32(params, "carrierFreq", 17500.0).clamp(15000.0, 20000.0);
        let amp = get_f32(params, "amp", 0.5);
        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("sequence")
            .to_lowercase();

        let mut paths: Vec<String> = Vec::new();
        if let Some(v) = params.get("audio_paths") {
            if let Some(s) = v.as_str() {
                for p in s.split(';') {
                    let p = p.trim();
                    if !p.is_empty() {
                        paths.push(p.to_string());
                    }
                }
            } else if let Some(arr) = v.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        paths.push(s.to_string());
                    }
                }
            }
        }
        if paths.is_empty() {
            if let Some(p) = params.get("audio_path").and_then(|v| v.as_str()) {
                paths.push(p.to_string());
            }
        }

        let mut segments: Vec<Vec<f32>> = Vec::new();
        for p in &paths {
            if let Some(seg) = load_and_modulate(p, sample_rate as u32, carrier) {
                segments.push(seg);
            }
        }

        let total_samples = (duration * sample_rate) as usize;
        if segments.is_empty() || total_samples == 0 {
            return Self {
                samples: vec![0.0; total_samples * 2],
                position: 0,
                remaining_samples: total_samples,
            };
        }

        let mut mono = vec![0.0f32; total_samples];
        if mode == "stack" {
            for seg in &segments {
                for i in 0..total_samples {
                    mono[i] += seg[i % seg.len()];
                }
            }
            if segments.len() > 1 {
                for v in &mut mono {
                    *v /= segments.len() as f32;
                }
            }
        } else {
            let mut pos = 0usize;
            let mut idx = 0usize;
            let pause = sample_rate as usize;
            while pos < total_samples {
                let seg = &segments[idx % segments.len()];
                let len = seg.len().min(total_samples - pos);
                mono[pos..pos + len].copy_from_slice(&seg[..len]);
                pos += len;
                if pos >= total_samples {
                    break;
                }
                let pad = pause.min(total_samples - pos);
                pos += pad;
                idx += 1;
            }
        }

        let max_val = mono.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if max_val > 0.0 {
            for v in &mut mono {
                *v = *v / max_val * amp;
            }
        }

        let mut samples = Vec::with_capacity(total_samples * 2);
        for s in &mono {
            samples.push(*s);
            samples.push(*s);
        }

        Self {
            samples,
            position: 0,
            remaining_samples: total_samples,
        }
    }
}

fn load_audio_file(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no default track")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("unknown channel count")?
        .count();

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        };
        let decoded = decoder.decode(&packet)?;
        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let sbuf = sample_buf.as_mut().unwrap();
        sbuf.copy_interleaved_ref(decoded);
        let data = sbuf.samples();
        for frame in data.chunks(channels) {
            let sum: f32 = frame.iter().copied().sum();
            samples.push(sum / channels as f32);
        }
    }
    Ok((samples, sample_rate))
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let duration = input.len() as f64 / src_rate as f64;
    let n_out = (duration * dst_rate as f64).round() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t = i as f64 / dst_rate as f64;
        let pos = t * src_rate as f64;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let x0 = input[idx];
        let x1 = if idx + 1 < input.len() {
            input[idx + 1]
        } else {
            x0
        };
        out.push(((1.0 - frac) * x0 as f64 + frac * x1 as f64) as f32);
    }
    out
}

fn load_and_modulate(path: &str, sample_rate: u32, carrier: f32) -> Option<Vec<f32>> {
    let (mut data, sr) = load_audio_file(path).ok()?;
    if sr != sample_rate {
        data = resample_linear(&data, sr, sample_rate);
    }
    if data.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(data.len());
    for (i, sample) in data.into_iter().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let m = sin_lut(2.0 * std::f32::consts::PI * carrier * t);
        out.push(sample * m);
    }
    let max_val = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_val > 0.0 {
        for v in &mut out {
            *v /= max_val;
        }
    }
    Some(out)
}

/// Parse sweeps from JSON params - handles multiple formats:
/// 1. Python dict format: [{start_min, end_min, start_max, end_max, start_q, end_q, start_casc, end_casc}, ...]
/// 2. Tuple format: [[min, max], ...]
/// 3. Simple dict format: [{min, max, q, cascade}, ...]
fn parse_sweeps_as_noise_sweeps(
    params: &HashMap<String, Value>,
    key: &str,
) -> Option<Vec<NoiseSweep>> {
    let sweeps = params.get(key)?;
    let arr = sweeps.as_array()?;

    if arr.is_empty() {
        return None;
    }

    // Try to detect format by checking first element
    let first = arr.first()?;

    // Check if it's the full dict format with start_min/end_min keys (Python preset format)
    if first.get("start_min").is_some() || first.get("end_min").is_some() {
        let parsed: Vec<NoiseSweep> = arr
            .iter()
            .filter_map(|item| {
                let obj = item.as_object()?;
                // Parse with explicit field extraction to handle all variations
                let start_min = obj
                    .get("start_min")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1000.0) as f32;
                let end_min = obj
                    .get("end_min")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(start_min as f64) as f32;
                let start_max = obj
                    .get("start_max")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(10000.0) as f32;
                let end_max = obj
                    .get("end_max")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(start_max as f64) as f32;
                let start_q = obj.get("start_q").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32;
                let end_q = obj
                    .get("end_q")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(start_q as f64) as f32;
                let start_casc = obj
                    .get("start_casc")
                    .and_then(|v| v.as_i64())
                    .or_else(|| {
                        obj.get("start_casc")
                            .and_then(|v| v.as_f64())
                            .map(|f| f as i64)
                    })
                    .unwrap_or(10) as usize;
                let end_casc = obj
                    .get("end_casc")
                    .and_then(|v| v.as_i64())
                    .or_else(|| {
                        obj.get("end_casc")
                            .and_then(|v| v.as_f64())
                            .map(|f| f as i64)
                    })
                    .unwrap_or(start_casc as i64) as usize;

                Some(NoiseSweep {
                    start_min,
                    end_min,
                    start_max,
                    end_max,
                    start_q,
                    end_q,
                    start_casc,
                    end_casc,
                })
            })
            .collect();

        if !parsed.is_empty() {
            return Some(parsed);
        }
    }

    // Check if it's the simple dict format with min/max/q/cascade keys
    if first.get("min").is_some() || first.get("max").is_some() {
        let parsed: Vec<NoiseSweep> = arr
            .iter()
            .filter_map(|item| {
                let obj = item.as_object()?;
                let min = obj.get("min").and_then(|v| v.as_f64()).unwrap_or(1000.0) as f32;
                let max = obj.get("max").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
                let q = obj.get("q").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32;
                let casc = obj
                    .get("cascade")
                    .and_then(|v| v.as_i64())
                    .or_else(|| {
                        obj.get("cascade")
                            .and_then(|v| v.as_f64())
                            .map(|f| f as i64)
                    })
                    .unwrap_or(10) as usize;

                Some(NoiseSweep {
                    start_min: min,
                    end_min: min,
                    start_max: max,
                    end_max: max,
                    start_q: q,
                    end_q: q,
                    start_casc: casc,
                    end_casc: casc,
                })
            })
            .collect();

        if !parsed.is_empty() {
            return Some(parsed);
        }
    }

    // Fall back to tuple format [[min, max], ...]
    if first.is_array() {
        let parsed: Vec<NoiseSweep> = arr
            .iter()
            .filter_map(|item| {
                let tuple = item.as_array()?;
                if tuple.len() >= 2 {
                    let min = tuple[0].as_f64()? as f32;
                    let max = tuple[1].as_f64()? as f32;
                    return Some(NoiseSweep {
                        start_min: min,
                        end_min: min,
                        start_max: max,
                        end_max: max,
                        start_q: 25.0,
                        end_q: 25.0,
                        start_casc: 10,
                        end_casc: 10,
                    });
                }
                None
            })
            .collect();

        if !parsed.is_empty() {
            return Some(parsed);
        }
    }

    None
}

/// Legacy function for backward compatibility - returns tuples only
fn parse_sweeps_from_params(params: &HashMap<String, Value>, key: &str) -> Vec<(f32, f32)> {
    if let Some(sweeps) = params.get(key) {
        if let Some(arr) = sweeps.as_array() {
            return arr
                .iter()
                .filter_map(|item| {
                    if let Some(tuple) = item.as_array() {
                        if tuple.len() >= 2 {
                            let min = tuple[0].as_f64()? as f32;
                            let max = tuple[1].as_f64()? as f32;
                            return Some((min, max));
                        }
                    }
                    None
                })
                .collect();
        }
    }
    vec![] // no default sweeps - steady state noise should not have sweeps
}

/// Parse Q values from JSON params - handles both single value and list
fn parse_q_from_params(params: &HashMap<String, Value>, key: &str, count: usize) -> Vec<f32> {
    if let Some(q) = params.get(key) {
        if let Some(n) = q.as_f64() {
            return vec![n as f32; count];
        }
        if let Some(arr) = q.as_array() {
            return arr
                .iter()
                .filter_map(|v| v.as_f64().map(|n| n as f32))
                .collect();
        }
    }
    vec![25.0; count] // default
}

/// Parse cascade values from JSON params - handles both single value and list
fn parse_casc_from_params(params: &HashMap<String, Value>, key: &str, count: usize) -> Vec<usize> {
    if let Some(c) = params.get(key) {
        if let Some(n) = c.as_i64() {
            return vec![n.max(1) as usize; count];
        }
        if let Some(n) = c.as_f64() {
            return vec![(n as i64).max(1) as usize; count];
        }
        if let Some(arr) = c.as_array() {
            return arr
                .iter()
                .filter_map(|v| {
                    v.as_i64()
                        .or_else(|| v.as_f64().map(|f| f as i64))
                        .map(|n| n.max(1) as usize)
                })
                .collect();
        }
    }
    vec![10; count] // default
}

/// Convert JSON params to NoiseParams struct for StreamingNoise initialization
fn noise_params_from_json(
    params: &HashMap<String, Value>,
    duration: f32,
    sample_rate: u32,
    is_transition: bool,
) -> NoiseParams {
    let mut color_params: HashMap<String, Value> = params
        .get("noise_parameters")
        .or_else(|| params.get("color_params"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();

    let noise_name = color_params
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| params.get("noise_type").and_then(|v| v.as_str()))
        .unwrap_or("pink")
        .to_string();

    color_params
        .entry("name".to_string())
        .or_insert(Value::String(noise_name.clone()));

    let lfo_waveform = params
        .get("lfo_waveform")
        .and_then(|v| v.as_str())
        .unwrap_or("sine")
        .to_string();

    let lfo_freq = get_f32(params, "lfo_freq", 1.0 / 12.0);
    let start_lfo_freq = get_f32(params, "start_lfo_freq", lfo_freq);
    let end_lfo_freq = get_f32(params, "end_lfo_freq", start_lfo_freq);

    let start_lfo_phase_offset = get_f32(params, "start_lfo_phase_offset_deg", 0.0);
    let end_lfo_phase_offset = get_f32(params, "end_lfo_phase_offset_deg", start_lfo_phase_offset);
    let start_intra_phase_offset = get_f32(params, "start_intra_phase_offset_deg", 0.0);
    let end_intra_phase_offset = get_f32(
        params,
        "end_intra_phase_offset_deg",
        start_intra_phase_offset,
    );

    // Parse sweeps - try the new comprehensive parser first which handles Python dict format
    // with start_min/end_min/start_q/end_q/start_casc/end_casc keys embedded in each sweep
    let sweeps_key = if is_transition {
        "start_sweeps"
    } else {
        "sweeps"
    };

    let sweeps: Vec<NoiseSweep> =
        if let Some(parsed_sweeps) = parse_sweeps_as_noise_sweeps(params, sweeps_key) {
            // Successfully parsed sweeps with embedded Q and cascade values
            parsed_sweeps
        } else {
            // Fall back to legacy parsing: separate sweeps tuples + Q/cascade arrays
            let sweeps_tuples = parse_sweeps_from_params(params, sweeps_key);
            let sweep_count = sweeps_tuples.len();

            // Parse Q and cascade values from separate keys
            let q_key = if is_transition { "start_q" } else { "notch_q" };
            let casc_key = if is_transition { "start_casc" } else { "casc" };
            let qs = parse_q_from_params(params, q_key, sweep_count);
            let cascs = parse_casc_from_params(params, casc_key, sweep_count);

            // Build NoiseSweep structs from separate components
            sweeps_tuples
                .iter()
                .enumerate()
                .map(|(i, (min, max))| NoiseSweep {
                    start_min: *min,
                    end_min: *min,
                    start_max: *max,
                    end_max: *max,
                    start_q: qs.get(i).copied().unwrap_or(25.0),
                    end_q: qs.get(i).copied().unwrap_or(25.0),
                    start_casc: cascs.get(i).copied().unwrap_or(10),
                    end_casc: cascs.get(i).copied().unwrap_or(10),
                })
                .collect()
        };

    let noise_params = NoiseParams {
        duration_seconds: duration,
        sample_rate,
        noise_parameters: color_params,
        seed: get_i64_opt(params, "seed"),
        lfo_waveform,
        transition: is_transition,
        lfo_freq,
        start_lfo_freq,
        end_lfo_freq,
        sweeps,
        start_lfo_phase_offset_deg: start_lfo_phase_offset,
        end_lfo_phase_offset_deg: end_lfo_phase_offset,
        start_intra_phase_offset_deg: start_intra_phase_offset,
        end_intra_phase_offset_deg: end_intra_phase_offset,
        initial_offset: get_f32(params, "initial_offset", 0.0),
        post_offset: get_f32(params, "post_offset", 0.0),
        input_audio_path: params
            .get("input_audio_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        exponent: get_f32_opt(params, "exponent"),
        high_exponent: get_f32_opt(params, "high_exponent"),
        distribution_curve: get_f32_opt(params, "distribution_curve"),
        lowcut: get_f32_opt(params, "lowcut"),
        highcut: get_f32_opt(params, "highcut"),
        amplitude: get_f32_opt(params, "amplitude"),
        start_time: get_f32(params, "start_time", 0.0),
        fade_in: get_f32(params, "fade_in", 0.0),
        fade_out: get_f32(params, "fade_out", 0.0),
        amp_envelope: Vec::new(),
        static_notches: Vec::new(),
    };

    crate::noise_params::apply_color_params(noise_params)
}

fn get_f32_opt(params: &HashMap<String, Value>, key: &str) -> Option<f32> {
    params.get(key).and_then(|v| v.as_f64()).map(|n| n as f32)
}

fn get_i64_opt(params: &HashMap<String, Value>, key: &str) -> Option<i64> {
    params.get(key).and_then(|v| v.as_i64())
}

impl NoiseSweptNotchVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 1.0);
        let noise_params = noise_params_from_json(params, duration, sample_rate as u32, false);

        let calibration_frames = ((duration * sample_rate) as usize)
            .min((sample_rate as usize) * 10)
            .max(1);
        let (generator, cached_peak) = StreamingNoise::new_with_calibrated_peak(
            &noise_params,
            sample_rate as u32,
            calibration_frames,
        );
        let total_samples = (duration * sample_rate) as usize;

        Self {
            generator,
            amp,
            remaining_samples: total_samples,
            cached_peak,
        }
    }

    pub fn cached_peak(&self) -> f32 {
        self.cached_peak
    }
}

impl Voice for NoiseSweptNotchVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        let to_process = frames.min(self.remaining_samples);

        if to_process == 0 {
            return;
        }

        // Generate noise into a temporary buffer
        let mut scratch = vec![0.0f32; to_process * 2];
        self.generator.generate(&mut scratch);

        // Mix into output with amplitude
        for i in 0..to_process {
            output[i * 2] += scratch[i * 2] * self.amp;
            output[i * 2 + 1] += scratch[i * 2 + 1] * self.amp;
        }

        self.remaining_samples = self.remaining_samples.saturating_sub(to_process);
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl NoiseSweptNotchTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 1.0);

        // Parse transition parameters
        let start_lfo_freq = get_f32(
            params,
            "start_lfo_freq",
            get_f32(params, "lfo_freq", 1.0 / 12.0),
        );
        let end_lfo_freq = get_f32(params, "end_lfo_freq", start_lfo_freq);
        let start_lfo_phase_offset_deg = get_f32(params, "start_lfo_phase_offset_deg", 0.0);
        let end_lfo_phase_offset_deg = get_f32(
            params,
            "end_lfo_phase_offset_deg",
            start_lfo_phase_offset_deg,
        );
        let start_intra_phase_offset_deg = get_f32(params, "start_intra_phase_offset_deg", 0.0);
        let end_intra_phase_offset_deg = get_f32(
            params,
            "end_intra_phase_offset_deg",
            start_intra_phase_offset_deg,
        );

        // Parse sweep ranges
        let start_sweeps = parse_sweeps_from_params(params, "start_sweeps");
        let end_sweeps = parse_sweeps_from_params(params, "end_sweeps");
        let sweep_count = start_sweeps.len();

        // Parse Q values
        let start_qs = parse_q_from_params(params, "start_q", sweep_count);
        let end_qs = parse_q_from_params(params, "end_q", sweep_count);

        // Create initial NoiseParams using start values
        let noise_params = noise_params_from_json(params, duration, sample_rate as u32, true);
        let calibration_frames = ((duration * sample_rate) as usize)
            .min((sample_rate as usize) * 10)
            .max(1);
        let (generator, cached_peak) = StreamingNoise::new_with_calibrated_peak(
            &noise_params,
            sample_rate as u32,
            calibration_frames,
        );
        let total_samples = (duration * sample_rate) as usize;

        Self {
            generator,
            amp,
            remaining_samples: total_samples,
            cached_peak,
            start_lfo_freq,
            end_lfo_freq,
            start_lfo_phase_offset_deg,
            end_lfo_phase_offset_deg,
            start_intra_phase_offset_deg,
            end_intra_phase_offset_deg,
            start_sweeps,
            end_sweeps,
            start_qs,
            end_qs,
            duration_samples: total_samples,
            sample_idx: 0,
        }
    }

    pub fn cached_peak(&self) -> f32 {
        self.cached_peak
    }
}

impl Voice for NoiseSweptNotchTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        let to_process = frames.min(self.remaining_samples);

        if to_process == 0 {
            return;
        }

        // Generate noise into a temporary buffer
        // Note: For full transition support, the StreamingNoise generator would need
        // to be modified to interpolate parameters. For now, we generate using the
        // initial parameters which provides basic noise functionality.
        let mut scratch = vec![0.0f32; to_process * 2];
        self.generator.generate(&mut scratch);

        // Mix into output with amplitude
        for i in 0..to_process {
            output[i * 2] += scratch[i * 2] * self.amp;
            output[i * 2 + 1] += scratch[i * 2 + 1] * self.amp;
        }

        self.remaining_samples = self.remaining_samples.saturating_sub(to_process);
        self.sample_idx += to_process;
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl BinauralBeatVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp_l = get_f32(params, "ampL", 0.5);
        let amp_r = get_f32(params, "ampR", 0.5);
        let base_freq = get_f32(params, "baseFreq", 200.0);
        let beat_freq = get_f32(params, "beatFreq", 4.0);
        let force_mono = get_bool(params, "forceMono", false);
        let left_high = get_bool(params, "leftHigh", false);
        let start_phase_l = get_f32(params, "startPhaseL", 0.0);
        let start_phase_r = get_f32(params, "startPhaseR", 0.0);
        let amp_osc_depth_l = get_f32(params, "ampOscDepthL", 0.0);
        let amp_osc_freq_l = get_f32(params, "ampOscFreqL", 0.0);
        let amp_osc_depth_r = get_f32(params, "ampOscDepthR", 0.0);
        let amp_osc_freq_r = get_f32(params, "ampOscFreqR", 0.0);
        let freq_osc_range_l = get_f32(params, "freqOscRangeL", 0.0);
        let freq_osc_freq_l = get_f32(params, "freqOscFreqL", 0.0);
        let freq_osc_range_r = get_f32(params, "freqOscRangeR", 0.0);
        let freq_osc_freq_r = get_f32(params, "freqOscFreqR", 0.0);
        let freq_osc_skew_l = get_f32(params, "freqOscSkewL", 0.0);
        let freq_osc_skew_r = get_f32(params, "freqOscSkewR", 0.0);
        let freq_osc_phase_offset_l = get_f32(params, "freqOscPhaseOffsetL", 0.0);
        let freq_osc_phase_offset_r = get_f32(params, "freqOscPhaseOffsetR", 0.0);
        let freq_osc_shape = LfoShape::from_str(
            params
                .get("freqOscShape")
                .and_then(|v| v.as_str())
                .unwrap_or("sine"),
        );
        let amp_osc_phase_offset_l = get_f32(params, "ampOscPhaseOffsetL", 0.0);
        let amp_osc_phase_offset_r = get_f32(params, "ampOscPhaseOffsetR", 0.0);
        let amp_osc_skew_l = get_f32(params, "ampOscSkewL", 0.0);
        let amp_osc_skew_r = get_f32(params, "ampOscSkewR", 0.0);
        let phase_osc_freq = get_f32(params, "phaseOscFreq", 0.0);
        let phase_osc_range = get_f32(params, "phaseOscRange", 0.0);

        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp_l,
            amp_r,
            base_freq,
            beat_freq,
            force_mono,
            left_high,
            start_phase_l,
            start_phase_r,
            amp_osc_depth_l,
            amp_osc_freq_l,
            amp_osc_depth_r,
            amp_osc_freq_r,
            freq_osc_range_l,
            freq_osc_freq_l,
            freq_osc_range_r,
            freq_osc_freq_r,
            freq_osc_skew_l,
            freq_osc_skew_r,
            freq_osc_phase_offset_l,
            freq_osc_phase_offset_r,
            freq_osc_shape,
            amp_osc_phase_offset_l,
            amp_osc_phase_offset_r,
            amp_osc_skew_l,
            amp_osc_skew_r,
            phase_osc_freq,
            phase_osc_range,
            phase_l: start_phase_l,
            phase_r: start_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl BinauralBeatTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let start_amp_l = get_f32(params, "startAmpL", get_f32(params, "ampL", 0.5));
        let end_amp_l = get_f32(params, "endAmpL", start_amp_l);
        let start_amp_r = get_f32(params, "startAmpR", get_f32(params, "ampR", 0.5));
        let end_amp_r = get_f32(params, "endAmpR", start_amp_r);
        let start_base_freq = get_f32(params, "startBaseFreq", get_f32(params, "baseFreq", 200.0));
        let end_base_freq = get_f32(params, "endBaseFreq", start_base_freq);
        let start_beat_freq = get_f32(params, "startBeatFreq", get_f32(params, "beatFreq", 4.0));
        let end_beat_freq = get_f32(params, "endBeatFreq", start_beat_freq);
        let start_force_mono = get_bool(
            params,
            "startForceMono",
            get_bool(params, "forceMono", false),
        );
        let end_force_mono = get_bool(params, "endForceMono", start_force_mono);
        let left_high = get_bool(params, "leftHigh", false);
        let start_start_phase_l = get_f32(
            params,
            "startStartPhaseL",
            get_f32(params, "startPhaseL", 0.0),
        );
        let end_start_phase_l = get_f32(params, "endStartPhaseL", start_start_phase_l);
        let start_start_phase_r = get_f32(
            params,
            "startStartPhaseR",
            get_f32(params, "startPhaseR", 0.0),
        );
        let end_start_phase_r = get_f32(params, "endStartPhaseR", start_start_phase_r);
        let start_phase_osc_freq = get_f32(
            params,
            "startPhaseOscFreq",
            get_f32(params, "phaseOscFreq", 0.0),
        );
        let end_phase_osc_freq = get_f32(params, "endPhaseOscFreq", start_phase_osc_freq);
        let start_phase_osc_range = get_f32(
            params,
            "startPhaseOscRange",
            get_f32(params, "phaseOscRange", 0.0),
        );
        let end_phase_osc_range = get_f32(params, "endPhaseOscRange", start_phase_osc_range);
        let start_amp_osc_depth_l = get_f32(
            params,
            "startAmpOscDepthL",
            get_f32(params, "ampOscDepthL", 0.0),
        );
        let end_amp_osc_depth_l = get_f32(params, "endAmpOscDepthL", start_amp_osc_depth_l);
        let start_amp_osc_freq_l = get_f32(
            params,
            "startAmpOscFreqL",
            get_f32(params, "ampOscFreqL", 0.0),
        );
        let end_amp_osc_freq_l = get_f32(params, "endAmpOscFreqL", start_amp_osc_freq_l);
        let start_amp_osc_depth_r = get_f32(
            params,
            "startAmpOscDepthR",
            get_f32(params, "ampOscDepthR", 0.0),
        );
        let end_amp_osc_depth_r = get_f32(params, "endAmpOscDepthR", start_amp_osc_depth_r);
        let start_amp_osc_freq_r = get_f32(
            params,
            "startAmpOscFreqR",
            get_f32(params, "ampOscFreqR", 0.0),
        );
        let end_amp_osc_freq_r = get_f32(params, "endAmpOscFreqR", start_amp_osc_freq_r);
        let start_amp_osc_phase_offset_l = get_f32(
            params,
            "startAmpOscPhaseOffsetL",
            get_f32(params, "ampOscPhaseOffsetL", 0.0),
        );
        let end_amp_osc_phase_offset_l = get_f32(
            params,
            "endAmpOscPhaseOffsetL",
            start_amp_osc_phase_offset_l,
        );
        let start_amp_osc_phase_offset_r = get_f32(
            params,
            "startAmpOscPhaseOffsetR",
            get_f32(params, "ampOscPhaseOffsetR", 0.0),
        );
        let end_amp_osc_phase_offset_r = get_f32(
            params,
            "endAmpOscPhaseOffsetR",
            start_amp_osc_phase_offset_r,
        );
        let start_freq_osc_range_l = get_f32(
            params,
            "startFreqOscRangeL",
            get_f32(params, "freqOscRangeL", 0.0),
        );
        let end_freq_osc_range_l = get_f32(params, "endFreqOscRangeL", start_freq_osc_range_l);
        let start_freq_osc_freq_l = get_f32(
            params,
            "startFreqOscFreqL",
            get_f32(params, "freqOscFreqL", 0.0),
        );
        let end_freq_osc_freq_l = get_f32(params, "endFreqOscFreqL", start_freq_osc_freq_l);
        let start_freq_osc_range_r = get_f32(
            params,
            "startFreqOscRangeR",
            get_f32(params, "freqOscRangeR", 0.0),
        );
        let end_freq_osc_range_r = get_f32(params, "endFreqOscRangeR", start_freq_osc_range_r);
        let start_freq_osc_freq_r = get_f32(
            params,
            "startFreqOscFreqR",
            get_f32(params, "freqOscFreqR", 0.0),
        );
        let end_freq_osc_freq_r = get_f32(params, "endFreqOscFreqR", start_freq_osc_freq_r);
        let start_freq_osc_skew_l = get_f32(
            params,
            "startFreqOscSkewL",
            get_f32(params, "freqOscSkewL", 0.0),
        );
        let end_freq_osc_skew_l = get_f32(params, "endFreqOscSkewL", start_freq_osc_skew_l);
        let start_freq_osc_skew_r = get_f32(
            params,
            "startFreqOscSkewR",
            get_f32(params, "freqOscSkewR", 0.0),
        );
        let end_freq_osc_skew_r = get_f32(params, "endFreqOscSkewR", start_freq_osc_skew_r);
        let start_freq_osc_phase_offset_l = get_f32(
            params,
            "startFreqOscPhaseOffsetL",
            get_f32(params, "freqOscPhaseOffsetL", 0.0),
        );
        let end_freq_osc_phase_offset_l = get_f32(
            params,
            "endFreqOscPhaseOffsetL",
            start_freq_osc_phase_offset_l,
        );
        let start_freq_osc_phase_offset_r = get_f32(
            params,
            "startFreqOscPhaseOffsetR",
            get_f32(params, "freqOscPhaseOffsetR", 0.0),
        );
        let end_freq_osc_phase_offset_r = get_f32(
            params,
            "endFreqOscPhaseOffsetR",
            start_freq_osc_phase_offset_r,
        );
        let start_amp_osc_skew_l = get_f32(
            params,
            "startAmpOscSkewL",
            get_f32(params, "ampOscSkewL", 0.0),
        );
        let end_amp_osc_skew_l = get_f32(params, "endAmpOscSkewL", start_amp_osc_skew_l);
        let start_amp_osc_skew_r = get_f32(
            params,
            "startAmpOscSkewR",
            get_f32(params, "ampOscSkewR", 0.0),
        );
        let end_amp_osc_skew_r = get_f32(params, "endAmpOscSkewR", start_amp_osc_skew_r);
        let freq_osc_shape = LfoShape::from_str(
            params
                .get("freqOscShape")
                .and_then(|v| v.as_str())
                .unwrap_or("sine"),
        );

        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            start_amp_l,
            end_amp_l,
            start_amp_r,
            end_amp_r,
            start_base_freq,
            end_base_freq,
            start_beat_freq,
            end_beat_freq,
            start_force_mono,
            end_force_mono,
            left_high,
            start_start_phase_l,
            end_start_phase_l,
            start_start_phase_r,
            end_start_phase_r,
            start_phase_osc_freq,
            end_phase_osc_freq,
            start_phase_osc_range,
            end_phase_osc_range,
            start_amp_osc_depth_l,
            end_amp_osc_depth_l,
            start_amp_osc_freq_l,
            end_amp_osc_freq_l,
            start_amp_osc_depth_r,
            end_amp_osc_depth_r,
            start_amp_osc_freq_r,
            end_amp_osc_freq_r,
            start_amp_osc_phase_offset_l,
            end_amp_osc_phase_offset_l,
            start_amp_osc_phase_offset_r,
            end_amp_osc_phase_offset_r,
            start_freq_osc_range_l,
            end_freq_osc_range_l,
            start_freq_osc_freq_l,
            end_freq_osc_freq_l,
            start_freq_osc_range_r,
            end_freq_osc_range_r,
            start_freq_osc_freq_r,
            end_freq_osc_freq_r,
            start_freq_osc_skew_l,
            end_freq_osc_skew_l,
            start_freq_osc_skew_r,
            end_freq_osc_skew_r,
            start_freq_osc_phase_offset_l,
            end_freq_osc_phase_offset_l,
            start_freq_osc_phase_offset_r,
            end_freq_osc_phase_offset_r,
            start_amp_osc_skew_l,
            end_amp_osc_skew_l,
            start_amp_osc_skew_r,
            end_amp_osc_skew_r,
            freq_osc_shape,
            curve,
            initial_offset,
            post_offset,
            sample_rate,
            remaining_samples: total_samples,
            phase_l: start_start_phase_l,
            phase_r: start_start_phase_r,
            sample_idx: 0,
            duration,
        }
    }
}

impl IsochronicToneVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let base_amp = get_f32(params, "amp", 0.5);
        let amp_l = get_f32(params, "ampL", base_amp);
        let amp_r = get_f32(params, "ampR", base_amp);
        let base_freq = get_f32(params, "baseFreq", 200.0);
        let beat_freq = get_f32(params, "beatFreq", 4.0);
        let force_mono = get_bool(params, "forceMono", false);
        let start_phase_l = get_f32(params, "startPhaseL", 0.0);
        let start_phase_r = get_f32(params, "startPhaseR", 0.0);
        let amp_osc_depth_l = get_f32(params, "ampOscDepthL", 0.0);
        let amp_osc_freq_l = get_f32(params, "ampOscFreqL", 0.0);
        let amp_osc_depth_r = get_f32(params, "ampOscDepthR", 0.0);
        let amp_osc_freq_r = get_f32(params, "ampOscFreqR", 0.0);
        let freq_osc_range_l = get_f32(params, "freqOscRangeL", 0.0);
        let freq_osc_freq_l = get_f32(params, "freqOscFreqL", 0.0);
        let freq_osc_range_r = get_f32(params, "freqOscRangeR", 0.0);
        let freq_osc_freq_r = get_f32(params, "freqOscFreqR", 0.0);
        let freq_osc_skew_l = get_f32(params, "freqOscSkewL", 0.0);
        let freq_osc_skew_r = get_f32(params, "freqOscSkewR", 0.0);
        let freq_osc_phase_offset_l = get_f32(params, "freqOscPhaseOffsetL", 0.0);
        let freq_osc_phase_offset_r = get_f32(params, "freqOscPhaseOffsetR", 0.0);
        let amp_osc_phase_offset_l = get_f32(params, "ampOscPhaseOffsetL", 0.0);
        let amp_osc_phase_offset_r = get_f32(params, "ampOscPhaseOffsetR", 0.0);
        let amp_osc_skew_l = get_f32(params, "ampOscSkewL", 0.0);
        let amp_osc_skew_r = get_f32(params, "ampOscSkewR", 0.0);
        let phase_osc_freq = get_f32(params, "phaseOscFreq", 0.0);
        let phase_osc_range = get_f32(params, "phaseOscRange", 0.0);
        let ramp_percent = get_f32(params, "rampPercent", 0.2);
        let gap_percent = get_f32(params, "gapPercent", 0.15);
        let pan = get_f32(params, "pan", 0.0);
        let pan_range_min = get_f32(params, "panRangeMin", pan).clamp(-1.0, 1.0);
        let pan_range_max = get_f32(params, "panRangeMax", pan).clamp(-1.0, 1.0);
        let pan_freq = get_f32(params, "panFreq", 0.0);
        let pan_phase = get_f32(params, "panPhase", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp_l,
            amp_r,
            base_freq,
            beat_freq,
            force_mono,
            start_phase_l,
            start_phase_r,
            amp_osc_depth_l,
            amp_osc_freq_l,
            amp_osc_depth_r,
            amp_osc_freq_r,
            freq_osc_range_l,
            freq_osc_freq_l,
            freq_osc_range_r,
            freq_osc_freq_r,
            freq_osc_skew_l,
            freq_osc_skew_r,
            freq_osc_phase_offset_l,
            freq_osc_phase_offset_r,
            amp_osc_phase_offset_l,
            amp_osc_phase_offset_r,
            amp_osc_skew_l,
            amp_osc_skew_r,
            phase_osc_freq,
            phase_osc_range,
            ramp_percent,
            gap_percent,
            pan,
            pan_range_min,
            pan_range_max,
            pan_freq,
            pan_phase,
            phase_l: start_phase_l,
            phase_r: start_phase_r,
            beat_phase: 0.0,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl IsochronicToneTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let base_amp = get_f32(params, "amp", 0.5);
        let start_amp_l = get_f32(params, "startAmpL", get_f32(params, "ampL", base_amp));
        let end_amp_l = get_f32(params, "endAmpL", start_amp_l);
        let start_amp_r = get_f32(params, "startAmpR", get_f32(params, "ampR", base_amp));
        let end_amp_r = get_f32(params, "endAmpR", start_amp_r);
        let start_base_freq = get_f32(params, "startBaseFreq", 200.0);
        let end_base_freq = get_f32(params, "endBaseFreq", start_base_freq);
        let start_beat_freq = get_f32(params, "startBeatFreq", 4.0);
        let end_beat_freq = get_f32(params, "endBeatFreq", start_beat_freq);
        let start_force_mono = get_bool(
            params,
            "startForceMono",
            get_bool(params, "forceMono", false),
        );
        let end_force_mono = get_bool(params, "endForceMono", start_force_mono);
        let start_start_phase_l = get_f32(
            params,
            "startStartPhaseL",
            get_f32(params, "startPhaseL", 0.0),
        );
        let end_start_phase_l = get_f32(params, "endStartPhaseL", start_start_phase_l);
        let start_start_phase_r = get_f32(
            params,
            "startStartPhaseR",
            get_f32(params, "startPhaseR", 0.0),
        );
        let end_start_phase_r = get_f32(params, "endStartPhaseR", start_start_phase_r);
        let start_phase_osc_freq = get_f32(
            params,
            "startPhaseOscFreq",
            get_f32(params, "phaseOscFreq", 0.0),
        );
        let end_phase_osc_freq = get_f32(params, "endPhaseOscFreq", start_phase_osc_freq);
        let start_phase_osc_range = get_f32(
            params,
            "startPhaseOscRange",
            get_f32(params, "phaseOscRange", 0.0),
        );
        let end_phase_osc_range = get_f32(params, "endPhaseOscRange", start_phase_osc_range);
        let start_amp_osc_depth_l = get_f32(
            params,
            "startAmpOscDepthL",
            get_f32(params, "ampOscDepthL", 0.0),
        );
        let end_amp_osc_depth_l = get_f32(params, "endAmpOscDepthL", start_amp_osc_depth_l);
        let start_amp_osc_freq_l = get_f32(
            params,
            "startAmpOscFreqL",
            get_f32(params, "ampOscFreqL", 0.0),
        );
        let end_amp_osc_freq_l = get_f32(params, "endAmpOscFreqL", start_amp_osc_freq_l);
        let start_amp_osc_depth_r = get_f32(
            params,
            "startAmpOscDepthR",
            get_f32(params, "ampOscDepthR", 0.0),
        );
        let end_amp_osc_depth_r = get_f32(params, "endAmpOscDepthR", start_amp_osc_depth_r);
        let start_amp_osc_freq_r = get_f32(
            params,
            "startAmpOscFreqR",
            get_f32(params, "ampOscFreqR", 0.0),
        );
        let end_amp_osc_freq_r = get_f32(params, "endAmpOscFreqR", start_amp_osc_freq_r);
        let start_amp_osc_phase_offset_l = get_f32(
            params,
            "startAmpOscPhaseOffsetL",
            get_f32(params, "ampOscPhaseOffsetL", 0.0),
        );
        let end_amp_osc_phase_offset_l = get_f32(
            params,
            "endAmpOscPhaseOffsetL",
            start_amp_osc_phase_offset_l,
        );
        let start_amp_osc_phase_offset_r = get_f32(
            params,
            "startAmpOscPhaseOffsetR",
            get_f32(params, "ampOscPhaseOffsetR", 0.0),
        );
        let end_amp_osc_phase_offset_r = get_f32(
            params,
            "endAmpOscPhaseOffsetR",
            start_amp_osc_phase_offset_r,
        );
        let start_freq_osc_range_l = get_f32(
            params,
            "startFreqOscRangeL",
            get_f32(params, "freqOscRangeL", 0.0),
        );
        let end_freq_osc_range_l = get_f32(params, "endFreqOscRangeL", start_freq_osc_range_l);
        let start_freq_osc_freq_l = get_f32(
            params,
            "startFreqOscFreqL",
            get_f32(params, "freqOscFreqL", 0.0),
        );
        let end_freq_osc_freq_l = get_f32(params, "endFreqOscFreqL", start_freq_osc_freq_l);
        let start_freq_osc_range_r = get_f32(
            params,
            "startFreqOscRangeR",
            get_f32(params, "freqOscRangeR", 0.0),
        );
        let end_freq_osc_range_r = get_f32(params, "endFreqOscRangeR", start_freq_osc_range_r);
        let start_freq_osc_freq_r = get_f32(
            params,
            "startFreqOscFreqR",
            get_f32(params, "freqOscFreqR", 0.0),
        );
        let end_freq_osc_freq_r = get_f32(params, "endFreqOscFreqR", start_freq_osc_freq_r);
        let start_freq_osc_skew_l = get_f32(
            params,
            "startFreqOscSkewL",
            get_f32(params, "freqOscSkewL", 0.0),
        );
        let end_freq_osc_skew_l = get_f32(params, "endFreqOscSkewL", start_freq_osc_skew_l);
        let start_freq_osc_skew_r = get_f32(
            params,
            "startFreqOscSkewR",
            get_f32(params, "freqOscSkewR", 0.0),
        );
        let end_freq_osc_skew_r = get_f32(params, "endFreqOscSkewR", start_freq_osc_skew_r);
        let start_freq_osc_phase_offset_l = get_f32(
            params,
            "startFreqOscPhaseOffsetL",
            get_f32(params, "freqOscPhaseOffsetL", 0.0),
        );
        let end_freq_osc_phase_offset_l = get_f32(
            params,
            "endFreqOscPhaseOffsetL",
            start_freq_osc_phase_offset_l,
        );
        let start_freq_osc_phase_offset_r = get_f32(
            params,
            "startFreqOscPhaseOffsetR",
            get_f32(params, "freqOscPhaseOffsetR", 0.0),
        );
        let end_freq_osc_phase_offset_r = get_f32(
            params,
            "endFreqOscPhaseOffsetR",
            start_freq_osc_phase_offset_r,
        );
        let start_amp_osc_skew_l = get_f32(
            params,
            "startAmpOscSkewL",
            get_f32(params, "ampOscSkewL", 0.0),
        );
        let end_amp_osc_skew_l = get_f32(params, "endAmpOscSkewL", start_amp_osc_skew_l);
        let start_amp_osc_skew_r = get_f32(
            params,
            "startAmpOscSkewR",
            get_f32(params, "ampOscSkewR", 0.0),
        );
        let end_amp_osc_skew_r = get_f32(params, "endAmpOscSkewR", start_amp_osc_skew_r);
        let start_ramp_percent = get_f32(
            params,
            "startRampPercent",
            get_f32(params, "rampPercent", 0.2),
        );
        let end_ramp_percent = get_f32(params, "endRampPercent", start_ramp_percent);
        let start_gap_percent = get_f32(
            params,
            "startGapPercent",
            get_f32(params, "gapPercent", 0.15),
        );
        let end_gap_percent = get_f32(params, "endGapPercent", start_gap_percent);
        let pan = get_f32(params, "pan", 0.0);
        let start_pan_range_min = get_f32(
            params,
            "startPanRangeMin",
            get_f32(params, "panRangeMin", pan),
        )
        .clamp(-1.0, 1.0);
        let end_pan_range_min =
            get_f32(params, "endPanRangeMin", start_pan_range_min).clamp(-1.0, 1.0);
        let start_pan_range_max = get_f32(
            params,
            "startPanRangeMax",
            get_f32(params, "panRangeMax", pan),
        )
        .clamp(-1.0, 1.0);
        let end_pan_range_max =
            get_f32(params, "endPanRangeMax", start_pan_range_max).clamp(-1.0, 1.0);
        let start_pan_freq = get_f32(params, "startPanFreq", get_f32(params, "panFreq", 0.0));
        let end_pan_freq = get_f32(params, "endPanFreq", start_pan_freq);
        let start_pan_phase = get_f32(params, "startPanPhase", get_f32(params, "panPhase", 0.0));
        let end_pan_phase = get_f32(params, "endPanPhase", start_pan_phase);
        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            start_amp_l,
            end_amp_l,
            start_amp_r,
            end_amp_r,
            start_base_freq,
            end_base_freq,
            start_beat_freq,
            end_beat_freq,
            start_force_mono,
            end_force_mono,
            start_start_phase_l,
            end_start_phase_l,
            start_start_phase_r,
            end_start_phase_r,
            start_phase_osc_freq,
            end_phase_osc_freq,
            start_phase_osc_range,
            end_phase_osc_range,
            start_amp_osc_depth_l,
            end_amp_osc_depth_l,
            start_amp_osc_freq_l,
            end_amp_osc_freq_l,
            start_amp_osc_depth_r,
            end_amp_osc_depth_r,
            start_amp_osc_freq_r,
            end_amp_osc_freq_r,
            start_amp_osc_phase_offset_l,
            end_amp_osc_phase_offset_l,
            start_amp_osc_phase_offset_r,
            end_amp_osc_phase_offset_r,
            start_freq_osc_range_l,
            end_freq_osc_range_l,
            start_freq_osc_freq_l,
            end_freq_osc_freq_l,
            start_freq_osc_range_r,
            end_freq_osc_range_r,
            start_freq_osc_freq_r,
            end_freq_osc_freq_r,
            start_freq_osc_skew_l,
            end_freq_osc_skew_l,
            start_freq_osc_skew_r,
            end_freq_osc_skew_r,
            start_freq_osc_phase_offset_l,
            end_freq_osc_phase_offset_l,
            start_freq_osc_phase_offset_r,
            end_freq_osc_phase_offset_r,
            start_amp_osc_skew_l,
            end_amp_osc_skew_l,
            start_amp_osc_skew_r,
            end_amp_osc_skew_r,
            start_ramp_percent,
            end_ramp_percent,
            start_gap_percent,
            end_gap_percent,
            pan,
            start_pan_range_min,
            end_pan_range_min,
            start_pan_range_max,
            end_pan_range_max,
            start_pan_freq,
            end_pan_freq,
            start_pan_phase,
            end_pan_phase,
            curve,
            initial_offset,
            post_offset,
            sample_rate,
            remaining_samples: total_samples,
            phase_l: start_start_phase_l,
            phase_r: start_start_phase_r,
            beat_phase: 0.0,
            sample_idx: 0,
            duration,
        }
    }
}

impl QamBeatVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp_l = get_f32(params, "ampL", 0.5);
        let amp_r = get_f32(params, "ampR", 0.5);
        let base_freq_l = get_f32(params, "baseFreqL", 200.0);
        let base_freq_r = get_f32(params, "baseFreqR", 204.0);
        let qam_am_freq_l = get_f32(params, "qamAmFreqL", 4.0);
        let qam_am_depth_l = get_f32(params, "qamAmDepthL", 0.5);
        let qam_am_phase_offset_l = get_f32(params, "qamAmPhaseOffsetL", 0.0);
        let qam_am_freq_r = get_f32(params, "qamAmFreqR", 4.0);
        let qam_am_depth_r = get_f32(params, "qamAmDepthR", 0.5);
        let qam_am_phase_offset_r = get_f32(params, "qamAmPhaseOffsetR", 0.0);
        let qam_am2_freq_l = get_f32(params, "qamAm2FreqL", 0.0);
        let qam_am2_depth_l = get_f32(params, "qamAm2DepthL", 0.0);
        let qam_am2_phase_offset_l = get_f32(params, "qamAm2PhaseOffsetL", 0.0);
        let qam_am2_freq_r = get_f32(params, "qamAm2FreqR", 0.0);
        let qam_am2_depth_r = get_f32(params, "qamAm2DepthR", 0.0);
        let qam_am2_phase_offset_r = get_f32(params, "qamAm2PhaseOffsetR", 0.0);
        let mod_shape_l = get_f32(params, "modShapeL", 1.0);
        let mod_shape_r = get_f32(params, "modShapeR", 1.0);
        let cross_mod_depth = get_f32(params, "crossModDepth", 0.0);
        let cross_mod_delay = get_f32(params, "crossModDelay", 0.0);
        let cross_mod_delay_samples = (cross_mod_delay * sample_rate) as usize;
        let harmonic_depth = get_f32(params, "harmonicDepth", 0.0);
        let harmonic_ratio = get_f32(params, "harmonicRatio", 2.0);
        let sub_harmonic_freq = get_f32(params, "subHarmonicFreq", 0.0);
        let sub_harmonic_depth = get_f32(params, "subHarmonicDepth", 0.0);
        let start_phase_l = get_f32(params, "startPhaseL", 0.0);
        let start_phase_r = get_f32(params, "startPhaseR", 0.0);
        let phase_osc_freq = get_f32(params, "phaseOscFreq", 0.0);
        let phase_osc_range = get_f32(params, "phaseOscRange", 0.0);
        let phase_osc_phase_offset = get_f32(params, "phaseOscPhaseOffset", 0.0);
        let beating_sidebands = get_bool(params, "beatingSidebands", false);
        let sideband_offset = get_f32(params, "sidebandOffset", 1.0);
        let sideband_depth = get_f32(params, "sidebandDepth", 0.1);
        let attack_time = get_f32(params, "attackTime", 0.0);
        let release_time = get_f32(params, "releaseTime", 0.0);
        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp_l,
            amp_r,
            base_freq_l,
            base_freq_r,
            qam_am_freq_l,
            qam_am_depth_l,
            qam_am_phase_offset_l,
            qam_am_freq_r,
            qam_am_depth_r,
            qam_am_phase_offset_r,
            qam_am2_freq_l,
            qam_am2_depth_l,
            qam_am2_phase_offset_l,
            qam_am2_freq_r,
            qam_am2_depth_r,
            qam_am2_phase_offset_r,
            mod_shape_l,
            mod_shape_r,
            cross_mod_depth,
            cross_mod_delay_samples,
            harmonic_depth,
            harmonic_ratio,
            sub_harmonic_freq,
            sub_harmonic_depth,
            phase_osc_freq,
            phase_osc_range,
            phase_osc_phase_offset,
            start_phase_l,
            start_phase_r,
            beating_sidebands,
            sideband_offset,
            sideband_depth,
            attack_time,
            release_time,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
            phase_l: start_phase_l,
            phase_r: start_phase_r,
            cross_env_l: if cross_mod_delay_samples == 0 {
                Vec::new()
            } else {
                vec![1.0; cross_mod_delay_samples]
            },
            cross_env_r: if cross_mod_delay_samples == 0 {
                Vec::new()
            } else {
                vec![1.0; cross_mod_delay_samples]
            },
            cross_idx: 0,
        }
    }
}

impl QamBeatTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let start_amp_l = get_f32(params, "startAmpL", get_f32(params, "ampL", 0.5));
        let end_amp_l = get_f32(params, "endAmpL", start_amp_l);
        let start_amp_r = get_f32(params, "startAmpR", get_f32(params, "ampR", 0.5));
        let end_amp_r = get_f32(params, "endAmpR", start_amp_r);
        let start_base_freq_l = get_f32(
            params,
            "startBaseFreqL",
            get_f32(params, "baseFreqL", 200.0),
        );
        let end_base_freq_l = get_f32(params, "endBaseFreqL", start_base_freq_l);
        let start_base_freq_r = get_f32(
            params,
            "startBaseFreqR",
            get_f32(params, "baseFreqR", 204.0),
        );
        let end_base_freq_r = get_f32(params, "endBaseFreqR", start_base_freq_r);
        let start_qam_am_freq_l = get_f32(
            params,
            "startQamAmFreqL",
            get_f32(params, "qamAmFreqL", 4.0),
        );
        let end_qam_am_freq_l = get_f32(params, "endQamAmFreqL", start_qam_am_freq_l);
        let start_qam_am_freq_r = get_f32(
            params,
            "startQamAmFreqR",
            get_f32(params, "qamAmFreqR", 4.0),
        );
        let end_qam_am_freq_r = get_f32(params, "endQamAmFreqR", start_qam_am_freq_r);
        let start_qam_am_depth_l = get_f32(
            params,
            "startQamAmDepthL",
            get_f32(params, "qamAmDepthL", 0.5),
        );
        let end_qam_am_depth_l = get_f32(params, "endQamAmDepthL", start_qam_am_depth_l);
        let start_qam_am_depth_r = get_f32(
            params,
            "startQamAmDepthR",
            get_f32(params, "qamAmDepthR", 0.5),
        );
        let end_qam_am_depth_r = get_f32(params, "endQamAmDepthR", start_qam_am_depth_r);
        let start_qam_am_phase_offset_l = get_f32(
            params,
            "startQamAmPhaseOffsetL",
            get_f32(params, "qamAmPhaseOffsetL", 0.0),
        );
        let end_qam_am_phase_offset_l =
            get_f32(params, "endQamAmPhaseOffsetL", start_qam_am_phase_offset_l);
        let start_qam_am_phase_offset_r = get_f32(
            params,
            "startQamAmPhaseOffsetR",
            get_f32(params, "qamAmPhaseOffsetR", 0.0),
        );
        let end_qam_am_phase_offset_r =
            get_f32(params, "endQamAmPhaseOffsetR", start_qam_am_phase_offset_r);
        let start_qam_am2_freq_l = get_f32(
            params,
            "startQamAm2FreqL",
            get_f32(params, "qamAm2FreqL", 0.0),
        );
        let end_qam_am2_freq_l = get_f32(params, "endQamAm2FreqL", start_qam_am2_freq_l);
        let start_qam_am2_freq_r = get_f32(
            params,
            "startQamAm2FreqR",
            get_f32(params, "qamAm2FreqR", 0.0),
        );
        let end_qam_am2_freq_r = get_f32(params, "endQamAm2FreqR", start_qam_am2_freq_r);
        let start_qam_am2_depth_l = get_f32(
            params,
            "startQamAm2DepthL",
            get_f32(params, "qamAm2DepthL", 0.0),
        );
        let end_qam_am2_depth_l = get_f32(params, "endQamAm2DepthL", start_qam_am2_depth_l);
        let start_qam_am2_depth_r = get_f32(
            params,
            "startQamAm2DepthR",
            get_f32(params, "qamAm2DepthR", 0.0),
        );
        let end_qam_am2_depth_r = get_f32(params, "endQamAm2DepthR", start_qam_am2_depth_r);
        let start_qam_am2_phase_offset_l = get_f32(
            params,
            "startQamAm2PhaseOffsetL",
            get_f32(params, "qamAm2PhaseOffsetL", 0.0),
        );
        let end_qam_am2_phase_offset_l = get_f32(
            params,
            "endQamAm2PhaseOffsetL",
            start_qam_am2_phase_offset_l,
        );
        let start_qam_am2_phase_offset_r = get_f32(
            params,
            "startQamAm2PhaseOffsetR",
            get_f32(params, "qamAm2PhaseOffsetR", 0.0),
        );
        let end_qam_am2_phase_offset_r = get_f32(
            params,
            "endQamAm2PhaseOffsetR",
            start_qam_am2_phase_offset_r,
        );
        let start_mod_shape_l =
            get_f32(params, "startModShapeL", get_f32(params, "modShapeL", 1.0));
        let end_mod_shape_l = get_f32(params, "endModShapeL", start_mod_shape_l);
        let start_mod_shape_r =
            get_f32(params, "startModShapeR", get_f32(params, "modShapeR", 1.0));
        let end_mod_shape_r = get_f32(params, "endModShapeR", start_mod_shape_r);
        let start_cross_mod_depth = get_f32(
            params,
            "startCrossModDepth",
            get_f32(params, "crossModDepth", 0.0),
        );
        let end_cross_mod_depth = get_f32(params, "endCrossModDepth", start_cross_mod_depth);
        let cross_mod_delay = get_f32(params, "crossModDelay", 0.0);
        let cross_mod_delay_samples = (cross_mod_delay * sample_rate) as usize;
        let harmonic_ratio = get_f32(params, "harmonicRatio", 2.0);
        let start_harmonic_depth = get_f32(
            params,
            "startHarmonicDepth",
            get_f32(params, "harmonicDepth", 0.0),
        );
        let end_harmonic_depth = get_f32(params, "endHarmonicDepth", start_harmonic_depth);
        let start_sub_harmonic_freq = get_f32(
            params,
            "startSubHarmonicFreq",
            get_f32(params, "subHarmonicFreq", 0.0),
        );
        let end_sub_harmonic_freq = get_f32(params, "endSubHarmonicFreq", start_sub_harmonic_freq);
        let start_sub_harmonic_depth = get_f32(
            params,
            "startSubHarmonicDepth",
            get_f32(params, "subHarmonicDepth", 0.0),
        );
        let end_sub_harmonic_depth =
            get_f32(params, "endSubHarmonicDepth", start_sub_harmonic_depth);
        let start_phase_osc_freq = get_f32(
            params,
            "startPhaseOscFreq",
            get_f32(params, "phaseOscFreq", 0.0),
        );
        let end_phase_osc_freq = get_f32(params, "endPhaseOscFreq", start_phase_osc_freq);
        let start_phase_osc_range = get_f32(
            params,
            "startPhaseOscRange",
            get_f32(params, "phaseOscRange", 0.0),
        );
        let end_phase_osc_range = get_f32(params, "endPhaseOscRange", start_phase_osc_range);
        let start_start_phase_l = get_f32(
            params,
            "startStartPhaseL",
            get_f32(params, "startPhaseL", 0.0),
        );
        let end_start_phase_l = get_f32(params, "endStartPhaseL", start_start_phase_l);
        let start_start_phase_r = get_f32(
            params,
            "startStartPhaseR",
            get_f32(params, "startPhaseR", 0.0),
        );
        let end_start_phase_r = get_f32(params, "endStartPhaseR", start_start_phase_r);
        let phase_osc_phase_offset = get_f32(params, "phaseOscPhaseOffset", 0.0);
        let beating_sidebands = get_bool(params, "beatingSidebands", false);
        let sideband_offset = get_f32(params, "sidebandOffset", 1.0);
        let sideband_depth = get_f32(params, "sidebandDepth", 0.1);
        let attack_time = get_f32(params, "attackTime", 0.0);
        let release_time = get_f32(params, "releaseTime", 0.0);
        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);
        let total_samples = (duration * sample_rate) as usize;

        Self {
            start_amp_l,
            end_amp_l,
            start_amp_r,
            end_amp_r,
            start_base_freq_l,
            end_base_freq_l,
            start_base_freq_r,
            end_base_freq_r,
            start_qam_am_freq_l,
            end_qam_am_freq_l,
            start_qam_am_depth_l,
            end_qam_am_depth_l,
            start_qam_am_freq_r,
            end_qam_am_freq_r,
            start_qam_am_depth_r,
            end_qam_am_depth_r,
            start_qam_am_phase_offset_l,
            end_qam_am_phase_offset_l,
            start_qam_am_phase_offset_r,
            end_qam_am_phase_offset_r,
            start_qam_am2_freq_l,
            end_qam_am2_freq_l,
            start_qam_am2_depth_l,
            end_qam_am2_depth_l,
            start_qam_am2_freq_r,
            end_qam_am2_freq_r,
            start_qam_am2_depth_r,
            end_qam_am2_depth_r,
            start_qam_am2_phase_offset_l,
            end_qam_am2_phase_offset_l,
            start_qam_am2_phase_offset_r,
            end_qam_am2_phase_offset_r,
            start_mod_shape_l,
            end_mod_shape_l,
            start_mod_shape_r,
            end_mod_shape_r,
            start_cross_mod_depth,
            end_cross_mod_depth,
            cross_mod_delay_samples,
            harmonic_ratio,
            start_harmonic_depth,
            end_harmonic_depth,
            start_sub_harmonic_freq,
            end_sub_harmonic_freq,
            start_sub_harmonic_depth,
            end_sub_harmonic_depth,
            start_phase_osc_freq,
            end_phase_osc_freq,
            start_phase_osc_range,
            end_phase_osc_range,
            start_start_phase_l,
            end_start_phase_l,
            start_start_phase_r,
            end_start_phase_r,
            phase_osc_phase_offset,
            beating_sidebands,
            sideband_offset,
            sideband_depth,
            attack_time,
            release_time,
            curve,
            initial_offset,
            post_offset,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
            phase_l: start_start_phase_l,
            phase_r: start_start_phase_r,
            cross_env_l: if cross_mod_delay_samples == 0 {
                Vec::new()
            } else {
                vec![1.0; cross_mod_delay_samples]
            },
            cross_env_r: if cross_mod_delay_samples == 0 {
                Vec::new()
            } else {
                vec![1.0; cross_mod_delay_samples]
            },
            cross_idx: 0,
        }
    }
}

impl StereoAmIndependentVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.25);
        let carrier_freq = get_f32(params, "carrierFreq", 200.0);
        let mod_freq_l = get_f32(params, "modFreqL", 4.0);
        let mod_depth_l = get_f32(params, "modDepthL", 0.8);
        let mod_phase_l = get_f32(params, "modPhaseL", 0.0);
        let mod_freq_r = get_f32(params, "modFreqR", 4.0);
        let mod_depth_r = get_f32(params, "modDepthR", 0.8);
        let mod_phase_r = get_f32(params, "modPhaseR", 0.0);
        let stereo_width_hz = get_f32(params, "stereo_width_hz", 0.2);
        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp,
            carrier_freq,
            stereo_width_hz,
            mod_freq_l,
            mod_depth_l,
            mod_phase_l,
            mod_freq_r,
            mod_depth_r,
            mod_phase_r,
            phase_carrier_l: 0.0,
            phase_carrier_r: 0.0,
            phase_mod_l: mod_phase_l,
            phase_mod_r: mod_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl StereoAmIndependentTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.25);
        let start_carrier_freq = get_f32(params, "startCarrierFreq", 200.0);
        let end_carrier_freq = get_f32(params, "endCarrierFreq", 250.0);
        let start_mod_freq_l = get_f32(params, "startModFreqL", 4.0);
        let end_mod_freq_l = get_f32(params, "endModFreqL", 6.0);
        let start_mod_depth_l = get_f32(params, "startModDepthL", 0.8);
        let end_mod_depth_l = get_f32(params, "endModDepthL", 0.8);
        let mod_phase_l = get_f32(params, "startModPhaseL", 0.0);
        let start_mod_freq_r = get_f32(params, "startModFreqR", 4.1);
        let end_mod_freq_r = get_f32(params, "endModFreqR", 5.9);
        let start_mod_depth_r = get_f32(params, "startModDepthR", 0.8);
        let end_mod_depth_r = get_f32(params, "endModDepthR", 0.8);
        let mod_phase_r = get_f32(params, "startModPhaseR", 0.0);
        let start_stereo_width_hz = get_f32(params, "startStereoWidthHz", 0.2);
        let end_stereo_width_hz = get_f32(params, "endStereoWidthHz", 0.2);
        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);
        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp,
            start_carrier_freq,
            end_carrier_freq,
            start_stereo_width_hz,
            end_stereo_width_hz,
            start_mod_freq_l,
            end_mod_freq_l,
            start_mod_depth_l,
            end_mod_depth_l,
            mod_phase_l,
            start_mod_freq_r,
            end_mod_freq_r,
            start_mod_depth_r,
            end_mod_depth_r,
            mod_phase_r,
            curve,
            initial_offset,
            post_offset,
            phase_carrier_l: 0.0,
            phase_carrier_r: 0.0,
            phase_mod_l: mod_phase_l,
            phase_mod_r: mod_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
        }
    }
}

impl WaveShapeStereoAmVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.15);
        let carrier_freq = get_f32(params, "carrierFreq", 200.0);
        let shape_mod_freq = get_f32(params, "shapeModFreq", 4.0);
        let shape_mod_depth = get_f32(params, "shapeModDepth", 0.8);
        let shape_amount = get_f32(params, "shapeAmount", 0.5);
        let stereo_mod_freq_l = get_f32(params, "stereoModFreqL", 4.1);
        let stereo_mod_depth_l = get_f32(params, "stereoModDepthL", 0.8);
        let stereo_mod_phase_l = get_f32(params, "stereoModPhaseL", 0.0);
        let stereo_mod_freq_r = get_f32(params, "stereoModFreqR", 4.0);
        let stereo_mod_depth_r = get_f32(params, "stereoModDepthR", 0.8);
        let stereo_mod_phase_r = get_f32(params, "stereoModPhaseR", std::f32::consts::FRAC_PI_2);
        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp,
            carrier_freq,
            shape_mod_freq,
            shape_mod_depth,
            shape_amount,
            stereo_mod_freq_l,
            stereo_mod_depth_l,
            stereo_mod_phase_l,
            stereo_mod_freq_r,
            stereo_mod_depth_r,
            stereo_mod_phase_r,
            phase_carrier: 0.0,
            phase_shape: 0.0,
            phase_stereo_l: stereo_mod_phase_l,
            phase_stereo_r: stereo_mod_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl WaveShapeStereoAmTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.15);
        let start_carrier_freq = get_f32(params, "startCarrierFreq", 200.0);
        let end_carrier_freq = get_f32(params, "endCarrierFreq", 100.0);
        let start_shape_mod_freq = get_f32(params, "startShapeModFreq", 4.0);
        let end_shape_mod_freq = get_f32(params, "endShapeModFreq", 8.0);
        let start_shape_mod_depth = get_f32(params, "startShapeModDepth", 0.8);
        let end_shape_mod_depth = get_f32(params, "endShapeModDepth", 0.8);
        let start_shape_amount = get_f32(params, "startShapeAmount", 0.5);
        let end_shape_amount = get_f32(params, "endShapeAmount", 0.5);
        let start_stereo_mod_freq_l = get_f32(params, "startStereoModFreqL", 4.1);
        let end_stereo_mod_freq_l = get_f32(params, "endStereoModFreqL", 6.0);
        let start_stereo_mod_depth_l = get_f32(params, "startStereoModDepthL", 0.8);
        let end_stereo_mod_depth_l = get_f32(params, "endStereoModDepthL", 0.8);
        let stereo_mod_phase_l = get_f32(params, "startStereoModPhaseL", 0.0);
        let start_stereo_mod_freq_r = get_f32(params, "startStereoModFreqR", 4.0);
        let end_stereo_mod_freq_r = get_f32(params, "endStereoModFreqR", 6.1);
        let start_stereo_mod_depth_r = get_f32(params, "startStereoModDepthR", 0.9);
        let end_stereo_mod_depth_r = get_f32(params, "endStereoModDepthR", 0.9);
        let stereo_mod_phase_r =
            get_f32(params, "startStereoModPhaseR", std::f32::consts::FRAC_PI_2);
        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);
        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp,
            start_carrier_freq,
            end_carrier_freq,
            start_shape_mod_freq,
            end_shape_mod_freq,
            start_shape_mod_depth,
            end_shape_mod_depth,
            start_shape_amount,
            end_shape_amount,
            start_stereo_mod_freq_l,
            end_stereo_mod_freq_l,
            start_stereo_mod_depth_l,
            end_stereo_mod_depth_l,
            stereo_mod_phase_l,
            start_stereo_mod_freq_r,
            end_stereo_mod_freq_r,
            start_stereo_mod_depth_r,
            end_stereo_mod_depth_r,
            stereo_mod_phase_r,
            curve,
            initial_offset,
            post_offset,
            phase_carrier: 0.0,
            phase_shape: 0.0,
            phase_stereo_l: stereo_mod_phase_l,
            phase_stereo_r: stereo_mod_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
        }
    }
}

impl SpatialAngleModulationVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.7);
        let carrier_freq = get_f32(params, "carrierFreq", 440.0);
        let beat_freq = get_f32(params, "beatFreq", 4.0);
        let path_radius = get_f32(params, "pathRadius", 1.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp,
            carrier_freq,
            beat_freq,
            path_radius,
            carrier_phase: 0.0,
            spatial_phase: 0.0,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl SpatialAngleModulationTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.7);
        let start_carrier_freq = get_f32(
            params,
            "startCarrierFreq",
            get_f32(params, "carrierFreq", 440.0),
        );
        let end_carrier_freq = get_f32(params, "endCarrierFreq", start_carrier_freq);
        let start_beat_freq = get_f32(params, "startBeatFreq", get_f32(params, "beatFreq", 4.0));
        let end_beat_freq = get_f32(params, "endBeatFreq", start_beat_freq);
        let start_path_radius = get_f32(
            params,
            "startPathRadius",
            get_f32(params, "pathRadius", 1.0),
        );
        let end_path_radius = get_f32(params, "endPathRadius", start_path_radius);

        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp,
            start_carrier_freq,
            end_carrier_freq,
            start_beat_freq,
            end_beat_freq,
            start_path_radius,
            end_path_radius,
            curve,
            initial_offset,
            post_offset,
            carrier_phase: 0.0,
            spatial_phase: 0.0,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
        }
    }
}

impl RhythmicWaveshapingVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.25);
        let carrier_freq = get_f32(params, "carrierFreq", 200.0);
        let mod_freq = get_f32(params, "modFreq", 4.0);
        let mod_depth = get_f32(params, "modDepth", 1.0);
        let shape_amount = get_f32(params, "shapeAmount", 5.0);
        let pan = get_f32(params, "pan", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp,
            carrier_freq,
            mod_freq,
            mod_depth,
            shape_amount,
            pan,
            carrier_phase: 0.0,
            lfo_phase: 0.0,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
        }
    }
}

impl RhythmicWaveshapingTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp = get_f32(params, "amp", 0.25);
        let start_carrier_freq = get_f32(
            params,
            "startCarrierFreq",
            get_f32(params, "carrierFreq", 200.0),
        );
        let end_carrier_freq = get_f32(params, "endCarrierFreq", start_carrier_freq);
        let start_mod_freq = get_f32(params, "startModFreq", get_f32(params, "modFreq", 4.0));
        let end_mod_freq = get_f32(params, "endModFreq", start_mod_freq);
        let start_mod_depth = get_f32(params, "startModDepth", get_f32(params, "modDepth", 1.0));
        let end_mod_depth = get_f32(params, "endModDepth", start_mod_depth);
        let start_shape_amount = get_f32(
            params,
            "startShapeAmount",
            get_f32(params, "shapeAmount", 5.0),
        );
        let end_shape_amount = get_f32(params, "endShapeAmount", start_shape_amount);
        let pan = get_f32(params, "pan", 0.0);

        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            amp,
            start_carrier_freq,
            end_carrier_freq,
            start_mod_freq,
            end_mod_freq,
            start_mod_depth,
            end_mod_depth,
            start_shape_amount,
            end_shape_amount,
            pan,
            curve,
            initial_offset,
            post_offset,
            carrier_phase: 0.0,
            lfo_phase: 0.0,
            sample_rate,
            remaining_samples: total_samples,
            sample_idx: 0,
            duration,
        }
    }
}

impl Voice for BinauralBeatVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let t = self.sample_idx as f32 / self.sample_rate;

            // Instantaneous frequency with vibrato
            let half_beat = self.beat_freq * 0.5;
            let phase_l_vib = self.freq_osc_freq_l * t
                + self.freq_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let phase_r_vib = self.freq_osc_freq_r * t
                + self.freq_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let vib_l = (self.freq_osc_range_l * 0.5)
                * match self.freq_osc_shape {
                    LfoShape::Triangle => {
                        skewed_triangle_phase(phase_l_vib.fract(), self.freq_osc_skew_l)
                    }
                    LfoShape::Sine => skewed_sine_phase(phase_l_vib.fract(), self.freq_osc_skew_l),
                };
            let vib_r = (self.freq_osc_range_r * 0.5)
                * match self.freq_osc_shape {
                    LfoShape::Triangle => {
                        skewed_triangle_phase(phase_r_vib.fract(), self.freq_osc_skew_r)
                    }
                    LfoShape::Sine => skewed_sine_phase(phase_r_vib.fract(), self.freq_osc_skew_r),
                };
            // Apply left_high to determine which ear gets the higher frequency
            let (freq_base_l, freq_base_r) = if self.left_high {
                (self.base_freq + half_beat, self.base_freq - half_beat)
            } else {
                (self.base_freq - half_beat, self.base_freq + half_beat)
            };
            let mut freq_l = freq_base_l + vib_l;
            let mut freq_r = freq_base_r + vib_r;

            if self.force_mono || self.beat_freq == 0.0 {
                freq_l = self.base_freq.max(0.0);
                freq_r = self.base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            // Advance phase
            let dt = 1.0 / self.sample_rate;
            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);

            // Phase modulation
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if self.phase_osc_freq != 0.0 || self.phase_osc_range != 0.0 {
                let dphi = (self.phase_osc_range * 0.5)
                    * sin_lut(2.0 * std::f32::consts::PI * self.phase_osc_freq * t);
                ph_l -= dphi;
                ph_r += dphi;
            }

            // Amplitude envelopes
            let amp_phase_l = self.amp_osc_freq_l * t
                + self.amp_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let amp_phase_r = self.amp_osc_freq_r * t
                + self.amp_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let env_l = 1.0
                - self.amp_osc_depth_l
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_l.fract(), self.amp_osc_skew_l)));
            let env_r = 1.0
                - self.amp_osc_depth_r
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_r.fract(), self.amp_osc_skew_r)));

            let sample_l = sin_lut(ph_l) * env_l * self.amp_l;
            let sample_r = sin_lut(ph_r) * env_r * self.amp_r;

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for BinauralBeatTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let amp_l = self.start_amp_l + (self.end_amp_l - self.start_amp_l) * alpha;
            let amp_r = self.start_amp_r + (self.end_amp_r - self.start_amp_r) * alpha;
            let base_freq =
                self.start_base_freq + (self.end_base_freq - self.start_base_freq) * alpha;
            let beat_freq =
                self.start_beat_freq + (self.end_beat_freq - self.start_beat_freq) * alpha;
            let force_mono = if self.start_force_mono == self.end_force_mono {
                self.start_force_mono
            } else {
                alpha >= 0.5
            };
            let phase_osc_freq = self.start_phase_osc_freq
                + (self.end_phase_osc_freq - self.start_phase_osc_freq) * alpha;
            let phase_osc_range = self.start_phase_osc_range
                + (self.end_phase_osc_range - self.start_phase_osc_range) * alpha;
            let amp_osc_depth_l = self.start_amp_osc_depth_l
                + (self.end_amp_osc_depth_l - self.start_amp_osc_depth_l) * alpha;
            let amp_osc_freq_l = self.start_amp_osc_freq_l
                + (self.end_amp_osc_freq_l - self.start_amp_osc_freq_l) * alpha;
            let amp_osc_depth_r = self.start_amp_osc_depth_r
                + (self.end_amp_osc_depth_r - self.start_amp_osc_depth_r) * alpha;
            let amp_osc_freq_r = self.start_amp_osc_freq_r
                + (self.end_amp_osc_freq_r - self.start_amp_osc_freq_r) * alpha;
            let amp_osc_phase_offset_l = self.start_amp_osc_phase_offset_l
                + (self.end_amp_osc_phase_offset_l - self.start_amp_osc_phase_offset_l) * alpha;
            let amp_osc_phase_offset_r = self.start_amp_osc_phase_offset_r
                + (self.end_amp_osc_phase_offset_r - self.start_amp_osc_phase_offset_r) * alpha;
            let freq_osc_range_l = self.start_freq_osc_range_l
                + (self.end_freq_osc_range_l - self.start_freq_osc_range_l) * alpha;
            let freq_osc_freq_l = self.start_freq_osc_freq_l
                + (self.end_freq_osc_freq_l - self.start_freq_osc_freq_l) * alpha;
            let freq_osc_range_r = self.start_freq_osc_range_r
                + (self.end_freq_osc_range_r - self.start_freq_osc_range_r) * alpha;
            let freq_osc_freq_r = self.start_freq_osc_freq_r
                + (self.end_freq_osc_freq_r - self.start_freq_osc_freq_r) * alpha;
            let freq_osc_skew_l = self.start_freq_osc_skew_l
                + (self.end_freq_osc_skew_l - self.start_freq_osc_skew_l) * alpha;
            let freq_osc_skew_r = self.start_freq_osc_skew_r
                + (self.end_freq_osc_skew_r - self.start_freq_osc_skew_r) * alpha;
            let freq_osc_phase_offset_l = self.start_freq_osc_phase_offset_l
                + (self.end_freq_osc_phase_offset_l - self.start_freq_osc_phase_offset_l) * alpha;
            let freq_osc_phase_offset_r = self.start_freq_osc_phase_offset_r
                + (self.end_freq_osc_phase_offset_r - self.start_freq_osc_phase_offset_r) * alpha;
            let amp_osc_skew_l = self.start_amp_osc_skew_l
                + (self.end_amp_osc_skew_l - self.start_amp_osc_skew_l) * alpha;
            let amp_osc_skew_r = self.start_amp_osc_skew_r
                + (self.end_amp_osc_skew_r - self.start_amp_osc_skew_r) * alpha;

            // instantaneous frequencies
            let half_beat = beat_freq * 0.5;
            let phase_l_vib =
                freq_osc_freq_l * t + freq_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let phase_r_vib =
                freq_osc_freq_r * t + freq_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let vib_l = (freq_osc_range_l * 0.5)
                * match self.freq_osc_shape {
                    LfoShape::Triangle => {
                        skewed_triangle_phase(phase_l_vib.fract(), freq_osc_skew_l)
                    }
                    LfoShape::Sine => skewed_sine_phase(phase_l_vib.fract(), freq_osc_skew_l),
                };
            let vib_r = (freq_osc_range_r * 0.5)
                * match self.freq_osc_shape {
                    LfoShape::Triangle => {
                        skewed_triangle_phase(phase_r_vib.fract(), freq_osc_skew_r)
                    }
                    LfoShape::Sine => skewed_sine_phase(phase_r_vib.fract(), freq_osc_skew_r),
                };
            // Apply left_high to determine which ear gets the higher frequency
            let (freq_base_l, freq_base_r) = if self.left_high {
                (base_freq + half_beat, base_freq - half_beat)
            } else {
                (base_freq - half_beat, base_freq + half_beat)
            };
            let mut freq_l = freq_base_l + vib_l;
            let mut freq_r = freq_base_r + vib_r;

            if force_mono || beat_freq == 0.0 {
                freq_l = base_freq.max(0.0);
                freq_r = base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if phase_osc_freq != 0.0 || phase_osc_range != 0.0 {
                let dphi = (phase_osc_range * 0.5)
                    * sin_lut(2.0 * std::f32::consts::PI * phase_osc_freq * t);
                ph_l -= dphi;
                ph_r += dphi;
            }

            let amp_phase_l =
                amp_osc_freq_l * t + amp_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let amp_phase_r =
                amp_osc_freq_r * t + amp_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let env_l = 1.0
                - amp_osc_depth_l
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_l.fract(), amp_osc_skew_l)));
            let env_r = 1.0
                - amp_osc_depth_r
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_r.fract(), amp_osc_skew_r)));

            let sample_l = sin_lut(ph_l) * env_l * amp_l;
            let sample_r = sin_lut(ph_r) * env_r * amp_r;

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for IsochronicToneVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;

            let phase_l_vib = self.freq_osc_freq_l * t
                + self.freq_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let phase_r_vib = self.freq_osc_freq_r * t
                + self.freq_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let vib_l = (self.freq_osc_range_l * 0.5)
                * skewed_sine_phase(phase_l_vib.fract(), self.freq_osc_skew_l);
            let vib_r = (self.freq_osc_range_r * 0.5)
                * skewed_sine_phase(phase_r_vib.fract(), self.freq_osc_skew_r);
            let mut freq_l = self.base_freq + vib_l;
            let mut freq_r = self.base_freq + vib_r;

            if self.force_mono {
                freq_l = self.base_freq.max(0.0);
                freq_r = self.base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            let cycle_len = if self.beat_freq > 0.0 {
                1.0 / self.beat_freq
            } else {
                0.0
            };
            let t_in_cycle = self.beat_phase * cycle_len;
            let iso_env =
                trapezoid_envelope(t_in_cycle, cycle_len, self.ramp_percent, self.gap_percent);

            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);
            self.beat_phase += self.beat_freq * dt;
            self.beat_phase = self.beat_phase.rem_euclid(1.0);

            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if self.phase_osc_freq != 0.0 || self.phase_osc_range != 0.0 {
                let dphi = (self.phase_osc_range * 0.5)
                    * sin_lut(2.0 * std::f32::consts::PI * self.phase_osc_freq * t);
                ph_l -= dphi;
                ph_r += dphi;
            }

            let amp_phase_l = self.amp_osc_freq_l * t
                + self.amp_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let amp_phase_r = self.amp_osc_freq_r * t
                + self.amp_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let env_l = 1.0
                - self.amp_osc_depth_l
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_l.fract(), self.amp_osc_skew_l)));
            let env_r = 1.0
                - self.amp_osc_depth_r
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_r.fract(), self.amp_osc_skew_r)));

            let mut sample_l = sin_lut(ph_l) * env_l * self.amp_l * iso_env;
            let mut sample_r = sin_lut(ph_r) * env_r * self.amp_r * iso_env;

            // Calculate time-varying pan based on pan envelope parameters
            let pan_min = self.pan_range_min.min(self.pan_range_max);
            let pan_max = self.pan_range_min.max(self.pan_range_max);
            let pan_center = (pan_min + pan_max) * 0.5;
            let pan_range = (pan_max - pan_min) * 0.5;
            let current_pan = if self.pan_freq != 0.0 && pan_range > 0.0 {
                // Oscillating pan using sine wave
                let pan_osc =
                    sin_lut(2.0 * std::f32::consts::PI * self.pan_freq * t + self.pan_phase);
                (pan_center + pan_range * pan_osc).clamp(-1.0, 1.0)
            } else {
                // Static pan
                pan_center
            };

            if current_pan.abs() > 1e-6 {
                let mono = 0.5 * (sample_l + sample_r);
                let (pl, pr) = pan2(mono, current_pan);
                sample_l = pl;
                sample_r = pr;
            }

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for IsochronicToneTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let amp_l = self.start_amp_l + (self.end_amp_l - self.start_amp_l) * alpha;
            let amp_r = self.start_amp_r + (self.end_amp_r - self.start_amp_r) * alpha;
            let base_freq =
                self.start_base_freq + (self.end_base_freq - self.start_base_freq) * alpha;
            let beat_freq =
                self.start_beat_freq + (self.end_beat_freq - self.start_beat_freq) * alpha;
            let force_mono = if self.start_force_mono == self.end_force_mono {
                self.start_force_mono
            } else {
                alpha >= 0.5
            };
            let phase_osc_freq = self.start_phase_osc_freq
                + (self.end_phase_osc_freq - self.start_phase_osc_freq) * alpha;
            let phase_osc_range = self.start_phase_osc_range
                + (self.end_phase_osc_range - self.start_phase_osc_range) * alpha;
            let amp_osc_depth_l = self.start_amp_osc_depth_l
                + (self.end_amp_osc_depth_l - self.start_amp_osc_depth_l) * alpha;
            let amp_osc_freq_l = self.start_amp_osc_freq_l
                + (self.end_amp_osc_freq_l - self.start_amp_osc_freq_l) * alpha;
            let amp_osc_depth_r = self.start_amp_osc_depth_r
                + (self.end_amp_osc_depth_r - self.start_amp_osc_depth_r) * alpha;
            let amp_osc_freq_r = self.start_amp_osc_freq_r
                + (self.end_amp_osc_freq_r - self.start_amp_osc_freq_r) * alpha;
            let amp_osc_phase_offset_l = self.start_amp_osc_phase_offset_l
                + (self.end_amp_osc_phase_offset_l - self.start_amp_osc_phase_offset_l) * alpha;
            let amp_osc_phase_offset_r = self.start_amp_osc_phase_offset_r
                + (self.end_amp_osc_phase_offset_r - self.start_amp_osc_phase_offset_r) * alpha;
            let freq_osc_range_l = self.start_freq_osc_range_l
                + (self.end_freq_osc_range_l - self.start_freq_osc_range_l) * alpha;
            let freq_osc_freq_l = self.start_freq_osc_freq_l
                + (self.end_freq_osc_freq_l - self.start_freq_osc_freq_l) * alpha;
            let freq_osc_range_r = self.start_freq_osc_range_r
                + (self.end_freq_osc_range_r - self.start_freq_osc_range_r) * alpha;
            let freq_osc_freq_r = self.start_freq_osc_freq_r
                + (self.end_freq_osc_freq_r - self.start_freq_osc_freq_r) * alpha;

            let freq_osc_skew_l = self.start_freq_osc_skew_l
                + (self.end_freq_osc_skew_l - self.start_freq_osc_skew_l) * alpha;
            let freq_osc_skew_r = self.start_freq_osc_skew_r
                + (self.end_freq_osc_skew_r - self.start_freq_osc_skew_r) * alpha;
            let freq_osc_phase_offset_l = self.start_freq_osc_phase_offset_l
                + (self.end_freq_osc_phase_offset_l - self.start_freq_osc_phase_offset_l) * alpha;
            let freq_osc_phase_offset_r = self.start_freq_osc_phase_offset_r
                + (self.end_freq_osc_phase_offset_r - self.start_freq_osc_phase_offset_r) * alpha;
            let amp_osc_skew_l = self.start_amp_osc_skew_l
                + (self.end_amp_osc_skew_l - self.start_amp_osc_skew_l) * alpha;
            let amp_osc_skew_r = self.start_amp_osc_skew_r
                + (self.end_amp_osc_skew_r - self.start_amp_osc_skew_r) * alpha;
            let ramp_percent =
                self.start_ramp_percent + (self.end_ramp_percent - self.start_ramp_percent) * alpha;
            let gap_percent =
                self.start_gap_percent + (self.end_gap_percent - self.start_gap_percent) * alpha;

            let phase_l_vib =
                freq_osc_freq_l * t + freq_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let phase_r_vib =
                freq_osc_freq_r * t + freq_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let vib_l =
                (freq_osc_range_l * 0.5) * skewed_sine_phase(phase_l_vib.fract(), freq_osc_skew_l);
            let vib_r =
                (freq_osc_range_r * 0.5) * skewed_sine_phase(phase_r_vib.fract(), freq_osc_skew_r);
            let mut freq_l = base_freq + vib_l;
            let mut freq_r = base_freq + vib_r;

            if force_mono {
                freq_l = base_freq.max(0.0);
                freq_r = base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            let cycle_len = if beat_freq > 0.0 {
                1.0 / beat_freq
            } else {
                0.0
            };
            let t_in_cycle = self.beat_phase * cycle_len;
            let iso_env = trapezoid_envelope(t_in_cycle, cycle_len, ramp_percent, gap_percent);

            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);
            self.beat_phase += beat_freq * dt;
            self.beat_phase = self.beat_phase.rem_euclid(1.0);

            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if phase_osc_freq != 0.0 || phase_osc_range != 0.0 {
                let dphi = (phase_osc_range * 0.5)
                    * sin_lut(2.0 * std::f32::consts::PI * phase_osc_freq * t);
                ph_l -= dphi;
                ph_r += dphi;
            }

            let amp_phase_l =
                amp_osc_freq_l * t + amp_osc_phase_offset_l / (2.0 * std::f32::consts::PI);
            let amp_phase_r =
                amp_osc_freq_r * t + amp_osc_phase_offset_r / (2.0 * std::f32::consts::PI);
            let env_l = 1.0
                - amp_osc_depth_l
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_l.fract(), amp_osc_skew_l)));
            let env_r = 1.0
                - amp_osc_depth_r
                    * (0.5 * (1.0 + skewed_sine_phase(amp_phase_r.fract(), amp_osc_skew_r)));

            let mut sample_l = sin_lut(ph_l) * env_l * amp_l * iso_env;
            let mut sample_r = sin_lut(ph_r) * env_r * amp_r * iso_env;

            // Calculate time-varying pan based on interpolated pan envelope parameters
            let pan_range_min = self.start_pan_range_min
                + (self.end_pan_range_min - self.start_pan_range_min) * alpha;
            let pan_range_max = self.start_pan_range_max
                + (self.end_pan_range_max - self.start_pan_range_max) * alpha;
            let pan_freq = self.start_pan_freq + (self.end_pan_freq - self.start_pan_freq) * alpha;
            let pan_phase =
                self.start_pan_phase + (self.end_pan_phase - self.start_pan_phase) * alpha;

            let pan_min = pan_range_min.min(pan_range_max);
            let pan_max = pan_range_min.max(pan_range_max);
            let pan_center = (pan_min + pan_max) * 0.5;
            let pan_range = (pan_max - pan_min) * 0.5;
            let current_pan = if pan_freq != 0.0 && pan_range > 0.0 {
                let pan_osc = sin_lut(2.0 * std::f32::consts::PI * pan_freq * t + pan_phase);
                (pan_center + pan_range * pan_osc).clamp(-1.0, 1.0)
            } else {
                pan_center
            };

            if current_pan.abs() > 1e-6 {
                let mono = 0.5 * (sample_l + sample_r);
                let (pl, pr) = pan2(mono, current_pan);
                sample_l = pl;
                sample_r = pr;
            }

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for QamBeatVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;

            let mut env_l = 1.0;
            if self.qam_am_freq_l != 0.0 && self.qam_am_depth_l != 0.0 {
                let phase = 2.0 * std::f32::consts::PI * self.qam_am_freq_l * t
                    + self.qam_am_phase_offset_l;
                let mod_l1 = if self.mod_shape_l == 1.0 {
                    cos_lut(phase)
                } else {
                    let c = cos_lut(phase);
                    c.signum() * c.abs().powf(1.0 / self.mod_shape_l)
                };
                env_l *= 1.0 + self.qam_am_depth_l * mod_l1;
            }

            let mut env_r = 1.0;
            if self.qam_am_freq_r != 0.0 && self.qam_am_depth_r != 0.0 {
                let phase = 2.0 * std::f32::consts::PI * self.qam_am_freq_r * t
                    + self.qam_am_phase_offset_r;
                let mod_r1 = if self.mod_shape_r == 1.0 {
                    cos_lut(phase)
                } else {
                    let c = cos_lut(phase);
                    c.signum() * c.abs().powf(1.0 / self.mod_shape_r)
                };
                env_r *= 1.0 + self.qam_am_depth_r * mod_r1;
            }

            if self.qam_am2_freq_l != 0.0 && self.qam_am2_depth_l != 0.0 {
                env_l *= 1.0
                    + self.qam_am2_depth_l
                        * cos_lut(
                            2.0 * std::f32::consts::PI * self.qam_am2_freq_l * t
                                + self.qam_am2_phase_offset_l,
                        );
            }
            if self.qam_am2_freq_r != 0.0 && self.qam_am2_depth_r != 0.0 {
                env_r *= 1.0
                    + self.qam_am2_depth_r
                        * cos_lut(
                            2.0 * std::f32::consts::PI * self.qam_am2_freq_r * t
                                + self.qam_am2_phase_offset_r,
                        );
            }

            let base_env_l = env_l;
            let base_env_r = env_r;

            if self.cross_mod_depth != 0.0 && self.cross_mod_delay_samples > 0 {
                let idx = self.cross_idx;
                env_l *= 1.0 + self.cross_mod_depth * (self.cross_env_r[idx] - 1.0);
                env_r *= 1.0 + self.cross_mod_depth * (self.cross_env_l[idx] - 1.0);
                self.cross_env_l[idx] = base_env_l;
                self.cross_env_r[idx] = base_env_r;
                self.cross_idx = (idx + 1) % self.cross_mod_delay_samples;
            }

            if self.sub_harmonic_freq != 0.0 && self.sub_harmonic_depth != 0.0 {
                let sub = cos_lut(2.0 * std::f32::consts::PI * self.sub_harmonic_freq * t);
                env_l *= 1.0 + self.sub_harmonic_depth * sub;
                env_r *= 1.0 + self.sub_harmonic_depth * sub;
            }

            self.phase_l += 2.0 * std::f32::consts::PI * self.base_freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * self.base_freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if self.phase_osc_freq != 0.0 || self.phase_osc_range != 0.0 {
                let dphi = (self.phase_osc_range * 0.5)
                    * sin_lut(
                        2.0 * std::f32::consts::PI * self.phase_osc_freq * t
                            + self.phase_osc_phase_offset,
                    );
                ph_l -= dphi;
                ph_r += dphi;
            }

            let mut sig_l = env_l * cos_lut(ph_l);
            let mut sig_r = env_r * cos_lut(ph_r);

            if self.harmonic_depth != 0.0 {
                sig_l += self.harmonic_depth * env_l * cos_lut(self.harmonic_ratio * ph_l);
                sig_r += self.harmonic_depth * env_r * cos_lut(self.harmonic_ratio * ph_r);
            }

            if self.beating_sidebands && self.sideband_depth != 0.0 {
                let side = 2.0 * std::f32::consts::PI * self.sideband_offset * t;
                sig_l += self.sideband_depth * env_l * cos_lut(ph_l - side);
                sig_r += self.sideband_depth * env_r * cos_lut(ph_r - side);
                sig_l += self.sideband_depth * env_l * cos_lut(ph_l + side);
                sig_r += self.sideband_depth * env_r * cos_lut(ph_r + side);
            }

            let mut env_mult = 1.0;
            if self.attack_time > 0.0 && t < self.attack_time {
                env_mult *= t / self.attack_time;
            }
            if self.release_time > 0.0 && t > (self.duration - self.release_time) {
                env_mult *= (self.duration - t) / self.release_time;
            }

            output[i * 2] += sig_l * self.amp_l * env_mult;
            output[i * 2 + 1] += sig_r * self.amp_r * env_mult;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for QamBeatTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let amp_l = self.start_amp_l + (self.end_amp_l - self.start_amp_l) * alpha;
            let amp_r = self.start_amp_r + (self.end_amp_r - self.start_amp_r) * alpha;
            let base_freq_l =
                self.start_base_freq_l + (self.end_base_freq_l - self.start_base_freq_l) * alpha;
            let base_freq_r =
                self.start_base_freq_r + (self.end_base_freq_r - self.start_base_freq_r) * alpha;
            let qam_am_freq_l = self.start_qam_am_freq_l
                + (self.end_qam_am_freq_l - self.start_qam_am_freq_l) * alpha;
            let qam_am_freq_r = self.start_qam_am_freq_r
                + (self.end_qam_am_freq_r - self.start_qam_am_freq_r) * alpha;
            let qam_am_depth_l = self.start_qam_am_depth_l
                + (self.end_qam_am_depth_l - self.start_qam_am_depth_l) * alpha;
            let qam_am_depth_r = self.start_qam_am_depth_r
                + (self.end_qam_am_depth_r - self.start_qam_am_depth_r) * alpha;
            let qam_am_phase_offset_l = self.start_qam_am_phase_offset_l
                + (self.end_qam_am_phase_offset_l - self.start_qam_am_phase_offset_l) * alpha;
            let qam_am_phase_offset_r = self.start_qam_am_phase_offset_r
                + (self.end_qam_am_phase_offset_r - self.start_qam_am_phase_offset_r) * alpha;
            let qam_am2_freq_l = self.start_qam_am2_freq_l
                + (self.end_qam_am2_freq_l - self.start_qam_am2_freq_l) * alpha;
            let qam_am2_freq_r = self.start_qam_am2_freq_r
                + (self.end_qam_am2_freq_r - self.start_qam_am2_freq_r) * alpha;
            let qam_am2_depth_l = self.start_qam_am2_depth_l
                + (self.end_qam_am2_depth_l - self.start_qam_am2_depth_l) * alpha;
            let qam_am2_depth_r = self.start_qam_am2_depth_r
                + (self.end_qam_am2_depth_r - self.start_qam_am2_depth_r) * alpha;
            let qam_am2_phase_offset_l = self.start_qam_am2_phase_offset_l
                + (self.end_qam_am2_phase_offset_l - self.start_qam_am2_phase_offset_l) * alpha;
            let qam_am2_phase_offset_r = self.start_qam_am2_phase_offset_r
                + (self.end_qam_am2_phase_offset_r - self.start_qam_am2_phase_offset_r) * alpha;
            let mod_shape_l =
                self.start_mod_shape_l + (self.end_mod_shape_l - self.start_mod_shape_l) * alpha;
            let mod_shape_r =
                self.start_mod_shape_r + (self.end_mod_shape_r - self.start_mod_shape_r) * alpha;
            let cross_mod_depth = self.start_cross_mod_depth
                + (self.end_cross_mod_depth - self.start_cross_mod_depth) * alpha;
            let harmonic_depth = self.start_harmonic_depth
                + (self.end_harmonic_depth - self.start_harmonic_depth) * alpha;
            let sub_harmonic_freq = self.start_sub_harmonic_freq
                + (self.end_sub_harmonic_freq - self.start_sub_harmonic_freq) * alpha;
            let sub_harmonic_depth = self.start_sub_harmonic_depth
                + (self.end_sub_harmonic_depth - self.start_sub_harmonic_depth) * alpha;
            let phase_osc_freq = self.start_phase_osc_freq
                + (self.end_phase_osc_freq - self.start_phase_osc_freq) * alpha;
            let phase_osc_range = self.start_phase_osc_range
                + (self.end_phase_osc_range - self.start_phase_osc_range) * alpha;

            let mut env_l = 1.0;
            if qam_am_freq_l != 0.0 && qam_am_depth_l != 0.0 {
                let phase = 2.0 * std::f32::consts::PI * qam_am_freq_l * t + qam_am_phase_offset_l;
                let mod_l1 = if mod_shape_l == 1.0 {
                    cos_lut(phase)
                } else {
                    let c = cos_lut(phase);
                    c.signum() * c.abs().powf(1.0 / mod_shape_l)
                };
                env_l *= 1.0 + qam_am_depth_l * mod_l1;
            }

            let mut env_r = 1.0;
            if qam_am_freq_r != 0.0 && qam_am_depth_r != 0.0 {
                let phase = 2.0 * std::f32::consts::PI * qam_am_freq_r * t + qam_am_phase_offset_r;
                let mod_r1 = if mod_shape_r == 1.0 {
                    cos_lut(phase)
                } else {
                    let c = cos_lut(phase);
                    c.signum() * c.abs().powf(1.0 / mod_shape_r)
                };
                env_r *= 1.0 + qam_am_depth_r * mod_r1;
            }

            if qam_am2_freq_l != 0.0 && qam_am2_depth_l != 0.0 {
                env_l *= 1.0
                    + qam_am2_depth_l
                        * cos_lut(
                            2.0 * std::f32::consts::PI * qam_am2_freq_l * t
                                + qam_am2_phase_offset_l,
                        );
            }
            if qam_am2_freq_r != 0.0 && qam_am2_depth_r != 0.0 {
                env_r *= 1.0
                    + qam_am2_depth_r
                        * cos_lut(
                            2.0 * std::f32::consts::PI * qam_am2_freq_r * t
                                + qam_am2_phase_offset_r,
                        );
            }

            let base_env_l = env_l;
            let base_env_r = env_r;

            if cross_mod_depth != 0.0 && self.cross_mod_delay_samples > 0 {
                let idx = self.cross_idx;
                env_l *= 1.0 + cross_mod_depth * (self.cross_env_r[idx] - 1.0);
                env_r *= 1.0 + cross_mod_depth * (self.cross_env_l[idx] - 1.0);
                self.cross_env_l[idx] = base_env_l;
                self.cross_env_r[idx] = base_env_r;
                self.cross_idx = (idx + 1) % self.cross_mod_delay_samples;
            }

            if sub_harmonic_freq != 0.0 && sub_harmonic_depth != 0.0 {
                let sub = cos_lut(2.0 * std::f32::consts::PI * sub_harmonic_freq * t);
                env_l *= 1.0 + sub_harmonic_depth * sub;
                env_r *= 1.0 + sub_harmonic_depth * sub;
            }

            self.phase_l += 2.0 * std::f32::consts::PI * base_freq_l * dt;
            self.phase_l = self.phase_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_r += 2.0 * std::f32::consts::PI * base_freq_r * dt;
            self.phase_r = self.phase_r.rem_euclid(2.0 * std::f32::consts::PI);
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if phase_osc_freq != 0.0 || phase_osc_range != 0.0 {
                let dphi = (phase_osc_range * 0.5)
                    * sin_lut(
                        2.0 * std::f32::consts::PI * phase_osc_freq * t
                            + self.phase_osc_phase_offset,
                    );
                ph_l -= dphi;
                ph_r += dphi;
            }

            let mut sig_l = env_l * cos_lut(ph_l);
            let mut sig_r = env_r * cos_lut(ph_r);

            if harmonic_depth != 0.0 {
                sig_l += harmonic_depth * env_l * cos_lut(self.harmonic_ratio * ph_l);
                sig_r += harmonic_depth * env_r * cos_lut(self.harmonic_ratio * ph_r);
            }

            if self.beating_sidebands && self.sideband_depth != 0.0 {
                let side = 2.0 * std::f32::consts::PI * self.sideband_offset * t;
                sig_l += self.sideband_depth * env_l * cos_lut(ph_l - side);
                sig_r += self.sideband_depth * env_r * cos_lut(ph_r - side);
                sig_l += self.sideband_depth * env_l * cos_lut(ph_l + side);
                sig_r += self.sideband_depth * env_r * cos_lut(ph_r + side);
            }

            let mut env_mult = 1.0;
            if self.attack_time > 0.0 && t < self.attack_time {
                env_mult *= t / self.attack_time;
            }
            if self.release_time > 0.0 && t > (self.duration - self.release_time) {
                env_mult *= (self.duration - t) / self.release_time;
            }

            output[i * 2] += sig_l * amp_l * env_mult;
            output[i * 2 + 1] += sig_r * amp_r * env_mult;

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for StereoAmIndependentVoice {
    fn process(&mut self, output: &mut [f32]) {
        let frames = output.len() / 2;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;

            let carrier_l = sin_lut(self.phase_carrier_l);
            let carrier_r = sin_lut(self.phase_carrier_r);
            let lfo_l = sin_lut(self.phase_mod_l);
            let lfo_r = sin_lut(self.phase_mod_r);
            let mod_l = 1.0 - self.mod_depth_l * (1.0 - lfo_l) * 0.5;
            let mod_r = 1.0 - self.mod_depth_r * (1.0 - lfo_r) * 0.5;

            output[i * 2] += carrier_l * mod_l * self.amp;
            output[i * 2 + 1] += carrier_r * mod_r * self.amp;

            let freq_l = self.carrier_freq - self.stereo_width_hz * 0.5;
            let freq_r = self.carrier_freq + self.stereo_width_hz * 0.5;
            self.phase_carrier_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_carrier_l = self.phase_carrier_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_carrier_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_carrier_r = self.phase_carrier_r.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_mod_l += 2.0 * std::f32::consts::PI * self.mod_freq_l * dt;
            self.phase_mod_l = self.phase_mod_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_mod_r += 2.0 * std::f32::consts::PI * self.mod_freq_r * dt;
            self.phase_mod_r = self.phase_mod_r.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for StereoAmIndependentTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let frames = output.len() / 2;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let mut alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let carrier_freq =
                self.start_carrier_freq + (self.end_carrier_freq - self.start_carrier_freq) * alpha;
            let stereo_width_hz = self.start_stereo_width_hz
                + (self.end_stereo_width_hz - self.start_stereo_width_hz) * alpha;
            let mod_freq_l =
                self.start_mod_freq_l + (self.end_mod_freq_l - self.start_mod_freq_l) * alpha;
            let mod_freq_r =
                self.start_mod_freq_r + (self.end_mod_freq_r - self.start_mod_freq_r) * alpha;
            let mod_depth_l =
                self.start_mod_depth_l + (self.end_mod_depth_l - self.start_mod_depth_l) * alpha;
            let mod_depth_r =
                self.start_mod_depth_r + (self.end_mod_depth_r - self.start_mod_depth_r) * alpha;

            let carrier_l = sin_lut(self.phase_carrier_l);
            let carrier_r = sin_lut(self.phase_carrier_r);
            let lfo_l = sin_lut(self.phase_mod_l);
            let lfo_r = sin_lut(self.phase_mod_r);
            let mod_l = 1.0 - mod_depth_l * (1.0 - lfo_l) * 0.5;
            let mod_r = 1.0 - mod_depth_r * (1.0 - lfo_r) * 0.5;

            output[i * 2] += carrier_l * mod_l * self.amp;
            output[i * 2 + 1] += carrier_r * mod_r * self.amp;

            let freq_l = carrier_freq - stereo_width_hz * 0.5;
            let freq_r = carrier_freq + stereo_width_hz * 0.5;
            self.phase_carrier_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_carrier_l = self.phase_carrier_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_carrier_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            self.phase_carrier_r = self.phase_carrier_r.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_mod_l += 2.0 * std::f32::consts::PI * mod_freq_l * dt;
            self.phase_mod_l = self.phase_mod_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_mod_r += 2.0 * std::f32::consts::PI * mod_freq_r * dt;
            self.phase_mod_r = self.phase_mod_r.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for WaveShapeStereoAmVoice {
    fn process(&mut self, output: &mut [f32]) {
        let frames = output.len() / 2;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;

            let carrier = sin_lut(self.phase_carrier);
            let shape_lfo_wave = sin_lut(self.phase_shape);
            let shape_env = 1.0 - self.shape_mod_depth * (1.0 - shape_lfo_wave) * 0.5;
            let modulated = carrier * shape_env;
            let sa = self.shape_amount.max(1e-6);
            let shaped = (modulated * sa).tanh() / sa.tanh();

            let stereo_lfo_l = sin_lut(self.phase_stereo_l);
            let stereo_lfo_r = sin_lut(self.phase_stereo_r);
            let mod_l = 1.0 - self.stereo_mod_depth_l * (1.0 - stereo_lfo_l) * 0.5;
            let mod_r = 1.0 - self.stereo_mod_depth_r * (1.0 - stereo_lfo_r) * 0.5;

            output[i * 2] += shaped * mod_l * self.amp;
            output[i * 2 + 1] += shaped * mod_r * self.amp;

            self.phase_carrier += 2.0 * std::f32::consts::PI * self.carrier_freq * dt;
            self.phase_carrier = self.phase_carrier.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_shape += 2.0 * std::f32::consts::PI * self.shape_mod_freq * dt;
            self.phase_shape = self.phase_shape.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_stereo_l += 2.0 * std::f32::consts::PI * self.stereo_mod_freq_l * dt;
            self.phase_stereo_l = self.phase_stereo_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_stereo_r += 2.0 * std::f32::consts::PI * self.stereo_mod_freq_r * dt;
            self.phase_stereo_r = self.phase_stereo_r.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for WaveShapeStereoAmTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let frames = output.len() / 2;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let mut alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let carrier_freq =
                self.start_carrier_freq + (self.end_carrier_freq - self.start_carrier_freq) * alpha;
            let shape_mod_freq = self.start_shape_mod_freq
                + (self.end_shape_mod_freq - self.start_shape_mod_freq) * alpha;
            let shape_mod_depth = self.start_shape_mod_depth
                + (self.end_shape_mod_depth - self.start_shape_mod_depth) * alpha;
            let shape_amount =
                self.start_shape_amount + (self.end_shape_amount - self.start_shape_amount) * alpha;
            let stereo_mod_freq_l = self.start_stereo_mod_freq_l
                + (self.end_stereo_mod_freq_l - self.start_stereo_mod_freq_l) * alpha;
            let stereo_mod_freq_r = self.start_stereo_mod_freq_r
                + (self.end_stereo_mod_freq_r - self.start_stereo_mod_freq_r) * alpha;
            let stereo_mod_depth_l = self.start_stereo_mod_depth_l
                + (self.end_stereo_mod_depth_l - self.start_stereo_mod_depth_l) * alpha;
            let stereo_mod_depth_r = self.start_stereo_mod_depth_r
                + (self.end_stereo_mod_depth_r - self.start_stereo_mod_depth_r) * alpha;

            let carrier = sin_lut(self.phase_carrier);
            let shape_lfo_wave = sin_lut(self.phase_shape);
            let shape_env = 1.0 - shape_mod_depth * (1.0 - shape_lfo_wave) * 0.5;
            let modulated = carrier * shape_env;
            let sa = shape_amount.max(1e-6);
            let shaped = (modulated * sa).tanh() / sa.tanh();

            let stereo_lfo_l = sin_lut(self.phase_stereo_l);
            let stereo_lfo_r = sin_lut(self.phase_stereo_r);
            let mod_l = 1.0 - stereo_mod_depth_l * (1.0 - stereo_lfo_l) * 0.5;
            let mod_r = 1.0 - stereo_mod_depth_r * (1.0 - stereo_lfo_r) * 0.5;

            output[i * 2] += shaped * mod_l * self.amp;
            output[i * 2 + 1] += shaped * mod_r * self.amp;

            self.phase_carrier += 2.0 * std::f32::consts::PI * carrier_freq * dt;
            self.phase_carrier = self.phase_carrier.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_shape += 2.0 * std::f32::consts::PI * shape_mod_freq * dt;
            self.phase_shape = self.phase_shape.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_stereo_l += 2.0 * std::f32::consts::PI * stereo_mod_freq_l * dt;
            self.phase_stereo_l = self.phase_stereo_l.rem_euclid(2.0 * std::f32::consts::PI);
            self.phase_stereo_r += 2.0 * std::f32::consts::PI * stereo_mod_freq_r * dt;
            self.phase_stereo_r = self.phase_stereo_r.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for SpatialAngleModulationVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;

            let sample = sin_lut(self.carrier_phase) * self.amp;
            let pan = sin_lut(self.spatial_phase) * self.path_radius;
            let (l, r) = pan2(sample, pan);
            output[i * 2] += l;
            output[i * 2 + 1] += r;

            self.carrier_phase += 2.0 * std::f32::consts::PI * self.carrier_freq * dt;
            self.carrier_phase = self.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
            self.spatial_phase += 2.0 * std::f32::consts::PI * self.beat_freq * dt;
            self.spatial_phase = self.spatial_phase.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for SpatialAngleModulationTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let carrier_freq =
                self.start_carrier_freq + (self.end_carrier_freq - self.start_carrier_freq) * alpha;
            let beat_freq =
                self.start_beat_freq + (self.end_beat_freq - self.start_beat_freq) * alpha;
            let path_radius =
                self.start_path_radius + (self.end_path_radius - self.start_path_radius) * alpha;

            let sample = sin_lut(self.carrier_phase) * self.amp;
            let pan = sin_lut(self.spatial_phase) * path_radius;
            let (l, r) = pan2(sample, pan);
            output[i * 2] += l;
            output[i * 2 + 1] += r;

            self.carrier_phase += 2.0 * std::f32::consts::PI * carrier_freq * dt;
            self.carrier_phase = self.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
            self.spatial_phase += 2.0 * std::f32::consts::PI * beat_freq * dt;
            self.spatial_phase = self.spatial_phase.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for RhythmicWaveshapingVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;

            let carrier = sin_lut(self.carrier_phase);
            let lfo = sin_lut(self.lfo_phase);
            let shape_lfo = 1.0 - self.mod_depth * (1.0 - lfo) * 0.5;
            let mod_input = carrier * shape_lfo;
            let amt = self.shape_amount.max(1e-6);
            let shaped = (mod_input * amt).tanh() / amt.tanh();
            let mono = shaped * self.amp;
            let (l, r) = pan2(mono, self.pan);
            output[i * 2] += l;
            output[i * 2 + 1] += r;

            self.carrier_phase += 2.0 * std::f32::consts::PI * self.carrier_freq * dt;
            self.carrier_phase = self.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
            self.lfo_phase += 2.0 * std::f32::consts::PI * self.mod_freq * dt;
            self.lfo_phase = self.lfo_phase.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for RhythmicWaveshapingTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.sample_idx as f32 / self.sample_rate;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let carrier_freq =
                self.start_carrier_freq + (self.end_carrier_freq - self.start_carrier_freq) * alpha;
            let mod_freq = self.start_mod_freq + (self.end_mod_freq - self.start_mod_freq) * alpha;
            let mod_depth =
                self.start_mod_depth + (self.end_mod_depth - self.start_mod_depth) * alpha;
            let shape_amount =
                self.start_shape_amount + (self.end_shape_amount - self.start_shape_amount) * alpha;

            let carrier = sin_lut(self.carrier_phase);
            let lfo = sin_lut(self.lfo_phase);
            let shape_lfo = 1.0 - mod_depth * (1.0 - lfo) * 0.5;
            let mod_input = carrier * shape_lfo;
            let amt = shape_amount.max(1e-6);
            let shaped = (mod_input * amt).tanh() / amt.tanh();
            let mono = shaped * self.amp;
            let (l, r) = pan2(mono, self.pan);
            output[i * 2] += l;
            output[i * 2 + 1] += r;

            self.carrier_phase += 2.0 * std::f32::consts::PI * carrier_freq * dt;
            self.carrier_phase = self.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
            self.lfo_phase += 2.0 * std::f32::consts::PI * mod_freq * dt;
            self.lfo_phase = self.lfo_phase.rem_euclid(2.0 * std::f32::consts::PI);

            self.remaining_samples -= 1;
            self.sample_idx += 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl Voice for SubliminalEncodeVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let sample = self.samples[self.position];
            output[i * 2] += sample;
            output[i * 2 + 1] += self.samples[self.position + 1];
            self.position += 2;
            self.remaining_samples -= 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

impl VoiceKind {
    /// Returns the current accumulated phases (phase_l, phase_r) for voices that track stereo phase.
    /// This is used to maintain phase continuity when transitioning between voice instances.
    pub fn get_phases(&self) -> Option<(f32, f32)> {
        match self {
            VoiceKind::BinauralBeat(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::BinauralBeatTransition(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::IsochronicTone(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::IsochronicToneTransition(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::QamBeat(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::QamBeatTransition(v) => Some((v.phase_l, v.phase_r)),
            VoiceKind::VolumeEnvelope(v) => v.inner.get_phases(),
            // Other voice types don't track stereo carrier phases
            _ => None,
        }
    }

    /// Sets the accumulated phases for voices that track stereo phase.
    /// This ensures phase continuity when creating a new voice instance to continue
    /// from where the previous one left off.
    pub fn set_phases(&mut self, phase_l: f32, phase_r: f32) {
        match self {
            VoiceKind::BinauralBeat(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::BinauralBeatTransition(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::IsochronicTone(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::IsochronicToneTransition(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::QamBeat(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::QamBeatTransition(v) => {
                v.phase_l = phase_l;
                v.phase_r = phase_r;
            }
            VoiceKind::VolumeEnvelope(v) => {
                v.inner.set_phases(phase_l, phase_r);
            }
            // Other voice types don't track stereo carrier phases
            _ => {}
        }
    }
}

impl Voice for VoiceKind {
    fn process(&mut self, output: &mut [f32]) {
        match self {
            VoiceKind::BinauralBeat(v) => v.process(output),
            VoiceKind::BinauralBeatTransition(v) => v.process(output),
            VoiceKind::IsochronicTone(v) => v.process(output),
            VoiceKind::IsochronicToneTransition(v) => v.process(output),
            VoiceKind::QamBeat(v) => v.process(output),
            VoiceKind::QamBeatTransition(v) => v.process(output),
            VoiceKind::StereoAmIndependent(v) => v.process(output),
            VoiceKind::StereoAmIndependentTransition(v) => v.process(output),
            VoiceKind::WaveShapeStereoAm(v) => v.process(output),
            VoiceKind::WaveShapeStereoAmTransition(v) => v.process(output),
            VoiceKind::SpatialAngleModulation(v) => v.process(output),
            VoiceKind::SpatialAngleModulationTransition(v) => v.process(output),
            VoiceKind::RhythmicWaveshaping(v) => v.process(output),
            VoiceKind::RhythmicWaveshapingTransition(v) => v.process(output),
            VoiceKind::SubliminalEncode(v) => v.process(output),
            VoiceKind::VolumeEnvelope(v) => v.process(output),
            VoiceKind::NoiseSweptNotch(v) => v.process(output),
            VoiceKind::NoiseSweptNotchTransition(v) => v.process(output),
        }
    }

    fn is_finished(&self) -> bool {
        match self {
            VoiceKind::BinauralBeat(v) => v.is_finished(),
            VoiceKind::BinauralBeatTransition(v) => v.is_finished(),
            VoiceKind::IsochronicTone(v) => v.is_finished(),
            VoiceKind::IsochronicToneTransition(v) => v.is_finished(),
            VoiceKind::QamBeat(v) => v.is_finished(),
            VoiceKind::QamBeatTransition(v) => v.is_finished(),
            VoiceKind::StereoAmIndependent(v) => v.is_finished(),
            VoiceKind::StereoAmIndependentTransition(v) => v.is_finished(),
            VoiceKind::WaveShapeStereoAm(v) => v.is_finished(),
            VoiceKind::WaveShapeStereoAmTransition(v) => v.is_finished(),
            VoiceKind::SpatialAngleModulation(v) => v.is_finished(),
            VoiceKind::SpatialAngleModulationTransition(v) => v.is_finished(),
            VoiceKind::RhythmicWaveshaping(v) => v.is_finished(),
            VoiceKind::RhythmicWaveshapingTransition(v) => v.is_finished(),
            VoiceKind::SubliminalEncode(v) => v.is_finished(),
            VoiceKind::VolumeEnvelope(v) => v.is_finished(),
            VoiceKind::NoiseSweptNotch(v) => v.is_finished(),
            VoiceKind::NoiseSweptNotchTransition(v) => v.is_finished(),
        }
    }
}

pub fn voices_for_step(step: &StepData, sample_rate: f32) -> Vec<StepVoice> {
    let mut out: Vec<StepVoice> = Vec::new();
    for voice in &step.voices {
        if let Some(v) = create_voice(voice, step.duration as f32, sample_rate) {
            out.push(v);
        }
    }
    out
}

fn create_voice(data: &VoiceData, duration: f32, sample_rate: f32) -> Option<StepVoice> {
    let mut voice =
        match data.synth_function_name.as_str() {
            "binaural_beat" => {
                VoiceKind::BinauralBeat(BinauralBeatVoice::new(&data.params, duration, sample_rate))
            }
            "binaural_beat_transition" => VoiceKind::BinauralBeatTransition(
                BinauralBeatTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "isochronic_tone" => VoiceKind::IsochronicTone(IsochronicToneVoice::new(
                &data.params,
                duration,
                sample_rate,
            )),
            "isochronic_tone_transition" => VoiceKind::IsochronicToneTransition(
                IsochronicToneTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "qam_beat" => {
                VoiceKind::QamBeat(QamBeatVoice::new(&data.params, duration, sample_rate))
            }
            "qam_beat_transition" => VoiceKind::QamBeatTransition(QamBeatTransitionVoice::new(
                &data.params,
                duration,
                sample_rate,
            )),
            "rhythmic_waveshaping" => VoiceKind::RhythmicWaveshaping(
                RhythmicWaveshapingVoice::new(&data.params, duration, sample_rate),
            ),
            "rhythmic_waveshaping_transition" => VoiceKind::RhythmicWaveshapingTransition(
                RhythmicWaveshapingTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "stereo_am_independent" => VoiceKind::StereoAmIndependent(
                StereoAmIndependentVoice::new(&data.params, duration, sample_rate),
            ),
            "stereo_am_independent_transition" => VoiceKind::StereoAmIndependentTransition(
                StereoAmIndependentTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "wave_shape_stereo_am" => VoiceKind::WaveShapeStereoAm(WaveShapeStereoAmVoice::new(
                &data.params,
                duration,
                sample_rate,
            )),
            "wave_shape_stereo_am_transition" => VoiceKind::WaveShapeStereoAmTransition(
                WaveShapeStereoAmTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "spatial_angle_modulation" => VoiceKind::SpatialAngleModulation(
                SpatialAngleModulationVoice::new(&data.params, duration, sample_rate),
            ),
            "spatial_angle_modulation_transition" => VoiceKind::SpatialAngleModulationTransition(
                SpatialAngleModulationTransitionVoice::new(&data.params, duration, sample_rate),
            ),
            "subliminal_encode" => VoiceKind::SubliminalEncode(SubliminalEncodeVoice::new(
                &data.params,
                duration,
                sample_rate,
            )),
            "noise_swept_notch" | "noise" => VoiceKind::NoiseSweptNotch(NoiseSweptNotchVoice::new(
                &data.params,
                duration,
                sample_rate,
            )),
            "noise_swept_notch_transition" | "noise_transition" => {
                VoiceKind::NoiseSweptNotchTransition(NoiseSweptNotchTransitionVoice::new(
                    &data.params,
                    duration,
                    sample_rate,
                ))
            }
            _ => return None,
        };

    if let Some(env) = &data.volume_envelope {
        let env_vec = build_volume_envelope(env, duration, sample_rate as u32);
        voice =
            VoiceKind::VolumeEnvelope(Box::new(VolumeEnvelopeVoice::new(Box::new(voice), env_vec)));
    }
    let voice_type = match data.voice_type.to_lowercase().as_str() {
        "noise" => VoiceType::Noise,
        "binaural" => VoiceType::Binaural,
        _ => VoiceType::Other,
    };

    let normalization_peak = voice.normalization_peak();

    Some(StepVoice {
        kind: voice,
        voice_type,
        normalization_peak,
    })
}

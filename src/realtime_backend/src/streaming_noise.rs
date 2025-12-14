use crate::noise_params::NoiseParams;
use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type, Q_BUTTERWORTH_F32};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc;

// --- Constants for Python-compat OLA mode ---
const BLOCK_SIZE: usize = 4096;
const HOP_SIZE: usize = BLOCK_SIZE / 2; // 2048, 50% overlap

// --- Crossfade length for FFT noise regeneration ---
const CROSSFADE_SAMPLES: usize = 2048;

// --- Renormalization window for post-filter RMS tracking ---
const RENORM_WINDOW: usize = 4096;

// --- Helper Functions ---

/// Scipy-compatible sawtooth with width=0.5 (triangle wave)
/// Python: signal.sawtooth(phase, width=0.5)
fn scipy_sawtooth_triangle(phase: f32) -> f32 {
    let t = phase.rem_euclid(2.0 * std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let width = 0.5f32;
    if t < width {
        -1.0 + 2.0 * t / width
    } else {
        1.0 - 2.0 * (t - width) / (1.0 - width)
    }
}

fn resolved_noise_name(params: &NoiseParams) -> String {
    if let Some(Value::String(name)) = params.noise_parameters.get("name") {
        return name.clone();
    }

    "pink".to_string()
}

/// LFO value computation matching Python's behavior
/// Python "sine" uses cosine: np.cos(2 * np.pi * lfo_freq * t + phase_offset)
/// Python "triangle" uses scipy.signal.sawtooth(phase, width=0.5)
fn lfo_value(phase: f32, waveform: &str) -> f32 {
    if waveform.eq_ignore_ascii_case("triangle") {
        scipy_sawtooth_triangle(phase)
    } else {
        // "sine" in Python actually uses cosine
        crate::dsp::trig::cos_lut(phase)
    }
}

// --- Notch Filter Logic ---

#[derive(Clone)]
struct Coeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

#[derive(Clone, Copy)]
struct BiquadState64 {
    z1: f64,
    z2: f64,
}

impl BiquadState64 {
    fn new() -> Self {
        Self { z1: 0.0, z2: 0.0 }
    }
}

/// Compute notch coefficients in f64 (matching SciPy's float64 path).
/// IMPORTANT: We keep the coefficients in f64 all the way through filtering.
/// With large cascade counts, doing this in f32 can accumulate enough numeric
/// error to cause huge peak spikes (or broad attenuation), which then makes
/// peak-based normalization collapse the perceived loudness.
fn notch_coeffs_f64(freq: f64, q: f64, sample_rate: f64) -> Coeffs {
    let w0 = 2.0 * std::f64::consts::PI * freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    // SciPy iirnotch (biquad form):
    // b = [1, -2cos(w0), 1]
    // a = [1+alpha, -2cos(w0), 1-alpha]
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    Coeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// Apply a biquad to a block with persistent state.
///
/// IMPORTANT: In the original implementation, we reset state to zero on every call.
/// With large cascade counts that are re-applied on every streaming block, that creates
/// large block-edge transients (ringing) that inflate the *peak* seen during calibration.
/// Peak-based normalization then collapses the perceived loudness (exactly your symptom).
fn biquad_block(block: &mut [f64], coeffs: &Coeffs, st: &mut BiquadState64) {
    // Direct Form II Transposed (biquad), in f64.
    let mut z1 = st.z1;
    let mut z2 = st.z2;

    for sample in block.iter_mut() {
        let input = *sample;
        let out = input * coeffs.b0 + z1;
        z1 = input * coeffs.b1 - out * coeffs.a1 + z2;
        z2 = input * coeffs.b2 - out * coeffs.a2;
        *sample = out;
    }

    st.z1 = z1;
    st.z2 = z2;
}

// --- FFT Based Noise Generator (Matches Python's ColoredNoiseGenerator) ---

struct FftNoiseGenerator {
    buffer: Vec<f32>,
    next_buffer: Option<Vec<f32>>,
    cursor: usize,
    size: usize,
    exponent: f32,
    high_exponent: f32,
    distribution_curve: f32,
    lowcut: Option<f32>,
    highcut: Option<f32>,
    sample_rate: f32,
    lp_filters: Option<Vec<DirectForm2Transposed<f32>>>,
    hp_filters: Option<Vec<DirectForm2Transposed<f32>>>,
    base_amplitude: f32,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    rng: StdRng,
    normal: Normal<f32>,
    renorm_gain: f32,
    pre_rms_accum: f32,
    post_rms_accum: f32,
    rms_samples: usize,
}

impl FftNoiseGenerator {
    fn preset_for_type(nt: &str) -> Option<(f32, f32, f32, Option<f32>, Option<f32>, f32)> {
        match nt {
            "pink" => Some((1.0, 1.0, 1.0, None, None, 1.0)),
            "brown" => Some((2.0, 2.0, 1.0, None, None, 1.0)),
            "red" => Some((2.0, 1.5, 1.0, None, None, 1.0)),
            "green" => Some((0.0, 0.0, 1.0, Some(100.0), Some(8000.0), 1.0)),
            "blue" => Some((-1.0, -1.0, 1.0, None, None, 1.0)),
            "purple" => Some((-2.0, -2.0, 1.0, None, None, 1.0)),
            "deep brown" => Some((2.5, 2.0, 1.0, None, None, 1.0)),
            "white" => Some((0.0, 0.0, 1.0, None, None, 1.0)),
            _ => None,
        }
    }

    fn new(params: &NoiseParams, sample_rate: f32) -> Self {
        let noise_label = resolved_noise_name(params);
        let nt = noise_label.to_lowercase();
        let preset = Self::preset_for_type(nt.as_str());

        let exponent = params
            .exponent
            .or_else(|| preset.map(|p| p.0))
            .unwrap_or(0.0);
        let high_exponent = params
            .high_exponent
            .or_else(|| preset.map(|p| p.1))
            .unwrap_or(exponent);
        let distribution_curve = params
            .distribution_curve
            .or_else(|| preset.map(|p| p.2))
            .unwrap_or(1.0)
            .max(1e-6);
        let lowcut = params.lowcut.or_else(|| preset.and_then(|p| p.3));
        let highcut = params.highcut.or_else(|| preset.and_then(|p| p.4));
        let amplitude = params
            .amplitude
            .or_else(|| preset.map(|p| p.5))
            .unwrap_or(1.0);
        let seed = params.seed.unwrap_or(1).max(0) as u64;

        let requested = (params.duration_seconds.max(0.0) * sample_rate) as usize;
        let mut size = if requested > 0 { requested } else { 1 << 17 };
        if size < 8 {
            size = 8;
        }
        if size % 2 != 0 {
            size += 1;
        }

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(size);
        let fft_inverse = planner.plan_fft_inverse(size);

        let rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let nyquist = sample_rate / 2.0;

        let mut lp_filters = None;
        if let Some(fc) = lowcut {
            if fc > 0.0 && fc < nyquist {
                if let Some(c) = Coefficients::<f32>::from_params(
                    Type::HighPass,
                    sample_rate.hz(),
                    fc.hz(),
                    Q_BUTTERWORTH_F32,
                )
                .ok()
                {
                    lp_filters = Some(vec![DirectForm2Transposed::<f32>::new(c); 2]);
                }
            }
        }

        let mut hp_filters = None;
        if let Some(fc) = highcut {
            if fc > 0.0 && fc < nyquist {
                if let Some(c) = Coefficients::<f32>::from_params(
                    Type::LowPass,
                    sample_rate.hz(),
                    fc.hz(),
                    Q_BUTTERWORTH_F32,
                )
                .ok()
                {
                    hp_filters = Some(vec![DirectForm2Transposed::<f32>::new(c); 2]);
                }
            }
        }

        let mut gen = Self {
            buffer: Vec::new(),
            next_buffer: None,
            cursor: 0,
            size,
            exponent,
            high_exponent,
            distribution_curve,
            lowcut,
            highcut,
            sample_rate,
            lp_filters,
            hp_filters,
            base_amplitude: amplitude,
            fft_forward,
            fft_inverse,
            rng,
            normal,
            renorm_gain: 1.0,
            pre_rms_accum: 0.0,
            post_rms_accum: 0.0,
            rms_samples: 0,
        };

        gen.buffer = gen.regenerate_buffer();
        gen
    }

    fn regenerate_buffer(&mut self) -> Vec<f32> {
        let mut white: Vec<Complex<f32>> = (0..self.size)
            .map(|_| Complex::new(self.normal.sample(&mut self.rng), 0.0))
            .collect();

        self.fft_forward.process(&mut white);

        let nyquist = self.sample_rate / 2.0;
        let min_f = self.sample_rate / (self.size as f32);

        if !white.is_empty() {
            white[0] = Complex::new(0.0, 0.0);
        }

        for i in 1..=self.size / 2 {
            let freq = i as f32 * self.sample_rate / self.size as f32;
            if freq <= 0.0 {
                continue;
            }

            let log_min = min_f.ln();
            let log_max = nyquist.ln();
            let log_f = freq.ln();

            let denom = (log_max - log_min).max(1e-12);
            let mut log_norm = (log_f - log_min) / denom;
            log_norm = log_norm.clamp(0.0, 1.0);

            let interp = log_norm.powf(self.distribution_curve);
            let current_exp = self.exponent + (self.high_exponent - self.exponent) * interp;

            let scale = freq.powf(-current_exp / 2.0);

            white[i] *= scale;
            if i < self.size / 2 {
                white[self.size - i] = white[i].conj();
            }
        }

        self.fft_inverse.process(&mut white);

        let mut output: Vec<f32> = white.iter().map(|c| c.re / self.size as f32).collect();
        if let Some(max_val) = output.iter().fold(None, |acc: Option<f32>, &v| {
            Some(acc.map_or(v.abs(), |m| m.max(v.abs())))
        }) {
            if max_val > 1e-9 {
                for x in &mut output {
                    *x /= max_val;
                }
            }
        }

        output
    }

    fn crossfade_len(&self) -> usize {
        self.buffer.len().min(CROSSFADE_SAMPLES)
    }

    fn next(&mut self) -> f32 {
        if self.buffer.is_empty() {
            self.buffer = self.regenerate_buffer();
        }

        let crossfade_len = self.crossfade_len();

        if self.next_buffer.is_none() && self.cursor + crossfade_len >= self.buffer.len() {
            self.next_buffer = Some(self.regenerate_buffer());
        }

        let mut sample = if let Some(ref next_buf) = self.next_buffer {
            let crossfade_start = self.buffer.len().saturating_sub(crossfade_len);
            if self.cursor >= crossfade_start && crossfade_len > 0 && !next_buf.is_empty() {
                let idx = self.cursor - crossfade_start;
                let t = idx as f32 / crossfade_len as f32;
                let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                let fade_in = 1.0 - fade_out;
                let next_sample = next_buf.get(idx).copied().unwrap_or(0.0);
                self.buffer[self.cursor] * fade_out + next_sample * fade_in
            } else {
                self.buffer[self.cursor]
            }
        } else {
            self.buffer[self.cursor]
        };

        self.cursor += 1;

        if self.cursor >= self.buffer.len() {
            let consumed_from_next = if self.next_buffer.is_some() {
                crossfade_len
            } else {
                0
            };

            if let Some(next) = self.next_buffer.take() {
                let skip = consumed_from_next.min(next.len());
                self.buffer = next;
                self.cursor = skip;
            } else {
                self.buffer = self.regenerate_buffer();
                self.cursor = 0;
            }
        }

        let pre_filter_sample = sample;

        if let Some(ref mut filters) = self.lp_filters {
            for f in filters {
                sample = f.run(sample);
            }
        }
        if let Some(ref mut filters) = self.hp_filters {
            for f in filters {
                sample = f.run(sample);
            }
        }

        sample = self.apply_post_filter_renorm(pre_filter_sample, sample);

        sample * self.base_amplitude
    }

    fn apply_post_filter_renorm(&mut self, pre: f32, post: f32) -> f32 {
        self.pre_rms_accum += pre * pre;
        self.post_rms_accum += post * post;
        self.rms_samples += 1;

        if self.rms_samples >= RENORM_WINDOW {
            let pre_rms = (self.pre_rms_accum / self.rms_samples as f32).sqrt();
            let post_rms = (self.post_rms_accum / self.rms_samples as f32).sqrt();

            if pre_rms > 1e-6 && post_rms > 1e-6 {
                let target_gain = (pre_rms / post_rms).clamp(0.25, 16.0);
                // Smooth gain to avoid sudden jumps
                self.renorm_gain = 0.9 * self.renorm_gain + 0.1 * target_gain;
            } else {
                // Drift gain back toward unity if measurements are unreliable
                self.renorm_gain = 0.9 * self.renorm_gain + 0.1;
            }

            self.pre_rms_accum = 0.0;
            self.post_rms_accum = 0.0;
            self.rms_samples = 0;
        }

        post * self.renorm_gain
    }
}

// --- Precomputed Hann window (matching np.hanning) ---

fn hann_window(size: usize) -> Vec<f64> {
    // np.hanning(N) = 0.5 - 0.5 * cos(2*pi*n/(N-1)), n = 0..N-1
    (0..size)
        .map(|n| {
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (size as f64 - 1.0)).cos()
        })
        .collect()
}

// --- OLA (Overlap-Add) State for Python-compat streaming ---

struct OlaState {
    // Ring buffer for input samples (mono base noise)
    input_ring: Vec<f32>,
    input_write_pos: usize,
    input_samples_buffered: usize,

    // Overlap-add accumulators for each channel (ring buffer style)
    out_acc_l: Vec<f32>,
    out_acc_r: Vec<f32>,
    win_acc: Vec<f32>,

    // Position tracking within the accumulator ring
    acc_read_pos: usize,
    acc_write_pos: usize,

    // Samples ready to output
    samples_ready: usize,

    // Absolute sample index for time tracking (block start positions)
    absolute_block_start: usize,

    // Precomputed Hann window
    window: Vec<f64>,

    // Scratch buffers for block processing
    block_l: Vec<f64>,
    block_r: Vec<f64>,
}

impl OlaState {
    fn new() -> Self {
        let window = hann_window(BLOCK_SIZE);
        let acc_size = BLOCK_SIZE * 2;

        Self {
            input_ring: vec![0.0; BLOCK_SIZE],
            input_write_pos: 0,
            input_samples_buffered: 0,
            out_acc_l: vec![0.0; acc_size],
            out_acc_r: vec![0.0; acc_size],
            win_acc: vec![0.0; acc_size],
            acc_read_pos: 0,
            acc_write_pos: 0,
            samples_ready: 0,
            absolute_block_start: 0,
            window,
            block_l: vec![0.0; BLOCK_SIZE],
            block_r: vec![0.0; BLOCK_SIZE],
        }
    }
}

// --- Sweep parameters for varying mode ---

#[derive(Clone)]
struct SweepParams {
    start_min: f32,
    end_min: f32,
    start_max: f32,
    end_max: f32,
    start_q: f32,
    end_q: f32,
    start_casc: usize,
    end_casc: usize,
}

#[derive(Clone)]
struct SweepRuntime {
    max_casc: usize,
    // Each cascade stage must preserve its own state across blocks, like a true
    // series of biquads applied to a continuous signal.
    l_main: Vec<BiquadState64>,
    r_main: Vec<BiquadState64>,
    l_extra: Vec<BiquadState64>,
    r_extra: Vec<BiquadState64>,
}

impl SweepRuntime {
    fn new(max_casc: usize) -> Self {
        let max_casc = max_casc.max(1);
        Self {
            max_casc,
            l_main: vec![BiquadState64::new(); max_casc],
            r_main: vec![BiquadState64::new(); max_casc],
            l_extra: vec![BiquadState64::new(); max_casc],
            r_extra: vec![BiquadState64::new(); max_casc],
        }
    }
}

impl SweepParams {
    fn interpolate_at(&self, t: f32) -> (f32, f32, f32, usize) {
        let t = t.clamp(0.0, 1.0);
        let min_freq = self.start_min + (self.end_min - self.start_min) * t;
        let max_freq = self.start_max + (self.end_max - self.start_max) * t;
        let q = self.start_q + (self.end_q - self.start_q) * t;
        let casc_f = self.start_casc as f32 + (self.end_casc as f32 - self.start_casc as f32) * t;
        let casc = casc_f.round().max(1.0) as usize;
        (min_freq, max_freq, q, casc)
    }
}

pub struct StreamingNoise {
    sample_rate: f32,
    duration_samples: usize,

    // LFO parameters
    start_lfo_freq: f32,
    end_lfo_freq: f32,
    lfo_freq: f32,
    start_lfo_phase_offset: f32,
    end_lfo_phase_offset: f32,
    start_intra_offset: f32,
    end_intra_offset: f32,
    lfo_waveform: String,
    initial_offset: f32,

    // Sweep parameters (for varying mode)
    sweep_params: Vec<SweepParams>,

    // Persistent biquad states per sweep (per channel + per pass + per cascade stage)
    sweep_runtime: Vec<SweepRuntime>,

    // Mode flags
    transition: bool,

    // FFT Generator for all noise modes
    fft_gen: FftNoiseGenerator,

    // OLA state for Python-compat mode
    ola: OlaState,

    // Total samples output so far (for absolute time tracking)
    total_samples_output: usize,
}

impl StreamingNoise {
    pub fn new(params: &NoiseParams, sample_rate: u32) -> Self {
        let sample_rate_f = sample_rate as f32;
        let duration_samples = (params.duration_seconds * sample_rate_f) as usize;

        let lfo_freq = if params.transition {
            params.start_lfo_freq
        } else if params.lfo_freq != 0.0 {
            params.lfo_freq
        } else {
            1.0 / 12.0
        };

        let sweep_params: Vec<SweepParams> = params
            .sweeps
            .iter()
            .map(|sw| {
                let start_min = if sw.start_min > 0.0 {
                    sw.start_min
                } else {
                    1000.0
                };
                let end_min = if sw.end_min > 0.0 {
                    sw.end_min
                } else {
                    start_min
                };
                let start_max = if sw.start_max > 0.0 {
                    sw.start_max.max(start_min + 1.0)
                } else {
                    start_min + 9000.0
                };
                let end_max = if sw.end_max > 0.0 {
                    sw.end_max.max(end_min + 1.0)
                } else {
                    start_max
                };
                let start_q = if sw.start_q > 0.0 { sw.start_q } else { 25.0 };
                let end_q = if sw.end_q > 0.0 { sw.end_q } else { start_q };
                let start_casc = if sw.start_casc > 0 { sw.start_casc } else { 10 };
                let end_casc = if sw.end_casc > 0 {
                    sw.end_casc
                } else {
                    start_casc
                };
                SweepParams {
                    start_min,
                    end_min,
                    start_max,
                    end_max,
                    start_q,
                    end_q,
                    start_casc,
                    end_casc,
                }
            })
            .collect();

        let sweep_runtime: Vec<SweepRuntime> = sweep_params
            .iter()
            .map(|sp| {
                let max_casc = sp.start_casc.max(sp.end_casc).max(1);
                SweepRuntime::new(max_casc)
            })
            .collect();

        Self {
            sample_rate: sample_rate_f,
            duration_samples,
            start_lfo_freq: if params.start_lfo_freq > 0.0 {
                params.start_lfo_freq
            } else {
                lfo_freq
            },
            end_lfo_freq: if params.end_lfo_freq > 0.0 {
                params.end_lfo_freq
            } else {
                lfo_freq
            },
            lfo_freq,
            start_lfo_phase_offset: params.start_lfo_phase_offset_deg.to_radians(),
            end_lfo_phase_offset: params.end_lfo_phase_offset_deg.to_radians(),
            start_intra_offset: params.start_intra_phase_offset_deg.to_radians(),
            end_intra_offset: params.end_intra_phase_offset_deg.to_radians(),
            lfo_waveform: params.lfo_waveform.clone(),
            initial_offset: params.initial_offset,
            sweep_params,
            sweep_runtime,
            transition: params.transition,
            fft_gen: FftNoiseGenerator::new(params, sample_rate_f),
            ola: OlaState::new(),
            total_samples_output: 0,
        }
    }

    pub fn new_with_calibrated_peak(
        params: &NoiseParams,
        sample_rate: u32,
        calibration_frames: usize,
    ) -> (Self, f32) {
        let frames = calibration_frames.max(1);

        let mut calibration_gen = StreamingNoise::new(params, sample_rate);
        let mut scratch = vec![0.0f32; frames * 2];
        calibration_gen.generate(&mut scratch);

        // IMPORTANT: Using absolute max is extremely fragile for streaming.
        // Deep/high-Q cascades can create rare block-edge spikes (especially with brown noise)
        // that make `peak` enormous. Peak-normalization then makes everything sound *silent*.
        // We use a robust peak estimate (99.9th percentile of |x|) to avoid one-sample poison.
        let mut abs_vals: Vec<f32> = scratch.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let idx = ((abs_vals.len() as f64) * 0.999).floor() as usize;
        let idx = idx.min(abs_vals.len().saturating_sub(1));
        let peak = abs_vals.get(idx).copied().unwrap_or(0.0).max(1e-9);

        let generator = StreamingNoise::new(params, sample_rate);

        (generator, peak)
    }

    pub fn skip_samples(&mut self, n: usize) {
        let mut scratch = vec![0.0f32; n * 2];
        self.generate(&mut scratch);
    }

    fn next_base(&mut self) -> f32 {
        self.fft_gen.next()
    }

    /// Compute the transition fraction at a given absolute sample index
    fn transition_fraction(&self, sample_idx: usize) -> f32 {
        if !self.transition || self.duration_samples == 0 {
            return 0.0;
        }
        (sample_idx as f32 / self.duration_samples as f32).clamp(0.0, 1.0)
    }

    /// Interpolate LFO frequency at a given transition fraction
    fn interpolate_lfo_freq(&self, t: f32) -> f32 {
        if !self.transition {
            return self.lfo_freq;
        }
        self.start_lfo_freq + (self.end_lfo_freq - self.start_lfo_freq) * t
    }

    /// Interpolate phase offset at a given transition fraction
    fn interpolate_phase_offset(&self, t: f32) -> f32 {
        if !self.transition {
            return self.start_lfo_phase_offset;
        }
        self.start_lfo_phase_offset + (self.end_lfo_phase_offset - self.start_lfo_phase_offset) * t
    }

    /// Interpolate intra offset at a given transition fraction
    fn interpolate_intra_offset(&self, t: f32) -> f32 {
        if !self.transition {
            return self.start_intra_offset;
        }
        self.start_intra_offset + (self.end_intra_offset - self.start_intra_offset) * t
    }

    /// Compute LFO phase at a given sample index using Python's approach:
    /// t = sample_idx / sample_rate + initial_offset
    /// phase = 2 * pi * lfo_freq * t + phase_offset
    fn compute_lfo_phase(&self, sample_idx: usize, lfo_freq: f32, extra_phase_offset: f32) -> f32 {
        let t = sample_idx as f32 / self.sample_rate + self.initial_offset;
        2.0 * std::f32::consts::PI * lfo_freq * t + extra_phase_offset
    }

    /// Process a single block using overlap-add approach (Python-compat mode)
    fn process_ola_block(&mut self) {
        let acc_size = self.ola.out_acc_l.len();
        let block_start_idx = self.ola.absolute_block_start;

        // Get transition fraction at block center
        let block_center_idx = block_start_idx + BLOCK_SIZE / 2;
        let t = self.transition_fraction(block_center_idx);

        // Interpolate LFO parameters at block center
        let lfo_freq = self.interpolate_lfo_freq(t);
        let phase_offset = self.interpolate_phase_offset(t);
        let intra_offset = self.interpolate_intra_offset(t);

        // Compute LFO phase at block center for L and R channels
        // Python: phase = 2*pi*lfo_freq*t + phase_offset, where phase_offset is in radians
        // For L channel: no extra offset (base phase)
        // For R channel: add lfo_phase_offset
        let l_phase_base = self.compute_lfo_phase(block_center_idx, lfo_freq, 0.0);
        let r_phase_base = self.compute_lfo_phase(block_center_idx, lfo_freq, phase_offset);

        // Get LFO values at block center (frozen for entire block)
        let lfo_l_main = lfo_value(l_phase_base, &self.lfo_waveform);
        let lfo_r_main = lfo_value(r_phase_base, &self.lfo_waveform);
        let lfo_l_extra = lfo_value(l_phase_base + intra_offset, &self.lfo_waveform);
        let lfo_r_extra = lfo_value(r_phase_base + intra_offset, &self.lfo_waveform);

        // Copy input block from ring buffer and apply Hann window
        // Also compute RMS of the input for later compensation (matching Python's rms_in)
        let mut sum_sq_in: f64 = 0.0;
        for i in 0..BLOCK_SIZE {
            let ring_idx =
                (self.ola.input_write_pos + BLOCK_SIZE - self.ola.input_samples_buffered + i)
                    % BLOCK_SIZE;
            let base = self.ola.input_ring[ring_idx] as f64;
            let windowed = base * self.ola.window[i];
            self.ola.block_l[i] = windowed;
            self.ola.block_r[i] = windowed;
            sum_sq_in += windowed * windowed;
        }
        let rms_in = (sum_sq_in / BLOCK_SIZE as f64).sqrt();

        // Apply notch filters for each sweep (coefficients frozen per block at block center)
        // NOTE: We keep per-stage filter state across blocks to avoid block-edge ringing spikes.
        for (si, sp) in self.sweep_params.iter().enumerate() {
            let rt = &mut self.sweep_runtime[si];
            let (min_freq, max_freq, q, casc) = sp.interpolate_at(t);
            let casc = casc.min(rt.max_casc);
            let center_freq = (min_freq + max_freq) * 0.5;
            let freq_range = (max_freq - min_freq) * 0.5;

            // Left channel - main pass
            let notch_freq_l_main = center_freq + freq_range * lfo_l_main;
            if notch_freq_l_main > 0.0 && notch_freq_l_main < self.sample_rate * 0.49 {
                let coeffs =
                    notch_coeffs_f64(notch_freq_l_main as f64, q as f64, self.sample_rate as f64);
                for stage in 0..casc {
                    biquad_block(&mut self.ola.block_l, &coeffs, &mut rt.l_main[stage]);
                }
            }

            // Right channel - main pass
            let notch_freq_r_main = center_freq + freq_range * lfo_r_main;
            if notch_freq_r_main > 0.0 && notch_freq_r_main < self.sample_rate * 0.49 {
                let coeffs =
                    notch_coeffs_f64(notch_freq_r_main as f64, q as f64, self.sample_rate as f64);
                for stage in 0..casc {
                    biquad_block(&mut self.ola.block_r, &coeffs, &mut rt.r_main[stage]);
                }
            }

            // Extra pass if intra_offset is non-zero
            if intra_offset.abs() > 1e-6 {
                // Left channel - extra pass
                let notch_freq_l_extra = center_freq + freq_range * lfo_l_extra;
                if notch_freq_l_extra > 0.0 && notch_freq_l_extra < self.sample_rate * 0.49 {
                    let coeffs = notch_coeffs_f64(
                        notch_freq_l_extra as f64,
                        q as f64,
                        self.sample_rate as f64,
                    );
                    for stage in 0..casc {
                        biquad_block(&mut self.ola.block_l, &coeffs, &mut rt.l_extra[stage]);
                    }
                }

                // Right channel - extra pass
                let notch_freq_r_extra = center_freq + freq_range * lfo_r_extra;
                if notch_freq_r_extra > 0.0 && notch_freq_r_extra < self.sample_rate * 0.49 {
                    let coeffs = notch_coeffs_f64(
                        notch_freq_r_extra as f64,
                        q as f64,
                        self.sample_rate as f64,
                    );
                    for stage in 0..casc {
                        biquad_block(&mut self.ola.block_r, &coeffs, &mut rt.r_extra[stage]);
                    }
                }
            }
        }

        // RMS compensation: restore original loudness after notch filtering
        // This matches Python's behavior where it computes rms_in before filtering
        // and then scales output by (rms_in / rms_out) to restore loudness
        if rms_in > 1e-8 {
            let mut sum_sq_l: f64 = 0.0;
            let mut sum_sq_r: f64 = 0.0;
            for i in 0..BLOCK_SIZE {
                sum_sq_l += self.ola.block_l[i] * self.ola.block_l[i];
                sum_sq_r += self.ola.block_r[i] * self.ola.block_r[i];
            }
            let rms_l = (sum_sq_l / BLOCK_SIZE as f64).sqrt();
            let rms_r = (sum_sq_r / BLOCK_SIZE as f64).sqrt();

            // Apply gain to restore original RMS level.
            // Clamp is critical: with deep/high-Q cascades, tiny rms_out values can
            // create enormous gains that produce spikes. Those spikes poison peak
            // calibration and make the stream end up extremely quiet.
            if rms_l > 1e-8 {
                let gain_l = (rms_in / rms_l).clamp(0.25, 16.0);
                for sample in self.ola.block_l.iter_mut() {
                    *sample *= gain_l;
                }
            }
            if rms_r > 1e-8 {
                let gain_r = (rms_in / rms_r).clamp(0.25, 16.0);
                for sample in self.ola.block_r.iter_mut() {
                    *sample *= gain_r;
                }
            }
        }

        // Overlap-add: accumulate filtered blocks and window into ring accumulators
        let write_base = self.ola.acc_write_pos;
        for i in 0..BLOCK_SIZE {
            let acc_idx = (write_base + i) % acc_size;
            self.ola.out_acc_l[acc_idx] += self.ola.block_l[i] as f32;
            self.ola.out_acc_r[acc_idx] += self.ola.block_r[i] as f32;
            self.ola.win_acc[acc_idx] += self.ola.window[i] as f32;
        }

        // Advance write position by hop size
        self.ola.acc_write_pos = (self.ola.acc_write_pos + HOP_SIZE) % acc_size;
        self.ola.samples_ready += HOP_SIZE;

        // Advance absolute block start for next block
        self.ola.absolute_block_start += HOP_SIZE;
    }

    /// Generate stereo output using Python-compatible overlap-add processing
    pub fn generate(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let mut frames_written = 0;
        let acc_size = self.ola.out_acc_l.len();

        while frames_written < frames {
            // If we have ready samples, emit them
            if self.ola.samples_ready > 0 {
                let read_pos = self.ola.acc_read_pos;

                // Normalize by window accumulator and output (Python: out_acc / win_acc where win_acc > 1e-8)
                let win_val = self.ola.win_acc[read_pos];
                let l = if win_val > 1e-8 {
                    self.ola.out_acc_l[read_pos] / win_val
                } else {
                    0.0
                };
                let r = if win_val > 1e-8 {
                    self.ola.out_acc_r[read_pos] / win_val
                } else {
                    0.0
                };

                out[frames_written * 2] = l;
                out[frames_written * 2 + 1] = r;

                // Clear the emitted accumulator slots for reuse (ring buffer)
                self.ola.out_acc_l[read_pos] = 0.0;
                self.ola.out_acc_r[read_pos] = 0.0;
                self.ola.win_acc[read_pos] = 0.0;

                // Advance read position
                self.ola.acc_read_pos = (read_pos + 1) % acc_size;
                self.ola.samples_ready -= 1;
                self.total_samples_output += 1;
                frames_written += 1;
            } else {
                // Need to fill input buffer and process a block
                // Fill the input ring buffer with base noise samples until we have BLOCK_SIZE
                while self.ola.input_samples_buffered < BLOCK_SIZE {
                    let sample = self.next_base();
                    self.ola.input_ring[self.ola.input_write_pos] = sample;
                    self.ola.input_write_pos = (self.ola.input_write_pos + 1) % BLOCK_SIZE;
                    self.ola.input_samples_buffered += 1;
                }

                // Process the block
                self.process_ola_block();

                // After processing, we consumed HOP_SIZE samples worth from input perspective
                // The ring buffer still holds BLOCK_SIZE samples, but logically we've advanced by HOP_SIZE
                // We need to refill HOP_SIZE samples for the next block (50% overlap)
                self.ola.input_samples_buffered = BLOCK_SIZE - HOP_SIZE;
            }
        }
    }
}

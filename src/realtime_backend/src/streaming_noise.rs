use crate::noise_params::NoiseParams;
use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type, Q_BUTTERWORTH_F32};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

// --- Helper Functions ---

fn triangle_wave(phase: f32) -> f32 {
    let t = (phase / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
    2.0 * (2.0 * (t - (t + 0.5).floor())).abs() - 1.0
}

fn lfo_value(phase: f32, waveform: &str) -> f32 {
    if waveform.eq_ignore_ascii_case("triangle") {
        triangle_wave(phase)
    } else {
        crate::dsp::trig::cos_lut(phase)
    }
}

// --- Notch Filter Logic (Legacy & Sweeps) ---

#[derive(Clone)]
struct Coeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

fn notch_coeffs(freq: f32, q: f32, sample_rate: f32) -> Coeffs {
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let cos_w0 = crate::dsp::trig::cos_lut(w0);
    let alpha = crate::dsp::trig::sin_lut(w0) / (2.0 * q);
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

#[derive(Clone, Copy)]
struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadState {
    fn new() -> Self {
        Self { z1: 0.0, z2: 0.0 }
    }
    fn process(&mut self, input: f32, c: &Coeffs) -> f32 {
        let out = input * c.b0 + self.z1;
        self.z1 = input * c.b1 - out * c.a1 + self.z2;
        self.z2 = input * c.b2 - out * c.a2;
        out
    }
}

// --- FFT Based Noise Generator (Matches Python's ColoredNoiseGenerator) ---

struct FftNoiseGenerator {
    buffer: Vec<f32>,
    cursor: usize,
    size: usize,
    exponent: f32,
    high_exponent: f32,
    distribution_curve: f32,
    lowcut: Option<f32>,
    highcut: Option<f32>,
    sample_rate: f32,
    // Optional Cut Filters (4th order = 2 cascaded biquads)
    lp_filters: Option<Vec<DirectForm2Transposed<f32>>>,
    hp_filters: Option<Vec<DirectForm2Transposed<f32>>>,
    base_amplitude: f32,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    rng: StdRng,
    normal: Normal<f32>,
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
        let nt = params.noise_type.to_lowercase();
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

        let rng = StdRng::from_entropy();
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
        };

        gen.regenerate_buffer();
        gen
    }

    fn regenerate_buffer(&mut self) {
        let mut white: Vec<Complex<f32>> = (0..self.size)
            .map(|_| Complex::new(self.normal.sample(&mut self.rng), 0.0))
            .collect();

        self.fft_forward.process(&mut white);

        let nyquist = self.sample_rate / 2.0;
        let min_f = (self.sample_rate) / (self.size as f32);

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

        self.buffer = output;
        self.cursor = 0;
    }
    fn next(&mut self) -> f32 {
        if self.cursor >= self.buffer.len() {
            self.regenerate_buffer();
        }
        let mut sample = self.buffer[self.cursor];
        self.cursor = (self.cursor + 1) % self.buffer.len();

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

        sample * self.base_amplitude
    }
}

pub struct StreamingNoise {
    sample_rate: f32,
    lfo_freq: f32,
    sweeps: Vec<(f32, f32)>,
    qs: Vec<f32>,
    cascades: Vec<usize>,
    lfo_phase: f32,
    lfo_phase_offset: f32,
    intra_offset: f32,
    lfo_waveform: String,
    noise_type: String,

    // Legacy Generator States (Pink/Brown)
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    brown: f32,
    brown_decay: f32,
    brown_scale: f32,

    // FFT Generator for everything else
    fft_gen: Option<FftNoiseGenerator>,

    rng: StdRng,
    normal: Normal<f32>,

    // Notch Filter States
    states_main_l: Vec<Vec<BiquadState>>,
    states_extra_l: Vec<Vec<BiquadState>>,
    states_main_r: Vec<Vec<BiquadState>>,
    states_extra_r: Vec<Vec<BiquadState>>,
}

impl StreamingNoise {
    pub fn new(params: &NoiseParams, sample_rate: u32) -> Self {
        let lfo_freq = if params.transition {
            params.start_lfo_freq
        } else if params.lfo_freq != 0.0 {
            params.lfo_freq
        } else {
            1.0 / 12.0
        };

        let sweeps: Vec<(f32, f32)> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| {
                    let min = if sw.start_min > 0.0 {
                        sw.start_min
                    } else {
                        1000.0
                    };
                    let max = if sw.start_max > 0.0 {
                        sw.start_max.max(min + 1.0)
                    } else {
                        (min + 1.0).max(min)
                    };
                    (min, max)
                })
                .collect()
        } else {
            vec![(1000.0, 10000.0)]
        };

        let qs: Vec<f32> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| if sw.start_q > 0.0 { sw.start_q } else { 25.0 })
                .collect()
        } else {
            vec![25.0; sweeps.len()]
        };

        let casc: Vec<usize> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| if sw.start_casc > 0 { sw.start_casc } else { 10 })
                .collect()
        } else {
            vec![10usize; sweeps.len()]
        };

        // Determine Mode
        let nt = params.noise_type.to_lowercase();
        // Legacy if strictly Pink/Brown/Red/White AND NO custom params
        // Note: Python uses Pink via IIR in generate_pink_noise_samples, so we replicate that.
        let is_legacy_type = matches!(nt.as_str(), "pink" | "brown" | "red" | "white");
        let has_custom_params = params.exponent.is_some()
            || params.high_exponent.is_some()
            || params.distribution_curve.is_some()
            || params.lowcut.is_some()
            || params.highcut.is_some()
            || params.amplitude.is_some();

        let use_fft = !is_legacy_type || has_custom_params;
        // eprintln!("StreamingNoise::new: nt={}, custom={}, use_fft={}", nt, has_custom_params, use_fft);

        let fft_gen = if use_fft {
            Some(FftNoiseGenerator::new(params, sample_rate as f32))
        } else {
            None
        };

        let mk_states = |casc: &Vec<usize>| -> (Vec<Vec<BiquadState>>, Vec<Vec<BiquadState>>) {
            let main: Vec<Vec<BiquadState>> =
                casc.iter().map(|c| vec![BiquadState::new(); *c]).collect();
            let extra: Vec<Vec<BiquadState>> =
                casc.iter().map(|c| vec![BiquadState::new(); *c]).collect();
            (main, extra)
        };
        let (states_main_l, states_extra_l) = mk_states(&casc);
        let (states_main_r, states_extra_r) = mk_states(&casc);

        Self {
            sample_rate: sample_rate as f32,
            lfo_freq,
            sweeps,
            qs,
            cascades: casc,
            lfo_phase: 0.0,
            lfo_phase_offset: params.start_lfo_phase_offset_deg.to_radians(),
            intra_offset: params.start_intra_phase_offset_deg.to_radians(),
            lfo_waveform: params.lfo_waveform.clone(),
            noise_type: params.noise_type.clone(),
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            b5: 0.0,
            brown: 0.0,
            brown_decay: 0.9999,
            brown_scale: 0.02,
            fft_gen,
            rng: StdRng::from_entropy(),
            normal: Normal::new(0.0, 1.0).unwrap(),
            states_main_l,
            states_extra_l,
            states_main_r,
            states_extra_r,
        }
    }

    pub fn skip_samples(&mut self, n: usize) {
        let mut scratch = vec![0.0f32; n * 2];
        self.generate(&mut scratch);
    }

    fn apply_pass(
        mut sample: f32,
        states: &mut [Vec<BiquadState>],
        sweeps: &[(f32, f32)],
        qs: &[f32],
        sample_rate: f32,
        lfo_waveform: &str,
        phase: f32,
    ) -> f32 {
        let lfo = lfo_value(phase, lfo_waveform);
        for (i, sweep) in sweeps.iter().enumerate() {
            let center = (sweep.0 + sweep.1) * 0.5;
            let range = (sweep.1 - sweep.0) * 0.5;
            let freq = center + range * lfo;
            if freq >= sample_rate * 0.49 {
                continue;
            }
            let coeffs = notch_coeffs(freq, qs[i], sample_rate);
            for state in &mut states[i] {
                sample = state.process(sample, &coeffs);
            }
        }
        sample
    }

    fn next_base(&mut self) -> f32 {
        if let Some(ref mut gen) = self.fft_gen {
            return gen.next();
        }

        // Legacy IIR Fallback
        let gaussian: f32 = self.normal.sample(&mut self.rng);
        match self.noise_type.to_lowercase().as_str() {
            "brown" | "red" => {
                self.brown = self.brown_decay * self.brown + gaussian * self.brown_scale;
                self.brown
            }
            "white" => gaussian,
            _ => {
                // Pink
                let w = gaussian;
                self.b0 = 0.99886 * self.b0 + w * 0.0555179;
                self.b1 = 0.99332 * self.b1 + w * 0.0750759;
                self.b2 = 0.96900 * self.b2 + w * 0.1538520;
                self.b3 = 0.86650 * self.b3 + w * 0.3104856;
                self.b4 = 0.55000 * self.b4 + w * 0.5329522;
                self.b5 = -0.7616 * self.b5 - w * 0.0168980;
                (self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5) * 0.11
            }
        }
    }

    pub fn generate(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let sweeps = self.sweeps.clone();
        let qs = self.qs.clone();
        let sample_rate = self.sample_rate;
        let lfo_waveform = self.lfo_waveform.clone();
        let lfo_phase_offset = self.lfo_phase_offset;
        let intra_offset = self.intra_offset;

        for i in 0..frames {
            let base = self.next_base();
            let l_phase = self.lfo_phase;
            let r_phase = self.lfo_phase + lfo_phase_offset;
            let mut l = Self::apply_pass(
                base,
                &mut self.states_main_l,
                &sweeps,
                &qs,
                sample_rate,
                &lfo_waveform,
                l_phase,
            );
            if intra_offset != 0.0 {
                l = Self::apply_pass(
                    l,
                    &mut self.states_extra_l,
                    &sweeps,
                    &qs,
                    sample_rate,
                    &lfo_waveform,
                    l_phase + intra_offset,
                );
            }
            let mut r = Self::apply_pass(
                base,
                &mut self.states_main_r,
                &sweeps,
                &qs,
                sample_rate,
                &lfo_waveform,
                r_phase,
            );
            if intra_offset != 0.0 {
                r = Self::apply_pass(
                    r,
                    &mut self.states_extra_r,
                    &sweeps,
                    &qs,
                    sample_rate,
                    &lfo_waveform,
                    r_phase + intra_offset,
                );
            }
            out[i * 2] = l;
            out[i * 2 + 1] = r;
            self.lfo_phase += 2.0 * std::f32::consts::PI * self.lfo_freq / sample_rate;
        }
    }
}

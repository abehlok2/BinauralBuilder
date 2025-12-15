use crate::config::CONFIG;
use crate::gpu::GpuMixer;
use crate::models::{BackgroundNoiseData, StepData, TrackData, MAX_INDIVIDUAL_GAIN};
use crate::noise_params::NoiseParams;
use crate::streaming_noise::StreamingNoise;
use crate::voices::{voices_for_step, VoiceKind};
use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
pub trait Voice: Send + Sync {
    fn process(&mut self, output: &mut [f32]);
    fn is_finished(&self) -> bool;
}

#[derive(Clone, Copy)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
}

impl CrossfadeCurve {
    fn gains(self, ratio: f32) -> (f32, f32) {
        match self {
            CrossfadeCurve::Linear => (1.0 - ratio, ratio),
            CrossfadeCurve::EqualPower => {
                let theta = ratio * std::f32::consts::FRAC_PI_2;
                (
                    crate::dsp::trig::cos_lut(theta),
                    crate::dsp::trig::sin_lut(theta),
                )
            }
        }
    }
}

fn steps_have_continuous_voices(a: &StepData, b: &StepData) -> bool {
    if a.voices.len() != b.voices.len() {
        return false;
    }

    for (va, vb) in a.voices.iter().zip(&b.voices) {
        if va.synth_function_name != vb.synth_function_name {
            return false;
        }
        if va.params != vb.params {
            return false;
        }
        if va.is_transition != vb.is_transition {
            return false;
        }
        if va.voice_type.to_lowercase() != vb.voice_type.to_lowercase() {
            return false;
        }
    }

    true
}

pub struct TrackScheduler {
    pub track: TrackData,
    pub current_sample: usize,
    pub current_step: usize,
    pub active_voices: Vec<StepVoice>,
    pub next_voices: Vec<StepVoice>,
    pub sample_rate: f32,
    pub crossfade_samples: usize,
    pub current_crossfade_samples: usize,
    pub crossfade_curve: CrossfadeCurve,
    pub crossfade_envelope: Vec<f32>,
    crossfade_prev: Vec<f32>,
    crossfade_next: Vec<f32>,
    pub next_step_sample: usize,
    pub crossfade_active: bool,
    pub absolute_sample: u64,
    /// Whether playback is paused
    pub paused: bool,
    pub clips: Vec<LoadedClip>,
    pub background_noise: Option<BackgroundNoise>,
    pub scratch: Vec<f32>,
    /// Whether GPU accelerated mixing should be used when available
    pub gpu_enabled: bool,
    pub voice_gain: f32,
    pub noise_gain: f32,
    pub clip_gain: f32,
    pub master_gain: f32,
    #[cfg(feature = "gpu")]
    pub gpu: GpuMixer,
    /// Temporary buffer for mixing per-voice output
    voice_temp: Vec<f32>,
    /// Temporary buffer for accumulating noise voices separately
    noise_scratch: Vec<f32>,
    /// Accumulated phases (phase_l, phase_r) carried over from previous voices.
    /// Used to maintain phase continuity and prevent clicking when transitioning between steps.
    accumulated_phases: Vec<(f32, f32)>,
}

pub enum ClipSamples {
    Static(Vec<f32>),
    Streaming { data: Vec<f32>, finished: bool },
}

pub struct LoadedClip {
    samples: ClipSamples,
    start_sample: usize,
    position: usize,
    gain: f32,
}

pub struct BackgroundNoise {
    generator: StreamingNoise,
    gain: f32,
    start_sample: usize,
    fade_in_samples: usize,
    fade_out_samples: usize,
    amp_envelope: Vec<(usize, f32)>,
    duration_samples: Option<usize>,
    started: bool,
    playback_sample: usize,
}

impl BackgroundNoise {
    fn from_params(mut params: NoiseParams, base_gain: f32, device_rate: u32) -> Self {
        params.sample_rate = device_rate;
        let start_sample = (params.start_time.max(0.0) * device_rate as f32) as usize;
        let fade_in_samples = (params.fade_in.max(0.0) * device_rate as f32) as usize;
        let fade_out_samples = (params.fade_out.max(0.0) * device_rate as f32) as usize;
        let duration_samples = if params.duration_seconds > 0.0 {
            Some((params.duration_seconds * device_rate as f32) as usize)
        } else {
            None
        };

        let env_points: Vec<(usize, f32)> = params
            .amp_envelope
            .iter()
            .map(|pair| {
                let t = pair.get(0).copied().unwrap_or(0.0).max(0.0);
                let a = pair.get(1).copied().unwrap_or(1.0);
                (((t * device_rate as f32) as usize), a)
            })
            .collect();

        let generator = StreamingNoise::new(&params, device_rate);

        Self {
            generator,
            gain: base_gain,
            start_sample,
            fade_in_samples,
            fade_out_samples,
            amp_envelope: env_points,
            duration_samples,
            started: false,
            playback_sample: 0,
        }
    }

    fn envelope_at(&self, local_sample: usize) -> f32 {
        let mut amp = 1.0f32;

        if self.fade_in_samples > 0 && local_sample < self.fade_in_samples {
            amp *= (local_sample as f32 / self.fade_in_samples as f32).clamp(0.0, 1.0);
        }

        if let Some(dur) = self.duration_samples {
            if self.fade_out_samples > 0
                && local_sample >= dur.saturating_sub(self.fade_out_samples)
            {
                let pos = local_sample.saturating_sub(dur.saturating_sub(self.fade_out_samples));
                let denom = self.fade_out_samples.max(1) as f32;
                amp *= (1.0 - pos as f32 / denom).clamp(0.0, 1.0);
            }
        }

        if !self.amp_envelope.is_empty() {
            let mut prev = self.amp_envelope[0];
            for &(t, a) in &self.amp_envelope {
                if local_sample < t {
                    let span = (t.saturating_sub(prev.0)).max(1);
                    let frac = (local_sample.saturating_sub(prev.0)) as f32 / span as f32;
                    let interp = prev.1 + (a - prev.1) * frac;
                    return amp * interp;
                }
                prev = (t, a);
            }
            amp *= prev.1;
        }

        amp
    }

    fn mix_into(&mut self, buffer: &mut [f32], scratch: &mut Vec<f32>, global_start_sample: usize) {
        let frames = buffer.len() / 2;
        if frames == 0 {
            return;
        }

        if let Some(limit) = self.duration_samples {
            if self.playback_sample >= limit {
                return;
            }
        }

        let start_offset = if !self.started && global_start_sample < self.start_sample {
            self.start_sample.saturating_sub(global_start_sample)
        } else {
            0
        };

        if start_offset >= frames {
            return;
        }

        let mut usable_frames = frames - start_offset;
        if let Some(limit) = self.duration_samples {
            usable_frames = usable_frames.min(limit.saturating_sub(self.playback_sample));
        }
        if usable_frames == 0 {
            return;
        }

        let mix_frames = start_offset + usable_frames;
        let required_samples = mix_frames * 2;
        if scratch.len() < required_samples {
            scratch.resize(required_samples, 0.0);
        }
        scratch[..start_offset * 2].fill(0.0);
        self.generator
            .generate(&mut scratch[start_offset * 2..required_samples]);

        for i in 0..usable_frames {
            let env = self.envelope_at(self.playback_sample + i) * self.gain;
            let idx = (start_offset + i) * 2;
            buffer[idx] += scratch[idx] * env;
            buffer[idx + 1] += scratch[idx + 1] * env;
        }

        self.playback_sample += usable_frames;
        self.started = true;
    }

    /// Update just the gain value without recreating the generator.
    /// This preserves the noise generator state and avoids phase resets.
    fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }
}

/// Check if two noise configurations are compatible (same params, only gain differs).
/// When compatible, we can reuse the existing noise generator and just update the gain.
fn noise_configs_compatible(
    old: &Option<BackgroundNoiseData>,
    new: &Option<BackgroundNoiseData>,
) -> bool {
    match (old, new) {
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
        (Some(old_data), Some(new_data)) => {
            // Compare file paths - must be identical
            if old_data.file_path != new_data.file_path {
                return false;
            }
            // Compare params - must be identical for the generator to be reusable
            // We compare the JSON serialization to handle nested structures
            match (&old_data.params, &new_data.params) {
                (None, None) => true,
                (Some(_), None) | (None, Some(_)) => false,
                (Some(old_params), Some(new_params)) => {
                    // Compare all fields except sample_rate (set by device)
                    // by serializing and comparing as JSON
                    let old_json = serde_json::to_string(old_params);
                    let new_json = serde_json::to_string(new_params);
                    match (old_json, new_json) {
                        (Ok(o), Ok(n)) => o == n,
                        _ => false,
                    }
                }
            }
        }
    }
}

/// Check if only volume-related parameters changed between two track configurations.
/// Volume-related parameters (binaural_volume, noise_volume, normalization_level)
/// can be updated without rebuilding voices or seeking, preserving phase continuity.
fn is_volume_only_change(old: &TrackData, new: &TrackData) -> bool {
    // Must have same number of steps
    if old.steps.len() != new.steps.len() {
        return false;
    }

    // Must have same number of clips
    if old.clips.len() != new.clips.len() {
        return false;
    }

    // Global settings must match except for normalization_level
    if old.global_settings.sample_rate != new.global_settings.sample_rate
        || old.global_settings.crossfade_duration != new.global_settings.crossfade_duration
        || old.global_settings.crossfade_curve != new.global_settings.crossfade_curve
        || old.global_settings.output_filename != new.global_settings.output_filename
    {
        return false;
    }

    // Compare each step - everything must match except volume-related fields
    for (old_step, new_step) in old.steps.iter().zip(new.steps.iter()) {
        // Duration must match
        if (old_step.duration - new_step.duration).abs() > 1e-9 {
            return false;
        }

        // Voices must be identical
        if old_step.voices.len() != new_step.voices.len() {
            return false;
        }
        for (old_voice, new_voice) in old_step.voices.iter().zip(new_step.voices.iter()) {
            if old_voice.synth_function_name != new_voice.synth_function_name
                || old_voice.params != new_voice.params
                || old_voice.is_transition != new_voice.is_transition
                || old_voice.voice_type != new_voice.voice_type
            {
                return false;
            }
        }
    }

    // Clips must match exactly (including gain for now - could be relaxed later)
    for (old_clip, new_clip) in old.clips.iter().zip(new.clips.iter()) {
        if old_clip.file_path != new_clip.file_path
            || (old_clip.start - new_clip.start).abs() > 1e-9
            || (old_clip.amp - new_clip.amp).abs() > 1e-9
        {
            return false;
        }
    }

    // Noise config must be compatible (same params, only gain may differ)
    if !noise_configs_compatible(&old.background_noise, &new.background_noise) {
        return false;
    }

    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VoiceType {
    Binaural,
    Noise,
    Other,
}

pub struct StepVoice {
    pub kind: VoiceKind,
    pub voice_type: VoiceType,
    pub normalization_peak: f32,
}

impl StepVoice {
    fn process(&mut self, output: &mut [f32]) {
        self.kind.process(output);
    }

    fn is_finished(&self) -> bool {
        self.kind.is_finished()
    }
}

use crate::command::Command;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use std::io::Cursor;

fn decode_clip_reader<R: MediaSource + 'static>(
    reader: R,
    sample_rate: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mss = MediaSourceStream::new(Box::new(reader), Default::default());
    let probed = get_probe().format(
        &Hint::new(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no default track")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let src_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("unknown channel count")?
        .count();

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut samples: Vec<f32> = Vec::new();
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
            let l = frame[0];
            let r = if channels > 1 { frame[1] } else { frame[0] };
            samples.push(l);
            samples.push(r);
        }
    }
    if src_rate != sample_rate {
        samples = resample_linear_stereo(&samples, src_rate, sample_rate);
    }
    Ok(samples)
}

fn load_clip_bytes(data: &[u8], sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let cursor = Cursor::new(data.to_vec());
    decode_clip_reader(cursor, sample_rate)
}
fn load_clip_file(path: &str, sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if path.starts_with("data:") {
        if let Some(idx) = path.find(',') {
            let (_, b64) = path.split_at(idx + 1);
            let bytes = BASE64.decode(b64.trim())?;
            return load_clip_bytes(&bytes, sample_rate);
        } else {
            return Err("invalid data url".into());
        }
    }

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe().format(
        &Hint::new(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no default track")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let src_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("unknown channel count")?
        .count();

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut samples: Vec<f32> = Vec::new();
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
            let l = frame[0];
            let r = if channels > 1 { frame[1] } else { frame[0] };
            samples.push(l);
            samples.push(r);
        }
    }
    if src_rate != sample_rate {
        samples = resample_linear_stereo(&samples, src_rate, sample_rate);
    }
    Ok(samples)
}

fn resample_linear_stereo(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let frames = input.len() / 2;
    let duration = frames as f64 / src_rate as f64;
    let out_frames = (duration * dst_rate as f64).round() as usize;
    let mut out = vec![0.0f32; out_frames * 2];
    for i in 0..out_frames {
        let t = i as f64 / dst_rate as f64;
        let pos = t * src_rate as f64;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let idx2 = if idx + 1 < frames { idx + 1 } else { idx };
        for ch in 0..2 {
            let x0 = input[idx * 2 + ch];
            let x1 = input[idx2 * 2 + ch];
            out[i * 2 + ch] = ((1.0 - frac) * x0 as f64 + frac * x1 as f64) as f32;
        }
    }
    out
}

impl TrackScheduler {
    pub fn new(track: TrackData, device_rate: u32) -> Self {
        Self::new_with_start(track, device_rate, 0.0)
    }

    pub fn new_with_start(track: TrackData, device_rate: u32, start_time: f64) -> Self {
        let sample_rate = device_rate as f32;
        let crossfade_samples =
            (track.global_settings.crossfade_duration * sample_rate as f64) as usize;
        let crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };
        let mut clips = Vec::new();
        let cfg = &CONFIG;
        for c in &track.clips {
            let clip_samples = match load_clip_file(&c.file_path, device_rate) {
                Ok(samples) => ClipSamples::Static(samples),
                Err(_) => ClipSamples::Streaming {
                    data: Vec::new(),
                    finished: false,
                },
            };
            clips.push(LoadedClip {
                samples: clip_samples,
                start_sample: (c.start * sample_rate as f64) as usize,
                position: 0,
                gain: c.amp * cfg.clip_gain,
            });
        }

        let background_noise = if let Some(noise_cfg) = &track.background_noise {
            if !noise_cfg.file_path.is_empty() && noise_cfg.file_path.ends_with(".noise") {
                if let Ok(params) = crate::noise_params::load_noise_params(&noise_cfg.file_path) {
                    Some(BackgroundNoise::from_params(
                        params,
                        noise_cfg.amp * cfg.noise_gain,
                        device_rate,
                    ))
                } else {
                    None
                }
            } else if let Some(params) = &noise_cfg.params {
                Some(BackgroundNoise::from_params(
                    params.clone(),
                    noise_cfg.amp * cfg.noise_gain,
                    device_rate,
                ))
            } else {
                None
            }
        } else {
            None
        };

        let mut sched = Self {
            track,
            current_sample: 0,
            current_step: 0,
            active_voices: Vec::new(),
            next_voices: Vec::new(),
            sample_rate,
            crossfade_samples,
            current_crossfade_samples: 0,
            crossfade_curve,
            crossfade_envelope: Vec::new(),
            crossfade_prev: Vec::new(),
            crossfade_next: Vec::new(),
            next_step_sample: 0,
            crossfade_active: false,
            absolute_sample: 0,
            paused: false,
            clips,
            background_noise,
            scratch: Vec::new(),
            gpu_enabled: cfg.gpu,
            voice_gain: cfg.voice_gain,
            noise_gain: cfg.noise_gain,
            clip_gain: cfg.clip_gain,
            master_gain: 1.0,
            #[cfg(feature = "gpu")]
            gpu: GpuMixer::new(),
            voice_temp: Vec::new(),
            noise_scratch: Vec::new(),
            accumulated_phases: Vec::new(),
        };

        let start_samples = (start_time * sample_rate as f64) as usize;
        sched.seek_samples(start_samples);
        sched
    }

    fn seek_samples(&mut self, abs_samples: usize) {
        self.absolute_sample = abs_samples as u64;

        for clip in &mut self.clips {
            clip.position = if abs_samples > clip.start_sample {
                (abs_samples - clip.start_sample) * 2
            } else {
                0
            };
        }

        let mut remaining = abs_samples;
        self.current_step = 0;
        self.current_sample = 0;
        for (idx, step) in self.track.steps.iter().enumerate() {
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if remaining < step_samples {
                self.current_step = idx;
                self.current_sample = remaining;
                break;
            }
            remaining = remaining.saturating_sub(step_samples);
        }

        self.active_voices.clear();
        self.next_voices.clear();
        self.crossfade_active = false;
        self.current_crossfade_samples = 0;
        self.next_step_sample = 0;
        self.crossfade_prev.clear();
        self.crossfade_next.clear();
        self.accumulated_phases.clear();
        if let Some(noise) = &mut self.background_noise {
            noise.playback_sample = 0;
            noise.started = false;
            if abs_samples > noise.start_sample {
                let local = abs_samples - noise.start_sample;
                let skip = if let Some(limit) = noise.duration_samples {
                    local.min(limit)
                } else {
                    local
                };
                noise.generator.skip_samples(skip);
                noise.playback_sample = skip;
                noise.started = true;
            }
        }
    }

    /// Replace the current track data while preserving playback progress.
    pub fn update_track(&mut self, track: TrackData) {
        // Fast path: if only volume-related parameters changed, we can update
        // the track data in place without rebuilding voices or seeking.
        // This preserves perfect phase continuity for binaural_volume, noise_volume,
        // and normalization_level changes.
        if is_volume_only_change(&self.track, &track) {
            // Just update the track data - volumes are applied at render time
            // in render_step_audio via apply_gain_stage, so existing voices
            // will automatically use the new volume values.
            self.track = track.clone();

            // Update noise gain if noise is active (noise config is compatible)
            if let (Some(ref mut noise), Some(noise_cfg)) =
                (&mut self.background_noise, &track.background_noise)
            {
                noise.set_gain(noise_cfg.amp * self.noise_gain);
            }
            return;
        }

        // Full update path: structural changes require rebuilding voices
        let abs_samples = self.absolute_sample as usize;

        // Preserve the currently accumulated phases so that live updates do not
        // introduce discontinuities when the track data is replaced. Without
        // this, `seek_samples` would clear the cached phases and newly
        // constructed voices would restart at phase 0, producing an audible
        // reset whenever parameters change mid-stream.
        let preserved_phases = if !self.active_voices.is_empty() {
            Self::extract_phases_from_voices(&self.active_voices)
        } else if !self.next_voices.is_empty() {
            Self::extract_phases_from_voices(&self.next_voices)
        } else {
            self.accumulated_phases.clone()
        };

        self.crossfade_samples =
            (track.global_settings.crossfade_duration * self.sample_rate as f64) as usize;
        self.crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };

        // Check if we can reuse the existing noise generator (only gain changed).
        // This preserves the noise phase/LFO state and prevents audible resets.
        // Must be done BEFORE updating self.track to compare old vs new config.
        let old_noise_cfg = self.track.background_noise.clone();
        let new_noise_cfg = &track.background_noise;
        let can_reuse_noise =
            self.background_noise.is_some() && noise_configs_compatible(&old_noise_cfg, new_noise_cfg);

        self.track = track.clone();

        self.clips.clear();
        for c in &track.clips {
            let clip_samples = match load_clip_file(&c.file_path, self.sample_rate as u32) {
                Ok(samples) => ClipSamples::Static(samples),
                Err(_) => ClipSamples::Streaming {
                    data: Vec::new(),
                    finished: false,
                },
            };
            self.clips.push(LoadedClip {
                samples: clip_samples,
                start_sample: (c.start * self.sample_rate as f64) as usize,
                position: 0,
                gain: c.amp * self.clip_gain,
            });
        }

        if can_reuse_noise {
            // Only update the gain, preserving the generator state
            if let (Some(ref mut noise), Some(noise_cfg)) =
                (&mut self.background_noise, &track.background_noise)
            {
                noise.set_gain(noise_cfg.amp * self.noise_gain);
            }
        } else {
            // Recreate the noise generator (params changed or no existing generator)
            self.background_noise = if let Some(noise_cfg) = &track.background_noise {
                if !noise_cfg.file_path.is_empty() && noise_cfg.file_path.ends_with(".noise") {
                    if let Ok(params) = crate::noise_params::load_noise_params(&noise_cfg.file_path)
                    {
                        Some(BackgroundNoise::from_params(
                            params,
                            noise_cfg.amp * self.noise_gain,
                            self.sample_rate as u32,
                        ))
                    } else {
                        None
                    }
                } else if let Some(params) = &noise_cfg.params {
                    Some(BackgroundNoise::from_params(
                        params.clone(),
                        noise_cfg.amp * self.noise_gain,
                        self.sample_rate as u32,
                    ))
                } else {
                    None
                }
            } else {
                None
            };
        }

        self.seek_samples(abs_samples);

        // Restore the captured phases so the next render reuses the current
        // oscillator states and remains continuous across the update.
        self.accumulated_phases = preserved_phases;
        self.crossfade_prev.clear();
        self.crossfade_next.clear();
        #[cfg(feature = "gpu")]
        {
            self.gpu = GpuMixer::new();
        }
    }

    pub fn handle_command(&mut self, cmd: Command) {
        match cmd {
            Command::UpdateTrack(t) => self.update_track(t),
            Command::EnableGpu(enable) => {
                self.gpu_enabled = enable;
            }
            Command::SetPaused(p) => {
                if p {
                    self.pause();
                } else {
                    self.resume();
                }
            }
            Command::StartFrom(time) => {
                let samples = (time * self.sample_rate as f64) as usize;
                self.seek_samples(samples);
            }
            Command::SetMasterGain(gain) => {
                self.master_gain = gain.clamp(0.0, 1.0);
            }
            Command::PushClipSamples {
                index,
                data,
                finished,
            } => {
                if let Some(clip) = self.clips.get_mut(index) {
                    if let ClipSamples::Streaming {
                        data: buf,
                        finished: fin,
                    } = &mut clip.samples
                    {
                        buf.extend_from_slice(&data);
                        if finished {
                            *fin = true;
                        }
                    }
                }
            }
        }
    }

    fn apply_gain_stage(
        buffer: &mut [f32],
        norm_target: f32,
        volume: f32,
        has_content: bool,
        normalization_peak: f32,
    ) {
        if !has_content {
            buffer.fill(0.0);
            return;
        }

        let normalization_gain = if normalization_peak > 1e-9 && norm_target > 0.0 {
            (norm_target / normalization_peak).min(1.0)
        } else {
            1.0
        };

        // Clamp volume to MAX_INDIVIDUAL_GAIN to prevent clipping when sources combine
        let clamped_volume = volume.clamp(0.0, MAX_INDIVIDUAL_GAIN);
        let total_gain = normalization_gain * clamped_volume;

        if (total_gain - 1.0).abs() > f32::EPSILON {
            for s in buffer.iter_mut() {
                *s *= total_gain;
            }
        }
    }

    fn render_step_audio(&mut self, voices: &mut [StepVoice], step: &StepData, out: &mut [f32]) {
        let len = out.len();
        if self.scratch.len() != len {
            self.scratch.resize(len, 0.0);
        }
        if self.noise_scratch.len() != len {
            self.noise_scratch.resize(len, 0.0);
        }
        if self.voice_temp.len() != len {
            self.voice_temp.resize(len, 0.0);
        }

        let binaural_buf = &mut self.scratch;
        let noise_buf = &mut self.noise_scratch;
        binaural_buf.fill(0.0);
        noise_buf.fill(0.0);
        let mut binaural_count = 0usize;
        let mut noise_count = 0usize;
        let mut binaural_peak = 0.0f32;
        let mut noise_peak = 0.0f32;

        for voice in voices.iter_mut() {
            self.voice_temp.fill(0.0);
            voice.process(&mut self.voice_temp);
            match voice.voice_type {
                VoiceType::Noise => {
                    noise_count += 1;
                    noise_peak = noise_peak.max(voice.normalization_peak);
                    for i in 0..len {
                        noise_buf[i] += self.voice_temp[i];
                    }
                }
                _ => {
                    binaural_count += 1;
                    binaural_peak = binaural_peak.max(voice.normalization_peak);
                    for i in 0..len {
                        binaural_buf[i] += self.voice_temp[i];
                    }
                }
            }
        }

        let norm_target = step.normalization_level;
        Self::apply_gain_stage(
            binaural_buf,
            norm_target,
            step.binaural_volume,
            binaural_count > 0,
            binaural_peak,
        );
        Self::apply_gain_stage(
            noise_buf,
            norm_target,
            step.noise_volume,
            noise_count > 0,
            noise_peak,
        );

        out.fill(0.0);
        for i in 0..len {
            out[i] = binaural_buf[i] + noise_buf[i];
        }
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn current_step_index(&self) -> usize {
        self.current_step
    }

    pub fn elapsed_samples(&self) -> u64 {
        self.absolute_sample
    }

    /// Extracts accumulated phases from all voices that track phase.
    /// Returns a vector of (phase_l, phase_r) tuples for each voice that has phases.
    fn extract_phases_from_voices(voices: &[StepVoice]) -> Vec<(f32, f32)> {
        voices.iter().filter_map(|v| v.kind.get_phases()).collect()
    }

    /// Applies accumulated phases to newly created voices.
    /// This maintains phase continuity between voice instances to prevent clicking.
    fn apply_phases_to_voices(phases: &[(f32, f32)], voices: &mut [StepVoice]) {
        // Apply phases to voices that support them, matching by index
        let mut phase_iter = phases.iter();
        for voice in voices.iter_mut() {
            // Only apply to voices that have phase tracking
            if voice.kind.get_phases().is_some() {
                if let Some(&(phase_l, phase_r)) = phase_iter.next() {
                    voice.kind.set_phases(phase_l, phase_r);
                }
            }
        }
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        let frame_count = buffer.len() / 2;
        buffer.fill(0.0);

        if self.paused {
            return;
        }

        if self.current_step >= self.track.steps.len() {
            return;
        }

        if self.active_voices.is_empty() && !self.crossfade_active {
            let step = &self.track.steps[self.current_step];
            let mut new_voices = voices_for_step(step, self.sample_rate);
            // Apply accumulated phases from previous voices to maintain phase continuity
            Self::apply_phases_to_voices(&self.accumulated_phases, &mut new_voices);
            self.active_voices = new_voices;
        }

        // Check if we need to start crossfade into the next step
        if !self.crossfade_active
            && self.crossfade_samples > 0
            && self.current_step + 1 < self.track.steps.len()
        {
            let step = &self.track.steps[self.current_step];
            let next_step = &self.track.steps[self.current_step + 1];
            if !steps_have_continuous_voices(step, next_step) {
                let step_samples = (step.duration * self.sample_rate as f64) as usize;
                let fade_len = self.crossfade_samples.min(step_samples);
                if self.current_sample >= step_samples.saturating_sub(fade_len) {
                    // Extract phases from current voices before transitioning
                    self.accumulated_phases = Self::extract_phases_from_voices(&self.active_voices);
                    let mut new_next_voices = voices_for_step(next_step, self.sample_rate);
                    // Apply accumulated phases to the new voices for continuity
                    Self::apply_phases_to_voices(&self.accumulated_phases, &mut new_next_voices);
                    self.next_voices = new_next_voices;
                    self.crossfade_active = true;
                    self.next_step_sample = 0;
                    let next_samples = (next_step.duration * self.sample_rate as f64) as usize;
                    self.current_crossfade_samples =
                        self.crossfade_samples.min(step_samples).min(next_samples);
                    self.crossfade_envelope = if self.current_crossfade_samples <= 1 {
                        vec![0.0; self.current_crossfade_samples]
                    } else {
                        (0..self.current_crossfade_samples)
                            .map(|i| i as f32 / (self.current_crossfade_samples - 1) as f32)
                            .collect()
                    };
                }
            }
        }

        if self.crossfade_active {
            let len = buffer.len();
            let frames = len / 2;
            if self.crossfade_prev.len() != len {
                self.crossfade_prev.resize(len, 0.0);
            }
            if self.crossfade_next.len() != len {
                self.crossfade_next.resize(len, 0.0);
            }
            let mut prev_buf = std::mem::take(&mut self.crossfade_prev);
            let mut next_buf = std::mem::take(&mut self.crossfade_next);
            prev_buf.fill(0.0);
            next_buf.fill(0.0);

            let step = self.track.steps[self.current_step].clone();
            let mut voices = std::mem::take(&mut self.active_voices);
            self.render_step_audio(&mut voices, &step, &mut prev_buf);
            self.active_voices = voices;
            let next_step_idx = (self.current_step + 1).min(self.track.steps.len() - 1);
            let next_step = self.track.steps[next_step_idx].clone();
            let mut next_voices = std::mem::take(&mut self.next_voices);
            self.render_step_audio(&mut next_voices, &next_step, &mut next_buf);
            self.next_voices = next_voices;

            for i in 0..frames {
                let idx = i * 2;
                let progress = self.next_step_sample + i;
                if progress < self.current_crossfade_samples {
                    let ratio = if progress < self.crossfade_envelope.len() {
                        self.crossfade_envelope[progress]
                    } else {
                        progress as f32 / (self.current_crossfade_samples - 1) as f32
                    };
                    let (g_out, g_in) = self.crossfade_curve.gains(ratio);
                    buffer[idx] = prev_buf[idx] * g_out + next_buf[idx] * g_in;
                    buffer[idx + 1] = prev_buf[idx + 1] * g_out + next_buf[idx + 1] * g_in;
                } else {
                    buffer[idx] = next_buf[idx];
                    buffer[idx + 1] = next_buf[idx + 1];
                }
            }

            self.current_sample += frames;
            self.next_step_sample += frames;

            self.active_voices.retain(|v| !v.is_finished());
            self.next_voices.retain(|v| !v.is_finished());

            if self.next_step_sample >= self.current_crossfade_samples {
                // Update accumulated phases from the next_voices that are becoming active
                self.accumulated_phases = Self::extract_phases_from_voices(&self.next_voices);
                self.current_step += 1;
                self.current_sample = self.next_step_sample;
                self.next_step_sample = 0;
                self.active_voices = std::mem::take(&mut self.next_voices);
                self.crossfade_active = false;
                self.crossfade_envelope.clear();
                self.current_crossfade_samples = 0;
            }

            self.crossfade_prev = prev_buf;
            self.crossfade_next = next_buf;
        } else {
            if !self.active_voices.is_empty() {
                let step = self.track.steps[self.current_step].clone();
                let mut voices = std::mem::take(&mut self.active_voices);
                self.render_step_audio(&mut voices, &step, buffer);
                self.active_voices = voices;
            }

            self.active_voices.retain(|v| !v.is_finished());
            self.current_sample += frame_count;
            let step = &self.track.steps[self.current_step];
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if self.current_sample >= step_samples {
                // Extract phases from current voices before clearing to maintain phase continuity
                self.accumulated_phases = Self::extract_phases_from_voices(&self.active_voices);
                self.current_step += 1;
                self.current_sample = 0;
                self.active_voices.clear();
            }
        }

        for v in &mut buffer[..] {
            *v *= self.voice_gain;
        }

        let frames = frame_count;

        let start_sample = self.absolute_sample as usize;

        if let Some(noise) = &mut self.background_noise {
            if self.scratch.len() != buffer.len() {
                self.scratch.resize(buffer.len(), 0.0);
            }
            noise.mix_into(buffer, &mut self.scratch, start_sample);
        }

        for clip in &mut self.clips {
            if start_sample + frames < clip.start_sample {
                continue;
            }
            let mut pos = clip.position;
            if start_sample < clip.start_sample {
                let offset = clip.start_sample - start_sample;
                pos += offset * 2;
            }
            match &mut clip.samples {
                ClipSamples::Static(data) => {
                    for i in 0..frames {
                        let global_idx = start_sample + i;
                        if global_idx < clip.start_sample {
                            continue;
                        }
                        if pos + 1 >= data.len() {
                            break;
                        }
                        buffer[i * 2] += data[pos] * clip.gain;
                        buffer[i * 2 + 1] += data[pos + 1] * clip.gain;
                        pos += 2;
                    }
                }
                ClipSamples::Streaming { data, finished } => {
                    for i in 0..frames {
                        let global_idx = start_sample + i;
                        if global_idx < clip.start_sample {
                            continue;
                        }
                        if pos + 1 >= data.len() {
                            break;
                        }
                        buffer[i * 2] += data[pos] * clip.gain;
                        buffer[i * 2 + 1] += data[pos + 1] * clip.gain;
                        pos += 2;
                    }
                    if *finished && pos >= data.len() {
                        pos = data.len();
                    }
                    // Remove consumed samples to free memory
                    if pos > 4096 {
                        data.drain(0..pos);
                        clip.start_sample += pos / 2;
                        pos = 0;
                    }
                }
            }
            clip.position = pos;
        }

        if (self.master_gain - 1.0).abs() > f32::EPSILON {
            for v in buffer.iter_mut() {
                *v *= self.master_gain;
            }
        }

        self.absolute_sample += frame_count as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::CrossfadeCurve;

    #[test]
    fn test_fade_curves_match_python() {
        let samples = 5;
        for curve in [CrossfadeCurve::Linear, CrossfadeCurve::EqualPower] {
            for i in 0..samples {
                let ratio = i as f32 / (samples - 1) as f32;
                let (g_out, g_in) = curve.gains(ratio);
                let (exp_out, exp_in) = match curve {
                    CrossfadeCurve::Linear => (1.0 - ratio, ratio),
                    CrossfadeCurve::EqualPower => {
                        let theta = ratio * std::f32::consts::FRAC_PI_2;
                        (
                            crate::dsp::trig::cos_lut(theta),
                            crate::dsp::trig::sin_lut(theta),
                        )
                    }
                };
                assert!((g_out - exp_out).abs() < 1e-6);
                assert!((g_in - exp_in).abs() < 1e-6);
            }
        }
    }
}

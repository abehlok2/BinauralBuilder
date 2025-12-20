use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

fn default_zero_f64() -> f64 {
    0.0
}

fn default_amp() -> f32 {
    1.0
}

fn default_crossfade_duration() -> f64 {
    3.0
}
fn default_crossfade_curve() -> String {
    "linear".to_string()
}

/// Maximum individual gain for binaural/noise to prevent clipping when combined.
/// With both at max (0.48 + 0.48 = 0.96), the combined output stays under 1.0.
pub const MAX_INDIVIDUAL_GAIN: f32 = 0.48;

fn default_binaural_volume() -> f32 {
    MAX_INDIVIDUAL_GAIN
}

fn default_noise_volume() -> f32 {
    MAX_INDIVIDUAL_GAIN
}

fn default_normalization() -> f32 {
    0.95
}

fn default_voice_type() -> String {
    "binaural".to_string()
}

#[derive(Deserialize, Debug, Clone)]
pub struct VolumeEnvelope {
    #[serde(rename = "type")]
    pub envelope_type: String,
    pub params: HashMap<String, f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct VoiceData {
    #[serde(alias = "synthFunctionName", alias = "synth_function")]
    pub synth_function_name: String,
    #[serde(alias = "parameters", default)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(alias = "volumeEnvelope")]
    pub volume_envelope: Option<VolumeEnvelope>,
    #[serde(default, alias = "isTransition")]
    pub is_transition: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_voice_type", alias = "voice_type")]
    pub voice_type: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StepData {
    #[serde(alias = "Duration", alias = "durationSeconds", alias = "stepDuration")]
    pub duration: f64,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub start: Option<f64>,
    pub voices: Vec<VoiceData>,
    #[serde(default = "default_binaural_volume", alias = "binaural_volume")]
    pub binaural_volume: f32,
    #[serde(default = "default_noise_volume", alias = "noise_volume")]
    pub noise_volume: f32,
    #[serde(default = "default_normalization", alias = "normalization_level")]
    pub normalization_level: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GlobalSettings {
    #[serde(alias = "sampleRate")]
    pub sample_rate: u32,
    #[serde(default = "default_crossfade_duration", alias = "crossfadeDuration")]
    pub crossfade_duration: f64,
    #[serde(default = "default_crossfade_curve", alias = "crossfadeCurve")]
    pub crossfade_curve: String,
    #[serde(default, alias = "outputFilename")]
    pub output_filename: Option<String>,
    #[serde(default = "default_normalization", alias = "normalization_level")]
    pub normalization_level: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TrackData {
    #[serde(alias = "globalSettings", alias = "global")]
    pub global_settings: GlobalSettings,
    #[serde(alias = "progression")]
    pub steps: Vec<StepData>,
    #[serde(default, alias = "overlay_clips")]
    pub clips: Vec<ClipData>,
    #[serde(default, alias = "noise")]
    pub background_noise: Option<BackgroundNoiseData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ClipData {
    #[serde(alias = "path", alias = "file")]
    pub file_path: String,
    #[serde(default, alias = "start_time")]
    pub start: f64,
    #[serde(default = "default_amp", alias = "gain")]
    pub amp: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BackgroundNoiseData {
    #[serde(
        default,
        alias = "file",
        alias = "file_path",
        alias = "params_path",
        alias = "noise_file"
    )]
    pub file_path: String,
    #[serde(default = "default_amp", alias = "gain", alias = "amp")]
    pub amp: f32,
    #[serde(default)]
    pub params: Option<crate::noise_params::NoiseParams>,
    #[serde(default = "default_zero_f64", alias = "start_time_seconds")]
    pub start_time: f64,
    #[serde(default = "default_zero_f64")]
    pub fade_in: f64,
    #[serde(default = "default_zero_f64")]
    pub fade_out: f64,
    #[serde(default)]
    pub amp_envelope: Vec<[f32; 2]>,
}

impl TrackData {
    /// Resolve clip and noise file paths relative to the provided base directory.
    pub fn resolve_relative_paths<P: AsRef<Path>>(&mut self, base: P) {
        let base = base.as_ref();
        if let Some(noise) = &mut self.background_noise {
            if !noise.file_path.is_empty() {
                let p = Path::new(&noise.file_path);
                if p.is_relative() {
                    noise.file_path = base.join(p).to_string_lossy().into_owned();
                }
            }
        }
        for clip in &mut self.clips {
            if !clip.file_path.is_empty() {
                let p = Path::new(&clip.file_path);
                if p.is_relative() {
                    clip.file_path = base.join(p).to_string_lossy().into_owned();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrackData;

    #[test]
    fn background_noise_deserializes_with_file_path() {
        let json = r#"
        {
            "global_settings": {
                "sample_rate": 44100
            },
            "steps": [],
            "background_noise": {
                "file_path": "presets/test.noise",
                "gain": 0.8,
                "start_time": 1.5,
                "fade_in": 0.25,
                "fade_out": 0.5,
                "amp_envelope": [[0.0, 0.2], [1.0, 1.0]]
            }
        }
        "#;

        let track: TrackData = serde_json::from_str(json).expect("valid track data");
        let noise = track.background_noise.expect("background noise present");
        assert_eq!(noise.file_path, "presets/test.noise");
        assert!((noise.amp - 0.8).abs() < f32::EPSILON);
        assert!((noise.start_time - 1.5).abs() < f64::EPSILON);
        assert!((noise.fade_in - 0.25).abs() < f64::EPSILON);
        assert!((noise.fade_out - 0.5).abs() < f64::EPSILON);
        assert_eq!(noise.amp_envelope.len(), 2);
    }
}

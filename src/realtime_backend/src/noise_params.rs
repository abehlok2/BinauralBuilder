use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NoiseSweep {
    #[serde(default)]
    pub start_min: f32,
    #[serde(default)]
    pub end_min: f32,
    #[serde(default)]
    pub start_max: f32,
    #[serde(default)]
    pub end_max: f32,
    #[serde(default)]
    pub start_q: f32,
    #[serde(default)]
    pub end_q: f32,
    #[serde(default)]
    pub start_casc: usize,
    #[serde(default)]
    pub end_casc: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NoiseParams {
    #[serde(default)]
    pub duration_seconds: f32,
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub lfo_waveform: String,
    #[serde(default)]
    pub transition: bool,
    #[serde(default)]
    pub lfo_freq: f32,
    #[serde(default)]
    pub start_lfo_freq: f32,
    #[serde(default)]
    pub end_lfo_freq: f32,
    #[serde(default)]
    pub sweeps: Vec<NoiseSweep>,
    #[serde(default, alias = "color_params")]
    pub noise_parameters: HashMap<String, Value>,
    #[serde(default)]
    pub start_lfo_phase_offset_deg: f32,
    #[serde(default)]
    pub end_lfo_phase_offset_deg: f32,
    #[serde(default)]
    pub start_intra_phase_offset_deg: f32,
    #[serde(default)]
    pub end_intra_phase_offset_deg: f32,
    #[serde(default)]
    pub initial_offset: f32,
    #[serde(default)]
    pub post_offset: f32,
    #[serde(default)]
    pub input_audio_path: String,
    #[serde(default)]
    pub exponent: Option<f32>,
    #[serde(default)]
    pub high_exponent: Option<f32>,
    #[serde(default)]
    pub distribution_curve: Option<f32>,
    #[serde(default)]
    pub lowcut: Option<f32>,
    #[serde(default)]
    pub highcut: Option<f32>,
    #[serde(default)]
    pub amplitude: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub start_time: f32,
    #[serde(default)]
    pub fade_in: f32,
    #[serde(default)]
    pub fade_out: f32,
    #[serde(default)]
    pub amp_envelope: Vec<[f32; 2]>,
    #[serde(default)]
    pub static_notches: Vec<Value>,
}

pub fn load_noise_params(path: &str) -> Result<NoiseParams, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let params: NoiseParams = serde_json::from_reader(file)?;
    Ok(apply_color_params(params))
}

pub fn load_noise_params_from_str(data: &str) -> Result<NoiseParams, serde_json::Error> {
    serde_json::from_str(data).map(apply_color_params)
}

fn color_val(map: &HashMap<String, Value>, key: &str) -> Option<f32> {
    map.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
}

fn color_i64(map: &HashMap<String, Value>, key: &str) -> Option<i64> {
    map.get(key).and_then(|v| v.as_i64())
}

pub fn apply_color_params(mut params: NoiseParams) -> NoiseParams {
    let noise_name = params
        .noise_parameters
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("pink")
        .to_string();

    params
        .noise_parameters
        .entry("name".to_string())
        .or_insert(Value::String(noise_name.clone()));

    if params.exponent.is_none() {
        params.exponent = color_val(&params.noise_parameters, "exponent");
    }
    if params.high_exponent.is_none() {
        params.high_exponent = color_val(&params.noise_parameters, "high_exponent");
    }
    if params.distribution_curve.is_none() {
        params.distribution_curve = color_val(&params.noise_parameters, "distribution_curve");
    }
    if params.lowcut.is_none() {
        params.lowcut = color_val(&params.noise_parameters, "lowcut");
    }
    if params.highcut.is_none() {
        params.highcut = color_val(&params.noise_parameters, "highcut");
    }
    if params.amplitude.is_none() {
        params.amplitude = color_val(&params.noise_parameters, "amplitude");
    }
    if params.seed.is_none() {
        params.seed = color_i64(&params.noise_parameters, "seed");
    }
    params
}

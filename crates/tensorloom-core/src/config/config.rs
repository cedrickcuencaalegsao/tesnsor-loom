use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HiddenLayerConfig {
    pub units: usize,
    pub activation: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerConfig {
    pub in_features: usize,
    pub hidden_layers: Vec<HiddenLayerConfig>,
    pub out_features: usize,
    pub output_activation: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareTarget {
    pub target_mode: String, // "GPU_DISCRETE", "GPU_INTEGRATED", or "CPU"
    pub device_index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub layers: crate::config::LayerConfig, // Structural layers profile
    pub hardware: HardwareTarget,
    #[serde(default)]
    pub task: TaskConfig,         // The dynamic configuration payload
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskConfig {
    /// "mse" (default) or "cross_entropy"
    #[serde(default = "default_loss")]
    pub loss: String,
    /// Classes per output cell. Sudoku = 9.
    #[serde(default)]
    pub classes: usize,
    /// CSV stores class / value_scale. Your sudoku file: 9.0
    #[serde(default = "default_scale")]
    pub value_scale: f32,
}

fn default_loss() -> String { "mse".into() }
fn default_scale() -> f32 { 1.0 }

impl Default for TaskConfig {
    fn default() -> Self {
        Self { loss: default_loss(), classes: 0, value_scale: default_scale() }
    }
}


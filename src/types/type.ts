// Mirrors the Rust structs in tensorloom-core (serde field names are snake_case).

export interface HiddenLayerConfig {
  units: number;
  activation: string; // "relu" | "tanh" | "sigmoid"
}

export interface LayerConfig {
  in_features: number;
  hidden_layers: HiddenLayerConfig[];
  out_features: number;       // output units (= number of target columns in the CSV)
  output_activation: string;  // "none" | "relu" | "tanh" | "sigmoid"
}

export type LossKind = "mse" | "cross_entropy";

// Rust: TaskConfig. "mse" = regression. "cross_entropy" = every output column is a
// class (1..=classes) predicted with softmax, e.g. sudoku cells with classes = 9.
export interface TaskConfig {
  loss: LossKind;
  classes: number;      // classes per output cell (0 for regression)
  value_scale: number;  // CSV stores class / value_scale (sudoku file: 9)
}

export const DEFAULT_TASK: TaskConfig = {
  loss: "mse",
  classes: 0,
  value_scale: 1,
};

export type TargetMode = "CPU" | "GPU_DISCRETE" | "GPU_INTEGRATED";

export interface HardwareTarget {
  target_mode: TargetMode;
  device_index: number;
}

export interface TrainConfig {
  epochs: number;
  batch_size: number;
  lr: number;
  layers: LayerConfig;
  hardware: HardwareTarget;
  task?: TaskConfig; // Rust defaults to regression (mse) when omitted
}

export interface DetectedGpu {
  id: number;
  name: string;
  device_type: "Discrete" | "Integrated" | "Virtual";
  backend: string;
}

export interface HardwareInventory {
  gpus: DetectedGpu[];
  fallback_cpu: boolean;
}

// Rust: #[serde(tag = "type", content = "payload")]
export type TrainEvent =
  | { type: "EpochStarted"; payload: { epoch: number } }
  | {
      type: "BatchCompleted";
      payload: { loss: number; accuracy: number; progress: number };
    }
  | {
      type: "EpochCompleted";
      payload: { epoch: number; avg_loss: number; val_accuracy: number };
    }
  | { type: "TrainingFinished"; payload: { success: boolean } }
  | { type: "Network"; payload: NetworkSnapshot }
  | { type: "Log"; payload: { level: string; message: string } };

export interface EdgeBlock {
  rows: number; // shown "from" nodes
  cols: number; // shown "to" nodes
  weights: number[]; // row-major, rows * cols
}

export interface NetworkSnapshot {
  epoch: number;
  layer_sizes: number[]; // real widths, e.g. [3, 16, 1]
  shown_sizes: number[]; // widths actually sent, e.g. [3, 12, 1]
  node_activity: number[][]; // per layer, per shown node
  edges: EdgeBlock[]; // layers - 1 blocks
}

export type ExportFormat = "json" | "bin" | "onnx";

export interface TrainingRequest {
  config: TrainConfig;
  data: string;
  exportFormat: ExportFormat;
  outputPath: string;
}
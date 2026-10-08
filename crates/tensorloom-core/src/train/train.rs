use crate::config::TrainConfig;
use crate::events::{EdgeBlock, NetworkSnapshot, TrainEvent};
use crate::model::{PortableModel, PrimitiveLayer};
use std::sync::mpsc::Sender;

// Burn ML core imports
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

// ---------------------------------------------------------------------------
// Training hyper-parameters. Move these into TrainConfig (with serde defaults)
// when you want the UI to control them.
// ---------------------------------------------------------------------------
const SEED: u64 = 42;
const VAL_FRACTION: f32 = 0.15; // held-out share used for early stopping + honest metrics
const MIN_ROWS_FOR_SPLIT: usize = 20; // below this, validate on the training data
const WARMUP_FRACTION: f64 = 0.03; // LR warm-up, as a share of all steps
const MIN_LR_FRACTION: f64 = 0.01; // cosine decay floor, as a share of base LR
const GRAD_CLIP_NORM: f32 = 1.0;

/// Activation applied after a layer. Kept outside the Burn module so any
/// number of layers can each pick their own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Linear,
    Relu,
    Tanh,
    Sigmoid,
}

impl Act {
    fn parse(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "none" | "linear" => Ok(Act::Linear),
            "relu" => Ok(Act::Relu),
            "tanh" => Ok(Act::Tanh),
            "sigmoid" => Ok(Act::Sigmoid),
            other => Err(format!(
                "Unknown activation '{other}'. Use one of: none, relu, tanh, sigmoid."
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Act::Linear => "none",
            Act::Relu => "relu",
            Act::Tanh => "tanh",
            Act::Sigmoid => "sigmoid",
        }
    }

    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Act::Linear => x,
            Act::Relu => activation::relu(x),
            Act::Tanh => activation::tanh(x),
            Act::Sigmoid => activation::sigmoid(x),
        }
    }

    /// Weight init matched to the activation that follows the layer.
    /// He init keeps ReLU stacks from dying/vanishing; Xavier suits the rest.
    fn initializer(self) -> Initializer {
        match self {
            Act::Relu => Initializer::KaimingNormal {
                gain: 2f64.sqrt(),
                fan_out_only: false,
            },
            Act::Tanh => Initializer::XavierUniform { gain: 5.0 / 3.0 },
            Act::Linear | Act::Sigmoid => Initializer::XavierUniform { gain: 1.0 },
        }
    }
}

/// Core neural network: any number of dense layers.
#[derive(Module, Debug)]
pub struct TensorLoomNetwork<B: Backend> {
    pub layers: Vec<Linear<B>>,
}

impl<B: Backend> TensorLoomNetwork<B> {
    /// Prediction only (no intermediate clones) - used in the hot training path.
    pub fn forward(&self, input: Tensor<B, 2>, acts: &[Act]) -> Tensor<B, 2> {
        let mut x = input;
        for (layer, act) in self.layers.iter().zip(acts) {
            x = act.apply(layer.forward(x));
        }
        x
    }

    /// Returns the post-activation output of every layer; the last one is the prediction.
    pub fn forward_all(&self, input: Tensor<B, 2>, acts: &[Act]) -> Vec<Tensor<B, 2>> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut x = input;
        for (layer, act) in self.layers.iter().zip(acts) {
            x = act.apply(layer.forward(x));
            outputs.push(x.clone());
        }
        outputs
    }
}

// ---------------------------------------------------------------------------
// Data handling
// ---------------------------------------------------------------------------

/// Parsed dataset: row-major features and targets.
struct Dataset {
    features: Vec<f32>,
    targets: Vec<f32>,
    rows: usize,
}

/// Parse CSV where each row is `feature_1,...,feature_n,target_1,...,target_m`.
/// A non-numeric first row is treated as a header and skipped.
fn parse_csv(raw: &str, in_features: usize, out_features: usize) -> Result<Dataset, String> {
    let expected = in_features + out_features;
    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut rows = 0usize;

    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        let parsed: Result<Vec<f32>, _> = cells.iter().map(|c| c.parse::<f32>()).collect();

        let values = match parsed {
            Ok(v) => v,
            Err(_) if rows == 0 => continue, // header row
            Err(e) => return Err(format!("CSV line {}: {}", line_no + 1, e)),
        };

        if values.len() != expected {
            return Err(format!(
                "CSV line {}: expected {} columns ({} features + {} targets), found {}",
                line_no + 1,
                expected,
                in_features,
                out_features,
                values.len()
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("CSV line {}: NaN or infinite value", line_no + 1));
        }

        features.extend_from_slice(&values[..in_features]);
        targets.extend_from_slice(&values[in_features..]);
        rows += 1;
    }

    if rows == 0 {
        return Err("CSV contained no data rows".to_string());
    }
    Ok(Dataset { features, targets, rows })
}

/// Per-column mean / std (computed in f64 for stability).
struct ColStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

fn col_stats(data: &[f32], rows: usize, cols: usize) -> ColStats {
    let mut mean = vec![0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            mean[c] += data[r * cols + c] as f64;
        }
    }
    mean.iter_mut().for_each(|m| *m /= rows as f64);

    let mut var = vec![0f64; cols];
    for r in 0..rows {
        for c in 0..cols {
            let d = data[r * cols + c] as f64 - mean[c];
            var[c] += d * d;
        }
    }
    let std = var
        .iter()
        .map(|v| {
            let s = (v / rows as f64).sqrt() as f32;
            if s < 1e-8 { 1.0 } else { s } // constant column: avoid divide-by-zero
        })
        .collect();

    ColStats {
        mean: mean.iter().map(|&m| m as f32).collect(),
        std,
    }
}

fn standardize(data: &mut [f32], cols: usize, stats: &ColStats) {
    for (i, v) in data.iter_mut().enumerate() {
        let c = i % cols;
        *v = (*v - stats.mean[c]) / stats.std[c];
    }
}

fn gather<T: Copy>(data: &[T], idx: &[usize], cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(idx.len() * cols);
    for &r in idx {
        out.extend_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    out
}

/// Tiny deterministic RNG (xorshift64) so runs are reproducible without extra deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Classification support (softmax + cross-entropy over groups of outputs)
// ---------------------------------------------------------------------------

/// Input cell value d (0 = blank, 1..=classes = given) -> one-hot of width classes+1.
/// Output layout: row-major, cell i occupies [i*(classes+1) .. (i+1)*(classes+1)).
fn one_hot_inputs(feat: &[f32], classes: usize, scale: f32) -> Result<Vec<f32>, String> {
    let width = classes + 1;
    let mut out = vec![0f32; feat.len() * width];
    for (i, &v) in feat.iter().enumerate() {
        let d = (v * scale).round() as i64;
        if d < 0 || d as usize > classes {
            return Err(format!(
                "Input value {v} is not a valid digit 0..={classes} at value_scale {scale}"
            ));
        }
        out[i * width + d as usize] = 1.0;
    }
    Ok(out)
}

/// Target value 1..=classes -> class id 0..classes-1 (kept as f32 so the rest of the
/// pipeline is unchanged; converted to Int right before the loss).
fn class_labels(targets: &[f32], classes: usize, scale: f32) -> Result<Vec<f32>, String> {
    targets
        .iter()
        .map(|&v| {
            let d = (v * scale).round() as i64;
            if d < 1 || d > classes as i64 {
                Err(format!(
                    "Target value {v} is not a valid class 1..={classes} at value_scale {scale}"
                ))
            } else {
                Ok((d - 1) as f32)
            }
        })
        .collect()
}

/// logits [batch, groups*classes] + labels [batch, groups] -> scalar loss.
/// CrossEntropyLoss applies log-softmax internally, so the network emits raw logits.
fn batch_loss<B: Backend>(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 2, Int>,
    classes: usize,
) -> Tensor<B, 1> {
    let [bs, total] = logits.dims();
    let groups = total / classes;
    let ce = CrossEntropyLossConfig::new().init(&logits.device());
    ce.forward(
        logits.reshape([bs * groups, classes]),
        labels.reshape([bs * groups]),
    )
}

/// Returns (loss, per-cell accuracy, whole-board accuracy).
fn evaluate_ce<B: Backend>(
    model: &TensorLoomNetwork<B>,
    acts: &[Act],
    x: Tensor<B, 2>,
    labels: Tensor<B, 2, Int>,
    classes: usize,
) -> Result<(f32, f32, f32), String> {
    let logits = model.forward(x, acts);
    let [n, total] = logits.dims();
    let groups = total / classes;

    let loss = tensor_to_vec(batch_loss(logits.clone(), labels.clone(), classes))?[0];

    let pred = logits
        .reshape([n * groups, classes])
        .argmax(1)
        .reshape([n, groups])
        .float();
    let p = tensor_to_vec(pred)?;
    let t = tensor_to_vec(labels.float())?;

    let (mut cells_ok, mut boards_ok) = (0usize, 0usize);
    for r in 0..n {
        let mut all = true;
        for g in 0..groups {
            if p[r * groups + g] == t[r * groups + g] {
                cells_ok += 1;
            } else {
                all = false;
            }
        }
        if all {
            boards_ok += 1;
        }
    }
    Ok((
        loss,
        cells_ok as f32 / (n * groups) as f32,
        boards_ok as f32 / n as f32,
    ))
}

// ---------------------------------------------------------------------------
// Dynamic pipeline routing gateway
// ---------------------------------------------------------------------------
pub fn execute_training(
    config: TrainConfig,
    raw_csv_data: String,
    tx: Sender<TrainEvent>,
) -> Result<PortableModel, String> {
    match config.hardware.target_mode.as_str() {
        "GPU_DISCRETE" => {
            let device = WgpuDevice::DiscreteGpu(config.hardware.device_index);
            run_burn_loop::<Autodiff<Wgpu>>(config, raw_csv_data, tx, device)
        }
        "GPU_INTEGRATED" => {
            let device = WgpuDevice::IntegratedGpu(config.hardware.device_index);
            run_burn_loop::<Autodiff<Wgpu>>(config, raw_csv_data, tx, device)
        }
        _ => {
            let device = NdArrayDevice::Cpu;
            run_burn_loop::<Autodiff<NdArray>>(config, raw_csv_data, tx, device)
        }
    }
}

fn tensor_to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Result<Vec<f32>, String> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("tensor conversion failed: {:?}", e))
}

// ---------------------------------------------------------------------------
// Snapshot (what the UI draws): every node and every weight, no sampling
// ---------------------------------------------------------------------------

/// `sizes` = [input, hidden..., output]; `layer_means` has one activity vector per entry in `sizes`.
/// Takes the detached (inner) model, so no autodiff bound is needed.
fn build_snapshot<B: Backend>(
    epoch: usize,
    model: &TensorLoomNetwork<B>,
    sizes: &[usize],
    layer_means: Vec<Vec<f32>>,
) -> Result<NetworkSnapshot, String> {
    // Burn Linear weights are row-major [d_input, d_output], which matches
    // the frontend's `weights[r * cols + c]` indexing.
    let mut edges = Vec::with_capacity(model.layers.len());
    for (i, layer) in model.layers.iter().enumerate() {
        edges.push(EdgeBlock {
            rows: sizes[i],
            cols: sizes[i + 1],
            weights: tensor_to_vec(layer.weight.val())?,
        });
    }

    Ok(NetworkSnapshot {
        epoch,
        layer_sizes: sizes.to_vec(),
        shown_sizes: sizes.to_vec(), // everything is shown
        node_activity: layer_means,
        edges,
    })
}

// ---------------------------------------------------------------------------
// Evaluation, LR schedule, export
// ---------------------------------------------------------------------------

/// Returns (mean MSE across outputs, per-output R^2 clamped to [0, 1]).
/// R^2 is computed per column, so outputs with different scales/means are judged
/// fairly (a single global variance hides badly-fit outputs in multi-output models).
fn evaluate<B: Backend>(
    model: &TensorLoomNetwork<B>,
    acts: &[Act],
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    col_var: &[f32],
) -> Result<(f32, Vec<f32>), String> {
    let pred = model.forward(x, acts);
    let mse_cols = tensor_to_vec(pred.sub(y).powf_scalar(2.0).mean_dim(0))?;
    let r2 = mse_cols
        .iter()
        .zip(col_var)
        .map(|(m, v)| (1.0 - m / v).clamp(0.0, 1.0))
        .collect();
    let mse = mse_cols.iter().sum::<f32>() / mse_cols.len() as f32;
    Ok((mse, r2))
}

/// Linear warm-up followed by cosine decay.
fn lr_at(step: usize, total: usize, base: f64) -> f64 {
    let warm = (total as f64 * WARMUP_FRACTION).ceil().max(1.0);
    let s = step as f64;
    if s < warm {
        return base * (s + 1.0) / warm;
    }
    let p = ((s - warm) / (total as f64 - warm).max(1.0)).min(1.0);
    let floor = base * MIN_LR_FRACTION;
    floor + 0.5 * (base - floor) * (1.0 + (std::f64::consts::PI * p).cos())
}

fn export_layers<B: Backend>(
    model: &TensorLoomNetwork<B>,
    sizes: &[usize],
    acts: &[Act],
) -> Result<Vec<PrimitiveLayer>, String> {
    let mut out = Vec::with_capacity(model.layers.len());
    for (i, layer) in model.layers.iter().enumerate() {
        let (n_in, n_out) = (sizes[i], sizes[i + 1]);
        let bias = match &layer.bias {
            Some(b) => tensor_to_vec(b.val())?,
            None => vec![0.0; n_out],
        };
        out.push(PrimitiveLayer {
            weights: tensor_to_vec(layer.weight.val())?,
            bias,
            shape: (n_in, n_out),
            activation: acts[i].name().to_string(),
        });
    }
    Ok(out)
}

/// The network trains on standardized data, but the exported model must accept RAW
/// inputs and return RAW outputs. Both transforms are affine, so they fold exactly
/// into the first layer's weights/bias and (when the output is linear) the last
/// layer's. The exported model is then self-contained: no extra metadata required.
/// (Regression only: classification models use one-hot inputs and raw logits.)
fn fold_normalization(layers: &mut [PrimitiveLayer], x: &ColStats, y: Option<&ColStats>) {
    if let Some(first) = layers.first_mut() {
        let (n_in, n_out) = first.shape;
        for j in 0..n_out {
            let mut shift = 0f32;
            for i in 0..n_in {
                let w = first.weights[i * n_out + j];
                shift += w * x.mean[i] / x.std[i];
                first.weights[i * n_out + j] = w / x.std[i];
            }
            first.bias[j] -= shift;
        }
    }
    if let (Some(y), Some(last)) = (y, layers.last_mut()) {
        let (n_in, n_out) = last.shape;
        for j in 0..n_out {
            for i in 0..n_in {
                last.weights[i * n_out + j] *= y.std[j];
            }
            last.bias[j] = last.bias[j] * y.std[j] + y.mean[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Generic training loop
// ---------------------------------------------------------------------------
fn run_burn_loop<B>(
    config: TrainConfig,
    raw_data: String,
    tx: Sender<TrainEvent>,
    device: B::Device,
) -> Result<PortableModel, String>
where
    B: AutodiffBackend,
{
    // ---- Task: regression (MSE) or grouped classification (softmax + cross-entropy) ----
    let input_dim = config.layers.in_features; // CSV feature columns
    let output_dim = config.layers.out_features; // CSV target columns
    if input_dim == 0 || output_dim == 0 {
        return Err("Input features and output units must be at least 1".to_string());
    }
    if config.epochs == 0 {
        return Err("Epochs must be at least 1".to_string());
    }

    let ce = match config.task.loss.to_lowercase().as_str() {
        "mse" | "" => false,
        "cross_entropy" | "crossentropy" => true,
        other => {
            return Err(format!(
                "Unknown loss '{other}'. Use 'mse' or 'cross_entropy'."
            ))
        }
    };
    let classes = config.task.classes;
    let value_scale = config.task.value_scale;
    if ce && classes < 2 {
        return Err("cross_entropy needs task.classes >= 2 (e.g. 9 for sudoku)".to_string());
    }
    if ce && value_scale <= 0.0 {
        return Err("task.value_scale must be > 0".to_string());
    }

    // The network sees one-hot inputs and emits logits for every (cell, class) pair.
    let net_in = if ce { input_dim * (classes + 1) } else { input_dim };
    let net_out = if ce { output_dim * classes } else { output_dim };

    // ---- Topology: sizes = [input, hidden..., output], one activation per layer ----
    let mut sizes = vec![net_in];
    let mut acts: Vec<Act> = Vec::new();
    for (i, h) in config.layers.hidden_layers.iter().enumerate() {
        if h.units == 0 {
            return Err(format!("Hidden layer {} must have at least 1 unit", i + 1));
        }
        sizes.push(h.units);
        acts.push(Act::parse(&h.activation)?);
    }
    sizes.push(net_out);
    let output_act = Act::parse(&config.layers.output_activation)?;
    if ce && output_act != Act::Linear {
        return Err("cross_entropy requires output activation 'none' (raw logits)".to_string());
    }
    acts.push(output_act);

    // ---- Data: parse, (encode), shuffle, split ----
    let data = parse_csv(&raw_data, input_dim, output_dim)?;
    let n = data.rows;

    // Classification: features -> one-hot, targets -> class ids (as f32).
    let (features, targets) = if ce {
        (
            one_hot_inputs(&data.features, classes, value_scale)?,
            class_labels(&data.targets, classes, value_scale)?,
        )
    } else {
        (data.features, data.targets)
    };

    let mut rng = Rng::new(SEED);

    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let n_val = if n >= MIN_ROWS_FOR_SPLIT {
        ((n as f32 * VAL_FRACTION).round() as usize).clamp(1, n - 1)
    } else {
        0
    };
    let (val_idx, train_idx): (Vec<usize>, Vec<usize>) = if n_val == 0 {
        (order.clone(), order) // too little data to hold out: validate on train
    } else {
        (order[..n_val].to_vec(), order[n_val..].to_vec())
    };
    let (nt, nv) = (train_idx.len(), val_idx.len());

    let mut x_train_v = gather(&features, &train_idx, net_in);
    let mut y_train_v = gather(&targets, &train_idx, output_dim);
    let mut x_val_v = gather(&features, &val_idx, net_in);
    let mut y_val_v = gather(&targets, &val_idx, output_dim);

    // ---- Standardize (statistics from the training rows only: no leakage) ----
    // Regression: inputs are always standardized; targets only when the output
    // activation is linear. Classification: one-hot inputs and class ids are left alone.
    let x_stats = if ce {
        None
    } else {
        let s = col_stats(&x_train_v, nt, net_in);
        standardize(&mut x_train_v, net_in, &s);
        standardize(&mut x_val_v, net_in, &s);
        Some(s)
    };

    let y_stats = if !ce && output_act == Act::Linear {
        let s = col_stats(&y_train_v, nt, output_dim);
        standardize(&mut y_train_v, output_dim, &s);
        standardize(&mut y_val_v, output_dim, &s);
        Some(s)
    } else {
        None
    };

    // Per-output variance of the validation targets (denominator of R^2). Regression only.
    let col_var: Vec<f32> = if ce {
        Vec::new()
    } else {
        let val_stats = col_stats(&y_val_v, nv, output_dim);
        val_stats.std.iter().map(|s| (s * s).max(f32::EPSILON)).collect()
    };

    // ---- Batching ----
    let batch_size = (nt / 10).clamp(16, 256).min(nt);
    let steps_per_epoch = nt.div_ceil(batch_size);
    let total_steps = steps_per_epoch * config.epochs;

    let _ = tx.send(TrainEvent::Log {
        level: "info".into(),
        message: format!(
            "Loaded {} rows ({} train / {} val). Task: {}. Network: {}. Batch size {}, {} steps/epoch.",
            n,
            nt,
            nv,
            if ce {
                format!("classification ({} classes per output, softmax + cross-entropy)", classes)
            } else {
                "regression (MSE)".to_string()
            },
            sizes
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(" -> "),
            batch_size,
            steps_per_epoch
        ),
    });

    // ---- Model, optimizer, tensors ----
    let mut model: TensorLoomNetwork<B> = TensorLoomNetwork {
        layers: sizes
            .windows(2)
            .zip(&acts)
            .map(|(w, act)| {
                LinearConfig::new(w[0], w[1])
                    .with_initializer(act.initializer())
                    .init(&device)
            })
            .collect(),
    };

    // Adam (decoupled weight decay) + gradient clipping for stable, regularized training.
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(GRAD_CLIP_NORM)))
        .init::<B, TensorLoomNetwork<B>>();

    let x_train = Tensor::<B, 2>::from_data(TensorData::new(x_train_v, [nt, net_in]), &device);
    let y_train = Tensor::<B, 2>::from_data(TensorData::new(y_train_v, [nt, output_dim]), &device);
    let x_val = Tensor::<B, 2>::from_data(TensorData::new(x_val_v, [nv, net_in]), &device);
    let y_val = Tensor::<B, 2>::from_data(TensorData::new(y_val_v, [nv, output_dim]), &device);

    // Detached copies for evaluation / snapshots (no autodiff graph).
    let x_train_inner = x_train.clone().inner();
    let x_val_inner = x_val.inner();
    let y_val_inner = y_val.inner();

    let base_lr = config.lr as f64;
    let snapshot_every = (config.epochs / 20).max(1); // at most ~20 snapshots per run
    let patience = (config.epochs / 5).clamp(10, 100);

    let mut shuffled: Vec<i32> = (0..nt as i32).collect();
    let mut step = 0usize;

    let mut best_model: Option<TensorLoomNetwork<B::InnerBackend>> = None;
    let mut best_val = f32::INFINITY;
    let mut best_epoch = 0usize;
    // Regression: per-output R^2. Classification: [cell accuracy, board accuracy].
    let mut best_r2: Vec<f32> = vec![0.0; if ce { 2 } else { output_dim }];
    let mut stale = 0usize;
    let mut last_epoch = 0usize;

    for epoch in 1..=config.epochs {
        last_epoch = epoch;
        let _ = tx.send(TrainEvent::EpochStarted { epoch });

        // ---- One pass over the data in shuffled mini-batches ----
        rng.shuffle(&mut shuffled);
        let mut loss_sum = Tensor::<B, 1>::zeros([1], &device); // stays on-device until epoch end

        for chunk in shuffled.chunks(batch_size) {
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(chunk.to_vec(), [chunk.len()]),
                &device,
            );
            let xb = x_train.clone().select(0, idx.clone());
            let yb = y_train.clone().select(0, idx);

            let pred = model.forward(xb, &acts);
            let loss = if ce {
                batch_loss(pred, yb.int(), classes)
            } else {
                pred.sub(yb).powf_scalar(2.0).mean()
            };
            loss_sum = loss_sum + loss.clone().detach().mul_scalar(chunk.len() as f32);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = lr_at(step, total_steps, base_lr);
            model = optimizer.step(lr, model, grads);
            step += 1;
        }
        let train_loss = tensor_to_vec(loss_sum)?[0] / nt as f32;

        // ---- Validation on the detached model ----
        let inner = model.valid();
        let (val_loss, metrics, val_score) = if ce {
            let (l, cell_acc, board_acc) = evaluate_ce(
                &inner,
                &acts,
                x_val_inner.clone(),
                y_val_inner.clone().int(),
                classes,
            )?;
            (l, vec![cell_acc, board_acc], cell_acc) // UI score = per-cell accuracy
        } else {
            let (l, r2) = evaluate(
                &inner,
                &acts,
                x_val_inner.clone(),
                y_val_inner.clone(),
                &col_var,
            )?;
            let s = r2.iter().sum::<f32>() / r2.len() as f32; // macro-average R^2
            (l, r2, s)
        };

        // ---- Keep the best checkpoint; stop when validation stops improving ----
        if val_loss < best_val - 1e-7 {
            best_val = val_loss;
            best_epoch = epoch;
            best_r2 = metrics.clone();
            best_model = Some(inner.clone());
            stale = 0;
        } else {
            stale += 1;
        }

        // ---- Snapshot for the UI (throttled) ----
        if epoch == 1 || epoch == config.epochs || epoch % snapshot_every == 0 {
            let mut means = vec![tensor_to_vec(x_train_inner.clone().mean_dim(0))?];
            for out in inner.forward_all(x_train_inner.clone(), &acts) {
                means.push(tensor_to_vec(out.mean_dim(0))?);
            }
            let snapshot = build_snapshot(epoch, &inner, &sizes, means)?;
            let _ = tx.send(TrainEvent::Network(snapshot));
        }

        let _ = tx.send(TrainEvent::BatchCompleted {
            loss: train_loss,
            accuracy: val_score,
            progress: epoch as f32 / config.epochs as f32,
        });
        let _ = tx.send(TrainEvent::EpochCompleted {
            epoch,
            avg_loss: train_loss,
            val_accuracy: val_score, // held-out score (macro R^2, or per-cell accuracy)
        });

        if stale >= patience {
            let _ = tx.send(TrainEvent::Log {
                level: "info".into(),
                message: format!(
                    "Early stop at epoch {epoch}: no validation improvement for {patience} epochs (best was epoch {best_epoch})."
                ),
            });
            break;
        }
    }

    // ---- Export the BEST checkpoint, not whatever the last epoch happened to be ----
    let final_model = best_model.ok_or_else(|| "Training produced no model".to_string())?;

    let summary = if ce {
        format!(
            "cell accuracy={:.3}, full-board accuracy={:.3}",
            best_r2[0], best_r2[1]
        )
    } else {
        format!(
            "Per-output R^2: {}",
            best_r2
                .iter()
                .enumerate()
                .map(|(i, r)| format!("out{}={:.3}", i + 1, r))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let _ = tx.send(TrainEvent::Log {
        level: "info".into(),
        message: format!(
            "Best epoch {best_epoch}/{last_epoch}, val loss {best_val:.5}. {summary}"
        ),
    });

    let mut exportable_layers = export_layers(&final_model, &sizes, &acts)?;
    if let Some(xs) = &x_stats {
        fold_normalization(&mut exportable_layers, xs, y_stats.as_ref());
    }

    let _ = tx.send(TrainEvent::TrainingFinished { success: true });

    let metadata = if ce {
        // Parsed by PortableModel::classification() in model/inference.rs - keep the format.
        format!(
            "TensorLoom Engine v1.1 | task=classification | classes={} | value_scale={}",
            classes, value_scale
        )
    } else {
        "TensorLoom Engine v1.1 Cross-Platform".to_string()
    };

    Ok(PortableModel {
        layers: exportable_layers,
        metadata,
    })
}

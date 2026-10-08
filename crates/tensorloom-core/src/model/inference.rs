// Pure-Rust forward pass over an exported PortableModel. No Burn needed, so it is
// cheap to call and works the same on every platform.
//
// Two kinds of model are supported:
//   * regression      - layers map raw inputs to raw outputs (unchanged behaviour)
//   * classification  - trained with softmax + cross-entropy. The exporter marks these in
//                       `metadata` ("task=classification | classes=N | value_scale=S").
//                       For them `input_size()`/`output_size()`/`predict()` use the same
//                       CSV-style cell values you trained with (e.g. 81 sudoku cells as
//                       digit/9); the one-hot encoding and the per-cell argmax happen inside.

use super::PortableModel;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutputEvaluation {
    pub index: usize,
    pub mse: f32,
    /// 1.0 = perfect, 0.0 = no better than predicting the average, < 0 = worse than that.
    pub r2: f32,
    pub mean_target: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evaluation {
    pub rows: usize,
    /// Average of the per-output MSE values.
    pub mse: f32,
    /// Average of the per-output R² values (1.0 = perfect, 0.0 = no better than the mean, < 0 = worse).
    pub r2: f32,
    /// Average of the per-output target means.
    pub mean_target: f32,
    /// One entry per model output, in output order.
    #[serde(default)]
    pub per_output: Vec<OutputEvaluation>,
    /// Classification only: share of output cells predicted correctly.
    #[serde(default)]
    pub cell_accuracy: Option<f32>,
    /// Classification only: share of rows where every output cell is correct.
    #[serde(default)]
    pub board_accuracy: Option<f32>,
    /// Classification only: accuracy on cells that were blank (0) in the input.
    /// This is the honest number for puzzles, because given clues are trivially "correct".
    #[serde(default)]
    pub blank_accuracy: Option<f32>,
}

/// Settings of a classification model, read from `PortableModel::metadata`.
#[derive(Debug, Clone, Copy)]
pub struct ClassInfo {
    /// Classes per output cell (sudoku = 9).
    pub classes: usize,
    /// CSV stores class / value_scale (sudoku file = 9).
    pub value_scale: f32,
}

fn activate(name: &str, v: f32) -> Result<f32, String> {
    match name {
        "relu" => Ok(v.max(0.0)),
        "none" | "linear" | "" => Ok(v),
        "sigmoid" => Ok(1.0 / (1.0 + (-v).exp())),
        "tanh" => Ok(v.tanh()),
        other => Err(format!("unknown activation '{other}'")),
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Digits already used in the row, column and 3x3 box of `cell` (index 0 is unused).
fn used_digits(b: &[u8], cell: usize) -> [bool; 10] {
    let (r, c) = (cell / 9, cell % 9);
    let (br, bc) = (r / 3 * 3, c / 3 * 3);
    let mut used = [false; 10];
    for i in 0..9 {
        used[b[r * 9 + i] as usize] = true;
        used[b[i * 9 + c] as usize] = true;
        used[b[(br + i / 3) * 9 + bc + i % 3] as usize] = true;
    }
    used
}

impl PortableModel {
    /// Some(..) when the model was exported from a softmax + cross-entropy run.
    pub fn classification(&self) -> Option<ClassInfo> {
        let mut is_cls = false;
        let mut classes = 0usize;
        let mut scale = 1.0f32;
        for part in self.metadata.split('|') {
            let part = part.trim();
            if part == "task=classification" {
                is_cls = true;
            } else if let Some(v) = part.strip_prefix("classes=") {
                classes = v.trim().parse().unwrap_or(0);
            } else if let Some(v) = part.strip_prefix("value_scale=") {
                scale = v.trim().parse().unwrap_or(1.0);
            }
        }
        if is_cls && classes >= 2 && scale > 0.0 {
            Some(ClassInfo { classes, value_scale: scale })
        } else {
            None
        }
    }

    /// Number of input values a caller provides (cells for classification models).
    pub fn input_size(&self) -> Option<usize> {
        let n = self.layers.first().map(|l| l.shape.0)?;
        Some(match self.classification() {
            Some(c) => n / (c.classes + 1),
            None => n,
        })
    }

    /// Number of output values a caller receives (cells for classification models).
    pub fn output_size(&self) -> Option<usize> {
        let n = self.layers.last().map(|l| l.shape.1)?;
        Some(match self.classification() {
            Some(c) => n / c.classes,
            None => n,
        })
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum()
    }

    /// Check that shapes, buffers and activations are consistent before running anything.
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("model has no layers".to_string());
        }
        for (k, layer) in self.layers.iter().enumerate() {
            let (n_in, n_out) = layer.shape;
            if layer.weights.len() != n_in * n_out {
                return Err(format!(
                    "layer {k}: expected {} weights for shape ({n_in}, {n_out}), found {}",
                    n_in * n_out,
                    layer.weights.len()
                ));
            }
            if layer.bias.len() != n_out {
                return Err(format!(
                    "layer {k}: expected {n_out} biases, found {}",
                    layer.bias.len()
                ));
            }
            activate(&layer.activation, 0.0).map_err(|e| format!("layer {k}: {e}"))?;
            if let Some(next) = self.layers.get(k + 1) {
                if next.shape.0 != n_out {
                    return Err(format!(
                        "layer {k} outputs {n_out} values but layer {} expects {}",
                        k + 1,
                        next.shape.0
                    ));
                }
            }
        }
        if let Some(c) = self.classification() {
            let n_in = self.layers[0].shape.0;
            let n_out = self.layers[self.layers.len() - 1].shape.1;
            if n_in % (c.classes + 1) != 0 {
                return Err(format!(
                    "classification model: input width {n_in} is not a multiple of {}",
                    c.classes + 1
                ));
            }
            if n_out % c.classes != 0 {
                return Err(format!(
                    "classification model: output width {n_out} is not a multiple of {}",
                    c.classes
                ));
            }
        }
        Ok(())
    }

    /// Run one input row through the model.
    /// Regression: raw outputs. Classification: the predicted value of every cell
    /// (class + 1) / value_scale, i.e. the same format as the CSV targets.
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.validate()?;
        self.predict_unchecked(input)
    }

    /// Same as `predict`, but assumes `validate()` already passed (used for big batches).
    fn predict_unchecked(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        match self.classification() {
            None => self.network_forward(input),
            Some(c) => {
                let probs = self.class_probs(input, c)?;
                Ok(probs
                    .iter()
                    .map(|p| (argmax(p) + 1) as f32 / c.value_scale)
                    .collect())
            }
        }
    }

    /// Raw network pass: input vector of the first layer's width -> last layer's output.
    fn network_forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let expected = self.layers[0].shape.0;
        if input.len() != expected {
            return Err(format!("expected {expected} input values, got {}", input.len()));
        }
        if let Some(bad) = input.iter().find(|v| !v.is_finite()) {
            return Err(format!("input contains a non-finite value ({bad})"));
        }

        let mut current = input.to_vec();
        for (k, layer) in self.layers.iter().enumerate() {
            let (n_in, n_out) = layer.shape;
            // Weights are row-major [n_in, n_out]: weight from input i to output j.
            // Row-wise accumulation is cache friendly and skips zero inputs, which makes
            // the (mostly-zero) one-hot first layer very cheap.
            let mut next = layer.bias.clone();
            for i in 0..n_in {
                let x = current[i];
                if x == 0.0 {
                    continue;
                }
                let row = &layer.weights[i * n_out..(i + 1) * n_out];
                for j in 0..n_out {
                    next[j] += x * row[j];
                }
            }
            for v in next.iter_mut() {
                *v = activate(&layer.activation, *v).map_err(|e| format!("layer {k}: {e}"))?;
            }
            current = next;
        }
        Ok(current)
    }

    /// Classification: cell values (class / value_scale, 0 = blank) -> per-cell class
    /// probabilities (softmax over each group of `classes` logits).
    fn class_probs(&self, cells: &[f32], c: ClassInfo) -> Result<Vec<Vec<f32>>, String> {
        let width = c.classes + 1;
        let expected = self.layers[0].shape.0 / width;
        if cells.len() != expected {
            return Err(format!("expected {expected} input values, got {}", cells.len()));
        }
        let mut one_hot = vec![0f32; cells.len() * width];
        for (i, &v) in cells.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("input contains a non-finite value ({v})"));
            }
            let d = (v * c.value_scale).round();
            if d < 0.0 || d > c.classes as f32 {
                return Err(format!(
                    "cell {i}: value {v} is not a valid digit 0..={} at value_scale {}",
                    c.classes, c.value_scale
                ));
            }
            one_hot[i * width + d as usize] = 1.0;
        }
        let logits = self.network_forward(&one_hot)?;
        Ok(logits.chunks(c.classes).map(softmax).collect())
    }

    /// Solve a 9x9 sudoku with a classification model (9 classes, 81 cells).
    /// `board` has 81 digits, 0 = blank. Each round the network scores every blank cell,
    /// digits that already appear in the cell's row/column/box are ruled out, and the single
    /// most confident (cell, digit) is filled in. Repeats until the board is full.
    /// Returns an error if it gets stuck (a blank cell with no legal digit left).
    pub fn solve_sudoku(&self, board: &[u8]) -> Result<Vec<u8>, String> {
        self.validate()?;
        let c = self
            .classification()
            .ok_or("this model was not trained with cross_entropy")?;
        if c.classes != 9 || self.input_size() != Some(81) || self.output_size() != Some(81) {
            return Err("model is not a 9-class, 81-cell sudoku model".to_string());
        }
        if board.len() != 81 || board.iter().any(|&d| d > 9) {
            return Err("board must contain 81 digits in 0..=9 (0 = blank)".to_string());
        }

        let mut b = board.to_vec();
        while b.contains(&0) {
            let cells: Vec<f32> = b.iter().map(|&d| d as f32 / c.value_scale).collect();
            let probs = self.class_probs(&cells, c)?;

            let mut best: Option<(usize, usize, f32)> = None; // (cell, digit, confidence)
            for cell in 0..81 {
                if b[cell] != 0 {
                    continue;
                }
                let used = used_digits(&b, cell);
                let legal_sum: f32 = (1..=9).filter(|&d| !used[d]).map(|d| probs[cell][d - 1]).sum();
                if !legal_sum.is_finite() || legal_sum <= 0.0 {
                    return Err(format!(
                        "stuck: cell (row {}, col {}) has no legal digit left",
                        cell / 9 + 1,
                        cell % 9 + 1
                    ));
                }
                for d in 1..=9 {
                    if used[d] {
                        continue;
                    }
                    let p = probs[cell][d - 1] / legal_sum;
                    if best.map_or(true, |(_, _, bp)| p > bp) {
                        best = Some((cell, d, p));
                    }
                }
            }
            let (cell, digit, _) = best.ok_or("stuck: no candidate found")?;
            b[cell] = digit as u8;
        }
        Ok(b)
    }
}

/// Parse `feature_1,...,feature_n,target_1,...,target_m` rows. A non-numeric first row is a header.
fn parse_rows(
    raw: &str,
    in_features: usize,
    out_features: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let expected_cols = in_features + out_features;
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for (line_no, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Result<Vec<f32>, _> = line.split(',').map(|c| c.trim().parse::<f32>()).collect();
        let values = match parsed {
            Ok(v) => v,
            Err(_) if rows.is_empty() => continue, // header
            Err(e) => return Err(format!("CSV line {}: {}", line_no + 1, e)),
        };
        if values.len() != expected_cols {
            return Err(format!(
                "CSV line {}: expected {} columns ({} features + {} targets), found {}",
                line_no + 1,
                expected_cols,
                in_features,
                out_features,
                values.len()
            ));
        }
        rows.push(values[..in_features].to_vec());
        targets.push(values[in_features..].to_vec());
    }

    if rows.is_empty() {
        return Err("CSV contained no data rows".to_string());
    }
    Ok((rows, targets))
}

/// Score a model on a CSV that includes the target column(s), one per model output.
pub fn evaluate(model: &PortableModel, raw_csv: &str) -> Result<Evaluation, String> {
    model.validate()?;
    let n_in = model.input_size().ok_or("model has no layers")?;
    let n_out = model.output_size().ok_or("model has no layers")?;
    let cls = model.classification();

    let (rows, targets) = parse_rows(raw_csv, n_in, n_out)?;
    let n = rows.len() as f32;

    // Per-output target means.
    let mut means = vec![0.0f32; n_out];
    for t in &targets {
        for (m, v) in means.iter_mut().zip(t) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }

    let mut sq_err = vec![0.0f32; n_out];
    let mut sq_var = vec![0.0f32; n_out];
    let (mut cells_ok, mut boards_ok) = (0usize, 0usize);
    let (mut blank_ok, mut blank_total) = (0usize, 0usize);

    for (row, t) in rows.iter().zip(&targets) {
        let pred = model.predict_unchecked(row)?;
        let mut all_correct = true;
        for j in 0..n_out {
            let e = pred[j] - t[j];
            let d = t[j] - means[j];
            sq_err[j] += e * e;
            sq_var[j] += d * d;

            if let Some(c) = cls {
                let hit = (pred[j] * c.value_scale).round() == (t[j] * c.value_scale).round();
                if hit {
                    cells_ok += 1;
                } else {
                    all_correct = false;
                }
                // Puzzle-style data: input cell j and output cell j are the same cell.
                if n_in == n_out && (row[j] * c.value_scale).round() == 0.0 {
                    blank_total += 1;
                    if hit {
                        blank_ok += 1;
                    }
                }
            }
        }
        if cls.is_some() && all_correct {
            boards_ok += 1;
        }
    }

    let per_output: Vec<OutputEvaluation> = (0..n_out)
        .map(|j| {
            let mse = sq_err[j] / n;
            let var = sq_var[j] / n;
            let r2 = if var > f32::EPSILON { 1.0 - mse / var } else { 0.0 };
            OutputEvaluation { index: j, mse, r2, mean_target: means[j] }
        })
        .collect();

    let k = n_out as f32;
    Ok(Evaluation {
        rows: rows.len(),
        mse: per_output.iter().map(|o| o.mse).sum::<f32>() / k,
        r2: per_output.iter().map(|o| o.r2).sum::<f32>() / k,
        mean_target: per_output.iter().map(|o| o.mean_target).sum::<f32>() / k,
        per_output,
        cell_accuracy: cls.map(|_| cells_ok as f32 / (rows.len() * n_out) as f32),
        board_accuracy: cls.map(|_| boards_ok as f32 / rows.len() as f32),
        blank_accuracy: if cls.is_some() && blank_total > 0 {
            Some(blank_ok as f32 / blank_total as f32)
        } else {
            None
        },
    })
}
# Tensor Loom

Tensor Loom is a cross-platform neural-network training engine written in Rust. It provides a configurable dense neural-network architecture with CPU and GPU execution, deterministic training, regression and classification support, validation metrics, live training events, network snapshots, early stopping, and portable model export.

The training engine is built around the [Burn](https://burn.dev/) deep-learning framework and supports CPU execution through `NdArray` and GPU execution through `WGPU`.

## Features

* Rust-based neural-network training engine
* Dynamic fully connected neural networks
* Configurable hidden layers
* Per-layer activation functions
* CPU training
* GPU training through WGPU
* Discrete GPU and integrated GPU selection
* Regression using Mean Squared Error
* Classification using softmax and cross-entropy
* CSV dataset loading
* Automatic header detection
* Input and output validation
* Feature standardization
* Deterministic dataset shuffling
* Train/validation splitting
* AdamW optimizer
* Gradient clipping
* Learning-rate warm-up
* Cosine learning-rate decay
* Early stopping
* Best-model checkpointing
* Training progress events
* Network visualization snapshots
* Per-output R² evaluation for regression
* Cell and full-board accuracy for classification
* Portable model export
* Normalization folding for self-contained regression models

---

## Architecture

Tensor Loom builds a dense neural network dynamically from the supplied training configuration.

```text
                    Tensor Loom
                         │
                         ▼
                  Training Config
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
         Regression             Classification
             │                       │
             ▼                       ▼
        Raw Features             One-Hot Input
             │                       │
             └───────────┬───────────┘
                         ▼
                  Neural Network
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Linear       Hidden       Output
           Layer        Layers       Layer
             │           │           │
             └───────────┴───────────┘
                         │
                         ▼
                    Evaluation
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Validation             Metrics
              │                     │
              └──────────┬──────────┘
                         ▼
                   Best Checkpoint
                         │
                         ▼
                  PortableModel
```

The network is represented as a collection of dynamically created `Linear` layers. Each layer has its own activation function.

---

# Supported Activations

Tensor Loom currently supports:

| Activation | Configuration name |
| ---------- | ------------------ |
| Linear     | `none` / `linear`  |
| ReLU       | `relu`             |
| Tanh       | `tanh`             |
| Sigmoid    | `sigmoid`          |

Activation names are parsed dynamically during network construction.

Weight initialization is selected according to the activation:

* ReLU → Kaiming Normal
* Tanh → Xavier Uniform
* Linear → Xavier Uniform
* Sigmoid → Xavier Uniform

This allows initialization to match the activation used by each layer.

---

# Hardware Acceleration

Tensor Loom can dynamically select the training backend based on the configured hardware target.

```text
GPU_DISCRETE
      │
      ▼
WGPU Discrete GPU

GPU_INTEGRATED
      │
      ▼
WGPU Integrated GPU

CPU
      │
      ▼
NdArray CPU Backend
```

The training gateway selects the backend using the configured `target_mode` and `device_index`.

### Supported Targets

```text
GPU_DISCRETE
GPU_INTEGRATED
CPU
```

For GPU execution, Tensor Loom uses Burn's WGPU backend. CPU execution uses the Burn NdArray backend.

---

# Dataset Format

Tensor Loom accepts CSV datasets.

The expected format is:

```text
feature_1,feature_2,...,feature_n,target_1,...,target_m
```

For example:

```csv
x1,x2,y
1.0,2.0,3.0
2.0,3.0,5.0
3.0,4.0,7.0
4.0,5.0,9.0
```

The first row may contain column names. If the first row cannot be parsed as numbers, Tensor Loom treats it as a header.

Every data row must contain exactly:

```text
input_features + output_features
```

values. NaN and infinite values are rejected.

---

# Regression

Regression uses Mean Squared Error (MSE).

```text
Input
  │
  ▼
Dense Network
  │
  ▼
Predicted Values
  │
  ▼
MSE Loss
```

Regression inputs are standardized using statistics calculated from the training data.

When the output activation is linear, target values are also standardized during training.

## Regression Metrics

Tensor Loom calculates:

* Mean Squared Error
* Per-output R²
* Macro-average R²

R² is calculated separately for every output column and clamped between `0` and `1`.

Example:

```text
Per-output R²:
out1=0.932
out2=0.887
out3=0.951
```

---

# Classification

Tensor Loom also supports grouped classification using cross-entropy.

Classification converts input values into one-hot encoded vectors.

For `N` classes, each input value uses:

```text
classes + 1
```

values, where `0` represents a blank/unset value and `1..N` represent classes.

The output layer produces:

```text
output_features × classes
```

logits.

The output activation must be:

```text
none
```

because cross-entropy expects raw logits.

## Classification Metrics

Tensor Loom reports:

* Per-cell accuracy
* Full-board accuracy

This makes the classification system suitable for grouped problems such as board-based classification tasks.

For example:

```text
Cell Accuracy: 94.2%
Full Board Accuracy: 71.8%
```

The implementation evaluates both individual classification groups and whether the entire sample was predicted correctly.

---

# Data Preprocessing

Tensor Loom performs preprocessing before training.

## 1. CSV Parsing

The raw CSV is converted into feature and target arrays.

## 2. Dataset Shuffling

A deterministic xorshift64 random-number generator is used.

```text
Seed = 42
```

This makes training runs reproducible.

## 3. Train/Validation Split

When the dataset contains at least 20 rows:

```text
15% → Validation
85% → Training
```

For smaller datasets, the implementation does not hold out a separate validation set and instead validates using the training data.

## 4. Standardization

Regression features are standardized using:

```text
z = (x - mean) / standard_deviation
```

The statistics are calculated only from the training rows to prevent validation leakage.

---

# Neural Network Configuration

The network topology follows:

```text
[input → hidden layers → output]
```

For example:

```text
[8 → 32 → 16 → 4]
```

represents:

```text
8 input units
      │
      ▼
32 hidden units
      │
      ▼
16 hidden units
      │
      ▼
4 output units
```

Every hidden layer specifies:

* Number of units
* Activation function

The output layer also has its own activation configuration.

---

# Training

Tensor Loom uses mini-batch gradient-based training.

The optimizer is:

```text
AdamW
```

with gradient clipping:

```text
Maximum gradient norm = 1.0
```

This provides weight decay and protects training from excessively large gradients.

---

# Learning Rate Schedule

The learning rate uses two phases.

```text
Learning Rate
     │
     │       ╭──────────────╮
     │      ╱                ╲
     │     ╱                  ╲
     │    ╱                    ╲
     │───╯                      ╲____
     │
     └───────────────────────────────► Steps
       Warm-up       Cosine Decay
```

## Warm-up

The learning rate begins with a linear warm-up covering approximately:

```text
3%
```

of the total training steps.

## Cosine Decay

After warm-up, the learning rate follows cosine decay until reaching:

```text
1%
```

## of the original learning rate.

# Early Stopping

Tensor Loom keeps the best validation checkpoint during training.

When validation loss stops improving, a stale-epoch counter is incremented.

Training stops after the configured patience threshold is reached.

The patience is derived from the number of configured epochs and is bounded between:

```text
10
```

and:

```text
100
```

epochs.

The final exported model is always the **best validation checkpoint**, rather than simply the model from the final epoch.

---

# Training Events

The engine communicates training progress through a Rust `Sender<TrainEvent>`.

Events include:

```text
Log
EpochStarted
Network
BatchCompleted
EpochCompleted
TrainingFinished
```

This allows a frontend or application layer to monitor training without directly controlling the training loop.

Example event flow:

```text
Training Started
      │
      ▼
EpochStarted
      │
      ▼
BatchCompleted
      │
      ▼
Network Snapshot
      │
      ▼
EpochCompleted
      │
      ▼
       ...
      │
      ▼
TrainingFinished
```

---

# Network Visualization Snapshots

Tensor Loom can produce `NetworkSnapshot` data for visualization.

A snapshot contains:

```text
epoch
layer_sizes
shown_sizes
node_activity
edge_blocks
```

Each edge block contains:

```text
rows
cols
weights
```

The weight layout follows row-major indexing:

```text
weights[row * columns + column]
```

This matches the frontend representation used for displaying network connections.

Unlike a sampled visualization, the current implementation sends the complete network:

```text
shown_sizes = layer_sizes
```

and includes every node and weight.

Snapshots are throttled to approximately 20 per training run, while always including the first and final epochs.

---

# Model Export

After training, Tensor Loom exports the best model as a `PortableModel`.

Each layer is represented using:

```text
PrimitiveLayer
```

containing:

```text
weights
bias
shape
activation
```

The layer shape is:

```text
(input_features, output_features)
```

The exported model also contains metadata describing the model type and configuration.

---

# Portable Model

A typical exported model follows this structure:

```text
PortableModel
│
├── layers
│   ├── PrimitiveLayer
│   │   ├── weights
│   │   ├── bias
│   │   ├── shape
│   │   └── activation
│   │
│   ├── PrimitiveLayer
│   └── ...
│
└── metadata
```

For classification models, the metadata contains:

```text
TensorLoom Engine v1.1
task=classification
classes=<number>
value_scale=<scale>
```

Regression models use:

```text
TensorLoom Engine v1.1 Cross-Platform
```

The metadata format is intended to remain compatible with the model inference layer.

---

# Normalization Folding

Regression models are trained using standardized values, but exported models need to accept raw values.

Tensor Loom therefore folds the normalization transformations directly into the model's first and final layers.

```text
Raw Input
   │
   ▼
Exported Layer
   │
   ▼
Neural Network
   │
   ▼
Raw Output
```

This means an exported regression model does not need a separate normalization configuration to reproduce the trained model's behavior.

The transformation is mathematically folded into the first layer's weights and biases and, when applicable, the final linear layer.

---

# Training Pipeline

The complete training pipeline can be summarized as:

```text
                 CSV Dataset
                      │
                      ▼
                Parse CSV Data
                      │
                      ▼
              Validate Dimensions
                      │
                      ▼
             Shuffle Deterministically
                      │
                      ▼
               Train / Validation
                    Split
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        Regression        Classification
             │                 │
        Standardize       One-Hot Encode
             │                 │
             └────────┬────────┘
                      ▼
                 Build Network
                      │
                      ▼
                Initialize Weights
                      │
                      ▼
                  AdamW Training
                      │
                      ▼
               Gradient Clipping
                      │
                      ▼
              Learning Rate Schedule
                      │
                      ▼
                  Validation
                      │
                      ▼
               Best Checkpoint
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
          Snapshot          Metrics
              │                │
              └───────┬────────┘
                      ▼
                Early Stopping
                      │
                      ▼
               Export Best Model
                      │
                      ▼
                PortableModel
```

---

# Configuration Requirements

The training engine expects a `TrainConfig` containing information for:

```text
epochs
batch_size
learning rate
layer configuration
hardware configuration
task configuration
```

The layer configuration provides:

```text
in_features
out_features
hidden_layers
output_activation
```

The hardware configuration determines:

```text
target_mode
device_index
```

The task configuration determines:

```text
loss
classes
value_scale
```

---

# Example Network

A regression network could be configured as:

```text
Input: 8
Hidden: 32 ReLU
Hidden: 16 ReLU
Output: 2 Linear
```

Resulting topology:

```text
8 → 32 → 16 → 2
```

A classification network could instead transform its input and output dimensions according to the configured number of classes:

```text
Input:
features × (classes + 1)

Output:
outputs × classes
```

The classification output consists of raw logits that are passed to cross-entropy loss.

---

# Design Goals

Tensor Loom's training engine is designed around several principles:

### Cross-platform execution

The same training pipeline can run on:

```text
CPU
GPU
```

using Burn's backend abstraction.

### Deterministic experiments

Dataset shuffling uses a fixed seed:

```text
42
```

making training runs reproducible.

### Hardware-aware training

The application can select between discrete GPUs, integrated GPUs, and CPU execution.

### Frontend-friendly training

Training communicates through events rather than coupling the training loop directly to a user interface.

### Portable inference

The trained model is exported into a lightweight representation that can be consumed independently by the inference layer.

### Stable training

Tensor Loom combines:

```text
AdamW
+ Gradient Clipping
+ Learning Rate Warm-up
+ Cosine Decay
+ Validation
+ Early Stopping
```

to provide a more controlled training process.

---

# Error Handling

The training pipeline validates several conditions before and during training.

Examples include:

```text
Empty CSV dataset
Invalid CSV column count
NaN or infinite values
Invalid activation function
Zero input features
Zero output features
Zero epochs
Invalid loss function
Invalid classification class count
Invalid classification value scale
Invalid classification output activation
Zero-unit hidden layer
```

Errors are returned as Rust `Result<_, String>` values rather than causing the training process to panic.

---

# Technology Stack

| Component           | Technology                |
| ------------------- | ------------------------- |
| Language            | Rust                      |
| ML Framework        | Burn                      |
| CPU Backend         | NdArray                   |
| GPU Backend         | WGPU                      |
| GPU API             | WGPU                      |
| Optimizer           | AdamW                     |
| Regression Loss     | MSE                       |
| Classification Loss | Cross-Entropy             |
| Data Format         | CSV                       |
| Model Format        | Portable Rust structures  |
| Communication       | `std::sync::mpsc::Sender` |

---

# Current Engine Version

The exported model identifies itself as:

```text
TensorLoom Engine v1.1
```

Classification metadata additionally stores the class count and value scale.

---

# Project Status

Tensor Loom currently contains a functional neural-network training pipeline supporting:

* Dynamic dense networks
* Regression
* Grouped classification
* CPU execution
* GPU execution
* Validation
* Early stopping
* Training visualization snapshots
* Portable model export

The training engine is designed to serve as the computational core of the broader Tensor Loom application.

---

# License

See the repository's `LICENSE` file for licensing information.

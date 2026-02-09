# Rental Churn Prediction System

## Overview

This repository implements a deep learning framework for predicting rental payment behavior. The system models churn prediction as a temporal sequence regression task, predicting `target_days_to_payment` for rental invoices based on historical payment patterns and invoice metadata.

## Problem Formulation

The system addresses the prediction of payment delays in rental contracts. Given a sequence of historical invoices for a user and the metadata of the current invoice, the model predicts the number of days until payment. This formulation enables:

- Identification of payment delay patterns across rental contracts
- Temporal modeling of user payment behavior
- Risk assessment for rental payment defaults

## Architecture

The model employs a hierarchical transformer architecture with the following components:

### 1. Feature Tokenization
- **Fourier Features**: Continuous variables are embedded using random Fourier features to address spectral bias, computed as:
  ```
  γ(v) = concat[sin(2πBv), cos(2πBv)]
  ```
  where B ∈ ℝ^(D/2) is a random frequency matrix.

- **Categorical Embeddings**: Learned embedding tables for discrete features.

### 2. Invoice Encoder
- Single-layer transformer block with multi-head self-attention
- SwiGLU activation in feed-forward network
- Mean pooling to produce invoice-level representations

### 3. Sequence Encoder
- Three-layer causal transformer with masked self-attention
- Rotary Positional Embeddings (RoPE) for relative position encoding
- Models temporal dependencies across invoice sequences

### 4. Temporal Cross-Attention
- Dynamically links current invoice features with historical patterns
- Gated Residual Network (GRN) for controlled information flow

### 5. Prediction Head
- Two-layer GRN with bottleneck architecture
- Regression output for continuous day prediction

## Technical Components

### Rotary Positional Embeddings (RoPE)
Position information is encoded through rotation in the complex plane:

```
[x'₁]   [cos(mθ)  -sin(mθ)] [x₁]
[x'₂] = [sin(mθ)   cos(mθ)] [x₂]
```

where θᵢ = 10000^(-2i/d), ensuring relative position dependency in attention scores.

### Gated Residual Networks
Information flow is controlled through learned gating mechanisms:

```
η₁ = ELU(W₁x + b₁)
η₂ = W₂η₁ + b₂
gate = σ(W₃η₂ + b₃)
out = LayerNorm(gate ⊙ η₂ + (1 - gate) ⊙ Wₛx)
```

### SwiGLU Activation
The feed-forward network uses SwiGLU activation:

```
SwiGLU(x) = (xWG) ⊙ SiLU(xW₁) · W₂
```

where SiLU(x) = x · σ(x).

## Model Specifications

- **Total Parameters**: 5,943,042
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Sequence Encoder Layers**: 3
- **Invoice Encoder Layers**: 1

## Installation

```bash
# Create conda environment
conda create -n rental-churn python=3.9
conda activate rental-churn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m main.train
```

Configuration parameters are specified in `main/config.py`:
- Batch size
- Learning rate schedule
- Model dimensions
- Training epochs

### Inference

```bash
python -m main.infer --checkpoint <path_to_checkpoint>
```

### API Deployment

```bash
python -m main.api
```

Launches a REST API for real-time prediction requests.

### Ablation Studies

```bash
python -m main.ablate
```

Executes systematic ablation experiments to assess component contributions.

## Data Structure

The input data (`raw_data`) consolidates financial events at the invoice level. Key features include:

- **Temporal**: `vencimentoData`, `pagamentoData`, `Dias_atraso`
- **Financial**: `valor_brl`, `valor_pago_brl`, `valor_recebiveis`
- **Categorical**: `situacao`, `formaPagamento`, `parcelaTipo`
- **Engineered**: Rolling statistics, lag features, cyclical time encodings

## Output Format

Training produces:
- `checkpoint.pt`: Full model checkpoint with optimizer state
- `model_state_dict.pt`: Model weights only
- `model_metadata.pt`: Feature encoders and preprocessing parameters
- `plots/`: Training curves and validation metrics

Inference generates:
- `inference_results.csv`: Predictions with associated metadata
- Performance metrics (MAE, RMSE, quantile losses)

## Project Structure

```
rental-churn/
├── core/                 # Core model components
│   ├── model.py         # Architecture implementation
│   ├── dataset.py       # Data loading and batching
│   ├── trainer.py       # Training loop
│   ├── inference.py     # Inference pipeline
│   └── data.py          # Preprocessing utilities
├── main/                # Entry points
│   ├── train.py         # Training script
│   ├── infer.py         # Inference script
│   ├── api.py           # REST API
│   └── config.py        # Configuration
├── runs/                # Experiment outputs
├── data/                # Data directory
└── notebooks/           # Analysis notebooks
```

## Results

Model checkpoints and evaluation metrics are stored in timestamped directories under `runs/`. Each run contains:

- Training and validation loss curves
- Prediction distribution analysis
- Error quantile breakdowns
- Attention weight visualizations

## Technical Details

### Sequence Processing
- Maximum sequence length: Configurable in training config
- Padding strategy: Right-padded with masking
- Batching: Variable-length sequences with attention masks

### Optimization
- Optimizer: AdamW with weight decay
- Learning rate schedule: Cosine annealing with warmup
- Loss function: Huber loss (smooth L1)

### Regularization
- Stochastic depth (drop path)
- Embedding dropout
- Attention dropout
- Label smoothing for quantile predictions

## Requirements

- PyTorch >= 2.0
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- fastapi (for API deployment)

See `requirements.txt` for complete dependency specifications.

## References

- **Fourier Features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2021
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", arXiv 2020

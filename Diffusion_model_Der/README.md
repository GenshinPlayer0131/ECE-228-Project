
# RF-Diffusion

## Overview

RF-Diffusion is a supervised 1D U-Net denoising model designed to reconstruct clean RF signals from noisy observations. The model learns a direct mapping from noisy RF measurements to clean reference signals using Mean Squared Error (MSE) regression.

The goal is to improve signal quality while preserving the temporal structure of RFID measurements.

---

## Model Architecture

The model uses a one-dimensional U-Net architecture consisting of:

- Encoder
  - 1D Convolution Layers
  - Downsampling Operations

- Bottleneck
  - Latent Feature Representation

- Decoder
  - Transposed Convolutions
  - Signal Reconstruction

- Skip Connections
  - Preserve low-level signal information
  - Improve reconstruction quality

Input:

```text
Noisy RF Signal Window
Shape: (1, 64)
```

Output:

```text
Denoised RF Signal Window
Shape: (1, 64)
```

---

## Data Preprocessing

Before training:

1. Normalize clean and noisy signals
2. Create overlapping windows
3. Window length = 64
4. Stride = 32
5. Split into training and evaluation samples

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Batch Size | 16 |
| Epochs | 100 |
| Window Length | 64 |
| Stride | 32 |
| Loss Function | MSE |

---

## Evaluation Metrics

The model is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Normalized Mean Squared Error (NMSE)
- Pearson Correlation Coefficient

---

## Results

| Metric | Noisy Signal | Denoised Signal |
|----------|----------|----------|
| MSE | 0.314707 | 0.021180 |
| RMSE | 0.560987 | 0.145532 |
| NMSE | 0.552624 | 0.037191 |
| Correlation | 0.782365 | 0.977219 |

### Improvement

- MSE Reduction: 93.27%
- NMSE Reduction: 93.27%
- RMSE Reduction: 73.98%
- Correlation Gain: +0.194854

---

## Output

The evaluation script generates:

- Denoised signal predictions
- Reconstruction plots
- MSE comparison plots
- Training loss curves
- Performance metrics

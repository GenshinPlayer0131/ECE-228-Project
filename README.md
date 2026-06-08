# ECE-228 Project

## RF Signal Denoising Using Deep Learning and Diffusion Models

### Team Members

- Derrick Eliasi
- Yizhou Fang
- Alan Aquino
- Tianxing Fan

---

## Project Overview

RFID localization systems rely on radio-frequency (RF) measurements such as phase angle, RSSI, and Doppler frequency. These measurements are often corrupted by multipath propagation, environmental interference, and hardware imperfections, introducing noise that degrades localization performance.

The objective of this project is to investigate multiple denoising approaches capable of reconstructing clean RF signals from noisy observations. We evaluate several diffusion-based architectures alongside a supervised deep learning baseline and compare their effectiveness using standard signal reconstruction metrics.

---

## Dataset

The dataset consists of paired clean and noisy RF signal measurements. All data was collected from a previous project Yizhou and Derrick had done.

---

## Methods Evaluated

### S1 – Denoise Diffusion (RFFI)

A DDPM-based denoiser trained on one-dimensional RF phase windows. The model performs noise prediction using a 1D U-Net backbone and utilizes DDIM step-skipping during inference for accelerated denoising.

### S2 – Conditional Denoising Diffusion Model (CDDM)

A conditional 1D DDPM architecture where noisy RF measurements are provided as conditioning information. A residual convolutional network predicts diffusion noise throughout the reverse diffusion process.

### S3 – Conditional Score-Based Diffusion for Imputation (CSDI)

A conditional diffusion framework utilizing residual convolution blocks and sinusoidal timestep embeddings. The noisy signal is incorporated as a conditioning channel while the model learns noise prediction through the DDPM process.

### S4 – RF Diffusion

A supervised one-dimensional U-Net denoising network that directly maps noisy RF signals to clean RF signals using Mean Squared Error (MSE) regression. The model employs an encoder-decoder architecture with skip connections and performs deterministic single-pass denoising.

### Baseline

- Wiener Filter

---

## Evaluation Metrics

The methods are evaluated using:

- Mean Squared Error (MSE)
- Normalized Mean Squared Error (NMSE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation Coefficient

---

## Results Summary

### Method Comparison

| Method | MSE Reduction | NMSE Reduction | RMSE Reduction | Correlation Gain |
|----------|----------|----------|----------|----------|
| Conditional 1D Diffusion | 99.68% | 99.68% | 94.38% | +0.151 |
| CDDM-style Denoiser | 98.56% | 98.56% | 87.98% | +0.148 |
| RF Diffusion | 93.27% | 93.27% | 73.98% | +0.195 |
| Denoise Diffusion (RFFI) | 81.31% | 81.50% | 56.67% | +0.017 |
| Wiener Filter (Baseline) | 71.50% | 71.43% | 46.62% | +0.014 |

---

## Key Findings

- All diffusion-based and deep learning approaches significantly outperformed the Wiener filter baseline.
- Conditional 1D Diffusion achieved the largest reduction in reconstruction error, reducing MSE by 99.68%.
- CDDM achieved similar performance, producing over 98% reduction in both MSE and NMSE.
- RF Diffusion achieved a 93.27% reduction in MSE while producing the largest correlation improvement among all methods.
- All successful implementations increased correlation coefficients to approximately 0.98–1.00, demonstrating strong preservation of the underlying RF signal structure.
- The results indicate that denoised RF measurements can provide more reliable inputs for downstream RFID localization and sensing applications.

---

## Repository Structure

```text
ECE-228-Project
│
├── Diffusion_Model_Fan/
│   └── Denoise Diffusion (RFFI) implementation
│
├── Diffusion_model_Alan/
│   └── CDDM implementation
│
├── Diffusion_model_Der/
│   └── RF Diffusion implementation
│
├── Diffusion_model_yzf/
│   └── Conditional 1D Diffusion / CSDI implementation
│
├── data/
│   └── RF signal datasets
│
├── README.md
│
└── .DS_Store
```

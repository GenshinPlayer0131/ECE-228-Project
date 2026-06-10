# Conditional Denoising Diffusion Model (CDDM)

## Overview

This code implements a Conditional Denoising Diffusion Model (CDDM) for RF signal denoising. The model takes a noisy RF signal as the input and reconstructs the denoised clean signal and then calculate the metrics.


## Required Packages

Install the required Python packages:

```bash
pip install numpy pandas matplotlib torch
```

## How to Run

Please use the following command:
```bash
python train_rf_cddm_1d.py --data_dir <directory containing input data> --output_dir <output directory>
```


## Evaluation Metrics

The code calculates the following as the metrics: (all between the reconstructed denoised signal and the ground truth)

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Normalized Mean Squared Error (NMSE)
- Pearson correlation coefficient

---

## Results

| Metric | Noisy Signal | Denoised Signal |
|----------|----------|----------|
| MSE | 11.304969 | 0.163194 |
| RMSE | 1.074802 | 0.015515 |
| NMSE | 3.362286 | 0.403973 |
| Correlation | 0.848676 | 0.997393 |

### Improvement

- MSE Reduction: 98.56%
- NMSE Reduction: 98.56%
- RMSE Reduction: 87.99%
- Correlation Gain: +0.149

---

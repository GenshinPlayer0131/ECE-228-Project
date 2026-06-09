# Conditional 1D DDPM RF Signal Denoising

This code implements a conditional 1D diffusion model for RF/RFID signal denoising. The model uses a noisy RF signal as the condition and learns to reconstruct the clean reference signal through a DDPM-style denoising process.

## Code Purpose

The goal of this code is to remove noise from a 1D RF signal and evaluate the denoising performance.

The model takes:

```text
Input condition: noisy RF signal
Training target: clean RF signal
Output: denoised RF signal
```

The code reports:

```text
MSE
NMSE
RMSE
Correlation Coefficient
MSE Reduction (%)
NMSE Reduction (%)
RMSE Reduction (%)
Correlation Gain
Correlation Gain (%)
```

## Dataset Format

The input dataset should be a CSV file with three columns:

```csv
time,clean,noisy
```

Example:

```csv
time,clean,noisy
0.000,4.52,7.10
0.010,4.55,3.92
0.020,4.58,6.44
```

Column meaning:

| Column  | Description               |
| ------- | ------------------------- |
| `time`  | Time index or timestamp   |
| `clean` | Clean reference RF signal |
| `noisy` | Noisy measured RF signal  |

## Required Packages

Install the required Python packages:

```bash
pip install numpy pandas matplotlib torch
```

## How to Run

Put the dataset file in the same folder as the Python code.

Then run:

```bash
python diffusion_model_yzf.py
```

In the code, set the CSV path:

```python
CSV_PATH = "sample_0.csv"
```

If the file is in a subfolder, use:

```python
CSV_PATH = "Diffusion_model/sample_0.csv"
```

On Windows, avoid using normal backslashes because they can cause path errors. Use one of these formats:

```python
CSV_PATH = r"E:\UCSD\Spring2026\ECE 228\project\Diffusion_model\sample_0.csv"
```

or:

```python
CSV_PATH = "E:/UCSD/Spring2026/ECE 228/project/Diffusion_model/sample_0.csv"
```

## Main Code Structure

### 1. Load Data

The code reads the CSV file:

```python
df = pd.read_csv(CSV_PATH)

time = df["time"].values.astype(np.float32)
clean = df["clean"].values.astype(np.float32)
noisy = df["noisy"].values.astype(np.float32)
```

The clean signal is used as the ground truth, and the noisy signal is used as the conditional input.

### 2. Normalize Data

The code normalizes both clean and noisy signals using the mean and standard deviation of the noisy signal:

```python
mean_val = noisy.mean()
std_val = noisy.std() + 1e-8

clean_norm = (clean - mean_val) / std_val
noisy_norm = (noisy - mean_val) / std_val
```

Normalization helps the neural network train more stably.

### 3. Create Sliding Windows

The signal is divided into overlapping windows:

```python
SEQ_LEN = 64
```

Each training sample contains:

```text
x0   = clean signal window
cond = noisy signal window
```

The noisy window is used as the condition for the diffusion model.

### 4. Forward Diffusion

The code adds Gaussian noise to the clean signal window:

```python
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
```

This is implemented by:

```python
x_t = q_sample(x0, t, noise)
```

The model learns to predict the added noise.

### 5. Conditional Diffusion Model

The neural network input is:

```text
x_t
timestep t
noisy signal condition
```

The noised signal and noisy condition are concatenated:

```python
x = torch.cat([x_t, cond], dim=1)
```

The model uses:

```text
1D convolution layers
residual blocks
sinusoidal timestep embedding
MSE noise prediction loss
```

The model output is the predicted noise:

```python
predicted_noise = model(x_t, t, cond)
```

### 6. Training Loss

The training objective is:

```python
loss = F.mse_loss(predicted_noise, noise)
```

This trains the model to predict the Gaussian noise added during diffusion.

### 7. Reverse Denoising

During inference, the code starts from random noise and applies the learned reverse diffusion process:

```python
for step in reversed(range(T)):
    x_t = p_sample(model, x_t, t, cond)
```

The noisy RF signal is used as the condition during every reverse step.

### 8. Full Signal Reconstruction

Since the model denoises short windows, the full signal is reconstructed by averaging overlapping denoised windows:

```python
denoised_norm = output_sum / (output_count + 1e-8)
```

The final denoised signal is converted back to the original scale:

```python
denoised = denoised_norm * std_val + mean_val
```

## Important Hyperparameters

The main parameters are:

```python
SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 800
LR = 1e-3

T = 200
BETA_START = 1e-4
BETA_END = 0.02
```

| Parameter    | Meaning                              |
| ------------ | ------------------------------------ |
| `SEQ_LEN`    | Length of each signal window         |
| `BATCH_SIZE` | Number of windows per training batch |
| `EPOCHS`     | Number of training epochs            |
| `LR`         | Learning rate                        |
| `T`          | Number of diffusion steps            |
| `BETA_START` | Starting noise schedule value        |
| `BETA_END`   | Ending noise schedule value          |

## Evaluation Metrics

The code compares:

```text
clean vs noisy
clean vs denoised
```

### MSE

```python
mse = np.mean((y_true - y_pred) ** 2)
```

### NMSE

```python
nmse = np.sum((y_true - y_pred) ** 2) / (np.sum(y_true ** 2) + 1e-8)
```

### RMSE

```python
rmse = np.sqrt(mse)
```

### Correlation Coefficient

```python
corr = np.corrcoef(y_true, y_pred)[0, 1]
```

### MSE Reduction

```python
mse_reduction = (mse_noisy - mse_denoised) / (mse_noisy + 1e-8) * 100
```

### Correlation Gain

```python
corr_gain = corr_denoised - corr_noisy
```

## Example Output

Example result from the code:

```text
================ Metrics ================
Noisy Signal:
MSE  = 11.304970
NMSE = 1.074802
RMSE = 3.362286
Corr = 0.848676

Denoised Signal:
MSE  = 0.035700
NMSE = 0.003394
RMSE = 0.188945
Corr = 0.999699

================ Improvement ================
MSE Reduction  = 99.68%
NMSE Reduction = 99.68%
RMSE Reduction = 94.38%
Correlation Gain = 0.151023
Correlation Gain (%) = 17.80%
```

## Generated Plots

The code generates two figures.

### 1. Denoising Result

This plot shows:

```text
clean signal
noisy signal
denoised signal
```

It is used to visually compare the denoised output with the clean reference.

### 2. Training Loss

This plot shows the diffusion model training loss over epochs.

A decreasing loss indicates that the model is learning to predict the added diffusion noise.

## Notes

This code is designed for a simple supervised denoising setting where both clean and noisy signals are available.

For a larger RFID localization dataset, the same structure can be extended to multiple input features, such as:

```text
RSSI
phase
antenna ID
tag ID
frequency
timestamp
```

For the current implementation, the method should be described as:

```text
Conditional 1D DDPM Denoiser
```

It should not be called pure CSDI or full CDDM. It is a DDPM-based conditional denoising model for 1D RF time-series signals.

## **ECE 228 Final Project — Diffusion Denoising for RFID Phase**
## **Alan Aquino A17452171**
---

### Overview

We train a diffusion model to learn the prior over clean RFID phase windows. At inference we'll use that prior to denoise noisy phase from real RFID captures and then run motion analysis (velocity from phase slope) on the cleaned signal.

The architecture follows Yin et al., *"Noise-Robust Radio Frequency Fingerprint Identification Using Denoise Diffusion Model"* (arXiv:2503.05514), but with a few changes for our project.

---

### How faithful is this to the paper?

| Paper | Our code | Match? |
|---|---|---|
| Forward process (Eq. 2): `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` | Same | Yes |
| Linear β schedule, `β` from `1e-5` to `1.5e-3`, `T=1000` | Same | Yes |
| ε-prediction loss (Eq. 4): MSE between ε and ε_θ | Same | Yes |
| Adamax, lr `1e-4`, batch 32 | Same | Yes |
| Hierarchical Diffusion Transformer (HDT) backbone | 1D U-Net | No — see below |
| SNR Mapping (Eq. 7) + DDIM step-skipping (Eq. 6) at inference | Same | Yes |
| DDIM reverse (Eq. 5), deterministic | Same | Yes |
| Downstream: device classification | Downstream: motion analysis | Different task |
| Classical baselines for comparison | Added (moving avg, Wiener, wavelet) | Beyond paper |

The training half mirrors the paper closely. The architecture and downstream task are where we diverge.

---

### What we changed and why

**HDT → 1D U-Net.** The paper's HDT is built for complex-valued IQ (In-phase and Quadrature) signals from a Wi-Fi preamble. Our data is 1D real unwrapped phase (256 samples). A 1D U-Net is the standard choice for that shape, much smaller, and trains fine on a single GPU.

**Classification → motion analysis.** The paper feeds denoised signals into a Transformer classifier to identify RFID dongles. We are not intrested in device identity we are intresrted in recovering phase trajectories so we can compute `v = (λ / 4π) · dφ/dt` accurately. So no classifier in our pipeline.

**Noise model.** The paper trains under AWGN on Wi-Fi preambles. Our noisy version has Gaussian + multipath-like sin + local spikes + bursts (see `data_reading/noise.py`) — closer to what real RFID phase actually looks like in a multipath room.

**Classical baselines.** The paper compares against itself (with vs. without denoising). We compare against three classical denoisers: moving average, Wiener filter, and wavelet thresholding. This gives an honest answer to "is diffusion actually better than what we could've done with `scipy.signal`?"

---

### File layout

```
dm_helpers.py   # config, dataset loader, schedule builder, sanity check, shape test
dm_classes.py   # Dataset wrapper + U-Net + building blocks
dm_main.py      # training loop + script entry point
inference.py    # SNR mapping + DDIM denoising 
baselines.py    # moving average / Wiener / wavelet (classical comparisons)
evaluate.py     # runs all 4 methods on the dataset and plots NMSE vs. SNR
test_one.py     # quick single-window test (2high/sample_0.csv) 
data_reading/
  noise.py      # generates (clean, noisy) window pairs from raw RFID CSVs
rfid_diffusion_dataset/   # generated training data (one CSV per window)
```

---

### Classes

**`RFPhaseDataset`** (`dm_classes.py`)
PyTorch Dataset wrapping the loaded windows. `__getitem__` returns the clean window (shape `(1, 256)`) for diffusion training. Noisy + time arrays are kept as `.noisy_windows` and `.times` for later use at inference / evaluation.

**`SinusoidalEmbedding`** (`dm_classes.py`)
Standard transformer-style sinusoidal embedding for the timestep `t`. Maps an integer step in `[0, T)` to a vector of length `time_dim=128` so the U-Net knows how noisy the input is.

**`ResBlock1D`** (`dm_classes.py`)
The basic building block of the U-Net. GroupNorm → SiLU → Conv1d → add timestep embedding → GroupNorm → SiLU → Conv1d, with a residual skip. Same pattern as Ho et al.'s original DDPM but with 1D convs.

**`UNet1D`** (`dm_classes.py`)
The noise predictor ε_θ. Takes `(B, 1, 256)` noisy window + timestep `t`, returns `(B, 1, 256)` predicted noise.
- Encoder: 3 ResBlocks at channels `32 → 64 → 128`, downsampled with avg pool.
- Bottleneck: 1 ResBlock at 128 channels, length 64.
- Decoder: mirrors encoder, with skip connections concatenated from each encoder level.
- ~280k parameters. Small enough for CPU training, fast on any GPU.

---

### Helper functions

**`load_paired_windows`** (`dm_helpers.py`)
Walks `rfid_diffusion_dataset/` recursively, loads every `sample_*.csv`, takes the `clean` and `noisy` columns. Normalizes each window by the clean window's mean and std so all windows live on the same scale. Skips degenerate windows where `std < 0.3` (over-smoothed flat lines). Returns `(clean, noisy, time, sources)` arrays.

**`sanity_check`** (`dm_helpers.py`)
Prints dataset stats (total count, mean, std, source breakdown) and plots 6 random windows showing clean (blue) on top of noisy (red). Warns if total windows is under 500. This is what tells us the data is healthy.

**`make_schedule`** (`dm_helpers.py`)
Builds the noise schedule once and returns a dict of tensors:
- `betas`: linear from `1e-5` to `1.5e-3`, length `T`
- `alphas`, `alpha_bars`: derived from betas
- `sqrt_alpha_bars`, `sqrt_one_minus_alpha_bars`: precomputed for fast use in the forward process

**`quick_shape_test`** (`dm_helpers.py`)
Runs one fake forward pass through the model to confirm the output shape matches the input shape. Catches U-Net bugs before they show up 5 minutes into training.

**`train`** (`dm_main.py`)
Standard DDPM training loop. For each batch of clean windows `x_0`:
1. Sample random timestep `t ∈ [0, T)`
2. Sample noise `ε ~ N(0, I)`
3. Build `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`  (Eq. 2)
4. Predict noise: `ε_θ = model(x_t, t)`
5. Loss: `MSE(ε_θ, ε)`  (Eq. 4)
6. Backward + optimizer step (with mixed precision on GPU)

Saves the model + schedule to `diffusion_ckpt.pt` and a loss curve to `training_loss.png`.

---

### Inference (SNR mapping + DDIM)

**`snr_map_db`** (`inference.py`)
Precomputes `γ_t` (SNR at every timestep) from the schedule, in dB. This is the lookup table that SNR mapping uses.

**`true_snr_db`** / **`estimate_snr_db`** (`inference.py`)
Two ways to measure input SNR. `true_snr_db(clean, noisy)` is exact (used in evaluation since we have paired data). `estimate_snr_db(noisy)` works from noisy alone via a small smoothing trick (for real deployment).

**`find_t_star`** (`inference.py`)
Paper Eq. 7: `t* = argmin_t |γ_map(t) − γ_input|`. One-line argmin on the SNR map. This is the SNR-mapping trick.

**`make_ddim_timesteps`** (`inference.py`)
Paper Eq. 6: build the step-skipping schedule `t* → 0` in `n_steps` stops. Lets us run ~20 reverse steps instead of all 1000.

**`ddim_denoise`** (`inference.py`)
Paper Eq. 5: deterministic DDIM reverse process from `t*` down to 0. For each step, predicts `ε_θ`, recovers a guess of `x_0`, then projects forward to the next step. Output is the denoised window.

**`denoise`** (`inference.py`)
The high-level wrapper. Pass it `(noisy_window, model, schedule)` → get back denoised window + the chosen `t*`.

---

### Baselines

**`moving_average`** (`baselines.py`)
Uniform 9-sample low-pass via `np.convolve`. Cheap, smears spikes.

**`wiener`** (`baselines.py`)
`scipy.signal.wiener` with window=11. Adaptive low-pass, smooths more in low-variance regions. Strongest classical baseline for stationary Gaussian noise.

**`wavelet_threshold`** (`baselines.py`)
`db4` wavelet decomposition + Donoho's universal soft threshold + inverse transform. Preserves sharp features better than the other two; best classical method for non-stationary noise.

---

### Evaluation

**`evaluate`** (`evaluate.py`)
For every window in the dataset: runs `raw` (no denoising), all 3 baselines, and diffusion+SNR-mapping. Records NMSE, input SNR, and source group. Prints a mean-NMSE-per-method summary, a per-source-group breakdown, and saves two plots:
- `evaluation.png` — NMSE vs. input SNR, one line per method.
- `denoise_examples.png` — 3 random windows with noisy / wiener / diffusion / clean overlaid.

**`test_one.py`**
Single-window test on `2high/sample_0.csv`. Prints true vs. estimated SNR, the chosen `t*`, NMSE per method, and saves `test_one.png` with one subplot per denoiser. Use this for fast iteration before kicking off the full `evaluate.py` sweep.

---

### Overfitting: what we guard against, and what we don't

We hold out a fixed 15% of windows (`split_indices` in `dm_helpers.py`, seeded so it's identical every run). The model trains only on the other 85%; `dm_main.py` reports a held-out **validation loss** next to the train loss each epoch, and `evaluate.py` scores every method **only on the held-out split** — so the NMSE numbers are out-of-sample, not memorized.

That split is the point: it lets us *measure* overfitting instead of just asserting it isn't happening. The two loss curves in `training_loss.png` tracking each other is the evidence.

What we deliberately **don't** add: dropout, weight decay, or early stopping. The diffusion objective is already a strong regularizer — every step draws a fresh timestep `t` and fresh noise `ε`, so a single clean window is never seen as the same training example twice (effectively unlimited augmentation). Paired with a deliberately small model (~280k params), that's enough: the train/val gap stays flat for all 20 epochs (`training_loss.png`), so there's nothing for early stopping to catch.

The paper (Yin et al., arXiv:2503.05514, Sec. V-A) *does* use validation-loss-driven LR scheduling and early stopping — they halve the LR after 20 epochs without val-loss improvement and stop after 30. That makes sense for their setup: an HDT backbone is far larger than our U-Net and they train on 30,000 packets per device, so they have both the capacity to overfit and a long enough schedule that an automatic stop matters. At our scale (280k params, ~530 train windows, 20 epochs) the curves never diverge, so we monitor the val loss but don't need the machinery. If the val curve ever pulled away from train we'd add `weight_decay` first — but spending regularization we don't need just trades overfitting for underfitting.

---

### Results

![Sanity check](Result%20Graphs/sanity_check.png)

![Training loss](Result%20Graphs/training_loss.png)

Train and held-out validation loss fall together and stay overlapped for all 20 epochs — no gap opening up, which is what "not overfitting" looks like (see the section above).

![Single-window test](Result%20Graphs/test_one.png)

The single-window test (`2high/sample_0.csv`, input SNR +13.7 dB) is the per-method ranking in miniature: raw NMSE `0.0427`, moving avg `0.0123`, Wiener `0.0121`, wavelet `0.0090`, diffusion `0.0070` at `t*=230`. Diffusion tracks the clean curve through the dip and the sharp recovery without the residual ripple the classical filters leave behind.

![NMSE vs. SNR](Result%20Graphs/evaluation.png)

Swept across the 94 held-out windows (out-of-sample — none seen in training), diffusion wins at almost every SNR. It's the clear leader from ~5 dB up — roughly 3–4× lower NMSE than raw and a solid margin below wavelet, the best classical method. The one exception is the lowest bin (~−2.5 dB): when the input is that noisy, all three classical denoisers actually beat diffusion, and diffusion barely improves on raw. So diffusion is the better choice in the regime we care about, but it isn't a free lunch at very low SNR.

**Oracle vs. blind SNR.** SNR mapping needs an input SNR to pick the starting step `t*`, and where that number comes from matters (see below). We plot two diffusion curves: **oracle SNR** (solid) uses the true SNR computed from the clean window, and **blind SNR** (dashed) uses `estimate_snr_db`, which works from the noisy signal alone — the deployable case. They sit almost on top of each other (median NMSE 0.019 vs. 0.021). Because the SNR→`t*` map is monotonic, a few-dB error in the estimate only nudges `t*` by a handful of steps, so the blind estimator costs us almost nothing. That's the result that says this would actually work in deployment, not just with ground truth.

#### Where the SNR comes from

In our pipeline `t*` is chosen by `find_t_star` from either the true or estimated SNR (the two curves above). It's worth being precise about what the paper does, because it differs from a fully blind deployment:

- The paper defines `t* = argmin_t |γ_map(t) − γ|`, where **γ is "the SNR of a signal input to the noise predictor"** (Yin et al., arXiv:2503.05514, Eq. 7, Sec. IV-B). At inference (Sec. II-B) the received packet is "extracted along with the SNR value of the current signal" — i.e. γ is assumed to be measured at the receiver; no blind-estimation algorithm is given.
- In their experiments γ is in fact **known by construction**: clean packets are all captured above 40 dB, and low-SNR cases are produced by adding artificial Gaussian noise to a *target* SNR (Sec. V-A). So their reported numbers correspond to our **oracle** curve.
- Our **blind** curve (`estimate_snr_db`) goes a step beyond the paper by estimating γ from the noisy window alone, and shows the oracle assumption costs almost nothing here.

![Denoise examples](Result%20Graphs/denoise_examples.png)

---

### Pipeline

```
[Raw RFID CSVs]
       ↓
   noise.py → (clean, noisy, time) windows
       ↓
[rfid_diffusion_dataset/]
       ↓
   load_paired_windows → RFPhaseDataset
       ↓
   sanity_check  (visual confirmation)
       ↓
   train  →  UNet1D learns ε_θ
       ↓
[diffusion_ckpt.pt]
       ↓
   inference.denoise:  estimate SNR  →  find t*  →  DDIM reverse from t* to 0
       ↓
   evaluate.py:  diffusion vs. moving_avg / wiener / wavelet → NMSE vs. SNR
       ↓
   (TODO) v = (λ / 4π) · dφ/dt   motion analysis on denoised phase
```

---

### How to run

Full pipeline in order:

```powershell
# 1. one-time setup
pip install torch numpy pandas matplotlib scipy PyWavelets

# 2. regenerate the (clean, noisy) windows from raw RFID CSVs
python data_reading\noise.py

# 3. sanity check + train the diffusion model
python dm_main.py

# 4. (optional) test SNR mapping + DDIM denoising on a few windows
python inference.py

# 5. (optional, fast) single-window comparison plot
python test_one.py

# 6. full comparison: diffusion vs. classical baselines on every window
python evaluate.py
```

What each step writes:

| Step | Outputs |
|---|---|
| `noise.py` | windows in `rfid_diffusion_dataset/<group>/sample_*.csv` |
| `dm_main.py` | `sanity_check.png`, `diffusion_ckpt.pt`, `training_loss.png` |
| `inference.py` | console: SNR map preview + denoising NMSE on 3 windows |
| `test_one.py` | `test_one.png` — one subplot per denoiser for 2high/sample_0.csv |
| `evaluate.py` | `evaluation.png` (NMSE vs. SNR per method), `denoise_examples.png`, console summary tables |

---

### Variable naming

Where it doesn't hurt readability, variables match the paper's symbols:

| Code | Paper |
|---|---|
| `x0`, `xt` | x_0, x_t |
| `epsilon` | ε |
| `epsilon_theta` | ε_θ |
| `betas`, `alphas`, `alpha_bars` | β, α, ᾱ |
| `sqrt_alpha_bar_t`, `sqrt_one_minus_alpha_bar_t` | √ᾱ_t, √(1−ᾱ_t) |
| `num_steps` | T |

So the training inner loop reads like the paper's Equations 2 and 4 directly.

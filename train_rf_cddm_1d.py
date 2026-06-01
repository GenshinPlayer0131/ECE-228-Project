from pathlib import Path
import argparse
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Metrics Functions
# ============================================================


def compute_signal_metrics(clean, estimate, eps=1e-12):
    clean = np.asarray(clean).reshape(-1)
    estimate = np.asarray(estimate).reshape(-1)

    error = estimate - clean

    mse = np.mean(error ** 2)

    nmse = np.sum(error ** 2) / (np.sum(clean ** 2) + eps)

    rmse = np.sqrt(mse)

    if np.std(clean) < eps or np.std(estimate) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(clean, estimate)[0, 1]

    return {
        "MSE": mse,
        "NMSE": nmse,
        "RMSE": rmse,
        "CorrCoef": corr,
    }


def compare_noisy_and_denoised(clean, noisy, denoised):
    noisy_metrics = compute_signal_metrics(clean, noisy)
    denoised_metrics = compute_signal_metrics(clean, denoised)

    rows = []

    for key in noisy_metrics.keys():
        noisy_val = noisy_metrics[key]
        denoised_val = denoised_metrics[key]

        if key in ["MSE", "NMSE", "RMSE"]:
            improvement_abs = noisy_val - denoised_val
    
            improvement_percent = 100.0 * (noisy_val - denoised_val) / (abs(noisy_val) + 1e-12)

        elif key == "CorrCoef":
            improvement_abs = denoised_val - noisy_val
            improvement_percent = 100.0 * (denoised_val - noisy_val) / (abs(noisy_val) + 1e-12)

        rows.append({
            "Metric": key,
            "Noisy": noisy_val,
            "Denoised": denoised_val,
            "Improvement": improvement_abs,
            "Improvement_%": improvement_percent,
        })

    return pd.DataFrame(rows)


# ============================================================
# 1. CSV loading
# ============================================================

def load_rf_csv(path: Path):
    df = pd.read_csv(path)

    if "clean" in df.columns and "noisy" in df.columns:
        clean = df["clean"].to_numpy(np.float32)
        noisy = df["noisy"].to_numpy(np.float32)
    else:
        clean = df.iloc[:, 1].to_numpy(np.float32)
        noisy = df.iloc[:, 2].to_numpy(np.float32)

    return clean, noisy


def make_windows(clean, noisy, window_len=128, stride=16):
    assert len(clean) == len(noisy)

    n = len(clean)

    if n < window_len:
        pad = window_len - n
        clean = np.pad(clean, (0, pad), mode="edge")
        noisy = np.pad(noisy, (0, pad), mode="edge")
        n = window_len

    clean_windows = []
    noisy_windows = []

    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        clean_windows.append(clean[start:end])
        noisy_windows.append(noisy[start:end])

    if clean_windows[-1].shape[0] != window_len or (n - window_len) % stride != 0:
        clean_windows.append(clean[-window_len:])
        noisy_windows.append(noisy[-window_len:])

    clean_windows = np.stack(clean_windows, axis=0)
    noisy_windows = np.stack(noisy_windows, axis=0)

    return clean_windows, noisy_windows


def compute_train_stats(csv_files, window_len, stride):
    all_clean = []
    all_noisy = []

    for p in csv_files:
        clean, noisy = load_rf_csv(p)
        cw, nw = make_windows(clean, noisy, window_len, stride)
        all_clean.append(cw.reshape(-1))
        all_noisy.append(nw.reshape(-1))

    all_clean = np.concatenate(all_clean)
    all_noisy = np.concatenate(all_noisy)

    stats = {
        "clean_mean": float(all_clean.mean()),
        "clean_std": float(all_clean.std() + 1e-8),
        "noisy_mean": float(all_noisy.mean()),
        "noisy_std": float(all_noisy.std() + 1e-8),
    }

    return stats


class RFWindowDataset(Dataset):
    def __init__(self, csv_files, stats, window_len=128, stride=16):
        self.clean_windows = []
        self.noisy_windows = []

        self.stats = stats

        for p in csv_files:
            clean, noisy = load_rf_csv(p)
            cw, nw = make_windows(clean, noisy, window_len, stride)

            self.clean_windows.append(cw)
            self.noisy_windows.append(nw)

        self.clean_windows = np.concatenate(self.clean_windows, axis=0)
        self.noisy_windows = np.concatenate(self.noisy_windows, axis=0)

    def __len__(self):
        return self.clean_windows.shape[0]

    def __getitem__(self, idx):
        clean = self.clean_windows[idx]
        noisy = self.noisy_windows[idx]

        clean = (clean - self.stats["clean_mean"]) / self.stats["clean_std"]
        noisy = (noisy - self.stats["noisy_mean"]) / self.stats["noisy_std"]

        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)

        return clean, noisy


# ============================================================
# 2. Conditional 1D denoising network
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResBlock1D(nn.Module):
    def __init__(self, channels, time_dim, dilation=1):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.time_proj = nn.Linear(time_dim, channels)

        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        time_term = self.time_proj(F.silu(t_emb)).unsqueeze(-1)
        h = h + time_term

        h = self.conv2(F.silu(self.norm2(h)))

        return x + h


class ConditionalDenoiser1D(nn.Module):
    def __init__(self, base_channels=64, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv1d(2, base_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            ResBlock1D(base_channels, time_dim, dilation=1),
            ResBlock1D(base_channels, time_dim, dilation=2),
            ResBlock1D(base_channels, time_dim, dilation=4),
            ResBlock1D(base_channels, time_dim, dilation=8),
            ResBlock1D(base_channels, time_dim, dilation=16),
            ResBlock1D(base_channels, time_dim, dilation=1),
        ])

        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_t, cond, t):
        t_emb = self.time_embedding(t)

        x = torch.cat([x_t, cond], dim=1)
        h = self.input_conv(x)

        for block in self.blocks:
            h = block(h, t_emb)

        eps_pred = self.output(h)
        return eps_pred


# ============================================================
# 3. DDPM utilities
# ============================================================

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion1D:
    def __init__(
        self,
        timesteps=200,
        beta_start=1e-4,
        beta_end=2e-2,
        device="cuda",
    ):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )

    def q_sample(self, x0, t, noise):
        sqrt_ab = extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_omab = extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)

        return sqrt_ab * x0 + sqrt_omab * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, cond, t, t_index):
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_ab_t = extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        eps_pred = model(x_t, cond, t)

        model_mean = sqrt_recip_alpha_t * (
            x_t - betas_t * eps_pred / sqrt_one_minus_ab_t
        )

        if t_index == 0:
            return model_mean

        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)

        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, cond):
        model.eval()

        x = torch.randn_like(cond)

        batch_size = cond.shape[0]

        for i in reversed(range(self.timesteps)):
            t = torch.full(
                (batch_size,),
                i,
                device=cond.device,
                dtype=torch.long,
            )
            x = self.p_sample(model, x, cond, t, i)

        return x


# ============================================================
# 4. Training and evaluation
# ============================================================

def split_files(csv_files, train_ratio=0.8, seed=0):
    rng = random.Random(seed)
    csv_files = list(csv_files)
    #rng.shuffle(csv_files)

    n_train = max(1, int(len(csv_files) * train_ratio))

    #train_files = csv_files[:n_train]
    #test_files = csv_files[n_train:]

    #test_files = csv_files[:len(csv_files) - n_train]
    #train_files = csv_files[len(csv_files) - n_train:]

    test_files = csv_files[:1]
    train_files = csv_files[0:1]

    if len(test_files) == 0:
        test_files = train_files[-1:]
        train_files = train_files[:-1]

    return train_files, test_files


def train(args):
    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))


    train_files, test_files = split_files(csv_files, args.train_ratio, args.seed)

    stats = compute_train_stats(train_files, args.window_len, args.stride)

    train_dataset = RFWindowDataset(
        train_files,
        stats,
        window_len=args.window_len,
        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Device:", device)

    model = ConditionalDenoiser1D(
        base_channels=args.channels,
        time_dim=args.time_dim,
    ).to(device)

    diffusion = GaussianDiffusion1D(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for clean, noisy in train_loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            b = clean.shape[0]

            t = torch.randint(
                0,
                args.timesteps,
                (b,),
                device=device,
                dtype=torch.long,
            )

            eps = torch.randn_like(clean)
            x_t = diffusion.q_sample(clean, t, eps)

            eps_pred = model(x_t, noisy, t)

            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item() * b
            total_count += b

        avg_loss = total_loss / total_count

        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | train diffusion loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                },
                save_path / "best_model.pt",
            )

    print("Training finished.")

    evaluate_and_plot(
        model=model,
        diffusion=diffusion,
        test_file=test_files[0],
        stats=stats,
        device=device,
        output_dir=Path(args.output_dir),
    )


@torch.no_grad()
def evaluate_and_plot(model, diffusion, test_file, stats, device, output_dir):
    clean, noisy = load_rf_csv(test_file)

    clean_norm = (clean - stats["clean_mean"]) / stats["clean_std"]
    noisy_norm = (noisy - stats["noisy_mean"]) / stats["noisy_std"]

    cond = torch.tensor(noisy_norm, dtype=torch.float32).view(1, 1, -1).to(device)

    pred_norm = diffusion.sample(model, cond)
    pred_norm = pred_norm.cpu().numpy()[0, 0]

    pred = pred_norm * stats["clean_std"] + stats["clean_mean"]

    '''
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_denoised = np.mean((pred - clean) ** 2)

    snr_before = 10 * np.log10(
        np.mean(clean ** 2) / (np.mean((noisy - clean) ** 2) + 1e-12)
    )
    snr_after = 10 * np.log10(
        np.mean(clean ** 2) / (np.mean((pred - clean) ** 2) + 1e-12)
    )

    print("\nEvaluation file:", test_file.name)
    print(f"MSE noisy    : {mse_noisy:.6f}")
    print(f"MSE denoised : {mse_denoised:.6f}")
    print(f"SNR before   : {snr_before:.3f} dB")
    print(f"SNR after    : {snr_after:.3f} dB")
    '''

    metrics_df = compare_noisy_and_denoised(
        clean=clean,
        noisy=noisy,
        denoised=pred,
    )

    print("\nEvaluation metrics:")
    print(metrics_df.to_string(index=False))

    metrics_path = output_dir / "rf_denoising_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics to:", metrics_path)


    samples = np.arange(len(clean))

    plt.figure(figsize=(18, 6))
    plt.plot(samples, clean, label="clean Signal", linewidth=1.5)
    plt.plot(samples, noisy, label="noisy Signal", linewidth=1.5, alpha=0.65)
    plt.plot(samples, pred, label="denoised Prediction", linewidth=2.0)

    plt.title("RF Signal Denoising")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "rf_signal_denoising_ddpm.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()

    print("Saved plot to:", fig_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./rf_cddm_output")

    parser.add_argument("--window_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)

    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
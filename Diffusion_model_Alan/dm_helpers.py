import pandas as pd
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


# ---- shared config (single source of truth, imported by dm_classes and dm_data) ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "rfid_diffusion_dataset"
CHECKPOINT_PATH = PROJECT_ROOT / "diffusion_ckpt.pt"
SANITY_PLOT_PATH = PROJECT_ROOT / "sanity_check.png"
LOSS_PLOT_PATH = PROJECT_ROOT / "training_loss.png"

WINDOW = 256
T = 1000
BETA_START = 1e-5
BETA_END = 1.5e-3
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_paired_windows(data_root=DATA_ROOT, window=WINDOW):
    clean_list = []
    noisy_list = []
    time_list = []
    sources = []
    csv_paths = sorted(Path(data_root).rglob("sample_*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No sample_*.csv files found under {data_root}")

    for path in csv_paths:
        df = pd.read_csv(path)
        if not {"time", "clean", "noisy"}.issubset(df.columns):
            continue
        if len(df) < window:
            continue

        clean = df["clean"].to_numpy(dtype=np.float64)[:window]
        noisy = df["noisy"].to_numpy(dtype=np.float64)[:window]
        t = df["time"].to_numpy(dtype=np.float64)[:window]

        mean = clean.mean()
        std = clean.std()
        if std < 0.3:
            continue

        clean_list.append(((clean - mean) / std).astype(np.float32))
        noisy_list.append(((noisy - mean) / std).astype(np.float32))
        time_list.append(t.astype(np.float64))
        rel = path.relative_to(data_root)
        sources.append(str(rel))

    if not clean_list:
        raise RuntimeError("Found files but no usable (clean, noisy, time) windows.")

    return (
        np.stack(clean_list),
        np.stack(noisy_list),
        np.stack(time_list),
        sources,
    )

def sanity_check(dataset, n_plot=6, seed=0):
    arr = dataset.windows
    print(f"  windows total : {len(dataset)}")
    print(f"  window length : {arr.shape[1]}")
    print(f"  clean mean    : {arr.mean():+.4f}   clean std: {arr.std():.4f}")
    print(f"  clean min/max : {arr.min():.2f} / {arr.max():.2f}")

    groups = {}
    for src in dataset.sources:
        bucket = src.split("/")[0] if "/" in src else src.split("\\")[0] if "\\" in src else "(top)"
        groups[bucket] = groups.get(bucket, 0) + 1
    print(f"  source groups :")
    for k, v in sorted(groups.items()):
        print(f"    {k}: {v} windows")

    if len(dataset) < 500:
        print(f"  WARNING: only {len(dataset)} windows — likely too few to train a diffusion model.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_plot, len(dataset)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, i in zip(axes.flatten(), idx):
        ax.plot(dataset.noisy_windows[i], color="tab:red", alpha=0.5, label="noisy")
        ax.plot(dataset.windows[i], color="tab:blue", linewidth=1.5, label="clean")
        ax.set_title(f"#{i}  {dataset.sources[i]}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Sanity check — clean (blue) vs noisy (red), normalized")
    fig.tight_layout()
    fig.savefig(SANITY_PLOT_PATH, dpi=100)
    print(f"  saved plot    : {SANITY_PLOT_PATH.name}")
    plt.close(fig)


def make_schedule(num_steps=T, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):
    # num_steps == T in the paper (Yin et al., Sec. III-A) — total diffusion timesteps.
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }

def quick_shape_test(model):
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, WINDOW, device=DEVICE)
        t = torch.randint(0, T, (2,), device=DEVICE)
        y = model(x, t)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    print(f"  forward pass ok: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    model.train()

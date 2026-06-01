import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm_helpers import DATA_ROOT, PROJECT_ROOT, WINDOW
from inference import load_trained_model, denoise, true_snr_db, estimate_snr_db, snr_map_db, find_t_star
from baselines import BASELINES


SAMPLE_PATH = DATA_ROOT / "2high" / "sample_0.csv"
OUT_PLOT = PROJECT_ROOT / "test_one.png"


def nmse(estimate, clean):
    return float(np.var(estimate - clean) / (np.var(clean) + 1e-12))


def main(n_ddim_steps=20):
    print(f"loading sample: {SAMPLE_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(SAMPLE_PATH).iloc[:WINDOW]
    clean_raw = df["clean"].to_numpy(dtype=np.float64)
    noisy_raw = df["noisy"].to_numpy(dtype=np.float64)
    t_axis = df["time"].to_numpy(dtype=np.float64)

    # normalize the same way load_paired_windows does (mean+std of clean)
    mu, sd = clean_raw.mean(), clean_raw.std()
    clean = ((clean_raw - mu) / sd).astype(np.float32)
    noisy = ((noisy_raw - mu) / sd).astype(np.float32)

    print("loading model + schedule")
    model, schedule = load_trained_model()

    snr_true = true_snr_db(clean, noisy)
    snr_est = estimate_snr_db(noisy)
    smap = snr_map_db(schedule)
    t_star_true = find_t_star(snr_true, smap)
    t_star_est = find_t_star(snr_est, smap)

    print(f"\nSNR (true,  from clean+noisy): {snr_true:+6.2f} dB   →  t* = {t_star_true}")
    print(f"SNR (estimated, noisy only)  : {snr_est:+6.2f} dB   →  t* = {t_star_est}")

    print(f"\nrunning denoisers ({n_ddim_steps} DDIM steps for diffusion):")
    outputs = {"raw (noisy)": noisy}
    for name, fn in BASELINES.items():
        outputs[name] = fn(noisy)
    denoised, t_star_used = denoise(noisy, model, schedule, snr_db_value=snr_true, n_steps=n_ddim_steps)
    outputs[f"diffusion (t*={t_star_used})"] = denoised

    print(f"\n{'method':28s}  NMSE")
    print("  " + "-" * 38)
    for name, out in outputs.items():
        print(f"  {name:26s}  {nmse(out, clean):.4f}")

    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 2.0 * len(outputs)), sharex=True)
    for ax, (name, out) in zip(axes, outputs.items()):
        ax.plot(t_axis, clean, color="tab:blue", linewidth=1.5, label="clean", linestyle="--")
        if name.startswith("raw"):
            ax.plot(t_axis, out, color="tab:red", linewidth=0.8, alpha=0.7, label="noisy")
        else:
            ax.plot(t_axis, out, color="tab:green", linewidth=1.2, label=name.split()[0])
        ax.set_title(f"{name}   NMSE = {nmse(out, clean):.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Single-sample test: 2high/sample_0.csv   (input SNR = {snr_true:+.1f} dB)")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=110)
    plt.close(fig)
    print(f"\nsaved {OUT_PLOT.name}")


if __name__ == "__main__":
    main()

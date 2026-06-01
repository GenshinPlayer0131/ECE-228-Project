import numpy as np
import matplotlib.pyplot as plt

from dm_classes import RFPhaseDataset
from dm_helpers import PROJECT_ROOT, split_indices
from inference import load_trained_model, denoise, true_snr_db, estimate_snr_db
from baselines import BASELINES

# two diffusion variants:
#   diffusion (oracle SNR) — t* picked from the true SNR; matches what the paper does,
#       since Yin et al. synthesize noise to a known target SNR (arXiv:2503.05514, Sec. V-A).
#   diffusion (blind SNR)  — t* picked from estimate_snr_db(noisy) alone; the deployable case.
DIFFUSION_ORACLE = "diffusion (oracle SNR)"
DIFFUSION_BLIND = "diffusion (blind SNR)"


EVAL_PLOT_PATH = PROJECT_ROOT / "evaluation.png"
EXAMPLE_PLOT_PATH = PROJECT_ROOT / "denoise_examples.png"


def nmse(estimate, clean):
    return float(np.var(estimate - clean) / (np.var(clean) + 1e-12))


def group_of(source):
    return source.split("/")[0] if "/" in source else source.split("\\")[0] if "\\" in source else "(top)"


def evaluate(n_ddim_steps=20, max_windows=None):
    print("loading model + dataset")
    model, schedule = load_trained_model()
    dataset = RFPhaseDataset()

    # score only the held-out split (same seed as dm_main) so diffusion is judged
    # on windows it never trained on — these NMSE numbers are out-of-sample.
    _, val_idx = split_indices(len(dataset))
    eval_idx = val_idx if max_windows is None else val_idx[:max_windows]
    n = len(eval_idx)
    print(f"evaluating on {n} held-out windows ({n_ddim_steps} DDIM steps each)\n")

    methods = ["raw"] + list(BASELINES.keys()) + [DIFFUSION_ORACLE, DIFFUSION_BLIND]
    results = {m: {"nmse": [], "snr_in_db": [], "group": []} for m in methods}

    for count, i in enumerate(eval_idx):
        clean = dataset.windows[i]
        noisy = dataset.noisy_windows[i]
        group = group_of(dataset.sources[i])
        snr_in = true_snr_db(clean, noisy)
        snr_est = estimate_snr_db(noisy)

        outputs = {"raw": noisy}
        for name, fn in BASELINES.items():
            outputs[name] = fn(noisy)
        # oracle: t* from the true SNR; blind: t* from the noisy-only estimate
        outputs[DIFFUSION_ORACLE] = denoise(noisy, model, schedule, snr_db_value=snr_in, n_steps=n_ddim_steps)[0]
        outputs[DIFFUSION_BLIND] = denoise(noisy, model, schedule, snr_db_value=snr_est, n_steps=n_ddim_steps)[0]

        for m in methods:
            results[m]["nmse"].append(nmse(outputs[m], clean))
            results[m]["snr_in_db"].append(snr_in)
            results[m]["group"].append(group)

        if (count + 1) % 25 == 0 or count == n - 1:
            print(f"  {count+1}/{n}")

    print("\n=== mean NMSE per method ===")
    for m in methods:
        arr = np.array(results[m]["nmse"])
        print(f"  {m:12s}  mean NMSE = {arr.mean():.3f}   median = {np.median(arr):.3f}")

    print("\n=== mean NMSE per method, per source group ===")
    groups = sorted(set(results["raw"]["group"]))
    header = "  group".ljust(20) + " ".join(f"{m:>11s}" for m in methods)
    print(header)
    for g in groups:
        row = [f"  {g}".ljust(20)]
        for m in methods:
            arr = np.array([v for v, grp in zip(results[m]["nmse"], results[m]["group"]) if grp == g])
            row.append(f"{arr.mean():>11.3f}")
        print(" ".join(row))

    snr_bins = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40])
    bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    snr_in_arr = np.array(results["raw"]["snr_in_db"])
    for m in methods:
        nmse_arr = np.array(results[m]["nmse"])
        means = []
        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            mask = (snr_in_arr >= lo) & (snr_in_arr < hi)
            means.append(nmse_arr[mask].mean() if mask.any() else np.nan)
        # dashed for the blind diffusion curve so oracle vs. blind is easy to compare
        style = "--" if m == DIFFUSION_BLIND else "-"
        ax.plot(bin_centers, means, marker="o", linestyle=style, label=m)
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("NMSE (lower is better)")
    ax.set_yscale("log")
    ax.set_title(f"Denoising comparison ({n} windows)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVAL_PLOT_PATH, dpi=100)
    plt.close(fig)
    print(f"\nsaved {EVAL_PLOT_PATH.name}")

    plot_examples(dataset, model, schedule, n_ddim_steps, pool=eval_idx)
    return results


def plot_examples(dataset, model, schedule, n_ddim_steps=20, n_examples=3, pool=None):
    rng = np.random.default_rng(0)
    pool = np.arange(len(dataset)) if pool is None else np.asarray(pool)
    idx = rng.choice(pool, size=min(n_examples, len(pool)), replace=False)

    fig, axes = plt.subplots(n_examples, 1, figsize=(11, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        clean = dataset.windows[i]
        noisy = dataset.noisy_windows[i]
        snr_in = true_snr_db(clean, noisy)
        denoised, t_star = denoise(noisy, model, schedule, snr_db_value=snr_in, n_steps=n_ddim_steps)
        wiener_out = BASELINES["wiener"](noisy)

        ax.plot(noisy, color="lightgray", linewidth=0.8, label="noisy")
        ax.plot(wiener_out, color="tab:orange", linewidth=1.0, alpha=0.8, label="wiener")
        ax.plot(denoised, color="tab:green", linewidth=1.2, label="diffusion")
        ax.plot(clean, color="tab:blue", linewidth=1.5, label="clean", linestyle="--")
        ax.set_title(f"window {i}  ({dataset.sources[i]})  SNR_in={snr_in:+.1f} dB  t*={t_star}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(EXAMPLE_PLOT_PATH, dpi=100)
    plt.close(fig)
    print(f"saved {EXAMPLE_PLOT_PATH.name}")


if __name__ == "__main__":
    evaluate(n_ddim_steps=20)

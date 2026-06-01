import numpy as np
import torch

from dm_helpers import make_schedule, DEVICE, CHECKPOINT_PATH
from dm_classes import UNet1D


def snr_map_db(schedule):
    # gamma_t = alpha_bar_t / (1 - alpha_bar_t)   (paper Eq. 3, normalized x_0)
    alpha_bars = schedule["alpha_bars"].detach().cpu().numpy()
    gamma = alpha_bars / (1.0 - alpha_bars + 1e-12)
    return 10.0 * np.log10(gamma + 1e-12)


def true_snr_db(clean, noisy):
    noise = noisy - clean
    return 10.0 * np.log10(np.var(clean) / (np.var(noise) + 1e-12))


def estimate_snr_db(noisy, smooth_n=5):
    kernel = np.ones(smooth_n) / smooth_n
    smooth = np.convolve(noisy, kernel, mode="same")
    residual = noisy - smooth
    return 10.0 * np.log10(np.var(smooth) / (np.var(residual) + 1e-12))


def find_t_star(snr_db_value, snr_map):
    # paper Eq. 7
    return int(np.argmin(np.abs(snr_map - snr_db_value)))


def make_ddim_timesteps(t_star, n_steps):
    # paper Eq. 6: ti = t* - i * delta_t, delta_t = t*/n_steps
    if t_star <= 0:
        return np.array([0], dtype=np.int64)
    n_steps = min(n_steps, t_star)
    delta = t_star / n_steps
    steps = [int(round(t_star - i * delta)) for i in range(n_steps)]
    steps.append(0)
    steps = sorted(set(s for s in steps if s >= 0), reverse=True)
    return np.array(steps, dtype=np.int64)


@torch.no_grad()
def ddim_denoise(model, noisy_window, snr_db_value, schedule, n_steps=20):
    model.eval()
    alpha_bars = schedule["alpha_bars"]

    snr_map = snr_map_db(schedule)
    t_star = find_t_star(snr_db_value, snr_map)

    timesteps = make_ddim_timesteps(t_star, n_steps)

    x = torch.as_tensor(noisy_window, dtype=torch.float32, device=DEVICE).view(1, 1, -1)

    for i in range(len(timesteps) - 1):
        t = int(timesteps[i])
        t_prev = int(timesteps[i + 1])
        t_tensor = torch.tensor([t], device=DEVICE)

        epsilon_theta = model(x, t_tensor)

        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0, device=DEVICE)

        # paper Eq. 5 (deterministic DDIM)
        pred_x0 = (x - torch.sqrt(1.0 - ab_t) * epsilon_theta) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * pred_x0 + torch.sqrt(1.0 - ab_prev) * epsilon_theta

    return x.squeeze().cpu().numpy(), t_star


def load_trained_model(checkpoint_path=CHECKPOINT_PATH):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = UNet1D().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    schedule = {k: v.to(DEVICE) for k, v in ckpt["schedule"].items()}
    return model, schedule


def denoise(noisy_window, model, schedule, snr_db_value=None, n_steps=20):
    if snr_db_value is None:
        snr_db_value = estimate_snr_db(noisy_window)
    return ddim_denoise(model, noisy_window, snr_db_value, schedule, n_steps=n_steps)


if __name__ == "__main__":
    from dm_classes import RFPhaseDataset

    print("loading model + schedule")
    model, schedule = load_trained_model()

    print("loading dataset")
    dataset = RFPhaseDataset()

    print("\nSNR map (first/last 5 entries, dB):")
    smap = snr_map_db(schedule)
    print(f"  t=0..4:    {smap[:5].round(2)}")
    print(f"  t=996..999: {smap[-5:].round(2)}")

    print("\ndenoising 3 windows:")
    for i in [0, len(dataset) // 2, len(dataset) - 1]:
        clean = dataset.windows[i]
        noisy = dataset.noisy_windows[i]
        snr_true = true_snr_db(clean, noisy)
        snr_est = estimate_snr_db(noisy)
        denoised, t_star = denoise(noisy, model, schedule, snr_db_value=snr_true)
        nmse_in = np.var(noisy - clean) / np.var(clean)
        nmse_out = np.var(denoised - clean) / np.var(clean)
        print(
            f"  window {i:3d}: SNR true={snr_true:+5.1f} dB  est={snr_est:+5.1f} dB  "
            f"t*={t_star:4d}  NMSE in={nmse_in:.3f}  out={nmse_out:.3f}"
        )

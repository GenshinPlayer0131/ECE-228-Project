import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dm_helpers import (
    sanity_check, make_schedule, quick_shape_test, split_indices,
    CHECKPOINT_PATH, LOSS_PLOT_PATH,
    WINDOW, BETA_START, BETA_END, BATCH_SIZE, EPOCHS, LR, DEVICE,
)
from dm_classes import RFPhaseDataset, UNet1D


def _val_loss(model, val_x0, schedule, num_steps, seed=1234):
    """Validation MSE on held-out windows.

    Uses a *fixed* (t, epsilon) draw via a seeded generator so the number is
    comparable epoch-to-epoch — otherwise fresh noise each call would make the
    val curve too jittery to read against the train curve.
    """
    if val_x0 is None or len(val_x0) == 0:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    B = val_x0.size(0)
    t = torch.randint(0, num_steps, (B,), generator=g).to(DEVICE)
    epsilon = torch.randn(val_x0.shape, generator=g).to(DEVICE)
    x0 = val_x0.to(DEVICE)
    sqrt_ab = schedule["sqrt_alpha_bars"][t].view(B, 1, 1)
    sqrt_1m_ab = schedule["sqrt_one_minus_alpha_bars"][t].view(B, 1, 1)
    xt = sqrt_ab * x0 + sqrt_1m_ab * epsilon

    model.eval()
    with torch.no_grad():
        loss = F.mse_loss(model(xt, t), epsilon)
    model.train()
    return loss.item()


def train(model, dataset, schedule, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, val_x0=None):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    sqrt_alpha_bars = schedule["sqrt_alpha_bars"]
    sqrt_one_minus_alpha_bars = schedule["sqrt_one_minus_alpha_bars"]
    num_steps = len(schedule["betas"])  # T in the paper (Yin et al., Sec. III-A)

    # mixed precision: halves VRAM use and speeds up Tensor-Core matmuls on the GPU.
    # disabled automatically on CPU so this stays a no-op there.
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler(DEVICE, enabled=use_amp)

    model.to(DEVICE).train()
    losses = []
    val_losses = []
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for x0 in loader:
            x0 = x0.to(DEVICE)
            B = x0.size(0)
            t = torch.randint(0, num_steps, (B,), device=DEVICE)
            epsilon = torch.randn_like(x0)

            sqrt_alpha_bar_t = sqrt_alpha_bars[t].view(B, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
                epsilon_theta = model(xt, t)
                loss = F.mse_loss(epsilon_theta, epsilon)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n += 1

        avg = running / max(n, 1)
        losses.append(avg)
        v = _val_loss(model, val_x0, schedule, num_steps)
        val_losses.append(v)
        print(f"  epoch {epoch+1:3d}/{epochs}   train={avg:.5f}   val={v:.5f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "schedule": {k: v.detach().cpu() for k, v in schedule.items()},
            "config": {"window": WINDOW, "T": num_steps, "beta_start": BETA_START, "beta_end": BETA_END},
        },
        CHECKPOINT_PATH,
    )
    print(f"  saved checkpoint to {CHECKPOINT_PATH.name}")

    fig = plt.figure(figsize=(7, 4))
    plt.plot(losses, marker="o", label="train")
    if val_x0 is not None and len(val_x0) > 0:
        plt.plot(val_losses, marker="o", label="val (held-out)")
        plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (predicted noise vs true noise)")
    plt.title("Diffusion model training loss")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=100)
    plt.close(fig)
    print(f"  saved loss plot to  {LOSS_PLOT_PATH.name}")
    return losses, val_losses




if __name__ == "__main__":
    print(f"device: {DEVICE}")

    print("\n[1] loading windows")
    dataset = RFPhaseDataset()

    print("\n[2] sanity check")
    sanity_check(dataset)

    print("\n[3] building model")
    model = UNet1D().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")
    quick_shape_test(model)

    print("\n[4] training")
    # hold out a fixed slice so we can watch for overfitting (train vs. val loss)
    # and so evaluate.py scores diffusion on windows it never trained on.
    train_idx, val_idx = split_indices(len(dataset))
    print(f"  split: {len(train_idx)} train / {len(val_idx)} held-out windows")
    train_set = Subset(dataset, train_idx)
    val_x0 = torch.from_numpy(dataset.windows[val_idx]).unsqueeze(1)
    schedule = make_schedule()
    train(model, train_set, schedule, val_x0=val_x0)

    print("\ndone.")

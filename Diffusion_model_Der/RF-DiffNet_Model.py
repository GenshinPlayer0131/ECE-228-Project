import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

dataDir = "2high"
windowSize = 64
stride = 32
batchSize = 16
epochs = 100
learningRate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
results_dir = "results"

os.makedirs(
    results_dir,
    exist_ok=True
)

print("Using device:", device)

class RFDataset(Dataset):
    def __init__(self, folder):
        self.clean_windows = []
        self.noisy_windows = []
        files = ["sample_0.csv"]

        print("Files found:", files)
        for file in files:
            path = os.path.join(folder, file)
            df = pd.read_csv(path)

            if "clean" not in df.columns or "noisy" not in df.columns:
                print("Skipping:", file)
                continue
            clean = df["clean"].values
            noisy = df["noisy"].values

            print(file, "length =", len(clean))

            # Normalize
            clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
            noisy = (noisy - np.mean(noisy)) / (np.std(noisy) + 1e-8)

            # Create windows
            for i in range(0, len(clean) - windowSize, stride):
                clean_win = clean[i:i + windowSize]
                noisy_win = noisy[i:i + windowSize]
                self.clean_windows.append(clean_win)
                self.noisy_windows.append(noisy_win)

        self.clean_windows = np.array(self.clean_windows)
        self.noisy_windows = np.array(self.noisy_windows)

        print("TOTAL WINDOWS:", len(self.noisy_windows))

    def __len__(self):
        return len(self.clean_windows)

    def __getitem__(self, idx):

        clean = self.clean_windows[idx]
        noisy = self.noisy_windows[idx]
        clean = torch.tensor(clean, dtype=torch.float32)
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)

        return noisy, clean

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose1d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u1 = self.up1(b)

        if u1.shape[-1] != e2.shape[-1]:
            diff = e2.shape[-1] - u1.shape[-1]
            u1 = nn.functional.pad(u1, (0, diff))

        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)

        if u2.shape[-1] != e1.shape[-1]:
            diff = e1.shape[-1] - u2.shape[-1]
            u2 = nn.functional.pad(u2, (0, diff))

        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)
        out = self.final(d2)
        return out

dataset = RFDataset(dataDir)

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check CSV folder or window size.")

loader = DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True
)

print("Dataset size:", len(dataset))

model = SimpleUNet().to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learningRate
)

losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        pred = model(noisy)
        loss = criterion(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

torch.save(
    model.state_dict(),
    os.path.join(
        results_dir,
        "rf_denoise_model.pth"
    )
)
print("Model saved.")

plt.figure(figsize=(8, 5))
plt.plot(losses)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.grid(True)

plt.savefig(
    os.path.join(
        results_dir,
        "training_loss.png"
    ),
    dpi = 300,
    bbox_inches="tight"
)

plt.show()

model.eval()

all_clean = []
all_noisy = []
all_pred = []

with torch.no_grad():
    for noisy, clean in dataset:
        inp = noisy.unsqueeze(0).to(device)
        pred = model(inp)
        noisy_np = noisy.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()
        all_noisy.extend(noisy_np)
        all_clean.extend(clean_np)
        all_pred.extend(pred_np)
all_noisy = np.array(all_noisy)
all_clean = np.array(all_clean)
all_pred = np.array(all_pred)

mse_noisy = np.mean((all_clean - all_noisy) ** 2)
rmse_noisy = np.sqrt(mse_noisy)
nmse_noisy = np.sum(
    (all_clean - all_noisy) ** 2
) / (
    np.sum(all_clean ** 2) + 1e-8
)

corr_noisy = np.corrcoef(
    all_clean,
    all_noisy
)[0, 1]

mse = np.mean((all_clean - all_pred) ** 2)
rmse = np.sqrt(mse)
nmse = np.sum(
    (all_clean - all_pred) ** 2
) / (
    np.sum(all_clean ** 2) + 1e-8
)

corr = np.corrcoef(
    all_clean,
    all_pred
)[0, 1]

print("RF DENOISING PERFORMANCE")

print("\nNoisy")
print(f"MSE               : {mse_noisy:.6f}")
print(f"RMSE              : {rmse_noisy:.6f}")
print(f"NMSE              : {nmse_noisy:.6f}")
print(f"Correlation Coeff : {corr_noisy:.6f}")

print("\nDenoised")
print(f"MSE               : {mse:.6f}")
print(f"RMSE              : {rmse:.6f}")
print(f"NMSE              : {nmse:.6f}")
print(f"Correlation Coeff : {corr:.6f}")

print("\improved")
print(
    f"MSE Reduction (%) : "
    f"{100 * (mse_noisy - mse) / mse_noisy:.2f}"
)

print(
    f"Correlation Gain  : "
    f"{corr - corr_noisy:.6f}"
)

results_df = pd.DataFrame({
    "Metric": [
        "MSE",
        "RMSE",
        "NMSE",
        "Correlation"
    ],
    "Noisy": [
        mse_noisy,
        rmse_noisy,
        nmse_noisy,
        corr_noisy
    ],
    "Denoised": [
        mse,
        rmse,
        nmse,
        corr
    ]
})

print(results_df)

with open(
    os.path.join(
        results_dir,
        "summary.txt"
    ),
    "w"
) as f:

    f.write("RF DENOISING RESULTS\n\n")
    f.write(
        f"Noisy MSE: {mse_noisy:.6f}\n"
    )
    f.write(
        f"Denoised MSE: {mse:.6f}\n"
    )
    f.write(
        f"Noisy RMSE: {rmse_noisy:.6f}\n"
    )
    f.write(
        f"Denoised RMSE: {rmse:.6f}\n"
    )
    f.write(
        f"Noisy NMSE: {nmse_noisy:.6f}\n"
    )
    f.write(
        f"Denoised NMSE: {nmse:.6f}\n"
    )
    f.write(
        f"Noisy Correlation: {corr_noisy:.6f}\n"
    )
    f.write(
        f"Denoised Correlation: {corr:.6f}\n"
    )
    f.write(
        f"MSE Reduction (%): "
        f"{100*(mse_noisy-mse)/mse_noisy:.2f}\n"
    )

results_df.to_csv(
    os.path.join(
        results_dir,
        "rf_denoising_metrics.csv"
    ),
    index = False
)

plt.figure(figsize=(15, 6))

plt.plot(
    all_clean,
    label = "Clean Signal",
    linewidth = 3
)

plt.plot(
    all_noisy,
    label = "Noisy Signal",
    alpha = 0.5
)

plt.plot(
    all_pred,
    label = "Denoised Prediction",
    linewidth=2
)

plt.title(
    f"RF Signal Denoising\n"
    f"MSE={mse:.4f} | "
    f"RMSE={rmse:.4f} | "
    f"NMSE={nmse:.4f} | "
    f"Corr={corr:.4f}"
)

plt.xlabel("Sample Index")
plt.ylabel("Normalized Amplitude")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()
plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["Noisy", "Denoised"],
    [mse_noisy, mse]
)

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"{y:.4f}",
        ha = "center",
        va = "bottom"
    )

plt.ylabel("MSE")
plt.title("Mean Squared Error Comparison")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "signal_comparison.png"
    ),
    dpi=300,
    bbox_inches = "tight"
)

plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["Noisy", "Denoised"],
    [corr_noisy, corr]
)
for bar in bars:
    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"{y:.4f}",
        ha = "center",
        va = "bottom"
    )

plt.ylabel("Correlation Coefficient")
plt.title("Correlation Comparison")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "mse_comparison.png"
    ),
    dpi = 300,
    bbox_inches = "tight"
)

plt.savefig(
    os.path.join(
        results_dir,
        "correlation_comparison.png"
    ),
    dpi = 300,
    bbox_inches = "tight"
)

plt.savefig(
    os.path.join(
        results_dir,
        "sample0_waveform.png"
    ),
    dpi = 300,
    bbox_inches = "tight"
)

plt.savefig(
    "sample0_reconstruction.png",
    dpi = 300,
    bbox_inches = "tight"
)


plt.show()
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dm_helpers import load_paired_windows, DATA_ROOT, WINDOW



class RFPhaseDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, window=WINDOW):
        self.windows, self.noisy_windows, self.times, self.sources = load_paired_windows(data_root, window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]).unsqueeze(0)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, channels=(32, 64, 128), time_dim=128):
        super().__init__()
        c1, c2, c3 = channels
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(1, c1, 3, padding=1)
        self.enc1 = ResBlock1D(c1, c1, time_dim)
        self.enc2 = ResBlock1D(c1, c2, time_dim)
        self.enc3 = ResBlock1D(c2, c3, time_dim)
        self.mid = ResBlock1D(c3, c3, time_dim)
        self.dec3 = ResBlock1D(c3 + c3, c2, time_dim)
        self.dec2 = ResBlock1D(c2 + c2, c1, time_dim)
        self.dec1 = ResBlock1D(c1 + c1, c1, time_dim)
        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv1d(c1, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h0 = self.in_conv(x)
        s1 = self.enc1(h0, t_emb)
        h = F.avg_pool1d(s1, 2)
        s2 = self.enc2(h, t_emb)
        h = F.avg_pool1d(s2, 2)
        s3 = self.enc3(h, t_emb)
        m = self.mid(s3, t_emb)
        h = self.dec3(torch.cat([m, s3], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
        h = self.dec2(torch.cat([h, s2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
        h = self.dec1(torch.cat([h, s1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))

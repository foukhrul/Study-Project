#!/usr/bin/env python3
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT
from core_pfcp import load_pfcp_packets, DEFAULT_PFCP_ROOT
from task2_flows_similarity import build_flows_all_windows, MAX_PACKETS_PER_FLOW


# -------------------------------
# 0) Utilities
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# 1) Dataset: flows → tensors
# -------------------------------

class FlowDataset(Dataset):
    """
    Takes the 'flows' list from build_flows_all_windows and prepares tensors.

    We use only NORMAL (no-attack) flows, because Task 3 wants
    synthetic data for normal network operation.
    """

    def __init__(self, flows, window_minutes: int, normalize=True):
        # filter by window and NON-attack
        filtered = [
            f for f in flows
            if f["window_minutes"] == window_minutes and not f["is_attack"]
        ]
        if not filtered:
            raise ValueError(f"No normal flows found for window={window_minutes}")

        self.window_minutes = window_minutes
        self.normalize = normalize

        # features are 1D vectors: (MAX_PACKETS_PER_FLOW * 3,)
        X = np.stack([f["features"] for f in filtered], axis=0).astype(np.float32)

        # reshape to (N, C, L) where C=3 features, L=MAX_PACKETS_PER_FLOW
        C = 3
        L = MAX_PACKETS_PER_FLOW
        X = X.reshape(-1, L, C).transpose(0, 2, 1)  # (N, C, L)

        if normalize:
            # simple per-feature standardization over the whole dataset
            mean = X.mean(axis=(0, 2), keepdims=True)
            std = X.std(axis=(0, 2), keepdims=True) + 1e-8
            X = (X - mean) / std
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

        self.X = torch.from_numpy(X)  # (N, 3, L)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# -------------------------------
# 2) Discriminator (Task 3b)
# -------------------------------

class Discriminator(nn.Module):
    """
    1D CNN discriminator:

    - 9 Conv1d layers with LeakyReLU
    - Dropout after conv2..conv8
    - Then flatten and 4 fully connected layers (ReLU)
    - outputs a single scalar in [0,1)
    """

    def __init__(self, in_channels=3, base_channels=32, length=MAX_PACKETS_PER_FLOW, dropout_p=0.3):
        super().__init__()
        self.length = length

        convs = []
        channels = in_channels
        for i in range(9):
            out_ch = base_channels * min(4, 1 + i // 3)  # increase channels a bit
            conv = nn.Conv1d(channels, out_ch, kernel_size=3, padding=1)
            convs.append(conv)
            channels = out_ch
        self.convs = nn.ModuleList(convs)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(dropout_p)

        # feature size after convolutions
        conv_feat_dim = channels * length

        # 4 fully connected layers
        fc_hidden = 256
        self.fc1 = nn.Linear(conv_feat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, fc_hidden)
        self.fc4 = nn.Linear(fc_hidden, 1)
        self.fc_act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, L)
        Returns:
          out: (B, 1)
          feat: (B, conv_feat_dim)  # for custom generator loss
        """
        # 9 convs with LeakyReLU + dropout after all except first & last
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.act(x)
            if 0 < i < len(self.convs) - 1:
                x = self.drop(x)

        # flatten conv features
        feat = x.view(x.size(0), -1)

        # fully-connected head
        h = self.fc1(feat)
        h = self.fc_act(h)
        h = self.fc2(h)
        h = self.fc_act(h)
        h = self.fc3(h)
        h = self.fc_act(h)
        h = self.fc4(h)
        out = self.out_act(h)  # (B,1) in (0,1)

        return out, feat


# -------------------------------
# 3) Generator (Task 3c)
# -------------------------------

class Generator(nn.Module):
    """
    Reverse CNN generator:

    - Input: noise z
    - 3 fully-connected layers (ReLU)
    - reshape to (C, L0)
    - 8 Conv1d layers with ReLU
      * first 4 conv layers each preceded by Upsample (nearest)
    - Output: (B, 3, MAX_PACKETS_PER_FLOW)
    """

    def __init__(self, noise_dim=128, base_channels=64, target_length=MAX_PACKETS_PER_FLOW):
        super().__init__()
        self.noise_dim = noise_dim
        self.target_length = target_length

        # choose small base length and upsample 4 times
        # 3 * 2^4 = 48, then we'll pad/crop to 50
        self.base_length = 3
        self.upsamples = 4

        # 3 fully-connected layers
        fc_hidden = 256
        self.fc1 = nn.Linear(noise_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, base_channels * self.base_length)
        self.fc_act = nn.ReLU()

        # 8 conv layers
        convs = []
        channels = base_channels
        for i in range(8):
            out_ch = max(16, base_channels // (2 ** (i // 2)))
            conv = nn.Conv1d(channels, out_ch, kernel_size=3, padding=1)
            convs.append(conv)
            channels = out_ch
        self.convs = nn.ModuleList(convs)
        self.act = nn.ReLU()

        # upsample before first 4 convs
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # final projection to 3 channels (features)
        self.to_out = nn.Conv1d(channels, 3, kernel_size=3, padding=1)

    def forward(self, z):
        """
        z: (B, noise_dim)
        returns: (B, 3, target_length)
        """
        h = self.fc_act(self.fc1(z))
        h = self.fc_act(self.fc2(h))
        h = self.fc_act(self.fc3(h))  # (B, base_channels * base_length)

        # reshape to (B, C, L0)
        batch_size = h.size(0)
        # infer C from fc3 out_features and base_length
        C0 = h.size(1) // self.base_length
        x = h.view(batch_size, C0, self.base_length)  # (B, C0, L0)

        length = self.base_length
        for i, conv in enumerate(self.convs):
            if i < 4:
                # upsample (double the length)
                x = self.upsample(x)
                length *= 2
            x = conv(x)
            x = self.act(x)

        x = self.to_out(x)  # (B, 3, current_length)

        # pad/crop to target length
        cur_len = x.size(2)
        if cur_len < self.target_length:
            pad_len = self.target_length - cur_len
            pad = torch.zeros(x.size(0), x.size(1), pad_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)
        elif cur_len > self.target_length:
            x = x[:, :, : self.target_length]

        return x


# -------------------------------
# 4) Training & custom loss (Task 3d)
# -------------------------------

def generator_feature_matching_loss(feat_real, feat_fake):
    """
    Custom loss on discriminator features (input to first FC layer):
    minimize difference between mean features of real and fake.
    """
    # feat_real, feat_fake: (B, D)
    mu_real = feat_real.mean(dim=0)
    mu_fake = feat_fake.mean(dim=0)
    return torch.mean((mu_real - mu_fake) ** 2)


def train_gan(
    dataset,
    noise_dim=128,
    batch_size=64,
    epochs=20,
    lr_d=1e-4,
    lr_g=1e-4,
    feature_loss_weight=10.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    D = Discriminator(in_channels=3, length=MAX_PACKETS_PER_FLOW).to(device)
    G = Generator(noise_dim=noise_dim, target_length=MAX_PACKETS_PER_FLOW).to(device)

    bce = nn.BCELoss()
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        D.train()
        G.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        n_batches = 0

        for real in dataloader:
            real = real.to(device)  # (B, 3, L)
            B = real.size(0)

            # -----------------------------
            # 1) Update Discriminator
            # -----------------------------
            opt_D.zero_grad()

            # real labels=1, fake labels=0
            real_labels = torch.ones(B, 1, device=device)
            fake_labels = torch.zeros(B, 1, device=device)

            # D(real)
            out_real, _ = D(real)
            loss_real = bce(out_real, real_labels)

            # D(fake)
            z = torch.randn(B, noise_dim, device=device)
            fake = G(z).detach()
            out_fake, _ = D(fake)
            loss_fake = bce(out_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_D.step()

            # -----------------------------
            # 2) Update Generator
            # -----------------------------
            opt_G.zero_grad()
            z = torch.randn(B, noise_dim, device=device)
            fake = G(z)
            out_fake, feat_fake = D(fake)

            # GAN loss: want D(fake) → 1
            g_adv_loss = bce(out_fake, real_labels)

            # feature-matching loss: compare features of real vs fake
            with torch.no_grad():
                _, feat_real = D(real)

            g_feat_loss = generator_feature_matching_loss(feat_real, feat_fake)

            g_loss = g_adv_loss + feature_loss_weight * g_feat_loss
            g_loss.backward()
            opt_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            n_batches += 1

        avg_d = total_d_loss / max(1, n_batches)
        avg_g = total_g_loss / max(1, n_batches)
        print(
            f"[GAN] Epoch {epoch}/{epochs} - D_loss={avg_d:.4f}, "
            f"G_loss={avg_g:.4f}"
        )

    return G, D


# -------------------------------
# 5) Hyperparameter tuning (Task 3e)
# -------------------------------

def tune_gan_hyperparameters(dataset, device):
    """
    Very simple grid search over a few hyperparameters:
      - noise_dim
      - feature_loss_weight
      - epochs (short, just for comparison)
    We pick config with lowest final generator loss proxy.
    """

    configs = [
        {"noise_dim": 64, "feature_loss_weight": 5.0, "epochs": 5},
        {"noise_dim": 64, "feature_loss_weight": 10.0, "epochs": 5},
        {"noise_dim": 128, "feature_loss_weight": 5.0, "epochs": 5},
        {"noise_dim": 128, "feature_loss_weight": 10.0, "epochs": 5},
    ]

    best_cfg = None
    best_score = float("inf")

    for cfg in configs:
        print(f"\n[GAN-TUNE] Trying config: {cfg}")
        G, D = train_gan(
            dataset=dataset,
            noise_dim=cfg["noise_dim"],
            feature_loss_weight=cfg["feature_loss_weight"],
            epochs=cfg["epochs"],
            device=device,
        )
        # simple proxy: evaluate feature-matching loss on a small batch
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        real = next(iter(loader)).to(device)
        B = real.size(0)
        z = torch.randn(B, cfg["noise_dim"], device=device)
        fake = G(z)
        _, feat_real = D(real)
        _, feat_fake = D(fake)
        score = generator_feature_matching_loss(feat_real, feat_fake).item()
        print(f"[GAN-TUNE] Config {cfg} -> feature loss proxy: {score:.6f}")

        if score < best_score:
            best_score = score
            best_cfg = cfg

    print(f"\n[GAN-TUNE] Best config: {best_cfg} with score={best_score:.6f}")
    return best_cfg


# -------------------------------
# 6) Main: load data, build flows, train/tune GAN
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task 3: GAN-based synthetic flow generation for 5G datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["nidd", "pfcp"],
        default="nidd",
        help="Dataset to use (default: nidd)",
    )
    parser.add_argument(
        "--nidd_root",
        default=DEFAULT_NIDD_ROOT,
        help="Root folder for 5G-NIDD CSV/PCAP files",
    )
    parser.add_argument(
        "--pfcp_root",
        default=DEFAULT_PFCP_ROOT,
        help="Root folder for 5G-PFCP CSV/PCAP files",
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=[1, 3, 5],
        default=1,
        help="Time window in minutes for flows (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for GAN training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for GAN training",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=128,
        help="Dimension of the generator noise input",
    )
    parser.add_argument(
        "--feature_loss_weight",
        type=float,
        default=10.0,
        help="Weight for feature-matching loss in generator objective",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="If set, perform simple hyperparameter tuning instead of single training run",
    )
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Task3] Using device: {device}")

    # load dataset
    if args.dataset == "nidd":
        print(f"[Task3] Loading 5G-NIDD from: {args.nidd_root}")
        df = load_nidd_packets(args.nidd_root)
        ds_name = "NIDD"
    else:
        print(f"[Task3] Loading 5G-PFCP from: {args.pfcp_root}")
        df = load_pfcp_packets(args.pfcp_root)
        ds_name = "PFCP"

    if df is None or df.empty:
        print("[Task3] DataFrame is empty, aborting.")
        return

    print(f"[Task3] Loaded {len(df):,} packets.")

    # build flows (reuse Task 2 logic)
    flows = build_flows_all_windows(df, window_list=(1, 3, 5))
    print(f"[Task3] Total flows generated (all windows): {len(flows):,}")

    # create dataset for chosen window, normal flows only
    flow_dataset = FlowDataset(flows, window_minutes=args.window, normalize=True)
    print(f"[Task3] Using {len(flow_dataset)} NORMAL flows for window={args.window}min")

    if args.tune:
        print("[Task3] Starting hyperparameter tuning...")
        best_cfg = tune_gan_hyperparameters(flow_dataset, device=device)
        print("[Task3] Hyperparameter tuning finished.")
        print(f"[Task3] Best config: {best_cfg}")
    else:
        print("[Task3] Training GAN with single configuration...")
        G, D = train_gan(
            dataset=flow_dataset,
            noise_dim=args.noise_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_d=1e-4,
            lr_g=1e-4,
            feature_loss_weight=args.feature_loss_weight,
            device=device,
        )
        # optionally save generator
        out_dir = "task3_models"
        os.makedirs(out_dir, exist_ok=True)
        gen_path = os.path.join(out_dir, f"gan_G_{ds_name.lower()}_{args.window}min.pt")
        torch.save(G.state_dict(), gen_path)
        print(f"[Task3] Saved generator to: {gen_path}")


if __name__ == "__main__":
    main()

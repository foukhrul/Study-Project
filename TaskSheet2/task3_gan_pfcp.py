""" TaskSheet2 - Task 3 (PFCP GAN)
Trains a simple GAN on REAL PFCP normal metadata using:
  - pfcp_real_dirsize.csv
  - pfcp_real_iat.csv
Then generates synthetic PFCP flows and saves them in:
  TaskSheet2/synth_pfcp_gan/ """

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------- PATHS ---------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "TaskSheet2")

REAL_DIRSIZE = os.path.join(DATA_DIR, "pfcp_real_dirsize.csv")
REAL_IAT     = os.path.join(DATA_DIR, "pfcp_real_iat.csv")

OUT_SYNTH = os.path.join(DATA_DIR, "synth_pfcp_gan")
os.makedirs(OUT_SYNTH, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("REAL_DIRSIZE:", REAL_DIRSIZE)
print("REAL_IAT    :", REAL_IAT)
print("OUT_SYNTH   :", OUT_SYNTH)

# কতগুলো sample নেব (পুরা ~2M না, কিন্তু 1-day এর বড় subset)
MAX_SAMPLES = 100_000   # চাইলে 50_000 করতে পারো

# --------------------- LOAD TRAIN DATA ---------------------

def load_training_vector() -> np.ndarray:
    """
    Build training matrix X of shape (N, 2):

      X[:, 0] = normalized dir+size values
      X[:, 1] = normalized IAT values

    We take a subset of size <= MAX_SAMPLES for practicality.
    """
    print("\n[LOAD] Real PFCP metadata for GAN training")

    df1 = pd.read_csv(REAL_DIRSIZE)
    df2 = pd.read_csv(REAL_IAT)

    v1 = df1["value"].values.astype(float)
    v2 = df2["value"].values.astype(float)

    v1_min, v1_max = v1.min(), v1.max()
    v2_min, v2_max = v2.min(), v2.max()

    v1_norm = (v1 - v1_min) / (v1_max - v1_min + 1e-9)
    v2_norm = (v2 - v2_min) / (v2_max - v2_min + 1e-9)

    L = min(len(v1_norm), len(v2_norm), MAX_SAMPLES)
    v1_norm = v1_norm[:L]
    v2_norm = v2_norm[:L]

    X = np.stack([v1_norm, v2_norm], axis=1)

    print(f"  dir+size: {len(v1):,} values (min={v1_min:.3f}, max={v1_max:.3f})")
    print(f"  iat     : {len(v2):,} values (min={v2_min:.3f}, max={v2_max:.3f})")
    print(f"  Using SUBSET L={L:,} for GAN, X.shape={X.shape}")
    return X


X_np = load_training_vector()
X_tensor = torch.tensor(X_np, dtype=torch.float32)

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)  # batch=64, not too big


# --------------------- GAN MODEL ---------------------

LATENT_DIM = 16


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # keep outputs in [0,1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # probability real/fake
        )

    def forward(self, x):
        return self.net(x)


G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

EPOCHS = 10   # subset + 10 epochs = manageable

print("\n[GAN] Training started...\n")


# --------------------- TRAIN LOOP ---------------------

for epoch in range(EPOCHS):
    for batch in loader:
        real = batch[0]
        bsz = real.size(0)

        # ----- Train Discriminator -----
        z = torch.randn(bsz, LATENT_DIM)
        fake = G(z).detach()

        out_real = D(real)
        out_fake = D(fake)

        loss_real = criterion(out_real, torch.ones_like(out_real))
        loss_fake = criterion(out_fake, torch.zeros_like(out_fake))
        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ----- Train Generator -----
        z = torch.randn(bsz, LATENT_DIM)
        fake = G(z)
        out_fake = D(fake)

        loss_G = criterion(out_fake, torch.ones_like(out_fake))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"D_loss={loss_D.item():.4f} | G_loss={loss_G.item():.4f}"
    )

print("\n[GAN] Training finished on PFCP subset.\n")


# --------------------- SYNTHETIC FLOW GENERATION ---------------------

def generate_synthetic_flows(n_flows: int = 20, length: int = 80) -> None:
    """
    Generate n_flows synthetic PFCP flows.
    Each flow: 'length' samples with:
        dirsize_norm, iat_norm  in [0,1]
    """
    for idx in range(n_flows):
        z = torch.randn(length, LATENT_DIM)

        # এখানে .numpy() না, শুধু Python list নিই
        with torch.no_grad():
            out_tensor = G(z)          # shape: (length, 2)

        out_list = out_tensor.tolist()  # normal nested Python list

        dirsize_vals = [row[0] for row in out_list]
        iat_vals     = [row[1] for row in out_list]

        df = pd.DataFrame({
            "dirsize_norm": dirsize_vals,
            "iat_norm": iat_vals,
        })

        fname = os.path.join(OUT_SYNTH, f"synth_pfcp_flow_{idx:03d}.csv")
        df.to_csv(fname, index=False)
        print("[SAVE]", fname)

    print("\n[SYNTH DONE] Synthetic PFCP flows saved in:", OUT_SYNTH)



if __name__ == "__main__":
    generate_synthetic_flows(n_flows=20, length=80)

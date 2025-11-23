#!/usr/bin/env python3
import os
import argparse

import torch
import pandas as pd

from task3_gan import Generator, MAX_PACKETS_PER_FLOW


def generate_and_save_flows(
    model_path: str,
    out_dir: str,
    num_flows: int = 10,
    noise_dim: int = 128,
):
    """
    Load a trained generator and save synthetic flows to CSV files.

    Each flow is saved as: synthetic_flow_<i>.csv with columns:
      - size
      - direction
      - iat
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Gen] Using device: {device}")

    # 1) Load generator with same noise_dim and target length as during training
    G = Generator(noise_dim=noise_dim, target_length=MAX_PACKETS_PER_FLOW).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    print(f"[Gen] Loaded generator from: {model_path}")

    os.makedirs(out_dir, exist_ok=True)

    # 2) Sample random noise
    z = torch.randn(num_flows, noise_dim, device=device)

    # 3) Generate synthetic flows
    with torch.no_grad():
        fake = G(z)  # shape: (num_flows, 3, MAX_PACKETS_PER_FLOW)
    fake_flows = fake.cpu().numpy()

    print(f"[Gen] Generated {num_flows} flows with shape: {fake_flows.shape}")

    # 4) Save each flow as a CSV
    for i, f in enumerate(fake_flows):
        # f has shape (3, L) where 0=size, 1=direction, 2=iat
        df = pd.DataFrame({
            "size": f[0],
            "direction": f[1],
            "iat": f[2],
        })
        out_path = os.path.join(out_dir, f"synthetic_flow_{i:03d}.csv")
        df.to_csv(out_path, index=False)
        print(f"[Gen] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic flows from a trained GAN generator and save to CSV."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained generator .pt file (e.g. task3_models/gan_G_nidd_1min.pt)",
    )
    parser.add_argument(
        "--out_dir",
        default="task3_synthetic_flows",
        help="Directory to save synthetic flow CSVs",
    )
    parser.add_argument(
        "--num_flows",
        type=int,
        default=10,
        help="How many synthetic flows to generate",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=128,
        help="Noise dimension used when training the generator",
    )

    args = parser.parse_args()

    generate_and_save_flows(
        model_path=args.model_path,
        out_dir=args.out_dir,
        num_flows=args.num_flows,
        noise_dim=args.noise_dim,
    )


if __name__ == "__main__":
    main()

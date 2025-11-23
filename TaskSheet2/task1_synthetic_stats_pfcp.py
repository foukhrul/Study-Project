#!/usr/bin/env python3

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths (hard-coded) ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "TaskSheet2")
OUT_DIR = os.path.join(DATA_DIR, "pfcp_task1_plots")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------- Loading & normalizing ----------

def load_metadata_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    print(f"[Load] {filename}")
    df = pd.read_csv(path)
    df = df[["pair", "protocol", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    print(
        f"  rows={len(df):,}, pairs={df['pair'].nunique():,}, "
        f"prot={df['protocol'].nunique():,}"
    )
    return df


def normalize_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    v = df["value"].values.astype(float)
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    if v_max == v_min:
        df["value_norm"] = 0.5
    else:
        df["value_norm"] = (v - v_min) / (v_max - v_min)
    print(f"[Norm] {name}: min={v_min:.6f}, max={v_max:.6f}")
    return df


# ---------- MinMax statistic ----------

def build_minmax_models(
    real_df: pd.DataFrame,
) -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
    models: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
    for (pair, proto), g in real_df.groupby(["pair", "protocol"]):
        vals = g["value_norm"].values
        if len(vals) == 0:
            continue
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        width = v_max - v_min
        min_err = v_min - width / 4.0
        max_err = v_max + width / 4.0
        models[(pair, proto)] = (v_min, v_max, min_err, max_err)
    print(f"[MinMax] models={len(models):,}")
    return models


def minmax_deviations(
    models: Dict[Tuple[str, str], Tuple[float, float, float, float]],
    synth_df: pd.DataFrame,
) -> np.ndarray:
    devs: List[float] = []
    for (pair, proto), g in synth_df.groupby(["pair", "protocol"]):
        key = (pair, proto)
        if key not in models:
            continue
        _, _, min_err, max_err = models[key]
        vals = g["value_norm"].values
        for v in vals:
            if v < min_err:
                devs.append(float(min_err - v))
            elif v > max_err:
                devs.append(float(v - max_err))
            else:
                devs.append(0.0)
    devs_arr = np.array(devs, dtype=float)
    print(f"[MinMax] deviations count={len(devs_arr):,}")
    return devs_arr


# ---------- Entropy statistic ----------

def shannon_entropy(vals: np.ndarray, bins: int = 20) -> float:
    if len(vals) == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist.astype(float) / float(total)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def entropies_per_pair(df: pd.DataFrame, bins: int = 20) -> np.ndarray:
    hs: List[float] = []
    for _, g in df.groupby(["pair", "protocol"]):
        v = g["value_norm"].values
        hs.append(shannon_entropy(v, bins=bins))
    arr = np.array(hs, dtype=float)
    print(f"[Entropy] count={len(arr):,}")
    return arr


# ---------- CDF / CCDF helpers ----------

def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    n = len(x)
    y = np.arange(1, n + 1) / float(n)
    return x, y


def empirical_ccdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    n = len(x)
    y = (n - np.arange(0, n)) / float(n)
    return x, y


# ---------- Plotting ----------

def plot_entropy_cdf(ent: np.ndarray, title: str, fname: str) -> None:
    if len(ent) == 0:
        print(f"[Plot] {title} – no data")
        return
    x, y = empirical_cdf(ent)
    plt.figure()
    plt.plot(x, y, marker=".", linestyle="-")
    plt.xlabel("Entropy H")
    plt.ylabel("CDF  P(H ≤ x)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"[Plot] saved {fname}")


def plot_minmax_ccdf(devs: np.ndarray, title: str, fname: str) -> None:
    if len(devs) == 0:
        print(f"[Plot] {title} – no data")
        return
    x, y = empirical_ccdf(devs)
    plt.figure()
    plt.semilogy(x, y, marker=".", linestyle="-")
    plt.xlabel("Deviation from [min_err, max_err]")
    plt.ylabel("CCDF  P(dev ≥ x)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"[Plot] saved {fname}")


# ---------- Main ----------

def main() -> None:
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR :", OUT_DIR)
    ensure_dir(OUT_DIR)

    # 1) load 4 CSV (2.5-day real + synthetic)
    real_dir = load_metadata_csv("pfcp_real_dirsize.csv")
    real_iat = load_metadata_csv("pfcp_real_iat.csv")
    synth_dir = load_metadata_csv("pfcp_synth_dirsize.csv")
    synth_iat = load_metadata_csv("pfcp_synth_iat.csv")

    # 2) normalize
    real_dir = normalize_df(real_dir, "real dir+size")
    real_iat = normalize_df(real_iat, "real iat")
    synth_dir = normalize_df(synth_dir, "synth dir+size")
    synth_iat = normalize_df(synth_iat, "synth iat")

    # 3) MinMax model from real, deviations on synthetic (use dir+size)
    models = build_minmax_models(real_dir)
    devs_dir = minmax_deviations(models, synth_dir)

    # 4) Entropy per (pair, proto) for all 4 datasets
    ent_real_dir = entropies_per_pair(real_dir)
    ent_real_iat = entropies_per_pair(real_iat)
    ent_synth_dir = entropies_per_pair(synth_dir)
    ent_synth_iat = entropies_per_pair(synth_iat)

    # 5) Plots: entropy CDF (real vs synth, both types)
    plot_entropy_cdf(
        ent_real_dir,
        "Entropy CDF - REAL PFCP (dir+size)",
        "entropy_cdf_real_dirsize.png",
    )
    plot_entropy_cdf(
        ent_synth_dir,
        "Entropy CDF - SYNTH PFCP (dir+size)",
        "entropy_cdf_synth_dirsize.png",
    )
    plot_entropy_cdf(
        ent_real_iat,
        "Entropy CDF - REAL PFCP (IAT)",
        "entropy_cdf_real_iat.png",
    )
    plot_entropy_cdf(
        ent_synth_iat,
        "Entropy CDF - SYNTH PFCP (IAT)",
        "entropy_cdf_synth_iat.png",
    )

    # 6) Plot: MinMax deviation CCDF for synthetic (dir+size)
    plot_minmax_ccdf(
        devs_dir,
        "MinMax CCDF - deviation (SYNTH vs REAL, dir+size)",
        "minmax_ccdf_dirsize.png",
    )

    print("\n[Done] Task 1 (a–c) PFCP stats & plots generated.")


if __name__ == "__main__":
    main()

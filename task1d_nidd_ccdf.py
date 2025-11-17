#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT


def compute_ccdf(values):
    """
    Manual CCDF computation.
    Input:  iterable/array of numeric values
    Output: (xs, ys) where ys[i] = P(X >= xs[i])
    """
    vals = sorted(values)
    n = len(vals)
    xs, ys = [], []

    if n == 0:
        return xs, ys

    prev = None
    for i, v in enumerate(vals):
        if v != prev:
            ccdf = (n - i) / n
            xs.append(v)
            ys.append(ccdf)
            prev = v

    return xs, ys


def safe_name(proto):
    """Make protocol name safe for filenames."""
    return "".join(ch if ch.isalnum() else "_" for ch in proto) or "UNKNOWN"


def plot_ccdf_for_metric(df, metric_col, metric_label, out_dir, with_attack_flag):
    """
    metric_col       : 'header_len' or 'app_len'
    metric_label     : nice name for axis/filename
    with_attack_flag : True => only Attack packets, False => Normal+Unlabeled
    """
    if with_attack_flag:
        subset = df[df["is_attack"] == True].copy()
        tag = "with_attack"
        title_tag = "with attacks"
    else:
        subset = df[df["is_attack"] == False].copy()
        tag = "without_attack"
        title_tag = "without attacks"

    # শুধু positive length + NaN বাদ
    series = subset[[metric_col, "app_proto"]].dropna()
    series = series[series[metric_col] > 0]

    if series.empty:
        print(f"[NIDD-1d] No data for {metric_label} ({title_tag})")
        return

    protocols = sorted(series["app_proto"].unique())

    print(
        f"[NIDD-1d] Plotting {metric_label} CCDFs {title_tag} "
        f"for {len(protocols)} app protocols..."
    )

    for proto in protocols:
        vals = series.loc[series["app_proto"] == proto, metric_col].tolist()
        if len(vals) < 2:
            # sample খুব কম হলে CCDF তেমন অর্থপূর্ণ না
            continue

        xs, ys = compute_ccdf(vals)
        if not xs:
            continue

        plt.figure()
        plt.step(xs, ys, where="post")

        plt.xlabel(f"{metric_label} (bytes)")
        plt.ylabel("P(X ≥ x)")
        plt.title(
            f"5G-NIDD {metric_label} CCDF ({title_tag})\n"
            f"Application protocol: {proto}"
        )
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xscale("log")  # length range বড়, log-scale এ ভালো দেখা যায়

        fname = f"nidd_ccdf_{metric_col}_{tag}_{safe_name(proto)}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"  -> saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Task 1d (NIDD): CCDF of header and payload length per app protocol."
    )
    parser.add_argument(
        "--nidd_root",
        default=DEFAULT_NIDD_ROOT,
        help="Root folder containing 5G-NIDD CSV/PCAP files",
    )
    parser.add_argument(
        "--out_dir",
        default="nidd_task1d_plots",
        help="Output directory for CCDF plots",
    )
    args = parser.parse_args()

    root = args.nidd_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing 5G-NIDD dataset from: {root}")
    df = load_nidd_packets(root)

    if df is None or df.empty:
        print("[NIDD-1d] DataFrame is empty, nothing to analyze.")
        return

    required_cols = {"header_len", "app_len", "app_proto", "is_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[NIDD-1d] Missing columns in DataFrame: {missing}")
        print("Check core_nidd.py to ensure these fields are created.")
        return

    print(f"[NIDD-1d] Loaded {len(df):,} packets into DataFrame for CCDF computation.")

    # 1) Header length CCDF (with & without attacks)
    plot_ccdf_for_metric(
        df=df,
        metric_col="header_len",
        metric_label="Header length",
        out_dir=out_dir,
        with_attack_flag=True,
    )
    plot_ccdf_for_metric(
        df=df,
        metric_col="header_len",
        metric_label="Header length",
        out_dir=out_dir,
        with_attack_flag=False,
    )

    # 2) Application payload length CCDF (with & without attacks)
    plot_ccdf_for_metric(
        df=df,
        metric_col="app_len",
        metric_label="Application payload length",
        out_dir=out_dir,
        with_attack_flag=True,
    )
    plot_ccdf_for_metric(
        df=df,
        metric_col="app_len",
        metric_label="Application payload length",
        out_dir=out_dir,
        with_attack_flag=False,
    )

    print("Task 1d (NIDD) CCDF plots generated.")
    print(f"Plots saved in folder: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

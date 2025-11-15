"""
Task 1d (5G-NIDD) – CCDFs for header length & application payload length

Plots *complementary* CDFs (CCDF = P(X >= x)) for:
  - header_len (bytes) per application-layer protocol
  - app_len    (bytes) per application-layer protocol
in two scenarios:
  - WITH attacks   (is_attack == 1)
  - WITHOUT attacks (is_attack == 0)
শর্ত: CCDF হিসাব করতে কোনো ready-made library ব্যবহার করা যাবে না,
মানে নিজে sort + counting দিয়ে কম্পিউট করতে হবে। এখানে ঠিক সেটাই করছি। """

import argparse
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT

# ---------- small helpers ----------

def compute_ccdf(values: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ values: 1D Series of non-negative lengths (header_len বা app_len)
    Returns:
        x: sorted unique values
        ccdf: P(X >= x) for each x
    কোনো external CCDF library ব্যবহার করছি না:
      1) values sort করি
      2) unique value + count বের করি
      3) reverse cumulative count করে ভাগ করি total দিয়ে """
    arr = values.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return None

    # non-negative only (length)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return None

    # sort ascending
    arr.sort()

    # unique value + frequency
    uniq, counts = np.unique(arr, return_counts=True)

    # cumulative count from right: for CCDF P(X >= x)
    # উদাহরণ: counts = [c0, c1, c2]
    # cum_rev = [c0+c1+c2, c1+c2, c2]
    cum_rev = counts[::-1].cumsum()[::-1]

    ccdf = cum_rev / float(arr.size)

    return uniq, ccdf


def plot_ccdf_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    title_prefix: str,
    outfile_prefix: str,
    top_k_app: int = 5,
) -> None:
    """ metric_col: "header_len" অথবা "app_len"
    title_prefix: প্লটের টাইটেলে prefix হিসেবে use হবে
    outfile_prefix: output ফাইলের নামের prefix (e.g. "nidd_header" / "nidd_payload")
    top_k_app: সবচেয়ে বেশি frequent application protocols এর মধ্যে কতটা show করব """

    if metric_col not in df.columns:
        print(f"[1d] Column '{metric_col}' not found in DataFrame, skipping.")
        return

    # Clean basic
    df = df.copy()
    df["app_proto"] = df["app_proto"].fillna("UNKNOWN").astype(str)
    df["is_attack"] = df["is_attack"].astype(bool)

    # WITH attacks / WITHOUT attacks
    subsets = {
        "with_attacks":  df[df["is_attack"]],
        "without_attacks": df[~df["is_attack"]],
    }

    for mode, sub in subsets.items():
        if sub.empty:
            print(f"[1d] Subset '{mode}' empty for metric '{metric_col}', skipping plot.")
            continue

        # সবচেয়ে বেশি packet থাকা app_proto থেকে top_k_app বেছে নিই
        app_counts = sub["app_proto"].value_counts()
        top_apps = list(app_counts.head(top_k_app).index)

        print(f"[1d] {metric_col} – {mode}: plotting top {len(top_apps)} app_proto:", top_apps)

        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        for app in top_apps:
            vals = sub.loc[sub["app_proto"] == app, metric_col]
            ccdf_data = compute_ccdf(vals)
            if ccdf_data is None:
                continue

            x, ccdf = ccdf_data
            # log-x scale useful for length distributions
            ax.step(x, ccdf, where="post", label=app)

        ax.set_xscale("log")
        ax.set_xlabel(f"{metric_col} (bytes)")
        ax.set_ylabel("CCDF  P(X ≥ x)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{title_prefix} – {mode.replace('_', ' ').title()}")

        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        outfile = f"{outfile_prefix}_{metric_col}_{mode}.png"
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)
        print(f"[1d] Saved plot: {outfile}")

        # চাইলে plt.show() দিতে পারো; report বানানোর সময় সাধারণত শুধু save করাই যথেষ্ট
        plt.close()


# ---------- main analysis ----------

def analyze_1d_nidd(df: pd.DataFrame) -> None:
    """
    Main driver for Task 1d – 5G-NIDD
    """

    if df is None or df.empty:
        print("[NIDD-1d] DataFrame is empty, nothing to analyze.")
        return

    required = {"app_proto", "is_attack", "header_len", "app_len"}
    missing = required - set(df.columns)
    if missing:
        print(f"[NIDD-1d] Missing columns {missing} in DataFrame.")
        print("Ensure core_nidd.py creates 'header_len' and 'app_len' like in PFCP core.")
        return

    print(f"[NIDD-1d] Total packets in DataFrame: {len(df):,}")

    # শুধু positive length এর উপর কাজ করব
    df = df[(df["header_len"] >= 0) & (df["app_len"] >= 0)].copy()
    print(f"[NIDD-1d] Packets with valid header/app lengths: {len(df):,}")

    # Header length CCDFs
    plot_ccdf_for_metric(
        df,
        metric_col="header_len",
        title_prefix="5G-NIDD Header Length CCDF by App Protocol",
        outfile_prefix="nidd_header",
        top_k_app=5,   # চাইলে বাড়াতে/কমাতে পারো
    )

    # App payload length CCDFs
    plot_ccdf_for_metric(
        df,
        metric_col="app_len",
        title_prefix="5G-NIDD App Payload Length CCDF by App Protocol",
        outfile_prefix="nidd_payload",
        top_k_app=5,
    )

    print("End of Task 1d (5G-NIDD CCDFs)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1d (5G-NIDD): CCDFs of header & payload length per app protocol."
    )
    parser.add_argument(
        "-r",
        "--root",
        default=DEFAULT_NIDD_ROOT,
        help="Path to 5G-NIDD root folder (default: %(default)s)",
    )
    parser.add_argument(
        "--max-rows-csv",
        type=int,
        default=0,
        help="Max rows per CSV when building flow_dict (0 = full, default: %(default)s)",
    )
    parser.add_argument(
        "--max-pkts-pcap",
        type=int,
        default=None,
        help="Max packets per pcap (for quick testing; default: all)",
    )

    args = parser.parse_args()

    print(f"Processing 5G-NIDD dataset from: {args.root}")

    df = load_nidd_packets(
        root=args.root,
        max_rows_per_csv=args.max_rows_csv if args.max_rows_csv > 0 else None,
        max_packets_per_pcap=args.max_pkts_pcap,
    )

    analyze_1d_nidd(df)

if __name__ == "__main__":
    main()

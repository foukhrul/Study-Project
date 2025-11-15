"""Packet length stats per application protocol
- Inter-arrival time (IAT) stats, comparing
  * packets WITH attacks
  * packets WITHOUT attacks """

import argparse
import pandas as pd

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT


def length_stats_by_app(df: pd.DataFrame, title: str) -> None:
    """Per-application protocol packet length statistics."""
    print(f"\n[Packet length stats] {title}")

    if df.empty:
        print("  (no packets in this subset)")
        return

    grp = df.groupby("app_proto")["pkt_len"]

    stats_df = pd.DataFrame(
        {
            "count": grp.count(),
            "mean": grp.mean(),
            "std": grp.std(ddof=1),
            "median": grp.median(),
        }
    )

    stats_df = stats_df.sort_values("count", ascending=False)

    print(stats_df.to_string(float_format=lambda x: f"{x:,.3f}"))


def iat_stats(df: pd.DataFrame, title: str) -> None:
    """ Inter-arrival time (IAT) statistics:- unordered host pair (min(src_ip, dst_ip), max(...))
    - then diff ts within each pair """
    print(f"\n[Inter-arrival stats] {title}")

    if df.empty:
        print("  (no packets in this subset)")
        return

    tmp = df[["src_ip", "dst_ip", "ts"]].copy()
    tmp["a"] = tmp[["src_ip", "dst_ip"]].min(axis=1)
    tmp["b"] = tmp[["src_ip", "dst_ip"]].max(axis=1)

    tmp = tmp.sort_values(["a", "b", "ts"])

    iats = tmp.groupby(["a", "b"])["ts"].diff().dropna()
    iats = iats[iats >= 0]

    if iats.empty:
        print("  No inter-arrival intervals could be computed.")
        return

    print(f"  count : {len(iats):,}")
    print(f"  mean  : {iats.mean():.6f}  seconds")
    print(f"  std   : {iats.std(ddof=1):.6f}  seconds")
    print(f"  median: {iats.median():.6f}  seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1b (5G-NIDD): packet length and inter-arrival time statistics."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_NIDD_ROOT,
        help="Root folder containing 5G-NIDD CSV/PCAP files "
             "(default: 5G_NIDD next to Code/)",
    )
    args = parser.parse_args()

    print(f"Processing 5G-NIDD dataset from: {args.root}")

    # full dataset: no max_rows / max_pkts limit
    df = load_nidd_packets(root=args.root)

    if df is None or df.empty:
        print("DataFrame is empty, nothing to analyze.")
        return

    required_cols = {
        "ts",
        "pkt_len",
        "app_proto",
        "is_attack",
        "src_ip",
        "dst_ip",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns in DataFrame: {missing}")
        print("Check core_nidd.py to ensure these fields are created.")
        return

    # remove non-positive lengths
    df = df[df["pkt_len"] > 0].copy()

    if df.empty:
        print("No packets with positive length — nothing to analyze.")
        return

    # split WITH vs WITHOUT attacks
    df_attack = df[df["is_attack"].astype(bool)].copy()
    df_no_attack = df[~df["is_attack"].astype(bool)].copy()

    # --------- Packet length stats by application-layer protocol ---------
    length_stats_by_app(
        df_attack,
        "Application protocols – WITH attacks (Attack packets only)",
    )
    length_stats_by_app(
        df_no_attack,
        "Application protocols – WITHOUT attacks (Normal+Unlabeled packets)",
    )

    # ----------------- Inter-arrival time stats -----------------
    iat_stats(
        df_attack,
        "WITH attacks (only Attack packets, unordered host pairs)",
    )
    iat_stats(
        df_no_attack,
        "WITHOUT attacks (Normal+Unlabeled packets, unordered host pairs)",
    )

    print("\nEnd of Task 1b (5G-NIDD)")


if __name__ == "__main__":
    main()

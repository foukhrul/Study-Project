#!/usr/bin/env python3
"""How many pairs of hosts are in the dataset?
- For how many of these pairs are there packets with attacks?
- Inter-arrival time (IAT) statistics between consecutive packets
  that are:
    * exchanged between the SAME pair of hosts
    * transmitted over the SAME application-layer protocol type
  separately:
    * WITHOUT attacks (only non-attack packets)
    * WITH attacks (only attack packets)
IAT stats reported: count, mean, standard deviation, median """

import argparse
import pandas as pd

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT


def print_header(title: str) -> None:
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


def host_pair_stats(df: pd.DataFrame) -> None:
    """ Compute: total number of unordered host pairs
      - number of host pairs that have at least one attack packet"""

    required = {"src_ip", "dst_ip", "is_attack"}
    missing = required - set(df.columns)
    if missing:
        print(f"[NIDD-1c] Missing required columns for host-pair stats: {missing}")
        return

    hp = df[["src_ip", "dst_ip", "is_attack"]].copy()

    # unordered pair (min, max)
    hp["a"] = hp[["src_ip", "dst_ip"]].min(axis=1)
    hp["b"] = hp[["src_ip", "dst_ip"]].max(axis=1)

    group_hp = hp.groupby(["a", "b"])

    total_pairs = len(group_hp)
    pairs_with_attack = (group_hp["is_attack"].sum() > 0).sum()

    print_header("Task 1c: Host-pair statistics (5G-NIDD)")
    print(f"Total host pairs:                 {total_pairs:,}")
    print(f"Host pairs with at least 1 attack: {pairs_with_attack:,}")
    if total_pairs > 0:
        frac = 100.0 * pairs_with_attack / total_pairs
        print(f"Fraction of host pairs with attacks: {frac:.2f}%")
    print()


def iat_stats_by_hostpair_app(df: pd.DataFrame, title: str) -> None:
    """ Inter-arrival time stats between consecutive packets that:
      - belong to the same unordered host pair
      - have the same application-layer protocol (app_proto)
    df: subset of packets (either only non-attack or only attack)
        must contain columns: src_ip, dst_ip, app_proto, ts """

    required = {"src_ip", "dst_ip", "app_proto", "ts"}
    missing = required - set(df.columns)
    if missing:
        print(f"[NIDD-1c] Missing required columns for IAT stats: {missing}")
        return

    print_header(title)

    if df.empty:
        print("No packets in this subset — cannot compute IAT.")
        return

    tmp = df[["src_ip", "dst_ip", "app_proto", "ts"]].copy()

    # unordered host pair
    tmp["a"] = tmp[["src_ip", "dst_ip"]].min(axis=1)
    tmp["b"] = tmp[["src_ip", "dst_ip"]].max(axis=1)

    # sort by (host pair, app_proto, timestamp)
    tmp_sorted = tmp.sort_values(["a", "b", "app_proto", "ts"])

    # IAT = difference between consecutive timestamps inside each (host_pair, app_proto) group
    iats = (
        tmp_sorted
        .groupby(["a", "b", "app_proto"])["ts"]
        .diff()
        .dropna()
    )

    # নিরাপত্তার জন্য negative interval বাদ দেই (ঘড়ি / ordering সমস্যা থাকলে)
    iats = iats[iats >= 0]

    if iats.empty:
        print("No valid inter-arrival intervals (need at least 2 packets per group).")
        return

    count = int(iats.shape[0])
    mean = float(iats.mean())
    std = float(iats.std(ddof=1)) if count > 1 else 0.0
    median = float(iats.median())

    print(f"Number of IAT samples : {count:,}")
    print(f"Mean IAT              : {mean:.6f} seconds")
    print(f"Std dev of IAT        : {std:.6f} seconds")
    print(f"Median IAT            : {median:.6f} seconds")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1c (5G-NIDD): host-pair count and IAT statistics."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_NIDD_ROOT,
        help="Path to 5G-NIDD root folder (default: %(default)s)",
    )

    args = parser.parse_args()

    print(f"Processing 5G-NIDD dataset from: {args.root}")

    # Full dataset; no row/packet limits from here
    df = load_nidd_packets(root=args.root)

    if df is None or df.empty:
        print("[NIDD-1c] DataFrame is empty, nothing to analyze.")
        return

    # Basic cleaning
    df = df.copy()
    df["is_attack"] = df["is_attack"].astype(bool)
    df["app_proto"] = df["app_proto"].fillna("UNKNOWN")

    # --- 1) Host pair stats ---
    host_pair_stats(df)

    # --- 2) IAT stats (without attacks) ---
    df_no_attack = df[~df["is_attack"]].copy()
    iat_stats_by_hostpair_app(
        df_no_attack,
        "Inter-arrival time – WITHOUT attacks "
        "(same host pair, same application protocol, non-attack packets only)",
    )

    # --- 3) IAT stats (with attacks) ---
    df_attack = df[df["is_attack"]].copy()
    iat_stats_by_hostpair_app(
        df_attack,
        "Inter-arrival time – WITH attacks "
        "(same host pair, same application protocol, attack packets only)",
    )

    print("End of Task 1c (5G-NIDD)")


if __name__ == "__main__":
    main()

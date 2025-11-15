#!/usr/bin/env python3
"""
1) Only packets that contain application data (has_app_data == True):

   For EACH application-layer protocol (app_proto) and for EACH of:
     • packets WITH attacks (is_attack == True)
     • packets WITHOUT attacks (is_attack == False; i.e., Normal + Unlabeled)
   -> compute: average total packet length (bytes)
        - standard deviation of packet length (bytes)
        - median packet length (bytes)

2) For ALL packets (no has_app_data filter): For EACH of: packets WITH attacks. packets WITHOUT attacks
   considering only packets that are exchanged between the same pair of hosts (same host_pair, irrespective of direction):

   -> compute: average inter-arrival time between two consecutive packets (seconds)
        - standard deviation of inter-arrival time (seconds), median inter-arrival time (seconds)
"""

import argparse
import numpy as np
import pandas as pd

from core_pfcp import load_pfcp_packets, DEFAULT_PFCP_ROOT


def length_stats_by_app_proto(df: pd.DataFrame, title: str) -> None:
    """
    df: subset where has_app_data == True and is_attack is fixed (True/False).
    Computes count, mean, std, median of pkt_len per app_proto.
    """
    if df.empty:
        print(f"\n[Packet length stats] {title}")
        print("  (no packets in this subset)")
        return

    grouped = df.groupby("app_proto")["pkt_len"]
    stats = grouped.agg(["count", "mean", "std"])
    stats["median"] = grouped.median()

    # nicer formatting: sort by count (descending)
    stats = stats.sort_values("count", ascending=False)

    # print section
    print(f"\n[Packet length stats] {title}")
    # use to_string for aligned table
    print(stats.to_string(float_format=lambda x: f"{x:,.3f}"))


def iat_stats_by_host_pair(df: pd.DataFrame, title: str) -> None:
    """
    Compute inter-arrival times between consecutive packets that belong to the
    same host_pair (unordered src/dst), then aggregate all intervals globally.
    df: subset where is_attack is fixed (True/False).
    """
    if df.empty:
        print(f"\n[Inter-arrival stats] {title}")
        print("  (no packets in this subset)")
        return

    # sort by host_pair then timestamp
    work = df[["host_pair", "ts"]].copy()
    work = work.sort_values(["host_pair", "ts"])

    # diff per host_pair
    work["iat"] = work.groupby("host_pair")["ts"].diff()

    # keep only positive intervals (ignore NaN and 0 or negative)
    iats = work["iat"].dropna()
    iats = iats[iats > 0]

    if iats.empty:
        print(f"\n[Inter-arrival stats] {title}")
        print("  (no valid inter-arrival intervals)")
        return

    count = len(iats)
    mean = float(iats.mean())
    std = float(iats.std(ddof=1)) if count > 1 else float("nan")
    median = float(iats.median())

    print(f"\n[Inter-arrival stats] {title}")
    print(f"  #Intervals: {count:,}")
    print(f"  Mean   IAT: {mean:.6f} seconds")
    print(f"  Std    IAT: {std:.6f} seconds")
    print(f"  Median IAT: {median:.6f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Task 1b (5G-PFCP): "
                    "packet length & inter-arrival time statistics."
    )
    parser.add_argument(
        "--pfcp_root",
        default=DEFAULT_PFCP_ROOT,
        help="Root folder containing 5G-PFCP PCAP files",
    )
    args = parser.parse_args()

    root = args.pfcp_root
    print(f"Processing 5G-PFCP dataset from: {root}")

    df = load_pfcp_packets(root)

    if df is None or df.empty:
        print("[PFCP-1b] DataFrame is empty, nothing to analyze.")
        return

    required_cols = {
        "pkt_len",       # total packet length (frame.len)
        "app_proto",     # application-layer protocol
        "is_attack",     # True / False
        "has_app_data",  # True if there is application-layer payload
        "ts",            # timestamp (frame.time_epoch)
        "host_pair",     # unordered pair of hosts (from core_pfcp)
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[PFCP-1b] Missing required columns: {missing}")
        print("Check core_pfcp.py to ensure these fields are created.")
        return

    print(f"[PFCP-1b] Loaded {len(df):,} packets into DataFrame.")

    # 1) Packet length statistics (only packets with app data)
    #    for each app_proto, with & without attacks

    df_app = df[df["has_app_data"] == True].copy()

    df_with_atk = df_app[df_app["is_attack"] == True]
    df_without_atk = df_app[df_app["is_attack"] == False]  # Normal + Unlabeled

    length_stats_by_app_proto(
        df_with_atk,
        "Application protocols – WITH attacks (only Attack packets, has_app_data=True)",
    )
    length_stats_by_app_proto(
        df_without_atk,
        "Application protocols – WITHOUT attacks (Normal+Unlabeled, has_app_data=True)",
    )

    # 2) Inter-arrival time statistics (ALL packets)
    #    intervals between consecutive pkts from same host_pair
    #    with & without attacks
    df_atk_iat = df[df["is_attack"] == True].copy()
    df_natk_iat = df[df["is_attack"] == False].copy()

    iat_stats_by_host_pair(
        df_atk_iat,
        "WITH attacks (Attack packets only, same host_pair)",
    )
    iat_stats_by_host_pair(
        df_natk_iat,
        "WITHOUT attacks (Normal+Unlabeled packets, same host_pair)",
    )

    print("Task 1b (5G-PFCP) analysis complete.")


if __name__ == "__main__":
    main()

import pandas as pd

from core_pfcp import load_pfcp_packets, DEFAULT_PFCP_ROOT

def length_stats_by_app_proto(df: pd.DataFrame, title: str) -> None:

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
    """ Compute inter-arrival times between consecutive packets that belong to the
    same host_pair (unordered src/dst), then aggregate all intervals globally.."""
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
    root = DEFAULT_PFCP_ROOT
    print(f"Processing 5G-PFCP dataset from: {root}")

    # Load PFCP packets
    df = load_pfcp_packets(root)

    # Required fields must be present
    required_cols = {
        "pkt_len",       # total packet length (frame.len)
        "app_proto",     # application-layer protocol
        "is_attack",     # True / False
        "has_app_data",  # True if PFCP has application payload
        "ts",            # timestamp
        "host_pair",     # unordered src-dst pair
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return

    print(f"Loaded {len(df):,} packets into DataFrame.")

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

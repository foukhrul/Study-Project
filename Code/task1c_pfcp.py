import numpy as np

from core_pfcp import load_pfcp_packets, DEFAULT_PFCP_ROOT

def compute_iat_stats(df_subset, desc):
    """df_subset: DataFrame already filtered (e.g. only attacks / only non-attacks)
    Condition: same host_pair + same app_proto
    Returns: dict with mean, std, median, count """
    if df_subset.empty:
        print(f"[Task 1c – IAT stats] {desc}: no packets in subset.")
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}

    iats = []

    # Group by host pair + application protocol
    grouped = df_subset.groupby(["host_pair", "app_proto"], sort=False)

    for (_, _), group in grouped:
        if len(group) < 2:
            continue

        ts = np.sort(group["ts"].to_numpy())
        diffs = np.diff(ts)

        # this below filter is to avoid small and zero interval:
        diffs = diffs[diffs > 0.0]
        if diffs.size == 0:
            continue

        iats.append(diffs)

    if not iats:
        print(f"[Task 1c – IAT stats] {desc}: no valid intervals found.")
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}

    all_iats = np.concatenate(iats)

    stats = {
        "mean": float(all_iats.mean()),
        "std": float(all_iats.std(ddof=0)),
        "median": float(np.median(all_iats)),
        "count": int(all_iats.size),
    }

    print(f"[Task 1c – IAT stats] {desc}")
    print("  Condition: same host pair + same application-layer protocol")
    print("  Time unit: seconds (frame.time_epoch)")
    print(f"  Average IAT: {stats['mean']:.6f} s")
    print(f"  Std IAT:     {stats['std']:.6f} s")
    print(f"  Median IAT:  {stats['median']:.6f} s")
    print(f"  #Intervals:  {stats['count']:,}")
    print()
    return stats


def main():
    root = DEFAULT_PFCP_ROOT
    print("5G-PFCP – Host Pair & IAT Analysis")
    print(f"Processing 5G-PFCP dataset from: {root}")

    df = load_pfcp_packets(root)

    if df is None or df.empty:
        print("DataFrame is empty, nothing to analyze.")
        return

    required_cols = {"host_pair", "ts", "app_proto", "is_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[PFCP-1c] Missing required columns: {missing}")
        print("Check core_pfcp.py to ensure these fields are created.")
        return

    # -------- Host pair summary --------
    total_pairs = df["host_pair"].nunique()
    pairs_with_attack = df.loc[df["is_attack"], "host_pair"].nunique()
    frac_pairs_with_attack = (
        pairs_with_attack / total_pairs if total_pairs > 0 else 0.0
    )

    print("[Host pairs]")
    print(f"  Total unique host pairs in dataset: {total_pairs}")
    print(f"  Host pairs with at least one Attack packet: {pairs_with_attack}")
    print(f"  Fraction of host pairs with attacks: {frac_pairs_with_attack*100:.2f}%")
    print()

    # -------- IAT stats WITHOUT attacks (Normal + Unlabeled) --------
    df_wo = df[~df["is_attack"]].copy()
    compute_iat_stats(df_wo, desc="WITHOUT attacks")

    # -------- IAT stats WITH attacks (only Attack packets) --------
    df_w = df[df["is_attack"]].copy()
    compute_iat_stats(df_w, desc="WITH attacks")

    print("End of Task 1c (5G-PFCP)")


if __name__ == "__main__":
    main()

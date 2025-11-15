
"""
- load_nidd_packets() দিয়ে CSV + PCAP থেকে merged DataFrame বানায়
- Basic stats print করে:
    * total packets, Normal/Attack/Unlabeled split
    * unique host pairs, কী fraction attack traffic বহন করছে
    * application-layer protocol stats
    * transport-layer protocol stats
    * transport x application pair stats """

import argparse
import pandas as pd

from core_nidd import load_nidd_packets, DEFAULT_NIDD_ROOT


def print_header(title: str):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


def analyze_nidd(df: pd.DataFrame):
    """ df: load_nidd_packets() থেকে পাওয়া DataFrame
    expected columns:  - label ("Normal"/"Attack"/"Unlabeled") - is_attack (0/1)
        - transport (TCP/UDP/ICMP/SCTP/OTHER)
        - app_proto (tshark protocol col, e.g. "GTP/UDP") - src_ip, dst_ip """
    if df.empty:
        print("[NIDD-1a] DataFrame is empty, nothing to analyze.")
        return

    print_header("Task 1a: 5G-NIDD Analysis")

    total = len(df)
    labels = df["label"].fillna("Unlabeled")

    normal_count = (labels == "Normal").sum()
    attack_count = (labels == "Attack").sum()
    unlabeled_count = total - normal_count - attack_count

    def pct(x):
        return 100.0 * x / total if total > 0 else 0.0

    print(f"Total Packets: {total:,}")
    print(f"  Normal:    {normal_count:,} ({pct(normal_count):.2f}%)")
    print(f"  Attack:    {attack_count:,} ({pct(attack_count):.2f}%)")
    print(f"  Unlabeled: {unlabeled_count:,} ({pct(unlabeled_count):.2f}%)")
    print()

    # Host-pair statistics
    # -----------------------
    hp = df[["src_ip", "dst_ip", "is_attack"]].copy()
    hp["a"] = hp[["src_ip", "dst_ip"]].min(axis=1)
    hp["b"] = hp[["src_ip", "dst_ip"]].max(axis=1)

    group_hp = hp.groupby(["a", "b"])
    unique_pairs = len(group_hp)
    pairs_with_attack = (group_hp["is_attack"].sum() > 0).sum()

    frac_attack_pairs = (
        100.0 * pairs_with_attack / unique_pairs if unique_pairs > 0 else 0.0
    )

    print("Host Pairs:")
    print(f"  Unique Pairs: {unique_pairs:,}")
    print(f"  Pairs with Attack Traffic: {pairs_with_attack:,}")
    print(f"  Fraction of host pairs with attacks: {frac_attack_pairs:.2f}%")
    print()

    # Application-layer stats

    print("Application-Layer Protocols (ALL):")
    app = df["app_proto"].fillna("UNKNOWN")
    is_attack = df["is_attack"].astype(bool)

    total_attacks = is_attack.sum()

    app_counts = app.value_counts()
    for proto, cnt in app_counts.items():
        mask_proto = app == proto
        attacks_this_proto = (mask_proto & is_attack).sum()

        if total_attacks > 0:
            attack_share = 100.0 * attacks_this_proto / total_attacks
        else:
            attack_share = 0.0

        attack_rate = 100.0 * attacks_this_proto / cnt if cnt > 0 else 0.0

        print(
            f"  {proto:<24} "
            f"{cnt:>10,} ({100.0*cnt/total:5.2f}%) | "
            f"Attack share: {attack_share:5.2f}% | "
            f"Attack rate: {attack_rate:5.2f}%"
        )
    print()

    # Transport-layer stats
    print("Transport-Layer Protocols:")
    transport = df["transport"].fillna("OTHER")

    t_counts = transport.value_counts()
    for t, cnt in t_counts.items():
        mask_t = transport == t
        attacks_this_t = (mask_t & is_attack).sum()

        if total_attacks > 0:
            attack_share = 100.0 * attacks_this_t / total_attacks
        else:
            attack_share = 0.0

        attack_rate = 100.0 * attacks_this_t / cnt if cnt > 0 else 0.0

        print(
            f"  {t:<12} "
            f"{cnt:>10,} ({100.0*cnt/total:5.2f}%) | "
            f"Attack share: {attack_share:5.2f}% | "
            f"Attack rate: {attack_rate:5.2f}%"
        )
    print()

    # Transport × Application
    print("Transport x Application Pairs (ALL):")
    combo_counts = df.groupby(["transport", "app_proto"]).size().sort_values(
        ascending=False
    )

    for (t, proto), cnt in combo_counts.items():
        mask_combo = (transport == t) & (app == proto)
        attacks_this = (mask_combo & is_attack).sum()
        attack_rate = 100.0 * attacks_this / cnt if cnt > 0 else 0.0

        print(
            f"  {t:<10} + {proto:<24} "
            f"{cnt:>10,} ({100.0*cnt/total:5.2f}%) | "
            f"Attack: {attack_rate:5.2f}%"
        )


    print("End of Task 1a (5G-NIDD)")


def main():
    parser = argparse.ArgumentParser(
        description="Task 1a: 5G-NIDD dataset analysis",
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
        help="Max rows per CSV to read when building flow_dict (0 = full, default: %(default)s)",
    )
    parser.add_argument(
        "--max-pkts-pcap",
        type=int,
        default=None,
        help="Max packets per pcap to parse (for quick testing; default: all)",
    )

    args = parser.parse_args()

    print(f"Processing 5G-NIDD dataset from: {args.root}")
    print(f"max_rows_per_csv = {args.max_rows_csv}")

    df = load_nidd_packets(
        root=args.root,
        max_rows_per_csv=args.max_rows_csv if args.max_rows_csv > 0 else None,
        max_packets_per_pcap=args.max_pkts_pcap,
    )

    if df.empty:
        print("DataFrame is empty, nothing to analyze.")
        return

    analyze_nidd(df)


if __name__ == "__main__":
    main()

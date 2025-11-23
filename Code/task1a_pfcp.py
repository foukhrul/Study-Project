import pandas as pd

from core_pfcp import load_pfcp_packets

DEFAULT_PFCP_ROOT = "/Users/eshtisabbiroutlook.com/PycharmProjects/5G_Project/5G_PFCP"

def report_task1a_pfcp(df: pd.DataFrame):
    """Prints Task 1a statistics for the 5G-PFCP dataset."""
    if df is None or df.empty:
        print("[DataFrame is empty, nothing to analyze.")
        return

    required_cols = {
        "label",
        "is_attack",
        "app_proto",
        "transport_proto",
        "host_pair",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[Missing required columns: {missing}")
        print("Check core_pfcp.py to ensure these fields are created.")
        return

    total_pkts = len(df)

    print("Task 1a – 5G-PFCP: Basic Traffic & Protocol Statistics")
    print(f"Total packets in PFCP dataset: {total_pkts:,}")

    # ---------------- Label distribution ----------------
    label_counts = df["label"].value_counts(dropna=False)
    print("\n[Label distribution]")
    for label, cnt in label_counts.items():
        frac = cnt / total_pkts if total_pkts > 0 else 0.0
        print(f"  {str(label):10s}: {cnt:10,} ({frac:.2%})")

    # ---------------- Host pair statistics ----------------
    print("\n[Host pairs]")
    # host_pair field is a tuple (src_ip, dst_ip) sorted in core_pfcp
    unique_pairs = df["host_pair"].nunique()
    # host pairs where at least one packet is attack
    attack_pairs = df.loc[df["is_attack"] == True, "host_pair"].nunique()

    print(f"  Unique host pairs           : {unique_pairs:,}")
    print(f"  Host pairs with attack traffic: {attack_pairs:,}")
    if unique_pairs > 0:
        print(f"  Fraction of host pairs with attacks: {attack_pairs / unique_pairs:.2%}")

    # ---------------- Application-layer protocols ----------------

    print("\n[Application-layer protocols]")
    app_counts = df["app_proto"].fillna("UNKNOWN").value_counts()
    print(f"Number of application-layer protocols: {len(app_counts)}")
    app_group = df.groupby("app_proto", dropna=False)
    app_stats = []

    for proto, g in app_group:
        count = len(g)
        frac_total = count / total_pkts if total_pkts > 0 else 0.0

        attack_cnt = g["is_attack"].sum()  # True=1, False=0
        attack_share = attack_cnt / label_counts.get("Attack", 1) if label_counts.get("Attack", 0) > 0 else 0.0
        attack_rate = attack_cnt / count if count > 0 else 0.0

        app_stats.append(
            {
                "app_proto": proto if pd.notna(proto) else "UNKNOWN",
                "count": count,
                "frac_total": frac_total,
                "attack_count": int(attack_cnt),
                "attack_share": attack_share,
                "attack_rate": attack_rate,
            }
        )

    # sort by count descending
    app_stats.sort(key=lambda x: x["count"], reverse=True)

    print(
        "\nexisted application-layer protocols , "
        "and fraction / attack share / attack rate they have:"
    )
    print(
        f"{'App Proto':25s} {'Count':>12s} {'Frac(total)':>12s} "
        f"{'AtkCount':>10s} {'AtkShare':>10s} {'AtkRate':>10s}"
    )
    for s in app_stats:
        print(
            f"{s['app_proto'][:24]:25s} "
            f"{s['count']:12,} "
            f"{s['frac_total']:12.2%} "
            f"{s['attack_count']:10,d} "
            f"{s['attack_share']:10.2%} "
            f"{s['attack_rate']:10.2%}"
        )

    # ---------------- Transport-layer protocols ----------------
    print("\n[Transport-layer protocols]")
    trans_group = df.groupby("transport_proto", dropna=False)
    trans_stats = []

    for proto, g in trans_group:
        count = len(g)
        frac_total = count / total_pkts if total_pkts > 0 else 0.0

        attack_cnt = g["is_attack"].sum()
        attack_share = attack_cnt / label_counts.get("Attack", 1) if label_counts.get("Attack", 0) > 0 else 0.0
        attack_rate = attack_cnt / count if count > 0 else 0.0

        trans_stats.append(
            {
                "transport_proto": proto if pd.notna(proto) else "UNKNOWN",
                "count": count,
                "frac_total": frac_total,
                "attack_count": int(attack_cnt),
                "attack_share": attack_share,
                "attack_rate": attack_rate,
            }
        )

    trans_stats.sort(key=lambda x: x["count"], reverse=True)

    print(
        "\nHow many and which transport-layer protocols exist, "
        "and what fraction / attack share / attack rate they have:"
    )
    print(
        f"{'Transport':15s} {'Count':>12s} {'Frac(total)':>12s} "
        f"{'AtkCount':>10s} {'AtkShare':>10s} {'AtkRate':>10s}"
    )
    for s in trans_stats:
        print(
            f"{s['transport_proto'][:14]:15s} "
            f"{s['count']:12,} "
            f"{s['frac_total']:12.2%} "
            f"{s['attack_count']:10,d} "
            f"{s['attack_share']:10.2%} "
            f"{s['attack_rate']:10.2%}"
        )

    # ---------------- Transport × Application matrix (optional but nice) ----------------
    print("\n[Transport × Application protocol pairs]")
    pair_counts = df.groupby(["transport_proto", "app_proto"]).size().reset_index(name="count")
    pair_counts = pair_counts.sort_values("count", ascending=False)

    print(
        f"{'Transport':15s} {'App Proto':25s} "
        f"{'Count':>12s} {'Frac(total)':>12s}"
    )
    for _, row in pair_counts.iterrows():
        trans = str(row["transport_proto"])
        app = str(row["app_proto"])
        cnt = int(row["count"])
        frac = cnt / total_pkts if total_pkts > 0 else 0.0
        print(
            f"{trans[:14]:15s} {app[:24]:25s} "
            f"{cnt:12,} {frac:12.2%}"
        )

def main():
    df = load_pfcp_packets(DEFAULT_PFCP_ROOT)
    report_task1a_pfcp(df)


if __name__ == "__main__":
    main()

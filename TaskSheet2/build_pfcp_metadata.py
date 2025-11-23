#!/usr/bin/env python3
"""
Build PFCP real + synthetic metadata CSVs for TaskSheet2 Task 1.

Creates 4 files in TaskSheet2/:

  pfcp_real_dirsize.csv
  pfcp_real_iat.csv
  pfcp_synth_dirsize.csv
  pfcp_synth_iat.csv
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------- Paths (hard-coded for your project) ----------

# .../5G_Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PFCP_ROOT = os.path.join(PROJECT_ROOT, "5G_PFCP")
SYNTH_DIR = os.path.join(PROJECT_ROOT, "Code", "task3_synth_pfcp_1min")
OUT_DIR = os.path.join(PROJECT_ROOT, "TaskSheet2")

# ---------- Import core_pfcp from Code/ ----------

CODE_DIR = os.path.join(PROJECT_ROOT, "Code")
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from core_pfcp import load_pfcp_packets  # type: ignore


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def choose_protocol(row: pd.Series) -> str:
    """Prefer app_proto, then transport_proto, else ip_proto/UNKNOWN."""
    app = str(row.get("app_proto", "")).strip()
    if app and app.lower() != "nan":
        return app
    transp = str(row.get("transport_proto", "")).strip()
    if transp and transp.lower() != "nan":
        return transp
    ip_proto = str(row.get("ip_proto", "")).strip()
    if ip_proto:
        return ip_proto
    return "UNKNOWN"


def compute_signed_size(row: pd.Series) -> float:
    """Encode direction+size as signed length using host_pair ordering."""
    src = str(row["src_ip"]).strip()
    dst = str(row["dst_ip"]).strip()
    pkt_len = float(row["pkt_len"])

    a, b = sorted([src, dst])
    sign = 1.0 if src == a else -1.0
    return sign * pkt_len


def build_real_metadata_dirsize(df: pd.DataFrame) -> pd.DataFrame:
    """Real: direction+size → columns: pair, protocol, value."""
    print("\n[REAL] Building direction+size metadata ...")

    mask = (~df["is_attack"]) & (df["has_app_data"])
    sub = df.loc[mask].copy()
    print(f"  Normal + has_app_data packets: {len(sub):,}")

    sub["protocol"] = sub.apply(choose_protocol, axis=1)
    sub["value"] = sub.apply(compute_signed_size, axis=1)

    meta = sub[["host_pair", "protocol", "value"]].copy()
    meta = meta.rename(columns={"host_pair": "pair"})

    print(
        f"  Rows: {len(meta):,}, "
        f"pairs: {meta['pair'].nunique():,}, "
        f"protocols: {meta['protocol'].nunique():,}"
    )
    return meta


def build_real_metadata_iat(df: pd.DataFrame) -> pd.DataFrame:
    """Real: IAT → columns: pair, protocol, value (seconds)."""
    print("\n[REAL] Building IAT metadata ...")

    mask = (~df["is_attack"]) & (df["has_app_data"])
    sub = df.loc[mask].copy()
    print(f"  Normal + has_app_data packets: {len(sub):,}")

    sub["protocol"] = sub.apply(choose_protocol, axis=1)

    records: List[Tuple[str, str, float]] = []

    grouped = sub.groupby(["host_pair", "protocol"])
    for (pair, proto), g in grouped:
        g_sorted = g.sort_values("ts")
        ts = g_sorted["ts"].values.astype(float)
        if len(ts) < 2:
            continue
        iats = np.diff(ts)
        iats = iats[iats > 0.0]
        for v in iats:
            records.append((pair, proto, float(v)))

    if not records:
        print("  WARNING: no IAT values found.")
        return pd.DataFrame(columns=["pair", "protocol", "value"])

    meta = pd.DataFrame(records, columns=["pair", "protocol", "value"])
    print(
        f"  Rows: {len(meta):,}, "
        f"pairs: {meta['pair'].nunique():,}, "
        f"protocols: {meta['protocol'].nunique():,}"
    )
    return meta


def build_synth_metadata_from_flows(
    synth_dir: str,
    real_pairs_protocols: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic flows (size, direction, iat) → same metadata format.
    We cycle through real (pair, protocol) list to assign labels.
    """
    print("\n[SYNTH] Building metadata from synthetic flows ...")
    csv_files = [
        os.path.join(synth_dir, f)
        for f in sorted(os.listdir(synth_dir))
        if f.endswith(".csv")
    ]
    if not csv_files:
        print(f"  No synthetic CSVs found in {synth_dir}")
        return (
            pd.DataFrame(columns=["pair", "protocol", "value"]),
            pd.DataFrame(columns=["pair", "protocol", "value"]),
        )

    if not real_pairs_protocols:
        real_pairs_protocols = [("synthetic_pair", "PFCP")]

    dirsize_records: List[Tuple[str, str, float]] = []
    iat_records: List[Tuple[str, str, float]] = []

    for idx, path in enumerate(csv_files):
        df = pd.read_csv(path)
        if not {"size", "direction", "iat"}.issubset(df.columns):
            print(f"  WARNING: {path} missing size/direction/iat, skipping")
            continue

        pair, proto = real_pairs_protocols[idx % len(real_pairs_protocols)]

        sizes = df["size"].values.astype(float)
        dirs = df["direction"].values.astype(float)
        vals_dirsize = sizes * dirs
        for v in vals_dirsize:
            dirsize_records.append((pair, proto, float(v)))

        iats = df["iat"].values.astype(float)
        iats = iats[iats > 0.0]
        for v in iats:
            iat_records.append((pair, proto, float(v)))

    dirsize_df = pd.DataFrame(dirsize_records, columns=["pair", "protocol", "value"])
    iat_df = pd.DataFrame(iat_records, columns=["pair", "protocol", "value"])

    print(
        f"  Synthetic dir+size rows: {len(dirsize_df):,}, "
        f"pairs: {dirsize_df['pair'].nunique():,}, "
        f"protocols: {dirsize_df['protocol'].nunique():,}"
    )
    print(
        f"  Synthetic IAT rows: {len(iat_df):,}, "
        f"pairs: {iat_df['pair'].nunique():,}, "
        f"protocols: {iat_df['protocol'].nunique():,}"
    )
    return dirsize_df, iat_df


def main() -> None:
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("PFCP_ROOT    :", PFCP_ROOT)
    print("SYNTH_DIR    :", SYNTH_DIR)
    print("OUT_DIR      :", OUT_DIR)

    ensure_dir(OUT_DIR)

    # ---- REAL PFCP ----
    print("\n[REAL] Loading PFCP packets ...")
    df_pfcp = load_pfcp_packets(PFCP_ROOT)
    print(f"[REAL] Total PFCP packets: {len(df_pfcp):,}")

    real_dir_df = build_real_metadata_dirsize(df_pfcp)
    real_iat_df = build_real_metadata_iat(df_pfcp)

    real_dir_csv = os.path.join(OUT_DIR, "pfcp_real_dirsize.csv")
    real_iat_csv = os.path.join(OUT_DIR, "pfcp_real_iat.csv")

    real_dir_df.to_csv(real_dir_csv, index=False)
    real_iat_df.to_csv(real_iat_csv, index=False)

    print(f"[REAL] Saved: {real_dir_csv}")
    print(f"[REAL] Saved: {real_iat_csv}")

    real_pairs_protocols = sorted(set(zip(real_dir_df["pair"], real_dir_df["protocol"])))

    # ---- SYNTHETIC PFCP ----
    synth_dir_df, synth_iat_df = build_synth_metadata_from_flows(
        SYNTH_DIR, real_pairs_protocols
    )

    synth_dir_csv = os.path.join(OUT_DIR, "pfcp_synth_dirsize.csv")
    synth_iat_csv = os.path.join(OUT_DIR, "pfcp_synth_iat.csv")

    synth_dir_df.to_csv(synth_dir_csv, index=False)
    synth_iat_df.to_csv(synth_iat_csv, index=False)

    print(f"[SYNTH] Saved: {synth_dir_csv}")
    print(f"[SYNTH] Saved: {synth_iat_csv}")

    print("\n[Done] All four PFCP metadata CSVs have been created.")


if __name__ == "__main__":
    main()

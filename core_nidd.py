import os
import sys
import subprocess
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

# ----- Default NIDD root (Code/ থেকে এক লেভেল উপরে গিয়ে 5G_NIDD) -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NIDD_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "5G_NIDD"))

# Protocol / Port helpers

# প্রোটোকল নাম → নম্বর (IPv4 proto field এর মত)
PROTO_MAP = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17,
    "ipv6-icmp": 58,
    "sctp": 132,
    "other": 0,
}

# কিছু common service name → port নম্বর
PORT_NAME_MAP = {
    "http": 80,
    "https": 443,
    "domain-s": 53,   # argus style DNS
    "dns": 53,
    "ntp": 123,
    "ssh": 22,
}


def convert_port(val) -> int:
    """ CSV থেকে আসা port value কে int-এ convert করে: - '443'    -> 443 - 'https'  -> 443
    - '0x0016' -> 22 """
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if not s:
        return 0

    lower = s.lower()
    # service name
    if lower in PORT_NAME_MAP:
        return PORT_NAME_MAP[lower]

    # hex format
    if lower.startswith("0x"):
        try:
            return int(lower, 16)
        except ValueError:
            return 0

    # normal int
    try:
        return int(lower)
    except ValueError:
        return 0


def canon_flow_key(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    proto_num: int,
) -> Tuple[str, str, int, int, int]:
    """ Symmetric 5-tuple: (ip1, ip2, p1, p2, proto) যে direction-ই হোক, একই flow একই key পাবে।"""
    src_ip = str(src_ip)
    dst_ip = str(dst_ip)
    sport = int(sport)
    dport = int(dport)
    proto_num = int(proto_num)

    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a <= b:
        ip1, p1 = a
        ip2, p2 = b
    else:
        ip1, p1 = b
        ip2, p2 = a
    return ip1, ip2, p1, p2, proto_num

# CSV → flow_dict (labels)

def build_nidd_flow_dict_from_csvs(
    folder: str,
    max_rows_per_file: Optional[int] = None,
    verbose: bool = False,
) -> Dict[Tuple[str, str, int, int, int], Dict[str, Any]]:
    """ 5G-NIDD CSV ফাইলগুলো থেকে flow_dict বানায়। key: (ip1, ip2, p1, p2, proto_num)  # symmetric
    val: { "label": "Normal"/"Attack"/"Unlabeled",
          "is_attack": bool,
          "attack_type": str,
          "src_file": "Goldeneye1.csv", ...}
    এখানে আমরা মূলত BS1_each_attack_csv / BS2_each_attack_csv use করব।
    BTS_1 / BTS_2 / Combined / Encoded / *argus* নামের CSV গুলো ignore করা হবে। """

    flow_dict: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue

            fname_low = f.lower()

            # aggregated / encoded / argus CSV গুলো বাদ
            if (
                "bts_1.csv" in fname_low
                or "bts_2.csv" in fname_low
                or "combined.csv" in fname_low
                or "encoded.csv" in fname_low
                or "argus" in fname_low
            ):
                # ইচ্ছে করে কিছু print করছি না, যাতে output clean থাকে
                continue

            csv_path = os.path.join(root, f)

            try:
                df = pd.read_csv(
                    csv_path,
                    low_memory=False,
                    nrows=max_rows_per_file if max_rows_per_file and max_rows_per_file > 0 else None,
                )
            except Exception as e:
                if verbose:
                    print(f"[NIDD] Error reading CSV {csv_path}: {e}")
                continue

            # সব column lower-case
            df.columns = df.columns.str.strip().str.lower()

            # required columns (BS1_each_attack_csv / BS2_each_attack_csv structure অনুযায়ী)
            required_base = ["srcaddr", "dstaddr", "sport", "dport", "proto"]

            # label column খোঁজা
            label_col = None
            if "label" in df.columns:
                label_col = "label"
            elif "attack type" in df.columns:
                label_col = "attack type"
            elif "attack tool" in df.columns:
                label_col = "attack tool"

            required = required_base + ([label_col] if label_col else [])
            missing = [c for c in required if c not in df.columns]

            if missing:
                if verbose:
                    print(f"[NIDD] Skipping {csv_path}: missing {missing}")
                continue

            for _, row in df.iterrows():
                src_ip = str(row.get("srcaddr", "")).strip()
                dst_ip = str(row.get("dstaddr", "")).strip()
                if not src_ip or not dst_ip:
                    continue

                sport = convert_port(row.get("sport", 0))
                dport = convert_port(row.get("dport", 0))

                proto_str = str(row.get("proto", "")).strip().lower()
                if proto_str.isdigit():
                    proto_num = int(proto_str)
                else:
                    proto_num = PROTO_MAP.get(proto_str, 0)

                raw_label = str(row.get(label_col, "")).strip().lower()

                if "benign" in raw_label or "normal" in raw_label:
                    label = "Normal"
                    is_attack = False
                elif raw_label and raw_label != "nan":
                    label = "Attack"
                    is_attack = True
                else:
                    label = "Unlabeled"
                    is_attack = False

                attack_type = raw_label if is_attack else "Benign"

                key = canon_flow_key(src_ip, dst_ip, sport, dport, proto_num)
                flow_dict[key] = {
                    "label": label,
                    "is_attack": is_attack,
                    "attack_type": attack_type,
                    "src_file": os.path.basename(csv_path),
                }

    print(f"Flow dictionary built from CSVs: {len(flow_dict)} labeled flows")
    print(f"Found {len(flow_dict)} CSV-labeled flows")
    return flow_dict

# pcap/pcapng → packet iterator (tshark)

def iter_nidd_packets(
    pcap_path: str,
    flow_dict: Optional[Dict[Tuple[str, str, int, int, int], Dict[str, Any]]] = None,
    max_packets: Optional[int] = None,
):
    """ এক একটা packet থেকে তথ্য বের করে dict আকারে yield করে।
আপাতত outer ip.src / ip.dst + TCP/UDP ports + ip.proto ব্যবহার করছি। """

    cmd: List[str] = [
        "tshark", "-r", pcap_path, "-T", "fields",
        "-n",
        # GTP dissector enforce করে রাখি, কিন্তু এখানে inner IP ব্যবহার করছিনা
        "-d", "udp.port==2152,gtp",
        # outer IP + transport ports + proto + protocol column + ts + frame len
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "udp.srcport",
        "-e", "sctp.srcport",
        "-e", "tcp.dstport",
        "-e", "udp.dstport",
        "-e", "sctp.dstport",
        "-e", "ip.proto",
        "-e", "_ws.col.Protocol",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("tshark not found. Install Wireshark/tshark first.")
        return

    matched = 0
    total = 0
    filename = os.path.basename(pcap_path)

    while True:
        line = proc.stdout.readline()
        if not line:
            break

        line = line.rstrip("\n")
        if not line:
            continue

        total += 1
        if max_packets is not None and total > max_packets:
            break

        parts = line.split("\t")
        if len(parts) < 12:
            continue

        (
            ip_src,
            ip_dst,
            tcp_sport,
            udp_sport,
            sctp_sport,
            tcp_dport,
            udp_dport,
            sctp_dport,
            ip_proto_str,
            proto_col,
            ts_str,
            frame_len_str,
        ) = parts[:12]

        src_ip = ip_src.strip()
        dst_ip = ip_dst.strip()

        # transport + ports
        sport_str = ""
        dport_str = ""
        transport = "OTHER"

        if tcp_sport or tcp_dport:
            transport = "TCP"
            sport_str = tcp_sport or "0"
            dport_str = tcp_dport or "0"
        elif udp_sport or udp_dport:
            transport = "UDP"
            sport_str = udp_sport or "0"
            dport_str = udp_dport or "0"
        elif sctp_sport or sctp_dport:
            transport = "SCTP"
            sport_str = sctp_sport or "0"
            dport_str = sctp_dport or "0"
        else:
            lp = proto_col.lower()
            if "icmp" in lp:
                transport = "ICMP"
            elif "sctp" in lp:
                transport = "SCTP"

        try:
            proto_num = int(ip_proto_str) if ip_proto_str.strip() else 0
        except ValueError:
            proto_num = 0

        try:
            ts = float(ts_str)
        except ValueError:
            ts = 0.0

        try:
            frame_len = int(frame_len_str)
        except ValueError:
            frame_len = 0

        src_port_int = convert_port(sport_str)
        dst_port_int = convert_port(dport_str)

        label = "Unlabeled"
        is_attack = False
        src_file = None
        attack_type = None

        if flow_dict is not None and src_ip and dst_ip:
            key = canon_flow_key(src_ip, dst_ip, src_port_int, dst_port_int, proto_num)
            meta = flow_dict.get(key)
            if meta:
                matched += 1
                label = meta["label"]
                is_attack = bool(meta["is_attack"])
                src_file = meta.get("src_file")
                attack_type = meta.get("attack_type")

        yield {
            "file": filename,
            "ts": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port_int,
            "dst_port": dst_port_int,
            "ip_proto": proto_num,
            "transport": transport,         # TCP/UDP/ICMP/...
            "app_proto": proto_col,         # tshark protocol column (e.g., GTP/UDP, HTTP)
            "pkt_len": frame_len,
            "label": label,
            "is_attack": int(is_attack),
            "src_file": src_file,
            "attack_type": attack_type,
        }

    ret = proc.wait()
    stderr = proc.stderr.read().strip()
    sys.stderr.write(
        f"iter_nidd_packets: pcap={filename}, "
        f"total={total}, matched={matched}, exit={ret}\n"
    )
    if ret != 0 and stderr:
        sys.stderr.write(
            f"tshark stderr for {filename}:\n{stderr}\n"
        )

# High-level loader: CSV + PCAP → DataFrame

def load_nidd_packets(
    root: str,
    max_rows_per_csv: Optional[int] = None,
    max_packets_per_pcap: Optional[int] = None,
) -> pd.DataFrame:
    """ 5G-NIDD root থেকে:
      1) সব CSV পড়ে flow_dict বানায়
      2) সব pcap/pcapng থেকে packet পড়ে label assign করে
      3) সব মিলিয়ে pandas DataFrame return দেয়  """
    root = os.path.abspath(root)
    print(f"NIDD root: {root}")

    # 1) CSV → flow_dict
    flow_dict = build_nidd_flow_dict_from_csvs(
        root,
        max_rows_per_file=max_rows_per_csv,
        verbose=False,   # skip message গুলো output-এ দেখাবো না
    )

    # 2) pcap list
    pcap_files: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            lf = f.lower()
            if lf.endswith(".pcap") or lf.endswith(".pcapng"):
                pcap_files.append(os.path.join(r, f))

    print(f"Found {len(pcap_files)} pcap/pcapng files")

    rows: List[Dict[str, Any]] = []
    for p in pcap_files:
        print(f"Reading pcap: {p}")
        for pkt in iter_nidd_packets(
            p,
            flow_dict=flow_dict,
            max_packets=max_packets_per_pcap,
        ):
            rows.append(pkt)

    if not rows:
        print("No packets loaded. Check paths & tshark.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} packets into DataFrame.")
    return df

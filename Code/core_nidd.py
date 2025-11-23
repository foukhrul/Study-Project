import os
import subprocess
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NIDD_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "5G_NIDD"))

# Protocol / Port helpers
PROTO_MAP = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17,
    "ipv6-icmp": 58,
    "sctp": 132,
    "other": 0,
}

PORT_NAME_MAP = {
    "http": 80,
    "https": 443,
    "domain-s": 53,   # argus style DNS
    "dns": 53,
    "ntp": 123,
    "ssh": 22,
}


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def convert_port(val) -> int:
    """ port value into int
      - '443'    -> 443
      - 'https'  -> 443
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
    """ Symmetric 5-tuple: (ip1, ip2, p1, p2, proto) same key same flow for any direction """
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

def build_nidd_flow_dict_from_csvs(folder: str) \
        -> Dict[Tuple[str, str, int, int, int], Dict[str, Any]]:

    """ flow_dict from csv # symmetric val:{
          "label": "Normal"/"Attack"/"Unlabeled",
          "is_attack": bool,
          "attack_type": str,
          "src_file": "Goldeneye1.csv", ...} """

    flow_dict: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue

            fname_low = f.lower()

            # aggregated / encoded / argus CSV excluded
            if (
                "bts_1.csv" in fname_low
                or "bts_2.csv" in fname_low
                or "combined.csv" in fname_low
                or "encoded.csv" in fname_low
                or "argus" in fname_low
            ):
                # no print to keep output clean
                continue

            csv_path = os.path.join(root, f)

            try:
                df = pd.read_csv(
                    csv_path,
                    low_memory=False,
                )
            except Exception as e:
                continue

            # all column lower case
            df.columns = df.columns.str.strip().str.lower()

            required = ["srcaddr", "dstaddr", "sport", "dport", "proto"]

            # to find label column
            label_col = None
            if "label" in df.columns:
                label_col = "label"
            elif "attack type" in df.columns:
                label_col = "attack type"
            elif "attack tool" in df.columns:
                label_col = "attack tool"

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

    return flow_dict

# pcap/pcapng → packet iterator (tshark)

def iter_nidd_packets(
    pcap_path: str,
    flow_dict: Optional[Dict[Tuple[str, str, int, int, int], Dict[str, Any]]] = None,
):
    """ yield as dict from each packet
   use outer ip.src / ip.dst + TCP/UDP/SCTP ports + ip.proto,
  and find tcp.len / udp.length / data.len থেকে app_len/header_len """

    cmd: List[str] = [
        "tshark", "-r", pcap_path, "-T", "fields",
        "-n",
        # GTP dissector enforce করে রাখি, কিন্তু এখানে inner IP ব্যবহার করছিনা
        "-d", "udp.port==2152,gtp",
        # outer IP + transport ports + proto + protocol column + frame len + ts + L4 payload
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
        "-e", "frame.len",
        "-e", "frame.time_epoch",
        "-e", "tcp.len",
        "-e", "udp.length",
        "-e", "data.len",
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
    #read line from tshark nd stops when output ends
    while True:
        line = proc.stdout.readline()
        if not line:
            break

        line = line.rstrip("\n")
        if not line:
            continue
        #count valid line
        total += 1

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
            frame_len_str,
            ts_str,
        ) = parts[:12]

        tcp_len = parts[12].strip() if len(parts) > 12 else ""
        udp_len = parts[13].strip() if len(parts) > 13 else ""
        data_len = parts[14].strip() if len(parts) > 14 else ""

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

        frame_len = _safe_int(frame_len_str, default=0)

        src_port_int = convert_port(sport_str)
        dst_port_int = convert_port(dport_str)

        # ---- approx application payload length (L4 payload) ----
        l4_payload = None
        for candidate in (tcp_len, udp_len):
            if candidate:
                l4_payload = _safe_int(candidate, default=0)
                break
        if l4_payload is None and data_len:
            l4_payload = _safe_int(data_len, default=0)
        if l4_payload is None:
            l4_payload = 0

        app_len = max(l4_payload, 0)
        header_len = max(frame_len - app_len, 0) if frame_len > 0 else 0
        has_app_data = app_len > 0

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

        # host_pair (unordered): min(ip1, ip2) <-> max(ip1, ip2)
        if src_ip and dst_ip:
            ip1, ip2 = sorted([src_ip, dst_ip])
            host_pair = f"{ip1} <-> {ip2}"
        else:
            host_pair = None

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
            "header_len": header_len,
            "app_len": app_len,
            "has_app_data": has_app_data,
            "host_pair": host_pair,
            "label": label,
            "is_attack": int(is_attack),
            "src_file": src_file,
            "attack_type": attack_type,
        }

    ret = proc.wait()
    #collect any error that tshark print
    stderr = proc.stderr.read().strip()

# High-level loader: CSV + PCAP → DataFrame

def load_nidd_packets(root: str) -> pd.DataFrame:
    """ read csv make flow_dict, read pcap/pcapng and assign label then return pd df """
    root = os.path.abspath(root)
    print(f"NIDD root: {root}")

    # 1) CSV → flow_dict
    flow_dict = build_nidd_flow_dict_from_csvs(root)

    # 2) pcap list
    pcap_files: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            lf = f.lower()
            if lf.endswith(".pcap") or lf.endswith(".pcapng"):
                pcap_files.append(os.path.join(r, f))

    rows: List[Dict[str, Any]] = []
    for p in pcap_files:
        for pkt in iter_nidd_packets( p, flow_dict=flow_dict):
            rows.append(pkt)

    if not rows:
        print("No packets loaded. Check paths & tshark.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} packets into DataFrame.")
    return df

import os
import ipaddress
import subprocess

import pandas as pd

DEFAULT_PFCP_ROOT = "/Users/eshtisabbiroutlook.com/PycharmProjects/5G_Project/5G_PFCP"

# helper
def _safe_int(x, default=0):
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

# to normalize csv label
def normalize_label_pfcp(label):

    s = str(label).lower().strip()
    if s == "normal":
        return "Normal"
    if s.startswith("mal") or "attack" in s or "malic" in s or "Malicious" in s:
        return "Attack"
    return "Unlabeled"


def flow_id(src_ip, dst_ip, src_port, dst_port, proto):
    # Symmetric 5-tuple

    src_port_str, dst_port_str = str(src_port), str(dst_port)
    try:
        a, b = ipaddress.ip_address(src_ip), ipaddress.ip_address(dst_ip)
        if a < b:
            ips, ports = (src_ip, dst_ip), (src_port_str, dst_port_str)
        elif b < a:
            ips, ports = (dst_ip, src_ip), (dst_port_str, src_port_str)
        else:
            p = sorted([int(src_port_str), int(dst_port_str)])
            ips, ports = (src_ip, dst_ip), (str(p[0]), str(p[1]))
        return f"{ips[0]}-{ips[1]}-{ports[0]}-{ports[1]}-{proto}"
    except ValueError:
        ip_list = sorted([src_ip, dst_ip])
        port_list = sorted([src_port_str, dst_port_str])
        return f"INVALID-{ip_list[0]}-{ip_list[1]}-{port_list[0]}-{port_list[1]}-{proto}"


PROTO_NAME_MAP = {
    "6": "TCP",
    "17": "UDP",
    "132": "SCTP",
    "1": "ICMP",
    "58": "ICMPv6",
    "0": "Other",
}

# LABEL DICT (from CSVs)
def build_pfcp_flow_dict(root_folder):
    flow_dict = {}

    for root, _, files in os.walk(root_folder):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            csv_path = os.path.join(root, f)
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df.columns = df.columns.str.strip()
                cols_lower = [c.lower() for c in df.columns]

                # column detection (very flexible)
                def find_col(possible_keywords):
                    for c_orig, c_low in zip(df.columns, cols_lower):
                        for kw in possible_keywords:
                            if kw in c_low:
                                return c_orig
                    return None

                src_ip_col = find_col(["source ip", "src ip", "srcaddr"])
                dst_ip_col = find_col(["destination ip", "dst ip", "dstaddr"])
                src_port_col = find_col(["source port", "src port", "sport"])
                dst_port_col = find_col(["destination port", "dst port", "dport"])
                proto_col = find_col(["protocol", "proto"])
                label_col = find_col(["label", "attack type", "attack tool"])

                missing = []
                for name, col in [
                    ("source IP", src_ip_col),
                    ("destination IP", dst_ip_col),
                    ("source port", src_port_col),
                    ("destination port", dst_port_col),
                    ("protocol", proto_col),
                    ("label", label_col),
                ]:
                    if col is None:
                        missing.append(name)

                if missing:
                    print(f"Skipping CSV {csv_path}: Missing {', '.join(missing)}")
                    continue

                for _, row in df.iterrows():
                    src_ip = str(row[src_ip_col]).strip()
                    dst_ip = str(row[dst_ip_col]).strip()
                    src_port = str(row[src_port_col]).strip()
                    dst_port = str(row[dst_port_col]).strip()
                    proto_val = str(row[proto_col]).strip()

                    if proto_val.isdigit():
                        proto = proto_val
                    else:
                        proto = "0"

                    label = normalize_label_pfcp(row[label_col])

                    fid = flow_id(src_ip, dst_ip, src_port, dst_port, proto)
                    flow_dict[fid] = label

            except Exception as e:
                print(f"Error in CSV {csv_path}: {e}")
                continue

    print(f"Flow dictionary size: {len(flow_dict)} flows")
    return flow_dict

# PACKET ITERATOR (PCAP -> rows)
def iter_pfcp_packets(pcap_path, flow_dict):

    cmd = [
        "tshark", "-r", pcap_path, "-T", "fields",
        "-e", "ip.src", "-e", "ip.dst",
        "-e", "tcp.srcport", "-e", "udp.srcport", "-e", "sctp.srcport",
        "-e", "tcp.dstport", "-e", "udp.dstport", "-e", "sctp.dstport",
        "-e", "ip.proto", "-e", "_ws.col.Protocol",
        "-e", "frame.len", "-e", "frame.time_epoch",
        "-e", "tcp.len", "-e", "udp.length",
        "-e", "data.len",
        "-n",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=900
        )
    except subprocess.CalledProcessError as e:
        print(f"tshark failed on {pcap_path} (exit {e.returncode})")
        if e.stderr:
            print("  --- tshark stderr ---")
            print(e.stderr.strip())
        return
    except subprocess.TimeoutExpired:
        print(f"Timeout reading {pcap_path}")
        return
    except Exception as e:
        print(f"Failed to process {pcap_path}: {e}")
        return

    filename = os.path.basename(pcap_path)

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")

        # ---- basic fields ----
        src_ip = parts[0].strip() if len(parts) > 0 and parts[0].strip() else "0.0.0.0"
        dst_ip = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "0.0.0.0"

        src_port = next(
            (p.strip() for p in parts[2:5] if p.strip()), "0"
        )
        dst_port = next(
            (p.strip() for p in parts[5:8] if p.strip()), "0"
        )

        ip_proto = parts[8].strip() if len(parts) > 8 and parts[8].strip() else "0"
        app_proto = parts[9].strip() if len(parts) > 9 and parts[9].strip() else "UNKNOWN"

        transport_proto = PROTO_NAME_MAP.get(ip_proto, "Other")

        pkt_len = _safe_int(parts[10] if len(parts) > 10 else 0, default=0)
        ts = _safe_float(parts[11] if len(parts) > 11 else 0.0, default=0.0)

        tcp_len = parts[12].strip() if len(parts) > 12 else ""
        udp_len = parts[13].strip() if len(parts) > 13 else ""
        data_len = parts[14].strip() if len(parts) > 14 else ""

        # ---- approximate application payload length ----
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
        header_len = max(pkt_len - app_len, 0) if pkt_len > 0 else 0

        fid = flow_id(src_ip, dst_ip, src_port, dst_port, ip_proto)
        label = flow_dict.get(fid, "Unlabeled")
        is_attack = (label == "Attack")

        host_pair = " <-> ".join(sorted([src_ip, dst_ip]))
        has_app_data = app_len > 0

        yield {
            "file": filename,
            "ts": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "ip_proto": ip_proto,          # numeric code as string
            "transport_proto": transport_proto,
            "app_proto": app_proto,
            "pkt_len": pkt_len,
            "header_len": header_len,
            "app_len": app_len,
            "host_pair": host_pair,
            "label": label,
            "is_attack": is_attack,
            "has_app_data": has_app_data,
        }

#  LOAD WHOLE DATASET INTO DATAFRAME
def load_pfcp_packets(root_folder=DEFAULT_PFCP_ROOT):
    flow_dict = build_pfcp_flow_dict(root_folder)

    rows = []
    for r, _, files in os.walk(root_folder):
        for f in files:
            if not f.lower().endswith((".pcap", ".pcapng")):
                continue
            pcap_path = os.path.join(r, f)
            #print(f"[PFCP] Reading pcap: {pcap_path}")
            for row in iter_pfcp_packets(pcap_path, flow_dict):
                rows.append(row)

    if not rows:
        print("No packets loaded. Check paths & tshark installation / pcap format.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["ts"] = df["ts"].astype(float)
    df["pkt_len"] = df["pkt_len"].astype(int)
    df["header_len"] = df["header_len"].astype(int)
    df["app_len"] = df["app_len"].astype(int)
    df["is_attack"] = df["is_attack"].astype(bool)
    df["has_app_data"] = df["has_app_data"].astype(bool)

    print(f"Loaded {len(df):,} packets into DataFrame.")
    return df


import os
import matplotlib.pyplot as plt

from core_pfcp import load_pfcp_packets

# Local default PFCP root (only for this script)
DEFAULT_PFCP_ROOT = "/Users/eshtisabbiroutlook.com/PycharmProjects/5G_Project/5G_PFCP"


def compute_ccdf(values):
    """ Manual CCDF computation. Input:  iterable of numeric values
    Output: (xs, ys) where ys[i] = P(X >= xs[i]) """
    vals = sorted(values)
    n = len(vals)
    xs, ys = [], []

    if n == 0:
        return xs, ys

    prev = None
    for i, v in enumerate(vals):
        if v != prev:
            ccdf = (n - i) / n
            xs.append(v)
            ys.append(ccdf)
            prev = v

    return xs, ys


def safe_name(proto):
    """Make protocol name safe for filenames."""
    return "".join(ch if ch.isalnum() else "_" for ch in proto) or "UNKNOWN"


def plot_ccdf_for_metric(df, metric_col, metric_label, out_dir, with_attack_flag):
    """ Attack / non-attack subset """
    if with_attack_flag:
        subset = df[df["is_attack"] == True].copy()
        tag = "with_attack"
        title_tag = "with attacks"
    else:
        subset = df[df["is_attack"] == False].copy()
        tag = "without_attack"
        title_tag = "without attacks"

    # only positive length + NaN exclude
    series = subset[[metric_col, "app_proto"]].dropna()
    series = series[series[metric_col] > 0]

    if series.empty:
        print(f"[PFCP-1d] No data for {metric_label} ({title_tag})")
        return

    protocols = sorted(series["app_proto"].unique())

    print(
        f"[PFCP-1d] Plotting {metric_label} CCDFs {title_tag} "
        f"for {len(protocols)} app protocols..."
    )
#oi protocol er shob row theke metric_col er value gulo nibe list akare.
    for proto in protocols:
        vals = series.loc[series["app_proto"] == proto, metric_col].tolist()
        if len(vals) < 2:
            # if less sample ccdf is meaningless
            continue

        xs, ys = compute_ccdf(vals)
        if not xs:
            continue

        plt.figure()
        plt.step(xs, ys, where="post")

        plt.xlabel(f"{metric_label} (bytes)")
        plt.ylabel("P(X ≥ x)")
        plt.title(
            f"PFCP {metric_label} CCDF ({title_tag})\n"
            f"Application protocol: {proto}"
        )
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xscale("log")

        fname = f"pfcp_ccdf_{metric_col}_{tag}_{safe_name(proto)}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"  -> saved: {out_path}")


def main():
    root = DEFAULT_PFCP_ROOT
    out_dir = "pfcp_task1d_plots"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing 5G-PFCP dataset from: {root}")
    df = load_pfcp_packets(root)

    if df.empty:
        print("DataFrame is empty, nothing to analyze.")
        return

    required_cols = {"header_len", "app_len", "app_proto", "is_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return

    plot_ccdf_for_metric(df, "header_len", "Header length", out_dir, True)
    plot_ccdf_for_metric(df, "header_len", "Header length", out_dir, False)

    plot_ccdf_for_metric(df, "app_len", "Application payload length", out_dir, True)
    plot_ccdf_for_metric(df, "app_len", "Application payload length", out_dir, False)

    print("Task 1d (PFCP) CCDF plots generated.")
    print(f"Plots saved in: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
""" This toolbox can start:
- Practical Sheet 1, PFCP Task 1 analysis
- Practical Sheet 1, PFCP GAN (Task 3)
- Task Sheet 2, PFCP Task 1 (stats, MinMax, entropy)
- Task Sheet 2, PFCP Task 3 GAN (my new GAN code)
Usage:
  python toolbox.py -h
  python toolbox.py -s ps1_task1_pfcp
  python toolbox.py -s ps1_gan_pfcp
  python toolbox.py -s ts2_pfcp_stats
  python toolbox.py -s ts2_pfcp_gan """

import os
import sys
import subprocess
import argparse

# project root = .../5G_Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(rel_path: str) -> None:
    """
    Run another Python script relative to PROJECT_ROOT.
    """
    full_path = os.path.join(PROJECT_ROOT, rel_path)

    if not os.path.exists(full_path):
        print(f"[ERROR] Script not found: {full_path}")
        return

    cmd = [sys.executable, full_path]
    print(f"\n[TOOLBOX] Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5G PFCP Toolbox (-s <mode> to select operation)."
    )

    parser.add_argument(
        "-s",
        dest="mode",
        metavar="<mode>",
        help=(
            "Select operation mode:\n"
            "  ps1_task1_pfcp  - Practical Sheet 1: PFCP Task 1 analysis\n"
            "  ps1_gan_pfcp    - Practical Sheet 1: PFCP GAN (Task 3)\n"
            "  ts2_pfcp_stats  - Task Sheet 2: PFCP Task 1 stats (MinMax, entropy)\n"
            "  ts2_pfcp_gan    - Task Sheet 2: PFCP Task 3 PFCP GAN\n"
        ),
    )

    args = parser.parse_args()

    if not args.mode:
        print("No mode selected. Use -h to see available modes.")
        return

    mode = args.mode

    # map modes to scripts (relative to PROJECT_ROOT)
    if mode == "ps1_task1_pfcp":
        run_script(os.path.join("Code", "task1a_pfcp.py"))

    elif mode == "ps1_gan_pfcp":
        run_script(os.path.join("Code", "task3_gan.py"))

    elif mode == "ts2_pfcp_stats":
        run_script(os.path.join("TaskSheet2", "task1_synthetic_stats_pfcp.py"))

    elif mode == "ts2_pfcp_gan":
        run_script(os.path.join("TaskSheet2", "task3_gan_pfcp.py"))

    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("Use -h to see available modes.")


if __name__ == "__main__":
    main()

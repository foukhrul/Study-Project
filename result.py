import os
import subprocess

# --------- CONFIG ---------
TASKS = [
    ("task1a_pfcp.py", "task1a_pfcp.txt"),
    #("task1a_nidd.py", "task1a_nidd.txt"),
    #("task1b_pfcp.py", "task1b_pfcp.txt"),
    #("task1b_nidd.py", "task1b_nidd.txt"),
    #("task1c_pfcp.py", "task1c_pfcp.txt"),
    #("task1c_nidd.py", "task1c_nidd.txt"),
    #("task1d_pfcp_ccdf.py", "task1d_pfcp_summary.txt"),
    #("task1d_nidd_ccdf.py", "task1d_nidd_summary.txt"),
]

RESULT_FOLDER = "results"
PYTHON = "python3"
# --------------------------

def run_and_save(script_name, output_file):
    print(f"Running: {script_name}")
    print(f"Saving output to: {output_file}")

    with open(os.path.join(RESULT_FOLDER, output_file), "w") as f:
        subprocess.run([PYTHON, script_name], stdout=f, stderr=subprocess.STDOUT)


def main():
    # create results folder
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"Results will be saved in: {RESULT_FOLDER}/\n")

    # run all tasks one by one
    for script, outfile in TASKS:
        if not os.path.exists(script):
            print(f"⚠️ Warning: {script} not found — skipping.")
            continue
        run_and_save(script, outfile)

if __name__ == "__main__":
    main()

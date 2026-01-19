"""Extract a summary from an nsys report file."""

import json
import sqlite3
from typing import Literal

fpath_in_baseline = "results/nsys/baseline.sqlite"
fpath_in_mod = "results/nsys/mod.sqlite"
fpath_out = "results/summary.json"


def get_nsys_summary(fpath_in) -> dict:
    """Extract summary from nsys report file."""
    conn = sqlite3.connect(fpath_in)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ANALYSIS_DETAILS")
    row = cursor.fetchone()
    columns = ["nsys_" + description[0] for description in cursor.description]
    summary = dict(zip(columns, row))
    summary["nsys_duration_s"] = (
        summary["nsys_duration"] / 1e9
    )  # Convert from ns to s
    # TODO: Extract the top 3 most expensive kernels by total time
    conn.close()
    return summary


def get_run_summary(case_name: Literal["baseline", "mod"]) -> dict:
    """Extract non-nsys run summary from log file."""
    fpath = f".calkit/slurm/logs/amip-{case_name}.out"
    with open(fpath, "r") as f:
        lines = f.readlines()
    found_step_40 = False
    res = {}
    for line in lines:
        if "n_steps_completed = 40" in line:
            found_step_40 = True
        if found_step_40:
            if "estimated_sypd =" in line:
                parts = line.strip().split()
                res["sypd"] = float(parts[-1].replace('"', ""))
            elif "n_steps_completed =" in line:
                parts = line.strip().split()
                res["n_steps_completed"] = int(parts[-1])
            elif "wall_time_total =" in line:
                res["wall_time_total"] = (
                    line.split("=")[-1].strip().replace('"', "")
                )
    return res


baseline = get_nsys_summary(fpath_in_baseline) | get_run_summary("baseline")
mod = get_nsys_summary(fpath_in_mod) | get_run_summary("mod")

results = {
    "nsys_speedup_pct": (1 - mod["nsys_duration"] / baseline["nsys_duration"])
    * 100,
    "speedup_pct": (1 - baseline["sypd"] / mod["sypd"]) * 100,
    "baseline": baseline,
    "mod": mod,
}

with open(fpath_out, "w") as f:
    json.dump(results, f, indent=2)

speedup_icon = "🚀"
slowdown_icon = "🐢"

if results["nsys_speedup_pct"] >= 0:
    nsys_icon = speedup_icon
else:
    nsys_icon = slowdown_icon
print(f"{nsys_icon} NSYS speedup: {results['nsys_speedup_pct']:.1f}%")
if results["speedup_pct"] >= 0:
    run_icon = speedup_icon
else:
    run_icon = slowdown_icon
print(f"{run_icon} Normal run speedup: {results['speedup_pct']:.1f}%")

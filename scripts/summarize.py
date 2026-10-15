"""Extract a summary from an nsys report file."""

import json
import re
import sqlite3
from typing import Literal

# The coupler defines SYPD as (t_end - t_start) / walltime / 365.25.
DAYS_PER_YEAR = 365.25

fpath_in_baseline = "results/nsys/baseline.sqlite"
fpath_in_mod = "results/nsys/mod.sqlite"
fpath_step_cmp = "results/step-comparison.json"
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
    """Extract non-nsys run summary from log file.

    `SYPD:` covers the whole timed window and so carries the run's deferred
    compilation. The `Progress` series is kept as data but is not used to subtract
    compilation: with `walltime_debug` its reports start at step 2, so the coupler's
    own "discard the first interval" heuristic discards almost nothing, and with only
    four diagnostics writes in the window the periodic work swamps any rate fitted
    across reports. results/step-comparison.json separates compilation from steady
    state per step instead.
    """
    fpath = f".calkit/scheduler/logs/amip-{case_name}.out"
    with open(fpath, "r") as f:
        lines = f.readlines()
    res = {}
    progress = []
    step = None
    for line in lines:
        if "Info: Progress" in line:
            step = None
        elif (m := re.search(r"\bstep = (\d+)", line)) is not None:
            step = int(m.group(1))
        elif (m := re.search(r"sypd ≈ ([\d.eE+-]+)", line)) is not None:
            if step is not None:
                progress.append((step, float(m.group(1))))
                step = None
        elif (m := re.search(r"Simulation took ([\d.eE+-]+) seconds", line)) is not None:
            res["wall_time_total_s"] = float(m.group(1))
        elif (m := re.search(r"Info: SYPD: ([\d.eE+-]+)", line)) is not None:
            res["sypd"] = float(m.group(1))
        elif (
            m := re.search(r"Walltime per coupling step: ([\d.eE+-]+)", line)
        ) is not None:
            res["walltime_per_step_s"] = float(m.group(1))

    res["progress_steps"] = [s for s, _ in progress]
    res["progress_sypd"] = [v for _, v in progress]

    # The coupler steps a little before starting its timer, so the timed window is
    # shorter than the run; recover its length rather than assuming the config's.
    n_timed = round(res["wall_time_total_s"] / res["walltime_per_step_s"])
    res["n_steps_timed"] = n_timed
    res["dt_cpl_s"] = (
        res["sypd"] * DAYS_PER_YEAR * res["wall_time_total_s"] / n_timed
    )
    if progress:
        res["n_steps_total"] = progress[-1][0]
    return res


with open(fpath_step_cmp) as f:
    step_cmp = json.load(f)

baseline = get_nsys_summary(fpath_in_baseline) | get_run_summary("baseline")
mod = get_nsys_summary(fpath_in_mod) | get_run_summary("mod")

results = {
    "nsys_speedup_pct": (1 - mod["nsys_duration"] / baseline["nsys_duration"])
    * 100,
    "speedup_pct": (1 - baseline["sypd"] / mod["sypd"]) * 100,
    "steady_state_pct": step_cmp["steady_state"]["saved_pct"],
    "steady_state_pct_ex_diagnostics": step_cmp["steady_state"][
        "saved_pct_ex_diagnostics"
    ],
    "compile_penalty_s": step_cmp["compile_penalty"]["total_s"],
    "compile_penalty_steps": step_cmp["compile_penalty"]["steps"],
    "typical_step_saved_ms": step_cmp["typical_step"]["saved_median_ms"],
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
print(
    f"   steady state {results['steady_state_pct']:.2f}%"
    f"  ({results['steady_state_pct_ex_diagnostics']:.2f}% excluding diagnostics steps)"
)
print(
    f"   typical ordinary step saves {results['typical_step_saved_ms']:.1f} ms;"
    f" the mod arm pays {results['compile_penalty_s']:.1f} s more compilation"
    f" over {results['compile_penalty_steps']} steps"
)

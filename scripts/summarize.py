"""Extract a summary from an nsys report file."""

import json
import re
import sqlite3
from typing import Literal

# The coupler defines SYPD as (t_end - t_start) / walltime / 365.25.
DAYS_PER_YEAR = 365.25

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
    """Extract non-nsys run summary from log file.

    Two SYPD figures come out of the log. `SYPD:` is the whole timed window, so it
    carries the run's deferred compilation; the `Progress` reports discard the first
    reporting interval's wall clock as compilation, so their final value is the
    coupler's own compile-discounted rate. The difference is an estimate of what
    compilation cost. It is a floor: the reports scale the first retained interval as
    if the discarded steps ran at that rate, which is why the series drifts down, and
    compilation continues well past the first interval. results/step-comparison.json
    measures the steady state directly instead.
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
    simulated_timed_s = res["sypd"] * DAYS_PER_YEAR * res["wall_time_total_s"]
    dt_cpl_s = simulated_timed_s / n_timed
    res["n_steps_timed"] = n_timed
    res["dt_cpl_s"] = dt_cpl_s

    if progress:
        n_total, sypd_excl = progress[-1]
        simulated_total_s = n_total * dt_cpl_s
        wall_excl_s = simulated_total_s / (sypd_excl * DAYS_PER_YEAR)
        res["n_steps_total"] = n_total
        res["sypd_compile_excluded"] = sypd_excl
        res["wall_time_compile_excluded_s"] = wall_excl_s
        res["ms_per_step_compile_excluded"] = 1e3 * wall_excl_s / n_total
        res["compile_s"] = (
            res["wall_time_total_s"] - n_timed * wall_excl_s / n_total
        )
        res["compile_frac_of_wall"] = res["compile_s"] / res["wall_time_total_s"]
    return res


baseline = get_nsys_summary(fpath_in_baseline) | get_run_summary("baseline")
mod = get_nsys_summary(fpath_in_mod) | get_run_summary("mod")

results = {
    "nsys_speedup_pct": (1 - mod["nsys_duration"] / baseline["nsys_duration"])
    * 100,
    "speedup_pct": (1 - baseline["sypd"] / mod["sypd"]) * 100,
    "speedup_pct_compile_excluded": (
        1
        - baseline["sypd_compile_excluded"] / mod["sypd_compile_excluded"]
    )
    * 100,
    "compile_s_delta": mod["compile_s"] - baseline["compile_s"],
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
excl_icon = (
    speedup_icon if results["speedup_pct_compile_excluded"] >= 0 else slowdown_icon
)
print(
    f"{excl_icon} Normal run speedup, compilation excluded: "
    f"{results['speedup_pct_compile_excluded']:.1f}%"
)
for name, arm in (("baseline", baseline), ("mod", mod)):
    print(
        f"   {name}: {arm['wall_time_total_s']:.1f} s over {arm['n_steps_timed']} steps, "
        f"{arm['compile_s']:.1f} s of it compilation "
        f"({100 * arm['compile_frac_of_wall']:.1f}%), "
        f"{arm['ms_per_step_compile_excluded']:.1f} ms/step without it"
    )
print(f"   compilation, mod - baseline: {results['compile_s_delta']:+.1f} s")

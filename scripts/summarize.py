"""Extract a summary from an nsys report file."""

import json
import re
import sqlite3
from typing import Literal

# The coupler defines SYPD as (t_end - t_start) / walltime / 365.25.
DAYS_PER_YEAR = 365.25
# The arms' kernel-activity spans must stay within this factor of each other.
# The honest gap is about 0.75 (the mod arm finishes the same window sooner);
# the truncation this guards against came in at 0.018.
MIN_TRACE_SPAN_RATIO = 0.5

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


def get_trace_coverage(fpath_in) -> dict:
    """What the trace actually holds, which `ANALYSIS_DETAILS` cannot say.

    On 2026-11-11 a baseline profile exited 0 and updated `dvc.lock` while every
    CUPTI table stopped 0.850 s after the capture range opened -- two of the 120
    steps, every per-step kernel at 1/57 of its real launch count, and nothing
    in the log saying so. `duration` still reported the whole 51.6 s session, so
    the session metadata is blind to it; the span of the kernel records is not.
    """
    conn = sqlite3.connect(fpath_in)
    span_ns, launches = conn.execute(
        "SELECT MAX(end) - MIN(start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    conn.close()
    return {
        "kernel_span_s": span_ns / 1e9,
        "kernel_launches": launches,
    }


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

baseline = (
    get_nsys_summary(fpath_in_baseline)
    | get_trace_coverage(fpath_in_baseline)
    | get_run_summary("baseline")
)
mod = (
    get_nsys_summary(fpath_in_mod)
    | get_trace_coverage(fpath_in_mod)
    | get_run_summary("mod")
)

# Both arms profile the same 120 steps, so their traces must cover comparable
# spans -- the honest device difference between them is tens of percent, never a
# factor. Checked before anything is written, because a truncated trace keeps
# every per-kernel average plausible while the totals are short by the
# truncation factor, which reads as a treatment.
spans = (baseline["kernel_span_s"], mod["kernel_span_s"])
if min(spans) < MIN_TRACE_SPAN_RATIO * max(spans):
    raise SystemExit(
        f"trace spans disagree: baseline {spans[0]:.3f} s"
        f" ({baseline['kernel_launches']} launches), mod {spans[1]:.3f} s"
        f" ({mod['kernel_launches']} launches). One arm's collection stopped"
        " early; re-measure it with CALKIT_FORCE=1 calkit dvc repro"
        " --single-item --force <arm>-nsys"
    )

results = {
    "nsys_speedup_pct": (1 - mod["nsys_duration"] / baseline["nsys_duration"])
    * 100,
    "speedup_pct": (1 - baseline["sypd"] / mod["sypd"]) * 100,
    # True means the steady-state figures below come from a frozen snapshot
    # whose own source commit is unrecoverable, so they may predate the
    # end-to-end figures beside them. See results/step-comparison.json.
    "steady_state_from_undated_snapshot": step_cmp.get("provenance", {}).get(
        "sources_carry_no_provenance", True
    ),
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
    f" the mod arm is slower on {results['compile_penalty_steps']} ordinary"
    f" steps totalling {results['compile_penalty_s']:.1f} s -- one-sided, so"
    f" read the count, not the total"
)

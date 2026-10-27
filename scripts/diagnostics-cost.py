"""Price the diagnostics by differencing a diagnostics-off run against the baseline.

Both runs are the baseline arm on the same config and differ only in whether
diagnostics are enabled, so the wall-clock difference is what diagnostics cost in
total -- computation and writing together. The per-step breakdown cannot separate
them: only the write events stand out as expensive steps, while the sampling that
feeds them is spread across ordinary steps and hides in the median.
"""

import json
import re

DAYS_PER_YEAR = 365.25
STEP_CMP = "results/step-comparison.json"
REPEATS = "results/sypd-repeats.json"
OUT = "results/diagnostics-cost.json"


def read_log(case):
    """Pull the coupler's own timing lines out of a scheduler log."""
    res = {}
    with open(f".calkit/scheduler/logs/amip-{case}.out") as f:
        for line in f:
            if (m := re.search(r"Simulation took ([\d.eE+-]+) seconds", line)) is not None:
                res["wall_time_total_s"] = float(m.group(1))
            elif (m := re.search(r"Info: SYPD: ([\d.eE+-]+)", line)) is not None:
                res["sypd"] = float(m.group(1))
            elif (m := re.search(r"Walltime per coupling step: ([\d.eE+-]+)", line)) is not None:
                res["walltime_per_step_s"] = float(m.group(1))
    missing = {"wall_time_total_s", "sypd", "walltime_per_step_s"} - res.keys()
    if missing:
        raise SystemExit(f"amip-{case} log is missing {sorted(missing)}")
    res["n_steps_timed"] = round(res["wall_time_total_s"] / res["walltime_per_step_s"])
    return res


def main():
    on, off = read_log("baseline"), read_log("nodiag")

    # The two runs must have stepped the same window, or the difference is a
    # different simulation rather than a different diagnostics setting.
    if on["n_steps_timed"] != off["n_steps_timed"]:
        raise SystemExit(
            f"runs stepped different windows: {on['n_steps_timed']} with diagnostics "
            f"vs {off['n_steps_timed']} without"
        )

    saved_s = on["wall_time_total_s"] - off["wall_time_total_s"]
    total_pct = 100 * saved_s / on["wall_time_total_s"]

    # What the write steps alone carry, measured independently by the step
    # breakdown. The residual is the sampling spread across ordinary steps.
    step_cmp = json.load(open(STEP_CMP))
    write_pct = step_cmp["classes"]["diagnostics"]["pct_of_baseline_wall"]

    # The residual is a difference across two runs, so it is only a measurement
    # if it is bigger than the spread between two runs of the same code. That
    # spread is now measured, and it is not. Forcing a repeat of amip-baseline
    # on 2026-10-26 moved the residual from 0.32 to 0.00 points without any code
    # changing, which is the demonstration rather than the argument.
    spread = json.load(open(REPEATS))["latest_treatment"]["spread_points"]
    sampling_pct = total_pct - write_pct

    out = {
        "with_diagnostics": on,
        "without_diagnostics": off,
        "n_steps_timed": on["n_steps_timed"],
        "saved_s": saved_s,
        "total_pct_of_wall": total_pct,
        "sypd_with": on["sypd"],
        "sypd_without": off["sypd"],
        "sypd_headroom_pct": 100 * (off["sypd"] / on["sypd"] - 1),
        "write_steps_pct_of_wall": write_pct,
        "sampling_pct_of_wall": sampling_pct,
        "repeat_spread_points": spread,
        "sampling_resolvable": abs(sampling_pct) > spread,
        "note": (
            "total_pct_of_wall is a difference of two whole runs, so it carries each "
            "run's deferred compilation as well as the diagnostics themselves, and the "
            "diagnostics-off run compiles less -- read it as an upper bound. "
            "write_steps_pct_of_wall comes from a separate step-breakdown run on the "
            "same arm and config, so sampling_pct_of_wall is a difference across two "
            "measurements rather than one decomposition. `sampling_resolvable` "
            "compares it against repeat_spread_points, the spread between independent "
            "pairs of the same code in results/sypd-repeats.json: when it is false the "
            "residual is smaller than the run-to-run variation of the runs it is "
            "differenced from, so it bounds the sampling share and does not measure it. "
            "The 120-step nsys window cannot answer this at all: "
            "it is shorter than the 720-step daily write period."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"{out['n_steps_timed']} steps: {on['wall_time_total_s']:.0f}s with diagnostics, "
        f"{off['wall_time_total_s']:.0f}s without -- {saved_s:.0f}s = {total_pct:.1f}% of wall"
    )
    print(
        f"SYPD {on['sypd']:.4f} -> {off['sypd']:.4f}, "
        f"{out['sypd_headroom_pct']:.1f}% headroom if diagnostics were free"
    )
    print(
        f"write steps carry {write_pct:.1f}% of wall, leaving {sampling_pct:.2f} points "
        f"as the sampling residual -- "
        + ("above" if out["sampling_resolvable"] else "below")
        + f" the {spread:.2f}-point spread between repeats of one pair, so it is "
        + ("a measurement" if out["sampling_resolvable"] else "a bound only")
    )


main()

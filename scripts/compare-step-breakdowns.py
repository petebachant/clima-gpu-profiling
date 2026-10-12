"""Pair the two arms' per-step wall clock and say where the fold's saving lands.

Both arms run the same config, so step index i is the same work in both and the
difference is paired rather than a comparison of aggregates.
"""

import json
import statistics as st
import sys
import tomllib

BASELINE = "results/step-breakdown-baseline.toml"
MOD = "results/step-breakdown.toml"
FOLD = "results/param-fold.json"
OUT = "results/step-comparison.json"

# Periodic work, by step index. Radiation is dt_rad=600secs at dt=30secs, and
# the diagnostics write lands on one step in every 120.
RAD_PERIOD, RAD_PHASE = 20, 16
DIAG_PERIOD, DIAG_PHASE = 120, 116


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    b, m = load(BASELINE), load(MOD)
    if b["steps"] != m["steps"]:
        sys.exit(f"arms measured different windows: {b['steps']} vs {m['steps']}")
    bw, mw = b["wall_ms"], m["wall_ms"]
    n = len(bw)

    diag = [i for i in range(n) if i % DIAG_PERIOD == DIAG_PHASE]
    rad = [i for i in range(n) if i % RAD_PERIOD == RAD_PHASE and i not in set(diag)]
    plain = [i for i in range(n) if i % RAD_PERIOD != RAD_PHASE]

    tb, tm = sum(bw), sum(mw)
    saved = tb - tm

    def cls(idx):
        sb, sm = sum(bw[i] for i in idx), sum(mw[i] for i in idx)
        return {
            "steps": len(idx),
            "baseline_mean_ms": sb / len(idx),
            "mod_mean_ms": sm / len(idx),
            "saved_per_step_ms": (sb - sm) / len(idx),
            "saved_s": (sb - sm) / 1000,
            "pct_of_saving": 100 * (sb - sm) / saved,
            "pct_of_baseline_wall": 100 * sb / tb,
        }

    # The typical step's saving. The mean is dragged down by a handful of steps
    # where the mod arm pays first-call compilation for the specialized methods,
    # so the median is what the steady state converts.
    pdiff = [bw[i] - mw[i] for i in plain]
    med_diff = st.median(pdiff)

    # What the fold would have returned without that compilation: every plain
    # step at the median rate, periodic steps as measured.
    cf = med_diff * len(plain) + sum(bw[i] - mw[i] for i in range(n) if i in set(rad) | set(diag))

    pen = [(i, mw[i] - bw[i]) for i in plain if mw[i] > bw[i]]
    pen_total = sum(v for _, v in pen)

    fold = json.load(open(FOLD))
    # One launch per step in the profiled window, so its launch count is the
    # window length and the device saving converts to a per-step figure.
    profiled_steps = fold["kernels"]["microphysics_quadrature"]["baseline"]["launches"]
    device_per_step = fold["fold_attributable"]["ms_saved"] / profiled_steps

    out = {
        "steps": n,
        "baseline_total_s": tb / 1000,
        "mod_total_s": tm / 1000,
        "baseline_mean_ms": tb / n,
        "mod_mean_ms": tm / n,
        "saved_s": saved / 1000,
        "saved_pct": 100 * saved / tb,
        "typical_step": {
            "baseline_median_ms": st.median([bw[i] for i in plain]),
            "mod_median_ms": st.median([mw[i] for i in plain]),
            "saved_median_ms": med_diff,
            "saved_mean_ms": st.mean(pdiff),
            "device_predicted_ms": device_per_step,
            "conversion_pct": 100 * med_diff / device_per_step,
        },
        "counterfactual": {
            "saved_s": cf / 1000,
            "saved_pct": 100 * cf / tb,
            "shortfall_points": 100 * cf / tb - 100 * saved / tb,
        },
        "compile_penalty": {
            "steps": len(pen),
            "pct_of_plain_steps": 100 * len(pen) / len(plain),
            "total_s": pen_total / 1000,
            "ms_per_plain_step": pen_total / len(plain),
            "worst_step": max(pen, key=lambda t: t[1])[0],
            "worst_step_ms": max(v for _, v in pen),
            "last_step_over_200ms": max((i for i, v in pen if v > 200), default=None),
        },
        "classes": {"diagnostics": cls(diag), "radiation": cls(rad), "plain": cls(plain)},
        # Kept so the answer can show why the source TOMLs' own summary fields
        # must not be read across arms: the threshold moves with the median.
        "own_threshold": {
            "baseline_median_ms": b["wall_ms_median"],
            "mod_median_ms": m["wall_ms_median"],
            "baseline_expensive_wall_frac": 100 * b["expensive_wall_frac"],
            "mod_expensive_wall_frac": 100 * m["expensive_wall_frac"],
            "baseline_expensive_steps": b["expensive_steps"],
            "mod_expensive_steps": m["expensive_steps"],
        },
    }
    d = out["classes"]["diagnostics"]
    d["multiple_of_plain_step"] = d["baseline_mean_ms"] / out["classes"]["plain"]["baseline_mean_ms"]
    out["note"] = (
        "Paired by step index: both arms run the same config, so index i is the "
        "same work in both. expensive_steps and expensive_wall_frac in the two "
        "source TOMLs are thresholded against each arm's OWN median and are not "
        "comparable across arms; this file uses fixed periodic classes instead."
    )

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"{n} steps: baseline {tb/1000:.1f}s, mod {tm/1000:.1f}s, saved {saved/1000:.2f}s = {100*saved/tb:.2f}%")
    print(f"typical plain step saves {med_diff:.2f} ms vs {device_per_step:.2f} ms predicted from device time")
    print(f"compile penalty {pen_total/1000:.1f}s on {len(pen)} steps; counterfactual saving {100*cf/tb:.2f}%")
    for k, v in out["classes"].items():
        print(f"  {k:12s} n={v['steps']:5d} {v['pct_of_baseline_wall']:5.1f}% of wall, {v['pct_of_saving']:5.1f}% of saving")


main()

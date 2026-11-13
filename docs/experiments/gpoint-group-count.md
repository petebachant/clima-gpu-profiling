# Longwave g-point group count

The longwave g-point split widens every workspace the solve indexes by column from `ncol` to `ncol * ngpt_groups`, and `RRTMGP.RTE.DEFAULT_NGPT_GROUPS` was set to 16 because 16 was the first group count that worked.
This experiment sweeps the group count to find where the saving stops, and converts it to SYPD by timing real coupler steps in the same process.
Two results: the saving does not stop at 16, and the split is worth <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.74<!-- /calkit value -->% of SYPD at the current default.

## Why this is an in-block comparison and not a difference of runs

`ngpt_groups` turned out to be a constructor argument rather than a compile-time constant, which is what makes this measurable at all on this machine.
All points are taken in one process against one settled AMIP state, rebuilding only the longwave RTE's group workspaces between them.
Everything the solve reads -- atmospheric state, lookups, boundary conditions, and the narrow optics, sources and flux buffers the ungrouped path uses -- is the live simulation's own, so two points differ by the group count and by nothing else.
This matters because the alternative, running the model once per group count and differencing, cannot resolve the effect: one arm of this stack wandered 7.73% between two runs of identical code, which is most of the way to the whole quantity being measured.
The in-block design replaces that noise with a repeat spread of <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.165<!-- /calkit value -->%, so the best saving is resolved <!-- calkit value key=sweep.resolution_vs_spread format="{:.0f}" path=results/docs/gpoint-group-count.json -->82<!-- /calkit value --> times over the sweep's own scatter.

`ngroups = 1` passes `nothing` and takes the ungrouped specialization, so the baseline is a point in the same sweep rather than a separate measurement.
The timing covers the whole both-skies solve, which charges the grouped path for the reduction and the aerosol mask it adds, so these are net figures and not the kernel's alone.

```python calkit stage name=values environment=py outputs=[{path: results/docs/gpoint-group-count.json, storage: git}] inputs=[results/gpoint-sweep.toml, results/gpoint-wallclock.toml, results/gpoint-groups.json, results/gpoint-record/prediction.json, results/sw-gpoint-sweep.toml, results/top-kernels.csv, results/null-census.json]
import csv
import json
import math
import tomllib

sweep = tomllib.load(open("results/gpoint-sweep.toml", "rb"))
wall = tomllib.load(open("results/gpoint-wallclock.toml", "rb"))
prior = json.load(open("results/gpoint-record/prediction.json"))
nsys = json.load(open("results/gpoint-groups.json"))
# The noise floor an identical-arm pair reproduces to, which is what says
# whether a default change could be confirmed by running the model at all.
nulls = json.load(open("results/null-census.json"))
sw = tomllib.load(open("results/sw-gpoint-sweep.toml", "rb"))

counts = sweep["group_counts"]
pts = sweep["points"]
base_ms = pts["1"]["solve_ms"]
best = sweep["best_change_pct"]
spread = sweep["repeat_spread_pct"]


def point(n):
    p = pts[str(n)]
    gib = p["workspace_gib"]
    return {
        "ngroups": n,
        "gpt_per_group": sweep["n_gpt_lw"] // n,
        "solve_ms": round(p["solve_ms"], 1),
        "change_pct": round(p["change_pct"], 2),
        "saved_ms": round(base_ms - p["solve_ms"], 1),
        "workspace_gib": round(gib, 2),
        # What the memory actually bought, so the counts are comparable at
        # different sizes rather than only by their saving.
        "pct_per_gib": round(-p["change_pct"] / gib, 2) if gib else None,
        "flux_deviation": p["worst_flux_deviation"],
    }


# The step from the previous count, which is the quantity a saturating curve
# would drive to zero and this one does not.
def increments():
    out = {}
    for prev, n in zip(counts, counts[1:]):
        out[f"{prev}_to_{n}"] = round(
            pts[str(n)]["change_pct"] - pts[str(prev)]["change_pct"], 2
        )
    return out


# Wave quantization, as a prediction rather than a story. At one resident block
# per SM a point's time is its wave count rounded up to a whole wave times its
# per-block work, and per-block work falls as 1/n because each group covers
# ngpt/n g-points. The grid is linear in the group count, which is checked here
# rather than assumed.
waves = nsys["waves"]
sms = waves["baseline"]["blocks_resident"]
per_group = waves["baseline"]["blocks"]
assert waves["baseline"]["blocks_resident_per_sm"] == 1, "more than one block is resident per SM"
assert waves["mod"]["blocks"] == per_group * waves["groups"], "grid is not linear in the group count"


def quantized(n):
    return math.ceil(per_group * n / sms) / n


def predicted(n):
    return 100.0 * (quantized(n) / quantized(1) - 1.0)


def quantization(n):
    return {
        "blocks": per_group * n,
        "waves": round(per_group * n / sms, 2),
        "waves_ceil": math.ceil(per_group * n / sms),
        "predicted_change_pct": round(predicted(n), 2),
        "residual_pct": round(pts[str(n)]["change_pct"] - predicted(n), 2),
    }


# What one more reservation would test: double the best count, holding the
# residual fixed, which is the only part of the curve the model does not cover.
nxt = 2 * max(counts)
residual_at_best = pts[str(max(counts))]["change_pct"] - predicted(max(counts))
next_point = {
    "ngroups": nxt,
    "blocks": per_group * nxt,
    "waves": round(per_group * nxt / sms, 2),
    "predicted_change_pct": round(predicted(nxt), 2),
    "predicted_gain_pts": round(predicted(nxt) - predicted(max(counts)), 2),
    "predicted_total_pct": round(predicted(nxt) + residual_at_best, 2),
    "workspace_gib": round(pts[str(max(counts))]["workspace_gib"] * 2, 2),
}

lw = nsys["longwave"]
# The nsys pass times the window, not a call, so divide out its launch count to
# compare an absolute per-solve cost against the sweep's.
per_launch = lw["baseline"]["elapsed_ms"] / lw["baseline"]["launches"]
sweep_saved = base_ms - pts[str(sweep["sufficient_ngroups"])]["solve_ms"]

# SYPD has three defensible denominators and they differ by more than the
# effect, so all three are derived rather than one being chosen silently. A
# block is 180 steps; the daily diagnostics write lands in one block in every
# write_period_steps / block_steps, and the FIRST write also compiles its path,
# which is a one-off a long run amortizes away.
def wallclock():
    secs = sorted(wall["block_seconds"])
    n = len(secs)
    nw = (n * wall["block_steps"]) // wall["write_period_steps"]
    ordinary, writes = secs[: n - nw], secs[n - nw :]
    blocks_per_write = wall["write_period_steps"] // wall["block_steps"]

    def mean(xs):
        return sum(xs) / len(xs)

    def sd(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    ord_mean = mean(ordinary)
    ord_se_pct = 100.0 * (sd(ordinary) / len(ordinary) ** 0.5) / ord_mean
    # The steady write cost, with the first write's compilation dropped.
    steady_write = mean(writes[:-1]) if len(writes) > 1 else writes[0]
    cycle_s = (blocks_per_write - 1) * ord_mean + steady_write
    window_s = sum(secs)
    events_per_cycle = wall["write_period_steps"] // wall["rad_period_steps"]
    events_per_block = wall["block_steps"] // wall["rad_period_steps"]
    sim_cycle = wall["write_period_steps"] * wall["dt_cpl_s"]

    def sypd(sim, wallsec):
        return sim / wallsec / 365.25

    arms = {}
    for k, a in wall["arms"].items():
        per_event = a["solve_saved_per_event_ms"] / 1e3
        saved_cycle = events_per_cycle * per_event
        saved_window = (n * wall["block_steps"] / wall["rad_period_steps"]) * per_event
        arms[k] = {
            "ngroups": a["ngroups"],
            "saved_per_event_ms": round(a["solve_saved_per_event_ms"], 1),
            # Ordinary steps only, which is the rate between writes.
            "gain_ordinary_pct": round(
                100.0 * events_per_block * per_event / ord_mean, 3
            ),
            # Steady state: writes at their real cadence, first-write compile dropped.
            "gain_steady_pct": round(100.0 * saved_cycle / cycle_s, 3),
            "sypd_steady": round(sypd(sim_cycle, cycle_s), 4),
            "sypd_steady_ungrouped": round(sypd(sim_cycle, cycle_s + saved_cycle), 4),
            # The timed window exactly as it ran, compile included.
            "gain_window_pct": round(100.0 * saved_window / window_s, 3),
            # The ratio's own uncertainty: the denominator's relative error
            # scales the gain, it does not add to it.
            "gain_steady_uncertainty_pts": round(
                (100.0 * saved_cycle / cycle_s) * ord_se_pct / 100.0, 4
            ),
        }

    return {
        "blocks": n,
        "block_steps": wall["block_steps"],
        "timed_steps": n * wall["block_steps"],
        "rad_period_steps": wall["rad_period_steps"],
        "write_period_steps": wall["write_period_steps"],
        "write_blocks": nw,
        "ordinary_blocks": n - nw,
        "ordinary_mean_s": round(ord_mean, 2),
        "ordinary_spread_pct": round(100.0 * (ordinary[-1] - ordinary[0]) / ord_mean, 2),
        "ordinary_rel_se_pct": round(ord_se_pct, 3),
        "ordinary_step_ms": round(1e3 * ord_mean / wall["block_steps"], 1),
        "first_write_block_s": round(writes[-1], 1),
        "steady_write_block_s": round(steady_write, 1),
        "write_overhead_pct": round(100.0 * (steady_write - ord_mean) / ord_mean, 1),
        "steady_step_ms": round(1e3 * cycle_s / wall["write_period_steps"], 1),
        "window_total_s": round(window_s, 1),
        "window_sypd": round(
            sypd(n * wall["block_steps"] * wall["dt_cpl_s"], window_s), 4
        ),
        "arms": arms,
    }


# The prediction this document made for 32 before it was measured, read out of a
# committed snapshot rather than retyped, and scored against two things: the
# measurement, and the amount the 16-group figure itself moved between the two
# atmospheric states the sweeps were taken at.
def prediction():
    pred = prior["predicted"]
    measured = pts["32"]["change_pct"]
    err = measured - pred["predicted_total_pct"]
    shift = pts["16"]["change_pct"] - prior["measured_then"]["16"]
    return {
        "source_commit": prior["source_commit"][:7],
        "ngroups": pred["ngroups"],
        "predicted_pct": pred["predicted_total_pct"],
        "predicted_quantization_pct": pred["predicted_change_pct"],
        "predicted_gib": pred["workspace_gib"],
        "measured_pct": round(measured, 2),
        "measured_gib": round(pts["32"]["workspace_gib"], 2),
        "error_pts": round(err, 2),
        "prior_16_pct": prior["measured_then"]["16"],
        "current_16_pct": round(pts["16"]["change_pct"], 2),
        "state_shift_pts": round(shift, 2),
        # The honest score: a prediction is only as good as the method's own
        # reproducibility across states, which is the larger of the two here.
        "error_within_state_shift": abs(err) <= abs(shift),
        "prior_spread_pct": prior["repeat_spread_pct"],
    }


# The claim the correctness section makes is that the deviation does not
# accumulate with the number of partials, so what is checked is monotonicity
# across the counts and not the size of the worst one.
def correctness():
    grouped = [n for n in counts if n > 1]
    devs = [pts[str(n)]["worst_flux_deviation"] for n in grouped]
    return {
        "grouped_counts": len(grouped),
        "min_deviation": min(devs),
        "max_deviation": max(devs),
        "spread_pct": round(100.0 * (max(devs) - min(devs)) / min(devs), 1),
        "rises_with_count": devs == sorted(devs),
        "worst_at_largest": devs[-1] == max(devs),
    }


# What a change to the default is worth against the whole device budget, which
# is the figure it has to justify. One default sets the group count for both
# solves, so both are priced. The sweeps time one solve in isolation and the
# profile says how many solves the window runs and how large the budget is, so
# the conversion is absolute milliseconds times launches rather than a share of
# a share -- the mistake that read a per-call share as a device share once here
# already.
def decision():
    rows = {}
    for r in csv.DictReader(open("results/top-kernels.csv")):
        for band, prefix in (
            ("lw", "rte_lw_2stream_solve_grouped"),
            ("sw", "rte_sw_2stream_solve_grouped"),
        ):
            if r["kernel"].startswith(prefix):
                rows[band] = r
    assert set(rows) == {"lw", "sw"}, f"profile has {sorted(rows)}, not both grouped solves"

    # The budget, derived from each row's time and its own share of it. The two
    # rows agreeing is the check that they came from one profile.
    budgets = {
        b: 100.0 * float(r["total_ms_mod"]) / float(r["pct_of_gpu_time_mod"])
        for b, r in rows.items()
    }
    device_ms = sum(budgets.values()) / len(budgets)
    assert abs(budgets["lw"] - budgets["sw"]) / device_ms < 0.01, "rows disagree on the device total"

    bands = {"lw": pts, "sw": sw["points"]}
    launches = {b: float(rows[b]["invocations_mod"]) for b in rows}
    default = sweep["default_ngpt_groups"]
    assert sw["default_ngpt_groups"] == default, "the two sweeps disagree on the default"
    above = [n for n in counts if n > default]
    assert len(above) >= 2, "needs two counts above the default to price a second doubling"

    def at(n):
        saved = {
            b: bands[b][str(default)]["solve_ms"] - bands[b][str(n)]["solve_ms"] for b in bands
        }
        window_ms = sum(saved[b] * launches[b] for b in bands)
        gib = sum(
            bands[b][str(n)]["workspace_gib"] - bands[b][str(default)]["workspace_gib"]
            for b in bands
        )
        return {
            "ngroups": n,
            "lw_saved_per_solve_ms": round(saved["lw"], 1),
            "sw_saved_per_solve_ms": round(saved["sw"], 1),
            "window_saved_ms": round(window_ms, 1),
            "pct_of_device": round(100.0 * window_ms / device_ms, 3),
            "extra_gib": round(gib, 2),
            "device_pct_per_gib": round(100.0 * window_ms / device_ms / gib, 4) if gib else None,
        }

    first, last = at(above[0]), at(above[-1])
    marg_pct = last["pct_of_device"] - first["pct_of_device"]
    marg_gib = last["extra_gib"] - first["extra_gib"]
    return {
        "device_ms": round(device_ms, 1),
        "default_ngroups": default,
        "default_gib": round(
            sum(bands[b][str(default)]["workspace_gib"] for b in bands), 2
        ),
        "launches": int(launches["lw"]),
        # The conversion rests on the sweep and the profile describing the same
        # kernel, at different states, so the gap between them is recorded.
        "profile_vs_sweep_pct": {
            b: round(
                100.0
                * (float(rows[b]["mean_ms_mod"]) - bands[b][str(default)]["solve_ms"])
                / bands[b][str(default)]["solve_ms"],
                1,
            )
            for b in bands
        },
        "points": {str(n): at(n) for n in above},
        "marginal_pct_of_device": round(marg_pct, 3),
        "marginal_gib": round(marg_gib, 2),
        "marginal_device_pct_per_gib": round(marg_pct / marg_gib, 4),
        "per_gib_ratio": round(first["device_pct_per_gib"] / (marg_pct / marg_gib), 1),
        "null_floor_pct": nulls["current_window_mean_pct"],
        "best_below_null_floor": last["pct_of_device"] < nulls["current_window_mean_pct"],
    }


out = {
    "sweep": {
        "ncol": sweep["ncol"],
        "nlay": sweep["nlay"],
        "n_gpt_lw": sweep["n_gpt_lw"],
        "float_type": sweep["float_type"],
        "repeats": sweep["repeats"],
        "warmup_steps": sweep["warmup_steps"],
        "group_counts": ", ".join(str(n) for n in counts),
        "repeat_spread_pct": round(spread, 3),
        "best_change_pct": round(best, 2),
        "resolution_vs_spread": round(abs(best) / spread, 1),
        "sufficient_ngroups": sweep["sufficient_ngroups"],
        "default_ngpt_groups": sweep["default_ngpt_groups"],
        "default_is_sufficient_only": sweep["default_is_sufficient_only"],
        # The guard on the headline: "smallest sufficient" is bounded by what was
        # swept, so when it lands on the largest count the sweep has not found a
        # knee at all and cannot distinguish enough from most-tried.
        "sufficient_is_sweep_maximum": sweep["sufficient_ngroups"] == max(counts),
        "threshold_pct": round(best + spread, 2),
    },
    "points": {str(n): point(n) for n in counts},
    "increments": increments(),
    "quantization": {
        "sms": sms,
        "blocks_per_group": per_group,
        "points": {str(n): quantization(n) for n in counts},
        "next": next_point,
    },
    "wallclock": wallclock(),
    "correctness": correctness(),
    "decision": decision(),
    "prediction": prediction(),
    "nsys": {
        "baseline_per_launch_ms": round(per_launch, 1),
        "launches": lw["baseline"]["launches"],
        "sweep_ungrouped_ms": round(base_ms, 1),
        "per_launch_gap_pct": round(100.0 * (per_launch - base_ms) / base_ms, 1),
        "window_change_pct": round(lw["change_pct"], 2),
        "window_saved_ms": round(lw["saved_ms"], 1),
        "window_saved_per_launch_ms": round(
            lw["saved_ms"] / lw["baseline"]["launches"], 1
        ),
        "sweep_saved_per_solve_ms": round(sweep_saved, 1),
        "pct_of_device_baseline": lw["pct_of_device_baseline"],
    },
}
json.dump(out, open("results/docs/gpoint-group-count.json", "w"), indent=2)
```

## The curve

The sweep ran <!-- calkit value key=sweep.repeats path=results/docs/gpoint-group-count.json -->7<!-- /calkit value --> timed repeats per point at group counts <!-- calkit value key=sweep.group_counts path=results/docs/gpoint-group-count.json -->1, 4, 8, 16, 32, 64<!-- /calkit value -->, on <!-- calkit value key=sweep.ncol format="{:,}" path=results/docs/gpoint-group-count.json -->24,576<!-- /calkit value --> columns and <!-- calkit value key=sweep.nlay path=results/docs/gpoint-group-count.json -->80<!-- /calkit value --> layers at <!-- calkit value key=sweep.n_gpt_lw path=results/docs/gpoint-group-count.json -->256<!-- /calkit value --> longwave g-points.

| groups | g-points each | solve (ms) | change | workspace (GiB) | per GiB |
|---|---|---|---|---|---|
| 1 | <!-- calkit value key=points.1.gpt_per_group path=results/docs/gpoint-group-count.json -->256<!-- /calkit value --> | <!-- calkit value key=points.1.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->671.4<!-- /calkit value --> | -- | 0 | -- |
| 4 | <!-- calkit value key=points.4.gpt_per_group path=results/docs/gpoint-group-count.json -->64<!-- /calkit value --> | <!-- calkit value key=points.4.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->651.3<!-- /calkit value --> | <!-- calkit value key=points.4.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.99<!-- /calkit value -->% | <!-- calkit value key=points.4.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.46<!-- /calkit value --> | <!-- calkit value key=points.4.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->6.50<!-- /calkit value -->%/GiB |
| 8 | <!-- calkit value key=points.8.gpt_per_group path=results/docs/gpoint-group-count.json -->32<!-- /calkit value --> | <!-- calkit value key=points.8.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->640.4<!-- /calkit value --> | <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-4.61<!-- /calkit value -->% | <!-- calkit value key=points.8.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.91<!-- /calkit value --> | <!-- calkit value key=points.8.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->5.04<!-- /calkit value -->%/GiB |
| 16 | <!-- calkit value key=points.16.gpt_per_group path=results/docs/gpoint-group-count.json -->16<!-- /calkit value --> | <!-- calkit value key=points.16.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->603.9<!-- /calkit value --> | <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.05<!-- /calkit value -->% | <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.81<!-- /calkit value --> | <!-- calkit value key=points.16.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->5.54<!-- /calkit value -->%/GiB |
| 32 | <!-- calkit value key=points.32.gpt_per_group path=results/docs/gpoint-group-count.json -->8<!-- /calkit value --> | <!-- calkit value key=points.32.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->586.3<!-- /calkit value --> | <!-- calkit value key=points.32.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-12.67<!-- /calkit value -->% | <!-- calkit value key=points.32.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.62<!-- /calkit value --> | <!-- calkit value key=points.32.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.50<!-- /calkit value -->%/GiB |

## The sweep has no knee, and after three tries that is the result

The smallest group count within one repeat spread of the best saving is <!-- calkit value key=sweep.sufficient_ngroups path=results/docs/gpoint-group-count.json -->64<!-- /calkit value -->, against a default of <!-- calkit value key=sweep.default_ngpt_groups path=results/docs/gpoint-group-count.json -->16<!-- /calkit value -->, so `default_is_sufficient_only` is <!-- calkit value key=sweep.default_is_sufficient_only path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->.
Sixty-four groups reach <!-- calkit value key=points.64.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.53<!-- /calkit value -->% where sixteen reach <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.05<!-- /calkit value -->%, so the default gives back about a quarter of the saving this sweep can reach.
The threshold is measured rather than assumed: a point counts as sufficient if it is within <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.165<!-- /calkit value -->% of the best, which puts the bar at <!-- calkit value key=sweep.threshold_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.37<!-- /calkit value -->% and only 64 clears it.

`sufficient_is_sweep_maximum` is <!-- calkit value key=sweep.sufficient_is_sweep_maximum path=results/docs/gpoint-group-count.json -->true<!-- /calkit value --> for the third sweep running, and the third time stops being a caveat and becomes the finding.
"Smallest sufficient" is bounded by what was swept, so landing on the largest count measured says only that the curve is still falling at the endpoint: the bound has moved from 16-or-beyond to 32-or-beyond to 64-or-beyond, and the rule will go on returning whatever count the sweep stopped at.
A threshold cannot answer this question, because the saving does not saturate before the memory does -- the per-GiB column falls monotonically past 16, from <!-- calkit value key=points.16.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->5.54<!-- /calkit value -->%/GiB there to <!-- calkit value key=points.32.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.50<!-- /calkit value --> at 32 and <!-- calkit value key=points.64.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.87<!-- /calkit value --> at 64.
So the group count is a price to pay or not, and it is priced below against the device budget instead of against the sweep's own best.

## The increments rise to 16 and then fall

| step | change in saving |
|---|---|
| 1 to 4 | <!-- calkit value key=increments.1_to_4 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.99<!-- /calkit value --> points |
| 4 to 8 | <!-- calkit value key=increments.4_to_8 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.62<!-- /calkit value --> points |
| 8 to 16 | <!-- calkit value key=increments.8_to_16 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-5.44<!-- /calkit value --> points |
| 16 to 32 | <!-- calkit value key=increments.16_to_32 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.62<!-- /calkit value --> points |
| 32 to 64 | <!-- calkit value key=increments.32_to_64 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.86<!-- /calkit value --> points |

The earliest sweep stopped at 16 and so saw only the rising half, which read as a curve that was still accelerating.
With 32 and 64 measured the shape is a hump: the step into 16 is the largest, the step into 32 is less than half of it, and the step into 64 is <!-- calkit value key=increments.32_to_64 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.86<!-- /calkit value --> points, a third of the step before it.
Saturation is real and slow, and it took two points past the sweep that could not see it to show up.

## Wave quantization accounts for the jumps and not the rest

The grid is linear in the group count -- <!-- calkit value key=quantization.blocks_per_group path=results/docs/gpoint-group-count.json -->96<!-- /calkit value --> blocks ungrouped, checked in the values block against the count recorded at sixteen groups rather than assumed -- and one block is resident per SM at 255 registers.
So a point's time is its wave count rounded up to a whole wave times its per-block work, and per-block work falls as 1/n because each group covers one nth of the g-points.
On <!-- calkit value key=quantization.sms path=results/docs/gpoint-group-count.json -->108<!-- /calkit value --> SMs that model predicts nothing at all for 4 and 8 groups, whose block counts divide into whole waves, and a step at 16, again at 32 and a smaller one at 64, where the rounding up costs progressively less than the extra parallelism buys.

| groups | blocks | waves | predicted | measured | residual |
|---|---|---|---|---|---|
| 4 | <!-- calkit value key=quantization.points.4.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->384<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.56<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=points.4.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.99<!-- /calkit value -->% | <!-- calkit value key=quantization.points.4.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.99<!-- /calkit value --> |
| 8 | <!-- calkit value key=quantization.points.8.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->768<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->7.11<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-4.61<!-- /calkit value -->% | <!-- calkit value key=quantization.points.8.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-4.61<!-- /calkit value --> |
| 16 | <!-- calkit value key=quantization.points.16.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->1,536<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->14.22<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-6.25<!-- /calkit value -->% | <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.05<!-- /calkit value -->% | <!-- calkit value key=quantization.points.16.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.80<!-- /calkit value --> |
| 32 | <!-- calkit value key=quantization.points.32.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->3,072<!-- /calkit value --> | <!-- calkit value key=quantization.points.32.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->28.44<!-- /calkit value --> | <!-- calkit value key=quantization.points.32.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-9.38<!-- /calkit value -->% | <!-- calkit value key=points.32.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-12.67<!-- /calkit value -->% | <!-- calkit value key=quantization.points.32.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.30<!-- /calkit value --> |
| 64 | <!-- calkit value key=quantization.points.64.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->6,144<!-- /calkit value --> | <!-- calkit value key=quantization.points.64.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->56.89<!-- /calkit value --> | <!-- calkit value key=quantization.points.64.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.94<!-- /calkit value -->% | <!-- calkit value key=points.64.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.53<!-- /calkit value -->% | <!-- calkit value key=quantization.points.64.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.59<!-- /calkit value --> |

The residual is a few points at every count above one and does not grow with the count, so most of what the increments do is quantization and the rest is something flat that arrives as soon as there is more than one group.
This sweep does not isolate that part.

## The prediction this document made for 32, and how it scored

The earlier version of this document predicted 32 groups from the quantization model plus the residual at 16, before 32 was measured.
That prediction is read here out of a committed snapshot of the earlier run rather than retyped, at `<!-- calkit value key=prediction.source_commit path=results/docs/gpoint-group-count.json -->8d91bb5<!-- /calkit value -->`.

| | |
|---|---|
| predicted | <!-- calkit value key=prediction.predicted_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-12.08<!-- /calkit value -->% |
| measured | <!-- calkit value key=prediction.measured_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-12.67<!-- /calkit value -->% |
| error | <!-- calkit value key=prediction.error_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.59<!-- /calkit value --> points |
| predicted workspace | <!-- calkit value key=prediction.predicted_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.63<!-- /calkit value --> GiB |
| measured workspace | <!-- calkit value key=prediction.measured_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.62<!-- /calkit value --> GiB |

The memory prediction was essentially exact, which is unsurprising since it is arithmetic on shapes.
The timing prediction came within <!-- calkit value key=prediction.error_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.59<!-- /calkit value --> points, and the right way to score that is against the method's own reproducibility rather than against zero.
Between the two sweeps the 16-group figure itself moved from <!-- calkit value key=prediction.prior_16_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-8.95<!-- /calkit value -->% to <!-- calkit value key=prediction.current_16_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.05<!-- /calkit value -->%, a shift of <!-- calkit value key=prediction.state_shift_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.10<!-- /calkit value --> points, because the second sweep was taken after the timed blocks at a different atmospheric state.
So `error_within_state_shift` is <!-- calkit value key=prediction.error_within_state_shift path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->: the prediction is confirmed at about the precision this method actually has, which is set by the state and not by the repeat spread.

That last point is worth holding onto, because it bounds every percentage in this document.
The repeat spread of <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.165<!-- /calkit value -->% is the scatter of seven timings of one state, and it is nearly four times smaller than the <!-- calkit value key=prediction.state_shift_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.10<!-- /calkit value --> points the same quantity moved between two states.
A figure here is reproducible to a fraction of a percent at a fixed state and to about a point across states.

## What it is worth in SYPD, measured

The same job timed <!-- calkit value key=wallclock.blocks path=results/docs/gpoint-group-count.json -->12<!-- /calkit value --> blocks of <!-- calkit value key=wallclock.block_steps path=results/docs/gpoint-group-count.json -->180<!-- /calkit value --> real coupler steps -- <!-- calkit value key=wallclock.timed_steps format="{:,}" path=results/docs/gpoint-group-count.json -->2,160<!-- /calkit value --> steps in all, with the live grouped solver, the real diagnostics and the real coupling -- using the wall clock the coupler's own SYPD is defined on.
Blocks are <!-- calkit value key=wallclock.block_steps path=results/docs/gpoint-group-count.json -->180<!-- /calkit value --> steps because that is a whole number of <!-- calkit value key=wallclock.rad_period_steps path=results/docs/gpoint-group-count.json -->30<!-- /calkit value -->-step radiation periods, so every block carries identical radiation work.

The one arithmetic step is the comparison arm: the grouped SYPD is measured, and the ungrouped SYPD is that measured block with the measured in-block saving added back, rather than a second run differenced against it.
That is justified because the solve is synchronous on the step's critical path, and the alternative cannot resolve a sub-percent effect on this machine.
It is stated here rather than left implicit, and it is the only place in this document where a number is not measured directly.

SYPD has three defensible denominators and they differ by more than the effect, so all three are derived:

| basis | step (ms) | gain at 16 groups |
|---|---|---|
| ordinary steps, between writes | <!-- calkit value key=wallclock.ordinary_step_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->288.9<!-- /calkit value --> | <!-- calkit value key=wallclock.arms.16.gain_ordinary_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.779<!-- /calkit value -->% |
| steady state, writes at their cadence | <!-- calkit value key=wallclock.steady_step_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->304.9<!-- /calkit value --> | <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.738<!-- /calkit value -->% |
| the timed window exactly as it ran | -- | <!-- calkit value key=wallclock.arms.16.gain_window_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.700<!-- /calkit value -->% |

The steady-state row is the one to quote: the daily diagnostics write lands in <!-- calkit value key=wallclock.write_blocks path=results/docs/gpoint-group-count.json -->3<!-- /calkit value --> of the <!-- calkit value key=wallclock.blocks path=results/docs/gpoint-group-count.json -->12<!-- /calkit value --> blocks and costs <!-- calkit value key=wallclock.write_overhead_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->22.1<!-- /calkit value -->% on the block it falls in, so a basis that excludes it is too fast and the raw window is too slow.
The raw window is too slow because the first write also compiles its path: that block took <!-- calkit value key=wallclock.first_write_block_s format="{:.1f}" path=results/docs/gpoint-group-count.json -->99.5<!-- /calkit value --> s against <!-- calkit value key=wallclock.steady_write_block_s format="{:.1f}" path=results/docs/gpoint-group-count.json -->63.5<!-- /calkit value --> s for the later ones, a one-off a long run amortizes away.

On the steady-state basis the split takes SYPD from <!-- calkit value key=wallclock.arms.16.sypd_steady_ungrouped format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0697<!-- /calkit value --> to <!-- calkit value key=wallclock.arms.16.sypd_steady format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0776<!-- /calkit value --> at the default 16 groups, a gain of <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.738<!-- /calkit value -->%.
At 32 groups it would be <!-- calkit value key=wallclock.arms.32.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.930<!-- /calkit value -->%, from a solve saving of <!-- calkit value key=wallclock.arms.32.saved_per_event_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->85.1<!-- /calkit value --> ms per radiation event against <!-- calkit value key=wallclock.arms.16.saved_per_event_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->67.5<!-- /calkit value --> ms at 16.

The uncertainty on that gain is <!-- calkit value key=wallclock.arms.16.gain_steady_uncertainty_pts format="{:.4f}" path=results/docs/gpoint-group-count.json -->0.0032<!-- /calkit value --> percentage points, which is smaller than it looks like it should be, and the reason is worth stating.
The gain is a ratio of a measured saving to a measured wall clock, not a difference of two runs, so the denominator's relative error scales the gain instead of adding to it: <!-- calkit value key=wallclock.ordinary_rel_se_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.436<!-- /calkit value -->% of <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.738<!-- /calkit value -->% is a few thousandths of a point.
The `gain_below_block_iqr` flag the measurement script writes is therefore true and should not be read as "unresolved" -- it compares a gain against a block-to-block spread as though the gain had been obtained by differencing blocks, which is exactly what this method avoids.

These SYPD figures are steady-state rates measured mid-run and are **not** comparable to the pipeline's end-to-end SYPD, which includes first-call compilation over the whole job.
For orientation, the same window read as a single measurement gives <!-- calkit value key=wallclock.window_sypd format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0218<!-- /calkit value -->, and it carries that one-off compile.

## What it costs

Sixteen groups take <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.81<!-- /calkit value --> GiB of wide longwave workspace, thirty-two take <!-- calkit value key=points.32.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.62<!-- /calkit value --> and sixty-four take <!-- calkit value key=points.64.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->7.23<!-- /calkit value -->, measured as a free-memory delta across the workspace construction rather than computed from shapes.
That is the whole cost of the treatment and it scales linearly in the group count, so this is the constraint that decides how far the count can go.
The model predicts <!-- calkit value key=quantization.next.ngroups path=results/docs/gpoint-group-count.json -->128<!-- /calkit value --> groups at <!-- calkit value key=quantization.next.predicted_total_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.53<!-- /calkit value -->% for <!-- calkit value key=quantization.next.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->14.46<!-- /calkit value --> GiB, which is <!-- calkit value key=quantization.next.predicted_gain_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value --> points of quantization gain for another doubling of memory: past 64 the wave count's rounding stops changing at all, so the model says its half of the curve is finished.

Before 64 was measured, this document predicted at `9e2bfcc` that "the curve and the memory are now pulling in the same direction and 64 is unlikely to be worth it".
The measurement bears that out, and the section below prices both doublings in device time rather than in points of one solve.

## What a default change is worth against the device budget

Points of a solve are not the unit a default change is decided in, and one default sets the group count for both the longwave and the shortwave, so both are priced here.
The sweeps time one solve in isolation and the profile says how many of them the window runs and how large the budget is, so the conversion is absolute milliseconds times launch count rather than a share of a share.
It rests on the sweep and the profile describing the same kernel, which is checked rather than assumed: the profiled mod arm's per-solve cost differs from the sweep's by <!-- calkit value key=decision.profile_vs_sweep_pct.lw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-1.4<!-- /calkit value -->% in the longwave and <!-- calkit value key=decision.profile_vs_sweep_pct.sw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-1.4<!-- /calkit value -->% in the shortwave, at a different atmospheric state.

| groups | longwave | shortwave | over the window | of device | extra workspace |
|---|---|---|---|---|---|
| 32 | <!-- calkit value key=decision.points.32.lw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->17.6<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.sw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->9.3<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.window_saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->107.4<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.323<!-- /calkit value -->% | <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.86<!-- /calkit value --> GiB |
| 64 | <!-- calkit value key=decision.points.64.lw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->23.4<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.sw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->14.5<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.window_saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->151.5<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.455<!-- /calkit value -->% | <!-- calkit value key=decision.points.64.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->11.55<!-- /calkit value --> GiB |

Against a device budget of <!-- calkit value key=decision.device_ms format="{:,.0f}" path=results/docs/gpoint-group-count.json -->33,306<!-- /calkit value --> ms, thirty-two groups are worth <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.323<!-- /calkit value -->% for <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.86<!-- /calkit value --> GiB on top of the default's <!-- calkit value key=decision.default_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.86<!-- /calkit value --> GiB -- the same figure twice, because the workspace is linear in the count and doubling it doubles the workspace.
Sixty-four are worth <!-- calkit value key=decision.points.64.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.455<!-- /calkit value -->%, so the second doubling adds <!-- calkit value key=decision.marginal_pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.132<!-- /calkit value --> points of device time for <!-- calkit value key=decision.marginal_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->7.69<!-- /calkit value --> GiB -- <!-- calkit value key=decision.per_gib_ratio format="{:.1f}" path=results/docs/gpoint-group-count.json -->4.9<!-- /calkit value --> times worse per GiB than the first.
Both figures are small enough that they cannot be confirmed by running the model: `best_below_null_floor` is <!-- calkit value key=decision.best_below_null_floor path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->, because even 64 groups come to a fraction of the <!-- calkit value key=decision.null_floor_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->1.869<!-- /calkit value -->% that two identical arms of this stack reproduce each other to.
They also inherit this experiment's state dependence: the 16-group figure itself moved <!-- calkit value key=prediction.state_shift_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.10<!-- /calkit value --> points between the two states sweeps have been taken at, so read the device shares as good to roughly a tenth of themselves rather than to the digits they are printed with.

## Correctness

The split is not bit-for-bit and cannot be, because a g-point sum taken in N partials is a different summation order.
The worst flux deviation from the ungrouped path over the <!-- calkit value key=correctness.grouped_counts path=results/docs/gpoint-group-count.json -->5<!-- /calkit value --> grouped counts is <!-- calkit value key=correctness.max_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->8.24e-04<!-- /calkit value --> W/m2, against fluxes of order a few hundred W/m2, and it does not grow with the group count.
The five points span only <!-- calkit value key=correctness.spread_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->8.0<!-- /calkit value -->% of their own smallest value, <!-- calkit value key=correctness.min_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->7.63e-04<!-- /calkit value --> W/m2, and `rises_with_count` is <!-- calkit value key=correctness.rises_with_count path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->: the worst value does sit at 64 groups, but 4 groups ties it exactly and 8 and 16 are the lowest on the curve.
A reordering error that accumulated with the number of partials would rise across the sweep, and this one is flat, which is the expected behavior for <!-- calkit value key=sweep.float_type path=results/docs/gpoint-group-count.json -->Float32<!-- /calkit value --> rounding rather than for a bug in the reduction.

## Two independent cross-checks against the nsys pass

The sweep and the profiling pass measure the same solve by different means, on different stacks and at different points in the simulation, so agreement is a check on both.
The nsys baseline arm spent <!-- calkit value key=nsys.baseline_per_launch_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->665.6<!-- /calkit value --> ms per launch in the ungrouped longwave kernel, against the sweep's <!-- calkit value key=nsys.sweep_ungrouped_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->671.4<!-- /calkit value --> ms for its ungrouped point, a gap of <!-- calkit value key=nsys.per_launch_gap_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->-0.9<!-- /calkit value -->%.
That is closer than the previous sweep managed, and for a reason that is not to anyone's credit: this sweep ran after <!-- calkit value key=wallclock.timed_steps format="{:,}" path=results/docs/gpoint-group-count.json -->2,160<!-- /calkit value --> steps rather than after <!-- calkit value key=sweep.warmup_steps path=results/docs/gpoint-group-count.json -->3<!-- /calkit value -->, so its state is nearer the profiled window's than the earlier one's was.
The saving agrees too: the pass measured <!-- calkit value key=nsys.window_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.43<!-- /calkit value -->% at 16 groups over its window, against the sweep's in-block <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-10.05<!-- /calkit value -->% at the same group count.

## Verdict

The split is worth <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.738<!-- /calkit value -->% of SYPD at the shipped default, measured rather than converted, and that remains the result worth quoting.
Raising the default is worth having and worth very little: thirty-two groups is the setting, at <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.323<!-- /calkit value -->% of device time for <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.86<!-- /calkit value --> GiB, and the last doubling to 64 costs <!-- calkit value key=decision.marginal_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->7.69<!-- /calkit value --> GiB more for <!-- calkit value key=decision.marginal_pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.132<!-- /calkit value --> points, <!-- calkit value key=decision.per_gib_ratio format="{:.1f}" path=results/docs/gpoint-group-count.json -->4.9<!-- /calkit value --> times worse per GiB.
Since neither figure can be resolved by running the model, the change should ride along with the next pass that re-profiles for some other reason rather than spend a reservation to re-measure the arms around it.
The edit is no longer a constant, either: on `pb/gpoint-groups` the 16 this sweep measured has been replaced by a grid-sized default, `clamp(cld(TARGET_GPT_GROUP_COLUMNS, ncol), 1, MAX_NGPT_GROUPS)`, which at this grid's <!-- calkit value key=sweep.ncol format="{:,}" path=results/docs/gpoint-group-count.json -->24,576<!-- /calkit value --> columns evaluates to the same 16, so raising it means doubling both of those and not editing one number.
What this document still does not establish is a knee, a mechanism for the flat residual, or any figure reproducible across atmospheric states to better than about a point.

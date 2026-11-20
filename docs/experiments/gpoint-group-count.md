# Longwave g-point group count

The longwave g-point split widens every workspace the solve indexes by column from `ncol` to `ncol * ngpt_groups`, and the group count was set to 16 because 16 was the first count that worked.
This experiment sweeps the group count to find where the saving stops, and converts it to SYPD by timing real coupler steps in the same process.
Two results: on the split kernels the saving is all but finished by eight groups, and sixteen are worth <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->2.39<!-- /calkit value -->% of SYPD against the ungrouped path.

## Why this is an in-block comparison and not a difference of runs

`ngpt_groups` turned out to be a constructor argument rather than a compile-time constant, which is what makes this measurable at all on this machine.
All points are taken in one process against one settled AMIP state, rebuilding only the longwave RTE's group workspaces between them.
Everything the solve reads -- atmospheric state, lookups, boundary conditions, and the narrow optics, sources and flux buffers the ungrouped path uses -- is the live simulation's own, so two points differ by the group count and by nothing else.
This matters because the alternative, running the model once per group count and differencing, cannot resolve the effect: one arm of this stack wandered 7.73% between two runs of identical code, which is most of the way to the whole quantity being measured.
The in-block design replaces that noise with a repeat spread of <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.327<!-- /calkit value -->%, so the best saving is resolved <!-- calkit value key=sweep.resolution_vs_spread format="{:.0f}" path=results/docs/gpoint-group-count.json -->104<!-- /calkit value --> times over the sweep's own scatter.

`ngroups = 1` passes `nothing` and takes the ungrouped specialization, so the baseline is a point in the same sweep rather than a separate measurement.
The timing covers the whole both-skies solve, which charges the grouped path for the reduction and the aerosol mask it adds, so these are net figures and not the kernel's alone.

```python calkit stage name=values environment=py outputs=[{path: results/docs/gpoint-group-count.json, storage: git}] inputs=[results/gpoint-sweep.toml, results/gpoint-wallclock.toml, results/gpoint-groups.json, results/gpoint-record/prediction.json, results/sw-gpoint-sweep.toml, results/experiments/radiation-phase-split/split.toml, results/top-kernels.csv, results/null-census.json]
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
# Registers and the driver-checked occupancy of the split kernels, which is
# what sets how many of a round's blocks are resident at once.
split = tomllib.load(open("results/experiments/radiation-phase-split/split.toml", "rb"))

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


# Wave quantization, rebuilt for the split. A grouped solve launches
# ceil(n_gpt/n) rounds, each a grid of ncol*n threads, and a round costs its
# block count rounded up to a whole wave. Counting is in threads because the
# split halved the block size, so block counts are not comparable across it --
# the same mistake that made the group count read 32 in export-gpoint-groups.
# ngroups = 1 is the ungrouped specialization, a different kernel at a
# different register count, so the model is anchored at the smallest grouped
# count rather than spanning that boundary.
waves = nsys["waves"]
sms = waves["baseline"]["blocks_resident"] // waves["baseline"]["blocks_resident_per_sm"]
block_threads = split["sweeps_block_threads"]
warps_per_block = block_threads // 32
grouped = [n for n in counts if n > 1]
anchor = min(grouped)


# Residency comes from the split measurement, where every kernel's occupancy was
# checked against the driver rather than only against the model.
def resident_blocks(kernel):
    warps = split["kernels"][kernel][f"driver_warps_per_sm_b{block_threads}"]
    assert warps % warps_per_block == 0, "warps per SM is not a whole number of blocks"
    return sms * (warps // warps_per_block)


def model(ngpt, resident):
    def cost(n):
        blocks = math.ceil(sweep["ncol"] * n / block_threads)
        return math.ceil(ngpt / n) * math.ceil(blocks / resident)

    return cost


def fit(ngpt, resident, points):
    cost = model(ngpt, resident)
    out = {}
    for n in grouped:
        blocks = math.ceil(sweep["ncol"] * n / block_threads)
        pred = 100.0 * (cost(n) / cost(anchor) - 1.0)
        meas = 100.0 * (points[str(n)]["solve_ms"] / points[str(anchor)]["solve_ms"] - 1.0)
        out[str(n)] = {
            "ngroups": n,
            "rounds": math.ceil(ngpt / n),
            "blocks_per_round": blocks,
            "waves_per_round": math.ceil(blocks / resident),
            "predicted_change_pct": round(pred, 2),
            "measured_change_pct": round(meas, 2),
            "residual_pts": round(meas - pred, 2),
        }
    return out


resident_lw = resident_blocks("lw_sweeps")
resident_sw = resident_blocks("sw_sweeps")
lw_fit = fit(sweep["n_gpt_lw"], resident_lw, pts)
sw_fit = fit(sw["n_gpt_sw"], resident_sw, sw["points"])
worst_lw = max(abs(p["residual_pts"]) for p in lw_fit.values())
worst_sw = max(abs(p["residual_pts"]) for p in sw_fit.values())

# What one more reservation would test. The model is flat past the best count
# in both bands, so this is the count at which it says the quantization half of
# the curve is finished, not a count worth running.
nxt = 2 * max(counts)
cost_lw = model(sweep["n_gpt_lw"], resident_lw)
next_point = {
    "ngroups": nxt,
    "rounds": math.ceil(sweep["n_gpt_lw"] / nxt),
    "blocks_per_round": math.ceil(sweep["ncol"] * nxt / block_threads),
    "predicted_change_pct": round(100.0 * (cost_lw(nxt) / cost_lw(anchor) - 1.0), 2),
    "predicted_gain_pts": round(
        100.0 * (cost_lw(nxt) - cost_lw(max(counts))) / cost_lw(anchor), 2
    ),
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
    # Whether the prediction and the measurement describe the same code, tested
    # on the geometry rather than on the size of the disagreement. The snapshot
    # recorded the grid it predicted for; the split halved the block size, so a
    # different block count at the same group count means a different arm and
    # the score is not a score.
    comparable = pred["blocks"] == lw_fit[str(pred["ngroups"])]["blocks_per_round"]
    return {
        "arms_comparable": comparable,
        "predicted_blocks": pred["blocks"],
        "current_blocks": lw_fit[str(pred["ngroups"])]["blocks_per_round"],
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
        # reproducibility across states -- but only while the two runs measured
        # the same kernel. Once they do not, the comparison is not a score at
        # any precision, so the flag is withheld rather than reported true.
        "error_within_state_shift": (abs(err) <= abs(shift)) if comparable else None,
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
# A band is however many kernels its solve launches. It was one named
# `rte_*_2stream_solve_grouped`; the optics split made it an optics kernel and a
# sweeps kernel, so a prefix pinned to the old name finds nothing and this
# analysis asserts the profile is missing a band it is actually running.
BAND_PREFIXES = {
    "lw": ("rte_lw_2stream_solve_grouped", "_lw_group_optics", "_lw_group_sweeps"),
    "sw": ("rte_sw_2stream_solve_grouped", "_sw_group_optics", "_sw_group_sweeps"),
}
# How many solves the window runs, counted on the UNTREATED arm, whose kernel
# is launched once per solve and which no change here can move. Counting on the
# mod arm would need a kernel the split launches once per solve rather than once
# per round -- the reduction is one, but at a couple of milliseconds it does not
# reach a top-twenty table, so the count would silently go missing.
NSYS_BANDS = {"lw": "longwave", "sw": "shortwave"}


def decision():
    rows = {b: [] for b in BAND_PREFIXES}
    for r in csv.DictReader(open("results/top-kernels.csv")):
        for band, prefixes in BAND_PREFIXES.items():
            if any(r["kernel"].startswith(p) for p in prefixes):
                rows[band].append(r)
    assert all(rows.values()), f"profile has {sorted(b for b, v in rows.items() if v)}, not both grouped solves"
    solves = {b: float(nsys[n]["baseline"]["launches"]) for b, n in NSYS_BANDS.items()}
    # The reduction runs once per grouped solve too, so the two counts agreeing
    # says the treated arm ran the same number of solves as the arm counted.
    assert solves["lw"] == solves["sw"] == float(nsys["overhead"]["reduction"]["launches"]), (
        f"solve counts disagree: {solves} against the reduction"
    )

    # The budget, derived from each band's time and its own share of it. The two
    # bands agreeing is the check that they came from one profile.
    budgets = {
        b: 100.0 * sum(float(r["total_ms_mod"]) for r in rs)
        / sum(float(r["pct_of_gpu_time_mod"]) for r in rs)
        for b, rs in rows.items()
    }
    device_ms = sum(budgets.values()) / len(budgets)
    assert abs(budgets["lw"] - budgets["sw"]) / device_ms < 0.01, "bands disagree on the device total"

    bands = {"lw": pts, "sw": sw["points"]}
    launches = solves
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

    # The split made the downward direction the interesting one, so the count
    # below the default is priced on the same terms: a negative saving is what
    # lowering costs, and a negative extra_gib is what it returns.
    below = max(n for n in counts if 1 < n < default)
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
        # solve, at different states, so the gap between them is recorded. A
        # band's per-solve cost is summed over every kernel its solve launches
        # and divided by the solves, not read off one kernel's mean, because
        # since the split a kernel's mean is a round rather than a solve.
        "profile_vs_sweep_pct": {
            b: round(
                100.0
                * (
                    sum(float(r["total_ms_mod"]) for r in rows[b]) / solves[b]
                    - bands[b][str(default)]["solve_ms"]
                )
                / bands[b][str(default)]["solve_ms"],
                1,
            )
            for b in bands
        },
        "points": {str(n): at(n) for n in above},
        "below": at(below),
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
        "block_threads": block_threads,
        "anchor_ngroups": anchor,
        "resident_blocks": resident_lw,
        "resident_blocks_sw": resident_sw,
        "points": lw_fit,
        "shortwave_points": sw_fit,
        "worst_residual_pts": round(worst_lw, 2),
        "worst_residual_pts_sw": round(worst_sw, 2),
        # The band the model covers and the band it does not, as a flag rather
        # than as a sentence, so the prose cannot keep a claim the fit lost.
        "fits_longwave": worst_lw < 2.0,
        "fits_shortwave": worst_sw < 2.0,
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
| 1 | <!-- calkit value key=points.1.gpt_per_group path=results/docs/gpoint-group-count.json -->256<!-- /calkit value --> | <!-- calkit value key=points.1.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->670.6<!-- /calkit value --> | -- | 0 | -- |
| 4 | <!-- calkit value key=points.4.gpt_per_group path=results/docs/gpoint-group-count.json -->64<!-- /calkit value --> | <!-- calkit value key=points.4.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->549.2<!-- /calkit value --> | <!-- calkit value key=points.4.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-18.11<!-- /calkit value -->% | <!-- calkit value key=points.4.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.40<!-- /calkit value --> | <!-- calkit value key=points.4.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->45.22<!-- /calkit value -->%/GiB |
| 8 | <!-- calkit value key=points.8.gpt_per_group path=results/docs/gpoint-group-count.json -->32<!-- /calkit value --> | <!-- calkit value key=points.8.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->459.2<!-- /calkit value --> | <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-31.53<!-- /calkit value -->% | <!-- calkit value key=points.8.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.79<!-- /calkit value --> | <!-- calkit value key=points.8.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->39.86<!-- /calkit value -->%/GiB |
| 16 | <!-- calkit value key=points.16.gpt_per_group path=results/docs/gpoint-group-count.json -->16<!-- /calkit value --> | <!-- calkit value key=points.16.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->454.6<!-- /calkit value --> | <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-32.21<!-- /calkit value -->% | <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.57<!-- /calkit value --> | <!-- calkit value key=points.16.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->20.46<!-- /calkit value -->%/GiB |
| 32 | <!-- calkit value key=points.32.gpt_per_group path=results/docs/gpoint-group-count.json -->8<!-- /calkit value --> | <!-- calkit value key=points.32.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->441.7<!-- /calkit value --> | <!-- calkit value key=points.32.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-34.12<!-- /calkit value -->% | <!-- calkit value key=points.32.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.15<!-- /calkit value --> | <!-- calkit value key=points.32.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->10.85<!-- /calkit value -->%/GiB |

## The split put a knee in the curve, and the sweep found it

The smallest group count within one repeat spread of the best saving is <!-- calkit value key=sweep.sufficient_ngroups path=results/docs/gpoint-group-count.json -->32<!-- /calkit value -->, against a default of <!-- calkit value key=sweep.default_ngpt_groups path=results/docs/gpoint-group-count.json -->16<!-- /calkit value -->, so `default_is_sufficient_only` is <!-- calkit value key=sweep.default_is_sufficient_only path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->.
For the first time `sufficient_is_sweep_maximum` is <!-- calkit value key=sweep.sufficient_is_sweep_maximum path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->: 64 groups measure <!-- calkit value key=points.64.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-33.98<!-- /calkit value -->% against 32 groups' <!-- calkit value key=points.32.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-34.12<!-- /calkit value -->%, so the curve turns over inside the sweep instead of still falling at its endpoint.
The threshold is measured rather than assumed -- a point counts as sufficient if it is within <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.327<!-- /calkit value -->% of the best, which puts the bar at <!-- calkit value key=sweep.threshold_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-33.80<!-- /calkit value -->% -- and 32 clears it while 64 misses by less than a repeat spread.

What matters is where the knee sits relative to the default, and it sits barely above it.
Eight groups reach <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-31.53<!-- /calkit value -->% where the best point in the sweep reaches <!-- calkit value key=sweep.best_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-34.12<!-- /calkit value -->%, at <!-- calkit value key=points.8.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.79<!-- /calkit value --> GiB against the default's <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.57<!-- /calkit value -->.
The per-GiB column falls monotonically across the whole sweep, from <!-- calkit value key=points.4.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->45.22<!-- /calkit value -->%/GiB at 4 groups to <!-- calkit value key=points.8.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->39.86<!-- /calkit value --> at 8, <!-- calkit value key=points.16.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->20.46<!-- /calkit value --> at 16, <!-- calkit value key=points.32.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->10.85<!-- /calkit value --> at 32 and <!-- calkit value key=points.64.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->5.41<!-- /calkit value --> at 64.
So the group count remains a price to pay or not, and it is priced below against the device budget rather than against the sweep's own best.

## Almost all of the saving arrives by eight groups

| step | change in saving |
|---|---|
| 1 to 4 | <!-- calkit value key=increments.1_to_4 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-18.11<!-- /calkit value --> points |
| 4 to 8 | <!-- calkit value key=increments.4_to_8 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.42<!-- /calkit value --> points |
| 8 to 16 | <!-- calkit value key=increments.8_to_16 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.68<!-- /calkit value --> points |
| 16 to 32 | <!-- calkit value key=increments.16_to_32 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.92<!-- /calkit value --> points |
| 32 to 64 | <!-- calkit value key=increments.32_to_64 format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.14<!-- /calkit value --> points |

The two steps at or below eight groups carry <!-- calkit value key=increments.1_to_4 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-18.11<!-- /calkit value --> and <!-- calkit value key=increments.4_to_8 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.42<!-- /calkit value --> points; the three above it carry <!-- calkit value key=increments.8_to_16 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.68<!-- /calkit value -->, <!-- calkit value key=increments.16_to_32 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.92<!-- /calkit value --> and <!-- calkit value key=increments.32_to_64 format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.14<!-- /calkit value -->.
The last of those is positive, which is the turn the earlier sweeps never reached.

This is a different curve from the one the earlier sweeps measured, and the difference is not the grouping.
The committed snapshot of the sweep taken before the optics split has 16 groups at <!-- calkit value key=prediction.prior_16_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-8.95<!-- /calkit value -->%, where the same count now measures <!-- calkit value key=prediction.current_16_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-32.21<!-- /calkit value -->%.
Splitting the optics out of the sweeps is what moved it: the saving the group count has left to find is whatever the split did not already take.

## Wave quantization accounts for the longwave curve, and not the shortwave's

The split changed both halves of the old model's premise, so the model is rebuilt rather than refitted.
A grouped solve launches <!-- calkit value key=quantization.points.16.rounds path=results/docs/gpoint-group-count.json -->16<!-- /calkit value --> rounds at the default, each a grid of `ncol * ngroups` threads in blocks of <!-- calkit value key=quantization.block_threads path=results/docs/gpoint-group-count.json -->128<!-- /calkit value -->, so a round costs its block count rounded up to a whole wave and the solve costs that times its rounds.
Counting is in threads rather than blocks because the split halved the block size, and a block count is not comparable across that boundary -- the same mistake that read the group count as 32 in `export-gpoint-groups.py` before it was fixed.
Residency comes from the split measurement, where every kernel's occupancy was checked against the driver and not only against the model: <!-- calkit value key=quantization.resident_blocks path=results/docs/gpoint-group-count.json -->324<!-- /calkit value --> of the longwave's blocks are resident at once on <!-- calkit value key=quantization.sms path=results/docs/gpoint-group-count.json -->108<!-- /calkit value --> SMs, and <!-- calkit value key=quantization.resident_blocks_sw path=results/docs/gpoint-group-count.json -->432<!-- /calkit value --> of the shortwave's.
One group is the ungrouped specialization, a different kernel at a different register count, so the model is anchored at <!-- calkit value key=quantization.anchor_ngroups path=results/docs/gpoint-group-count.json -->4<!-- /calkit value --> groups rather than spanning that boundary, and every figure in the table below is relative to that anchor rather than to the ungrouped point.

| groups | rounds | blocks/round | waves/round | predicted | measured | residual |
|---|---|---|---|---|---|---|
| 4 | <!-- calkit value key=quantization.points.4.rounds path=results/docs/gpoint-group-count.json -->64<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->768<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.waves_per_round path=results/docs/gpoint-group-count.json -->3<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=quantization.points.4.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=quantization.points.4.residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value --> |
| 8 | <!-- calkit value key=quantization.points.8.rounds path=results/docs/gpoint-group-count.json -->32<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->1,536<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.waves_per_round path=results/docs/gpoint-group-count.json -->5<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-16.67<!-- /calkit value -->% | <!-- calkit value key=quantization.points.8.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-16.39<!-- /calkit value -->% | <!-- calkit value key=quantization.points.8.residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.28<!-- /calkit value --> |
| 16 | <!-- calkit value key=quantization.points.16.rounds path=results/docs/gpoint-group-count.json -->16<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->3,072<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.waves_per_round path=results/docs/gpoint-group-count.json -->10<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-16.67<!-- /calkit value -->% | <!-- calkit value key=quantization.points.16.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-17.22<!-- /calkit value -->% | <!-- calkit value key=quantization.points.16.residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-0.55<!-- /calkit value --> |
| 32 | <!-- calkit value key=quantization.points.32.rounds path=results/docs/gpoint-group-count.json -->8<!-- /calkit value --> | <!-- calkit value key=quantization.points.32.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->6,144<!-- /calkit value --> | <!-- calkit value key=quantization.points.32.waves_per_round path=results/docs/gpoint-group-count.json -->19<!-- /calkit value --> | <!-- calkit value key=quantization.points.32.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-20.83<!-- /calkit value -->% | <!-- calkit value key=quantization.points.32.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-19.56<!-- /calkit value -->% | <!-- calkit value key=quantization.points.32.residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.27<!-- /calkit value --> |
| 64 | <!-- calkit value key=quantization.points.64.rounds path=results/docs/gpoint-group-count.json -->4<!-- /calkit value --> | <!-- calkit value key=quantization.points.64.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->12,288<!-- /calkit value --> | <!-- calkit value key=quantization.points.64.waves_per_round path=results/docs/gpoint-group-count.json -->38<!-- /calkit value --> | <!-- calkit value key=quantization.points.64.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-20.83<!-- /calkit value -->% | <!-- calkit value key=quantization.points.64.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-19.39<!-- /calkit value -->% | <!-- calkit value key=quantization.points.64.residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.45<!-- /calkit value --> |

The worst residual across the five grouped points is <!-- calkit value key=quantization.worst_residual_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.45<!-- /calkit value --> points, so `fits_longwave` is <!-- calkit value key=quantization.fits_longwave path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->.
The model reproduces the plateaus and not merely the trend: 8 and 16 groups cost the same number of wave-rounds and measure <!-- calkit value key=quantization.points.8.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-16.39<!-- /calkit value -->% and <!-- calkit value key=quantization.points.16.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-17.22<!-- /calkit value -->%, and 32 and 64 cost the same and measure <!-- calkit value key=quantization.points.32.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-19.56<!-- /calkit value -->% and <!-- calkit value key=quantization.points.64.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-19.39<!-- /calkit value -->%.
That is what the earlier model could not do: it left a residual of a few points at every count, flat in the count, which this one does not have.

The same model on the shortwave fails, and not by a margin.
Its worst residual is <!-- calkit value key=quantization.worst_residual_pts_sw format="{:.2f}" path=results/docs/gpoint-group-count.json -->17.03<!-- /calkit value --> points and `fits_shortwave` is <!-- calkit value key=quantization.fits_shortwave path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->.
It also gets the last point the wrong way round: 64 groups cost more wave-rounds there than 32, so it predicts <!-- calkit value key=quantization.shortwave_points.64.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.57<!-- /calkit value -->% against 32's <!-- calkit value key=quantization.shortwave_points.32.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-6.25<!-- /calkit value -->%, while the measurement has 64 the faster of the two at <!-- calkit value key=quantization.shortwave_points.64.measured_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-13.45<!-- /calkit value -->%.

The band the model fails on is the band whose cost depends on how many columns are lit, and this document's own conversion says so independently.
Comparing each band's per-solve cost in the profile against the sweep's -- the same code at two atmospheric states -- the longwave differs by <!-- calkit value key=decision.profile_vs_sweep_pct.lw format="{:.1f}" path=results/docs/gpoint-group-count.json -->-1.0<!-- /calkit value -->% and the shortwave by <!-- calkit value key=decision.profile_vs_sweep_pct.sw format="{:.1f}" path=results/docs/gpoint-group-count.json -->-27.1<!-- /calkit value -->%.
A solve whose cost moves by a quarter between states is not one whose threads carry equal work, and packing uneven work into blocks keeps improving with more groups long after the wave count has stopped changing.
That is a reading of two numbers rather than a test, and this sweep does not isolate it.

## The prediction this document made for 32 can no longer be scored

The earlier version of this document predicted 32 groups from the quantization model plus the residual at 16, before 32 was measured.
That prediction is read here out of a committed snapshot of the earlier run rather than retyped, at `<!-- calkit value key=prediction.source_commit path=results/docs/gpoint-group-count.json -->8d91bb5<!-- /calkit value -->`.

It cannot be scored against this sweep, and the test for that is mechanical rather than a judgment about how far apart the numbers look.
The snapshot recorded the grid it was predicting for: <!-- calkit value key=prediction.predicted_blocks format="{:,}" path=results/docs/gpoint-group-count.json -->3,072<!-- /calkit value --> blocks at 32 groups, where the same count now launches <!-- calkit value key=prediction.current_blocks format="{:,}" path=results/docs/gpoint-group-count.json -->6,144<!-- /calkit value --> blocks per round because the split halved the block size.
So `arms_comparable` is <!-- calkit value key=prediction.arms_comparable path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->, and `error_within_state_shift` is withheld rather than reported, because a prediction made for one kernel and measured on another is not confirmed at any precision.

This is the failure the repository's own rule names: when a value marker's sign or boolean flips, re-read the sentences around it.
Here the boolean did not flip.
The prediction error grew from under a point to <!-- calkit value key=prediction.error_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-22.04<!-- /calkit value --> points and the state shift it is compared against grew to <!-- calkit value key=prediction.state_shift_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-23.26<!-- /calkit value -->, both for the same reason, so the flag would have gone on reading true while meaning nothing -- a confirmation manufactured by the thing it was supposed to be independent of.

What survives is the memory arithmetic, which is shape arithmetic and does not depend on the kernel: <!-- calkit value key=prediction.predicted_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.63<!-- /calkit value --> GiB predicted against <!-- calkit value key=prediction.measured_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.15<!-- /calkit value --> measured.

What also survives is the bound on precision, which still holds and still matters.
The repeat spread of <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.327<!-- /calkit value -->% is the scatter of seven timings of one state, and the same quantity moves by about a point between two states of the same code.
A figure here is reproducible to a fraction of a percent at a fixed state and to about a point across states, and to nothing at all across a change of kernel.

## What it is worth in SYPD, measured

The same job timed <!-- calkit value key=wallclock.blocks path=results/docs/gpoint-group-count.json -->12<!-- /calkit value --> blocks of <!-- calkit value key=wallclock.block_steps path=results/docs/gpoint-group-count.json -->180<!-- /calkit value --> real coupler steps -- <!-- calkit value key=wallclock.timed_steps format="{:,}" path=results/docs/gpoint-group-count.json -->2,160<!-- /calkit value --> steps in all, with the live grouped solver, the real diagnostics and the real coupling -- using the wall clock the coupler's own SYPD is defined on.
Blocks are <!-- calkit value key=wallclock.block_steps path=results/docs/gpoint-group-count.json -->180<!-- /calkit value --> steps because that is a whole number of <!-- calkit value key=wallclock.rad_period_steps path=results/docs/gpoint-group-count.json -->30<!-- /calkit value -->-step radiation periods, so every block carries identical radiation work.

The one arithmetic step is the comparison arm: the grouped SYPD is measured, and the ungrouped SYPD is that measured block with the measured in-block saving added back, rather than a second run differenced against it.
That is justified because the solve is synchronous on the step's critical path, and the alternative cannot resolve a sub-percent effect on this machine.
It is stated here rather than left implicit, and it is the only place in this document where a number is not measured directly.

SYPD has three defensible denominators and they differ by more than the effect, so all three are derived:

| basis | step (ms) | gain at 16 groups |
|---|---|---|
| ordinary steps, between writes | <!-- calkit value key=wallclock.ordinary_step_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->282.2<!-- /calkit value --> | <!-- calkit value key=wallclock.arms.16.gain_ordinary_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.551<!-- /calkit value -->% |
| steady state, writes at their cadence | <!-- calkit value key=wallclock.steady_step_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->301.4<!-- /calkit value --> | <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.389<!-- /calkit value -->% |
| the timed window exactly as it ran | -- | <!-- calkit value key=wallclock.arms.16.gain_window_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.273<!-- /calkit value -->% |

The steady-state row is the one to quote: the daily diagnostics write lands in <!-- calkit value key=wallclock.write_blocks path=results/docs/gpoint-group-count.json -->3<!-- /calkit value --> of the <!-- calkit value key=wallclock.blocks path=results/docs/gpoint-group-count.json -->12<!-- /calkit value --> blocks and costs <!-- calkit value key=wallclock.write_overhead_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->27.2<!-- /calkit value -->% on the block it falls in, so a basis that excludes it is too fast and the raw window is too slow.
The raw window is too slow because the first write also compiles its path: that block took <!-- calkit value key=wallclock.first_write_block_s format="{:.1f}" path=results/docs/gpoint-group-count.json -->97.8<!-- /calkit value --> s against <!-- calkit value key=wallclock.steady_write_block_s format="{:.1f}" path=results/docs/gpoint-group-count.json -->64.6<!-- /calkit value --> s for the later ones, a one-off a long run amortizes away.

On the steady-state basis the split takes SYPD from <!-- calkit value key=wallclock.arms.16.sypd_steady_ungrouped format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0648<!-- /calkit value --> to <!-- calkit value key=wallclock.arms.16.sypd_steady format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0902<!-- /calkit value --> at the default 16 groups, a gain of <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.389<!-- /calkit value -->%.
At 32 groups it would be <!-- calkit value key=wallclock.arms.32.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.531<!-- /calkit value -->%, from a solve saving of <!-- calkit value key=wallclock.arms.32.saved_per_event_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->228.8<!-- /calkit value --> ms per radiation event against <!-- calkit value key=wallclock.arms.16.saved_per_event_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->216.0<!-- /calkit value --> ms at 16.

The uncertainty on that gain is <!-- calkit value key=wallclock.arms.16.gain_steady_uncertainty_pts format="{:.4f}" path=results/docs/gpoint-group-count.json -->0.0090<!-- /calkit value --> percentage points, which is smaller than it looks like it should be, and the reason is worth stating.
The gain is a ratio of a measured saving to a measured wall clock, not a difference of two runs, so the denominator's relative error scales the gain instead of adding to it: <!-- calkit value key=wallclock.ordinary_rel_se_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.378<!-- /calkit value -->% of <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.389<!-- /calkit value -->% is a few thousandths of a point.
The `gain_below_block_iqr` flag the measurement script writes is therefore true and should not be read as "unresolved" -- it compares a gain against a block-to-block spread as though the gain had been obtained by differencing blocks, which is exactly what this method avoids.

These SYPD figures are steady-state rates measured mid-run and are **not** comparable to the pipeline's end-to-end SYPD, which includes first-call compilation over the whole job.
For orientation, the same window read as a single measurement gives <!-- calkit value key=wallclock.window_sypd format="{:.4f}" path=results/docs/gpoint-group-count.json -->1.0373<!-- /calkit value -->, and it carries that one-off compile.

## What it costs

Sixteen groups take <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.57<!-- /calkit value --> GiB of wide longwave workspace, thirty-two take <!-- calkit value key=points.32.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.15<!-- /calkit value --> and sixty-four take <!-- calkit value key=points.64.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->6.28<!-- /calkit value -->, measured as a free-memory delta across the workspace construction rather than computed from shapes.
That is the whole cost of the treatment and it scales linearly in the group count, so this is the constraint that decides how far the count can go.
The model says <!-- calkit value key=quantization.next.ngroups path=results/docs/gpoint-group-count.json -->128<!-- /calkit value --> groups would run the same <!-- calkit value key=quantization.next.rounds path=results/docs/gpoint-group-count.json -->2<!-- /calkit value --> rounds over <!-- calkit value key=quantization.next.blocks_per_round format="{:,}" path=results/docs/gpoint-group-count.json -->24,576<!-- /calkit value --> blocks each and land at <!-- calkit value key=quantization.next.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-20.83<!-- /calkit value -->%, which is <!-- calkit value key=quantization.next.predicted_gain_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value --> points of quantization gain for <!-- calkit value key=quantization.next.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->12.56<!-- /calkit value --> GiB: past 32 the rounding stops changing at all, so the model says its half of the longwave curve is finished.

Before 64 was measured, this document predicted at `9e2bfcc` that "the curve and the memory are now pulling in the same direction and 64 is unlikely to be worth it".
The split changed the kernel underneath that prediction, so it is not scored here either -- but 64 did measure worse than 32 in the longwave, and the section below prices both doublings in device time rather than in points of one solve.

## What a default change is worth against the device budget

Points of a solve are not the unit a default change is decided in, and one default sets the group count for both the longwave and the shortwave, so both are priced here.
The sweeps time one solve in isolation and the profile says how many of them the window runs and how large the budget is, so the conversion is absolute milliseconds times launch count rather than a share of a share.
It rests on the sweep and the profile describing the same solve, which is checked rather than assumed: the profiled mod arm's per-solve cost differs from the sweep's by <!-- calkit value key=decision.profile_vs_sweep_pct.lw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-1.0<!-- /calkit value -->% in the longwave and <!-- calkit value key=decision.profile_vs_sweep_pct.sw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-27.1<!-- /calkit value -->% in the shortwave, at a different atmospheric state.
Those two disagreements are not the same size, and the shortwave's is large enough that its conversion below should be read as an order of magnitude rather than a figure.

| groups | longwave | shortwave | over the window | of device | extra workspace |
|---|---|---|---|---|---|
| 32 | <!-- calkit value key=decision.points.32.lw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->12.9<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.sw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->12.9<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.window_saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->103.0<!-- /calkit value --> ms | <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.318<!-- /calkit value -->% | <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.38<!-- /calkit value --> GiB |
| 64 | <!-- calkit value key=decision.points.64.lw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->11.9<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.sw_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->15.9<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.window_saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->111.2<!-- /calkit value --> ms | <!-- calkit value key=decision.points.64.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.344<!-- /calkit value -->% | <!-- calkit value key=decision.points.64.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->10.12<!-- /calkit value --> GiB |

Against a device budget of <!-- calkit value key=decision.device_ms format="{:,.0f}" path=results/docs/gpoint-group-count.json -->32,356<!-- /calkit value --> ms, thirty-two groups are worth <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.318<!-- /calkit value -->% for <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.38<!-- /calkit value --> GiB on top of the default's <!-- calkit value key=decision.default_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.39<!-- /calkit value --> GiB -- the same figure twice, because the workspace is linear in the count and doubling it doubles the workspace.
Sixty-four are worth <!-- calkit value key=decision.points.64.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.344<!-- /calkit value -->%, so the second doubling adds <!-- calkit value key=decision.marginal_pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.026<!-- /calkit value --> points of device time for <!-- calkit value key=decision.marginal_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->6.74<!-- /calkit value --> GiB -- <!-- calkit value key=decision.per_gib_ratio format="{:.1f}" path=results/docs/gpoint-group-count.json -->24.4<!-- /calkit value --> times worse per GiB than the first.
Lowering the default is now the direction worth pricing too, and it is the better trade: <!-- calkit value key=decision.below.ngroups path=results/docs/gpoint-group-count.json -->8<!-- /calkit value --> groups give back <!-- calkit value key=decision.below.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.68<!-- /calkit value --> GiB for <!-- calkit value key=decision.below.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->-0.161<!-- /calkit value -->% of device time, which is more device time per GiB than the doubling to 32 buys.
Across the whole swept range the default sits in the middle of a plateau rather than at a decision point.
Both figures are small enough that they cannot be confirmed by running the model: `best_below_null_floor` is <!-- calkit value key=decision.best_below_null_floor path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->, because even 64 groups come to a fraction of the <!-- calkit value key=decision.null_floor_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->1.869<!-- /calkit value -->% that two identical arms of this stack reproduce each other to.
They also inherit this experiment's state dependence, which the conversion's own check measures: the longwave solve differs by <!-- calkit value key=decision.profile_vs_sweep_pct.lw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-1.0<!-- /calkit value -->% between the sweep's state and the profile's and the shortwave by <!-- calkit value key=decision.profile_vs_sweep_pct.sw format="{:+.1f}" path=results/docs/gpoint-group-count.json -->-27.1<!-- /calkit value -->%, so read the longwave's share to roughly a tenth of itself and the shortwave's as an order of magnitude.

## Correctness

The split is not bit-for-bit and cannot be, because a g-point sum taken in N partials is a different summation order.
The worst flux deviation from the ungrouped path over the <!-- calkit value key=correctness.grouped_counts path=results/docs/gpoint-group-count.json -->5<!-- /calkit value --> grouped counts is <!-- calkit value key=correctness.max_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->8.85e-04<!-- /calkit value --> W/m2, against fluxes of order a few hundred W/m2, and it does not grow with the group count.
The five points span only <!-- calkit value key=correctness.spread_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->20.8<!-- /calkit value -->% of their own smallest value, <!-- calkit value key=correctness.min_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->7.32e-04<!-- /calkit value --> W/m2, and `rises_with_count` is <!-- calkit value key=correctness.rises_with_count path=results/docs/gpoint-group-count.json -->false<!-- /calkit value -->: the worst value does sit at 64 groups, but 4 groups ties it exactly and 8 and 16 are the lowest on the curve.
A reordering error that accumulated with the number of partials would rise across the sweep, and this one is flat, which is the expected behavior for <!-- calkit value key=sweep.float_type path=results/docs/gpoint-group-count.json -->Float32<!-- /calkit value --> rounding rather than for a bug in the reduction.

## Two independent cross-checks against the nsys pass

The sweep and the profiling pass measure the same solve by different means, on different stacks and at different points in the simulation, so agreement is a check on both.
The nsys baseline arm spent <!-- calkit value key=nsys.baseline_per_launch_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->667.2<!-- /calkit value --> ms per launch in the ungrouped longwave kernel, against the sweep's <!-- calkit value key=nsys.sweep_ungrouped_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->670.6<!-- /calkit value --> ms for its ungrouped point, a gap of <!-- calkit value key=nsys.per_launch_gap_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->-0.5<!-- /calkit value -->%.
That is closer than the previous sweep managed, and for a reason that is not to anyone's credit: this sweep ran after <!-- calkit value key=wallclock.timed_steps format="{:,}" path=results/docs/gpoint-group-count.json -->2,160<!-- /calkit value --> steps rather than after <!-- calkit value key=sweep.warmup_steps path=results/docs/gpoint-group-count.json -->3<!-- /calkit value -->, so its state is nearer the profiled window's than the earlier one's was.
The saving agrees too: the pass measured <!-- calkit value key=nsys.window_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-32.51<!-- /calkit value -->% at 16 groups over its window, against the sweep's in-block <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-32.21<!-- /calkit value -->% at the same group count.

## Verdict

The split is worth <!-- calkit value key=wallclock.arms.16.gain_steady_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->2.389<!-- /calkit value -->% of SYPD at the shipped default, measured rather than converted, and that remains the result worth quoting.
Raising the default is worth having and worth very little: thirty-two groups is the setting if the memory is spare, at <!-- calkit value key=decision.points.32.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.318<!-- /calkit value -->% of device time for <!-- calkit value key=decision.points.32.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.38<!-- /calkit value --> GiB, and the last doubling to 64 costs <!-- calkit value key=decision.marginal_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->6.74<!-- /calkit value --> GiB more for <!-- calkit value key=decision.marginal_pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.026<!-- /calkit value --> points, <!-- calkit value key=decision.per_gib_ratio format="{:.1f}" path=results/docs/gpoint-group-count.json -->24.4<!-- /calkit value --> times worse per GiB.
Since neither figure can be resolved by running the model -- and lowering to <!-- calkit value key=decision.below.ngroups path=results/docs/gpoint-group-count.json -->8<!-- /calkit value --> groups costs only <!-- calkit value key=decision.below.pct_of_device format="{:.3f}" path=results/docs/gpoint-group-count.json -->-0.161<!-- /calkit value -->% while returning <!-- calkit value key=decision.below.extra_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.68<!-- /calkit value --> GiB -- the honest recommendation is to leave the default alone.
The optics split took the saving the group count used to be the only way to reach, and what the count still moves is smaller than this stack can measure end to end.
The edit is no longer a constant, either: on `pb/gpoint-groups` the 16 this sweep measured has been replaced by a grid-sized default, `clamp(cld(TARGET_GPT_GROUP_COLUMNS, ncol), 1, MAX_NGPT_GROUPS)`, which at this grid's <!-- calkit value key=sweep.ncol format="{:,}" path=results/docs/gpoint-group-count.json -->24,576<!-- /calkit value --> columns evaluates to the same 16, so raising it means doubling both of those and not editing one number.
What this document still does not establish is why the shortwave keeps gaining past the point where its wave count stops changing, or any figure reproducible across atmospheric states to better than about a point -- and in the shortwave, not to better than a quarter of itself.

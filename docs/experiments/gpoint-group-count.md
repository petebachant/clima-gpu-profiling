# Longwave g-point group count

The longwave g-point split widens every workspace the solve indexes by column from `ncol` to `ncol * ngpt_groups`, and `RRTMGP.RTE.DEFAULT_NGPT_GROUPS` was set to 16 because 16 was the first group count that worked.
This experiment sweeps the group count to find where the saving stops, so the default rests on a measurement instead of on the order things were tried.
It answers the question in the opposite direction from the one expected: the saving does not saturate below 16, and the curve is still falling steeply at the largest count measured.

## Why this is an in-block comparison and not a difference of runs

`ngpt_groups` turned out to be a constructor argument rather than a compile-time constant, which is what makes this measurable at all on this machine.
All points are taken in one process against one settled AMIP state, rebuilding only the longwave RTE's group workspaces between them.
Everything the solve reads -- atmospheric state, lookups, boundary conditions, and the narrow optics, sources and flux buffers the ungrouped path uses -- is the live simulation's own, so two points differ by the group count and by nothing else.
This matters because the alternative, running the model once per group count and differencing, cannot resolve the effect: one arm of this stack wandered 7.73% between two runs of identical code, which is most of the way to the whole quantity being measured.
The in-block design replaces that noise with a repeat spread of <!-- calkit value key=sweep.repeat_spread_pct format="{:.3f}" path=results/docs/gpoint-group-count.json -->0.183<!-- /calkit value -->%, so the best saving is resolved <!-- calkit value key=sweep.resolution_vs_spread format="{:.0f}" path=results/docs/gpoint-group-count.json -->49<!-- /calkit value --> times over the sweep's own scatter.

`ngroups = 1` passes `nothing` and takes the ungrouped specialization, so the baseline is a point in the same sweep rather than a separate measurement.
The timing covers the whole both-skies solve, which charges the grouped path for the reduction and the aerosol mask it adds, so these are net figures and not the kernel's alone.

```python calkit stage name=values environment=py outputs=[{path: results/docs/gpoint-group-count.json, storage: git}] inputs=[results/gpoint-sweep.toml, results/gpoint-groups.json]
import json
import math
import tomllib

sweep = tomllib.load(open("results/gpoint-sweep.toml", "rb"))
nsys = json.load(open("results/gpoint-groups.json"))

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

The sweep ran <!-- calkit value key=sweep.repeats path=results/docs/gpoint-group-count.json -->7<!-- /calkit value --> timed repeats per point at group counts <!-- calkit value key=sweep.group_counts path=results/docs/gpoint-group-count.json -->1, 4, 8, 16<!-- /calkit value -->, on <!-- calkit value key=sweep.ncol format="{:,}" path=results/docs/gpoint-group-count.json -->24,576<!-- /calkit value --> columns and <!-- calkit value key=sweep.nlay path=results/docs/gpoint-group-count.json -->80<!-- /calkit value --> layers at <!-- calkit value key=sweep.n_gpt_lw path=results/docs/gpoint-group-count.json -->256<!-- /calkit value --> longwave g-points.

| groups | g-points each | solve (ms) | change | saved (ms) | workspace (GiB) | per GiB |
|---|---|---|---|---|---|---|
| 1 | <!-- calkit value key=points.1.gpt_per_group path=results/docs/gpoint-group-count.json -->256<!-- /calkit value --> | <!-- calkit value key=points.1.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->639.1<!-- /calkit value --> | -- | -- | 0 | -- |
| 4 | <!-- calkit value key=points.4.gpt_per_group path=results/docs/gpoint-group-count.json -->64<!-- /calkit value --> | <!-- calkit value key=points.4.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->624.4<!-- /calkit value --> | <!-- calkit value key=points.4.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.29<!-- /calkit value -->% | <!-- calkit value key=points.4.saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->14.6<!-- /calkit value --> | <!-- calkit value key=points.4.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.46<!-- /calkit value --> | <!-- calkit value key=points.4.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->4.99<!-- /calkit value -->%/GiB |
| 8 | <!-- calkit value key=points.8.gpt_per_group path=results/docs/gpoint-group-count.json -->32<!-- /calkit value --> | <!-- calkit value key=points.8.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->616.2<!-- /calkit value --> | <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.57<!-- /calkit value -->% | <!-- calkit value key=points.8.saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->22.8<!-- /calkit value --> | <!-- calkit value key=points.8.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.91<!-- /calkit value --> | <!-- calkit value key=points.8.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.92<!-- /calkit value -->%/GiB |
| 16 | <!-- calkit value key=points.16.gpt_per_group path=results/docs/gpoint-group-count.json -->16<!-- /calkit value --> | <!-- calkit value key=points.16.solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->581.8<!-- /calkit value --> | <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-8.95<!-- /calkit value -->% | <!-- calkit value key=points.16.saved_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->57.2<!-- /calkit value --> | <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.81<!-- /calkit value --> | <!-- calkit value key=points.16.pct_per_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->4.93<!-- /calkit value -->%/GiB |

## The default is justified, and that is not the same as being enough

The smallest group count within one repeat spread of the best saving is <!-- calkit value key=sweep.sufficient_ngroups path=results/docs/gpoint-group-count.json -->16<!-- /calkit value -->, against a default of <!-- calkit value key=sweep.default_ngpt_groups path=results/docs/gpoint-group-count.json -->16<!-- /calkit value -->, so the default carries no excess: `default_is_sufficient_only` is <!-- calkit value key=sweep.default_is_sufficient_only path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->.
Dropping to 8 would give back more than half the saving and dropping to 4 would give back three quarters of it, so the memory is buying something at every step and there is no cheaper group count that performs like 16.

The honest reading stops there, because `sufficient_is_sweep_maximum` is <!-- calkit value key=sweep.sufficient_is_sweep_maximum path=results/docs/gpoint-group-count.json -->true<!-- /calkit value -->.
"Smallest sufficient" is bounded by what was swept, and it landed on the largest count measured, which means this sweep did not find a knee -- it established that the knee is at 16 or beyond.
A sweep whose answer is its own upper endpoint cannot distinguish "16 is enough" from "16 is the most we tried", and the increments say which of those is more likely.

## The increments do not decay

| step | change in saving |
|---|---|
| 1 to 4 | <!-- calkit value key=increments.1_to_4 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.29<!-- /calkit value --> points |
| 4 to 8 | <!-- calkit value key=increments.4_to_8 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-1.28<!-- /calkit value --> points |
| 8 to 16 | <!-- calkit value key=increments.8_to_16 format="{:.2f}" path=results/docs/gpoint-group-count.json -->-5.38<!-- /calkit value --> points |

A saturating curve drives these toward zero and this one does the opposite: the step from 8 to 16 is the largest of the three and is four times the step from 4 to 8, so the marginal return is rising rather than falling at the top of the range.
Each doubling also halves the g-points per group, so the last step is where a group first drops to 16 g-points, and the per-GiB column shows 16 paying as well as the cheapest point despite costing four times the memory.

## Wave quantization accounts for the jump and not for the rest

The grid is linear in the group count -- <!-- calkit value key=quantization.blocks_per_group path=results/docs/gpoint-group-count.json -->96<!-- /calkit value --> blocks ungrouped, checked in the values block against the count recorded at sixteen groups rather than assumed -- and one block is resident per SM at 255 registers.
So a point's time is its wave count rounded up to a whole wave times its per-block work, and per-block work falls as 1/n because each group covers one nth of the g-points.
On <!-- calkit value key=quantization.sms path=results/docs/gpoint-group-count.json -->108<!-- /calkit value --> SMs that model predicts nothing at all for 4 and 8 groups, whose block counts divide into whole waves, and <!-- calkit value key=quantization.points.16.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-6.25<!-- /calkit value -->% for 16, whose <!-- calkit value key=quantization.points.16.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->1,536<!-- /calkit value --> blocks need <!-- calkit value key=quantization.points.16.waves_ceil path=results/docs/gpoint-group-count.json -->15<!-- /calkit value --> whole waves to carry <!-- calkit value key=quantization.points.16.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->14.22<!-- /calkit value --> waves of work.

| groups | blocks | waves | predicted | measured | residual |
|---|---|---|---|---|---|
| 4 | <!-- calkit value key=quantization.points.4.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->384<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.56<!-- /calkit value --> | <!-- calkit value key=quantization.points.4.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=points.4.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.29<!-- /calkit value -->% | <!-- calkit value key=quantization.points.4.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.29<!-- /calkit value --> |
| 8 | <!-- calkit value key=quantization.points.8.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->768<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->7.11<!-- /calkit value --> | <!-- calkit value key=quantization.points.8.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->0.00<!-- /calkit value -->% | <!-- calkit value key=points.8.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.57<!-- /calkit value -->% | <!-- calkit value key=quantization.points.8.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.57<!-- /calkit value --> |
| 16 | <!-- calkit value key=quantization.points.16.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->1,536<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->14.22<!-- /calkit value --> | <!-- calkit value key=quantization.points.16.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-6.25<!-- /calkit value -->% | <!-- calkit value key=points.16.change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-8.95<!-- /calkit value -->% | <!-- calkit value key=quantization.points.16.residual_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-2.70<!-- /calkit value --> |

The model earns its place by predicting the one step that looked anomalous and getting most of it, which moves the surprise off the shape of the curve and onto a residual that is roughly flat.
What is left over is a few points at every count above one, not growing with the count, which this sweep does not isolate and which is not the quantity the increments appeared to show.
The honest statement is that most of the 8-to-16 jump is an artifact of <!-- calkit value key=quantization.points.16.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->1,536<!-- /calkit value --> blocks not dividing <!-- calkit value key=quantization.sms path=results/docs/gpoint-group-count.json -->108<!-- /calkit value --> evenly, and that the rest of the saving arrives as soon as there is more than one group and then stays about the same size.

## What the model predicts for 32, which is a measurement and not a conclusion

Doubling again puts <!-- calkit value key=quantization.next.blocks format="{:,}" path=results/docs/gpoint-group-count.json -->3,072<!-- /calkit value --> blocks into <!-- calkit value key=quantization.next.waves format="{:.2f}" path=results/docs/gpoint-group-count.json -->28.44<!-- /calkit value --> waves, which quantizes to <!-- calkit value key=quantization.next.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-9.38<!-- /calkit value -->% against the <!-- calkit value key=quantization.points.16.predicted_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-6.25<!-- /calkit value -->% at sixteen, a further <!-- calkit value key=quantization.next.predicted_gain_pts format="{:.2f}" path=results/docs/gpoint-group-count.json -->-3.12<!-- /calkit value --> points.
Holding the residual at the value it takes at 16 groups, that puts 32 groups near <!-- calkit value key=quantization.next.predicted_total_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-12.08<!-- /calkit value -->% for <!-- calkit value key=quantization.next.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->3.63<!-- /calkit value --> GiB of workspace.
This is a prediction from a model fitted to nothing, which is the useful kind, and adding 32 to `GROUPS` falsifies it in one reservation.
It is not a result: holding the residual fixed is an assumption, and a model that predicts zero where the sweep measured two and three and a half points has already shown it is incomplete.

## What it costs

Sixteen groups take <!-- calkit value key=points.16.workspace_gib format="{:.2f}" path=results/docs/gpoint-group-count.json -->1.81<!-- /calkit value --> GiB of device memory, measured as a free-memory delta across the workspace construction rather than computed from shapes.
That is the whole cost of the treatment, and it scales linearly in the group count, so a 32-group point would ask for roughly 3.6 GiB.
Memory is the constraint that will decide how far this can go, not time.

## Correctness

The split is not bit-for-bit and cannot be, because a g-point sum taken in N partials is a different summation order.
The worst flux deviation from the ungrouped path is <!-- calkit value key=points.16.flux_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->7.93e-04<!-- /calkit value --> W/m2 at 16 groups, against fluxes of order a few hundred W/m2, and it does not grow with the group count: 4 and 8 groups both sit at <!-- calkit value key=points.4.flux_deviation format="{:.2e}" path=results/docs/gpoint-group-count.json -->7.63e-04<!-- /calkit value --> W/m2.
A reordering error that accumulated with the number of partials would rise across the sweep, and this one is flat, which is the expected behavior for <!-- calkit value key=sweep.float_type path=results/docs/gpoint-group-count.json -->Float32<!-- /calkit value --> rounding rather than for a bug in the reduction.

## Two independent cross-checks against the nsys pass

The sweep and the profiling pass measure the same solve by different means, on different stacks and at different points in the simulation, so agreement is a check on both.
The nsys baseline arm spent <!-- calkit value key=nsys.baseline_per_launch_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->674.7<!-- /calkit value --> ms per launch in the ungrouped longwave kernel, against the sweep's <!-- calkit value key=nsys.sweep_ungrouped_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->639.1<!-- /calkit value --> ms for its ungrouped point, a gap of <!-- calkit value key=nsys.per_launch_gap_pct format="{:.1f}" path=results/docs/gpoint-group-count.json -->5.6<!-- /calkit value -->%.
The saving agrees too: the pass measured <!-- calkit value key=nsys.window_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-9.86<!-- /calkit value -->% over its window and the sweep measured <!-- calkit value key=sweep.best_change_pct format="{:.2f}" path=results/docs/gpoint-group-count.json -->-8.95<!-- /calkit value -->% in block, which is <!-- calkit value key=nsys.window_saved_per_launch_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->66.5<!-- /calkit value --> ms against <!-- calkit value key=nsys.sweep_saved_per_solve_ms format="{:.1f}" path=results/docs/gpoint-group-count.json -->57.2<!-- /calkit value --> ms per solve.
Neither pair should match exactly -- the arms differ, the cloud state differs after <!-- calkit value key=sweep.warmup_steps path=results/docs/gpoint-group-count.json -->3<!-- /calkit value --> warmup steps rather than 120, and one of the two was taken under a profiler -- so the point is that two methods with unrelated failure modes put the cost and the saving in the same place.
The sweep's figure is the smaller of the two, which is the direction to prefer for a claim.

## Verdict

`DEFAULT_NGPT_GROUPS = 16` is supported as the smallest count that reaches the measured best saving, and the open question has changed rather than closed: it is no longer whether 16 is too many but whether it is too few.
The saving is worth <!-- calkit value key=nsys.pct_of_device_baseline path=results/docs/gpoint-group-count.json -->7.71<!-- /calkit value -->% of the device at the window share of the longwave kernel, so the next point is worth one reservation if the memory is affordable.
What this document does not establish is a knee, a mechanism for the rising increments, or any wall-clock figure: the device saving converts to well under the AMIP noise floor and is not measurable end to end on this machine.

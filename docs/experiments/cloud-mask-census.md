# Which cells the SGS closure skips, and whether any thread order lets it

`sgs-substep-cost` established that the CloudMicrophysics calls are 75.9% of the explicit SGS 1M closure call and that the clear cells do not share warps: 43.6% of environment cells carry no condensate while only 2.48% of 32-cell warps are clear in every lane.
Its verdict was that the way into this kernel is to change which lanes run the microphysics rather than which points skip it.
This experiment measures the two things that decide how: the rate at which the closure's own early-out fires, and how the cells it fires at distribute over warps under a thread order other than the one the launch happens to use.

The result is that the order is the whole difference, and that it costs nothing to change: the cells the guard already skips are almost never a whole warp under the current order and are one warp in ten under a level-aligned one.

## The guard that is already there, and what decides whether it fires

`_sum_1m_over_sgs` returns `zero_1m_tendency` before building the node rule when three conditions hold together: the cell carries no condensate, it has no equilibrium cloud, and its node temperatures stay clear of the mixed-phase band (`T - 8√T′T′ >= T_freeze` or `T + 8√T′T′ <= T_icenuc`).
The middle condition is written `!isfinite(λ_eq)`, and `λ_eq` is `-Inf` only where a *truncated* displacement makes the equilibrium condensate vanish: `_compute_sgs_moments` returns the mean cloud excess itself whenever `isinf(c)`, with `c` the `sgs_displacement_truncation`.

Whether the guard can fire at all therefore rests on one parameter, and the docstring of `compute_sgs_saturation_adjustment` gives its default as `Inf`, which would make the condition false at every cell and the early-out dead code.
The running model resolves it to <!-- calkit value key=guard.truncation format="{:.1f}" path=results/docs/cloud-mask-census.json -->3.0<!-- /calkit value --> instead, so the truncation is active and `λ_eq` is `-Inf` at <!-- calkit value key=guard.eq_nonfinite_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->65.9<!-- /calkit value -->% of cells.
The guard is live, and it fires at <!-- calkit value key=guard.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% of the <!-- calkit value key=cells format="{:,}" path=results/docs/cloud-mask-census.json -->1,941,504<!-- /calkit value --> environment cells.

That is why the census reads the parameter and counts the condition rather than inferring either from the source: a documented default and a resolved value disagreed here, and the difference is between a treatment that collects an existing saving and one that has to create it.

## Why the thread order is the variable

A warp runs the microphysics unless all 32 of its lanes can skip it, so the rate that matters is per warp and not per cell — and which cells share a warp is a property of the launch, not of the field.
The pointwise closure kernel indexes the parent layout's own order, `(v, i, j, h)` with the vertical fastest, and `subscope_indices` hands consecutive thread ranks consecutive indices, so a warp is 32 consecutive levels of one column.
Clear cells are not distributed that way: precipitation threads down through a column, so a 32-level run almost always contains a cell with condensate.
A level-aligned order — horizontal fastest, so a warp is two spectral elements at one level — groups cells that are clear or cloudy together, because dryness is a property of height far more than of a column.

The two orders are a permutation of the same cell set and run the same arithmetic on each cell, with no reduction across cells, so they return identical results to the bit.
What they do not share is the memory access pattern: the level-aligned order strides by the level count where the current one is contiguous.
That is affordable here and only here — §2a measured this kernel at 4.7% of DRAM peak, 7% of L2 and 3.3% of L1, against an occupancy ceiling of 12.4%, and concluded that coalescing and layout are irrelevant to it.

## What the threshold sweep is for

A dry cell's node rule does not place its nodes at the mean.
`conditional_rule` sorts the actual cloud edge `t_a = -λ_lagrange/α`, the equilibrium edge `t_e = -λ_eq` and the shaft edge into four intervals, and the outermost interval is a semi-infinite tail that always keeps two nodes, with weight `Q(|edge|/σ_S)`.
So the census also sweeps a criterion that replaces the guard's `!isfinite(λ_eq)` term with the distance of the nearer edge, `min(t_a, t_e)/σ_S`, over 2, 4, 6 and 8 standard deviations — which is what a *widened* criterion would have to be built on, since widening means giving up a tail weight rather than nothing.

## The census

`ClimaAtmos.jl-mod/src` is an input because the closure the probe calls is ClimaAtmos's own code and an edit there moves the mask, which is the reason to tag this experiment and freeze it once it has run.

```sh calkit stage name=cloud-mask environment=clima outputs=[{path: results/experiments/cloud-mask-census/mask.toml, storage: git}] inputs=[scripts/measure-cloud-mask.jl, scripts/run-julia-script.sh, ClimaAtmos.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=90]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-cloud-mask.jl \
  --out results/experiments/cloud-mask-census/mask.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/cloud-mask-census.json, storage: git}] inputs=[results/experiments/cloud-mask-census/mask.toml, results/experiments/sgs-substep-cost/substeps.toml, results/top-kernels.csv]
import json
import tomllib

d = tomllib.load(open("results/experiments/cloud-mask-census/mask.toml", "rb"))
sgs = tomllib.load(open("results/experiments/sgs-substep-cost/substeps.toml", "rb"))
expl = sgs["explicit"]
masks = d["masks"]

# The device budget and the profiled closure kernel, from one row of the
# profile so the denominator is the profile's own.
row = None
with open("results/top-kernels.csv") as f:
    header = f.readline().rstrip("\n").split(",")
    for line in f:
        parts = line.rstrip("\n").split(",")
        if parts and parts[0].startswith("set_environment_1m_tendency"):
            row = dict(zip(header, parts))
            break
if row is None:
    raise SystemExit("no set_environment_1m_tendency row in results/top-kernels.csv")
device_ms = 100.0 * float(row["total_ms_mod"]) / float(row["pct_of_gpu_time_mod"])

# What the explicit path costs over the profiled window, derived the way
# `sgs-substep-cost` derives it: the probe times one call, and the window runs
# two of them per step.
STEPS = 120
EXPLICIT_CALLS = 2
explicit_ms = expl["ms_in_use"] * EXPLICIT_CALLS * STEPS
cm_ms = explicit_ms * expl["substep_share_pct"] / 100.0

ORDERS = d["orders"]
WARP, BLOCK = "32", "128"


# A skipped cell leaves before the node rule is built, so it stops paying for
# the CM calls AND for the rule around them -- but the column loop in the same
# call is a separate kernel that no criterion reaches. So a saving is bracketed
# rather than converted: the CM calls are the floor, the whole call the ceiling,
# and timing the treatment is what resolves it.
def bracket(f):
    return {
        "floor_pct": round(100.0 * f * cm_ms / device_ms, 2),
        "ceiling_pct": round(100.0 * f * explicit_ms / device_ms, 2),
    }


def priced(name):
    m = masks[name]
    out = {
        "cell_pct": round(100.0 * m["cell_fraction"], 2),
        "compaction_saving_pct": round(100.0 * m["compaction_saving_fraction"], 1),
        "compaction": bracket(m["compaction_saving_fraction"]),
    }
    for order in ORDERS:
        f = m[order + "_saving_fraction"]
        out[order] = dict(
            warp_pct=round(
                100.0 * m["orders"][order][WARP]["all_skippable_fraction"], 2
            ),
            block_pct=round(
                100.0 * m["orders"][order][BLOCK]["all_skippable_fraction"], 2
            ),
            saving_pct=round(100.0 * f, 1),
            **bracket(f),
        )
    return out


thresholds = [int(k) for k in d["edge_sigmas"]]
sweep = {str(k): priced(f"edges_beyond_{k}_sigma") for k in thresholds}
# A threshold no cell reaches is degenerate rather than strict. A dry cell's
# actual-cloud edge is `-λ_lagrange/α` with `λ_lagrange` from
# `_continuous_lagrange(0, ασ)`, whose two Newton steps land at a fixed
# multiple of σ, so the sweep stops where that fixed point lands and not where
# the field does -- which is why the top of the sweep is reported rather than
# read as a strictness the criterion could be set to.
reached = [k for k in thresholds if sweep[str(k)]["cell_pct"] > 0]
unreached = [k for k in thresholds if sweep[str(k)]["cell_pct"] == 0]

# The live criterion is the guard as written, which is what a reorder alone
# collects; the widened one drops the other two terms and keeps only
# "no condensate", which is what a criterion change could add on top.
live = priced("guard_live")
widened = priced("no_condensate")
REORDER = "level_aligned"

out = {
    "cells": d["cells"],
    "nlev": d["nlev"],
    "warp_width": int(WARP),
    "block_width": int(BLOCK),
    "guard": {
        "cell_pct": live["cell_pct"],
        "eq_nonfinite_pct": round(100.0 * d["eq_nonfinite_cells"] / d["cells"], 1),
        "fires": d["eq_nonfinite_cells"] > 0,
        "truncation": d["sgs_displacement_truncation"],
        "curvature": d["sgs_variance_curvature"],
    },
    "live": live,
    "widened": widened,
    "temperature_band": priced("temperature_band"),
    "threshold_count": len(thresholds),
    "sweep": sweep,
    "reached_thresholds": len(reached),
    "strictest_reached": max(reached) if reached else None,
    "first_unreached": min(unreached) if unreached else None,
    "reorder": REORDER,
    "reorder_beats_current": live[REORDER]["saving_pct"] > 0,
    # The share of what a perfect gather of the same cells could remove that a
    # reorder alone reaches. This is what decides whether a compacted index
    # list, which needs a device-side prefix scan, is worth building.
    "reorder_share_of_compaction_pct": round(
        100.0 * live[REORDER]["saving_pct"] / live["compaction_saving_pct"]
    )
    if live["compaction_saving_pct"]
    else None,
    # What widening the criterion to "no condensate" would add on top of the
    # reorder, which is a separate and not bit-for-bit decision.
    "widening_gain_pct": round(
        widened[REORDER]["saving_pct"] - live[REORDER]["saving_pct"], 1
    ),
    "widening_floor_pct": round(
        widened[REORDER]["floor_pct"] - live[REORDER]["floor_pct"], 2
    ),
    "device": {
        "device_ms": round(device_ms, 1),
        "explicit_ms": round(explicit_ms, 1),
        "explicit_pct": round(100.0 * explicit_ms / device_ms, 2),
        "cm_calls_ms": round(cm_ms, 1),
        "cm_calls_pct": round(100.0 * cm_ms / device_ms, 2),
    },
    # Two probes, two runs, one state each: the gap between their timings of the
    # same call is what says whether the mask describes the field the cost was
    # measured on.
    "state": {
        "probe_call_ms": round(d["explicit_call_ms"], 2),
        "sgs_call_ms": round(expl["ms_in_use"], 2),
        "call_gap_pct": round(
            100.0 * (d["explicit_call_ms"] - expl["ms_in_use"]) / expl["ms_in_use"], 2
        ),
    },
}
json.dump(out, open("results/docs/cloud-mask-census.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/cloud-mask-census.json -->

## Result

The two probes describe the same field.
This one times the explicit call at <!-- calkit value key=state.probe_call_ms format="{:.2f}" path=results/docs/cloud-mask-census.json -->16.23<!-- /calkit value --> ms against `sgs-substep-cost`'s <!-- calkit value key=state.sgs_call_ms format="{:.2f}" path=results/docs/cloud-mask-census.json -->16.23<!-- /calkit value --> ms, a gap of <!-- calkit value key=state.call_gap_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->-0.01<!-- /calkit value -->%, so the mask below belongs to the state the cost was measured on and the two can be multiplied.

### The order is the whole difference

| criterion | cells | warps now | warps level-aligned | 128-blocks level-aligned |
|---|---|---|---|---|
| the guard as written | <!-- calkit value key=live.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% | <!-- calkit value key=live.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.31<!-- /calkit value -->% | <!-- calkit value key=live.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->10.26<!-- /calkit value -->% | <!-- calkit value key=live.level_aligned.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->7.80<!-- /calkit value -->% |
| no condensate | <!-- calkit value key=widened.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->43.56<!-- /calkit value -->% | <!-- calkit value key=widened.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->2.48<!-- /calkit value -->% | <!-- calkit value key=widened.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->26.04<!-- /calkit value -->% | <!-- calkit value key=widened.level_aligned.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->23.30<!-- /calkit value -->% |
| outside the mixed-phase band | <!-- calkit value key=temperature_band.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->51.81<!-- /calkit value -->% | <!-- calkit value key=temperature_band.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->9.24<!-- /calkit value -->% | <!-- calkit value key=temperature_band.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->42.91<!-- /calkit value -->% | <!-- calkit value key=temperature_band.level_aligned.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->35.70<!-- /calkit value -->% |

The guard fires at a quarter of the cells and at <!-- calkit value key=live.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.31<!-- /calkit value -->% of warps, which is to say it is being paid for and almost never collected: an existing early-out, already in the code, already correct, reaching three warps in a thousand.
Under a level-aligned order the same cells make <!-- calkit value key=live.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->10.26<!-- /calkit value -->% of warps entirely skippable, which removes <!-- calkit value key=live.level_aligned.saving_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->10.0<!-- /calkit value -->% of the warp-work the kernel does now.
The gain survives at block granularity, where a skipped group frees its registers rather than only its issue slots: <!-- calkit value key=live.level_aligned.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->7.80<!-- /calkit value -->% of 128-thread blocks are entirely skippable against <!-- calkit value key=live.vertical_fastest.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.00<!-- /calkit value -->% now.

A perfect gather of the same cells would remove <!-- calkit value key=live.compaction_saving_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->22.7<!-- /calkit value -->%, so the reorder reaches <!-- calkit value key=reorder_share_of_compaction_pct format="{:.0f}" path=results/docs/cloud-mask-census.json -->44<!-- /calkit value -->% of the compaction bound without a prefix scan, a compacted index list, or a single scattered access the field does not already make.

### What it is worth, as a bracket rather than a number

A skipped cell leaves before the node rule is built, so it stops paying for the CloudMicrophysics calls and for the rule around them — but the column loop in the same call is a separate kernel that no criterion reaches, so the saving is not a fixed share of anything the profile reports.
Against the mod arm's <!-- calkit value key=device.device_ms format="{:,.1f}" path=results/docs/cloud-mask-census.json -->33,306.1<!-- /calkit value --> ms device budget, the CM calls on the explicit path are <!-- calkit value key=device.cm_calls_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->8.88<!-- /calkit value -->% and the whole explicit call is <!-- calkit value key=device.explicit_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->11.70<!-- /calkit value -->%, so the reorder is worth between <!-- calkit value key=live.level_aligned.floor_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.89<!-- /calkit value -->% and <!-- calkit value key=live.level_aligned.ceiling_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->1.17<!-- /calkit value -->% of device on the explicit path alone.
Both ends are floors across the whole closure: the guard is active on all eight calls a step makes, and the six implicit ones are another 7.86% of device whose CM share this fit cannot separate.

### The threshold sweep says the criterion cannot be widened by strictness

Of the <!-- calkit value key=threshold_count format="{:d}" path=results/docs/cloud-mask-census.json -->4<!-- /calkit value --> thresholds swept, <!-- calkit value key=reached_thresholds format="{:d}" path=results/docs/cloud-mask-census.json -->3<!-- /calkit value --> are reached by any cell at all, and the ones that are give the same rate as the guard to two decimals.
The first unreached threshold is <!-- calkit value key=first_unreached format="{:d}" path=results/docs/cloud-mask-census.json -->8<!-- /calkit value --> standard deviations, and that is a property of the code rather than of the field: a condensate-free cell's actual-cloud edge is `-λ_lagrange/α` with `λ_lagrange = _continuous_lagrange(0, ασ)`, whose two Newton steps land at a fixed multiple of σ, so no cell's nearer edge can exceed it however dry the cell is.
So the edge distance is not the term to turn.

What is left is the mixed-phase band.
It is the term that halves the criterion — <!-- calkit value key=widened.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->43.56<!-- /calkit value -->% of cells carry no condensate against the guard's <!-- calkit value key=live.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% — and dropping it would add <!-- calkit value key=widening_gain_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->14.2<!-- /calkit value --> points of warp-work, at least <!-- calkit value key=widening_floor_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->1.25<!-- /calkit value -->% of device, on top of the reorder.
That is not a free change: inside the band `q_smin` is the ice saturation rather than the single-phase one, so a dry cell there can still deposit, and a criterion that skips it would have to test the excess against ice saturation instead of dropping the test.
That is a second experiment, and it is worth one.

## Verdict

**The thread order is a treatment and it is bit-for-bit.** The two orders run the same arithmetic on the same cells with no reduction between them, so the only thing that changes is which cells share a warp. The guard that collects the saving is already in the code and already fires at <!-- calkit value key=live.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% of cells; it reaches <!-- calkit value key=live.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.31<!-- /calkit value -->% of warps under the order the launch uses and <!-- calkit value key=live.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->10.26<!-- /calkit value -->% under a level-aligned one. Build the order, not the compacted index list: it reaches <!-- calkit value key=reorder_share_of_compaction_pct format="{:.0f}" path=results/docs/cloud-mask-census.json -->44<!-- /calkit value -->% of what a perfect gather could and needs no scan, no index buffer and no allocation per call.

**The number this experiment does not give is the one the treatment will.** The bracket above spans <!-- calkit value key=live.level_aligned.floor_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.89<!-- /calkit value -->% to <!-- calkit value key=live.level_aligned.ceiling_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->1.17<!-- /calkit value -->% of device because a warp census cannot know how much of a skipped point body was CM and how much was the rule, and because the strided reads the order implies are cheap by §2a's measurement rather than by this one's. Both are settled by timing the ordered launch against the current one in one process on one state, which is the same in-block comparison that resolved the substep cost, and that is the next step.

**Read the parameter, not the docstring.** This experiment was designed around the guard being dead, on the strength of `sgs_displacement_truncation`'s documented default of `Inf`. The model resolves it to <!-- calkit value key=guard.truncation format="{:.1f}" path=results/docs/cloud-mask-census.json -->3.0<!-- /calkit value -->, the truncation is active, `λ_eq` is `-Inf` at <!-- calkit value key=guard.eq_nonfinite_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->65.9<!-- /calkit value -->% of cells, and the guard fires. Had the census not counted the condition it was built to explain, the treatment would have been a repaired criterion with an accuracy argument to make, instead of a launch-configuration change with none.

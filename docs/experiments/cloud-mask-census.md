# Which cells the SGS closure could skip, and whether any thread order lets it

`sgs-substep-cost` established that the CloudMicrophysics calls are 75.9% of the explicit SGS 1M closure call and that the clear cells do not share warps: 43.6% of environment cells carry no condensate while only 2.48% of 32-cell warps are clear in every lane, so 97.5% of warps run the cloudy path.
Its verdict was that the way into this kernel is to change which lanes run the microphysics rather than which points skip it.
This experiment measures the two things that decide how: the rate at which the closure's own early-out actually fires, and how a skippable cell set distributes over warps under thread orders other than the one the launch happens to use.

## The guard that is already there, and why it may never fire

`_sum_1m_over_sgs` returns `zero_1m_tendency` before building the node rule when three conditions hold together: the cell carries no condensate, it has no equilibrium cloud, and its node temperatures stay clear of the mixed-phase band (`T - 8√T′T′ >= T_freeze` or `T + 8√T′T′ <= T_icenuc`).
The middle condition is written `!isfinite(λ_eq)`, and `λ_eq` is `-Inf` only where a *truncated* displacement makes the equilibrium condensate vanish: `_compute_sgs_moments` returns `mu_S_cloud` itself whenever `isinf(c)`, and `c` is `sgs_displacement_truncation`, whose default is `Inf`.
The curvature-aware branch would make `λ_eq` the fitted multiplier instead, and it is off by default (`sgs_variance_curvature: false`).
Neither is set in this benchmark's configuration, so the condition should be false at every cell and the guard should never fire.

That is a claim about the running model, not about the source, so the census counts it: `masks.guard_live.cell_fraction` is the rate at which the early-out fires and `eq_nonfinite_cells` is its middle term on its own.

## What a repaired criterion would have to be

A dry cell's node rule does not place its nodes at the mean.
`conditional_rule` sorts the actual cloud edge `t_a = -λ_lagrange/α`, the equilibrium edge `t_e = -λ_eq` and the shaft edge into four intervals, and the outermost interval is a semi-infinite tail that always keeps two nodes, with weight `Q(|edge|/σ_S)`.
So a condensate-free, subsaturated cell still has nodes carrying condensate, with a weight that falls off as the edges move out but never reaches zero.
Skipping such a cell is therefore not bit-for-bit; it is below rounding once the edges are far enough out, and how far is a number rather than a judgment.

The census replaces the dead `!isfinite(λ_eq)` term with the distance of the nearer edge, `min(t_a, t_e) / σ_S`, and sweeps the threshold over 2, 4, 6 and 8 standard deviations, so the hit rate is reported against the tail weight being given up rather than at one chosen strictness.

## Why the thread order is the other variable

A warp runs the microphysics unless all 32 of its lanes can skip it, so the rate that matters is per warp and not per cell — and which cells share a warp is a property of the launch, not of the field.
The pointwise closure kernel indexes the parent layout's own order, `(v, i, j, h)` with the vertical fastest, so a warp is 32 consecutive levels of one column.
Clear cells are not distributed that way: precipitation threads down through a column, so a 32-level run almost always contains a cell with condensate, which is the 2.48%.
A level-aligned order — horizontal fastest, so a warp is two spectral elements at one level — groups cells that are clear or cloudy together, because dryness is a property of height far more than of a column.

Both are censused here, against the per-cell rate that perfect compaction would reach and no order can beat.
The orders are a permutation of the same cell set and the same arithmetic, so this census prices a launch-configuration change against a compacted index list before either is built.

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

# What the CM calls on the explicit path cost over the profiled window, derived
# the same way `sgs-substep-cost` derives it: the probe times one call, and the
# window runs two of them per step.
STEPS = 120
EXPLICIT_CALLS = 2
explicit_ms = expl["ms_in_use"] * EXPLICIT_CALLS * STEPS
cm_ms = explicit_ms * expl["substep_share_pct"] / 100.0

ORDERS = d["orders"]
WIDTH = "32"


def priced(name):
    m = masks[name]
    out = {
        "cell_pct": round(100.0 * m["cell_fraction"], 2),
        "compaction_saving_pct": round(100.0 * m["compaction_saving_fraction"], 1),
        "compaction_device_pct": round(
            100.0 * m["compaction_saving_fraction"] * cm_ms / device_ms, 2
        ),
    }
    for order in ORDERS:
        f = m[order + "_saving_fraction"]
        out[order] = {
            "warp_pct": round(
                100.0 * m["orders"][order][WIDTH]["all_skippable_fraction"], 2
            ),
            "saving_pct": round(100.0 * f, 1),
            "device_pct": round(100.0 * f * cm_ms / device_ms, 2),
        }
    return out


# The sweep over the edge threshold, which is where the criterion's strictness
# is priced rather than chosen.
thresholds = [int(k) for k in d["edge_sigmas"]]
sweep = {str(k): priced(f"edges_beyond_{k}_sigma") for k in thresholds}

# The widest threshold is the one a treatment would ship, since its tail weight
# is the smallest; the narrowest says what the strictness costs.
strict, loose = str(max(thresholds)), str(min(thresholds))
best_order = max(
    ORDERS, key=lambda o: sweep[strict][o]["saving_pct"]
)

out = {
    "cells": d["cells"],
    "nlev": d["nlev"],
    "group_width": int(WIDTH),
    "thresholds": thresholds,
    # Does the closure's own early-out fire at all?
    "guard": {
        "cell_pct": round(100.0 * masks["guard_live"]["cell_fraction"], 4),
        "eq_nonfinite_cells": d["eq_nonfinite_cells"],
        "fires": d["eq_nonfinite_cells"] > 0,
        "truncation": d["sgs_displacement_truncation"],
        "curvature": d["sgs_variance_curvature"],
    },
    "no_condensate": priced("no_condensate"),
    "temperature_band": priced("temperature_band"),
    "sweep": sweep,
    "strict_threshold": int(strict),
    "loose_threshold": int(loose),
    # Which order wins at the strictest threshold, and by how much over the one
    # the launch uses now. A reorder that does not beat the current order is a
    # verdict against the cheap treatment and for the compacted index list.
    "best_order": best_order,
    "best_order_device_pct": sweep[strict][best_order]["device_pct"],
    "current_order_device_pct": sweep[strict]["vertical_fastest"]["device_pct"],
    "reorder_beats_current": (
        sweep[strict][best_order]["saving_pct"]
        > sweep[strict]["vertical_fastest"]["saving_pct"]
    ),
    "compaction_device_pct": sweep[strict]["compaction_device_pct"],
    # The share of the compaction bound that a reorder alone would reach, which
    # is what decides whether the index list is worth building.
    "reorder_share_of_compaction_pct": round(
        100.0
        * sweep[strict][best_order]["device_pct"]
        / sweep[strict]["compaction_device_pct"],
        1,
    )
    if sweep[strict]["compaction_device_pct"]
    else None,
    "device": {
        "device_ms": round(device_ms, 1),
        "explicit_ms": round(explicit_ms, 1),
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

To be written against the measurement.

## Verdict

To be written against the measurement.

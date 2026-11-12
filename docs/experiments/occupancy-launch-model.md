# The launch model, and what a register is worth

ClimaCore does not launch a thread per point.
It launches exactly one occupancy wave --- as many threads as the register file allows resident at once --- and strides the loop over the rest of its slices.
So the grid of every slice-loop kernel is *derived from its register count*, and a kernel's parallelism is not a property of the problem but of how many registers `ptxas` happened to use.
This document transcribes that model from ClimaCore, checks it against every launch in a profile pair, and prices the one lever it exposes.

The question it answers is the one the microphysics regression left open: `set_microphysics_tendency_cache` runs <!-- calkit value key=targets.threads path=results/experiments/occupancy-launch-model/stats.json -->55296<!-- /calkit value --> threads, which is exactly `ncol`, and that looks like a kernel decomposed one-thread-per-column.
It is not.
The coincidence is arithmetic --- <!-- calkit value key=model.sm_count path=results/experiments/occupancy-launch-model/stats.json -->108<!-- /calkit value --> multiprocessors times <!-- calkit value key=targets.block_size path=results/experiments/occupancy-launch-model/stats.json -->512<!-- /calkit value --> threads is <!-- calkit value key=targets.threads path=results/experiments/occupancy-launch-model/stats.json -->55296<!-- /calkit value --> --- and the kernel is a point loop whose wave happens to be the same size as the horizontal grid.

## Why this experiment declares no profiling stages

Every figure here comes from profiles that have already been paid for, so this experiment costs no GPU time and declares no `clima` stage.
What it declares instead is an analysis over two committed snapshots: `results/launch-inventory.json` for the launch geometry and `results/retired/sedimentation-fold.json` for the calibration.

That indirection is required rather than convenient.
`results/nsys/baseline.sqlite` holds whichever experiment profiled last, so it cannot be a stage dependency --- the pipeline would regenerate it and the write-up would silently restate itself against a different run.
`scripts/measure_occupancy.py` reads it once, by hand, and commits the geometry it needs with the source profiles identified by hash, which is the same arrangement `results/launch-probe.json` uses and for the same reason.

This document is therefore **not frozen**, unlike the other experiment write-ups here.
Freezing exists to stop a document being regenerated against a profile it did not describe, and nothing this one reads is transient: its inputs change only when someone deliberately re-runs the snapshot.

## Analysis

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/occupancy-launch-model/stats.json, storage: git}] inputs=[scripts/measure_occupancy.py, results/launch-inventory.json, results/retired/sedimentation-fold.json, results/retired/fold-on-current-stack.json, results/regression-record/attribution-1103.json, results/quadrature-registers-baseline.toml, results/quadrature-registers-mod.toml, results/quadrature-registers-frames.toml]
import json
import sys

sys.path.insert(0, "scripts")
from measure_occupancy import (
    A100,
    WARPS_PER_SM,
    calibration,
    max_active_blocks,
    predict_step,
    step_table,
    warps_per_sm,
)

inv = json.load(open("results/launch-inventory.json"))
base = inv["arms"]["baseline"]

# Device ms to points of SYPD, measured on this stack rather than assumed.
SYPD_PCT_PER_MS = json.load(open("results/retired/fold-on-current-stack.json"))["device"][
    "sypd_pct_per_device_ms"
]

steps = step_table()
current = next(s for s in steps if s["min_registers"] <= 128 <= s["max_registers"])
# The next step up is one register below the current band, by construction.
nxt = next(s for s in steps if s["max_registers"] == current["min_registers"] - 1)
full = next(s for s in steps if s["warps_per_sm"] == WARPS_PER_SM)

# The targets: the kernels that fill a wave, are capped by registers, and sit in
# the widest band below the cap. They are the three heaviest kernels on the
# device, and the fourth (set_precipitation_velocities) is already treated in the
# mod arm by the sedimentation fold, so it is reported but not counted.
cap = [
    r
    for r in base["top_kernels"]
    if r["fills_one_wave"] and r["limiter"] == "registers"
]
targets = [r for r in cap if r["warps_per_sm"] == current["warps_per_sm"]]
counted = [r for r in targets if r["block_size"] == 512]

# nsys's local-memory-per-thread field, under either spelling the inventory has
# used. It is reported, never treated as a spill measurement: see below.
def nsys_local(r):
    v = r.get("nsys_local_memory_per_thread", r.get("spill_bytes_per_thread"))
    return 0 if v is None else v

target_ms = round(sum(r["total_ms"] for r in counted), 1)

cal = calibration()
assert cal["all_model_agrees"], "the model disagrees with the fold's recorded warps per SM"
exponent = cal["exponent_mean"]

# What the 2026-11-03 fusion did to this kernel's occupancy, which decides
# whether the register lever recovers the regression or adds to it. The shape
# fields are read from the attribution snapshot, which reports the register
# count and the block size of one launch rather than the maximum of each.
attr = json.load(open("results/regression-record/attribution-1103.json"))
mp = next(
    r
    for r in attr["functions"]
    if r["function"] == "set_microphysics_tendency_cache"
)
fusion = {
    "pre_registers": mp["pre_registers"],
    "pre_block_size": mp["pre_threads_per_block"],
    "pre_warps_per_sm": warps_per_sm(mp["pre_registers"]),
    "now_registers": mp["now_registers"],
    "now_block_size": mp["now_threads_per_block"],
    "now_warps_per_sm": warps_per_sm(mp["now_registers"]),
    "pre_ms": mp["pre_ms"],
    "now_ms": mp["now_ms"],
    "ms_change_pct": round(100 * (mp["now_ms"] / mp["pre_ms"] - 1), 1),
}
fusion["occupancy_rose"] = fusion["now_warps_per_sm"] > fusion["pre_warps_per_sm"]
fusion["warps_gained"] = fusion["now_warps_per_sm"] - fusion["pre_warps_per_sm"]

# What each occupancy step is worth, and whether it clears the noise floor that
# separates two runs of identical code.
prediction = []
for target in (96, 80, 72, 64, 56, 48, 32):
    w = warps_per_sm(target)
    p = predict_step(target_ms, current["warps_per_sm"], w, exponent)
    points = round(p["saved_ms"] * SYPD_PCT_PER_MS, 2)
    p.update(
        target_registers=target,
        sypd_points=points,
        vs_null_floor=round(points / cal["null_floor_pct"], 2),
        resolvable=points > cal["null_floor_pct"],
    )
    prediction.append(p)

first = prediction[0]
recommended = next(
    (p["target_registers"] for p in prediction if p["vs_null_floor"] >= 1.5), None
)

# The one structural saving the kernel offers from ClimaAtmos, measured by
# compiling the real broadcast rather than by estimating it. The two snapshots
# come from one script run over the two arms, and the non-quadrature sibling is
# the in-run control: it shares the launcher and the layout and is untouched by
# the treatment, so an equal control with an unequal target is the treatment's
# own effect.
import tomllib

TARGET_CASE = "quadrature broadcast (L1297)"
CONTROL_CASE = "non-quadrature sibling"
probe = {
    a: tomllib.load(open(f"results/quadrature-registers-{a}.toml", "rb"))
    for a in ("baseline", "mod")
}
assert not any(p["climaatmos_dirty"] for p in probe.values()), "probe ran on a dirty tree"
eager, lazy = probe["baseline"]["cases"], probe["mod"]["cases"]
attempt = {
    "treatment": "rebuild the eight conditional-moment nodes on demand "
    "instead of materializing the node table",
    "eager_commit": probe["baseline"]["climaatmos_commit"][:9],
    "lazy_commit": probe["mod"]["climaatmos_commit"][:9],
    "eager_registers": eager[TARGET_CASE]["registers"],
    "lazy_registers": lazy[TARGET_CASE]["registers"],
    "eager_spill_bytes": eager[TARGET_CASE]["local_bytes"],
    "lazy_spill_bytes": lazy[TARGET_CASE]["local_bytes"],
    "control_eager_registers": eager[CONTROL_CASE]["registers"],
    "control_lazy_registers": lazy[CONTROL_CASE]["registers"],
    "probe_target_registers": eager[TARGET_CASE]["registers"],
    "profiled_target_registers": current["max_registers"],
}
attempt["register_change"] = attempt["lazy_registers"] - attempt["eager_registers"]
attempt["spill_change_bytes"] = (
    attempt["lazy_spill_bytes"] - attempt["eager_spill_bytes"]
)
attempt["control_unchanged"] = (
    attempt["control_eager_registers"] == attempt["control_lazy_registers"]
)
attempt["probe_error_registers"] = (
    attempt["probe_target_registers"] - attempt["profiled_target_registers"]
)
# The table is live across the integration, five fields of eight nodes.
attempt["node_table_registers"] = 8 * 5
attempt["helped"] = attempt["register_change"] < 0

# What any register treatment on this kernel could reach at best. The node
# function is `@noinline`, so `ptxas` allocates for the largest frame on the
# call graph rather than the sum: a barrier splits a frame and never shrinks
# one, which makes the largest frame the floor of every barrier arrangement.
# Each frame is measured by compiling it alone -- the traversal with a stub node
# function in place of the evaluator, the evaluator at one point, and
# CloudMicrophysics' bulk tendency with nothing of ClimaAtmos around it -- which
# needs no edit to ClimaAtmos, since `sum_over_conditional_nodes` takes the node
# function as an argument.
fr = tomllib.load(open("results/quadrature-registers-frames.toml", "rb"))
assert not fr["climaatmos_dirty"], "frame census ran on a dirty tree"
fc = fr["cases"]
frames = {
    "census_commit": fr["climaatmos_commit"][:9],
    "whole_kernel_registers": fc[TARGET_CASE]["registers"],
    "traversal_registers": fc["traversal frame (stub node function)"]["registers"],
    "node_function_registers": fc["node function frame (recon + BMT)"]["registers"],
    "bmt_registers": fc["BMT frame (CloudMicrophysics only)"]["registers"],
    "bmt_warps_per_sm": fc["BMT frame (CloudMicrophysics only)"]["warps_per_sm"],
    "step_to_gain_registers": nxt["max_registers"],
    "kernel_warps_per_sm_profiled": current["warps_per_sm"],
}
# What ClimaAtmos's own reconstruction adds on top of the package it calls.
frames["climaatmos_registers"] = (
    frames["node_function_registers"] - frames["bmt_registers"]
)
frames["bmt_over_step"] = frames["bmt_registers"] - frames["step_to_gain_registers"]
# The question the census settles: if CloudMicrophysics' own frame already needs
# more registers than the next step allows, no ClimaAtmos-side treatment can
# move this kernel's occupancy, however perfectly it works.
frames["occupancy_reachable"] = frames["bmt_over_step"] <= 0
frames["bmt_in_kernels_own_band"] = (
    frames["bmt_warps_per_sm"] == frames["kernel_warps_per_sm_profiled"]
)

# The census's own numbers refute the rule first used to read it. If a barrier
# made a kernel's registers the MAXIMUM over the frames on its call graph, the
# whole kernel would compile at the largest frame; it compiles above every one of
# them, because values the traversal holds live ACROSS the call cannot occupy the
# registers the callee is using. So the frames add rather than max, and the floor
# a ClimaAtmos-side treatment could reach is the callee's frame -- which leaves
# real registers reachable, even though reaching them crosses no band.
frames["largest_single_frame"] = max(
    frames["traversal_registers"],
    frames["node_function_registers"],
    frames["bmt_registers"],
)
frames["kernel_above_largest_frame"] = (
    frames["whole_kernel_registers"] - frames["largest_single_frame"]
)
frames["frames_add_rather_than_max"] = frames["kernel_above_largest_frame"] > 0
# Registers between where the kernel compiles and the floor its callee sets.
# This is a DIFFERENCE between two probe readings, which is what the probe is
# good for; the band question below needs absolute counts, so it converts to the
# profile's scale first rather than asking a band about a probe reading.
frames["recoverable_registers"] = (
    frames["whole_kernel_registers"] - frames["bmt_registers"]
)
frames["probe_error_registers"] = attempt["probe_error_registers"]
frames["floor_registers_profile_scale"] = (
    frames["bmt_registers"] - frames["probe_error_registers"]
)
# Reachable, but worth no occupancy: the floor and the kernel share a band.
frames["recovery_crosses_a_band"] = (
    warps_per_sm(frames["floor_registers_profile_scale"])
    != warps_per_sm(attempt["profiled_target_registers"])
)
# Headroom the recovery would buy inside the band, which is what an edit that
# costs registers needs and the kernel currently does not have.
frames["headroom_recovery_would_buy"] = (
    attempt["profiled_target_registers"] - frames["floor_registers_profile_scale"]
)

# How much the heaviest target could gain before it loses a band. The step table
# above gives the bands; what it does not say on its own is where in its band
# this kernel sits, and it sits on the edge -- so every change to it, including
# one that removes work, is an occupancy change.
headroom = {
    "heaviest_target_registers": max(r["registers"] for r in counted),
    "band_max_registers": current["max_registers"],
    "warps_in_band": current["warps_per_sm"],
    "warps_one_band_down": next(
        (b["warps_per_sm"] for b in steps
         if b["min_registers"] > current["max_registers"]), None
    ),
}
headroom["registers"] = (
    headroom["band_max_registers"] - headroom["heaviest_target_registers"]
)
headroom["at_band_edge"] = headroom["registers"] == 0
headroom["warps_lost_at_edge"] = (
    headroom["warps_in_band"] - headroom["warps_one_band_down"]
)

# The obvious way out of a band is to shrink the block: a kernel that cannot fit
# 512 threads at 130 registers might fit several smaller blocks whose threads sum
# to more. It cannot, and the reason is that the register file is handed out per
# scheduler in fixed granules, not per block -- a warp's registers are rounded up
# to `regs_per_allocation`, and each of the four schedulers can hold only whole
# warps of that size. So the ceiling belongs to the register count alone. This
# sweeps every legal block size at every register count and checks that none of
# them beats the block ClimaCore picks, which is what makes the band a cliff
# rather than a step a launch can walk around.
def best_block_warps(regs):
    best = (0, None)
    for blk in range(A100["threads_per_warp"], A100["max_threads_per_block"] + 1,
                     A100["threads_per_warp"]):
        total, _ = max_active_blocks(blk, regs, 0)
        per_sm = total // A100["sm_count"]
        warps = per_sm * (blk // A100["threads_per_warp"])
        if warps > best[0]:
            best = (warps, blk)
    return best

granule_rows = []
for regs in range(1, A100["max_regs_per_thread"] + 1):
    best_warps, best_blk = best_block_warps(regs)
    granule_rows.append(
        {
            "registers": regs,
            "climacore_warps": warps_per_sm(regs),
            "best_warps_any_block": best_warps,
            "best_block_size": best_blk,
        }
    )

granule = {
    "regs_per_allocation": A100["regs_per_allocation"],
    "schedulers_per_sm": A100["schedulers_per_sm"],
    "regs_per_scheduler": A100["max_regs_per_sm"] // A100["schedulers_per_sm"],
    "checked_register_counts": len(granule_rows),
    # The whole claim, in one field: no block size beats the band.
    "no_block_size_beats_the_band": all(
        r["best_warps_any_block"] <= r["climacore_warps"] for r in granule_rows
    ),
    "climacore_is_optimal_everywhere": all(
        r["best_warps_any_block"] == r["climacore_warps"] for r in granule_rows
    ),
    # At the edge the kernel sits on, stated as its own numbers.
    "edge_registers": headroom["heaviest_target_registers"],
    "over_edge_registers": headroom["heaviest_target_registers"] + 2,
    "edge_warps": warps_per_sm(headroom["heaviest_target_registers"]),
    "over_edge_warps": warps_per_sm(headroom["heaviest_target_registers"] + 2),
    "over_edge_best_warps_any_block": best_block_warps(
        headroom["heaviest_target_registers"] + 2
    )[0],
    # Granule boundaries are the only cap targets worth asking ptxas for: a cap
    # between them spends registers for no warps.
    "useful_caps": [
        b["max_registers"] for b in steps if b["max_registers"] <= 128
    ],
}

# How much of the model's device time could absorb a fusion. `ptxas` spends the
# granule it has been charged for, so register counts cluster ON the boundaries
# and headroom is usually zero -- which turns "add live state" into "lose a
# fifth of the occupancy" across most of the model. Both files are committed
# snapshots, because the profile and the pre-pull profile are both transient.
census = json.load(open("results/register-headroom.json"))
ledger = json.load(open("results/regression-record/fusion-ledger.json"))

headroom_census = {
    "arm": census["provenance"]["arm"],
    "kernel_variants": census["device"]["kernel_variants"],
    "device_total_ms": census["device"]["device_total_ms"],
    "regs_per_allocation": census["device"]["regs_per_allocation"],
    "regs_per_scheduler": census["device"]["regs_per_scheduler"],
    "zero_headroom_pct": census["summary"]["zero_headroom_pct"],
    "at_most_one_register_pct": census["summary"]["at_most_one_register_pct"],
    "most_of_device_cannot_absorb_one_register": census["summary"][
        "most_of_device_cannot_absorb_one_register"
    ],
    "useful_cap_targets": census["summary"]["useful_cap_targets"],
    # Where a fusion is safe: the buckets with room to grow.
    "roomy_pct": round(
        sum(v["pct_of_device"] for k, v in census["by_headroom"].items() if int(k) >= 4),
        1,
    ),
}

fusion_ledger = {
    "window_grew_pct": ledger["window_grew_pct"],
    "launches_change_pct": ledger["launches_change_pct"],
    "broadcast_machinery_saved_ms": ledger["broadcast_machinery_saved_ms"],
    "receivers_net_ms": ledger["receivers_net_ms"],
    "paid_per_ms_saved_low": ledger["paid_per_ms_saved_low"],
    "paid_per_ms_saved_high": ledger["paid_per_ms_saved_high"],
    "new_named_ms": ledger["new_named_ms"],
    "new_named_share_of_growth_pct": ledger["new_named_share_of_growth_pct"],
    "recorded_all_other_growth_per_ms_saved": ledger[
        "recorded_all_other_growth_per_ms_saved"
    ],
    "fusion_about_break_even": ledger["fusion_about_break_even"],
}

out = {
    "model": {
        "device": inv["provenance"]["device"],
        "source": inv["provenance"]["model_source"],
        "sm_count": A100["sm_count"],
        "warps_per_sm_capacity": WARPS_PER_SM,
        "max_waves": inv["provenance"]["max_waves"],
        "n_steps": len(steps),
        "steps": steps,
        "current_band": current,
        "next_step": nxt,
        "full_occupancy_registers": full["max_registers"],
    },
    "validation": base["validation"],
    "inventory": {
        k: base[k]
        for k in (
            "device_total_ms",
            "kernel_variants",
            "fills_one_wave_variants",
            "fills_one_wave_ms",
            "fills_one_wave_pct_of_device",
            "below_one_wave_variants",
            "above_one_wave_variants",
            "register_limited_ms",
            "register_limited_variants",
            "register_limited_pct_of_device",
        )
    },
    "targets": {
        "n": len(counted),
        "total_ms": target_ms,
        "pct_of_device": round(100 * target_ms / base["device_total_ms"], 2),
        "warps_per_sm": current["warps_per_sm"],
        "occupancy_pct": current["occupancy_pct"],
        "registers": sorted(r["registers"] for r in counted),
        "max_registers": max(r["registers"] for r in counted),
        "min_registers": min(r["registers"] for r in counted),
        # nsys does not populate this field, so it is reported as what it is
        # rather than as a spill measurement; see the prose below. The committed
        # snapshot predates the rename, so both spellings are accepted.
        "nsys_local_memory_per_thread": sorted(
            {nsys_local(r) for r in counted}
        ),
        "nsys_reports_spill": any(nsys_local(r) for r in base["top_kernels"]),
        "probe_local_memory_bytes": attempt["eager_spill_bytes"],
        "threads": counted[0]["threads"],
        "block_size": counted[0]["block_size"],
        "grid_size": counted[0]["grid_size"],
        # Registers each kernel can shed before anything changes: the band is
        # wide, so a partial saving is worth exactly nothing.
        "dead_headroom_registers": {
            r["kernel"].split("__FILE_")[0] + f"@{r['registers']}": r["registers"]
            - current["min_registers"]
            for r in counted
        },
        "max_dead_headroom_registers": max(
            r["registers"] - current["min_registers"] for r in counted
        ),
        "kernels": [
            {"kernel": r["kernel"][:70], "total_ms": r["total_ms"], "registers": r["registers"]}
            for r in counted
        ],
    },
    "calibration": cal,
    "fusion": fusion,
    "attempt": attempt,
    "frames": frames,
    "headroom": headroom,
    "granule": granule,
    "headroom_census": headroom_census,
    "fusion_ledger": fusion_ledger,
    "prediction": {
        "sypd_pct_per_device_ms": SYPD_PCT_PER_MS,
        "exponent": exponent,
        "null_floor_pct": cal["null_floor_pct"],
        "steps": prediction,
        "first_step_registers": first["target_registers"],
        "first_step_saved_ms": first["saved_ms"],
        "first_step_sypd_points": first["sypd_points"],
        "first_step_vs_null_floor": first["vs_null_floor"],
        "first_step_resolvable": first["resolvable"],
        "recommended_target_registers": recommended,
    },
}

# ncol is a coincidence, not a decomposition; assert it so the claim is checked.
assert out["targets"]["threads"] == 55296, out["targets"]["threads"]
# The control carries the comparison, so an unequal control invalidates it.
assert attempt["control_unchanged"], attempt
# The census is only a closure of this lever while CloudMicrophysics' own frame
# sits in the band the whole kernel already occupies.
assert frames["bmt_in_kernels_own_band"], frames
# The band is a property of the register count, not of the launch: if some block
# size could beat it, the cliff below would be a detour rather than a wall.
# The max rule would make the kernel compile at its largest frame; it does not.
assert frames["frames_add_rather_than_max"], frames
assert not frames["recovery_crosses_a_band"], frames
assert granule["no_block_size_beats_the_band"], granule
# The precondition for fusion, as a checked claim rather than an impression.
assert headroom_census["most_of_device_cannot_absorb_one_register"], headroom_census
# The ledger must still account for the window it decomposes.
assert ledger["buckets_account_for_window"], ledger
assert fusion_ledger["fusion_about_break_even"], fusion_ledger
assert granule["climacore_is_optimal_everywhere"], granule

json.dump(out, open("results/experiments/occupancy-launch-model/stats.json", "w"), indent=2)
```

<!-- calkit values path=results/experiments/occupancy-launch-model/stats.json -->

## The model, and the check on it

`ClimaCore.jl/ext/cuda/cuda_utils.jl` computes a launch configuration by walking every block size, asking CUDA's occupancy arithmetic how many blocks of that size fit on a multiprocessor, and taking the configuration with the most total threads.
The number of waves it schedules is <!-- calkit value key=model.max_waves path=results/experiments/occupancy-launch-model/stats.json -->1<!-- /calkit value --> by default, so the grid holds exactly the blocks that are resident at once and the loop strides.
The consequence is that occupancy is a step function of the register count, with <!-- calkit value key=model.n_steps path=results/experiments/occupancy-launch-model/stats.json -->11<!-- /calkit value --> steps on an A100's <!-- calkit value key=model.warps_per_sm_capacity path=results/experiments/occupancy-launch-model/stats.json -->64<!-- /calkit value --> warps per multiprocessor, because registers are allocated to warps in units of 256 and to schedulers in whole warps.

The model's claim is falsifiable and is tested without reference to the launch being tested: given only a kernel's register count, predict the block size and grid it will be launched with.
On the <!-- calkit value key=validation.checked path=results/experiments/occupancy-launch-model/stats.json -->187<!-- /calkit value --> launches where the prediction is well posed --- no shared memory, and enough slices to fill a wave --- it gets the resident thread count right <!-- calkit value key=validation.thread_count_agree path=results/experiments/occupancy-launch-model/stats.json -->187<!-- /calkit value --> times out of <!-- calkit value key=validation.checked path=results/experiments/occupancy-launch-model/stats.json -->187<!-- /calkit value -->, or <!-- calkit value key=validation.thread_count_agree_pct path=results/experiments/occupancy-launch-model/stats.json -->100.0<!-- /calkit value -->%.
It gets the block *shape* right <!-- calkit value key=validation.block_shape_agree_pct path=results/experiments/occupancy-launch-model/stats.json -->93.05<!-- /calkit value -->% of the time, and all <!-- calkit value key=validation.shape_only_mismatches path=results/experiments/occupancy-launch-model/stats.json -->13<!-- /calkit value --> misses are shape alone with the thread count intact: they are the loops that hand a slice to a sub-block rather than a thread, which ask for a coarser granularity and so spread the same threads over more, smaller blocks.

A second check comes from outside this document.
`results/retired/sedimentation-fold.json` recorded warps per multiprocessor for two kernels in two arms back in October, computed by a different script from the nsys launch records, and the model reproduces all four values from the register count alone --- `all_model_agrees` is <!-- calkit value key=calibration.all_model_agrees path=results/experiments/occupancy-launch-model/stats.json -->true<!-- /calkit value -->.

## How much of the device this governs

Of <!-- calkit value key=inventory.kernel_variants path=results/experiments/occupancy-launch-model/stats.json -->503<!-- /calkit value --> kernel variants in the baseline profile, <!-- calkit value key=inventory.fills_one_wave_variants path=results/experiments/occupancy-launch-model/stats.json -->244<!-- /calkit value --> launch a grid that exactly fills one wave, and they are <!-- calkit value key=inventory.fills_one_wave_pct_of_device path=results/experiments/occupancy-launch-model/stats.json -->71.88<!-- /calkit value -->% of device time.
Within those, the ones whose wave is bounded by **registers** rather than by warps, blocks or shared memory are <!-- calkit value key=inventory.register_limited_variants path=results/experiments/occupancy-launch-model/stats.json -->76<!-- /calkit value --> variants and <!-- calkit value key=inventory.register_limited_ms path=results/experiments/occupancy-launch-model/stats.json -->20458.4<!-- /calkit value --> ms --- <!-- calkit value key=inventory.register_limited_pct_of_device path=results/experiments/occupancy-launch-model/stats.json -->58.89<!-- /calkit value -->% of the device.
The figure is the same in the mod arm to two decimal places, which is what one expects of a structural property rather than a treatment effect.

The remaining time splits two ways.
A further <!-- calkit value key=inventory.below_one_wave_variants path=results/experiments/occupancy-launch-model/stats.json -->218<!-- /calkit value --> variants launch *less* than a wave, so their grid is bounded by how few slices the loop had and occupancy is not what limits them.
Another <!-- calkit value key=inventory.above_one_wave_variants path=results/experiments/occupancy-launch-model/stats.json -->41<!-- /calkit value --> launch more than a wave, which means they were not launched by a slice loop at all: they are GPUArrays broadcasts, RRTMGP's own `@cuda` calls, and the matrix-field solvers.

## The targets

The targets are the <!-- calkit value key=targets.n path=results/experiments/occupancy-launch-model/stats.json -->3<!-- /calkit value --> kernels that sit in the widest band, at <!-- calkit value key=targets.warps_per_sm path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value --> of 64 warps, or <!-- calkit value key=targets.occupancy_pct path=results/experiments/occupancy-launch-model/stats.json -->25.0<!-- /calkit value -->% occupancy, and they are the three heaviest kernels on the device at <!-- calkit value key=targets.total_ms path=results/experiments/occupancy-launch-model/stats.json -->10200.2<!-- /calkit value --> ms, or <!-- calkit value key=targets.pct_of_device path=results/experiments/occupancy-launch-model/stats.json -->29.36<!-- /calkit value -->% of device time.
Their register counts are <!-- calkit value key=targets.registers path=results/experiments/occupancy-launch-model/stats.json -->[114, 116, 128]<!-- /calkit value -->, all launched as <!-- calkit value key=targets.grid_size path=results/experiments/occupancy-launch-model/stats.json -->108<!-- /calkit value --> blocks of <!-- calkit value key=targets.block_size path=results/experiments/occupancy-launch-model/stats.json -->512<!-- /calkit value --> threads, which is <!-- calkit value key=targets.threads path=results/experiments/occupancy-launch-model/stats.json -->55296<!-- /calkit value --> threads and the source of the one-thread-per-column misreading.
Nothing asked `ptxas` to economize: no cap is set at these call sites and the counts sit far below the 255-register ceiling, so it was free to use as many registers as it liked and chose these.
This document previously added "and they spill nothing", citing nsys's local memory per thread.
That was not a measurement: nsys leaves `localMemoryPerThread` at **0 for every kernel in the profile**, one distinct value across the whole table, so a zero there never meant a kernel does not spill --- which `nsys_reports_spill` now records as <!-- calkit value key=targets.nsys_reports_spill path=results/experiments/occupancy-launch-model/stats.json -->false<!-- /calkit value --> so a future nsys that does populate it is noticed.
What spill these kernels actually carry is unmeasured here. The register probe reads <!-- calkit value key=targets.probe_local_memory_bytes path=results/experiments/occupancy-launch-model/stats.json -->328<!-- /calkit value --> bytes of local memory per thread at the compile, but that figure covers the `@noinline` call frame and any local arrays as well as spilled registers, so it bounds spill rather than measuring it.
`ncu` separates them, and `results/kernel-resources.csv` has the fields --- but those stages are frozen at a revision where this kernel compiled to 255 registers, so the table describes different code and cannot answer it.

**The trap is the width of the band.** The band these kernels sit in runs from <!-- calkit value key=model.current_band.min_registers path=results/experiments/occupancy-launch-model/stats.json -->97<!-- /calkit value --> to <!-- calkit value key=model.current_band.max_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value --> registers, so the next step up requires reaching <!-- calkit value key=model.next_step.max_registers path=results/experiments/occupancy-launch-model/stats.json -->96<!-- /calkit value --> and the heaviest kernel can shed <!-- calkit value key=targets.max_dead_headroom_registers path=results/experiments/occupancy-launch-model/stats.json -->31<!-- /calkit value --> registers before anything at all changes.
A change that cuts fifteen registers off a 114-register kernel is a change that does nothing, and it will read as a null result rather than as a near miss.
That is worth knowing before spending a profiling run on one.

## What a register buys

Occupancy is not time.
More resident warps hide latency better but execute the same instructions, so time falls by less than warps rise, and the exponent relating the two has to be measured rather than assumed.

The sedimentation fold is the only treatment this repo has measured that moved a kernel across an occupancy step with a control population to resolve it against --- its worst control moved <!-- calkit value key=calibration.worst_control_delta_ms path=results/experiments/occupancy-launch-model/stats.json -->1.4<!-- /calkit value --> ms.
It gives two points, on two different kernels at two different warp ratios, and they agree closely: exponents of <!-- calkit value key=calibration.exponent_min path=results/experiments/occupancy-launch-model/stats.json -->-0.3283<!-- /calkit value --> and <!-- calkit value key=calibration.exponent_max path=results/experiments/occupancy-launch-model/stats.json -->-0.2996<!-- /calkit value -->, mean <!-- calkit value key=calibration.exponent_mean path=results/experiments/occupancy-launch-model/stats.json -->-0.3139<!-- /calkit value -->.
So time scales roughly as the cube root of resident warps, inverted: doubling occupancy buys about a fifth of the runtime, not half of it.

**This exponent is an upper bound in magnitude, and the reason is a confound that cannot be removed from the data.** The fold did not only raise occupancy; `Val`-folding the terminal-velocity parameters also removed work, and nothing in the measurement separates the two.
Whatever share of the fold's saving was less arithmetic rather than more warps makes the true occupancy exponent smaller than this, and the predictions below correspondingly optimistic.

## Prediction

Committed before any treatment is built, so that measuring one is a test.

| target | warps/SM | saved ms | SYPD points | vs noise floor |
|---|---|---|---|---|
| <!-- calkit value key=prediction.steps.0.target_registers path=results/experiments/occupancy-launch-model/stats.json -->96<!-- /calkit value --> regs | <!-- calkit value key=prediction.steps.0.warps_after path=results/experiments/occupancy-launch-model/stats.json -->20<!-- /calkit value --> | <!-- calkit value key=prediction.steps.0.saved_ms path=results/experiments/occupancy-launch-model/stats.json -->690.0<!-- /calkit value --> | <!-- calkit value key=prediction.steps.0.sypd_points path=results/experiments/occupancy-launch-model/stats.json -->1.83<!-- /calkit value --> | <!-- calkit value key=prediction.steps.0.vs_null_floor path=results/experiments/occupancy-launch-model/stats.json -->0.98<!-- /calkit value -->x |
| <!-- calkit value key=prediction.steps.1.target_registers path=results/experiments/occupancy-launch-model/stats.json -->80<!-- /calkit value --> regs | <!-- calkit value key=prediction.steps.1.warps_after path=results/experiments/occupancy-launch-model/stats.json -->24<!-- /calkit value --> | <!-- calkit value key=prediction.steps.1.saved_ms path=results/experiments/occupancy-launch-model/stats.json -->1219.0<!-- /calkit value --> | <!-- calkit value key=prediction.steps.1.sypd_points path=results/experiments/occupancy-launch-model/stats.json -->3.23<!-- /calkit value --> | <!-- calkit value key=prediction.steps.1.vs_null_floor path=results/experiments/occupancy-launch-model/stats.json -->1.73<!-- /calkit value -->x |
| <!-- calkit value key=prediction.steps.3.target_registers path=results/experiments/occupancy-launch-model/stats.json -->64<!-- /calkit value --> regs | <!-- calkit value key=prediction.steps.3.warps_after path=results/experiments/occupancy-launch-model/stats.json -->32<!-- /calkit value --> | <!-- calkit value key=prediction.steps.3.saved_ms path=results/experiments/occupancy-launch-model/stats.json -->1994.5<!-- /calkit value --> | <!-- calkit value key=prediction.steps.3.sypd_points path=results/experiments/occupancy-launch-model/stats.json -->5.28<!-- /calkit value --> | <!-- calkit value key=prediction.steps.3.vs_null_floor path=results/experiments/occupancy-launch-model/stats.json -->2.83<!-- /calkit value -->x |
| <!-- calkit value key=model.full_occupancy_registers path=results/experiments/occupancy-launch-model/stats.json -->32<!-- /calkit value --> regs | <!-- calkit value key=prediction.steps.6.warps_after path=results/experiments/occupancy-launch-model/stats.json -->64<!-- /calkit value --> | <!-- calkit value key=prediction.steps.6.saved_ms path=results/experiments/occupancy-launch-model/stats.json -->3599.0<!-- /calkit value --> | <!-- calkit value key=prediction.steps.6.sypd_points path=results/experiments/occupancy-launch-model/stats.json -->9.53<!-- /calkit value --> | <!-- calkit value key=prediction.steps.6.vs_null_floor path=results/experiments/occupancy-launch-model/stats.json -->5.1<!-- /calkit value -->x |

**The first occupancy step is not worth measuring end to end.** Reaching <!-- calkit value key=prediction.first_step_registers path=results/experiments/occupancy-launch-model/stats.json -->96<!-- /calkit value --> registers is predicted to save <!-- calkit value key=prediction.first_step_saved_ms path=results/experiments/occupancy-launch-model/stats.json -->690.0<!-- /calkit value --> ms, which is <!-- calkit value key=prediction.first_step_sypd_points path=results/experiments/occupancy-launch-model/stats.json -->1.83<!-- /calkit value --> points of SYPD against a measured floor of <!-- calkit value key=prediction.null_floor_pct path=results/experiments/occupancy-launch-model/stats.json -->1.869<!-- /calkit value --> points between two runs of identical code --- a ratio of <!-- calkit value key=prediction.first_step_vs_null_floor path=results/experiments/occupancy-launch-model/stats.json -->0.98<!-- /calkit value -->, and `first_step_resolvable` is <!-- calkit value key=prediction.first_step_resolvable path=results/experiments/occupancy-launch-model/stats.json -->false<!-- /calkit value -->.
It would be visible in the per-kernel device time and invisible in SYPD, which is the shape of result this repo has twice mistaken for a null.
The first target worth a full pair is <!-- calkit value key=prediction.recommended_target_registers path=results/experiments/occupancy-launch-model/stats.json -->80<!-- /calkit value --> registers.

The falsifier is two-sided, which is the point of writing it down.
If a treatment reaches an occupancy step and the kernel does not move at all, occupancy is not what limits these kernels and the register direction is dead.
If it moves by substantially *more* than the power law says, the exponent fitted from the fold does not transfer --- most likely because the fold's kernels were less latency-bound than these, which would make the remaining steps worth more than the table claims rather than less.

Neither has been tested, because no treatment has reached a step: the one structural saving the kernel offers from this side of the stack moves the register count the wrong way, which the last section records.

## What this rules out

**Register shaving was never the wrong idea; the arithmetic I used to dismiss it was wrong.** Reading <!-- calkit value key=targets.threads path=results/experiments/occupancy-launch-model/stats.json -->55296<!-- /calkit value --> threads as one per column makes the work look like the binding constraint, and under that reading the register file's capacity at <!-- calkit value key=targets.max_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value --> registers --- also <!-- calkit value key=targets.block_size path=results/experiments/occupancy-launch-model/stats.json -->512<!-- /calkit value --> threads per multiprocessor --- is a coincidence that leaves nothing to gain.
The causality is the other way round: the launcher computes the grid *from* the register count, so the two agree by construction and the register count is the only free variable.

**Un-fusing these kernels for parallelism would buy nothing.** They are already point loops over every level of every column, which is tens of times more slices than the <!-- calkit value key=targets.threads path=results/experiments/occupancy-launch-model/stats.json -->55296<!-- /calkit value --> threads a wave holds, and the loop strides over the rest: parallelism is abundant and unexploited, and splitting the kernel would not add any.
Splitting would only help by making each piece need fewer registers, which is the same lever reached a more expensive way.

**The occupancy lever is additive to the 2026-11-03 regression, not a recovery of it.** It is tempting to read a 25%-occupancy kernel that got three times slower in one upstream pull as an occupancy regression, and it is not one.
Before the fusion the same function ran at <!-- calkit value key=fusion.pre_registers path=results/experiments/occupancy-launch-model/stats.json -->150<!-- /calkit value --> registers in blocks of <!-- calkit value key=fusion.pre_block_size path=results/experiments/occupancy-launch-model/stats.json -->384<!-- /calkit value -->, which is <!-- calkit value key=fusion.pre_warps_per_sm path=results/experiments/occupancy-launch-model/stats.json -->12<!-- /calkit value --> warps per multiprocessor; after it, <!-- calkit value key=fusion.now_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value --> registers in blocks of <!-- calkit value key=fusion.now_block_size path=results/experiments/occupancy-launch-model/stats.json -->512<!-- /calkit value -->, which is <!-- calkit value key=fusion.now_warps_per_sm path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value -->.
Occupancy *rose* by <!-- calkit value key=fusion.warps_gained path=results/experiments/occupancy-launch-model/stats.json -->4<!-- /calkit value --> warps while the function went <!-- calkit value key=fusion.pre_ms path=results/experiments/occupancy-launch-model/stats.json -->4755.9<!-- /calkit value --> to <!-- calkit value key=fusion.now_ms path=results/experiments/occupancy-launch-model/stats.json -->7782.9<!-- /calkit value --> ms, or <!-- calkit value key=fusion.ms_change_pct path=results/experiments/occupancy-launch-model/stats.json -->63.6<!-- /calkit value -->%.
So nothing in the regression is explained by the lever this document prices, and conversely nothing the lever buys is a recovery of the regression: the two are independent, and the register predictions above are a gain over the current stack rather than a return to the previous one.

**A hard register cap is the wrong instrument for reaching a step.** ClimaCore exposes one, as the `maxregs` keyword on `foreach_slice`, and it works by letting `ptxas` spill whatever does not fit; the kernels here spill nothing today, so a cap buys occupancy by creating memory traffic.
This repo has already tried it twice --- a cap alone measured +0.43% and a 64-register cap measured -5.5%.
The register count has to fall because the kernel needs less live state, not because it was forbidden more.

**De-materializing the node table costs registers rather than saving them.** The quadrature kernel holds an eight-node conditional-moment rule live across its integration, five floats per node, so the table alone is <!-- calkit value key=attempt.node_table_registers path=results/experiments/occupancy-launch-model/stats.json -->40<!-- /calkit value --> registers --- the largest single piece of live state that `ClimaAtmos` owns here, and the only one of any size outside `CloudMicrophysics`, which neither arm devs.
Rebuilding each interval's node pair on demand instead, so that one pair is live where the table was, moved the compiled kernel from <!-- calkit value key=attempt.eager_registers path=results/experiments/occupancy-launch-model/stats.json -->130<!-- /calkit value --> registers to <!-- calkit value key=attempt.lazy_registers path=results/experiments/occupancy-launch-model/stats.json -->163<!-- /calkit value --> and its spill from <!-- calkit value key=attempt.eager_spill_bytes path=results/experiments/occupancy-launch-model/stats.json -->328<!-- /calkit value --> to <!-- calkit value key=attempt.lazy_spill_bytes path=results/experiments/occupancy-launch-model/stats.json -->360<!-- /calkit value --> bytes, a change of <!-- calkit value key=attempt.register_change path=results/experiments/occupancy-launch-model/stats.json -->33<!-- /calkit value -->.
The non-quadrature sibling compiled in the same two runs is the control: it shares the launcher, the layout and the parameter construction, it is untouched by the treatment, and it stayed at <!-- calkit value key=attempt.control_eager_registers path=results/experiments/occupancy-launch-model/stats.json -->102<!-- /calkit value --> registers in both arms, so nothing about the instrument moved.
The mechanism is that the generation's own working set --- a saturation-vapor-pressure evaluation per node, among others --- then lives at the same time as the evaluator's, and the peak is what `ptxas` allocates for, so the materialized table is the cheaper of the two arrangements rather than the more expensive.
Rematerializing to relieve pressure is a register allocator's own trick, and doing it by hand in the source competes with the allocator instead of helping it.

**A frame census closes the register lever on this kernel outright, and on stronger grounds than "the registers are in a package neither arm devs".** The node function is `@noinline`, so `ptxas` allocates for the largest frame on the call graph rather than the sum of the frames, which means a barrier can split a frame but never shrink one and the largest frame is the floor of every barrier arrangement ClimaAtmos could adopt.
Each frame compiles alone, with no edit to ClimaAtmos at all, because `sum_over_conditional_nodes` takes the node function as an argument: the traversal with a stub in the evaluator's place reads <!-- calkit value key=frames.traversal_registers path=results/experiments/occupancy-launch-model/stats.json -->84<!-- /calkit value --> registers, the real node function reads <!-- calkit value key=frames.node_function_registers path=results/experiments/occupancy-launch-model/stats.json -->112<!-- /calkit value -->, and CloudMicrophysics' substep-averaged bulk tendency with nothing of ClimaAtmos around it reads <!-- calkit value key=frames.bmt_registers path=results/experiments/occupancy-launch-model/stats.json -->102<!-- /calkit value -->.
ClimaAtmos's entire reconstruction is therefore worth <!-- calkit value key=frames.climaatmos_registers path=results/experiments/occupancy-launch-model/stats.json -->10<!-- /calkit value --> registers on top of the package it calls, and the next step up needs <!-- calkit value key=frames.step_to_gain_registers path=results/experiments/occupancy-launch-model/stats.json -->96<!-- /calkit value --> or fewer, which the bulk tendency misses on its own by <!-- calkit value key=frames.bmt_over_step path=results/experiments/occupancy-launch-model/stats.json -->6<!-- /calkit value -->.
So the bulk tendency by itself already runs at <!-- calkit value key=frames.bmt_warps_per_sm path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value --> warps per multiprocessor --- the same <!-- calkit value key=frames.kernel_warps_per_sm_profiled path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value --> the whole kernel gets today at <!-- calkit value key=attempt.profiled_target_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value --> registers --- and a ClimaAtmos-side treatment that removed every register it owns would land in the occupancy band it started in.
That is what retires the prediction above for this kernel: the saving is real arithmetic, but there is no arrangement of this stack that collects it.

**The census closes the occupancy lever, and the rule used to read it was wrong --- the frames add rather than max.** The paragraph above argues that an `@noinline` barrier makes a kernel's register count the largest frame on its call graph, so the largest frame is the floor of every arrangement.
The census's own numbers refute it: the largest single frame is <!-- calkit value key=frames.largest_single_frame path=results/experiments/occupancy-launch-model/stats.json -->112<!-- /calkit value --> registers and the whole kernel compiles at <!-- calkit value key=frames.whole_kernel_registers path=results/experiments/occupancy-launch-model/stats.json -->130<!-- /calkit value -->, which is <!-- calkit value key=frames.kernel_above_largest_frame path=results/experiments/occupancy-launch-model/stats.json -->18<!-- /calkit value --> above it.
The reason is that a value the traversal holds live *across* the call cannot sit in a register the callee is using, so the kernel pays the caller's live-across-call state on top of the callee's frame instead of the larger of the two --- and the eight-node rule is exactly such a value, held live across every one of the calls it drives.
What that changes is narrow but useful. The floor a ClimaAtmos-side treatment could reach is the callee's frame, <!-- calkit value key=frames.floor_registers_profile_scale path=results/experiments/occupancy-launch-model/stats.json -->100<!-- /calkit value --> registers once the probe's offset of <!-- calkit value key=frames.probe_error_registers path=results/experiments/occupancy-launch-model/stats.json -->2<!-- /calkit value --> is taken out, which leaves <!-- calkit value key=frames.recoverable_registers path=results/experiments/occupancy-launch-model/stats.json -->28<!-- /calkit value --> registers genuinely reachable rather than none.
It changes nothing about occupancy, because that floor and the profiled count sit in the same band, which is checked here rather than asserted.
It changes the headroom, which is the thing the section below shows this kernel has none of: shedding caller-live state would buy up to <!-- calkit value key=frames.headroom_recovery_would_buy path=results/experiments/occupancy-launch-model/stats.json -->28<!-- /calkit value --> registers of room inside the band, and room inside the band is precisely what an edit that costs registers needs and cannot currently find.

**The heaviest target has zero register headroom, which this document should have said the first time it drew the step table.** It compiles to <!-- calkit value key=headroom.heaviest_target_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value --> registers and its band ends at <!-- calkit value key=headroom.band_max_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value -->, so its headroom is <!-- calkit value key=headroom.registers path=results/experiments/occupancy-launch-model/stats.json -->0<!-- /calkit value --> registers: one more and it falls from <!-- calkit value key=headroom.warps_in_band path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value --> warps per multiprocessor to <!-- calkit value key=headroom.warps_one_band_down path=results/experiments/occupancy-launch-model/stats.json -->12<!-- /calkit value -->, losing <!-- calkit value key=headroom.warps_lost_at_edge path=results/experiments/occupancy-launch-model/stats.json -->4<!-- /calkit value -->.
The table above gives the bands and the section on targets gives the counts, and the conjunction --- that the largest of them sits exactly on an edge --- was never written down.
It is the most consequential fact in this document, because it inverts the advice: for a kernel at a band edge the first question about any edit is what it does to the register count, before anything about what the edit is for.

**The edge is a wall rather than a detour: no block size can recover a band once the register count has left it.** The step table lists a different block size for every band, which invites the obvious escape --- a kernel that cannot fit <!-- calkit value key=granule.edge_warps path=results/experiments/occupancy-launch-model/stats.json -->16<!-- /calkit value --> warps as one block of <!-- calkit value key=model.current_band.block_size path=results/experiments/occupancy-launch-model/stats.json -->512<!-- /calkit value --> threads at <!-- calkit value key=granule.over_edge_registers path=results/experiments/occupancy-launch-model/stats.json -->130<!-- /calkit value --> registers might fit more total threads as several smaller blocks --- and it does not work.
An A100 hands its register file out per scheduler in granules of <!-- calkit value key=granule.regs_per_allocation path=results/experiments/occupancy-launch-model/stats.json -->256<!-- /calkit value --> registers, <!-- calkit value key=granule.regs_per_scheduler path=results/experiments/occupancy-launch-model/stats.json -->16384<!-- /calkit value --> per scheduler across <!-- calkit value key=granule.schedulers_per_sm path=results/experiments/occupancy-launch-model/stats.json -->4<!-- /calkit value --> of them, and a warp's allocation is rounded up to a whole granule before a scheduler counts whether it fits.
So the ceiling is a property of the register count alone, and sweeping every legal block size at all <!-- calkit value key=granule.checked_register_counts path=results/experiments/occupancy-launch-model/stats.json -->256<!-- /calkit value --> register counts confirms it: none of them ever beats the block ClimaCore derives, and at <!-- calkit value key=granule.over_edge_registers path=results/experiments/occupancy-launch-model/stats.json -->130<!-- /calkit value --> registers the best any block size achieves is <!-- calkit value key=granule.over_edge_best_warps_any_block path=results/experiments/occupancy-launch-model/stats.json -->12<!-- /calkit value --> warps per multiprocessor, the same <!-- calkit value key=granule.over_edge_warps path=results/experiments/occupancy-launch-model/stats.json -->12<!-- /calkit value --> the derived launch gets.
The practical consequence is that the only register counts worth asking `ptxas` for are the granule boundaries, which on this device are <!-- calkit value key=granule.useful_caps path=results/experiments/occupancy-launch-model/stats.json -->[32, 40, 48, 56, 64, 72, 80, 96, 128]<!-- /calkit value --> --- a cap set anywhere between two of them spends registers and buys no warps.

## What this says about kernel fusion

ClimaAtmos #4261 proposes finding more kernel fusion, and a fusion was in fact carried out upstream between 2026-10-30 and 2026-11-03 --- ClimaCore #2710 and #2713, and ClimaAtmos #4934, "Fuse column loops and eliminate temporary global caches".
Launch count fell <!-- calkit value key=fusion_ledger.launches_change_pct path=results/experiments/occupancy-launch-model/stats.json -->-34.1<!-- /calkit value -->% while device time rose <!-- calkit value key=fusion_ledger.window_grew_pct path=results/experiments/occupancy-launch-model/stats.json -->36.36<!-- /calkit value -->%, and this document's own register model says why that is not the paradox it looks like.

**Fusion adds live state to the kernel that absorbs the work, and most of this model has nowhere to put it.** The census over all <!-- calkit value key=headroom_census.kernel_variants path=results/experiments/occupancy-launch-model/stats.json -->458<!-- /calkit value --> launch variants measures each kernel's headroom --- the registers it can gain before crossing a granule boundary --- and finds <!-- calkit value key=headroom_census.zero_headroom_pct path=results/experiments/occupancy-launch-model/stats.json -->34.5<!-- /calkit value -->% of device time at exactly zero and <!-- calkit value key=headroom_census.at_most_one_register_pct path=results/experiments/occupancy-launch-model/stats.json -->58.4<!-- /calkit value -->% with at most one register.
That is not bad luck, it is what the allocator does: `ptxas` spends the granule it has already been charged for, so register counts cluster *on* the boundaries rather than between them, and the kernels with real room --- <!-- calkit value key=headroom_census.roomy_pct path=results/experiments/occupancy-launch-model/stats.json -->32.4<!-- /calkit value -->% of device time at four registers or more --- are the minority.
So the precondition for a fusion to pay is that the fused kernel stays inside its granule, which is one compile to check rather than a profile, since ClimaCore's own launch record reports the register count of the kernel the model launches.

**Measured, that fusion was about break-even rather than a loss, and the regression it was blamed for was mostly something else.** It removed <!-- calkit value key=fusion_ledger.broadcast_machinery_saved_ms path=results/experiments/occupancy-launch-model/stats.json -->3164.3<!-- /calkit value --> ms of broadcast machinery and the kernels that absorbed the work came to <!-- calkit value key=fusion_ledger.receivers_net_ms path=results/experiments/occupancy-launch-model/stats.json -->3220.6<!-- /calkit value --> ms net over survivors plus removals, which is <!-- calkit value key=fusion_ledger.paid_per_ms_saved_low path=results/experiments/occupancy-launch-model/stats.json -->1.02<!-- /calkit value --> to <!-- calkit value key=fusion_ledger.paid_per_ms_saved_high path=results/experiments/occupancy-launch-model/stats.json -->1.64<!-- /calkit value --> ms paid per ms saved --- the range being whether a new anonymous broadcast, which a line move renames, is charged to the fusion or treated as new work.
The growth in the same pull is dominated instead by <!-- calkit value key=fusion_ledger.new_named_ms path=results/experiments/occupancy-launch-model/stats.json -->6165.2<!-- /calkit value --> ms of new *named* functions, <!-- calkit value key=fusion_ledger.new_named_share_of_growth_pct path=results/experiments/occupancy-launch-model/stats.json -->66.6<!-- /calkit value -->% of the window, the implicit 1M refresh above all.
This correction matters because the figure first recorded for that pull, <!-- calkit value key=fusion_ledger.recorded_all_other_growth_per_ms_saved path=results/experiments/occupancy-launch-model/stats.json -->3.93<!-- /calkit value --> ms per ms saved, charged the fusion every other change in the window; a fusion changes the kernel inventory, which is exactly the case where a name-paired subtotal misleads, and the aggregate that survives is the net over every matching name.

**The kernel's work, unlike its registers, was reachable --- and collecting it on this kernel cost more occupancy than the work was worth.** Reaching the remaining registers means devving CloudMicrophysics, which rewrites the AMIP manifest and invalidates every profiling stage; and the node count, up to sixteen evaluations of the 1M microphysics per cell, is a physics-accuracy choice rather than an implementation detail, so the largest lever left is a science question.
This section previously stopped there, and it was wrong to: the kernel was also doing arithmetic it did not need, in the same node path whose registers had just been priced.
`docs/experiments/quadrature-saturation.md` measures the removal of two dead or duplicated saturation evaluations per node --- a field of the node table that nothing read, and a recomputation behind the barrier of the value the traversal had just handed across it --- and both are bit-for-bit rather than approximately equal.
It also measures what the headroom above implies: the removal nets two registers the wrong way, because a value handed across the barrier lives across the call, and the kernel pays four warps for them and runs slower while doing less work.
Having a register probe in hand made the work finding harder to see rather than easier --- every question asked of this kernel for two passes was "how much live state does it hold" --- and then the probe could not price the work finding either, because its register reading is off by <!-- calkit value key=attempt.probe_error_registers path=results/experiments/occupancy-launch-model/stats.json -->2<!-- /calkit value --> and the headroom it had to resolve was <!-- calkit value key=headroom.registers path=results/experiments/occupancy-launch-model/stats.json -->0<!-- /calkit value -->.
The measurement above cost a handful of compile-only jobs of a few minutes each and no profiling pair, which is the use the probe is for: `scripts/measure-quadrature-registers.jl` reads the register count out of ClimaCore's own launch record, so it measures the kernel the model launches rather than a stand-in.
It reads <!-- calkit value key=attempt.probe_target_registers path=results/experiments/occupancy-launch-model/stats.json -->130<!-- /calkit value --> registers where the profile reads <!-- calkit value key=attempt.profiled_target_registers path=results/experiments/occupancy-launch-model/stats.json -->128<!-- /calkit value -->, a difference of <!-- calkit value key=attempt.probe_error_registers path=results/experiments/occupancy-launch-model/stats.json -->2<!-- /calkit value -->, so it is trusted for differences between arms and not for an absolute count.

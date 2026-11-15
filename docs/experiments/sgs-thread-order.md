# A level-aligned thread order on the SGS closure

`cloud-mask-census` found an early-out that is paid for and never collected.
`_sum_1m_over_sgs` returns a zero tendency at <!-- calkit value key=guard.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% of environment cells, and under the order the launch uses — the parent layout's own, vertical fastest, so a warp is 32 levels of one column — those cells fill a warp <!-- calkit value key=live.vertical_fastest.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->0.31<!-- /calkit value -->% of the time, because precipitation threads down a column and a 32-level run almost always contains one cloudy cell.
Reordered so that a warp is two spectral elements at one level, the same cells fill <!-- calkit value key=live.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->10.26<!-- /calkit value -->% of warps and <!-- calkit value key=live.level_aligned.block_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->7.80<!-- /calkit value -->% of <!-- calkit value key=block_width path=results/docs/cloud-mask-census.json -->128<!-- /calkit value -->-thread blocks, which is <!-- calkit value key=reorder_share_of_compaction_pct path=results/docs/cloud-mask-census.json -->44<!-- /calkit value -->% of what a perfect gather of them could reach.

This experiment applies that order and measures what it is worth.

## The treatment

`DataLayouts.LevelAlignedOrder` is a `DataMask` that marks every point active, as `NoMask` does, and changes only which points consecutive thread ranks take: it presents the cell grid in `(i, j, h, v)` order while indexing the layout in `(v, i, j, h)`.
It is a `DataMask` rather than a new argument because `copyto_capped!` already takes its index set from `maskable_slice_indices(mask, view, args...)`, which is the hook `IJHMask` uses to hand the loop a compacted list rather than a filter — so a thread order needs no signature change anywhere, and it composes with a real mask later.
`set_environment_1m_tendency!` takes the order as a trailing argument and defaults to the level-aligned one, with a guard that refuses it on a space that carries a mask of its own, since the one `copyto_capped!` argument cannot hold both.

The order is why this is cheap, and the reason it is allowed is measured rather than assumed: §2a puts this kernel at 4.7% of DRAM peak, 7% of L2 and 3.3% of L1 against an occupancy ceiling of 12.4%, so the strided reads a level-aligned order implies are spending the one resource the kernel has to spare.

## Why it must be bit-for-bit, and why that is checked rather than argued

The two orders evaluate the same function at the same cells, write one value per cell, and reduce nothing between cells, so every cell's result is independent of which warp computed it.
That makes the treatment exact, which is worth more than it sounds: the alternative route into this kernel — widening the skip criterion past the mixed-phase band — gives up a tail weight and needs an accuracy case, and learnings §4e records a quadrature collapse that looked safe at 10σ and moved `dq_icl_dt` by 23.8% of field RMS.

So the probe writes each order into its own destination field and compares the two as integers, on the fields' own storage, so that a NaN cannot pass as equal and padding is compared too.

## Both paths, because the guard is live on both

The closure runs eight times a step: twice on the explicit path and six times on the implicit one.
`sgs-substep-cost` could price only the explicit calls, because its handle was the substep count and `BMT.Instantaneous` has no substep loop to fit, which is why its compaction ceiling was stated as a floor across the closure.
The early-out the order collects is in `_sum_1m_over_sgs` ahead of the node rule and is reached whichever mode is passed — `mode` binds the positional argument before `args...`, so `isempty(args)` holds on both paths — so the implicit calls are not a control here but a second site to decide, and timing them needs no fit.

They are decided the other way, and that is the result that makes the order a per-call-site choice rather than a default.
An `Instantaneous` call costs a fifth of a `LinearizedAverage` one, so the same point body is a far smaller share of its own kernel: the strided reads a level-aligned order implies are charged against a kernel with much less arithmetic to hide them behind, and §2a's reading of 4.7% of DRAM peak was taken on the heavy one.
So both sites are timed and the order is asked for only where it is a resolved gain, which is why the probe also reports what a blanket default would have cost.

```sh calkit stage name=thread-order environment=clima outputs=[{path: results/experiments/sgs-thread-order/order.toml, storage: git}] inputs=[scripts/measure-thread-order.jl, scripts/run-julia-script.sh, ClimaAtmos.jl-mod/src, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=90]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-thread-order.jl \
  --out results/experiments/sgs-thread-order/order.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/sgs-thread-order.json, storage: git}] inputs=[results/experiments/sgs-thread-order/order.toml, results/top-kernels.csv, results/null-census.json, results/docs/cloud-mask-census.json]
import json
import statistics
import tomllib

d = tomllib.load(open("results/experiments/sgs-thread-order/order.toml", "rb"))
nulls = json.load(open("results/null-census.json"))

# The device budget and the profiled closure kernel, from one row of the
# profile so the denominator is the profile's own rather than a sum over a
# truncated table.
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
kernel_ms = float(row["total_ms_mod"])
kernel_pct = float(row["pct_of_gpu_time_mod"])
kernel_launches = int(float(row["invocations_mod"]))
device_ms = 100.0 * kernel_ms / kernel_pct

ARMS = ["explicit"] + (["implicit"] if d["has_implicit_path"] else [])


def arm(name):
    a = d[name]
    ds = a["order_effect"]["production"]["differences_ms"]
    third = len(ds) // 3
    return {
        "calls_per_step": a["calls_per_step"],
        "vertical_ms": round(a["vertical_ms"], 2),
        "level_aligned_ms": round(a["level_aligned_ms"], 2),
        "saved_ms_per_call": round(a["saved_ms_per_call"], 3),
        "saved_pct": round(a["saved_pct"], 2),
        # The range of the per-round differences, which is the error bar: a
        # per-block spread is set by the device drift the rounds cancel.
        "saved_min_ms": round(a["order_effect"]["production"]["min_ms"], 3),
        "saved_max_ms": round(a["order_effect"]["production"]["max_ms"], 3),
        "resolved": a["resolved"],
        "helps": a["saved_ms_per_call"] > 0 and a["resolved"],
        # The resolution test: how many of the rounds put the difference the
        # same way, and the two-sided sign-test probability of that.
        "rounds_agreeing": a["order_effect"]["production"]["rounds_agreeing"],
        "rounds_compared": a["order_effect"]["production"]["rounds_compared"],
        "sign_test_p": a["order_effect"]["production"]["sign_test_p"],
        # Does the effect hold still across the run? The device drifts, so an
        # effect whose magnitude grows over the rounds is pinned to a sign
        # rather than to a size, however many rounds agree on the sign.
        "saved_first_third_pct": round(
            100.0 * statistics.median(ds[:third]) / a["vertical_ms"], 2
        ),
        "saved_last_third_pct": round(
            100.0 * statistics.median(ds[-third:]) / a["vertical_ms"], 2
        ),
        "window_saved_ms": round(a["window_saved_ms"], 1),
        "device_pct": round(100.0 * a["window_saved_ms"] / device_ms, 3),
        # The same effect read at the other destination, which is what says the
        # order and not the field is what moved.
        "saved_pct_fresh": round(a["order_effect"]["fresh"]["pct"], 2),
        # And what the probe's own choice of field is worth at the order the
        # model uses today, which a single-block design reads as a treatment.
        "destination_pct": round(a["destination_effect"]["vertical"]["pct"], 2),
        "destination_rounds_agreeing": a["destination_effect"]["vertical"][
            "rounds_agreeing"
        ],
        "destination_sign_test_p": a["destination_effect"]["vertical"]["sign_test_p"],
        "destination_resolved": a["destination_effect"]["vertical"]["resolved"],
        "bitwise_identical": a["bitwise_identical"],
        "values_compared": a["values_compared"],
        "values_differing": a["values_differing"],
        "worst_abs_difference": a["worst_abs_difference"],
    }


arms = {name: arm(name) for name in ARMS}
closure_ms = sum(d[name]["vertical_ms"] * d[name]["calls_per_step"] * d["steps"] for name in ARMS)

# What ships: the order is asked for at the call sites where it is a resolved
# gain and not at the others, so the device figure is the sum over those rather
# than over the closure. The blanket figure is carried beside it, because a
# default applied everywhere is the obvious thing to do and is worse.
helped = [name for name in ARMS if arms[name]["helps"]]
hurt = [name for name in ARMS if not arms[name]["helps"]]
shipped_ms = sum(arms[name]["window_saved_ms"] for name in helped)
blanket_ms = sum(arms[name]["window_saved_ms"] for name in ARMS)

out = {
    "arms": arms,
    "steps": d["steps"],
    "rounds": d["rounds"],
    "orders": d["orders"],
    "destinations": d["destinations"],
    "bitwise_identical": all(arms[name]["bitwise_identical"] for name in ARMS),
    "all_resolved": all(arms[name]["resolved"] for name in ARMS),
    "helped_arms": len(helped),
    "hurt_arms": len(hurt),
    "shipped_window_ms": round(shipped_ms, 1),
    "shipped_device_pct": round(100.0 * shipped_ms / device_ms, 3),
    "shipped_closure_pct": round(100.0 * shipped_ms / closure_ms, 2),
    "blanket_window_ms": round(blanket_ms, 1),
    "blanket_device_pct": round(100.0 * blanket_ms / device_ms, 3),
    # What the blanket default gives up against the targeted one, which is the
    # cost of not measuring the second call site.
    "blanket_cost_pct": round(100.0 * (shipped_ms - blanket_ms) / device_ms, 3),
    "device": {
        "device_ms": round(device_ms, 1),
        "closure_ms": round(closure_ms, 1),
        "closure_pct": round(100.0 * closure_ms / device_ms, 2),
    },
    # Does the probe describe the kernel the pipeline profiles? The profile
    # charges one row for all eight calls over the window, so reconstruct it
    # from the per-call times and the launch counts and compare.
    "profile": {
        "kernel_ms": round(kernel_ms, 1),
        "kernel_pct_of_device": kernel_pct,
        "calls_per_step": kernel_launches // d["steps"],
        "reconstructed_ms": round(closure_ms, 1),
        "reconstruction_residual_pct": round(100.0 * (closure_ms - kernel_ms) / kernel_ms, 2),
    },
    # A device-time change is measured on the device and needs no SYPD floor to
    # be believed, but whether an AMIP pair could confirm it is a separate
    # question and the null census is what answers it.
    "null_floor_pct": nulls["current_window_mean_pct"],
    "below_null_floor": 100.0 * shipped_ms / device_ms < nulls["current_window_mean_pct"],
}
json.dump(out, open("results/docs/sgs-thread-order.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/sgs-thread-order.json -->

## Result

The explicit call takes <!-- calkit value key=arms.explicit.vertical_ms -->16.32<!-- /calkit value --> ms under the order the launch takes unasked and <!-- calkit value key=arms.explicit.level_aligned_ms -->13.89<!-- /calkit value --> ms under the level-aligned one, a saving of <!-- calkit value key=arms.explicit.saved_ms_per_call -->2.429<!-- /calkit value --> ms per call, or <!-- calkit value key=arms.explicit.saved_pct -->14.88<!-- /calkit value -->%.
All <!-- calkit value key=arms.explicit.rounds_agreeing -->31<!-- /calkit value --> of <!-- calkit value key=arms.explicit.rounds_compared -->31<!-- /calkit value --> rounds put the difference the same way, which a two-sided sign test prices at p = <!-- calkit value key=arms.explicit.sign_test_p format="{:.1e}" -->9.3e-10<!-- /calkit value -->, and the second destination field puts the same effect at <!-- calkit value key=arms.explicit.saved_pct_fresh -->15.03<!-- /calkit value -->%.
The implicit call moves the other way by more: <!-- calkit value key=arms.implicit.vertical_ms -->3.63<!-- /calkit value --> ms becomes <!-- calkit value key=arms.implicit.level_aligned_ms -->4.33<!-- /calkit value --> ms, <!-- calkit value key=arms.implicit.saved_pct -->-18.47<!-- /calkit value -->% per call, again every round of <!-- calkit value key=arms.implicit.rounds_compared -->31<!-- /calkit value --> and again reproduced at the other destination at <!-- calkit value key=arms.implicit.saved_pct_fresh -->-18.25<!-- /calkit value -->%.
So the thing the design was built around holds, and holds at a size worth targeting rather than at a rounding error: the order is a gain on the heavy call and a loss on the light one.

The treatment is exact, which it had to be.
On each path, <!-- calkit value key=arms.explicit.values_compared format="{:,}" -->7,766,016<!-- /calkit value --> values were compared as integers on the field's own storage, with <!-- calkit value key=arms.explicit.values_differing -->0<!-- /calkit value --> differing and a worst absolute difference of <!-- calkit value key=arms.explicit.worst_abs_difference -->0.0<!-- /calkit value -->.

What ships is the order at the explicit site and nothing at the implicit one, which is <!-- calkit value key=shipped_window_ms -->582.9<!-- /calkit value --> ms of the <!-- calkit value key=steps -->120<!-- /calkit value -->-step window — <!-- calkit value key=shipped_device_pct -->1.75<!-- /calkit value -->% of the <!-- calkit value key=device.device_ms format="{:,}" -->33,306.1<!-- /calkit value --> ms device budget, and <!-- calkit value key=shipped_closure_pct -->8.93<!-- /calkit value -->% of the closure's own <!-- calkit value key=device.closure_ms format="{:,}" -->6,527.9<!-- /calkit value --> ms.
Applied everywhere instead, which is the obvious thing to do with a default and is what a probe that timed only the explicit path would have recommended, it comes to <!-- calkit value key=blanket_device_pct -->0.302<!-- /calkit value -->%.
So measuring the second call site is worth <!-- calkit value key=blanket_cost_pct -->1.448<!-- /calkit value -->% of device, which is several times the blanket figure it replaces.

The probe describes the kernel the pipeline profiles, which is worth checking rather than assuming, because the profile charges one row for all eight calls.
Reconstructing that row from the per-call times and the launch counts gives <!-- calkit value key=profile.reconstructed_ms format="{:,}" -->6,527.9<!-- /calkit value --> ms against <!-- calkit value key=profile.kernel_ms format="{:,}" -->6,550.0<!-- /calkit value --> ms measured, a residual of <!-- calkit value key=profile.reconstruction_residual_pct -->-0.34<!-- /calkit value -->%, and the row's <!-- calkit value key=profile.calls_per_step -->8<!-- /calkit value --> calls per step match the <!-- calkit value key=arms.explicit.calls_per_step -->2<!-- /calkit value --> + <!-- calkit value key=arms.implicit.calls_per_step -->6<!-- /calkit value --> the probe times.

Two of the four effects reported here are the ones the earlier designs got wrong, and they are now unresolved rather than large.
Timing each configuration in its own block had made the probe's choice of destination field look worth about a fifth of the call on one arm and a fifth the other way on the other arm — opposite signs for one factor on two arms of the same probe, which is what a device drift read as an effect looks like.
Interleaved, that factor is <!-- calkit value key=arms.explicit.destination_pct -->0.11<!-- /calkit value -->% on the explicit arm and <!-- calkit value key=arms.implicit.destination_pct -->0.25<!-- /calkit value -->% on the implicit one, and neither is resolved: <!-- calkit value key=arms.explicit.destination_rounds_agreeing -->19<!-- /calkit value --> and <!-- calkit value key=arms.implicit.destination_rounds_agreeing -->18<!-- /calkit value --> rounds of <!-- calkit value key=rounds -->31<!-- /calkit value --> agree on a sign, at p = <!-- calkit value key=arms.explicit.destination_sign_test_p format="{:.2f}" -->0.28<!-- /calkit value --> and p = <!-- calkit value key=arms.implicit.destination_sign_test_p format="{:.2f}" -->0.47<!-- /calkit value -->.

One figure here is pinned to its sign and not to its size.
The explicit arm's effect is flat across the run, <!-- calkit value key=arms.explicit.saved_first_third_pct -->15.03<!-- /calkit value -->% over the first third of the rounds against <!-- calkit value key=arms.explicit.saved_last_third_pct -->15.01<!-- /calkit value -->% over the last.
The implicit arm's grows, from <!-- calkit value key=arms.implicit.saved_first_third_pct -->-11.17<!-- /calkit value -->% to <!-- calkit value key=arms.implicit.saved_last_third_pct -->-19.22<!-- /calkit value -->%, which is why a shorter pass over the same code reported a smaller regression.
Every round agreeing establishes that the light call loses under the order; it does not establish by how much, and the blanket cost above inherits that looseness.

## Verdict

### The thread order is a per-call-site argument, not a default

The same reorder, of the same function, over the same cells, is worth <!-- calkit value key=arms.explicit.saved_pct -->14.88<!-- /calkit value -->% at one call site and <!-- calkit value key=arms.implicit.saved_pct -->-18.47<!-- /calkit value -->% at another, both resolved by every round of <!-- calkit value key=rounds -->31<!-- /calkit value -->.
Nothing about the field changed between those two statements; what changed is how much arithmetic the kernel had to hide the strided reads behind, and §2a's reading of a mostly-idle memory system was taken on the <!-- calkit value key=arms.explicit.vertical_ms -->16.32<!-- /calkit value --> ms call, not the <!-- calkit value key=arms.implicit.vertical_ms -->3.63<!-- /calkit value --> ms one.
A reorder whose whole justification is spare bandwidth is only justified where the bandwidth is spare, so the order belongs in the call, which is where `set_environment_1m_tendency!` now takes it.

That is also the general form of the finding, and it is cheap to get wrong in the other direction.
`LevelAlignedOrder` is a `DataMask`, so any ClimaCore pointwise loop can ask for it, and the temptation is to make it the layout's order or a space's property.
Measured per site, it pays <!-- calkit value key=shipped_device_pct -->1.75<!-- /calkit value -->% of device here and would have paid <!-- calkit value key=blanket_device_pct -->0.302<!-- /calkit value -->% as a default.

### Below the null floor is not the same as unmeasured

What ships is <!-- calkit value key=shipped_device_pct -->1.75<!-- /calkit value -->% of device, which is below the <!-- calkit value key=null_floor_pct -->1.869<!-- /calkit value -->% the current window's null census reports, so `below_null_floor` is <!-- calkit value key=below_null_floor -->true<!-- /calkit value -->.
That does not weaken this measurement, because it is not a SYPD difference: it is device time taken in-block with CUDA events over <!-- calkit value key=rounds -->31<!-- /calkit value --> interleaved rounds on one simulation state, where the quantity and the noise are both visible.
What it says is that an AMIP pair would not confirm it — the pair's own arms wander further than the treatment moves — which is worth knowing before spending the reservations to ask.

### What is left in this kernel

The closure is <!-- calkit value key=device.closure_pct -->19.6<!-- /calkit value -->% of device and the order collects <!-- calkit value key=shipped_closure_pct -->8.93<!-- /calkit value -->% of it, so the kernel is still the largest single item in the budget after this pass.
`cloud-mask-census` says where the rest of that early-out went: the guard fires at <!-- calkit value key=guard.cell_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->22.91<!-- /calkit value -->% of cells, the level-aligned order gathers them into <!-- calkit value key=live.level_aligned.warp_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->10.26<!-- /calkit value -->% of warps, and a perfect gather would reach <!-- calkit value key=live.compaction_saving_pct format="{:.1f}" path=results/docs/cloud-mask-census.json -->22.7<!-- /calkit value -->%, so this treatment collects about <!-- calkit value key=shipped_closure_pct -->8.93<!-- /calkit value -->% of the closure out of a ceiling nearer twice that.
The remaining lever is the criterion rather than the layout — testing a dry cell's excess against ice saturation at its coldest node instead of against the mixed-phase band.
`cloud-mask-census` prices it at <!-- calkit value key=widening_floor_pct format="{:.2f}" path=results/docs/cloud-mask-census.json -->1.25<!-- /calkit value -->% of device, and it gives up a tail weight, so unlike this pass it needs an accuracy case before it needs a timing.

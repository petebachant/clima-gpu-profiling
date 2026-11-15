# A level-aligned thread order on the SGS closure

`cloud-mask-census` found an early-out that is paid for and never collected.
`_sum_1m_over_sgs` returns a zero tendency at 22.91% of environment cells, and under the order the launch uses — the parent layout's own, vertical fastest, so a warp is 32 levels of one column — those cells fill a warp 0.31% of the time, because precipitation threads down a column and a 32-level run almost always contains one cloudy cell.
Reordered so that a warp is two spectral elements at one level, the same cells fill 10.26% of warps and 7.80% of 128-thread blocks, which is 44% of what a perfect gather of them could reach.

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
The early-out the order collects is in `_sum_1m_over_sgs` ahead of the node rule and is reached whichever mode is passed — `mode` binds the positional argument before `args...`, so `isempty(args)` holds on both paths — so the implicit path is not a control here but the larger half of the prize, and timing it needs no fit.

```sh calkit stage name=thread-order environment=clima outputs=[{path: results/experiments/sgs-thread-order/order.toml, storage: git}] inputs=[scripts/measure-thread-order.jl, scripts/run-julia-script.sh, ClimaAtmos.jl-mod/src, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=90]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-thread-order.jl \
  --out results/experiments/sgs-thread-order/order.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/sgs-thread-order.json, storage: git}] inputs=[results/experiments/sgs-thread-order/order.toml, results/top-kernels.csv, results/null-census.json]
import json
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
    pts = a["points"]
    return {
        "calls_per_step": a["calls_per_step"],
        "vertical_ms": round(pts["vertical"]["ms"], 2),
        "level_aligned_ms": round(pts["level_aligned"]["ms"], 2),
        "saved_ms_per_call": round(a["saved_ms_per_call"], 3),
        "saved_pct": round(a["saved_pct"], 2),
        "repeat_spread_ms": round(a["repeat_spread_ms"], 3),
        "resolved": a["resolved"],
        "window_saved_ms": round(a["window_saved_ms"], 1),
        "device_pct": round(100.0 * a["window_saved_ms"] / device_ms, 3),
        "bitwise_identical": a["bitwise_identical"],
        "values_compared": a["values_compared"],
        "values_differing": a["values_differing"],
        "worst_abs_difference": a["worst_abs_difference"],
    }


arms = {name: arm(name) for name in ARMS}
window_ms = sum(arms[name]["window_saved_ms"] for name in ARMS)
closure_ms = sum(
    d[name]["points"]["vertical"]["ms"] * d[name]["calls_per_step"] * d["steps"]
    for name in ARMS
)

# Does the probe describe the kernel the pipeline profiles? The profile charges
# one row for all eight calls over the window, so reconstruct it from the
# per-call times and the launch counts and compare. A large residual means the
# probe is timing something other than what the pipeline measured.
out = {
    "arms": arms,
    "steps": d["steps"],
    "orders": d["orders"],
    "bitwise_identical": all(arms[name]["bitwise_identical"] for name in ARMS),
    "all_resolved": all(arms[name]["resolved"] for name in ARMS),
    "window_saved_ms": round(window_ms, 1),
    "device_pct": round(100.0 * window_ms / device_ms, 3),
    "closure_saved_pct": round(100.0 * window_ms / closure_ms, 2),
    "device": {
        "device_ms": round(device_ms, 1),
        "closure_ms": round(closure_ms, 1),
        "closure_pct": round(100.0 * closure_ms / device_ms, 2),
    },
    "profile": {
        "kernel_ms": round(kernel_ms, 1),
        "kernel_pct_of_device": kernel_pct,
        "calls_per_step": kernel_launches // d["steps"],
        "reconstructed_ms": round(closure_ms, 1),
        "reconstruction_residual_pct": round(
            100.0 * (closure_ms - kernel_ms) / kernel_ms, 2
        ),
    },
    # A device-time change is measured on the device and needs no SYPD floor to
    # be believed, but whether an AMIP pair could confirm it is a separate
    # question and the null census is what answers it.
    "null_floor_pct": nulls["current_window_mean_pct"],
    "below_null_floor": 100.0 * window_ms / device_ms < nulls["current_window_mean_pct"],
}
json.dump(out, open("results/docs/sgs-thread-order.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/sgs-thread-order.json -->

## Result

To be written against the measurement.

## Verdict

To be written against the measurement.

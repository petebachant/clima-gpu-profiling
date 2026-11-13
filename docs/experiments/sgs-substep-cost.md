# What one CloudMicrophysics evaluation costs inside the SGS closure

The SGS 1M closure is the largest single item in the device budget, and every proposal for cutting it turns on one unmeasured quantity: how its time splits between the quadrature machinery and the CloudMicrophysics bulk-tendency calls inside it.
This experiment measures that split directly, by timing the live environment tendency call at several values of `n_substeps_quad` and fitting a line, and it measures the warp divergence the field actually has rather than inferring it from a fraction of points.

## Why the substep count is the handle

`docs/learnings.md` §4r put the CM call at about 3.7% of this kernel.
That figure came from skipping the call wherever the clear-air guard fires, which was 88.31% of points, and observing the kernel get 3.27% faster.
The inference from "skipping it at 88% of points buys 3%" to "the call is 4% of the cost" needs the per-point cost to be uniform, and it is not.
The guard fires at clear points, where the call returns early and is cheap; and under SIMT a warp runs the cloudy path if any one of its 32 lanes is cloudy, so at an 88% clear rate almost every warp pays it regardless.
The measurement bounds what *skipping* buys. It does not bound what the call *costs*, and §4r's conclusion that the kernel is not bound by microphysics rests on the second reading.

`n_substeps_quad` separates the two without touching any source.
CloudMicrophysics runs `_substep_1m` exactly `nsub` times per quadrature node in `LinearizedAverage` mode and nothing else in the kernel reads the count, so the time against `nsub` is a line whose slope is one CM evaluation over the whole field — paid under the real divergence — and whose intercept is everything else: the conditional-moment rule, the per-node saturation calls, the column loop that materializes the environment, and the data movement.

The implicit path is the control.
It runs `BMT.Instantaneous()`, which has no substep loop at all, so its slope against `nsub` must be flat — and it is 6 of the 8 calls the closure makes per step.

## Why this is an in-block comparison and not a difference of runs

Every point is taken in one process against one settled AMIP state, calling the real `set_environment_1m_tendency!` with a different substep count.
That function already takes the count as an argument, so these are the call sites' own arguments rather than a re-implementation of them, and two points differ by the substep count and by nothing else.
The alternative — running the model once per setting and differencing — cannot resolve the effect on this machine: one arm of this stack wandered 7.73% between two runs of identical code.

```sh calkit stage name=sgs-substep-sweep environment=clima outputs=[{path: results/experiments/sgs-substep-cost/substeps.toml, storage: git}] inputs=[scripts/measure-sgs-substeps.jl, scripts/run-julia-script.sh, ClimaAtmos.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=90]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-sgs-substeps.jl \
  --out results/experiments/sgs-substep-cost/substeps.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

`ClimaAtmos.jl-mod/src` is an input here, unlike in the g-point sweeps, because the closure it measures is ClimaAtmos's own code and an edit there moves the answer.
That makes this a GPU stage that any microphysics edit restales, which is the reason to tag the experiment and freeze it once it has run.

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/sgs-substep-cost.json, storage: git}] inputs=[results/experiments/sgs-substep-cost/substeps.toml, results/top-kernels.csv]
import json
import tomllib

d = tomllib.load(open("results/experiments/sgs-substep-cost/substeps.toml", "rb"))
expl = d["explicit"]

# The profiled kernel this probe times, and the device budget it is a share
# of. Both come from one row, so the budget is the profile's own denominator
# rather than a sum taken over a truncated table.
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
kernel_ms = float(row["total_ms_baseline"])
kernel_pct = float(row["pct_of_gpu_time_baseline"])
kernel_launches = int(float(row["invocations_baseline"]))
device_ms = 100.0 * kernel_ms / kernel_pct


def arm(a):
    pts = a["points"]
    return {
        "in_use": a["n_substeps_quad_in_use"],
        "ms_in_use": round(a["ms_in_use"], 1),
        "per_substep_ms": round(a["per_substep_ms"], 1),
        "fixed_ms": round(a["fixed_ms"], 1),
        "fit_r2": round(a["fit_r2"], 4),
        "substep_share_pct": round(a["substep_share_pct"], 1),
        "saving_one_fewer_substep_pct": round(a["saving_one_fewer_substep_pct"], 1),
        "repeat_spread_pct": round(a["repeat_spread_pct"], 3),
        "slope_resolved": a["slope_resolved"],
        "points": {k: round(v["ms"], 1) for k, v in sorted(pts.items())},
    }


clear_pct = 100.0 * d["condensate_free_fraction"]
warp_pct = 100.0 * d["fully_clear_warp_fraction"]
out = {
    "explicit": arm(expl),
    "divergence": {
        "points": d["points"],
        "condensate_free_pct": round(clear_pct, 2),
        "fully_clear_warp_pct": round(warp_pct, 2),
        # What the naive reading would have said, had the clear points been
        # spread at random rather than where they are.
        "random_order_clear_warp_pct": round(
            100.0 * d["condensate_free_fraction"] ** 32, 4
        ),
        "warps_paying_cm_pct": round(100.0 - warp_pct, 2),
        "compaction_ceiling_of_cm_pct": round(d["compaction_ceiling_of_cm_pct"], 1),
    },
    "n_substeps_quad": d["points"] and expl["n_substeps_quad_in_use"],
}
if d["has_implicit_path"]:
    out["implicit"] = arm(d["implicit"])

# The compaction ceiling in the only units that matter: a share of the kernel,
# and then of the device budget the pipeline measured.
ceiling_of_kernel = (
    expl["substep_share_pct"] * d["compaction_ceiling_of_cm_pct"] / 100.0
)
out["compaction"] = {
    "ceiling_pct_of_kernel": round(ceiling_of_kernel, 1),
    "ceiling_pct_of_device": round(ceiling_of_kernel * kernel_pct / 100.0, 2),
}

# Does the probe describe the profiled kernel? The profile charges one row for
# both paths over the whole window, so reconstruct it from the per-call times
# and the launch counts and compare. A large residual means the probe is
# timing something other than what the pipeline measured.
STEPS = 120
calls = {"explicit": 2, "implicit": 6}
per_step_ms = calls["explicit"] * expl["ms_in_use"]
if d["has_implicit_path"]:
    per_step_ms += calls["implicit"] * d["implicit"]["ms_in_use"]
reconstructed = per_step_ms * STEPS
out["profile"] = {
    "kernel_ms": round(kernel_ms, 1),
    "kernel_pct_of_device": kernel_pct,
    "kernel_launches": kernel_launches,
    "device_ms": round(device_ms, 1),
    "calls_per_step": kernel_launches // STEPS,
    "reconstructed_ms": round(reconstructed, 1),
    "reconstruction_residual_pct": round(
        100.0 * (reconstructed - kernel_ms) / kernel_ms, 1
    ),
}
json.dump(out, open("results/docs/sgs-substep-cost.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/sgs-substep-cost.json -->

## Result

Fill in once the stage has run.

## Verdict

Fill in once the stage has run.

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
# The mod columns, because the probe ran against the mod arm: the register cap
# is on this kernel, and the baseline row describes a different one.
kernel_ms = float(row["total_ms_mod"])
kernel_pct = float(row["pct_of_gpu_time_mod"])
kernel_launches = int(float(row["invocations_mod"]))
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
        "random_order_clear_warp_pct": f"{100.0 * d['condensate_free_fraction'] ** 32:.2e}",
        "warps_paying_cm_pct": round(100.0 - warp_pct, 2),
        "compaction_ceiling_of_cm_pct": round(d["compaction_ceiling_of_cm_pct"], 1),
    },
    "n_substeps_quad": d["points"] and expl["n_substeps_quad_in_use"],
}
if d["has_implicit_path"]:
    out["implicit"] = arm(d["implicit"])

# Device time, which is the only unit a treatment is decided in. The probe
# times ONE call; the explicit path makes two of them per step and the implicit
# path six, so a per-call share becomes device time through the launch counts
# and not through the profile row's percentage -- that row covers all eight.
STEPS = 120
EXPLICIT_CALLS = 2
IMPLICIT_CALLS = 6
explicit_ms = expl["ms_in_use"] * EXPLICIT_CALLS * STEPS
implicit_ms = (
    d["implicit"]["ms_in_use"] * IMPLICIT_CALLS * STEPS
    if d["has_implicit_path"]
    else 0.0
)


def pct_of_device(ms):
    return round(100.0 * ms / device_ms, 2)


# One fewer substep, over the two explicit calls a step makes.
substep_ms = expl["per_substep_ms"] * EXPLICIT_CALLS * STEPS
# What a perfect gather could take off the CM calls on the explicit path. The
# implicit path is excluded: `Instantaneous` has no substep loop, so this fit
# cannot separate its CM call from the machinery around it.
cm_ms = explicit_ms * expl["substep_share_pct"] / 100.0
compaction_ms = cm_ms * d["compaction_ceiling_of_cm_pct"] / 100.0
out["device"] = {
    "device_ms": round(device_ms, 1),
    "explicit_ms": round(explicit_ms, 1),
    "explicit_pct": pct_of_device(explicit_ms),
    "implicit_ms": round(implicit_ms, 1),
    "implicit_pct": pct_of_device(implicit_ms),
    "closure_pct": pct_of_device(explicit_ms + implicit_ms),
    "cm_calls_ms": round(cm_ms, 1),
    "cm_calls_pct": pct_of_device(cm_ms),
    "substep_saving_ms": round(substep_ms, 1),
    "substep_saving_pct": pct_of_device(substep_ms),
    "compaction_ceiling_ms": round(compaction_ms, 1),
    "compaction_ceiling_pct": pct_of_device(compaction_ms),
}
out["compaction"] = {
    "ceiling_pct_of_cm": round(d["compaction_ceiling_of_cm_pct"], 1),
    "ceiling_pct_of_explicit_call": round(
        expl["substep_share_pct"] * d["compaction_ceiling_of_cm_pct"] / 100.0, 1
    ),
}

# Does the probe describe the profiled kernel? The profile charges one row for
# both paths over the whole window, so reconstruct it from the per-call times
# and the launch counts and compare. A large residual means the probe is
# timing something other than what the pipeline measured.
reconstructed = explicit_ms + implicit_ms
out["profile"] = {
    "kernel_ms": round(kernel_ms, 1),
    "kernel_pct_of_device": kernel_pct,
    "kernel_launches": kernel_launches,
    "calls_per_step": kernel_launches // STEPS,
    "reconstructed_ms": round(reconstructed, 1),
    "reconstruction_residual_pct": round(
        100.0 * (reconstructed - kernel_ms) / kernel_ms, 2
    ),
}
json.dump(out, open("results/docs/sgs-substep-cost.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/sgs-substep-cost.json -->

## Result

The explicit call is a straight line in the substep count, at <!-- calkit value key=explicit.per_substep_ms format="{:.1f}" path=results/docs/sgs-substep-cost.json -->6.2<!-- /calkit value --> ms per substep against an intercept of <!-- calkit value key=explicit.fixed_ms format="{:.1f}" path=results/docs/sgs-substep-cost.json -->3.7<!-- /calkit value --> ms, with r2 = <!-- calkit value key=explicit.fit_r2 format="{:.4f}" path=results/docs/sgs-substep-cost.json -->0.9988<!-- /calkit value --> over the four points and a repeat spread of <!-- calkit value key=explicit.repeat_spread_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->1.48<!-- /calkit value -->%.
At the shipped `n_substeps_quad` of <!-- calkit value key=n_substeps_quad format="{:d}" path=results/docs/sgs-substep-cost.json -->2<!-- /calkit value --> that puts the CloudMicrophysics calls at <!-- calkit value key=explicit.substep_share_pct format="{:.1f}" path=results/docs/sgs-substep-cost.json -->75.9<!-- /calkit value -->% of the call, which is the quantity S4r was read as bounding at about 4%.
The implicit path is the control and comes out flat, <!-- calkit value key=implicit.per_substep_ms format="{:.3f}" path=results/docs/sgs-substep-cost.json -->-0.000<!-- /calkit value --> ms per substep against its own <!-- calkit value key=implicit.repeat_spread_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->1.38<!-- /calkit value -->% spread, exactly as `BMT.Instantaneous()` having no substep loop predicts.

The probe describes the kernel the pipeline profiles.
Reconstructing that kernel from the per-call times and the <!-- calkit value key=profile.calls_per_step format="{:d}" path=results/docs/sgs-substep-cost.json -->8<!-- /calkit value --> calls a step makes gives <!-- calkit value key=profile.reconstructed_ms format="{:,.1f}" path=results/docs/sgs-substep-cost.json -->6,512.9<!-- /calkit value --> ms against the <!-- calkit value key=profile.kernel_ms format="{:,.1f}" path=results/docs/sgs-substep-cost.json -->6,550.0<!-- /calkit value --> ms the mod arm's profile charges it, a residual of <!-- calkit value key=profile.reconstruction_residual_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->-0.57<!-- /calkit value -->%.

The divergence is the other half, and it is measured rather than inferred from a fraction of points.
Of <!-- calkit value key=divergence.points format="{:,}" path=results/docs/sgs-substep-cost.json -->1,941,504<!-- /calkit value --> environment cells, <!-- calkit value key=divergence.condensate_free_pct format="{:.1f}" path=results/docs/sgs-substep-cost.json -->43.6<!-- /calkit value -->% carry no condensate at all -- but only <!-- calkit value key=divergence.fully_clear_warp_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->2.48<!-- /calkit value -->% of warps are clear in all 32 lanes, so <!-- calkit value key=divergence.warps_paying_cm_pct format="{:.1f}" path=results/docs/sgs-substep-cost.json -->97.5<!-- /calkit value -->% of them run the cloudy path.
The clear cells do cluster, since scattering them at random would leave <!-- calkit value key=divergence.random_order_clear_warp_pct format="{}" path=results/docs/sgs-substep-cost.json -->2.82e-10<!-- /calkit value -->% of warps clear, but nowhere near enough to matter.
A perfect gather of the cloudy cells therefore removes at most <!-- calkit value key=compaction.ceiling_pct_of_cm format="{:.1f}" path=results/docs/sgs-substep-cost.json -->42.1<!-- /calkit value -->% of the CM calls, which is <!-- calkit value key=compaction.ceiling_pct_of_explicit_call format="{:.1f}" path=results/docs/sgs-substep-cost.json -->32.0<!-- /calkit value -->% of the explicit call.

In device time, against the mod arm's <!-- calkit value key=device.device_ms format="{:,.1f}" path=results/docs/sgs-substep-cost.json -->33,306.1<!-- /calkit value --> ms: the closure is <!-- calkit value key=device.closure_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->19.55<!-- /calkit value -->% of the budget, of which <!-- calkit value key=device.cm_calls_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->8.88<!-- /calkit value -->% is CM calls on the explicit path alone.
Dropping to one substep saves <!-- calkit value key=device.substep_saving_ms format="{:,.1f}" path=results/docs/sgs-substep-cost.json -->1,478.1<!-- /calkit value --> ms, or <!-- calkit value key=device.substep_saving_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->4.44<!-- /calkit value -->% of device.
Compaction's ceiling is <!-- calkit value key=device.compaction_ceiling_ms format="{:,.1f}" path=results/docs/sgs-substep-cost.json -->1,245.3<!-- /calkit value --> ms, or <!-- calkit value key=device.compaction_ceiling_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->3.74<!-- /calkit value -->% of device, and the two are sequential rather than additive because one substep leaves half as much for the other to gather.

Two things the ceiling is not.
The mask is the environment condensate alone, while the guard in `_sum_1m_over_sgs` also requires a non-finite `lambda_eq`, so the true hit rate is at most the figure above and the ceiling with it.
And <!-- calkit value key=divergence.condensate_free_pct format="{:.1f}" path=results/docs/sgs-substep-cost.json -->43.6<!-- /calkit value -->% is not comparable to S4r's 88.31%: that was quadrature points on a different stack, not environment cells on this one.

The implicit path is excluded from the compaction figure.
It is <!-- calkit value key=device.implicit_pct format="{:.2f}" path=results/docs/sgs-substep-cost.json -->7.86<!-- /calkit value -->% of device in six calls a step, and `Instantaneous` has no substep loop, so this fit cannot separate its CM call from the machinery around it -- which makes the ceiling above a floor under what compaction is worth across the whole closure.

## Verdict

S4r's measurement stands and its conclusion does not.
Skipping the CM call where the clear-air guard fires buys about 3%, and that remains true; it does not follow that the call is 4% of the kernel, because the guard fires where the call is cheap and because at this clear rate almost every warp runs the cloudy path regardless.
The call is <!-- calkit value key=explicit.substep_share_pct format="{:.1f}" path=results/docs/sgs-substep-cost.json -->75.9<!-- /calkit value -->% of the explicit kernel.
"Do not spend another experiment on smarter early-outs here" was right for early-outs and wrong as a verdict on the kernel: it is bound by microphysics, and the way to reach it is to change which lanes run it rather than which points skip it.

Neither item is adopted here.
The substep count is a physics-accuracy decision that belongs to the science owners, and this experiment exists to put a price on it rather than to take it.
Compaction is a treatment and is now worth building, against a measured ceiling instead of a sparsity argument.

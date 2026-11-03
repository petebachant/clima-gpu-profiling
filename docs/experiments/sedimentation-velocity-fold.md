# Folding the terminal-velocity parameters: an occupancy step that did not matter

The Chen 2022 snow fall speed became the ClimaAtmos default in the 2026-10-30 re-baseline, and the sedimentation call site hands its coefficients to the kernel as runtime structs.
Wrapping them in `Val` sheds registers exactly as the parameter fold does everywhere else, and it bought almost nothing end to end.
That combination is the point of the experiment: [the fold decision rule](../fold-decision-rule.md) could not separate occupancy headroom from device share, because every site it had measured confounded the two, and this one separates them.

Numbers come from `results/sedimentation-fold.json`, which is a frozen snapshot: it is derived from the shared `results/nsys/*.sqlite`, which the next experiment overwrites, so it carries both arms' refs and both database hashes.

```python calkit stage name=values environment=py outputs=[{path: results/docs/sedimentation-velocity-fold.json, storage: git}] inputs=[results/sedimentation-fold.json, results/fold-decision.json, results/fold-compile-cost.json, results/compile-time/compile-time.toml]
import json
import tomllib

fold = json.load(open("results/sedimentation-fold.json"))
cc = json.load(open("results/fold-compile-cost.json"))
# TOML, which a value marker cannot read, so the derivation happens here.
ct = tomllib.load(open("results/compile-time/compile-time.toml", "rb"))
rule = json.load(open("results/fold-decision.json"))["summary"]
e2e = fold["end_to_end"]


def kernel(name):
    k = fold["kernels"][name]
    return {
        "regs_before": k["registers"]["baseline"],
        "regs_after": k["registers"]["mod"],
        "regs_saved": k["registers_saved"],
        "warps_before": k["warps_per_sm"]["baseline"],
        "warps_after": k["warps_per_sm"]["mod"],
        "ms_before": k["total_ms"]["baseline"],
        "ms_after": k["total_ms"]["mod"],
        "change_pct": k["change_pct"],
        "pct_of_device": k["pct_of_device"],
    }


out = {
    "env": kernel("env_terminal_velocity"),
    "updraft": kernel("updraft_terminal_velocity"),
    "treated_saved_ms": fold["treated_saved_ms"],
    "treated_change_pct": fold["treated_change_pct"],
    "treated_pct_of_device": fold["treated_pct_of_device"],
    "saved_pct_of_device": fold["saved_pct_of_device"],
    "device_total_change_pct": fold["device_total_change_pct"],
    "drift_ms": fold["untreated_drift_ms"],
    "worst_control_ms": fold["worst_control_delta_ms"],
    "saving_resolved": fold["saving_resolved_against_controls"],
    "arm_total_resolved": fold["arm_total_resolved"],
    "sypd_baseline": round(e2e["sypd"]["baseline"], 4),
    "sypd_mod": round(e2e["sypd"]["mod"], 4),
    "e2e_speedup_pct": e2e["speedup_pct"],
    "e2e_floor_pct": e2e["floor_pct"],
    "e2e_nulls": e2e["floor_n_nulls"],
    "e2e_margin_points": e2e["margin_over_floor_points"],
    "e2e_spread_points": e2e["floor_spread_points"],
    "e2e_max_points": e2e["max_points_if_fully_converted"],
    "e2e_resolved": e2e["wall_clock_resolved"],
    "e2e_steps": e2e["n_steps_timed"],
    "mod_ref": fold["provenance"]["mod_ref"][:9],
    "baseline_ref": fold["provenance"]["baseline_ref"][:9],
    "paid_warps": rule["paid_warps"],
    "flat_warps": rule["flat_warps"],
    "smallest_paying_pct": rule["smallest_paying_pct_of_device"],
    "largest_flat_pct": rule["largest_flat_pct_of_device"],
    "crossed_flat_change_pct": rule["crossed_flat_change_pct"],
}

# The compile-time price. `first` is the body compiled first in each process;
# `rest` is the three that follow it, which is where a per-call-site cost would
# have to show up if there were one.
paths = cc["paths"]
first = next(k for k, v in paths.items() if v["measured_first"])
rest = [v for k, v in paths.items() if not v["measured_first"]]
host_jit_s = ct["warmup_off"]["totals"]["host_compile_s"]
out.update({
    "cc_unfolded_s": cc["unfolded_total_compile_s"],
    "cc_folded_s": cc["folded_total_compile_s"],
    "cc_cost_s": cc["fold_total_cost_s"],
    "cc_cost_pct": cc["fold_total_cost_pct"],
    "cc_floor_s": cc["repeat_spread_floor_s"],
    "cc_resolved": cc["resolved"],
    "cc_first_path": first,
    "cc_first_cost_s": paths[first]["compile_cost_s"],
    "cc_rest_cost_s": round(sum(v["compile_cost_s"] for v in rest), 3),
    "cc_rest_n": len(rest),
    "cc_regs_min": min(v["unfolded_registers"] for v in paths.values()),
    "cc_regs_max": max(v["unfolded_registers"] for v in paths.values()),
    "cc_warps": max(v["unfolded_warps_per_sm"] for v in paths.values()),
    "cc_paths_n": len(paths),
    "cc_in_benchmark_n": len(cc["fold_relevant_in_benchmark"]),
    "cc_host_jit_s": round(host_jit_s, 1),
    "cc_share_of_jit_pct": round(100.0 * cc["fold_total_cost_s"] / host_jit_s, 3),
})
json.dump(out, open("results/docs/sedimentation-velocity-fold.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/sedimentation-velocity-fold.json -->

## The treatment

`set_precipitation_velocities!` builds a closure over `params`, `cmc` and `cmp` and broadcasts it, so every quadrature point reads the Chen 2022 coefficient tables out of a struct.
Thirty lines below, in the same file, `set_microphysics_tendency_cache!` already does `_fold(dev, cmp), _fold(dev, thp)`; CliMA/ClimaAtmos.jl#4930 folded the tendency cache and left the sedimentation path alone.
The treatment is that one idiom applied at both `set_precipitation_velocities!` methods, with `_unwrapped` at the six `terminal_velocity` dispatch sites.

Only the velocity members are folded, as `(; cmc.liquid, cmc.ice, cmc.stokes, cmc.Ch2022)`, rather than `cmc` whole.
A type parameter has to be isbits and `cmc` also carries the aerosol-ML coefficients and the activation parameters, so folding all of it risks throwing at launch; those four members are every field the dispatch reads.

## What it bought

Both treated broadcasts shed registers and crossed an occupancy step.

| | registers | warps/SM | kernel time |
| --- | --- | --- | --- |
| environment velocity | <!-- calkit value key=env.regs_before -->78<!-- /calkit value --> → <!-- calkit value key=env.regs_after -->48<!-- /calkit value --> | <!-- calkit value key=env.warps_before -->24<!-- /calkit value --> → <!-- calkit value key=env.warps_after -->40<!-- /calkit value --> | <!-- calkit value key=env.change_pct -->-15.47<!-- /calkit value -->% |
| updraft velocity | <!-- calkit value key=updraft.regs_before -->92<!-- /calkit value --> → <!-- calkit value key=updraft.regs_after -->56<!-- /calkit value --> | <!-- calkit value key=updraft.warps_before -->20<!-- /calkit value --> → <!-- calkit value key=updraft.warps_after -->36<!-- /calkit value --> | <!-- calkit value key=updraft.change_pct -->-16.08<!-- /calkit value -->% |

Together they fall from <!-- calkit value key=env.ms_before -->185.9<!-- /calkit value --> + <!-- calkit value key=updraft.ms_before -->125.1<!-- /calkit value --> ms to <!-- calkit value key=env.ms_after -->157.2<!-- /calkit value --> + <!-- calkit value key=updraft.ms_after -->104.9<!-- /calkit value --> ms over the 120 profiled steps, a saving of <!-- calkit value key=treated_saved_ms -->48.9<!-- /calkit value --> ms, or <!-- calkit value key=treated_change_pct -->-15.71<!-- /calkit value -->% of the work they do.
That saving is <!-- calkit value key=saved_pct_of_device -->0.192<!-- /calkit value -->% of device time.

## The saving is a measurement; the arm total is not

The untreated broadcasts of the same block are the control: same stack, same launch shapes, source untouched, so whatever they move by is run-to-run noise at this scale.
The worst of them moves <!-- calkit value key=worst_control_ms -->1.4<!-- /calkit value --> ms, against the treated <!-- calkit value key=treated_saved_ms -->48.9<!-- /calkit value -->, so `saving_resolved` is <!-- calkit value key=saving_resolved -->true<!-- /calkit value --> and the per-kernel figures above can be read as written.

The arm total cannot.
It moved <!-- calkit value key=device_total_change_pct -->0.01<!-- /calkit value -->%, because the untreated kernels drifted by <!-- calkit value key=drift_ms -->51.0<!-- /calkit value --> ms in the opposite direction — a few tenths of a percent each on the radiation and microphysics-tendency kernels, which are large enough that tenths of a percent swamp the treatment.
So `arm_total_resolved` is <!-- calkit value key=arm_total_resolved -->false<!-- /calkit value -->, and the per-arm device total says nothing about this treatment.

## The wall clock was measured too, by accident, and it confirms the prediction

This experiment set out not to spend an AMIP pair on a fifth of a point, and then spent one anyway: `calkit run summarize` re-ran the stale `amip-mod` stage as a dependency, so the unprofiled pair exists.
It is reported here because a prediction that gets tested is worth more than one that is argued, and because the pair would otherwise sit in `results/summary.json` where the next experiment overwrites it.

Over <!-- calkit value key=e2e_steps -->2878<!-- /calkit value --> coupled steps the arms read <!-- calkit value key=sypd_baseline -->1.2967<!-- /calkit value --> and <!-- calkit value key=sypd_mod -->1.3222<!-- /calkit value --> SYPD, a difference of <!-- calkit value key=e2e_speedup_pct -->1.928<!-- /calkit value -->%.
The floor between two *identical* arms at this benchmark window is <!-- calkit value key=e2e_floor_pct -->1.869<!-- /calkit value -->% over <!-- calkit value key=e2e_nulls -->2<!-- /calkit value --> null tests, so the margin is <!-- calkit value key=e2e_margin_points -->0.059<!-- /calkit value --> points — against a <!-- calkit value key=e2e_spread_points -->1.196<!-- /calkit value -->-point spread between the two nulls that set that floor.
`wall_clock_resolved` is therefore <!-- calkit value key=e2e_resolved -->false<!-- /calkit value -->, and the <!-- calkit value key=e2e_speedup_pct -->1.928<!-- /calkit value -->% must not be quoted as this treatment's speedup.

The part of that which does not depend on this pair at all: the device saving is <!-- calkit value key=e2e_max_points -->0.192<!-- /calkit value --> points of device time, so even a perfect conversion to wall clock could not have cleared a floor whose own nulls disagree by <!-- calkit value key=e2e_spread_points -->1.196<!-- /calkit value --> points.
The pair was unresolvable before it ran, which is why it was not worth running — and it is the reason the per-kernel controls in the section above, not this section, are what prices the fold.

## Why it is worth recording anyway: headroom and share come apart

[The fold decision rule](../fold-decision-rule.md) reports that every site which paid started at <!-- calkit value key=paid_warps -->8<!-- /calkit value --> warps per SM and held at least <!-- calkit value key=smallest_paying_pct -->3.4<!-- /calkit value -->% of device time, while every flat site started at <!-- calkit value key=flat_warps -->20 and 32<!-- /calkit value --> warps and held at most <!-- calkit value key=largest_flat_pct -->0.4<!-- /calkit value -->%.
Two predictors moved together across all four sites, so the record could not say which one was doing the work.

This site splits them.
Its occupancy headroom is that of a flat site — <!-- calkit value key=updraft.warps_before -->20<!-- /calkit value --> warps per SM, the same as the flat case the rule names — and its device share is smaller still than any of them, at <!-- calkit value key=treated_pct_of_device -->1.22<!-- /calkit value -->%.
Yet it returned <!-- calkit value key=treated_change_pct -->-15.71<!-- /calkit value -->% on the kernels themselves, an order of magnitude more than the <!-- calkit value key=crossed_flat_change_pct -->-1.16<!-- /calkit value -->% the rule's one step-crossing flat site returned.

So the two predictors answer different questions, and the rule should be read as two.
Occupancy headroom predicts the change in the kernel: a fold that crosses a step makes its kernel substantially faster whether or not the kernel matters, and <!-- calkit value key=updraft.warps_before -->20<!-- /calkit value --> warps per SM is enough headroom to do it.
Device share predicts whether anyone notices, and it is the gate that decides whether to spend the GPU time at all — the earlier flat results were flat because their kernels were small, not because their occupancy was already adequate.

The practical form: screen on device share first, because it is free to read off a profile and it rejects the most candidates, then run the register ceiling test on what survives.
Screening in the other order is what produced this experiment.

## What it costs in compile time

The fold's price is structural: a runtime value in a type parameter makes a specialization no pkgimage can hold, so whatever it costs to infer and codegen is paid at run time, on every start.
The `fold-compile-cost` stage measures that directly, compiling the <!-- calkit value key=cc_paths_n -->4<!-- /calkit value --> terminal-velocity bodies that read the folded parameters, one arm per process so the second arm cannot reuse the first's inference of every shared callee.
Only the modes that read the parameters are measured, because the `Fixed` modes take `args...` and ignore both payloads by signature, so there is nothing for a fold to change in them.

The four paths cost <!-- calkit value key=cc_unfolded_s -->1.274<!-- /calkit value --> s to compile unfolded and <!-- calkit value key=cc_folded_s -->4.142<!-- /calkit value --> s folded, so the fold costs <!-- calkit value key=cc_cost_s -->2.868<!-- /calkit value --> s, against a repeat spread of <!-- calkit value key=cc_floor_s -->0.055<!-- /calkit value --> s between the two runs of each arm.
In relative terms that is <!-- calkit value key=cc_cost_pct -->225.12<!-- /calkit value -->%, which is worth stating plainly and is also the least useful way to read it: it more than triples a quantity whose absolute size is seconds.

**It is a one-time cost, not a per-call-site one.** Almost all of it lands on whichever body is compiled first — <!-- calkit value key=cc_first_cost_s -->2.851<!-- /calkit value --> s on `<!-- calkit value key=cc_first_path -->liquid_stokes<!-- /calkit value -->` — while the <!-- calkit value key=cc_rest_n -->3<!-- /calkit value --> bodies that follow it come to <!-- calkit value key=cc_rest_cost_s -->0.018<!-- /calkit value --> s between them.
That is the shape of specializing once on the payload types and reusing it: the first body pays to infer and codegen through the two parameter structs embedded in type parameters, and nothing after it pays again.
The harness compiles the paths in a fixed order, so it does not separate "the body compiled first" from "that particular body" — but a cost that was genuinely per-call-site would have to appear in the three that follow, and it does not.

Set against the startup it is part of, the figure is small. Host-side Julia JIT is <!-- calkit value key=cc_host_jit_s -->1628.0<!-- /calkit value --> s of a profiling job ([compile time](../compile-time.md)), so the fold adds <!-- calkit value key=cc_share_of_jit_pct -->0.176<!-- /calkit value -->% to it.

One caution on the register columns the results file also carries, which this section deliberately does not quote as a benefit: in isolation these bodies are <!-- calkit value key=cc_regs_min -->32<!-- /calkit value -->-<!-- calkit value key=cc_regs_max -->40<!-- /calkit value --> registers and already at <!-- calkit value key=cc_warps -->16<!-- /calkit value --> warps/SM, so the fold has no occupancy to buy there.
The benefit is the one measured above, in the fused kernels where these calls are inlined alongside everything else and the register pressure is a kernel's worth rather than a call's.

The last figure is the one that decides whether the cost buys anything in this configuration at all: of the <!-- calkit value key=cc_paths_n -->4<!-- /calkit value --> fold-relevant paths, the benchmark selects <!-- calkit value key=cc_in_benchmark_n -->1<!-- /calkit value --> — snow, at `Chen2022`, with liquid, ice and rain all left at `Fixed`.
So three quarters of the folded call sites compile the fold and then discard it, and the register saving measured above comes from the one that does not.

## Status

The treatment is on `ClimaAtmos.jl-mod` at <!-- calkit value key=mod_ref -->d86251d45<!-- /calkit value -->, against baseline <!-- calkit value key=baseline_ref -->f11850f4a<!-- /calkit value -->.
It is correct, free and upstreamable on its own merits — it sheds registers from a kernel that had no reason to be paying for its parameters — and its SYPD must not be quoted, because the AMIP pair that priced it came in <!-- calkit value key=e2e_margin_points -->0.059<!-- /calkit value --> points over the floor between identical arms.
The record that survives is the per-kernel one: both halves of a <!-- calkit value key=treated_change_pct -->-15.71<!-- /calkit value -->% change, resolved against untreated controls in the same block, on a site that separates two predictors the decision rule could not tell apart.

This pass is tagged `exp/2026-10-31-sedimentation-velocity-fold`, where `export-sedimentation-fold` is live and both nsys databases are in the DVC remote, so the snapshot above regenerates from the tag.
On main the stage is frozen, because those database paths hold whichever experiment profiled last.

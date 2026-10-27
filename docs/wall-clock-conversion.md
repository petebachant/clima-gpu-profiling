# How much wall clock a device saving returns, and how wide the floor under it is

A kernel optimization saves N ms of device time on every step.
This asks what fraction of that reaches the clock, and --- because the answer turned out to be "essentially all of it" --- what the smallest speedup worth believing on this benchmark is.

The conversion question is settled.
The floor question is not, and the honest state of it is recorded here rather than rounded off.

Numbers here are injected from `results/docs/wall-clock-conversion.json` by the block below and rewritten on every run, so none of them is typed.
Its `inputs` are the files the prose quotes, which is what ties this document to the pipeline and makes it go stale when they do.

The treatment measured in the step-breakdown and AMIP sections is read out of `results/fold-record/`, a verbatim copy of the measurement's tag that `scripts/snapshot-tag-records.py` re-makes on every pass, and the null test out of `results/null-record/` the same way.
The live `results/summary.json` describes whatever ran last, and what ran last is a null test, so reading it live here would render roughly zero under this prose.

```python calkit stage name=values environment=py outputs=[{path: results/docs/wall-clock-conversion.json, storage: git}] inputs=[results/fold-record/step-comparison.json, results/fold-record/summary.json, results/null-record/summary.json, results/null-record/step-comparison.json, results/null-record/param-fold.json, results/sypd-repeats.json, results/null-census.json]
import json


def load(path):
    return json.load(open(path))


step = load("results/fold-record/step-comparison.json")
summary = load("results/fold-record/summary.json")
null = load("results/null-record/summary.json")
null_step = load("results/null-record/step-comparison.json")
null_fold = load("results/null-record/param-fold.json")
repeats = load("results/sypd-repeats.json")["latest_treatment"]
census = load("results/null-census.json")

plain, diag = step["classes"]["plain"], step["classes"]["diagnostics"]
typ, pen, ss = step["typical_step"], step["compile_penalty"], step["steady_state"]
own = step["own_threshold"]

out = {
    "typical": {
        "saved_ms": round(typ["saved_median_ms"], 2),
        "device_predicted_ms": round(typ["device_predicted_ms"], 2),
        "conversion_pct": round(typ["conversion_pct"]),
    },
    "window": {
        "saved_pct": round(step["saved_pct"], 2),
        "counterfactual_pct": round(step["counterfactual"]["saved_pct"], 2),
        "shortfall_points": round(step["counterfactual"]["shortfall_points"], 2),
        "plain_steps": plain["steps"],
        "plain_wall_pct": round(plain["pct_of_baseline_wall"], 1),
        "diag_steps": diag["steps"],
        "diag_wall_pct": round(diag["pct_of_baseline_wall"], 1),
        "diag_multiple": round(diag["multiple_of_plain_step"]),
        "diag_saving_pct": round(diag["pct_of_saving"], 1),
    },
    "penalty": {
        "steps": pen["steps"],
        "pct_of_plain_steps": round(pen["pct_of_plain_steps"], 1),
        "total_s": round(pen["total_s"], 1),
        "ms_per_plain_step": round(pen["ms_per_plain_step"], 2),
        "worst_step_ms": round(pen["worst_step_ms"]),
        "last_step_over_200ms": pen["last_step_over_200ms"],
    },
    "steady_state": {
        "saved_pct": round(ss["saved_pct"], 2),
        "saved_pct_ex_diagnostics": round(ss["saved_pct_ex_diagnostics"], 2),
        "cycles": ss["cycles"],
        "from_step_720": round(ss["sensitivity"]["from_step_720"], 2),
        "from_step_1440": round(ss["sensitivity"]["from_step_1440"], 2),
        "from_step_2400": round(ss["sensitivity"]["from_step_2400"], 2),
    },
    "end_to_end": {
        "speedup_pct": round(summary["speedup_pct"], 2),
        "sypd_baseline": round(summary["baseline"]["sypd"], 4),
        "sypd_mod": round(summary["mod"]["sypd"], 4),
    },
    "null": {
        "amip_pct": round(null["speedup_pct"], 2),
        "nsys_pct": round(null["nsys_speedup_pct"], 2),
        "steady_state_pct": round(null["steady_state_pct"], 2),
        "step_pct": round(null_step["saved_pct"], 2),
        "device_pct": round(null_fold["fold_attributable"]["change_pct"], 2),
        "arm_pct": round(null_fold["device_total"]["change_pct"], 2),
    },
    "repeats": {
        "n": repeats["n_pairs"],
        "mean_pct": repeats["mean_pct"],
        "min_pct": repeats["min_pct"],
        "max_pct": repeats["max_pct"],
        "spread_points": repeats["spread_points"],
        "margin_points": repeats["margin_over_null_points"],
        "reverse_pct": repeats["by_order"]["mod_first"]["mean_pct"],
        "predicted_pct": repeats["predicted_if_order_explains_null_pct"],
        "order_effect_points": repeats["order_effect_points"],
        "corrected_pct": repeats["order_corrected_pct"],
        "same_gpu_pairs": repeats["same_gpu_pairs"],
    },
    "census": {
        "n": census["n_nulls"],
        "n_current_window": census["n_current_window"],
        "n_earlier": census["n_earlier_windows"],
        "earlier_abs_max_pct": census["earlier_abs_max_pct"],
        "skipped_unpaired": census["skipped_unpaired"],
    },
    "own_threshold": {
        "mod_frac": round(own["mod_expensive_wall_frac"], 1),
        "baseline_frac": round(own["baseline_expensive_wall_frac"], 1),
        "baseline_median_ms": round(own["baseline_median_ms"], 1),
        "mod_median_ms": round(own["mod_median_ms"], 1),
    },
}
json.dump(out, open("results/docs/wall-clock-conversion.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/wall-clock-conversion.json -->

## The typical step converts the saving completely

The median paired step saves <!-- calkit value key=typical.saved_ms -->19.73<!-- /calkit value --> ms against the <!-- calkit value key=typical.device_predicted_ms -->20.57<!-- /calkit value --> ms the profiled device time predicts, which is <!-- calkit value key=typical.conversion_pct -->96<!-- /calkit value -->%, across <!-- calkit value key=window.plain_steps -->2784<!-- /calkit value --> ordinary steps.
Nothing is lost between the kernel and the clock.

It also survives to the run total.
The window reports <!-- calkit value key=window.saved_pct -->10.57<!-- /calkit value -->% against the <!-- calkit value key=window.counterfactual_pct -->9.92<!-- /calkit value -->% that per-step rate would give, a shortfall of <!-- calkit value key=window.shortfall_points -->-0.64<!-- /calkit value --> points --- nothing.
So there is no conversion gap to explain.

## The benchmark that hid it

The benchmark this project measured on first was a one-day window at a 30-second coupled timestep, which wrote diagnostics 24 times a simulated day instead of once and left ordinary steps about half the wall clock, so an every-step saving had half the budget to act on and reported a fraction of itself.
The window rationale in AGENTS.md records why each number of the current window was chosen.

Mirroring the nightly configuration instead --- `dt` 120secs, `dt_rad` 1hours, four simulated days --- leaves ordinary steps <!-- calkit value key=window.plain_wall_pct -->68.1<!-- /calkit value -->% of the wall clock and the <!-- calkit value key=window.diag_steps -->4<!-- /calkit value --> daily diagnostics writes <!-- calkit value key=window.diag_wall_pct -->12.7<!-- /calkit value -->%, still <!-- calkit value key=window.diag_multiple -->129<!-- /calkit value -->x an ordinary step each and still absorbing almost none of the saving, <!-- calkit value key=window.diag_saving_pct -->3.5<!-- /calkit value -->% of it.
The periodic ceiling did not go away; it stopped dominating.

That is the lesson, and it is not about conversion, which was never the problem.
A benchmark whose periodic work dominates its wall clock will report a fraction of any every-step saving, and will do so without any sign that it is doing it.
Convert a kernel result through the step distribution before promising it as a speedup, and check what fraction of the window the ordinary steps actually hold.

## Compilation, and the steady state

Compilation shrank with the window.
There are <!-- calkit value key=penalty.steps -->35<!-- /calkit value --> ordinary steps, <!-- calkit value key=penalty.pct_of_plain_steps -->1.3<!-- /calkit value -->% of them, that are slower in the specialized arm: <!-- calkit value key=penalty.total_s -->4.5<!-- /calkit value --> s in total, or <!-- calkit value key=penalty.ms_per_plain_step -->1.61<!-- /calkit value --> ms averaged over every ordinary step, worst single step <!-- calkit value key=penalty.worst_step_ms -->277<!-- /calkit value --> ms and the last above 200 ms at step <!-- calkit value key=penalty.last_step_over_200ms -->799<!-- /calkit value -->.
It is first-call compilation of the specialized methods rather than noise, which would be symmetric, but at that size it no longer moves the headline.

The steady-state figure now agrees with the whole window rather than correcting it: <!-- calkit value key=steady_state.saved_pct -->9.66<!-- /calkit value -->% over the final <!-- calkit value key=steady_state.cycles -->2<!-- /calkit value --> diagnostics cycles, <!-- calkit value key=steady_state.saved_pct_ex_diagnostics -->10.67<!-- /calkit value -->% excluding the diagnostics steps, against <!-- calkit value key=window.saved_pct -->10.57<!-- /calkit value -->% for the window entire.
It is flat wherever the cutoff is put --- <!-- calkit value key=steady_state.from_step_720 -->9.91<!-- /calkit value -->% from step 720, <!-- calkit value key=steady_state.from_step_1440 -->9.66<!-- /calkit value -->% from step 1440, <!-- calkit value key=steady_state.from_step_2400 -->8.76<!-- /calkit value -->% from step 2400 --- which is what a run with little compilation left in it looks like.

## End to end, and the null test

On the unprofiled AMIP pair rather than the step-breakdown pair, the treatment is worth <!-- calkit value key=end_to_end.speedup_pct -->8.66<!-- /calkit value -->% of wall clock, <!-- calkit value key=end_to_end.sypd_baseline -->1.6067<!-- /calkit value --> to <!-- calkit value key=end_to_end.sypd_mod -->1.7591<!-- /calkit value --> SYPD --- the figures CliMA/ClimaAtmos.jl#4879 quotes.
Three measurements of one treatment on three pairs of runs will not agree to the last digit, and the SYPD pair is a throughput ratio while the step-breakdown figure is a duration reduction, so the two must never be quoted as one number.

The null test prices that disagreement instead of conceding it.
With CliMA/ClimaAtmos.jl#4879 merged both arms sit on one commit, so the same three measurements describe a treatment of nothing: <!-- calkit value key=null.amip_pct -->2.47<!-- /calkit value -->% on the unprofiled AMIP pair, <!-- calkit value key=null.step_pct -->-1.35<!-- /calkit value -->% on the step-breakdown pair, <!-- calkit value key=null.nsys_pct -->-1.64<!-- /calkit value -->% profiled, and <!-- calkit value key=null.steady_state_pct -->-0.74<!-- /calkit value -->% in steady state.
All four are near zero, and two have the wrong sign.

The device chain agrees it is zero where it should --- <!-- calkit value key=null.device_pct -->-0.0<!-- /calkit value -->% across the treated kernels, <!-- calkit value key=null.arm_pct -->-0.09<!-- /calkit value -->% over the whole arm --- which is the direct check that device-side nsys figures are the trustworthy ones and wall clock is the noisy one, rather than the reverse.

## What kind of floor: repeats, run order and GPU assignment

Repeating a pair says what kind of floor this is, and the answer is not the one this project assumed for months.
The treatment measured in the current pass --- the updraft cache, not the fold above --- has been run end to end <!-- calkit value key=repeats.n -->3<!-- /calkit value --> times on the same committed code, giving <!-- calkit value key=repeats.min_pct -->4.847<!-- /calkit value -->% to <!-- calkit value key=repeats.max_pct -->5.354<!-- /calkit value -->%: a mean of <!-- calkit value key=repeats.mean_pct -->5.086<!-- /calkit value -->% and a spread of <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points.

Run-to-run repeatability is therefore about half a point, several times tighter than the <!-- calkit value key=null.amip_pct -->2.47<!-- /calkit value -->% the null test found between two IDENTICAL arms.
Those two figures cannot both be jitter.
One of them is a confound that repeats, and a confound that repeats is not reduced by averaging.

Two candidates were available and both are now measured.
The arms run one after the other, so running second could be worth something; and SLURM hands out whichever device is free, so the arms need not even share a GPU.

**Run order is not it.**
Reversing the order --- mod first, baseline second, forced onto the same committed code --- was predicted to return <!-- calkit value key=repeats.predicted_pct -->0.123<!-- /calkit value -->% if the null's <!-- calkit value key=null.amip_pct -->2.47<!-- /calkit value -->% were entirely the advantage of running second, and it returned <!-- calkit value key=repeats.reverse_pct -->4.847<!-- /calkit value -->%.
The prediction was committed before the reversed pair ran, so the record can be checked against it rather than described after the fact.
Running second is worth <!-- calkit value key=repeats.order_effect_points -->0.105<!-- /calkit value --> points, and the order-corrected figure is <!-- calkit value key=repeats.corrected_pct -->4.952<!-- /calkit value -->%.

**Nor is the GPU.**
It is ruled out from the record rather than by spending GPU time: the pairs span a matched assignment, a mismatched one, and a second matched one --- <!-- calkit value key=repeats.same_gpu_pairs -->2<!-- /calkit value --> of <!-- calkit value key=repeats.n -->3<!-- /calkit value --> shared a device, and one of the others had the baseline arm on the GPU this cluster used to blacklist --- and they agree inside <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points regardless.

## The null-test census

The history says the same thing by counting.
There are <!-- calkit value key=census.n -->3<!-- /calkit value --> passes in this repository whose arms `record-treatment` found identical in content and whose measurement was committed by the same pass, and the <!-- calkit value key=census.n_earlier -->2<!-- /calkit value --> run on earlier benchmark windows land within <!-- calkit value key=census.earlier_abs_max_pct -->0.207<!-- /calkit value --> points of zero.
The <!-- calkit value key=null.amip_pct -->2.47<!-- /calkit value -->% is the only null at the window in use now.

The pairing test is what makes that count trustworthy.
A commit counts only where one pass wrote both `results/treatment.json` and `results/summary.json`, because before `record-treatment` carried `always_run` it could be skipped while the pass it was supposed to describe went ahead --- one such commit claims identical arms beside a +12.6% speedup.
In all, <!-- calkit value key=census.skipped_unpaired -->12<!-- /calkit value --> commits are rejected on that test, which includes the harmless case of a later commit re-committing an already counted pass.

## Where that leaves the floor

The likeliest reading is that this window's single null pair was an outlier draw rather than a bias, which would put the floor near the repeats' own <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points and leave this treatment's <!-- calkit value key=repeats.mean_pct -->5.086<!-- /calkit value -->% comfortably clear of it.

That is a reading and not a measurement.
It rests on one null pair at this configuration, and repeating the null test --- two unprofiled AMIP runs --- is still what would settle it.
Until it is settled the conservative course is to quote the <!-- calkit value key=repeats.margin_points -->2.619<!-- /calkit value -->-point margin over the null alongside the headline, and to treat a candidate worth a percent or two as unresolved by a single pair.

## Measure both arms

The one-arm version of the step-breakdown stage reported <!-- calkit value key=own_threshold.mod_frac -->36.6<!-- /calkit value -->% of wall in the mod arm's expensive steps against <!-- calkit value key=own_threshold.baseline_frac -->34.2<!-- /calkit value -->% in the baseline's, which reads as a treatment effect and is not one.
The threshold is 1.25x each arm's OWN median, and the fold moved the median from <!-- calkit value key=own_threshold.baseline_median_ms -->142.5<!-- /calkit value --> to <!-- calkit value key=own_threshold.mod_median_ms -->122.7<!-- /calkit value --> ms, so an unchanged distribution crosses a lower bar.
Those fields cannot be read across arms; a fixed periodic classification can.

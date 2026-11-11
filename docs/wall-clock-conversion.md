# How much wall clock a device saving returns, and how wide the floor under it is

A kernel optimization saves N ms of device time on every step.
This asks what fraction of that reaches the clock, and --- because the answer turned out to be "essentially all of it" --- what the smallest speedup worth believing on this benchmark is.

The conversion question is settled.
The floor question is not, and the honest state of it is recorded here rather than rounded off.

Numbers here are injected from `results/docs/wall-clock-conversion.json` by the block below and rewritten on every run, so none of them is typed.
Its `inputs` are the files the prose quotes, which is what ties this document to the pipeline and makes it go stale when they do.

The treatment measured in the step-breakdown and AMIP sections is read out of `results/fold-record/`, a verbatim copy of the measurement's tag that `scripts/snapshot-tag-records.py` re-makes on every pass, and the null test out of `results/null-record/` the same way.
The live `results/summary.json` describes whatever ran last, and what ran last is a null test, so reading it live here would render roughly zero under this prose.

```python calkit stage name=values environment=py outputs=[{path: results/docs/wall-clock-conversion.json, storage: git}] inputs=[results/fold-record/step-comparison.json, results/fold-record/summary.json, results/null-record/summary.json, results/null-record/step-comparison.json, results/null-record/param-fold.json, results/sypd-repeats.json, results/null-census.json, results/summary.json, results/kernel-population.json]
import json


def load(path):
    return json.load(open(path))


step = load("results/fold-record/step-comparison.json")
summary = load("results/fold-record/summary.json")
null = load("results/null-record/summary.json")
null_step = load("results/null-record/step-comparison.json")
null_fold = load("results/null-record/param-fold.json")
repeat_record = load("results/sypd-repeats.json")
current = repeat_record["latest_treatment"]
# Order and GPU assignment are read off the richest group of this treatment,
# which carries the most pairs and the only pairs that shared a GPU.
repeats = repeat_record["most_repeated_treatment"]
census = load("results/null-census.json")
# The live files describe whatever ran last, which is the pair the `current`
# block reports, so they are the right source for ITS device check and the
# wrong one for every pinned figure above.
live = load("results/summary.json")
pop = load("results/kernel-population.json")
# The nulls at the window in use now are the floor under every headline in
# this document; the ones on earlier windows measure a different noise.
cw = census["current_window_pct"]
cw_mean = census["current_window_mean_pct"]

# What the device side of this same pass can account for: the kernel-time
# saving spread over the profiled window, against an unprofiled baseline step.
nsys_steps = 120
device_saved_ms_per_step = (
    pop["baseline"]["kernel_time_ms"] - pop["mod"]["kernel_time_ms"]
) / nsys_steps
base_step_ms = live["baseline"]["walltime_per_step_s"] * 1000.0
device_predicted_pct = 100.0 * device_saved_ms_per_step / base_step_ms

plain, diag = step["classes"]["plain"], step["classes"]["diagnostics"]
typ, pen, ss = step["typical_step"], step["compile_penalty"], step["steady_state"]
# The same field for two identical arms. It is one-sided -- only the steps where
# the mod arm was slower -- so it is positive with no treatment at all, and the
# null is the only thing that says which part of it means anything.
null_pen = null_step["compile_penalty"]
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
        "null_steps": null_pen["steps"],
        "null_pct_of_plain_steps": round(null_pen["pct_of_plain_steps"], 1),
        "null_total_s": round(null_pen["total_s"], 1),
        "null_last_step_over_200ms": null_pen["last_step_over_200ms"],
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
    "current": {
        "n": current["n_pairs"],
        "mean_pct": current["mean_pct"],
        "margin_points": current["margin_over_null_points"],
        # Sign-bearing, so the prose need not assert a direction a later pass
        # can reverse.
        "clears_floor": current["margin_over_null_points"] > 0,
        "predicted_pct": current["predicted_if_order_explains_null_pct"],
        # The latest group is a null whenever upstream has taken the treatment,
        # and its percentage is then a draw from the floor rather than a
        # measurement of a change.
        "is_null": current["treatment"] == [],
        "device_predicted_pct": round(device_predicted_pct, 2),
        # Sign-bearing, like clears_floor above. A headline inside the null
        # spread of its own device prediction needs no caveat; one outside it
        # is unexplained by the measurement this project trusts.
        "exceeds_device_prediction": current["mean_pct"]
        > device_predicted_pct + census["current_window_spread_points"],
        "spread_points": current.get("spread_points"),
        # Each arm's own wander across the group. A pair differences two
        # runs, so it resolves nothing smaller than this.
        "arm_spread_baseline_pct": current.get("arm_spread_pct", {}).get("baseline"),
        "arm_spread_mod_pct": current.get("arm_spread_pct", {}).get("mod"),
        "widest_arm": current.get("widest_arm"),
        "effect_below_arm_spread": current.get("effect_below_arm_spread"),
        "fwd_pct": current.get("by_order", {}).get("baseline_first", {}).get("mean_pct"),
        "rev_pct": current.get("by_order", {}).get("mod_first", {}).get("mean_pct"),
        "order_effect_points": current.get("order_effect_points"),
        "corrected_pct": current.get("order_corrected_pct"),
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
        "current_min_pct": min(cw),
        "current_max_pct": max(cw),
        "current_mean_pct": round(cw_mean, 2),
        "current_spread_points": census["current_window_spread_points"],
        "current_all_same_sign": census["current_window_all_same_sign"],
    },
    "floor": {
        "mean_null_pct": cw_mean,
        "repeats_margin_points": round(repeats["mean_pct"] - cw_mean, 2),
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
It is first-call compilation of the specialized methods rather than noise, but not for the reason that sentence used to give here, which was that noise would be symmetric.
This figure is one-sided: it sums only the steps where the specialized arm was slower, so noise cannot cancel in it and arrives as a positive total like any real penalty.
The null test is what separates them, and it is emphatic -- two identical arms report <!-- calkit value key=penalty.null_total_s -->8.4<!-- /calkit value --> s over <!-- calkit value key=penalty.null_steps -->2207<!-- /calkit value --> steps, <!-- calkit value key=penalty.null_pct_of_plain_steps -->79.3<!-- /calkit value -->% of the ordinary ones, with the last above 200 ms as late as step <!-- calkit value key=penalty.null_last_step_over_200ms -->2613<!-- /calkit value -->.
So the seconds are the half of this figure that carries no information, being larger with no treatment than with one; the step count and how early the spikes stop are the half that does, and at <!-- calkit value key=penalty.pct_of_plain_steps -->1.3<!-- /calkit value -->% of steps ending by step <!-- calkit value key=penalty.last_step_over_200ms -->799<!-- /calkit value --> this one is compilation.
At that size it no longer moves the headline either way.

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
The treatment measured here --- the updraft cache, not the fold above --- has been run end to end <!-- calkit value key=repeats.n -->3<!-- /calkit value --> times on one stack and one set of committed code, giving <!-- calkit value key=repeats.min_pct -->4.847<!-- /calkit value -->% to <!-- calkit value key=repeats.max_pct -->5.354<!-- /calkit value -->%: a mean of <!-- calkit value key=repeats.mean_pct -->5.086<!-- /calkit value -->% and a spread of <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points.
Re-baselining onto a newer upstream main changes the code of both arms, so `record-treatment` files the result as a new group rather than a fourth repeat; the re-baseline's own pair was a null test, because upstream had merged the treatment and the two arms were identical, and it is the second of the two draws that set this window's floor below.
The current group is the one after it --- this pass's arm, carrying the SGS node removals, the pointwise register cap and the g-point groups, with the sedimentation fold now upstream in both arms --- and it holds <!-- calkit value key=current.n -->2<!-- /calkit value --> pairs of one set of committed code, <!-- calkit value key=current.fwd_pct -->6.384<!-- /calkit value -->% with the baseline first and <!-- calkit value key=current.rev_pct -->7.084<!-- /calkit value -->% with the mod first, a mean of <!-- calkit value key=current.mean_pct -->6.734<!-- /calkit value -->% over a spread of <!-- calkit value key=current.spread_points -->0.7<!-- /calkit value --> points, with `is_null` recording <!-- calkit value key=current.is_null -->false<!-- /calkit value -->.
Each arm's own wander across the two pairs is <!-- calkit value key=current.arm_spread_baseline_pct -->1.396<!-- /calkit value -->% for the baseline and <!-- calkit value key=current.arm_spread_mod_pct -->0.637<!-- /calkit value -->% for the mod, the <!-- calkit value key=current.widest_arm -->baseline<!-- /calkit value --> being the widest, and the effect clears both --- `effect_below_arm_spread` records <!-- calkit value key=current.effect_below_arm_spread -->false<!-- /calkit value -->.
A pair differences two runs, so it cannot resolve an effect smaller than the wander of the runs it differences --- and this is the first entry in this record where it comfortably can, the effect being several times the <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points those repeats spread over.
The device side agrees with it rather than arbitrating between draws: it prices this arm's treatments on this stack at <!-- calkit value key=current.device_predicted_pct -->6.1<!-- /calkit value -->% of wall clock, against the pair's <!-- calkit value key=current.mean_pct -->6.734<!-- /calkit value -->%.
Those two sit well inside the null census's own <!-- calkit value key=census.current_spread_points -->1.196<!-- /calkit value --> points of spread, so `exceeds_device_prediction` records <!-- calkit value key=current.exceeds_device_prediction -->false<!-- /calkit value -->, which here means the device chain and the unprofiled pair agree rather than disagree.
The reversed order was predicted to return <!-- calkit value key=current.predicted_pct -->1.45<!-- /calkit value -->% if run order explains the null --- a prediction committed before the repeat that tests it --- and it returned <!-- calkit value key=current.rev_pct -->7.084<!-- /calkit value -->%, above the forward draw rather than below it.
Running second is worth <!-- calkit value key=current.order_effect_points -->-0.35<!-- /calkit value --> points on this stack and the order-corrected figure is <!-- calkit value key=current.corrected_pct -->6.734<!-- /calkit value -->%, so the order effect here is measured, and this is the second stack on which run order fails to explain the null.
What is still missing is a null on this stack.

Run-to-run repeatability is therefore about half a point, several times tighter than the <!-- calkit value key=census.current_mean_pct -->1.87<!-- /calkit value -->% the null tests find between two IDENTICAL arms.
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
There are <!-- calkit value key=census.n -->4<!-- /calkit value --> passes in this repository whose arms `record-treatment` found identical in content and whose measurement was committed by the same pass, and the <!-- calkit value key=census.n_earlier -->2<!-- /calkit value --> run on earlier benchmark windows land within <!-- calkit value key=census.earlier_abs_max_pct -->0.207<!-- /calkit value --> points of zero.
At the window in use now there are <!-- calkit value key=census.n_current_window -->2<!-- /calkit value -->, returning <!-- calkit value key=census.current_min_pct -->1.271<!-- /calkit value -->% and <!-- calkit value key=census.current_max_pct -->2.467<!-- /calkit value -->%.

The pairing test is what makes that count trustworthy.
A commit counts only where one pass wrote both `results/treatment.json` and `results/summary.json`, because before `record-treatment` carried `always_run` it could be skipped while the pass it was supposed to describe went ahead --- one such commit claims identical arms beside a +12.6% speedup.
In all, <!-- calkit value key=census.skipped_unpaired -->13<!-- /calkit value --> commits are rejected on that test, which includes the harmless case of a later commit re-committing an already counted pass.

## Where that leaves the floor

The second null pair has now run, and it settled this the other way.
The reading here used to be that the window's single null was an outlier draw rather than a bias, which would have put the floor near the repeats' own <!-- calkit value key=repeats.spread_points -->0.507<!-- /calkit value --> points.
Instead both nulls at this window come back positive and both exceed that spread, at <!-- calkit value key=census.current_min_pct -->1.271<!-- /calkit value -->% and <!-- calkit value key=census.current_max_pct -->2.467<!-- /calkit value -->%, a mean of <!-- calkit value key=census.current_mean_pct -->1.87<!-- /calkit value -->% between identical arms, against <!-- calkit value key=census.earlier_abs_max_pct -->0.207<!-- /calkit value --> points for every null on an earlier window.

So the floor at this window is around <!-- calkit value key=floor.mean_null_pct -->1.869<!-- /calkit value --> points rather than half a point, and jitter of the repeats' width does not account for it.
What two draws cannot separate is whether the remainder is a bias favoring the mod arm or a jitter that is simply wider across stacks than within one --- the repeats share a stack and the nulls do not --- and the distinction can wait, because both readings cost a headline the same thing.
The conservative course is to subtract that mean: the margin for the treatment measured here is <!-- calkit value key=floor.repeats_margin_points -->3.22<!-- /calkit value --> points, and a candidate worth a percent or two stays unresolved however many times it is repeated, since a confound that repeats is not reduced by averaging.

## Measure both arms

The one-arm version of the step-breakdown stage reported <!-- calkit value key=own_threshold.mod_frac -->36.6<!-- /calkit value -->% of wall in the mod arm's expensive steps against <!-- calkit value key=own_threshold.baseline_frac -->34.2<!-- /calkit value -->% in the baseline's, which reads as a treatment effect and is not one.
The threshold is 1.25x each arm's OWN median, and the fold moved the median from <!-- calkit value key=own_threshold.baseline_median_ms -->142.5<!-- /calkit value --> to <!-- calkit value key=own_threshold.mod_median_ms -->122.7<!-- /calkit value --> ms, so an unchanged distribution crosses a lower bar.
Those fields cannot be read across arms; a fixed periodic classification can.

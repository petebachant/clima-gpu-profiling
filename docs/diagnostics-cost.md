# What diagnostics cost, and which half of them is measurable

Diagnostics are a periodic cost on the benchmark window, so they set a ceiling on any work to speed them up.
This prices that ceiling, splits it into the two halves that need different work, and is explicit about which half this design can measure and which it can only bound.

Numbers here are injected from `results/docs/diagnostics-cost.json` by the block below and rewritten on every run, so none of them is typed.
Its `inputs` are the files the prose quotes, which ties this document to the pipeline and makes it go stale when they do.

```python calkit stage name=values environment=py outputs=[{path: results/docs/diagnostics-cost.json, storage: git}] inputs=[results/diagnostics-cost.json, results/step-comparison.json]
import json

cost = json.load(open("results/diagnostics-cost.json"))
step = json.load(open("results/step-comparison.json"))
diag = step["classes"]["diagnostics"]

out = {
    "total_pct": round(cost["total_pct_of_wall"], 1),
    "sypd_headroom_pct": round(cost["sypd_headroom_pct"], 1),
    "write_pct": round(cost["write_steps_pct_of_wall"], 1),
    "sampling_pct": round(cost["sampling_pct_of_wall"], 2),
    "sampling_resolvable": cost["sampling_resolvable"],
    "sources_inconsistent": cost["sources_inconsistent"],
    "repeat_spread_points": cost["repeat_spread_points"],
    "margin_over_floor": round(
        cost["sampling_pct_of_wall"] - cost["repeat_spread_points"], 2
    ),
    "wall_on_s": round(cost["with_diagnostics"]["wall_time_total_s"]),
    "wall_off_s": round(cost["without_diagnostics"]["wall_time_total_s"]),
    "sypd_with": round(cost["sypd_with"], 4),
    "sypd_without": round(cost["sypd_without"], 4),
    "n_steps": cost["n_steps_timed"],
    "diag_steps": diag["steps"],
    "diag_multiple": round(diag["multiple_of_plain_step"]),
}
json.dump(out, open("results/docs/diagnostics-cost.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/diagnostics-cost.json -->

## The ceiling

Diagnostics cost <!-- calkit value key=total_pct -->11.4<!-- /calkit value -->% of the wall clock on the benchmark window, so the whole of them is worth <!-- calkit value key=sypd_headroom_pct -->12.9<!-- /calkit value -->% in SYPD: <!-- calkit value key=sypd_with -->1.2967<!-- /calkit value --> SYPD with them against <!-- calkit value key=sypd_without -->1.4634<!-- /calkit value --> without.
That is the number to compare an optimization's cost against before starting.

Read it as an upper bound.
It differences two whole runs, <!-- calkit value key=wall_on_s -->729<!-- /calkit value --> s against <!-- calkit value key=wall_off_s -->646<!-- /calkit value --> s over <!-- calkit value key=n_steps -->2878<!-- /calkit value --> steps, so it also contains the compilation the diagnostics-off run does not pay.
The direction of that bias is known and its size is not, which is the honest limit of a difference-of-runs design and the reason the write-step share below is cited from a separate measurement rather than derived from this one.

## The split matters more than the total

The two halves need different work.
The <!-- calkit value key=diag_steps -->4<!-- /calkit value --> daily write events carry <!-- calkit value key=write_pct -->10.8<!-- /calkit value -->% of wall on their own, at <!-- calkit value key=diag_multiple -->103<!-- /calkit value -->x an ordinary step each, which is I/O and reduction flushing concentrated in a handful of steps.
The rest is the sampling that feeds them, spread thinly across every ordinary step, where it is invisible in the median and cannot be found by looking for expensive steps.

## The sampling share is under a point, and only just resolvable

The residual left after the write steps is <!-- calkit value key=sampling_pct -->0.63<!-- /calkit value --> points of wall clock, against a <!-- calkit value key=repeat_spread_points -->0.507<!-- /calkit value -->-point spread between independent repeats of one AMIP pair of identical code.
It clears that floor by <!-- calkit value key=margin_over_floor -->0.12<!-- /calkit value --> points, so `sampling_resolvable` is <!-- calkit value key=sampling_resolvable -->true<!-- /calkit value --> and the split can be read at all; the margin is smaller than the floor it cleared, so read the sampling share as under a point rather than to the digit.
A sampling cost cannot be negative, so a residual below minus the floor is not a small share but the two differently-sourced measurements disagreeing by more than the quantity between them, which `sources_inconsistent` records, currently <!-- calkit value key=sources_inconsistent -->false<!-- /calkit value -->.
It was true before the 2026-10-30 re-baseline, on a stack where this residual came out negative, and comparing the magnitude against the floor would have reported that as resolved -- which is how a sign error gets published as a measurement.

The demonstration is on the record rather than in the argument.
Re-running the baseline arm alone, twice, with no code change at all, swung the residual from its original value to zero and most of the way back; those values are in the committed history of `results/diagnostics-cost.json`, and the spread they move inside is measured in [the wall-clock conversion document](wall-clock-conversion.md).

So the write steps carry the great majority of what this measurement can see, and pricing the sampling any more precisely than "under a point" would take a separately instrumented run rather than another difference of two.

## Why it takes a whole extra run

Nothing else in this pipeline can answer the question.
The 120-step nsys window is shorter than the 720-step daily write period, so it contains no write at all and its kernel table reports the sampling as ordinary every-step work.
The step breakdown finds the writes precisely, because they are the most expensive steps in the window, and by construction cannot find the sampling for the same reason.

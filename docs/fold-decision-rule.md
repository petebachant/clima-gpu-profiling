# When is folding parameters into the type worth it?

Wrapping a parameter set in `Val` so the compiler folds it to literals has
halved two kernels in this project and returned nothing on several other
attempts that look the same from outside: the same parameters, the same source
file, sometimes the same kernel. This asks whether a candidate can be priced before
the fold is written, and in particular whether counting the parameters tells you
anything.

Nine attempts are on the record -- five compile-only probes and the four kernels
folded in the model -- and three candidate rules can be tested against them.
Numbers here are injected from `results/fold-decision.json` and rewritten on
every run, so none of them is typed.

The analysis is declared here rather than as a stage of its own, so freezing
this file freezes the whole thing. Its inputs are four files earlier stages
wrote and it runs no GPU, which is also what it concludes: if deciding a fold
needed a profile, this document could not exist.

```python calkit stage name=analyze environment=py outputs=[{path: results/fold-decision.json, storage: git}] inputs=[scripts/export_fold_decision.py, results/val-params-registers.toml, results/pointwise-registers.json, results/val-mileage.json, results/fold-record/param-fold.json]
import sys

sys.path.insert(0, "scripts")
from export_fold_decision import main

main()
```

<!-- calkit values path=results/fold-decision.json -->

## The rule that fails: count the parameters

The parameter set that ClimaAtmos folds is `Microphysics1MParams` plus
`ThermodynamicsParameters`, which at `Float32` is <!-- calkit value key=summary.payload_bytes -->468<!-- /calkit value --> bytes, or <!-- calkit value key=summary.payload_registers -->117<!-- /calkit value --> registers at four bytes each.
Call that the payload: the most a fold could free if the kernel were holding the
whole struct.
That one payload produced savings of <!-- calkit value key=summary.savings_list -->0, 102, 105, 109, 116, 117<!-- /calkit value --> registers at <!-- calkit value key=summary.payload_sites -->6<!-- /calkit value --> sites, a spread of <!-- calkit value key=summary.saving_spread -->117<!-- /calkit value --> registers on an identical parameter set.
A number that takes every value from nothing to everything is not a predictor.

The payload does bound them. None of the <!-- calkit value key=summary.payload_sites -->6<!-- /calkit value --> savings exceeded it and the largest landed exactly on it, so the size prices the
prize and orders nothing -- a bound that outcomes from zero to all of it respect
predicts none of them. These are compile-only figures and they move when the
stack moves, the same way the in-model payoffs do in `docs/learnings.md` 4ad, so
it is a tight bound rather than a hard one.

Nor does a small payload mean a small prize. The Thermodynamics call carries a <!-- calkit value key=summary.td_payload_registers -->26<!-- /calkit value -->-register payload into a <!-- calkit value key=summary.td_registers -->32<!-- /calkit value -->-register kernel and saved nothing however the parameters arrived, which is a
kernel that never paid for its parameters in the first place.

What the payload does give is the scale, once a fold is known to land. Where
nothing boxed the parameters and the kernel was larger than the payload -- <!-- calkit value key=summary.landing_sites -->5<!-- /calkit value --> of the nine -- the measured saving was <!-- calkit value key=summary.recovered_min_pct -->87.2<!-- /calkit value -->% to <!-- calkit value key=summary.recovered_max_pct -->100.0<!-- /calkit value -->% of it.
So `sizeof(params)/4` brackets the prize within <!-- calkit value key=summary.recovered_spread_pct -->12.8<!-- /calkit value --> points of the payload and says nothing at all about the odds of collecting it.

## The rule that works: compile the body with const globals

Const globals are the ceiling of any constant-folding scheme -- a value the
compiler has at compile time, with no wrapper to lose it through -- so a body
whose register count does not move when its parameters become const globals
cannot be helped by `Val`, by `_fold`, or by anything else.

That is a compile-only test. `CUDA.@cuda launch=false` compiles a kernel and
`CUDA.registers` reads its register count without executing it, so the probe
needs no simulation, no coupled model and no profile; the
`val-params-registers` stage that runs it is minutes against hours for a
profile pair.

It predicted `Val` exactly at every site where both were measured, <!-- calkit value key=summary.ceiling_agree -->3<!-- /calkit value --> of <!-- calkit value key=summary.ceiling_sites -->3<!-- /calkit value -->, with a largest disagreement of <!-- calkit value key=summary.ceiling_disagreement -->0<!-- /calkit value --> registers.
More usefully, it flagged <!-- calkit value key=summary.ceiling_flagged -->2<!-- /calkit value --> sites in advance, and those are exactly the two where the fold went on to return nothing, with <!-- calkit value key=summary.ceiling_false_alarms -->0<!-- /calkit value --> sites condemned that would have paid.
Both failures the project actually hit were catchable in five minutes without
writing the fold.

The two it caught are worth separating, because only one of them is final.

One is a kernel that was never spending registers on its parameters, where there
is nothing to recover and the question is closed. The other is the struct field
boundary: `Microphysics1MEvaluator` holds the parameters as fields, so a caller
handing it a compile-time constant has that constant stored into a struct and
read back per point, and the body sits at the same register count whether the
parameters arrive as runtime arguments, as `Val`, or as const globals.
That one is a code shape to change rather than a verdict on the kernel, and the
same body with the parameters moved into the evaluator's type goes <!-- calkit value key=summary.boxed_registers -->255<!-- /calkit value --> to <!-- calkit value key=summary.unboxed_registers -->146<!-- /calkit value --> registers, <!-- calkit value key=summary.unboxed_saving -->109<!-- /calkit value --> saved.

So a `no` from the ceiling probe closes the fold as written and asks which of
the two causes it is -- and the second question is another compile, not a
profile. That is the sequence behind the largest payoff recorded here: the
probes said the boundary was erasing the fold, the ClimaAtmos change followed
from that, and the profile came last to price it.

## What sets the size: warps per SM, not registers

A fold returns time by letting another block onto the SM, and ClimaCore sizes
the launch block from the register count, so the quantity to watch is warps per
SM rather than registers.

The control is the folded kernel whose register count did not move: it returned <!-- calkit value key=summary.noise_band_pct -->1.89<!-- /calkit value -->%, which is the per-kernel noise this measurement resolves.
Both payoffs above that band started at <!-- calkit value key=summary.paid_warps -->8<!-- /calkit value --> warps per SM, and returned <!-- calkit value key=summary.best_change_pct -->-49.34<!-- /calkit value -->% going <!-- calkit value key=summary.best_warps_before -->8<!-- /calkit value --> to <!-- calkit value key=summary.best_warps_after -->16<!-- /calkit value --> warps and <!-- calkit value key=summary.biggest_change_pct -->-45.69<!-- /calkit value -->% going <!-- calkit value key=summary.biggest_warps_before -->8<!-- /calkit value --> to <!-- calkit value key=summary.biggest_warps_after -->12<!-- /calkit value -->.
Everything flat started at <!-- calkit value key=summary.flat_warps -->20 and 32<!-- /calkit value --> warps per SM.

Crossing a step is therefore necessary and nowhere near sufficient. The
cloud-fraction kernel shed <!-- calkit value key=summary.crossed_flat_registers_saved -->13<!-- /calkit value --> registers, crossed from <!-- calkit value key=summary.crossed_flat_warps_before -->32<!-- /calkit value --> to <!-- calkit value key=summary.crossed_flat_warps_after -->36<!-- /calkit value --> warps per SM, and returned <!-- calkit value key=summary.crossed_flat_change_pct -->-1.16<!-- /calkit value -->% -- less than the control that crossed nothing.

One honest limit: in this record the high-occupancy kernels are also the small
ones. Every flat site holds at most <!-- calkit value key=summary.largest_flat_pct_of_device -->0.4<!-- /calkit value -->% of device time while the smallest payoff holds <!-- calkit value key=summary.smallest_paying_pct_of_device -->3.4<!-- /calkit value -->%, so occupancy headroom and device share cannot be separated from these four
kernels alone. Operationally it does not matter, because both are free to check.

## How many kernels get past the screen

Of <!-- calkit value key=summary.kernels_profiled -->215<!-- /calkit value --> kernels in a 120-step run, <!-- calkit value key=summary.addressable -->36<!-- /calkit value --> (<!-- calkit value key=summary.addressable_pct -->16.74<!-- /calkit value -->%) are capped by registers alone and below full occupancy, and <!-- calkit value key=summary.at_full_occupancy -->110<!-- /calkit value --> are already at full occupancy and can gain nothing however well their parameters fold.
By time rather than count that is <!-- calkit value key=summary.ceiling_pct_of_device -->23.27<!-- /calkit value -->% of the device, of which <!-- calkit value key=summary.untreated_pct_of_device -->1.85<!-- /calkit value -->% over <!-- calkit value key=summary.untreated_kernels -->13<!-- /calkit value --> kernels is untreated -- and the largest single piece of it, <!-- calkit value key=summary.largest_untreated_pct -->0.904<!-- /calkit value -->% of the device needing <!-- calkit value key=summary.largest_untreated_shed -->47<!-- /calkit value --> registers to cross its step, is a ClimaCore broadcast kernel with no parameter struct in it to fold.

Both halves of that screen come from measurements the pipeline already holds:
the occupancy limiter from one static ncu pass over the whole run, the device
share from an existing nsys profile.

## The procedure

1. Is the parameter set `isbits`? If not, it cannot be a type parameter and the
   question is closed. Free.
2. From the existing ncu pass, are registers the kernel's sole occupancy limit,
   is it below full occupancy, and how many registers would it have to shed for
   one more block per SM? Free.
3. From the existing nsys profile, how much device time does it hold? A kernel
   under about a percent cannot repay the work whatever happens to it. Free.
4. Compile the body with the parameters as const globals and read
   `CUDA.registers`. If it does not move, stop: no folding scheme will help.
   Minutes, no simulation.
5. If it moves, expect a real fold to recover most of `sizeof(params)/4`, and
   compare that against the shed from step 2 to see whether it crosses.
6. Write the fold, keeping the wrapper intact all the way to the arithmetic, and
   check the register count moved the way the probe said. Minutes.
7. Profile only now, to price it.

Steps 1 through 5 are free or minutes, and they called every outcome on the
record for the body they were handed. Step 7 is hours, and it is about the size
of the answer rather than its sign.

## What this does not settle

The threshold between <!-- calkit value key=summary.paid_warps -->8<!-- /calkit value --> warps per SM, where folds paid, and <!-- calkit value key=summary.flat_warps -->20 and 32<!-- /calkit value -->, where they did not, is unmeasured; the record brackets it and does not locate it.

And a price has a shelf life. The same fold on the same kernels measured
different payoffs one upstream stack earlier, because the rewrites in between
changed the kernel bodies and moved their distance to the next occupancy step.
Steps 1 through 5 can be redone in minutes when that happens, which is the
practical argument for keeping the decision cheap rather than for trusting an
old profile.

# Agent instructions

## Read docs/learnings.md before proposing a direction

On 2026-09-17 a day of work went into GPU idle and launch overhead measured from
nsys. Section 2b-i, written six weeks earlier, had already established that
profiled idle is dominated by the profiler and that "the GPU idles X%, therefore
launch-count work is worth Y" is unsupported. The conclusion was in the file the
whole time.

Before starting a line of work, search `docs/learnings.md` for the quantity you
are about to rely on. The file is long, so grep it --- for the metric, for the
subsystem, and for the word "wrong" or "unsupported", since its corrections are
written as corrections.

Device-side numbers from nsys (kernel times, launch counts) are trustworthy.
Host-side numbers from nsys (idle, gaps between kernels) are not a model of the
real run. Wall-clock questions are answered by the unprofiled AMIP stages.

## Numbers in prose must be injected, not typed

A figure in `docs/`, `calkit.yaml` answers, or a commit message must come from a
file the pipeline produced. `docs/learnings.md` accumulated dozens of hand-typed
numbers that nothing could check, which is how a published tag ended up citing
figures its own results file did not contain.

The mechanism is calkit's Markdown stage. The analysis lives in an annotated
code block that writes a JSON output, and the prose refers to it by key:

```md
The marginal cost is <!-- calkit value key=marginal.host_us_per_launch
path=results/experiments/launch-cost/stats.json -->6.74<!-- /calkit value --> us.
```

Values are rewritten from the results file on every run, so the prose cannot
drift from the calculation, and the rendered numbers stay visible in Git.
`docs/experiments/launch-cost.md` is the worked example.

One experiment, one document under `docs/experiments/`, registered as a
`kind: markdown` stage and cited from a `calkit.yaml` question so
`scripts/verify-evidence.py` checks it. `docs/learnings.md` keeps the reasoning
and links to those documents; it should stop carrying figures of its own.

### An answer is one or two sentences; the reasoning goes in a document

A `calkit.yaml` answer is the claim and nothing else, with only the figures it
actually quotes templated by name. The six answers here had grown to between
400 and 1135 words each -- whole arguments folded into one paragraph, which is
unreadable on the hub and impossible to check against, since a reader cannot
tell which sentence the evidence is supposed to support.

The reasoning belongs in a document under `docs/`, cited as `kind: document`
evidence with a `section`:

```yaml
    evidence:
      - kind: document
        path: docs/wall-clock-conversion.md
        section: "What kind of floor: repeats, run order and GPU assignment"
```

This is not a convention this project invented. calkit **enforces** it: a
document evidence entry with no stage building it is reported by
`calkit check questions` as

    no pipeline stage builds this document, so nothing checks what it says
    against the results; build it with a Markdown stage

so a hand-written write-up is an error in the same way a typed-in number is.
The chain runs answer -> document -> Markdown stage -> results file -> the
stage that computed it, which is what makes a long-form explanation tie back to
the pipeline instead of drifting from it. There is no separate `details:` field
and there should not be; the reasoning is evidence.

Two consequences worth knowing before shortening an answer:

  * `scripts/verify-evidence.py` reports `NOT IN ANSWER` for every name in an
    evidence `values:` map that the answer no longer uses, so trim the maps to
    what the shortened answer templates. The figures that moved into the
    document are carried by its own value markers, which is where the checking
    moves too.
  * Several questions may cite one document at different `section`s. The Val
    questions do -- the occupancy gates and the dependency scope rest on the
    same device budget, and splitting them would duplicate it. `section` is
    free text, so write the heading as a reader would find it.

A document whose figures all come from existing results files still needs one
annotated code block, because a Markdown stage with no stage in it is an error.
Give it a `values` block: it is what declares the `inputs`, so the document goes
stale when the results do, and it is doing real work rather than copying
whenever a source is TOML -- value markers read JSON only.

### Keep value markers inline, one sentence per line

A `<!-- calkit value ... -->` that starts a line ends the paragraph and opens an
HTML block, so the rendered preview breaks apart and the document becomes hard
to read. The markers are long, which makes wrapping a sentence around them the
natural thing to do, and it is wrong.

Write one sentence per line and let the line run long. No line may begin with a
marker. `docs/experiments/launch-cost.md` is the worked example.

Values are re-injected on every `calkit run`, even when the stage itself is
skipped as unchanged --- verified by corrupting a value and watching it be
rewritten. So the prose cannot drift even between runs of the analysis.

### An experiment document declares its own profiling stages

Copy `docs/experiments/TEMPLATE.md`. The document declares the nsys runs it
needs, writing to `results/experiments/<slug>/`, so the experiment owns every
artifact it cites. Nothing is shared, so nothing can be overwritten by the next
experiment, and freezing the file freezes the whole experiment at once ---
`frozen` applies to every stage a Markdown file declares.

This removes the failure rather than compensating for it. The earlier
arrangement had the write-up read `results/nsys/baseline.sqlite`, which holds
whichever experiment profiled last; keeping it honest then needed a freeze AND a
separate statistics snapshot, and getting that wrong once destroyed a
measurement outright.

Storage: a profile pair is about 1.9 GB, which is affordable only because it is
a cache rather than the record. The stage can regenerate it from the tag, so
profiles may be garbage-collected (`calkit dvc gc --workspace` after pushing);
the durable record is the statistics JSON, stored in Git at a few kilobytes.

`launch-cost.md` is the worked example: it declares its own baseline and probe
stages, writing under `results/experiments/launch-cost/`, so nothing it cites
can be overwritten by the next experiment. The write-up it replaced read the
shared `results/nsys/` paths instead and needed both a freeze and a snapshot
file to stay honest; it was deleted on 2026-09-22 once its figures were
withdrawn, and `meas/2026-09-17-launch-overhead` still has it. Do not copy that
arrangement.

### Freeze an experiment's document on main, leave it live at its tag

An experiment's document is generated from artifacts that the next experiment
overwrites. Two mechanisms keep it honest, and both are needed:

  * **Snapshot** the small derived numbers into their own committed results
    file, so the write-up does not depend on a 648 MB profile still being in
    the DVC cache (`results/launch-probe.json`).
  * **Freeze the stage on main** and tag the experiment. At the tag the stage is
    not frozen and its inputs exist, so `git checkout <tag> && calkit run <stage>`
    reproduces it; on main `calkit run` leaves it alone instead of regenerating
    the write-up against a different run. `compile-time-study` established this
    pattern; `launch-cost` follows it.

Question evidence then cites the document and its results file with
`git_ref: <tag>`, which is what `git_ref` is for: the claim is pinned to the rev
where it can be checked, not to whatever main happens to hold.

Freezing has a failure mode this project has already hit (see the ncu stages
above): a frozen stage goes quietly stale and its output can be mismatched with
whatever it is displayed beside. So a frozen snapshot must record its own
provenance --- source commit, input hashes, what was being measured --- inside
the file, as `results/launch-probe.json` does.

### A transient artifact cannot be a stage dependency

`results/nsys/mod.sqlite` holds whichever experiment ran last. The launch-cost
probe measured it, and making it an input to the write-up caused the pipeline to
regenerate the mod profile WITHOUT the probe, destroying the measurement --- and
the recovered figures came from a cache file whose `dvc.lock` entry had never
been committed.

When an experiment's artifact is transient, commit its derived statistics as
their own small results file (`results/launch-probe.json`) and depend on that.

### A name-paired subtotal is not the treatment's effect

`results/param-fold.json` carries two totals over the treated kernels.
`fold_attributable` sums the ones with a counterpart NAME in both arms;
`device_saving` falls back to the family total wherever a target is unpaired,
and records the `basis` it used. Divide by `device_saving`.

The distinction is not pedantic. A parameter fold can collapse two
specializations into one, so an arm's name covers work the other arm splits in
two and there is nothing to pair --- which is the case the fold is applied for.
On 2026-10-27 that put the fold's whole saving outside `fold_attributable`,
which reported it two orders of magnitude low, and
`compare-step-breakdowns.py` divided the measured per-step saving by it and
wrote a conversion of tens of thousands of percent into
`results/step-comparison.json`. Nothing complained: no check asserts that a
conversion is near unity.

`vs_prior_stack` has the same blind spot by construction --- it compares each
target across stacks by name, so a kernel the fold merged has no row and a
payoff that moved INTO it reads as a payoff lost. It lists `unpairable_now` for
that reason. Read it against `device_saving`, not instead of it.

The general rule: when the point of a treatment is to change the kernel
inventory, the aggregate that survives is the sum over every matching name.
Section 8g of `docs/learnings.md` has the three figures that got it wrong.

### A re-baseline makes fields unavailable, not just stale

`results/sypd-repeats.json` groups pairs by diff digest, so pulling upstream
`main` into both arms starts a FRESH group --- correctly, since it is different
code. Its spread between repeats is then 0 and its `by_order` map has one entry,
so anything reading `latest_treatment` for the noise floor gets zero and
anything reading it for run order raises `KeyError`.

Zero is the dangerous one. It flipped `diagnostics-cost`'s verdict on the
sampling residual from "a bound only" to "a measurement", because a residual
clears a 0-point floor trivially. Read `most_repeated_treatment` for the floor
and `latest_treatment` only for the headline.

While there: that same test compared `abs(residual)` against the floor. A
sampling cost cannot be negative, so a residual past MINUS the floor is not a
small share --- it is two differently-sourced runs disagreeing by more than the
quantity between them, which `sources_inconsistent` now records. Taking the
magnitude of a signed quantity reads a sign error as a resolved measurement.

### Quote the margin over the null census, not over the last null

`results/null-census.json` collects every pass whose arms `record-treatment`
found identical, split by benchmark window because an earlier window's noise is
not this one's. Read `current_window_mean_pct` as the floor and quote a headline's
margin over it. Never quote the margin over whichever null ran most recently: a
single null is one draw, and on 2026-10-22 that draw was the larger of the two
this window has produced.

`current_window_all_same_sign` is the field that says what kind of floor it is.
Both nulls here are positive, which is a confound that repeats rather than
jitter around zero --- and a confound that repeats is not reduced by averaging,
so repeating a pair tightens the wrong quantity. Three repeats of one pair agree
inside half a point while the nulls disagree by over a point; the repeats
measure repeatability of one arm pair on one stack, which is not what a headline
needs. Section 8h of `docs/learnings.md` has it.

### When a value marker's sign or boolean flips, re-read the sentences

A Markdown stage re-injects its values on every run, so the numbers in a
document cannot drift. The claims built on them can. On 2026-10-31 the
diagnostics residual crossed from negative to positive, `sources_inconsistent`
went true to false, every marker rewrote itself, and the prose between them
still explained at length that a negative residual means the two sources
disagree --- directly after a marker rendering `false`.

A document that says *why* a figure means something holds a second, uncheckable
copy of that figure's sign in its sentences, and the pipeline reports the stage
as up to date either way. There is no mechanical guard. After a pass that moves
a sign or a boolean, grep the document and the `calkit.yaml` answer for the
field name and read what surrounds it.

## Assert that every scripted edit matched

`str.replace()` returns the string unchanged when the pattern is absent, so a
scripted edit that misses leaves the file half-modified and silently wrong.

On 2026-09-16 a signature rewrite in `compute_interp_frac_press` did not match
(the pattern assumed a multi-line signature; the file had it on one line) while
the edits to the body DID match. The function then took a struct in a parameter
named for an array and referenced an undefined variable. It parsed fine, and
failed 40 minutes later on the GPU with `InvalidIRError: unsupported dynamic
function invocation (call to length)`. That was the third failed launch of one
experiment.

    assert s.count(old) == 1, f"pattern matched {s.count(old)} times"
    s = s.replace(old, new, 1)

Assert on every replacement, and after a multi-part edit verify each piece is
present before submitting a job. A parse check is not enough -- the file above
parsed.

## Use American spelling everywhere

Code comments, commit messages, docs/learnings.md, calkit.yaml answers, PR
bodies, and replies in the terminal. Write optimization, behavior, analyze,
center, modeling, normalize, rematerialization, parallelize, labeled --
not the -ise/-isation/-our/-re forms.

This is a CliMA codebase and the surrounding prose is American; mixed spelling
in the same file reads as careless. Note that `analysis`, `optimistic`,
`precise`, `basis` and `emphasis` are the same in both and must not be
"corrected".


In this project we are trying to optimize an AMIP simulation.
We have submodules for relevant packages, with and without a `-mod` suffix,
which indicates our modified version used to compare against baseline.

NVIDIA Nsight Systems profiling is set up to run as part of a Calkit pipeline,
submitting SLURM jobs on the `clima` cluster one at a time
(so we don't take up too many resources).

The logs in `.calkit/scheduler/logs` print important information like estimated
SYPD and a table of kernel resource consumption.

The general process we follow is:

1. Make some modifications to suboptimal kernels.
2. Run `calkit run mod-nsys` to profile and wait for the SLURM job to finish.
   You can use `calkit slurm queue` to view the jobs associated with this
   project.
3. If the `estimated_sypd` is significantly higher than baseline, run the
   rest of the pipeline with `calkit run`.
   If not, try some more modifications in the `-mod` suffix submodules.
4. If we have significant speedup from the longer AMIP simulation, e.g.,
   greater than 5%, stop and let the user make commits to the submodules,
   push, and run `calkit save` on the repo to push the Nsight reports to the
   cloud for archival.

## Never touch a GPU without reserving one

`clima` is a shared machine. Never run anything that uses a GPU on the login
node or outside a scheduler reservation. Reserve one first with `srun`:

```sh
srun-gpu   # alias for: srun --gpus=1 --mpi=none --time=180 --pty bash
```

This applies to everything, not just the pipeline: quick kernel benchmarks,
`CUDA.jl` sanity checks, package test suites, and one-off scripts all need a
reservation. The Calkit pipeline stages already reserve their own GPUs through
the `clima` SLURM environment, so `calkit run` is fine as-is.

## `clima` is a shared single node cluster

We should usually only have one job in the queue during working hours,
when other typically need it, so we don't back things up behind us.

### GPU 5 was bad — guard now disabled (2026-08-13)

`clima` is one node with 8 A100s. GPU 5 was faulty; as of 2026-08-13 it reports
0 uncorrected ECC errors and is in normal use, so `BAD_GPUS` is now empty and
the guard is inert. Note SLURM never stopped offering it: `gres.conf` still
reads `File=/dev/nvidia[0-7]`. To re-arm, list the bad indices in
`scripts/gpu-guard.sh` — nothing else needs to change.

Historical detail follows. SLURM has no "allocate any GPU except N" flag,
so `scripts/gpu-guard.sh`
handles it from inside the job: SLURM sets `CUDA_VISIBLE_DEVICES` /
`SLURM_STEP_GPUS` to the *global* device index, so the guard checks that index
and calls `scontrol requeue` if it landed on a bad one. It is sourced at the top
of `run-nsys.sh`, `run-ncu.sh`, and `run-julia-script.sh`.

For interactive work, check what you were given and drop the allocation if it is
GPU 5:

```sh
srun-gpu
echo $CUDA_VISIBLE_DEVICES   # exit and re-run srun-gpu if this is 5
```

The proper fix needs a cluster admin, since `/etc/slurm/gres.conf` is root-owned:
change `File=/dev/nvidia[0-7]` to `File=/dev/nvidia[0-4,6-7]` and run
`scontrol reconfigure`.

Once the hardware is fixed, turn the guard off by setting `BAD_GPUS=""` in
`scripts/gpu-guard.sh` — and **do not delete the `source` block from the run
scripts**. `gpu-guard.sh` is deliberately not a stage dependency, so editing the
bad-GPU list invalidates nothing. But `run-nsys.sh`, `run-ncu.sh`, and
`run-julia-script.sh` *are* dependencies of the profiling stages, so removing the
block from them would invalidate every stage and force hours of re-profiling for
no scientific reason.

The `make-diffs` stage collects up all changes across all packages and
puts them in the `diffs` folder, so it can be archived along with the results.

This history of this repo serves as a record of numerical experiments.
If we see something interesting we should save it in the history by
running the pipeline so we ensure it's reproducible.

## Updating ClimaCoupler to reflect the latest `main`

We are running on branches that should be identical to `main` except for the
changes to their `experiments/AMIP/Manifest-v1.11.toml` files,
which dev in the local packages in the submodules.
So we can `cd` into each Coupler submodule and run

```sh
bash scripts/update-coupler-to-main.sh
```

## Freeze and unfreeze the two ncu stages together

`baseline-ncu` and `mod-ncu` are frozen by default because they cost ~1.5 hours
each, more than half the pipeline's cluster time. When they need refreshing,
**unfreeze both, run both, and re-freeze both** — never one alone.

`results/kernel-resources.csv` puts the two arms in adjacent `_baseline` and
`_mod` columns, so it reads as a single comparison. It only is one if both
exports came from the same revision. Freezing the stages separately breaks that
silently: on 2026-09-03 the baseline column turned out to be from 2026-08-20 (a
null test, both arms identical) and the mod column from 2026-08-23 (the
clear-air early-out), three days and one experiment apart. Nothing in the table
said so, and it had already been shared outside the repo.

`scripts/export-kernel-resources.py` prints each arm's source revision and
treatment when it runs, so the stage log records what the table describes. That
is a check, not a guard — the guard is running the two stages as a pair.

The same applies to any future pair of arm-specific stages.

## Use `calkit push`, not `git push`

**Enforced by a hook.** Run `bash scripts/install-git-hooks.sh` once per clone.
The `pre-push` hook refuses a bare `git push` while DVC-tracked data is missing
from the remote. `calkit push` sends DVC first and Git second, so it passes the
hook; `git push --no-verify` overrides it when Git-only is genuinely meant.

The hook exists because knowing the rule was not enough: on 2026-09-17 a whole
session's commits went out with `git push` inside compound commands, and the
user had to run `calkit push` by hand to ship the artifacts.

`git push` sends the git objects only. The pipeline's real outputs -- nsys
reports, their sqlite databases, the ncu exports -- are DVC-tracked, so they stay
in the local cache and reach nobody. `calkit push` sends both.

This is silent. `git status` is clean, `calkit status` reports "DVC: No changes"
(which means the local cache matches the workspace, NOT that the remote has the
data), and the local clone works because the cache is right there. It fails only
for someone else, and only when they try to open a report.

It also breaks a claim this repo makes about itself. `experiments.csv` advertises
that any row reproduces from `git checkout <commit> && git submodule update
--init --recursive && calkit run`, and an audit of submodule pointers can pass
while every nsys and ncu artifact is missing from the remote. That happened on
2026-09-09: pointers were verified fetchable for all ten tags, `git push` had
been used throughout, and five artifacts were sitting only in the local cache.

`calkit save` does the same for the archival case described above.

## The benchmark window, and why `--config_file` overrides are forbidden

`t_end` is **345600secs** (four simulated days, 2880 coupler steps at
dt = 120 s) and `scripts/run.jl` profiles **120 steps**. Both numbers are
deliberate:

  * The window mirrors `config/nightly_configs/amip_coarse_progedmf_1m_land.yml`,
    which is the configuration whose speed anyone cares about. It did not always:
    one simulated day at dt = 30 s put `sim_duration` one timestep below
    ClimaAtmos's `duration >= 86400` bucket, so the default diagnostics came out
    hourly instead of daily -- 24 write events a day rather than one, taking
    32.1% of the wall clock and leaving ordinary steps 51.9%. An every-step
    saving measured there reports a fraction of itself, silently. Fixed in
    CliMA/ClimaAtmos.jl#4900; the window no longer depends on the threshold
    either way.
  * 120 profiled steps = four simulated hours, a whole number of radiation
    (30-step), cloud-fraction (30-step) and gravity-wave (15-step) cycles, so
    each fires its exact long-run share wherever the window starts and the
    profile needs no offsetting or reweighting.
  * It is **not** a whole number of diagnostics cycles: the daily write is 720
    steps and the extra monthly-average block samples every 180. So the nsys
    profile contains no write event and reports the sampling as ordinary
    every-step work. Diagnostics cost is measured by differencing a
    diagnostics-off run instead -- the `amip-nodiag` and `diagnostics-cost`
    stages, which is also why the latter cannot derive the two halves of the
    cost from one measurement.
  * The 1200secs / 40-step window that preceded both was shorter than the
    gravity-wave period, so neither scheme ever fired. It was also about half
    first-call compilation: 0.2767 s per coupling step reported against 0.146 s
    measured in steady state with CUDA events.

The stale periods above outlived the config change by weeks, in this file, in
`scripts/run.jl` and in two `calkit.yaml` explanations. `run.jl` is an input to
both nsys stages, so correcting its comment costs ~2 GPU-hours of re-profiling
and is worth bundling with a pass that re-runs them for a real reason. It was
corrected on 2026-10-19, bundled with the re-baseline onto ClimaCore `cfdd310`
and ClimaAtmos `352fe98`, which had already invalidated both nsys stages -- so
the fix cost nothing.

Both arms' benchmark configs are edited **in the submodules** and must stay
byte-identical to each other. That is a deliberate deviation from "identical to
upstream main except the Manifest" and is the only one.

**Do not change these by passing an extra `--config_file`.** It does not merge.
The additional file drops the primary config's coupler keys back to defaults: on
2026-09-10 it silently changed `dt` and `dt_cpl` from 30secs to 400secs in all
four arms, a 13x coarser physics timestep, and invalidated a full pipeline run.

The failure is worth recognizing by shape, because the SYPD looked spectacular:
**0.297 -> 2.986, an apparent 10x.** The tell was that *walltime per coupling
step went UP*, 0.2767 -> 0.3667 s. A real speedup lowers it. Throughput per
simulated year rose only because 13x fewer steps were being taken per simulated
hour -- a different, coarser problem, not a faster solution to the same one.
**When SYPD and per-step walltime move the same direction, the timestep changed.**

## Never accumulate in a top-level Julia loop

`count += 1` inside a `for` at top level creates a *new local*, so the loop dies
with `UndefVarError` on the first iteration. Julia warns about it, but only
after the job has already started, which on this cluster means the failure costs
a GPU reservation and ~10 minutes of model warmup.

This has now bitten four separate measurement scripts. Always put the
accumulation in a function and return the counts:

```julia
function tally(args...)
    n = 0
    for ... ; n += 1 ; end
    return (; n)
end
(; n) = tally(args...)
```

Measurement scripts are the usual victims because they are written quickly and
run once. Run them for a single step locally before submitting if possible.

## A package the arms do not dev cannot be imported by name

`import CloudMicrophysics...` worked only while CM was dev'd in the mod arm.
After the 2026-09-14 re-baseline neither arm devs it, so it is in the AMIP
manifest but not in `Project.toml`, and importing it fails. Reach such a package
through one that *is* a direct dependency -- `CA.BMT`, `CA.TD` -- rather than
adding it to the environment, which would change the manifest and invalidate
every profiling stage.

## A tag is only meaningful if the whole pipeline is green at it

Before tagging, run `calkit status` and confirm no stale stages. Running one
stage and reading its output directly bypasses the consistency the pipeline
exists to enforce: the downstream stages that write `results/top-kernels.csv`,
`results/summary.json` and `results/treatment.json` still hold the previous
run's numbers, and nothing says so.

That published `meas/2026-09-13-binary-search-alone` with a `top-kernels.csv`
carrying the previous configuration's radiation figures, cited in the Q&A for
numbers it did not contain. The tag was deleted and the measurement redone.

`scripts/verify-evidence.py` checks every evidence item resolves at its
`git_ref` **and** carries the value its answer quotes. Run it before pushing a
tag. A check that only confirms the file exists is what let this through.

Editing a stage input after a run makes its stage stale even when the edit
cannot change behavior -- a comment rewrite counts. Either re-run before
tagging or tag the commit that was measured.

## Push every submodule before pushing the superproject

A committed submodule pointer to an **unpushed** commit is exactly as
unreproducible as a dirty tree, and far less visible. The superproject pushes
without complaint, `git status` is clean, and the local clone resolves fine
because the objects are still in the local store. It only fails for someone else:

```
fatal: remote error: upload-pack: not our ref 5ed945da...
fatal: Fetched in submodule path 'ClimaCoupler.jl-mod', but it did not contain
       5ed945da... Direct fetching of that commit failed.
```

That happened on 2026-09-09 to `ClimaCoupler.jl-mod`. The check below
used to test the exit status, which is always 0, so on 2026-10-14 it passed on an
unpushed `ClimaAtmos.jl-mod` pointer; count the output lines instead. Non-`-mod`
arms carry pointers too, so check them all. The cause is worth knowing
because it is not carelessness about the important repo — it is the opposite. The
three submodules with *code* changes were pushed; the one with only a *config*
change was forgotten, and its commits were the ones three experiment tags
depended on most directly, because the config value **was** the treatment.

Check before pushing:

```sh
for d in *-mod *.jl; do
  [ -d "$d" ] || continue
  ptr=$(git ls-tree HEAD "$d" | awk '{print $3}')
  [ -n "$ptr" ] || continue
  # Test the OUTPUT, not the exit status: `branch -r --contains` exits 0 and
  # prints nothing when no remote branch holds the commit.
  n=$(git -C "$d" branch -r --contains "$ptr" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || echo "$d NOT PUSHED ($ptr)"
done
```

To audit the whole history rather than just HEAD, loop the same check over
`git tag --list 'exp/*'` with `git ls-tree <tag>`.

### Do not squash a submodule branch that tags point into

Squashing away an experiment-and-revert pair is tidy for reviewers and
catastrophic for reproducibility: it orphans every superproject commit whose
pointer lands in the squashed range, silently. `git status` stays clean and the
local clone still works.

Both cases arose in this project and resolved oppositely:

  * **ClimaCore, 2026-09-06.** The net change was comment-only, no tags pointed
    into the range, so the experiment and its revert were squashed to one commit
    before pushing. That orphaned five *superproject* commits, which then had to
    be squashed as well -- repointing them was not an option, since they
    genuinely ran the reverted code.
  * **ClimaCoupler, 2026-09-09.** Also three experiment/revert pairs with an
    empty net diff, so equally tempting. But `exp/2026-09-08-quadrature-order-1`,
    `exp/2026-09-08-quadrature-order-2-isolated` and
    `exp/2026-09-09-order2-full-stack` point at the *intermediate* commits.
    Pushed in full.

The rule: check whether any tag's pointer lands inside the range before
rewriting it. If one does, push the history as-is.

## Never force a stage with `calkit run --force`

`calkit run <stage> --force` does not force that stage. It forces the stage and
everything upstream of it, so a one-second CPU stage re-runs the whole pipeline
--- on 2026-10-10 `calkit run build-experiment-index --force` submitted
`baseline-nsys` twice and cost two GPU reservations on a shared machine for a
stage that reads git tags and writes a CSV.

Force exactly one stage with DVC's own flag instead:

```sh
calkit dvc repro --single-item --force <stage>
```

`--single-item` is the part that matters: it runs the stage without its
dependencies.

### A scheduler stage needs `CALKIT_FORCE=1` as well, or it measures nothing

`--force` makes DVC re-run the stage *command*, and for a SLURM stage that
command is a submission wrapper, not the model. It looks for its own previous
job first, finds a completed one whose deps and command are unchanged and whose
exit code was 0, and harvests that result instead of submitting:

```
Job 'amip-baseline' already left the queue; using its result
```

The stage then "succeeds" in seconds and `dvc.lock` updates, so nothing in the
output says the measurement did not happen. That is the failure mode to
recognize: a forced GPU stage that finishes before a GPU could have started.

The branch is guarded by `elif not os.environ.get("CALKIT_FORCE")` in calkit's
`cli/scheduler.py`, so set it in the environment for a re-measurement:

```sh
CALKIT_FORCE=1 calkit dvc repro --single-item --force <stage>
```

This is how an independent repeat of an AMIP pair is taken --- the two pairs in
`results/sypd-repeats.json` ran the same committed code, and the second needed
both flags. Note the ordering constraint: a repeat **deletes and rewrites** the
scheduler logs, which are the stage's declared outputs, so record the previous
pair with `scripts/record-sypd-repeat.py` and commit it *before* forcing the
next one.

The cleanup is worse than the waste, because the cancelled job takes the
pipeline with it. A GPU stage declares its scheduler log
(`.calkit/scheduler/logs/<stage>.out`) as an OUTPUT, and DVC deletes a stage's
outputs before running it, so cancelling mid-run leaves the log rewritten and
the real artifacts missing. The stage then reads as stale on its own output and
drags `summarize` and `compare-kernels` with it. To recover: restore the log
with `git checkout HEAD -- .calkit/scheduler/logs/<stage>.out`, restore
`dvc.lock` the same way if the forced run updated it, then `calkit dvc checkout`
to pull the artifacts back out of the cache --- and check nothing is running
first, because a second `calkit run` will delete them again.

That recovery only worked because the artifacts were still in the DVC cache.
`calkit dvc gc --workspace` had been run earlier in the same session, before the
measurement, which is why it was safe; running it after a force-cancel would
have destroyed a 6-hour measurement.

## A stage that reports git state must always run

`record-treatment` records submodule SHAs, refs and whether a tree was dirty.
DVC decides staleness from file contents, and committing a submodule changes no
content, so the stage was skipped as unchanged while exactly what it reports had
changed --- it kept reporting a stale SHA and a dirty tree that was no longer
dirty, and `reproducible` stayed false after the problem was fixed.

It now carries `always_run: true`. It is CPU-only and takes seconds.

The general rule: if a stage's output depends on something DVC cannot hash ---
git metadata, the wall clock, the machine --- declaring inputs will not keep it
honest, so make it always run.

## Commit before running anything meant for the record

`experiments.csv` advertises that any row can be reproduced with
`git checkout <commit> && git submodule update --init --recursive && calkit run`.
That restores the **committed** state. A run made against a dirty working tree
measured code that exists nowhere afterwards, so the recipe cannot reproduce it —
however good the numbers are.

`record-treatment.py` writes a top-level `reproducible` flag and `dirty_trees`
list into `results/treatment.json`, and the experiment index carries a
`reproducible` column: `yes`, `NO (dirty tree)`, or `unknown` for rows predating
that file. Non-reproducible rows are visible in the log rather than buried.

Only **submodule** dirtiness counts. The superproject is dirty during every run —
`record-treatment` executes mid-pipeline while results and logs are being
written — so counting it would flag every row and mean nothing.

Iterating dirty is fine and is what the normal loop does. Just commit the
submodule change before the run whose result you intend to tag. It is also worth
not editing a stage input while a run is in flight: it invalidates stages
mid-pass and forces a re-run for a change that may not even affect behavior.

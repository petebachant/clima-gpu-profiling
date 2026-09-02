# What we have learned optimizing the AMIP GPU run

This is the long-form companion to the `questions` section of `calkit.yaml`.
That file is the entrypoint: it states each question, its answer, and the
evidence the answer rests on. This document holds the reasoning, the failed
attempts, and the measurements that are too long to live there.

Every number below came from a run in this repository. Where a result is a
single measurement rather than a reproduced one, it says so.

> **A note on how this is wired up.** `calkit.yaml` points at this file with a
> question-level `explanation: docs/learnings.md`, not as an entry in the
> question's `evidence` list. That distinction is the point: evidence is data
> supporting a claim -- a figure, a table, a result key, each of them a pipeline
> output that can be regenerated and checked -- whereas this is an elaboration
> of the claim itself, hand-written prose that no stage produces. Filing it as
> evidence produced an entry whose explanation read "see this for the
> explanation", which is a good sign the model was wrong.
>
> No such field exists in Calkit today, so this is a proposed extension; Calkit
> tolerates the extra key locally. The two earlier attempts are recorded because
> the reasoning may be useful upstream: a `documents:` object list with
> `kind: document` evidence (rejected -- it forces a new object kind for every
> sort of file, and the nearest existing kind, `publications`, wrongly implies
> intent to publish), and evidence carrying a bare `path` with no `kind`
> (reasonable on its own merits, since the path already identifies the artifact
> and the kind is derivable, but it does not fix the evidence-versus-explanation
> confusion).

## 1. How to read a result here

### Two different numbers are both called SYPD

- **The AMIP-stage figure** (`Info: SYPD:` in `amip-*.out`) is what
  `scripts/summarize.py` turns into `speedup_pct`, and it is the one that
  belongs in `experiments.csv`.
- **The nsys-stage figure** (`sypd ≈` progress callbacks) is instantaneous and
  runs far higher — 0.337 against 0.272 for the same baseline on 2026-08-24.

They are not comparable, and confusing them will make a treatment look like it
moved the baseline when it did not. `summarize.py` reports the nsys number
separately as `nsys_speedup_pct`; treat it as a leading indicator only.

### Three different numbers are all "the hot kernel's duration"

The same L1013 kernel is timed by three instruments, and they do not agree
because they are not measuring the same thing:

| number | source | what it is |
|---|---|---|
| **26.29 ms** | `results/top-kernels.csv` | nsys mean over the profiled AMIP run |
| **30.86 ms** | `results/ncu/*-details.csv` | Nsight Compute `Duration`, inflated by ncu's serialization and replay |
| **26.00 ms** | `results/launch-bounds/launch-bounds.toml` | the `launch-bounds-study` harness, 50 calls in its own process |

Never build a ratio out of two of them. Doing exactly that — dividing a scratch
total from one instrument by an ncu duration from another — is what produced the
understated 0.76% figure that §4 used to carry. Within a section, pick one
instrument and stay on it: §2's cost model is nsys throughout, §4's launch-bounds
comparisons are `launch-bounds.toml` throughout.

### A tight noise floor is not the same as sensitivity

Four null tests (byte-identical arms) landed at −0.43%, −0.09%, −0.06% and
+0.21%, so the metric is *repeatable* to a few tenths of a percent. That is a
statement about variance, not about what the metric can detect.

The launch-bounds experiment made this concrete: a **−2.11% reduction in total
CUDA kernel time**, reproduced across two runs, produced a SYPD change of
**−0.016%** — indistinguishable from zero. The measured window is short (1200 s
simulated, 40 steps) and dominated by first-call compilation, so a change that
only makes already-compiled kernels faster has little room to show. Roughly 88%
of a profiling job is host-side Julia JIT.

**Consequence:** a few percent of GPU kernel time is below this benchmark's
resolution. To move SYPD by 5% you need to remove *work*, not reschedule it.
The one historical change that did move it by that much — the CloudMicrophysics
register work, +15–17% — cut the hot kernel roughly in half.

### What each stage costs

| stage | typical elapsed |
|---|---|
| `*-ncu` | 1:21 – 1:39 |
| `*-nsys` | 0:44 – 0:53 |
| `amip-*` | 0:39 – 0:55 |

The two ncu stages are over half the pipeline's cluster time, so they are frozen
by default. nsys already reports registers per thread, and `launch-bounds-study`
records registers, occupancy target and spill growth directly. Unfreeze both ncu
stages *together* — otherwise the two arms describe different revisions — only
when a claim genuinely rests on measured spill traffic, stack frame,
achieved-versus-theoretical occupancy, or stall reasons.

## 2. What actually limits the flagship kernel

The hot kernel is the SGS-quadrature environment broadcast in
`ClimaAtmos/src/cache/microphysics_cache.jl`, about **16% of GPU time**.

### It is latency-bound, not any of the usual things

| | |
|---|---|
| registers / occupancy | 255 per thread, 12.4% achieved, "limited by the number of required registers" |
| pipes | ALU 27%, FMA 22%, XU 16% — nothing saturated |
| memory | DRAM 4.7%, L2 7%, L1 3.3% |
| dominant stall | fixed-latency execution dependency, 2.5 of 5.9 cycles (42%) |
| IPC | 1.34 of a possible 4 |

So the standard GPU playbook does not apply. Coalescing and layout are
irrelevant at 4.7% of DRAM peak; there are ~1.55M independent points, so it is
not parallelism-starved; and no pipe is near its roofline. Memory is the *idle*
resource and occupancy is the exhausted one — which is what makes tabulation
interesting and makes anything that adds spill traffic dangerous.

### The 255 registers are one CloudMicrophysics evaluation

The single-call updraft kernel (one evaluation per point, no quadrature) uses
**246 registers**. The nine-point quadrature kernel behind its `@noinline`
barrier uses **255**. The barrier is working, and the quadrature scaffolding
costs **9 registers**. On sm_80 the occupancy steps are 255 → 8 warps/SM,
168 → 12, 128 → 16, so nothing that saves 9 registers can change occupancy.

A caution on mechanism: `@noinline` does **not** "free the callee's registers."
Under the NVPTX ABI the kernel's register count is the maximum over its whole
call graph, so a non-inlined callee's frame still sets the allocation. That is
precisely why the kernel sits at 255 despite the barrier holding.

### A cost model for the kernel

The updraft kernel runs `n_substeps = 3`; each quadrature point runs
`n_substeps_quadrature = 2`. Normalising the two measured times
(3.63 ms and 26.29 ms / 9 = 2.92 ms per point):

```
a + 3b = 3.63     →   b ≈ 0.70 ms per substep
a + 2b = 2.92         a ≈ 1.51 ms fixed, per evaluation
```

`a` is a bound: the environment path does *more* fixed work per point than the
updraft path (transform, `q_sat(T̂)`, condensate diagnosis), so `b > 0.70` and
`a < 1.51`. Two consequences:

- **Fixed cost is ~52% of the kernel** (9 × 1.51 ≈ 13.6 ms of 26.3 ms).
- **Removing a quadrature point saves ~2.9 ms — about 11% of the kernel** — and
  attacks both halves, whereas cutting substeps only reaches the other ~48%.

## 2b. The kernel population, and where the time actually goes

Per-kernel tables answer "which kernel is slowest". They cannot answer "are there
too many kernels", and that turns out to matter more. From
`results/kernel-population.json`, regenerated by the `kernel-population` stage
from the nsys SQLite exports:

| | baseline |
|---|---|
| launches | 31,140 over 10 steps = **3,114/step** |
| distinct kernels | **343** |
| **GPU idle** | **30.9%** of the profiled span |
| launches under 25 µs | **17,941 (57.6%)**, for **13.4%** of kernel time |

More than half of all launches are short enough that the ~7.5 µs of host time to
*issue* one is comparable to running it, and together they account for an eighth
of the work. Meanwhile the GPU idles nearly a third of the time, of which launch
issuing is roughly a third (3,114 × 7.5 µs ≈ 23 ms against ~73 ms idle per step).

That splits the waste in two, and only the first has been worked:

- the GPU is **busy but inefficient** — occupancy, registers, spilling;
- the GPU is **idle** — kernel count, launch overhead, host gaps.

Idle time converts to wall clock far more directly than kernel time does (see the
conversion factor in section 5), so kernel-*count* work may beat kernel-*time*
work even though its share of GPU time looks small. By subsystem, spectral
element operators are 1,310 launches and 5.8% of kernel time; generic broadcasts
and `copyto_foreach` are 4,050 launches and 18.7%.

### 2b-i. Most of that idle is the profiler, not the model (2026-09-01)

The paragraphs above treat GPU idle as an optimization target. **They are wrong,
and the correction matters more than the original claim.** Instrumenting
`scripts/run.jl` with per-step `Base.GC_Diff` and then attributing the idle
directly:

| candidate cause | verdict |
|---|---|
| Julia GC | **ruled out** — `gc_pauses = 0` on every profiled step, `alloc_MB` flat at 7.9 |
| kernel compilation / allocation / sync | **ruled out** — 1.3 ms of CUDA API time inside 326.9 ms of large-gap idle |
| OS calls (mmap, futex, I/O) | **ruled out** — 41.1 ms of 344.5 ms, 12%, mostly driver `ioctl` |
| **Nsight Systems itself** | **the leading contributor** |

The `PROFILER_OVERHEAD` table accounts for **137.8 ms of 663.8 ms of idle
(20.8%)** explicitly, and it has exactly one row per large gap. CPU sampling
inside those gaps is dominated by `libToolsInjection64.so` — 78 of 223 samples,
35%, plus `pthread_mutex_lock`/`unlock` that is very likely the same library's
internal locking. Julia frames appear at 3 samples each.

The mechanism is the tracing volume: **1,020,691 traced CUDA API calls against
31,180 kernel launches — 33 per launch.** Every one carries injection cost, and
only the buffer flushes land in `PROFILER_OVERHEAD`; the per-event cost is
smeared across the timeline and never attributed.

**Consequence: do not project an idle fraction measured under nsys onto the real
run.** The two windows are not even measuring the same thing — the nsys window
runs 222.2 ms/step while the unprofiled AMIP stage runs 297.6 ms/step, so the
profiled window is not a scaled version of the real one. Any "the GPU idles
31%, therefore fusion/launch-count work is worth X" argument built on the
numbers in §2b is unsupported. Kernel *times* from nsys remain trustworthy
(they are device-side measurements); the *gaps between them* are not a model of
the real run's host behaviour.

The honest ground truth for wall-clock questions is the AMIP stage, and the one
calibration we have from it is the conversion ratio in §3a: −5.37% GPU kernel
time bought +1.87% SYPD.

**Compute idle from the union of kernel intervals, not from summed durations.**
An earlier hand analysis summed the per-kernel-name summary table, missed rows,
and concluded the GPU was ~50% idle when the timeline says ~31%. The union is
also the figure to hand anyone working on fusion: their change should move
launches and idle, not necessarily kernel time.

## 2c. Where the hot kernel's registers actually go

Measured by the `cm-registers` stage, which compiles each layer and reads its
register count rather than running a simulation. Two halves: CloudMicrophysics on
its own, and the ClimaCore broadcast it runs inside, on the real AMIP layout.

**Revision, because this stage is cheap and therefore not frozen.** The table
below is CloudMicrophysics.jl-mod at `68007fb10` with ClimaCore.jl-mod on
`pb/launch-bounds`. The broadcast half is read out of
`ClimaCoreCUDAExt.LAUNCH_BOUNDS_CACHE` and exists only on that branch — with
upstream ClimaCore the stage logs "ClimaCore lacks the launch-bounds record;
skipping the broadcast layers" and writes a *partial* file rather than failing.
A 2026-08-31 run with CloudMicrophysics 12 commits further along (`cbfc5954`)
and ClimaCore back on main produced exactly that: LinearizedAverage at 112
registers / 480 B local instead of 163 / 32 B, and no A–D rows. Two variables
moved at once, so it is a confounded measurement of a different revision pair,
not a correction to this one. Always check for the A–D keys before reading a
comparison out of `results/cm-registers.toml`.

| layer | registers | warps/SM |
|---|---|---|
| CM source terms only | 32 | 16 |
| CM + aggregate | 48 | 16 |
| CM one linearized implicit step | 58 | 16 |
| **CM full `LinearizedAverage`** (nsub 1/2/3) | **163** | 12 |
| ClimaCore bare broadcast, 3 fields in | 50 | 16 |
| ClimaCore broadcast, 7 fields in | 48 | 16 |
| **+ 4-field NamedTuple out** | **68** | 16 |
| **+ the microphysics call** | **255** (capped) | 8 |

Reading, in order of what it overturns:

**Field count is free.** Going from three input fields to seven costs −2
registers: reads stream through rather than being held. So none of the gap
between a standalone evaluation and the real kernel is field access, which is
what one would naturally assume it to be.

**The substep count is free.** 163 registers at `nsub` = 1, 2 and 3 alike, so
`microphysics_n_substeps_quadrature` buys accuracy at the cost of time but not of
occupancy.

**The framework costs 68 registers** — 48 for ClimaCore's broadcast and indexing
machinery, plus 20 to hold four output accumulators live. Worth noting the bare
broadcast floor of 48 exceeds CloudMicrophysics' entire source-terms layer at 32.

**But the physics still dominates: roughly 70/30.** The parts sum to 68 + 163 =
231 against an observed 255, and 255 is the hardware ceiling, so the true demand
is higher still — consistent with the real kernel's 2208-byte stack frame and 21%
spill overhead. An intermediate reading of this data, that the register problem
was mostly framework rather than physics, was wrong: it came from differencing
246 − 163 = 83 before the framework had been measured directly.

**Consequence.** Reaching 168 registers, the next occupancy step, means cutting
roughly 63 from a demand of at least 231. Neither lever is obviously enough
alone, but they are additive: trim the 20-register NamedTuple output *and* bring
CloudMicrophysics from 163 toward ~120, and the kernel crosses into 12 warps/SM,
at which point the launch-bounds mechanism converts it with no further work. That
is the same superadditive structure `experiments.csv` recorded for `cm-and-core`.

**Method note.** Pass parameter structs as kernel *arguments*. As `const`
globals the compiler folds their fields and reports 101 instead of 163 for the
full evaluation — understating pressure by 62 registers, enough to invert a
conclusion.

## 3. What has worked

| change | effect | notes |
|---|---|---|
| CloudMicrophysics register/spill work | ~5% SYPD alone | removes spilling in the hot kernel |
| ClimaCore automatic register cap | ~0.1% alone, ~9% combined | needs CM's headroom first; **superadditive** |
| Clear-air early-out (ClimaAtmos) | +1.79% SYPD, kernel −21.5% | skips clear, subsaturated, precipitation-free quadrature points |
| Launch bounds (ClimaCore) | kernel −2.11%, **SYPD null** | `exp/2026-08-24-launch-bounds`; accepts 9 of 13 candidates, wins −22.4% on L970, correctly rejects L1013. See §4 and §6.5 |
| **Source-term/linearize fusion (CloudMicrophysics)** | **+1.87% SYPD**, kernel −5.37%, L1013 −27.2% | `exp/2026-09-01-cm-fuse`, CM `pb/1m-spill-fuse`. Combined with the `sd` rematerialization already on `pb/1m-spill`. See §3a |

### 3a. The fusion result, and the two things it overturned

`pb/1m-spill-fuse` (CM `2d583673`) folds each 1M source term into the
linearization accumulators at its point of computation instead of materializing
all eighteen first — bit-for-bit, since each accumulator still receives its
contributions in `_linearize`'s order. Measured against a baseline whose
ClimaAtmos and ClimaCore are byte-identical, so the delta is CloudMicrophysics
alone:

| | baseline | mod | Δ |
|---|---|---|---|
| AMIP SYPD | 0.27079 | 0.27596 | **+1.87%** |
| total GPU kernel time | 1685.0 ms | 1594.4 ms | −5.37% |
| hot kernel L1013 | 279.2 ms | 203.4 ms | **−27.2%** |
| updraft kernel L970 | 37.6 ms | 26.3 ms | −30.2% |
| launches | 31,180 | 31,180 | 0.00% |

Every other top-ten kernel moved by ≤0.11% and the launch count is identical to
the unit, so the isolation is clean.

**The nsys screen was sign-wrong, not merely noisy.** Its estimated SYPD read
the mod arm as 3–4% *slower* (0.3579/0.3650 against 0.3738/0.3737), and
`nsys_speedup_pct` recorded −1.12% against the AMIP stage's +1.87%. The process
in `AGENTS.md` — profile with `mod-nsys`, and only continue to the full pipeline
if `estimated_sypd` is significantly higher — would have discarded this change.
§1 called that figure a leading indicator; it is now demonstrated that it can
invert on a real effect. **Do not use it as a stop rule.** Its value is
diagnostic (per-kernel times, launch counts), not directional.

**Standalone register counts do not predict the real kernel.** The
`cm-registers` stage measures a `LinearizedAverage` compiled on its own. With
the fusion it reports **116 registers / 536 B local**, *worse* than the 112 /
480 without it — while the real kernel got 27% faster. The reason is that the
standalone layer is a small function where the allocator has room, whereas the
fusion's benefit is specifically about liveness at the source-terms/linearize
boundary inside a 255-register kernel carrying nine quadrature points. Use
`cm-registers` to decompose *where* registers go, not to predict whether a
change will help.

**And the conversion ratio is the reusable number.** −5.37% GPU kernel time
bought +1.87% SYPD: about 35% pass-through. Launch bounds got −2.11% → −0.016%,
essentially zero. The difference is that this removed *work* rather than
rescheduling it, which is what §1 predicted would be required. Use ~1/3
pass-through as the estimator for a work-removing change, and ~0 for a
rescheduling one.

The superadditivity is the important pattern: the register cap was worth nothing
on its own because the kernel already spilled at its natural register count, and
worth a great deal once CloudMicrophysics had created headroom. Expect
occupancy-side and work-side changes to multiply, not add.

Launch bounds belongs in this table rather than the next one, and the distinction
is worth stating plainly because it has been misread once already: what §4
records as a failure is *forcing the annotation onto L1013*, which the shipped
guard refuses to do. The mechanism itself does what it was built to do — it just
has little to convert until CloudMicrophysics frees registers, so its measured
end-to-end effect today is a null rather than a loss.

## 4. What has not worked, and why

Recorded at least as carefully as the successes, because each one closes a
direction that otherwise looks attractive from the profile.

**Per-kernel L1 carveout** — −25.5% SYPD, +47.6% GPU time. A device-wide
per-kernel change that regressed everything.

**Register cap with a 1024-byte spill budget** — drove the hot kernel to 64
registers, −5.51% SYPD. A hard `maxregs` cap makes ptxas spill whatever does not
fit; below about 168 registers the spill costs more than the occupancy buys.

**GPUCompiler's on-disk kernel cache** — a no-op in this stack. The preference is
honoured (`disk_cache_enabled()` returns true) but compiling a kernel leaves 0
files in `disk_cache_path()`. Irrelevant anyway, since the cost is host-side JIT.

**AMIPWarmup (PrecompileTools)** — made the job 5.1% *slower*, with the penalty
landing in `CoupledSimulation`, the phase it targeted. It works in isolation
(grid build 45.4 → 32.4 s) but ~21 s of provable saving is ~1% of a 1836 s job,
so the ceiling was inside the noise before the package was written. Doing that
arithmetic first would have been cheaper than the experiment.

**Forcing launch bounds on the hot kernel** — ptxas *can* reach 168 registers and
12 warps/SM, but it costs +344 bytes of spill per thread and the kernel runs
**31.33 ms against 26.00 ms, 20.5% slower**
(`unguarded.kernels.L1013.mean_ms` against `off.kernels.L1013.mean_ms`).
Occupancy on this kernel is not reachable by codegen alone. The spill-growth
guard that rejects it is load-bearing: without it the change is net negative,
since the hot kernel is 16% of GPU time against the 2.3% of the kernel that
benefits.

Note that it is the *guard* that declines, not the kernel that fails. Of the two
rejection conditions in `uncached_launch_bounds`, the occupancy check passes —
the recorded decision has `bounded_regs = 168`, `bounded_warps = 12`,
`target_warps = 12` — and it is the spill check that fires, 344 bytes against a
256-byte budget. Saying the kernel "cannot reach 12 warps/SM" gets the mechanism
backwards and has already misled one reading of this file.

The reason it cannot afford the trade is that it had no slack left. ptxas can
meet a register target by rematerializing and rescheduling (nearly free) or by
spilling (costly), and prefers the first; L1013's demand is at least 231
registers, it is already pinned at the 255 ceiling, and it is already spilling
(2208-byte stack frame, 21.15% spill request overhead), so the whole reduction
comes out as new spill. The decision table shows that as a gradient — how far
above 168 a kernel starts predicts what reaching it costs:

| unbounded registers | spill growth | annotated |
|---|---|---|
| 255, 255, 255, 254 | 384, 344, 344, 336 B | **no** |
| 254, 234 | 192, 240 B | yes |
| 246 | 88 B | yes |
| 214, 202 | 56, 32 B | yes |
| 201, 196, 187, 173 | **0 B** | yes |

Everything starting at 201 or below reaches 12 warps/SM for free. All four
rejections are at 254–255, and the hot kernel is the worst case in the run. The
mechanism works best exactly where it matters least, which is why the treatment
nets out at −2.11% kernel time and a SYPD null. It also sharpens §6 item 5: if
one CloudMicrophysics evaluation brought the kernel's natural demand to ~200, it
would land in the free-of-charge band and convert with no spill at all.

**Point-level early-outs underdeliver because of warp divergence.** A warp
covers 32 grid points and pays for the slowest of them, so a per-point skip only
helps when the skippable points are spatially clustered. Measured on an AMIP
state, 77.7% of cells need no microphysics at all — no condensate at any
quadrature point, no precipitation — yet the clear-air early-out built on exactly
that condition won 21.5% of the kernel, not 77.7%. If skippable points were
scattered, `0.777^32` ≈ 0.02%, so essentially no warp would skip; the 21.5% is
what partial clustering by altitude and region recovers.

The corollary is that a criterion firing on a *small* fraction can be worth far
more than one firing on a large fraction. At the 0.3% of cells that straddle
saturation, `0.997^32` ≈ 91%, so nine warps in ten take the cheap path and
divergence stops being the obstacle. Always convert a "fraction of points"
estimate into a per-warp probability before believing it.

**Splitting the quadrature into per-point kernels** — dead, but not for the
reasons usually given. Launch overhead is negligible (9 extra launches × 10 calls
× ~7 µs ≈ 0.63 ms) and so is the extra field traffic (~1.1 GB/call ≈ 0.9 ms
against DRAM at 4.7% of peak). It fails because each split kernel still contains
one full CloudMicrophysics evaluation at ~246 registers, so every one of them
runs at the same 8 warps/SM.

**Fusing the environment scratch fields back into one point body** — the five
scratch kernels total **219 µs against the hot kernel's 26.00 ms, 0.84%**, and
all seven materialization kernels (those five plus `ᶜλ⁰` and `ᶜmu_S⁰`) total
277 µs, 1.07%. Fusing them would raise the hot kernel's register pressure to
save that. `foreach_point` is what the `@.` broadcast already lowers to, so
writing it explicitly changes syntax, not codegen. Both figures are
`off.kernels.*.mean_ms` from `launch-bounds.toml`, so numerator and denominator
come from the same instrument; an earlier version of this entry divided a
scratch total by the *ncu* duration of L1013 (30.86 ms) and understated the
share.

**Unrolling the quadrature loop** — the comment in `sum_over_quadrature_points`
saying loops beat `ntuple` for register reuse is still correct. Sweeping a
synthetic evaluator across register regimes:

| working set | loops | unrolled | |
|---|---|---|---|
| 78 regs | 64 B local, 0.742 ms | 80 regs, 144 B, 0.745 ms | tied |
| 193 regs | 64 B, 2.364 ms | 202 regs, 144 B, 2.616 ms | **unrolled 10.7% slower** |
| 255 regs | 1736 B, 28.161 ms | 255 regs, 1808 B, 27.972 ms | tied, both saturated |

Unrolling is never faster and *increases* local memory by a constant 80 bytes —
the opposite of the usual advice that dynamic indexing forces local memory and
unrolling fixes it. The diagnosis is right; the remedy is backwards.

**Fast math** — not tested, deliberately. It is Nsight Compute's own top
recommendation for this kernel's dominant stall and would attack both the
transcendental latency chains and the non-fused FP32 instructions. It is rejected
on **reproducibility** grounds, not performance grounds. Recorded here so it is
not rediscovered from the profile and re-proposed.

Also excluded by measurement: there is **no Float64** in this kernel (0% of FP64
peak, no `dfma` instructions), so the common "hidden Float64 promotion" diagnosis
does not apply. And parameter structs are not the driver — the updraft kernel
passes the same `cmp, thp` and sits 9 registers *lower*.

## 5. Methodology lessons

**A mechanism that wins on the GPU can still lose the run.** The first full
pipeline run of launch bounds improved CUDA kernel time by 2.55% and every
per-kernel prediction held, yet estimated SYPD fell ~19%. The cause was host-side:
the decision cache was keyed on the compiled kernel, so it called `cufunction` on
every launch just to build the key, from two call sites. An AMIP step spends most
of its host time issuing many small kernels, so the cost landed on the
launch-dense ranges at identical call counts — `copyto_foreach!` +116% per call,
`ldiv!` +92%, `set_implicit_precomputed_quantities!` +88%. Keying on
`(typeof(f), typeof(args))` fixed it: per-launch host cost with the target
enabled went from 6.95 → 11.47 µs (+65%) to 7.05 → 7.21 µs (+2.3%).

**Use the interactive harness to iterate, never to conclude.**
`scripts/repl_perf_setup.jl` times GPU kernels at one call site and is blind to
per-launch host overhead accumulated across a step. It reported the buggy
treatment above as a clean win. It is otherwise trustworthy — it reproduces the
pipeline's per-kernel times to within 1–2% — which is exactly what makes it
dangerous to over-trust.

**Measure the counterfactual, not just the shipped configuration.** The spill
budget that gates launch bounds was originally a guessed constant, and it
silently rejected the most expensive kernel in the run. Only by loosening it
until the kernel was accepted did we learn the guard was right (+20.5% slower)
rather than arbitrary. A rejected configuration never appears in an ordinary
run, which is why `launch-bounds-study` measures one deliberately.

**Do the arithmetic before the experiment.** AMIPWarmup and the disk cache both
had ceilings that were computable in advance and smaller than the noise.

## 6. Where the remaining headroom is

The binding constraint is the number of CloudMicrophysics evaluations:
**9 quadrature points × 2 substeps = 18 per grid point per call**. That is a
modelling choice, not a compiler outcome, and it is the only lever large enough
to clear the benchmark's resolution.

Ordered by preference — ClimaCore first, then ClimaAtmos, then CloudMicrophysics:

1. **~~Collapse the quadrature where the SGS PDF is degenerate.~~ DEAD, measured
   2026-09-01.** The idea was that where `σ_q` and `σ_T` are both zero the nine
   points map to the same state and the weights sum to one, making the collapse
   exact and free. Measured across all 1,548,288 cells: **0.00%** qualify. `σ_q`
   is never identically zero — its minimum is `3.25e-16`. There is no free,
   exact collapse. See §6a.
2. **Adapt the order to why the quadrature exists.** It resolves the `max(0, ·)`
   kink at saturation. Where the PDF sits wholly on one side —
   `|mu_S| > k·σ_S`, both already computed — the integrand is smooth and low
   order suffices. Error controlled by `k`. **Measured and viable: 98.74% of
   points and 83.22% of warps are more than 10σ clear of the kink. See §6a,
   including why this is a science question before it is a performance one.**
3. **Collapse the 2-D rule to 1-D in the saturation excess**, evaluating the
   smooth temperature-dependent rate coefficients at the mean. Error
   `O(σ_T²)`, 9 → 3 evaluations.
4. **Integrate the SGS average analytically instead of sampling it.** The closure
   already does this for the mean condensate — `λ_lagrange` is calibrated so that
   `E[max(0, λ + α·S′)] = q_c` in closed form. The tendencies need `E[q_c^p]` for
   power-law rates, which also has closed forms for a truncated Gaussian. This
   removes the quadrature entirely, but is a reformulation of the
   ClimaAtmos/CloudMicrophysics interface rather than a local edit.
5. **Reduce one CloudMicrophysics evaluation below 168 registers.** Least
   preferred layer, highest leverage: the launch-bounds machinery is *written*
   and measured, so this would immediately convert the hot kernel from 8 to 12
   warps/SM. Expect superadditivity, as with the earlier CM + register-cap pair.
   To find where the 246 registers go, bisect with temporary returns partway
   through the evaluation and watch `CUDA.registers` spike.

   **The target is ~133, not ~100.** Subtracting the 68-register framework from
   the 168 threshold gives 100, which is probably unreachable. But the
   launch-bounds decision table shows kernels *entering* at ≤201 registers reach
   12 warps/SM for **zero** bytes of spill, and the hot kernel's true demand is
   ~231 (163 CM + 68 framework). So the ask is ~30 registers, not ~63. That is
   the difference between "probably impossible" and "plausible", and it is the
   number to aim at.

   **Three CloudMicrophysics techniques exist, and they are orthogonal.** All
   three trade the idle resource (ALU at 26%, DRAM at 4.7%) for the exhausted
   one (registers), which is why they read as pessimizations anywhere that is
   not register-bound — worth stating in any PR description, since upstream's
   instinct is the opposite.

   - **Rematerialize** — `pb/1m-spill`. Removes the hoisted
     `sd = CM1.size_distr_parameters(...)` (λ⁻¹, n₀, v₀ for rain/snow/ice,
     pow/exp-heavy) that main computes once and threads through ~10 of the 18
     process calls, so each process recomputes what it needs instead of holding
     it live across the whole evaluation. Measured **+1.107%**
     (`2026-08-21-cm-only`).
   - **Fuse** — `pb/fuse-source`, commit `e619a1eb`. `_linearize` is already an
     accumulator over 13 values (M11…M44, e1, e2, e4) consuming each `src.S_*`
     in turn, so the 18 source terms exist as a batch only because they are
     produced in one function and consumed in the next. `_fused_linearize`
     folds each source term into its M/e accumulators at its point of
     computation, cutting peak liveness from ~18 src + 13 accumulators to ~13
     plus the term in hand. It is bit-for-bit: each accumulator receives its
     contributions in `_linearize`'s statement order, so only *when* each term
     is computed moves — no sum is reassociated. Measured **+3.299%**
     (`2026-07-06-run-with-cm-b8de82423fe-fuse-source`), the largest recorded
     CM-side number, though on a 2026-07 baseline (0.26113) not directly
     comparable to the August runs. That branch also hoists `p_vs_liq` /
     `p_vs_ice` (the two saturation vapor pressures, a log + exp each) into the
     processes that need them — the *opposite* direction from `pb/1m-spill`,
     and consistent with it: hoist what is expensive to compute and cheap to
     hold, rematerialize what is cheap to compute and expensive to hold.

   **Combined on `pb/1m-spill-fuse` (2026-08-31), not yet measured.**
   `_fused_linearize` ported onto the current `pb/1m-spill` tip, transcribed
   statement-for-statement from *today's* `_linearize` rather than merged from
   the July branch (which predates the `options`→`processes` rename and the `sd`
   hoist). The `p_vs_liq`/`p_vs_ice` hoist is deliberately NOT included, so the
   fusion can be attributed on its own; it touches three more files and is the
   obvious follow-up. Bit-for-bit equivalence verified on CPU over 520,000
   accumulator comparisons in Float32 and Float64, including the negative-input
   clamping paths, and pinned by a new test in `test/bulk_tendencies_tests.jl`
   so the fused and unfused implementations cannot silently diverge —
   `_microphysics_source_terms` + `_linearize` remain as the tested reference
   for the `Instantaneous` paths.

   **Check the current tip before building on it.** `pb/1m-spill` at
   `cbfc59545` ("Merge main into pb/1m-spill; re-apply 1M guards on the new rate
   API") measures 112 registers with **480 bytes** of local memory, against
   `68007fb10`'s 163 registers with **32 bytes**. Registers fell 31% while local
   memory rose 15×: the branch is now spilling its way down rather than fitting,
   which is the opposite of what it exists to do. Re-measure both revisions
   before reopening — `cm-registers` runs in minutes — and record `local_bytes`
   alongside `registers` every time, because a falling register count with
   rising local memory reads as progress if you only look at one column.

   **It is written, not merged, and not currently in either arm.** The mechanism
   lives on `ClimaCore.jl-mod` branch `pb/launch-bounds` (ClimaCore PR 2601) as
   two commits — `5428c6e2c` "Target occupancy with `__launch_bounds__` instead
   of a register cap" and `45bf1792c` "Key the launch-bounds cache on types, not
   on the compiled kernel" — on top of `505982708`. As of 2026-08-31 that branch
   is not an ancestor of `ClimaCore.jl-mod` HEAD (`74b837bfc`, on `main`), and
   both ClimaCore arms sit at the same commit, so the working tree carries no
   ClimaCore treatment at all. Re-pointing the mod arm at `pb/launch-bounds` is a
   prerequisite for this lever, not something that comes for free with it.

Splitting by *process* is the one live decomposition, but only in the variant
that materialises per-quadrature-point state into scratch first (~223 MB, cheap
against idle DRAM). Splitting with the quadrature loop left inside each process
kernel multiplies the ~1.5 ms fixed cost by the number of processes and is not
viable.

## 6a. How much work the SGS quadrature is actually doing (2026-09-01)

Measured by `scripts/measure-sgs-degeneracy.jl` on a settled AMIP state (three
steps in), reading `ᶜT′T′`, `ᶜq′q′`, `ᶜT⁰` and `ᶜq_tot_nonneg⁰` from
`p.precomputed` and the prognostic `Y.c.ρ`. Log:
`.calkit/scheduler/logs/sgs-degeneracy.out`.

### The distribution

| | |
|---|---|
| `σ_T` | median 0.0112 K, max 0.465 K |
| `σ_q` | median 2.25e-7, max 1.07e-3 |
| `σ_S` | median 3.04e-6, max 0.0336 |
| `\|mu_S\|` | median 1.64e-3 |
| **`\|mu_S\| / σ_S`** | **p1 7.8, p25 266, median 449, p75 899, p95 2761** |

`σ_S² = σ_q² + (∂q_sat/∂T)²σ_T² − 2·corr·σ_q·σ_T·(∂q_sat/∂T)`, with
`∂q_sat/∂T` from Clausius-Clapeyron. Cross-checked independently: the median
`|mu_S|`/median `σ_S` ≈ 540 against the measured median ratio of 449, same
order, so the formula is behaving.

**At the median cell the saturation excess sits 449 standard deviations from
the kink the quadrature exists to resolve.** Even the worst 1% of cells are
~8σ clear of it.

### And it survives the warp test, which is the part that usually kills these

| criterion | point frac | warp frac |
|---|---|---|
| exact: `σ_T == 0 && σ_q == 0` | 0.00% | 0.00% |
| `\|mu_S\| > 2·σ_S` | 99.76% | 94.30% |
| `\|mu_S\| > 3·σ_S` | 99.63% | 92.24% |
| `\|mu_S\| > 5·σ_S` | 99.36% | 89.12% |
| `\|mu_S\| > 10·σ_S` | 98.74% | **83.22%** |

Scattered, `0.9874³²` would give 66.7% of warps; measured 83.22%, so vertical
clustering adds ~1.25×. But the reason this works is not clustering — it is that
the point fraction is so close to one that warps qualify almost regardless.
Contrast the clear-air early-out: 77.7% of points returned 21.5%.

### What the quadrature costs, so the trade is a decision and not an abstraction

Priced by a config-only bounding run — `quadrature_order: 1` in the mod arm,
collapsing 3×3 points to 1, compared against the *mod* arm rather than the
baseline so the fusion is held fixed and only the quadrature moves:

| configuration | SYPD |
|---|---|
| baseline (upstream CloudMicrophysics) | 0.27079 |
| + fusion (`exp/2026-09-01-cm-fuse`) | 0.27596 |
| + quadrature collapsed to one point | **0.29973** |

**The nine-point quadrature costs 7.93% SYPD.** Removing it entirely would put
the flagship run 9.66% above today's baseline.

This is an **upper bound and not a candidate**: `quadrature_order: 1` changes
results, and the config was reverted immediately after. It exists so the science
question has a number attached — *is the quadrature worth 8% of the flagship
run's throughput, given the PDF sits 449σ from the feature it resolves?*

An adaptive collapse (item 2) captures only the warps that qualify, so expect
roughly `0.83 × 7.93% ≈ 6.6%` before subtracting the branch cost and the
divergence in the ~17% of mixed warps. The full 7.93% is available only if the
quadrature can go entirely.

### Three caveats, all load-bearing

**It is an approximation, not an exact collapse.** Being clear of the kink makes
the *condensate reconstruction* smooth, but the rest of the integrand — the
evaporation and sublimation rates, which depend on `T̂` and `q̂` — still varies
across the PDF. The error is `O(σ²·f″)`; the relative PDF width is tiny
(`σ_q/q_tot` ≈ 2e-5 at the median), so it should be small, but it must be
measured against the nine-point answer rather than assumed. Item 1 was exact;
this one has a knob.

**The 98.74% is dominated by cells where nothing happens anyway.** The median
cell is dry upper atmosphere. `σ_q` reaches 1.07e-3, comparable to `q_tot`, so
the cells that matter physically are exactly the minority where σ is large and
the quadrature *is* doing work. The saving is real but concentrated in cheap
cells, which means it **overlaps heavily with the clear-air early-out**
(ClimaAtmos `3b38b0e50`, +1.79%) — and that early-out is *not* in the currently
checked-out ClimaAtmos (v0.42.8). The two are not additive and must not be
estimated as though they were.

**It is a science question before it is a performance one.** If the SGS PDF
genuinely sits 449σ from saturation at the median cell, the nine-point
quadrature is buying almost nothing physically in this configuration — while
costing 9× on the largest kernel in the run. Either the covariance closure is
producing variances that are too small, or the quadrature is unnecessary here.
Those have *different* remedies: if the variances are wrong the fix is in the
closure and the quadrature stays; if they are right, the quadrature order should
be reconsidered outright rather than worked around. Put this to whoever owns
`_compute_sgs_moments` before writing kernel code.

### Two method errors made getting here, both worth not repeating

**The first version of this measurement used the wrong yardstick.** It compared
`σ_q` and `σ_T` against absolute constants and reported 99.77% of warps under
`σ < 1e-3`, which looks like a headline result and means nothing — "σ is small"
is dimensionally arbitrary. A tiny σ still needs the quadrature if the mean sits
on the kink; a large one does not if the cell is far from it. Only the ratio to
`σ_S` is meaningful.

**And `calkit scheduler batch` silently returned the previous result.** It keys
on job name, and with no declared dependencies it reported *"Job
'sgs-degeneracy' already left the queue; using its result"* — so the corrected
script's output was the old script's numbers. This was caught only because the
output *format* had changed; a threshold-only edit would have been reported as
fresh. **Always pass `--dep` for the script itself.** This is the same failure
as the `CloudMicrophysics.jl-mod` input bug in §3a: an under-declared dependency
reusing a cached result.

## 7. Reference

**A100 register occupancy steps** (65536 registers/SM, 4 schedulers, 256-register
allocation granularity), for `round_up(regs × 32, 256)` per warp:

| registers ≤ | warps/SM | occupancy |
|---|---|---|
| 255 | 8 | 12.5% |
| 168 | 12 | 18.75% |
| 128 | 16 | 25% |
| 96 | 21 | 32.8% |

The exact threshold for 12 warps is 168, not 170: it needs
`round_up(regs × 32, 256) ≤ 16384/3`.

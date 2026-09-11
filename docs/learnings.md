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

### 2a-i. SUPERSEDED 2026-09-03: the barrier no longer holds, and the cost is the broadcast

The subsection below was measured pre-fusion and its central claim — that the
`@noinline` barrier holds, so nine evaluations cost 9 registers more than one —
**is no longer true.** Re-measured with the rebased launch-bounds branch
(`pb/launch-bounds-v2`), on a harness that reproduces *both* real microphysics
kernels exactly:

| | pre-fusion | now |
|---|---|---|
| single-call (real L970, and layer D) | 246 | **153** → 12 warps/SM |
| nine-point (real L1013, and layer E) | 255 | **255** → 8 warps/SM |
| gap attributable to the quadrature | **9** | **102** |

The fusion cut the single-call path by 93 registers and the quadrature path by
none, which is why L970 crossed to 12 warps/SM and L1013 did not.

**And the 102 registers are the broadcast context, not the physics:**

| | registers |
|---|---|
| quadrature body, **scalar** kernel | **147** |
| quadrature form, **broadcast** (layer E) | **255** |
| direct form, **broadcast** (layer D) | 153 |
| CloudMicrophysics `LinearizedAverage` standalone | 116 |

Identical physics costs 147 standalone and 255 inside a broadcast. The broadcast
adds +108 for the quadrature form against +37 for the direct form — and layer B
rules out field count as the cause (7 fields = 32 registers, *fewer* than 3
fields at 40), so the three extra quadrature inputs are not it.

**Leading hypothesis: the quadrature loop is being unrolled in the broadcast
context.** `sum_over_quadrature_points` deliberately uses `for` loops rather than
`ntuple`, and its own comment gives the reason — "each loop iteration releases
registers from the previous one". ClimaCore compiles broadcasts with
`always_inline = true`. If LLVM unrolls the 3×3 loops there but not in the
scalar context, nine copies of the body go live simultaneously, which is the
right order of magnitude for +108.

**Not yet established**, and both matter before acting:

- Whether layer E's 255 is a demand or the hardware *cap*. The real kernel has
  zero spill at 255 (ncu, §3a), so its demand is genuinely ≤255; layer E's
  figure could be clipped.
- Whether the loop is actually unrolled. This is inferred from a register count.
  `@device_code_llvm` on the scalar and broadcast contexts would settle it, and
  is cheap.

If it holds, the fix is local — an unroll barrier inside the quadrature loop —
and it is the whole remaining obstacle to the occupancy win ncu ranks at 69%
estimated speedup. It also moves the target off CloudMicrophysics: the physics
is already at 147.

### The 255 registers are one CloudMicrophysics evaluation (PRE-FUSION; see 2a-i)

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

### 3b. Re-measured on the updated stack (2026-09-02), and it is additive

The dependency update (§3c) moved both arms, so the numbers above describe the
old stack. Re-run with everything at upstream main:

| | baseline | mod | fusion |
|---|---|---|---|
| AMIP SYPD | 0.28312 | 0.28874 | **+1.95%** |
| GPU kernel time | 1520.2 ms | 1424.8 ms | **−6.28%** |
| hot kernel L1013 | 279.2 ms | 199.3 ms | **−28.6%** |
| launches | 31,180 | 31,180 | 0.00% |

Everything else in the top six moved by ≤0.7%, both arms report 357 distinct
kernels with zero name mismatches, and launch counts are identical to the unit —
so the arm comparison is sound on this stack even though it is *not* sound
across the update (see §3c).

**A prediction that was wrong, and why.** Before the run I expected the fusion to
come in *below* +1.87%, reasoning that ClimaCore PR 2606 had removed ~170 ms of
kernel time from the denominator. It came in slightly higher. The error was
treating total kernel time as the denominator for SYPD: 2606's gains are largely
device allocations and Cartesian index arithmetic, which do not sit on the
microphysics kernel's critical path. The fusion's saving held at ~80 ms in
absolute terms — the baseline arm's L1013 is 279.2 ms on *both* stacks — so its
share rose as the total fell. **The two changes are independent and additive**,
which is the more useful conclusion than either number.

**Two samples for the conversion ratio now**, rather than one: −5.37% → +1.87%
(35%) and −6.28% → +1.95% (31%). Roughly a third of GPU kernel time reaching
SYPD is a defensible estimator for a work-removing change.

### The mechanism, finally measured: the fusion removes the spill entirely

Both ncu stages refreshed together on the current stack (2026-09-03), the first
ncu data describing the post-fusion, post-2606 kernel:

| L1013 | baseline | mod |
|---|---|---|
| registers per thread | 255 | 255 |
| **local memory spill overhead** | **32.92%** | **0%** |
| stack frame | 2,272 B | 2,048 B |
| achieved occupancy | 12.39% | 12.40% |

Both arms sit at the 255-register cap, at the same occupancy. **nsys therefore
shows nothing changed** — it reports registers per thread and not spilling — so
the entire mechanism behind the −28.6% is invisible in `top-kernels.csv` and
appears only here. The baseline spills a third of its memory traffic; the fused
version spills none.

This retires two open items:

- **"Does the fused L1013 still spill?"** No. The 2208-byte stack frame and 21%
  spill overhead recorded in §2 were properties of the *pre-fusion* kernel.
- **"Is the hot kernel memory-bound?"** It never was on *global* memory — DRAM
  at 4.7% — but it did carry heavy *local* traffic, and ncu's own memory
  recommendations were all about local loads and stores (0.9 of 32 bytes
  utilized per sector). That traffic is gone, and it was removed by cutting
  register pressure rather than by any memory optimisation. Spill is downstream
  of registers; there is no separate memory lever here.

Note the shape of the win: occupancy is *unchanged* at 12.4%. The kernel got
28.6% faster without gaining a single warp. Occupancy was never the only thing
the register pressure was costing.

## 3c. The upstream update is worth +4.36% on its own (2026-09-02)

Prompted by the nightly showing ~4%. Updated ClimaCore (24 commits), ClimaAtmos
(2), ClimaCoupler (63) in both arms, and merged main into CloudMicrophysics
`pb/1m-spill-fuse`.

| | SYPD |
|---|---|
| old baseline | 0.27079 |
| **new baseline** | **0.28312** |
| new mod (with fusion) | 0.28874 |

**This is the baseline moving, not a result of ours.** Upstream `main` is the
floor we build on and it changes continuously; a +4.36% baseline improvement
belongs to whoever wrote PR 2606. Our contribution is only ever the mod-vs-
baseline delta at a given revision, which is what `speedup_pct` measures and
what `experiments.csv` records. Do not add the two together and report a
cumulative figure — that silently claims upstream's work.

Absolute SYPD is still the right KPI for "how fast is the flagship model", and
`baseline_sypd` in `experiments.csv` is the column that shows when the floor
moved (0.26969 → 0.27079 → 0.28312). Keep the two questions separate: *what did
we contribute* is the delta; *how fast is it now* is the absolute.

The mechanism is ClimaCore **PR 2606**, and it is visible directly in the
profile: the CUDA.jl Cartesian broadcast path collapsed from ~168 ms in the
top ten alone to **one kernel kind, 30 launches, 0.4 ms**. Those broadcasts now
take ClimaCore's flattened linear-indexing path. Launch count did not change, so
nothing was merged — the same kernels got faster.

**PR 2598 no longer crashes AMIP.** `docs/misc/climacore-pr2598-report.md`
recorded a device-side `error_mismatched_spaces` on the first tendency
evaluation; on merged main all runs complete with zero errors. That report is
resolved.

### Per-kernel rows are not comparable across this update

Work moved *between* kernels: broadcasts that were separate unnamed Cartesian
kernels are now attributed to the enclosing NVTX-named kernel. So several
kernels appear to have roughly doubled —
`prep_hyperdiffusion_tendency` 19.5 → 38.8 ms, `apply_hyperdiffusion_tendency`
19.3 → 35.5, `horizontal_tracer_advection_tendency` 29.5 → 47.1 — and have not.
Launch counts identical to the unit rule out kernels being merged. Compare
**totals** across the update, and per-kernel rows only *within* a stack.

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

**Forcing launch bounds on the hot kernel** — *(SUPERSEDED 2026-09-04: this was
measured on a kernel spilling 21% at its natural register count, with no slack
to reschedule away. The fusion created the slack, and the same mechanism is now
accepted and worth −19.4% on the kernel. See §4a. The measurement below stands
as recorded; the conclusion drawn from it does not generalise.)* ptxas *can*
reach 168 registers and 12 warps/SM, but it costs +344 bytes of spill per thread
and the kernel runs **31.33 ms against 26.00 ms, 20.5% slower**
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

## 4a. Occupancy was not a dead end — it was waiting for headroom (2026-09-04)

§4 concluded that "occupancy is a dead end for the hot kernel itself." That was
correct on its evidence and is now false. The mechanism is unchanged; the kernel
underneath it changed.

| | pre-fusion | post-fusion |
|---|---|---|
| registers asked of ptxas | 168 | 168 |
| **spill growth to get there** | **+344 B** | **+48 B** |
| guard verdict (256 B budget) | rejected | **accepted** |
| measured effect | 20.5% *slower* | **19.4% faster** |

Measured end to end, with the mechanism on ClimaCore `pb/launch-bounds-v2`
(PR 2601 rebased onto the current base) and the fusion on CloudMicrophysics
`pb/1m-spill-fuse`:

| | SYPD | L1013 | registers | total kernel time |
|---|---|---|---|---|
| baseline | 0.28312 | 279.2 ms | 255 | 1520.2 ms |
| + fusion | 0.28874 (+1.95%) | 199.3 ms (−28.6%) | 255 | 1424.8 ms |
| **+ both** | **0.29056 (+2.56%)** | **160.7 ms (−42.4%)** | **168** | **1382.4 ms (−9.06%)** |

Launch bounds contributes **+0.63% SYPD** and **−19.4%** on the hot kernel on
top of the fusion. Every other top-five kernel moved by ≤0.14%, and nsys
confirms the mechanism fired in the real run: L1013 compiles at 168 registers in
the mod arm against 255 in baseline. **8 → 12 warps/SM on the largest kernel in
the run**, bought for 48 bytes of spill.

### The pattern this is the third instance of

| pair | alone | alone | together |
|---|---|---|---|
| CM register work + ClimaCore register cap | +1.107% | +0.433% | +2.372% |
| CM fusion + ClimaCore launch bounds | +1.95% | +0.433%¹ | **+2.56%** |

¹ the launch-bounds mechanism's own standalone measurement, from
`exp/2026-08-24-launch-bounds`, where it had no substrate and returned a SYPD
null.

Occupancy machinery converts register headroom; it cannot create it. Work
removal creates headroom but does not by itself convert it into warps. **Neither
is worth much alone and the pair is worth more than the sum** — three times now.
Expect it, and do not judge an occupancy-side change by its standalone number.

### And it nearly got thrown away

On 2026-09-01 the recommendation here was to close PR 2601: its payoff was
conditional on CloudMicrophysics work that did not exist, the target looked like
~100 registers, and it was worth −2.11% kernel time with a SYPD null on its own.
That reasoning was sound on the evidence available. The fusion invalidated the
evidence two days later by moving the target to a reachable place. The branch
survived only because the tag `exp/2026-08-24-launch-bounds` and the branch
itself were preserved rather than deleted — which is the argument for keeping
dead-end branches around rather than for the analysis that declared it dead.

**ncu's 69%-estimated-speedup figure for occupancy remains a poor estimator.**
The prediction made before this run — that it was optimistic, because the kernel
is latency-bound on execution dependencies rather than occupancy-starved, and
that the real gain would be "real but smaller" — held. −19.4% is real but
smaller.

## 4b. What the refreshed ncu reports actually show (2026-09-05)

The ncu pair was refreshed after the 16-warp step landed, because the table on
record still described the pre-launch-bounds mod arm. Only `mod-ncu` re-ran:
DVC found `baseline-ncu` up to date, correctly — the baseline arm has not
changed since its 2026-09-03 export, which was already post-dep-update. A
skipped baseline is not the mismatch AGENTS.md warns about; that one was two
arms from *different* project states. Here the baseline state is the same one,
and the skip is DVC saying so.

| | baseline | mod (128 reg / 16 warps) |
|---|---|---|
| duration | 31.70 ms | **11.38 ms** |
| registers | 255 | **128** |
| achieved occupancy | 12.39% | **24.60%** |
| SM throughput | 31.01% | **43.88%** |
| IPC | 1.27 | **1.79** |
| spill requests | 37,538,807 | **15,654,128** |
| spill overhead | 32.92% | **12.81%** |
| block size | 256 | **512** |

### The spill went down, not up

This was recorded backwards while the change was being made: the 16-warp target
was described as "reintroducing 208 bytes of spill," and §4a's budget table
frames spill growth as the cost paid for occupancy. On the real kernel, forcing
128 registers **halves** spill traffic against baseline — 37.5M requests to
15.7M, 32.92% to 12.81% overhead.

Both statements are true and they are about different baselines. `spill_growth`
in the guard is measured against the *unbounded* compile of the same kernel,
which is the right quantity for the guard's decision. It is not the quantity
that predicts the measured result, because the unbounded compile of the fused
kernel is not what the arm is being compared against. **When reporting a spill
number, say which compile it is relative to** — the guard's budget and the
profiler's overhead are not the same axis, and reading one as the other is what
produced the wrong description here.

### Part of the conversion anomaly is block size

The unexplained item from §3b — the mod arm gained SYPD out of proportion to its
summed kernel time — has a partial mechanism now. Block size went 256 → 512, so
each block covers twice the work and the grid-stride loop runs half as many
iterations. IPC rose 1.27 → 1.79 and SM throughput 31% → 44%: the kernel is not
merely holding more warps resident, it is *issuing* substantially better.
Summed kernel duration understates this, because part of the gain lands in how
work is scheduled rather than in any single kernel's wall time. This narrows the
anomaly; it does not close it, and the mechanism is inferred from the counters
rather than demonstrated.

### The next occupancy step does not exist

ncu still ranks occupancy first, at 55% estimated (down from 69%), on the same
grounds: 4.00 of 16 warps per scheduler, register-limited. The obvious read is
to take one more step. The register sweep says there is no step to take —
ptxas goes from 128 registers (16 warps) directly to 80 (24 warps), with nothing
between, and the hot kernel pays **+392 B** of spill growth to get there against
a 256 B budget. The nearest measured point on that axis, +344 B pre-fusion at 12
warps, ran **20.5% slower**.

So the register lever is not exhausted because occupancy saturated. It is
exhausted because the granularity of the next step overshoots. That is a
different claim and it fails differently: it would come back if the kernel body
shrank enough to reach 80 registers without the spill.

### Measured, not extrapolated (2026-09-06)

Raising `LAUNCH_BOUNDS_SPILL_BUDGET` to 512 lets the 24-warp target through, so
the step could be priced rather than argued about. Under nsys, over 10 launches
each:

| | L1013 total | per launch |
|---|---|---|
| 16 warps / 128 reg | 150.3 ms | 15.03 ms |
| 24 warps / 80 reg | **177.9 ms** | 17.79 ms |

**18.4% slower**, against the 20.5% the +344 B case lost pre-fusion. Reverted.
The 256 B budget now has a rejection on either side of its single acceptance,
and the two rejections cost almost the same amount — the budget is not a
conservative guess that happens to work, it is tracking something real.

Re-running the 16-warp build afterwards to restore `results/nsys/mod.sqlite`
produced an unplanned replicate: **15.07 ms** against the original 15.03 ms,
**0.31% apart**.

**That 0.31% was over-read, and this note originally over-read it.** It is one
pair. A second pair, taken on the evaluator build (§4d), came back **2.08%**
apart — 137.21 vs 140.10 ms. A repeat spread estimated from a single pair is not
a repeat spread; it is one draw from a distribution whose width is unknown. Treat
the instrument as good to a couple of percent on this kernel until there are
enough repeats to say otherwise. The 24-warp effect is large enough (+18.4%) to
survive that revision comfortably; it is still a sharper instrument than
full-AMIP SYPD, because it counts one kernel rather than a whole run's wall
time, but not by the order of magnitude first claimed. Prefer it for
kernel-local questions; it cannot answer whole-run ones, since it is blind to
everything that is not that kernel.

**A warning about the wrong number, because it nearly got reported as a win.**
The same run's `sypd ≈` lines read 0.4104 against the 16-warp run's 0.3784 —
apparently +8.5%. They are the progress logger's *running average* at coupler
steps 8 and 12 of a 10-step profiled run, so they carry JIT and profiler warmup
and are not the pipeline's `estimated_sypd`. When a proxy and a direct
measurement of the changed kernel disagree by 27 points in opposite directions,
the proxy is what is broken. Grepping a log for `sypd` will find these lines
first; they are not the metric.

## 4c. 80 registers is unreachable, and why the ladder says so (2026-09-06)

§4b closed on the idea that the 24-warp step "would come back if the kernel body
shrank enough to reach 80 registers without the spill", and named
CloudMicrophysics work removal as the way there. The register ladder already
measured says that is impossible, and it is worth stating plainly so nobody
spends a week on it.

| layer | registers | what it contains |
|---|---|---|
| B | 32 | 7 fields in, one scalar out |
| **C** | **64** | 7 fields in, NamedTuple out — **framework only, zero physics** |
| D | 153 | + the full non-quadrature microphysics |
| F | 193 | + quadrature machinery, 1 point |
| G | 215 | 4 points |
| E | 255 | 9 points |

**Layer C is the floor: 64 registers before a single line of physics runs.** An
80-register budget leaves 16 registers for all of microphysics, and the "source
terms only" layer alone is 44. Deleting the entire quadrature would land at 153.
There is no amount of CloudMicrophysics work removal that reaches 80, because
the thing standing in the way is not microphysics.

Note also B → C: **+32 registers for returning a 4-field NamedTuple instead of a
scalar.** That is half the framework floor, spent on output layout rather than
on any computation, and it is a ClimaCore question rather than a
CloudMicrophysics one. It is the only visible lever on the floor itself.

### The leak that is worth chasing instead

The same ladder shows something that should not be there. `Microphysics1MEvaluator`
carries a `@noinline` barrier precisely so that `sum_over_quadrature_points`
reuses registers across points, and the loop is hand-written (rather than
`ntuple`) for the same reason. If the barrier held, F, G and E would be equal.

    1 point   193
    4 points  215      +22
    9 points  255      +62      ~7.75 registers per extra point

**The barrier leaks, and the leak scales with point count.** That is the gap
between 255 and ~193, and closing it is worth real time even though it does not
reach 80: at the shipped 128-register bound, spill growth is 208 B at 255
unbounded but only 88 B at 193 (layer F), and spill is currently 12.81% of the
kernel's memory traffic.

So the target to chase is **not a lower register bound — it is a lower unbounded
register count, which buys less spill at the bound we already have.** Those are
different objectives and the ladder distinguishes them; conflating them is what
produced the 80-register goal in the first place.

### Attempt 1: it is not transform hoisting (rejected)

The first hypothesis for the leak was that `GaussianPhysicalPointTransform` is
small enough to inline, so LLVM computes all nine `(T_hat, q_hat)` pairs up
front and keeps eighteen values live across the barrier — which would scale with
point count exactly as observed. Marking the transform `@noinline` so each point
is transformed at its point of use:

| layer | before | after |
|---|---|---|
| D (no quadrature) | 153 | 153 |
| F (1 point) | 193 | 195 |
| G (4 points) | 215 | 216 |
| E (9 points) | **255** | **255** |

**No effect on registers.** Spill growth at the 128-register bound moved 208 →
192 B, which is not nothing but is not the mechanism either. Reverted — it added
an ABI cost for no register gain. The transform is not what is being kept alive.

### Attempt 2: it is not loop unrolling either (rejected)

Second hypothesis: with `N` a compile-time constant the loops fully unroll into
N² call sites, so the `@noinline` on the evaluator never produces a single
reusable call frame. Two tries:

- **`@noinline _opaque_trip_count(n) = n`** — changed *nothing*, to the byte.
  The reason is worth keeping: **`@noinline` blocks inlining, not constant
  propagation.** Julia's constprop runs independently and folded the identity
  straight back to `3`, so the loop still unrolled. A `@noinline` identity is
  not an optimization barrier.
- **`Base.compilerbarrier(:const, N)`** — the primitive that *does* block
  constprop. Registers: D unchanged, F unchanged, E unchanged at 255, G
  215 → 185.

That last number looks like a win and is not one, which is the real lesson here.

### The instrument was hiding the answer

`spill_growth_bytes` is a *difference* between the bounded and unbounded
compiles, and it was the only local-memory number recorded. A difference cannot
tell "demand fell" from "both compiles moved together", and a kernel pinned at
the 255-register hard cap reports 255 whatever its true demand — so neither
recorded quantity could answer whether a change did anything. Added absolute
`unbounded_local_bytes` / `bounded_local_bytes` to the launch-bounds decision
record. With those visible, barrier on vs off:

| layer | registers | unbounded local | bounded local |
|---|---|---|---|
| D | 153 → 153 | 1096 → 1096 | 1160 → 1160 |
| F (1 pt) | 193 → 193 | 592 → 592 | 680 → 680 |
| G (4 pt) | 215 → **185** | 640 → 640 | 816 → 816 |
| E (9 pt) | 255 → 255 | 1864 → 1864 | 2072 → 2072 |

**Every footprint is identical.** G's 30-register "improvement" is the compiler
choosing a different register/spill split at exactly the same total cost. The
barrier does nothing; it only moved where the compiler wrote the number down.
Reverted.

### What the absolute numbers say instead

| layer | unbounded local memory |
|---|---|
| F: 1 point | 592 B |
| G: 4 points | 640 B |
| E: 9 points | **1864 B** |

**The 9-point kernel spills 1864 bytes before launch bounds are applied at all.**
The 208 B of growth that the spill budget adjudicates — the number §4a and §4b
are built around — is an 11% increment on a kernel already deep in local memory.

This reframes the register story. Going 1 → 9 points does not mainly cost
registers (they are capped, so they cannot show it); it costs **+1272 bytes of
spill**. Reasoning about this kernel in registers was measuring the cap rather
than the demand, and the three failed attempts above are what that error looks
like from the inside: each targeted register pressure, and register pressure was
never the free variable.

## 4d. The evaluator payload, and what it does not buy (2026-09-06)

§4c ended by naming the evaluator's live state as the lead worth chasing. It was
the right lead and it produced the largest kernel-level win in the project, but
it does not do the thing it looked like it might.

`Microphysics1MEvaluator` stored `mp` and `tps` as fields, making it 472 B of
which **432 B was those two parameter structs** — identical for every cell and
every quadrature point. A struct built per cell is an alloca the ABI writes to
local memory before each `@noinline` call; the same values passed as arguments
stay in `.param` space and are read field-by-field. Threading them through
`sum_over_quadrature_points` as a defaulted `extra` tuple:

| | before | after |
|---|---|---|
| `sizeof(Microphysics1MEvaluator)` | 472 B | **40 B** |
| unbounded registers | 255 *(at the cap)* | **184** |
| unbounded local memory | 1864 B | **1432 B** |
| bounded local memory (shipped) | 2072 B | **1648 B** |
| L1013, 10 launches | 150.3 / 150.7 ms | **137.2 / 140.1 ms (−7.9%)** |

The −432 B is *exactly* `sizeof(mp) + sizeof(tps)`, which is what makes this
mechanism measured rather than inferred. Verified bit-for-bit: 400 randomized
states × 4 tendencies, identical bit patterns.

### The kernel win does not reach SYPD, and never could

Measured end to end over two runs: **3.78% and 4.52%, mean 4.15%** — against
4.25/4.36/4.10%, mean **4.24%**, for fusion+bounds@16 *without* this change.
The two ranges overlap and the difference is not resolvable.

Note the second sample also widens the SYPD noise picture: these two runs are
0.78% apart, against the 0.27% spread of the three-run set above and the 0.43%
"floor" quoted throughout this document. Like the kernel-level spread in §4b,
that floor is an estimate from few repeats and should be treated as a lower
bound on the true variability, not a constant.

The arithmetic was available before the run and should have been done first:
L1013 is now **10.1% of GPU time**, so −8.8% on it is ~0.95% of kernel time and
**~0.3% SYPD** at the ~⅓ pass-through this project keeps measuring. That is
below the noise floor by construction. **A kernel-level win on a kernel that is
10% of the budget cannot be validated by a whole-model measurement** — check the
share before spending 45 minutes of cluster time on the confirmation.

### It does not remove the need for the ClimaCore work

The tempting inference from 255 → 184 was that a little more slimming reaches
168 and 12 warps/SM unaided, making the launch-bounds branch unnecessary. That
is the wrong target. **Launch bounds already delivers 16 warps at 128
registers**; 168 would buy 12, which is a downgrade. The threshold that would
make ClimaCore redundant is ≤128 unbounded — 56 registers below where we are,
not 16 — and with a 64-register framework floor that leaves 64 for physics and
quadrature together, which currently cost 120.

**Occupancy at 16 warps is only reachable through launch bounds.** The
superadditive pattern holds a fourth time and the ClimaCore change is
load-bearing.

Nor is there an easy follow-up in narrowing the parameter struct:

    Microphysics1MParams = 328 B
      precip 132   terminal_velocity 76   process_params 56
      cloud   52   air_properties    12   processes        0 (singleton)

No large unused chunk — the physics touches most of it, so narrowing means an
invasive refactor for a fraction of 432 B.

## 4e. The adaptive quadrature collapse fails, and the criterion is why (2026-09-09)

§6a measured that the subgrid PDF sits a median 449σ from the saturation kink,
and that at 10σ **83.2% of 32-lane warps** are entirely clear of it — a far
better warp fraction than the clear-air early-out's 21.5%, which is why that one
returned only +1.79%. That made an adaptive collapse look promising: branch on
`|mu_S| > k·σ_S`, evaluate one point instead of nine where the PDF cannot reach
the kink.

The mechanism works. The accuracy does not.

| tendency (10σ) | rms / field RMS | max / field RMS |
|---|---|---|
| `dq_lcl_dt` | 1.8% | 0.72× |
| **`dq_icl_dt`** | **23.8%** | **7.0×** |
| `dq_rai_dt` | 3.3% | 2.1× |
| **`dq_sno_dt`** | **12.6%** | **30.6×** |

### Tightening the threshold does not help, which is the whole diagnosis

3σ → 10σ moves the ice error from 0.2316 to 0.2383 — **flat**. If the error came
from marginal cells near the kink, a stricter criterion would cut it. It does
not, so the error is intrinsic to collapsing rather than to the threshold.

**Distance from the kink does not bound curvature of the integrand.** Clear of
the kink the `max(0, ·)` is smooth — that is all the criterion establishes. It
says nothing about the rest of the integrand, and deposition, sublimation and
snow processes stay strongly nonlinear in T and q far from saturation. The
collapse error is O(σ²·f″) and f″ is large for reasons that have nothing to do
with saturation. §6a measures distance from the **kink**; the quantity that
governs collapse error is curvature of the **whole integrand**. Conflating them
is what motivated this branch, and it is the mistake to avoid repeating.

### What it did confirm

At 10σ roughly 98.7% of points collapse, but only **8.75% of cells change at
all** — so about 91% of collapses are **bit-identical**, not approximately
equal. The degeneracy claim is correct. The entire cost lives in the remaining
~9%, and no threshold on kink distance separates them from the rest.

### Two instrumentation errors, both mine, both worth not repeating

**A knob whose "off" value meant "always on".** `|mu_S| > 0·σ_S` is `|mu_S| > 0`,
true almost everywhere, so a threshold of zero collapsed *every* cell instead of
disabling the branch. The first error measurement therefore compared full
collapse against partial collapse and was discarded. It announced itself only
because the ordering inverted — a stricter threshold appeared to change *more*
cells (1.12% at 10σ against 0.36% at 3σ), which is impossible. **Build the
monotonicity check into the measurement**; it is what caught this.

**A synthetic distribution that sampled only the corner.** The first attempt used
randomized states with variances far larger than the real field's, so the branch
fired in 4% of cases — the marginal ones — and reported a 194% relative error on
tendencies of order 1e-8, a small-denominator artefact. Normalise by the field's
own RMS, and measure on the real state, which is what
`scripts/measure-adaptive-error.jl` does by calling the real cache function and
diffing its output rather than reconstructing the kernel's inputs.

The branch is kept and disabled. The mechanism — a real branch rather than a
mask, warp-uniform in 83% of warps, nearly free to evaluate because `mu_S` and
the σ's are already computed — is sound and worth reusing if someone finds a
criterion that actually bounds the error.

## 4f. The model is densely GPU-bound: there is no launch-overhead headroom (2026-09-09)

Asked whether a structural change to how kernels are created could reach +50%
SYPD, the obvious hypothesis was launch overhead. The model issues **3118 kernel
launches per coupler step** across **346 distinct kernels** at a mean duration of
43.7 µs, and nsys reports 30.5% GPU idle. That profile says "launch-bound", and
CUDA graphs are the textbook answer.

**It is wrong.** Measured without a profiler, using CUDA events around
`step!(cs)` (two event records against a ~145 ms step, so the instrument cannot
dominate what it measures):

| | median over 12 steps |
|---|---|
| wall | 145.2 ms |
| GPU span (first to last GPU activity) | **145.2 ms — 100.0% of wall** |
| gap (host time outside the GPU timeline) | **0.0 ms** |

`gpu_span` is a *span*, so it includes inter-kernel gaps and bounds GPU busy time
from above; the gap is therefore a **lower** bound on host time. It is zero. And
against nsys's ~136 ms/step of kernel time, the GPU timeline is **~94% kernel
execution** — only ~6% gaps.

**So CUDA graphs would buy at most ~6% of GPU time, and probably less.** 3118
launches per step sounds enormous but they pipeline; the GPU never starves. Any
route to a large speedup has to remove arithmetic, not improve scheduling.

The 30.5% idle figure that motivated the hypothesis is the nsys artefact §2b-i
already identified — the same profiled run reports 48.3% idle for the *mod* arm
against 30.5% for baseline while being **faster** in SYPD, which is incoherent.
Two independent observations now say that number is instrument, not model.

### An unresolved discrepancy, recorded rather than smoothed over

SYPD 0.29716 implies **276.6 ms per coupler step**. The measurement above gives a
median of 145.2 ms and a mean of 168.2 ms over 12 steps — **61% of the implied
value**. Roughly 108 ms/step is unaccounted for.

The likely explanation is periodic work outside a 12-step window: one step in the
sample took **421 ms**, nearly 3× the others, and the configuration has radiation
on a 600 s cycle (20 steps), gravity wave drag on 1800 s (60 steps), and hourly
diagnostics (120 steps). A short window samples the cheap steps and misses the
duty cycle.

If that is right it matters more than the launch-overhead question ever did:
**about 40% of long-run wall time would sit in periodic work that the per-step
kernel optimisation in this project never touches.** The flagship microphysics
kernel runs every step; radiation does not. Nobody has measured the duty cycle,
and doing so needs a window of at least 120 steps rather than 12.

Do not treat the 61% as established. It is the difference between two
instruments — a CUDA-event window and the model's own SYPD accounting — and this
project has repeatedly found that such differences are the instrument rather than
the model.

## 4g. The optimisation reverses over a realistic window (2026-09-10)

Every performance result in this project was measured in a 10-step profiling
window on a 40-step run. Lengthening the window to 120 steps (one simulated
hour, the phase-independent LCM) shows the headline result does not survive.

`set_microphysics_tendency_cache` L1013, mean ms per launch, 120 launches:

| launches | baseline | mod | mod/base |
|---|---|---|---|
| 1–12 | 27.49 | **14.16** | **0.51** |
| 25–36 | 28.40 | 19.69 | 0.69 |
| 49–60 | 28.72 | 27.38 | 0.95 |
| 61–72 | 28.73 | 30.49 | **1.06** |
| 97–108 | 28.70 | 37.00 | 1.29 |
| 109–120 | 29.14 | **38.13** | **1.31** |

**Baseline is flat (+6% across the window). The mod arm degrades monotonically,
crosses parity at launch ~60, and ends 31% SLOWER than the code it optimises.**

End to end over one simulated day: **baseline 0.21458 SYPD, mod 0.20111,
−6.70%** — against +4.72% measured on the old window. Walltime per coupling step
0.3828 s baseline against 0.4084 s mod.

### This is not a configuration error

All four arms ran `dt = 30secs`, `t_end = 86400secs`. The manifests dev the
optimised packages. The launch-bounds mechanism fired: `top-kernels.csv` records
255 registers for baseline and 128 for mod, exactly as designed. The
optimisation is present and working as specified; what it does is not what was
wanted.

### Probable mechanism, and the reason to distrust the guard

The likely cause is that the 128-register cap is adequate for the kernel's early
working set and inadequate later, so spill grows with the atmosphere's activity
while baseline's 255 registers absorb it. That is precisely the trade
`LAUNCH_BOUNDS_SPILL_BUDGET` exists to adjudicate — **but the guard evaluates
spill at COMPILE time, once, against one input. It cannot see a working set that
grows at run time.** A budget validated at three points (§4a, §4b) was validated
at three *compile-time* points, all of which said the same thing about a state
the kernel would leave within half a simulated hour.

This is inferred, not measured. Separating the three changes needs one run each
with the mechanism disabled; the register cap is the obvious suspect but the CM
fusion and the evaluator payload have not been ruled out.

### What this invalidates

Every SYPD and kernel figure in this project predating 2026-09-10 was measured
in the first 10 steps of a spin-up, which is the regime where this optimisation
looks best and is least representative. That includes +1.87%, +4.24%, +8.05%,
the −7.9% kernel result, and the quadrature-order comparisons. **The A/B
structure was sound; the window was not.** Re-measurement on the 120-step window
is required before any of those numbers is quoted again.

The general lesson is sharper than "use a longer window". **A short window at the
start of a run does not sample a representative state, it samples an
unrepresentative one — and the direction of the error is not random.** An
optimisation tuned against early-state behaviour will look best exactly where it
was tuned.

## 4h. SUPERSEDED — see 4i. The divergence is upstream CloudMicrophysics, not the fusion (2026-09-10)

Chasing the §4g reversal produced a bigger finding. The two arms do not compute
the same atmosphere.

### The model is deterministic, so the comparison is exact

A null test — the baseline arm run twice, same config, same everything —
is **bit-identical at every snapshot, all 12 prognostic fields, all statistics**.
There is no run-to-run nondeterminism to hide behind, which makes any difference
between arms real by construction.

    BASELINE vs BASELINE    0/12 fields differ at every step
    BASELINE vs MOD        12/12 fields differ from step 20 on

### Bisected to CloudMicrophysics

| configuration | diverges? |
|---|---|
| full mod (fusion + evaluator + launch bounds) | yes |
| launch bounds disabled | yes, identically |
| evaluator reverted, fusion only | **yes, byte-identical to full mod** |

The fusion-only run reproduces the full mod arm's divergence *to the digit*, so
`pb/evaluator-param-args` contributes nothing and its bit-identity does hold in
the coupled model. `pb/1m-spill-fuse` is the sole cause.

### What it changes

    field      step   baseline        mod       mod/base
    c.ρq_sno     20   9.92e-04   7.75e-04         0.781
    c.ρq_sno    120   6.66e-01   6.28e-01         0.943
    c.ρq_rai    120   3.81e-01   3.81e-01        1.0015
    c.ρq_tot    120   5.16e+03   5.16e+03        1.0000

**Total water is conserved to six digits**; the fusion shifts the snow/rain
partitioning, producing ~22% less snow at step 20 and settling near 6% less.
This is not a mass-conservation bug and not roundoff — it is a systematic
process-level difference that appears within 20 coupled steps.

### Why the equivalence test missed it

`test/bulk_tendencies_tests.jl` compares fused against unfused over 2000
randomized trials per float type and asserts zero mismatches. It passes. The
model still diverges.

**A randomized unit test samples a distribution the author chose; the model
samples the one the physics produces.** The same error shape as the 10-step
profiling window in §4g: a sample that looked broad and was not representative.
Whatever the fusion does differently, it does it in states the random sampler
generates rarely or never — and the commit message's claim that each accumulator
receives contributions "in `_linearize`'s order" is therefore not true in
general.

### What this invalidates

The fusion is not a pure optimisation and must not be presented as one. Every
performance comparison involving it measured two arms solving *different
problems*, so the attribution of any speedup to the fusion is unsound
independent of the window issue in §4g.

It does NOT explain §4g's kernel degradation. The mod arm produces *less*
condensate, which would make its microphysics cheaper, not 2.4x more expensive.
Those are two separate open problems.

## 4i. The arms were comparing different CloudMicrophysics versions (2026-09-11)

§4h blamed the source-term fusion for the state divergence. That was wrong, and
the way it was wrong is the lesson.

### The bisection, completed

| configuration | step-20 worst | step-120 worst |
|---|---|---|
| full mod (upstream drift + our CM changes) | 2.190e-01 | 5.752e-02 |
| **CM = upstream main at our branch point, NONE of our changes** | **2.190e-01** | **5.771e-02** |
| CM = v0.38.3, identical to baseline | **0** | **0** |

Removing every change we wrote leaves the divergence **identical to three
digits**. The cause is upstream CloudMicrophysics between v0.38.3 and
`cf58726f` — 3 source files, chiefly `MicrophysicsNonEq.jl` (64 lines) and
`Microphysics1MOptions.jl` (18 lines). None of it ours.

Two independent confirmations that our work is not responsible:

  * `_fused_linearize` matches `_linearize(_microphysics_source_terms(...))`
    **bit for bit on all 1,548,288 real model states** after 20 coupled steps,
    measured by `scripts/measure-fusion-mismatch.jl`. §4h's claim that the
    fusion's accumulation order "is not true in general" was unfounded.
  * With CM pinned to v0.38.3, the mod arm — still carrying ClimaCore's launch
    bounds and ClimaAtmos's evaluator payload — is **bit-identical to baseline at
    every snapshot**. Both of those changes are provably neutral in the full
    coupled model over 120 steps, which is a far stronger statement than the
    unit-level equivalence tests that were being quoted for them.

### The design flaw this exposes

**The baseline arm pins CloudMicrophysics to a registry release; the mod arm
tracks a branch off main.** So every CM experiment in this project has compared
our change *plus* however much upstream had moved, and `arms_differ_in` reports
`CloudMicrophysics.jl-mod` either way — it cannot distinguish the two. The signal
was visible earlier and went unread: the `make-diffs` split showed 39 files
against the 5 we authored, and that gap *is* the drift.

The fix is to pin both arms to the same CloudMicrophysics base, so the arms
differ only by what we wrote. Until that is done, no CM experiment in
`experiments.csv` isolates its stated treatment.

### On the attribution error

The bisection was sound at every step; the attribution ran ahead of it. Each
experiment showed *some part of the mod stack* was responsible and the part named
was the one already under discussion, rather than the one the evidence isolated.
**"The mod arm differs" is not "our change differs" whenever the mod arm also
carries a different upstream.** Check what else rides along before naming a
cause.

## 4j. A clean number: the evaluator payload change is −5.46% (2026-09-11)

The first performance result in this project measured without a confound.

| | registers | mean over 120 launches | first-qtr | last-qtr | ratio |
|---|---|---|---|---|---|
| baseline | 255 | 28.51 ms | 27.88 | 28.81 | 1.03 |
| + evaluator payload | 255 | **26.96 ms** | 26.35 | 27.20 | 1.03 |

**−5.46% on `set_microphysics_tendency_cache`, flat across the window.**

### Why this one is trustworthy where the others were not

  * **Phase-independent window.** 120 steps = LCM(20, 60, 120), so radiation,
    both gravity-wave schemes, cloud fraction and the diagnostics write each
    fire at their true long-run frequency. The old 10-step window fired the
    gravity-wave schemes 0 or 1 times depending on phase (§4g).
  * **Byte-identical everything else.** ClimaCore at the baseline commit,
    CloudMicrophysics at v0.38.3 in both arms. No upstream drift riding along,
    which is what invalidated every CM comparison (§4i).
  * **Bit-identical model state** over 120 coupled steps, on a model shown
    deterministic by a null test (§4h/§4i).
  * **Reproduced.** An earlier configuration — ClimaCore's branch present but
    its launch bounds correctly declining to apply — gave 28.53 → 26.98,
    −5.43%. Two setups, 0.03 points apart.
  * **Both arms flat** (ratio 1.03). No degradation, unlike §4g.

### The constraint this ran into

ClimaAtmos main now requires CloudMicrophysics **0.39** (`PrescribedIceNumber`),
which is incompatible with the v0.38.3 pin that removes the CM drift. Those two
cannot both hold, so the measurement runs at ClimaAtmos `31d34486b`. That is
sound here: the 7 commits to current main touch only `cloud_fraction.jl` in the
microphysics tree, not the two files the change lives in.

Measuring against current main needs CM 0.39 in **both** arms, where the drift
is common-mode and cancels. That is the dependency-update cycle §4i already
calls for, and it is the prerequisite for any future CloudMicrophysics
measurement.

### Scale

Honest framing for a reviewer: L1013 is ~14% of GPU time on this window, and
diagnostics are ~34%. A 5.46% kernel improvement is roughly 0.8% of GPU time.
This is a clean, real, modest result — not a headline.

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

**The nine-point quadrature costs 5.45% SYPD** (tagged 2026-09-08; an earlier
measurement on a slower stack put it at 7.93%, and the difference is the
optimisation work since). Removing it entirely would put
the flagship run 9.66% above today's baseline.

This is an **upper bound and not a candidate**: `quadrature_order: 1` changes
results, and the config was reverted immediately after. It exists so the science
question has a number attached — *is the quadrature worth 8% of the flagship
run's throughput, given the PDF sits 449σ from the feature it resolves?*

An adaptive collapse (item 2) captures only the warps that qualify, so expect
roughly `0.83 × 5.45% ≈ 4.5%` before subtracting the branch cost and the
divergence in the ~17% of mixed warps. The full 5.45% is available only if the
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

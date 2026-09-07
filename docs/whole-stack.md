# Improving AMIP performance takes changes in three repositories at once

The flagship AMIP simulation can be made meaningfully faster, and the current
best measured configuration is **+4.24% SYPD** (0.28312 → 0.29565, mean of three
runs). But the result does not live in any one package. It is the product of
changes in CloudMicrophysics, ClimaCore and ClimaAtmos, and **the two largest
contributors are worth almost nothing on their own.**

That is the finding worth carrying away. It is not a note about this particular
kernel; it is a claim about how performance work in this stack has to be
organised and reviewed.

## The decomposition

Every row is a full AMIP run against the same baseline (0.28312), tagged in
`experiments.csv` and reproducible from its commit.

| change | repo | measured **alone** | measured **cumulatively** |
|---|---|---|---|
| source-term / linearize fusion | CloudMicrophysics | **+1.87%** | +1.95% |
| `__launch_bounds__` occupancy targeting @12 warps | ClimaCore | **−0.016%** | +2.56% |
| same, retargeted @16 warps | ClimaCore | — | **+4.24%** |
| evaluator parameter payload | ClimaAtmos | not measured alone | +3.78% ¹ |

¹ single sample, inside the 0.43% noise floor — see "What ClimaAtmos actually
contributes" below.

All percentages follow this project's `speedup_pct` convention,
`(mod - baseline) / mod`, matching `experiments.csv`. The other convention,
dividing by baseline, gives visibly different numbers on the same run (the
ClimaAtmos row is 3.78% here and 3.93% that way), so a figure quoted without
its denominator cannot be compared against these.

**The `−0.016%` is the important number in this table.** Launch-bounds occupancy
targeting, measured by itself on 2026-08-24 against an unmodified stack, was a
null. Not a small win — nothing, slightly negative, indistinguishable from
run-to-run variation. On that evidence the ClimaCore pull request was closed as
a dead end on 2026-08-31.

It was not a dead end. The same mechanism, unchanged, is worth **+2.6 percentage
points** once the CloudMicrophysics fusion is present. The fusion frees register
headroom; the occupancy machinery converts headroom into resident warps. Neither
half can do the other's job:

- Removing work creates headroom but does not by itself turn it into occupancy,
  because the compiler will spend the slack on other things.
- Occupancy targeting converts headroom but cannot create it. Asked to hit a
  target the kernel cannot afford, it makes things worse — a 24-warp target
  costs +392 B of spill and runs the kernel **18.4% slower**.

This pattern has now appeared four separate times in this project. It should be
the default expectation for GPU work here, not a surprise.

## Why this is a review problem, not just an engineering one

Each of these changes, submitted to its own repository and reviewed on its own
merits, looks like this to a reviewer:

- **CloudMicrophysics**: a real but modest +1.87%, arguably not worth the
  complexity of a fused code path.
- **ClimaCore**: a null result. Correctly rejected by any reviewer applying the
  ordinary standard of "show me the improvement."
- **ClimaAtmos**: a kernel 8.8% faster, with no measurable effect on the model.

Every one of those judgements is locally correct and the combination is worth
+4.24%. A per-repository review process cannot see this. The ClimaCore branch
survived only because its git tag and branch were kept after the PR was closed,
and the fusion made it viable again three days later — that was luck, not
process.

**Recommendation:** land cross-repo performance work as a declared set, with one
shared write-up carrying the combined measurement, and review the members
against that rather than individually. A standalone null is not evidence of a
useless change when the change is an *enabler*.

## What ClimaAtmos actually contributes

The ClimaAtmos change looks negligible in the SYPD column and is easy to
misread. Its measured effect on the hot kernel is the **largest single kernel
improvement in the project**:

| | before | after |
|---|---|---|
| `sizeof(Microphysics1MEvaluator)` | 472 B | **40 B** |
| unbounded registers | 255 *(pinned at the hardware cap)* | **184** |
| unbounded local memory | 1864 B | **1432 B** |
| L1013, 10 launches | 150.3 ms | **137.2 ms (−8.8%)** |

The −432 B is exactly `sizeof(mp) + sizeof(tps)`: two parameter structs,
identical for every cell and every quadrature point, that were stored as fields
of a per-cell struct. A struct built per cell becomes an alloca the ABI writes
to local memory before every `@noinline` call; the same values passed as
arguments stay in `.param` space. Output is bit-for-bit identical across 400
randomized states × 4 tendencies.

It still cannot be justified on SYPD, and the arithmetic says so in advance:
L1013 is **10.1% of GPU time**, so −8.8% on it is ~0.95% of kernel time and
**~0.3% SYPD** at the ⅓ pass-through this project keeps measuring — below the
0.43% noise floor. No single AMIP run could ever have resolved it.

That is a general lesson about what to measure with what: **a whole-model
benchmark cannot validate an improvement to a kernel holding a small share of
the budget.** Check the share first. The kernel-level measurement used here
(summed kernel time over 10 launches under nsys) has a 0.31% repeat spread, so
it resolves this change at roughly 28× its noise. Use the instrument matched to
the claim.

## What it does not buy

Slimming the evaluator took registers off the 255 cap and it is tempting to read
that as a path to dropping the ClimaCore dependency. It is not. Launch bounds
already delivers **16 warps/SM at 128 registers**; reaching 168 registers
unaided would buy 12, which is worse than what we have. The threshold that would
make ClimaCore redundant is ≤128 registers unbounded — 56 below where we sit,
with a 64-register framework floor that leaves 64 for physics and quadrature
combined, which currently cost 120.

**Occupancy at 16 warps is reachable only through launch bounds.** The
three-repository coupling is real and cannot be refactored away.

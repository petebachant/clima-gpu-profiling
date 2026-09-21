# A CI gate for kernel resource regressions

This is a design derived from this project's measurements, not a tested CI
configuration. Nothing here has been implemented or validated.

Deliberately no figures are typed in this document. Every number the design
rests on is cited as evidence on the corresponding `calkit.yaml` question,
where `calkit check questions` resolves it against the results file and
reports when it moves.

## Why the obvious gate cannot work

The natural check --- benchmark the model, fail the build if it got slower ---
is unusable here, and this project has the numbers to say why. Full-AMIP SYPD
has a noise floor of a few tenths of a percent at best and around two percent
typically, while a change that made the hot kernel substantially faster is
worth a fraction of a percent end to end. A threshold loose enough to avoid
false alarms passes almost every regression worth catching; a tight one fails
constantly.

The quantities that actually diagnose these kernels are compile-time and
deterministic, so the gate should be built on those instead. Both come from
`CUDA.registers` and `CUDA.memory` without running the kernel or needing a
benchmark, so they are exactly reproducible and cost seconds.

## What to record

Registers per thread **and** absolute local-memory bytes, per kernel, as a
checked-in baseline, with the build failing on a regression or on crossing an
absolute limit.

Three specifics this project learned the hard way, each of which a naive
version of the check would get wrong.

### Registers alone are not enough, and are actively misleading

A kernel at the 255-register hardware cap on sm_80 reports 255 no matter how
far over it is. Register count therefore cannot distinguish "fine" from
"catastrophic", and cannot show whether a fix helped: three separate attempts
in this project read as no-ops for exactly this reason. Local-memory bytes must
be recorded alongside, in absolute terms.

### Record absolutes, not differences

The launch-bounds decision record originally logged only `spill_growth`, the
difference between two compiles. A difference cannot separate "demand fell"
from "both compiles moved together", and it hid a register change that turned
out to be the compiler relabeling identical total cost.

### Spilling at all is the signal worth alerting on

The flagship kernel spilled a large share of its memory traffic while looking
unremarkable in every timing table, because nsys does not collect spill data.
Any kernel compiling to nonzero local memory deserves a warning, and one at the
register cap deserves a hard failure.

## Where it would go

ClimaCore's launch-bounds decision record already computes every one of these
numbers per kernel at compile time, which makes it the natural place to build
this rather than a new piece of infrastructure.

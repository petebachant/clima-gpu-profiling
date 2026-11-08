# The hot kernel's redundant saturation work

The environment quadrature broadcast of `set_microphysics_tendency_cache!` is the heaviest kernel on the device, and `docs/experiments/occupancy-launch-model.md` priced its occupancy and found nothing to collect.
This experiment asks the other question about the same kernel: not whether it can be made more occupant, but whether it is doing arithmetic it does not need.
It was, twice over, and removing both is bit-for-bit --- the treated and untreated arms agree on a hash of every output bit of <!-- calkit value key=probe_summary.points path=results/experiments/quadrature-saturation/stats.json -->4368384<!-- /calkit value --> points, because the treatment only reuses a value the kernel had already computed and deletes one it never read.

## Why this experiment declares no profiling stages

It reads the pipeline's own arm profiles rather than taking its own pair, because the treatment is a change to one arm and the pipeline already profiles both arms of exactly that comparison.
That makes `results/nsys/mod.sqlite` an input, which is transient --- the next experiment overwrites it --- so the analysis below snapshots every derived number into `results/experiments/quadrature-saturation/stats.json`, which is a few kilobytes in Git and carries the arm refs and the digest of each profile it read.
Once this experiment is tagged, its stage is frozen on main, so a later pass cannot restate this write-up against a different run.

The compile-only probe is the second instrument and the one that carries the accuracy claim.
`scripts/measure-quadrature-registers.jl` compiles the real broadcast on the model's AMIP he16 layout, reads the register count out of ClimaCore's own launch record, times the launch with CUDA events, and hashes the output bits, in minutes of one GPU rather than the ~45 an nsys stage costs.
Its state is varied and almost every cell carries condensate, so it prices the full-work path and runs about twice the profile's per-launch time; its percentage is therefore an upper bound on the profile's, since a cell that takes an early-out never pays the work the treatment removes.

## Analysis

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/quadrature-saturation/stats.json, storage: git}] inputs=[scripts/export_quadrature_saturation.py, results/nsys/baseline.sqlite, results/nsys/mod.sqlite, results/quadrature-registers-work-baseline.toml, results/quadrature-registers-work-mod-satreuse-only.toml, results/quadrature-registers-work-mod.toml, results/experiments/occupancy-launch-model/stats.json]
import sys

sys.path.insert(0, "scripts")
from export_quadrature_saturation import main

stats = main("results/experiments/quadrature-saturation/stats.json")

# The whole claim of this treatment is that the answer did not change, so assert
# it rather than render a boolean and hope a reader checks it.
assert stats["probe_summary"]["bit_for_bit"], stats["probe_summary"]
assert stats["probe_summary"]["control_bit_for_bit"], stats["probe_summary"]
# Equal launch counts, or the two arms' kernels are not doing the same work and
# the per-kernel comparison below is between two different quantities.
assert stats["kernel"]["launches_agree"], stats["kernel"]
# The frame census is the reason no register treatment is attempted here. If
# CloudMicrophysics' own frame ever fits under the step, that conclusion is void.
assert not stats["frames"]["occupancy_reachable"], stats["frames"]
# The saving is arithmetic, not occupancy. Identical registers in an identical
# block in every state is what rules the alternative out, so assert all three.
assert stats["probe_summary"]["registers_unchanged"], stats["probe_summary"]
assert stats["probe_summary"]["block_size_unchanged"], stats["probe_summary"]
assert stats["probe_summary"]["one_kernel_each"], stats["probe_summary"]
```

## Why this is a work experiment and not an occupancy one

A kernel's register count is the largest frame on its call graph, not the sum of its parts, because the node function is `@noinline` and `ptxas` allocates for the peak.
A barrier can therefore split a frame but never shrink one, which makes the largest frame reachable from ClimaAtmos the floor of every barrier arrangement there is --- and that floor is measurable by passing stubs, with no edit to ClimaAtmos at all, since `sum_over_conditional_nodes` takes the node function as an argument.

The census puts the whole kernel at <!-- calkit value key=frames.whole_kernel_registers path=results/experiments/quadrature-saturation/stats.json -->130<!-- /calkit value --> registers, the node traversal with a stub node function at <!-- calkit value key=frames.traversal_registers path=results/experiments/quadrature-saturation/stats.json -->84<!-- /calkit value -->, the real node function at <!-- calkit value key=frames.node_function_registers path=results/experiments/quadrature-saturation/stats.json -->112<!-- /calkit value -->, and CloudMicrophysics' substep-averaged bulk tendency on its own at <!-- calkit value key=frames.bmt_registers path=results/experiments/quadrature-saturation/stats.json -->102<!-- /calkit value -->.
ClimaAtmos's own reconstruction is thus worth about ten registers on top of CloudMicrophysics', and the next occupancy step up needs <!-- calkit value key=frames.step_to_gain_registers path=results/experiments/quadrature-saturation/stats.json -->96<!-- /calkit value --> or fewer, which CloudMicrophysics misses on its own by <!-- calkit value key=frames.bmt_over_step path=results/experiments/quadrature-saturation/stats.json -->6<!-- /calkit value -->.
So the bulk tendency alone already sits at <!-- calkit value key=frames.bmt_warps_per_sm path=results/experiments/quadrature-saturation/stats.json -->16<!-- /calkit value --> resident warps per multiprocessor, which is exactly what the whole kernel gets today at <!-- calkit value key=kernel.baseline.registers path=results/experiments/quadrature-saturation/stats.json -->128<!-- /calkit value --> registers: a ClimaAtmos-side register treatment that worked perfectly would land in the same occupancy band it started in.
That is a stronger statement than "the registers are in a package neither arm devs", which is how this was left before, and it is why the lever pursued here is arithmetic rather than occupancy.

## What was removed

Two saturation specific humidities per node were being computed and discarded.

**The node carried a total water that nothing read, one saturation evaluation per node for each of the rule's <!-- calkit value key=work.nodes path=results/experiments/quadrature-saturation/stats.json -->8<!-- /calkit value --> nodes.** `ConditionalNode` held a `q` field, set by `_conditional_node` from a `TD.q_vap_saturation` call at the node temperature, and no production code path ever read it --- only two test assertions did.
It is a leftover from before the temperature-point split: `_point_value` now sets the state the evaluator receives per temperature point, so a copy of it at the node temperature is both unused and, when there is no split, the same number.
Deleting the field removes eight saturation evaluations per cell and eight registers from the eight-node table, which is live across the whole integration, and it leaves `conditional_rule` a function of the Gaussian geometry alone: it no longer needs the thermodynamic parameters, the density or the mean excess to place its nodes.

**The node function recomputed the saturation the traversal had just evaluated.** `_point_value` evaluates `TD.q_vap_saturation(thp, T_pt, ρ)` to build the point's total water, and the evaluator's first act was to call `TD.q_vap_saturation(tps, T_hat, ρ)` with `T_hat` bound to that same `T_pt`, the same density and the same parameters, to recover the saturation excess from the state it had just been handed.
Passing the value across the barrier instead removes up to sixteen evaluations per cell and takes the saturation computation's working set out of the frame that caps the kernel.
It is exact rather than nearly exact: the same float is reused in the same expression, so the reconstruction is not merely equivalent but identical.

Of the <!-- calkit value key=work.calls_before path=results/experiments/quadrature-saturation/stats.json -->40<!-- /calkit value --> saturation evaluations a cloudy cell performed in this kernel, <!-- calkit value key=work.calls_removed path=results/experiments/quadrature-saturation/stats.json -->24<!-- /calkit value --> were dead or duplicated, leaving the <!-- calkit value key=work.calls_after path=results/experiments/quadrature-saturation/stats.json -->16<!-- /calkit value --> that have to happen --- one per temperature point, each at its own temperature.

## What it is worth

On the model's own benchmark window the kernel went <!-- calkit value key=kernel.baseline.total_ms path=results/experiments/quadrature-saturation/stats.json -->4988.2<!-- /calkit value --> to <!-- calkit value key=kernel.mod.total_ms path=results/experiments/quadrature-saturation/stats.json -->4988.2<!-- /calkit value --> ms over <!-- calkit value key=kernel.baseline.launches path=results/experiments/quadrature-saturation/stats.json -->240<!-- /calkit value --> launches, a saving of <!-- calkit value key=kernel.saved_ms path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value --> ms or <!-- calkit value key=kernel.saved_pct path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value -->%.
Every other kernel in the profile is a control, since the treatment is one broadcast: of the <!-- calkit value key=control.kernels_over_100ms path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value --> paired kernels holding more than 100 ms, the median moved <!-- calkit value key=control.median_abs_change_pct path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value -->% in absolute value and the worst moved <!-- calkit value key=control.worst_change_pct path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value -->%.
At the <!-- calkit value key=end_to_end.sypd_pct_per_device_ms path=results/experiments/quadrature-saturation/stats.json -->0.00261<!-- /calkit value -->% of SYPD per device millisecond measured on this stack, the saving is <!-- calkit value key=end_to_end.sypd_points path=results/experiments/quadrature-saturation/stats.json -->0<!-- /calkit value --> points against a null floor of <!-- calkit value key=end_to_end.null_floor_pct path=results/experiments/quadrature-saturation/stats.json -->1.869<!-- /calkit value -->%, so whether one AMIP pair can resolve it end to end is a separate question from whether the kernel got faster.
The kernel figure is the one to quote: device-side kernel times from nsys are trustworthy here, and the saving is a change to one broadcast rather than to the schedule.

The probe prices the same treatment on its full-work state at <!-- calkit value key=probe_summary.saved_pct_total path=results/experiments/quadrature-saturation/stats.json -->11.74<!-- /calkit value -->%, and because it was run at each commit separately it also splits it.
Handing the saturation across the barrier is worth <!-- calkit value key=probe_summary.saved_pct_saturation_reuse path=results/experiments/quadrature-saturation/stats.json -->4.52<!-- /calkit value -->% and deleting the unread field a further <!-- calkit value key=probe_summary.saved_pct_dead_field path=results/experiments/quadrature-saturation/stats.json -->7.22<!-- /calkit value -->%, which is the opposite order from the one counting evaluations predicts: the field removes half as many saturation calls and is worth more than the duplicate, because it was also eight registers of state live across the whole traversal, and the spill fell by <!-- calkit value key=probe_summary.spill_moved_bytes path=results/experiments/quadrature-saturation/stats.json -->-32<!-- /calkit value --> bytes with it.
The saving is arithmetic and not occupancy, which the probe settles rather than argues: every state compiles to the same <!-- calkit value key=probe_summary.registers path=results/experiments/quadrature-saturation/stats.json -->130<!-- /calkit value --> registers in the same block of <!-- calkit value key=probe_summary.block_size path=results/experiments/quadrature-saturation/stats.json -->384<!-- /calkit value --> threads and launches one kernel, so the launch geometry is identical and only the work inside it changed.
The non-quadrature sibling compiled in the same runs is the in-run control: it shares the launcher, the layout and the parameter construction, it is untouched by the treatment, and it moved <!-- calkit value key=probe_summary.control_change_pct path=results/experiments/quadrature-saturation/stats.json -->-0.11<!-- /calkit value -->%.
Had the split not been measured it would have been reported backwards, which is the second time on this kernel that counting something has disagreed with compiling it.

## That the answer did not change

The probe writes an FNV-1a hash of the raw bits of all four tendency components over the broadcast's whole output: <!-- calkit value key=probe_summary.points path=results/experiments/quadrature-saturation/stats.json -->4368384<!-- /calkit value --> points of a varied state spanning clear, cloudy, mixed-phase and precipitating cells, which is the model's own AMIP he16 layout.
The untreated arm, the arm with the saturation reuse alone and the fully treated arm all produce the same hash, and so does the untouched sibling, which the analysis asserts rather than renders.
This is the reason to prefer this treatment over the node count: the other lever on this kernel is how many points the quadrature evaluates, and that changes the answer, whereas these two removals cannot --- one deletes a field nothing reads and the other reuses a float already in a register.

## What is left

The kernel is still the heaviest on the device, and what remains of it is the work it exists to do: up to sixteen evaluations of the 1M microphysics per cell, eight nodes each split at the freezing point, inside a package neither arm devs.
Both remaining levers are outside this experiment's reach.
Lowering the node count would be the largest single saving available and it is an accuracy decision, not an implementation one.
Devving CloudMicrophysics would reach the <!-- calkit value key=frames.bmt_registers path=results/experiments/quadrature-saturation/stats.json -->102<!-- /calkit value --> registers of the bulk tendency, where the <!-- calkit value key=frames.bmt_over_step path=results/experiments/quadrature-saturation/stats.json -->6<!-- /calkit value --> registers between it and the next occupancy step are, and it rewrites the AMIP manifest and invalidates every profiling stage to try.

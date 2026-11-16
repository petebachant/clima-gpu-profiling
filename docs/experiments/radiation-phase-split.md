# Can the radiation solve be split into pieces that clear an occupancy step?

The two 2-stream solvers are the largest block in the device budget after the SGS closure, and they are the clearest case in this project of a kernel held by one resource.
They compile to 255 registers per thread, which at 256 threads per block puts one block on a multiprocessor and 8 warps of a possible 64 on it, and ncu measures achieved occupancy at essentially the theoretical ceiling — so there is no latency-hiding shortfall to recover, the ceiling itself is the whole problem.
They also already spill at that budget, 11.7 million local-memory spilling requests longwave and 16.2 million shortwave, which is the fact that decides how to approach them.

Forcing the budget down has been tried twice and lost twice: 2 blocks per multiprocessor at 128 registers measured +3.93% and 128-thread blocks at 168 registers +2.91%.
Section 8t priced why, by compiling both: the forced variants reach 16 warps and roughly 2.3x the local memory, 248 to 576 bytes per thread longwave and 288 to 600 shortwave.
A kernel whose stall is already 57-61% long-scoreboard has nothing to hide added local traffic behind, so paying latency to buy occupancy is a losing trade on this family.

That is the argument for not pushing on registers at all.
`ptxas` is not being stingy at 255 — it is out of room, which is what the spill says, so the live state exceeds what the register file can hold and a cap only moves the overflow from registers into memory.
The structural alternative is to make the live state smaller by splitting the solve into phases, each compiled as its own kernel, so that no single piece has to hold the whole column sweep's working set at once.
Whether that can work is a question about where the live state sits, and it is answerable before anything is built.

## What decides it, and why the sweeps are not the answer

The phase that matters is the gas and aerosol optics.
Section 4l item 5 measured 99% of RRTMGP's time in the optics and 0.1% in the vertical sweeps, so a split that moves the sweeps into their own kernel moves almost none of the time to a higher occupancy, however well those kernels compile.
If the optics phase alone comes back near 255 registers, the split dies for the same reason the microphysics quadrature split died: each piece still contained one full CloudMicrophysics evaluation at ~246 registers, so every piece ran at the same 8 warps.

There is a second, nearer bar, and it costs nothing to ask.
At 256 threads the only step is 128 registers, because two blocks of 256 need 65536/(2*256); at 128 threads a third block fits at 170 registers, and at the cap both block sizes give the same 8 warps.
So a phase that cannot reach 128 may still gain a step at the smaller block, and that is a weaker requirement than the budget the two forced attempts failed to meet cleanly.

## The probe

Each phase is compiled as its own kernel with the live solver's own argument types, after one warmup step of the real coupled simulation, so the types and the inlining are the model's rather than a synthetic body's.
Section 4ab records why that matters: measuring an isolated body once reversed a conclusion, because the body had register room the real kernel did not.
`reaches_target` requires both the register count and zero spill, since a budget met by spilling is exactly what lost twice.
The controls are the whole fused and grouped kernels, which must come back at the cap or the probe is not describing the kernel the model runs.

This probe was written in an earlier pass and had never produced a reading: all seven phase stubs failed to compile.
Section 8t attributed that to the stubs calling the per-column bodies outside `@inbounds`, which was wrong — every stub already had one.
The cause was that the file imports CUDA rather than using it, so the bare `threadIdx()` in each stub was an undefined name, and the dynamic dispatch that followed buried the real reason several lines into an error whose first lines name only its consequences.
Seven stubs, seven failures, one per unqualified intrinsic.

```sh calkit stage name=phases environment=clima outputs=[{path: results/experiments/radiation-phase-split/phases.toml, storage: git}] inputs=[scripts/measure-radiation-phase-registers.jl, scripts/run-julia-script.sh, RRTMGP.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=60]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-radiation-phase-registers.jl \
  --out results/experiments/radiation-phase-split/phases.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/radiation-phase-split.json, storage: git}] inputs=[results/experiments/radiation-phase-split/phases.toml, results/top-kernels.csv]
import csv
import json
import tomllib

d = tomllib.load(open("results/experiments/radiation-phase-split/phases.toml", "rb"))
s = d["summary"]

# What the two solvers cost in the profiled window, and the device budget they
# sit in, taken from the profile's own rows so the denominator is the profile's.
with open("results/top-kernels.csv", newline="") as f:
    rows = list(csv.DictReader(f))


def ms_and_pct(prefix, arm):
    for r in rows:
        if r["kernel"].startswith(prefix):
            try:
                return float(r[f"total_ms_{arm}"]), float(r[f"pct_of_gpu_time_{arm}"])
            except ValueError:
                continue
    raise SystemExit(f"no {prefix} row for the {arm} arm")


lw_ms, lw_pct = ms_and_pct("rte_lw_2stream_solve_grouped", "mod")
sw_ms, sw_pct = ms_and_pct("rte_sw_2stream_solve_grouped", "mod")
device_ms = 100.0 * lw_ms / lw_pct
solver_ms = lw_ms + sw_ms

PHASES = [k for k, v in d.items() if isinstance(v, dict) and k != "summary"]


def phase(name):
    v = d[name]
    if "error" in v:
        # The first line only: an InvalidIRError's later Reason lines are the
        # consequences of its first, and the whole text is in the TOML.
        return {"compiled": False, "error_head": v["error"].split("\n")[0][:200]}
    return {
        "compiled": True,
        "registers": v["registers"],
        "local_bytes": v["local_bytes"],
        "warps_per_sm": v["warps_per_sm"],
        "warps_per_sm_b128": v.get("warps_per_sm_b128"),
        "at_cap": v["at_cap"],
        "reaches_target": v["reaches_target"],
        "gains_step": v.get("gains_step"),
        "gains_step_b128": v.get("gains_step_b128"),
    }


phases = {name: phase(name) for name in sorted(PHASES)}
failed = sorted(n for n, p in phases.items() if not p["compiled"])

# The occupancy ladder at both block sizes, so a phase's register count can be
# read against what it would buy rather than against the cap alone.
#
# This follows ClimaCore's `max_active_blocks`, a port of NVIDIA's
# `cudaOccMaxActiveBlocksPerMultiprocessor`, and must match the model the probe
# uses or this file would contradict its own `alt_target_registers`. Dividing
# the register file by regs*block instead puts the 128-thread step at 170
# registers; the granularity-correct answer is 168, because registers are
# allocated per warp in 256-register units and the file is divided per
# scheduler, not per multiprocessor.
RF, THREADS_PER_SM, MAX_BLOCKS = 65536, 2048, 32
REGS_PER_ALLOCATION, MAX_REGS_PER_BLOCK, SCHEDULERS, TPW = 256, 65536, 4, 32


def round_up(n, divisor):
    return -(-n // divisor) * divisor


def warps(regs, block):
    warps_per_block = -(-block // TPW)
    max_warps = THREADS_PER_SM // TPW
    if regs > 0:
        allocated_warps = round_up(warps_per_block, SCHEDULERS)
        allocated_regs_per_warp = round_up(regs * TPW, REGS_PER_ALLOCATION)
        if allocated_warps * allocated_regs_per_warp > MAX_REGS_PER_BLOCK:
            return 0
        per_scheduler = (RF // SCHEDULERS) // allocated_regs_per_warp
        max_warps = min(max_warps, per_scheduler * SCHEDULERS)
    blocks = max_warps // warps_per_block
    # Blocks of two warps are dealt unevenly to four schedulers unless each
    # gets the same number.
    if warps_per_block == 2:
        blocks = 4 * (blocks // 4)
    return min(blocks, MAX_BLOCKS) * warps_per_block


def ladder(block):
    out, seen = [], set()
    for regs in range(255, 31, -1):
        w = warps(regs, block)
        if w in seen or w == 0:
            continue
        seen.add(w)
        out.append({"max_registers": regs, "warps_per_sm": w,
                    "occupancy_pct": round(100.0 * w / 64, 1)})
    return out


out = {
    "phases": phases,
    "phases_measured": len(phases) - len(failed),
    "phases_failed": len(failed),
    "failed": failed,
    "probe_is_complete": not failed,
    # The controls: if these are not at the cap the probe is not describing the
    # kernel the model runs, and nothing below means anything.
    "controls_at_cap": bool(
        s["control_lw_grouped_at_cap"] and s["control_sw_grouped_at_cap"]
    ),
    "register_cap": s["register_cap"],
    "current_warps_per_sm": s["current_warps_per_sm"],
    "block_size": s["block_size"],
    "target_registers": s["target_registers"],
    "target_warps_per_sm": s["target_warps_per_sm"],
    "alt_block_size": s.get("alt_block_size"),
    "alt_target_registers": s.get("alt_target_registers"),
    "alt_target_warps_per_sm": s.get("alt_target_warps_per_sm"),
    # The verdict fields, which are what the experiment exists to set.
    "lw_optics_phase_fits": s["lw_optics_phase_fits"],
    "sw_optics_phase_fits": s["sw_optics_phase_fits"],
    "split_is_viable": s["split_is_viable"],
    "lw_optics_phase_gains_step_b128": s.get("lw_optics_phase_gains_step_b128"),
    "sw_optics_phase_gains_step_b128": s.get("sw_optics_phase_gains_step_b128"),
    "split_gains_a_step_b128": s.get("split_gains_a_step_b128"),
    "phases_that_fit": s["phases_that_fit"],
    "phases_that_gain_a_step_b128": s.get("phases_that_gain_a_step_b128", []),
    "forced": s["forced"],
    "forced_budget_is_clean": s["forced_budget_is_clean"],
    "ladder": {"b256": ladder(256), "b128": ladder(128)},
    "device": {
        "device_ms": round(device_ms, 1),
        "lw_ms": round(lw_ms, 1),
        "sw_ms": round(sw_ms, 1),
        "solver_ms": round(solver_ms, 1),
        "solver_pct_of_device": round(100.0 * solver_ms / device_ms, 2),
        "lw_pct_of_device": lw_pct,
        "sw_pct_of_device": sw_pct,
    },
    "grid": {"ncol": s["ncol"], "nlay": s["nlay"],
             "ngroups_lw": s["ngroups_lw"], "ngroups_sw": s["ngroups_sw"]},
}
json.dump(out, open("results/docs/radiation-phase-split.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/radiation-phase-split.json -->

## Result

Every variant produced a reading, for the first time since the probe was written: <!-- calkit value key=phases_measured -->15<!-- /calkit value --> measured and <!-- calkit value key=phases_failed -->0<!-- /calkit value --> failed.
All four whole-kernel controls came back at the <!-- calkit value key=register_cap -->255<!-- /calkit value -->-register cap, which is what says the probe is describing the kernels the model launches rather than a synthetic body.

The phase that carries the time unfuses well below the cap.
Longwave gas and aerosol optics compiles to <!-- calkit value key=phases.lw_gas_aerosol_optics.registers -->156<!-- /calkit value --> registers as its own kernel, and shortwave to <!-- calkit value key=phases.sw_gas_aerosol_optics.registers -->128<!-- /calkit value -->, against the <!-- calkit value key=register_cap -->255<!-- /calkit value --> the fused kernels sit at.
Both carry <!-- calkit value key=phases.lw_gas_aerosol_optics.local_bytes -->32<!-- /calkit value --> bytes of local memory, against <!-- calkit value key=phases.lw_grouped_whole.local_bytes -->304<!-- /calkit value --> and <!-- calkit value key=phases.sw_grouped_whole.local_bytes -->360<!-- /calkit value --> in the grouped kernels they would replace.

What that buys is an occupancy step on each solver, by different routes.
Shortwave reaches the <!-- calkit value key=target_registers -->128<!-- /calkit value -->-register budget that the only step at <!-- calkit value key=block_size -->256<!-- /calkit value --> threads needs, so <!-- calkit value key=current_warps_per_sm -->8<!-- /calkit value --> warps per multiprocessor become <!-- calkit value key=phases.sw_gas_aerosol_optics.warps_per_sm -->16<!-- /calkit value -->.
Longwave at <!-- calkit value key=phases.lw_gas_aerosol_optics.registers -->156<!-- /calkit value --> is above that budget but below the <!-- calkit value key=alt_target_registers -->168<!-- /calkit value --> a <!-- calkit value key=alt_block_size -->128<!-- /calkit value -->-thread block needs for a third resident block, so it reaches <!-- calkit value key=phases.lw_gas_aerosol_optics.warps_per_sm_b128 -->12<!-- /calkit value --> warps instead of <!-- calkit value key=current_warps_per_sm -->8<!-- /calkit value -->.

The contrast with forcing is the whole result.
Forced to the same two-blocks-per-multiprocessor budget, the fused kernels also reach <!-- calkit value key=target_registers -->128<!-- /calkit value --> registers and <!-- calkit value key=target_warps_per_sm -->16<!-- /calkit value --> warps, and carry <!-- calkit value key=forced.lw_fused_maxregs_128.local_bytes -->672<!-- /calkit value --> and <!-- calkit value key=forced.sw_fused_maxregs_128.local_bytes -->712<!-- /calkit value --> bytes of local memory — about 21x what the unfused optics phase needs, and roughly double what those same kernels carry unforced.
`maxregs` and launch bounds produce identical figures again, so there is no allocator setting that reaches the budget cleanly: `forced_budget_is_clean` is <!-- calkit value key=forced_budget_is_clean -->false<!-- /calkit value -->.
That local memory is the quantity the two rejected attempts were blamed on, and the difference between those attempts and this one is reaching a budget rather than being squeezed into it.

The other phases are much smaller still — accumulation at <!-- calkit value key=phases.lw_accumulate.registers -->24<!-- /calkit value --> registers, cloud optics at <!-- calkit value key=phases.lw_cloud_optics.registers -->42<!-- /calkit value -->, and the two vertical sweeps at <!-- calkit value key=phases.lw_two_stream_sweep.registers -->51<!-- /calkit value --> and <!-- calkit value key=phases.sw_two_stream_sweep.registers -->53<!-- /calkit value -->, the sweeps with no local memory at all.
They are also the phases that carry almost none of the time, which is why they are not the answer on their own.

One criterion had to be changed to read this correctly, and the change is worth stating rather than burying.
Requiring a phase to show exactly zero local bytes reports both optics phases as failures at <!-- calkit value key=phases.lw_gas_aerosol_optics.local_bytes -->32<!-- /calkit value --> bytes, and `lw_optics_phase_fits_no_spill_at_all` and `sw_optics_phase_fits_no_spill_at_all` are both false for that reason.
Section 8t records why zero is the wrong bar: `CUDA.memory(k).local` covers the call frame and local arrays as well as spilled registers, so it bounds spill rather than measuring it.
The reference that means something is the kernel the phase would replace, and against that both phases spill an order of magnitude less, so `split_is_viable` is <!-- calkit value key=split_is_viable -->true<!-- /calkit value --> and `split_gains_a_step_b128` is <!-- calkit value key=split_gains_a_step_b128 -->true<!-- /calkit value -->.

For scale: the two solvers are <!-- calkit value key=device.solver_ms format="{:,}" -->4,333.8<!-- /calkit value --> ms of the <!-- calkit value key=device.device_ms format="{:,}" -->33,304.7<!-- /calkit value --> ms window, or <!-- calkit value key=device.solver_pct_of_device -->13.01<!-- /calkit value -->% of device, over <!-- calkit value key=grid.ncol format="{:,}" -->24,576<!-- /calkit value --> columns and <!-- calkit value key=grid.nlay -->80<!-- /calkit value --> layers at <!-- calkit value key=grid.ngroups_lw -->16<!-- /calkit value --> g-point groups.

## Verdict

### The split is viable, and for exactly the reason the forced route was not

Both attempts to raise these kernels' occupancy by setting a register budget lost, and this probe says why in one comparison: the fused kernel cannot hold its working set in <!-- calkit value key=target_registers -->128<!-- /calkit value --> registers, so a budget moves the overflow into local memory, while the optics phase on its own does not need the overflow at all.
The live state, not the register count, was always the binding quantity — a cap addresses the symptom and a split addresses the cause.
That the phase carrying essentially all of the time is also the one that unfuses cleanly is what makes it worth building, and it is the opposite of the microphysics quadrature split, where every piece retained one full CloudMicrophysics evaluation and so every piece stayed at the same occupancy.

### An occupancy threshold comes from the allocator's rules, not from division

The budget a phase must reach is not the register file divided by threads.
Registers are allocated per warp in 256-register units, a block's warps round up to the scheduler count, and the file is divided per scheduler rather than per multiprocessor.
Dividing instead puts the <!-- calkit value key=alt_block_size -->128<!-- /calkit value -->-thread step at 170 registers; the granularity-correct answer is <!-- calkit value key=alt_target_registers -->168<!-- /calkit value -->, which is also the figure `launch-bounds-vs-maxregs` recorded long before this experiment.
This document's ladder and the probe now both follow ClimaCore's `max_active_blocks`, itself a port of NVIDIA's occupancy calculator, because a hand-rolled ladder got three of its steps wrong and would have contradicted the probe inside one results file.

### A register count is a precondition, not a payoff

Nothing here is a time.
A split adds a kernel launch and has to write the optics output to scratch and read it back, and the phases that consume it carry almost none of the time, so the traffic is new cost against an occupancy gain.
What makes that trade plausible rather than wishful is the same reading that justified the SGS thread order: these kernels run at a small fraction of DRAM peak, so bandwidth is the resource with room and registers the one without.
What makes it uncertain is that the 99%-in-optics figure predates the g-point grouping, and that an occupancy step bought the wrong way has twice made these same kernels measurably slower.
The next measurement is the split built and timed, not compiled.

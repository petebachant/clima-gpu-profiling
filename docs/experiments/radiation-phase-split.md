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
At 256 threads the only step is 128 registers, because two blocks of 256 need 65536/(2*256); at 128 threads a third block fits at 168 registers -- not the 170 a division gives, because a warp's registers are allocated in units of 256 and three warps of 5,632 overrun a scheduler's 16,384 -- and at the cap both block sizes give the same 8 warps.
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

```sh calkit stage name=phases environment=clima outputs=[{path: results/experiments/radiation-phase-split/phases.toml, storage: git}] inputs=[scripts/measure-radiation-phase-registers.jl, scripts/run-julia-script.sh, RRTMGP.jl-mod/src, RRTMGP.jl-mod/ext, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=60]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-radiation-phase-registers.jl \
  --out results/experiments/radiation-phase-split/phases.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Building it, and what it measures

The grouped solve already writes its optics to device arrays: `op` and `src` in the group workspace are `(ncol * ngroups, nlay)` and both halves of a round address them by wide column.
So the two phases exchange nothing across a kernel boundary that was not already going to memory, and the split is three changes rather than a rewrite — the post-optics half of each per-column body becomes a named function, each band gets an optics kernel and a sweeps kernel, and the host walks the rounds of the g-point loop instead of the kernel walking them.
The cloud mask moves from the start of the body into the sweeps half, because the gas and aerosol optics never read it and the sweeps kernel is the half with registers to spare.
Both halves launch at <!-- calkit value key=split.optics_block_threads path=results/docs/radiation-phase-split.json -->128<!-- /calkit value --> threads rather than the solve's usual <!-- calkit value key=split.kernels.lw_fused.launch_block path=results/docs/radiation-phase-split.json -->256<!-- /calkit value -->, which is the only tuning in the change and is there because a block's warps round up to the scheduler count: at four warps per block these register counts reach a third and a fourth resident block where at eight they reach one.

Three claims are measured on the live simulation's own state, lookups and buffers, which makes all three in-block comparisons rather than differences of runs:

  * `ungrouped` is upstream's serial g-point loop, with no group workspace at all.
  * `fused` is the grouped solve as one kernel per round, which is what the PR ships today.
  * `split` is the grouped solve as optics then sweeps, two kernels per round.

`fused` and `split` share one group workspace and differ only in its `split_optics` flag, so they cannot differ in what they allocated or in what state they read.
The split moves no arithmetic, so the two are asserted bit-for-bit — fluxes, cloud cover and the 550 nm optical depth compared as integers — and the gate runs before the timed blocks, so a broken kernel costs two solves rather than a reservation.

It also answers a question this experiment raised about its own instrument.
The occupancy model that predicted the split was worth building is a model; the CUDA driver's `cuOccupancyMaxActiveBlocksPerMultiprocessor` is the authority.
Every kernel here is reported three ways — the register count `ptxas` chose, the driver's answer, and the model's — so `occupancy_model_agrees_everywhere` either confirms the ladder this document reasons on or names where it is wrong.

```sh calkit stage name=split environment=clima outputs=[{path: results/experiments/radiation-phase-split/split.toml, storage: git}] inputs=[scripts/measure-optics-split.jl, scripts/run-julia-script.sh, RRTMGP.jl-mod/src, RRTMGP.jl-mod/ext, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/config/atmos_configs/climaatmos_progedmf_1m.yml] scheduler={options: [--gpus=1, --time=120]}
#!/usr/bin/env bash
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/measure-optics-split.jl \
  --out results/experiments/radiation-phase-split/split.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=values environment=py outputs=[{path: results/docs/radiation-phase-split.json, storage: git}] inputs=[results/experiments/radiation-phase-split/phases.toml, results/experiments/radiation-phase-split/split.toml, results/top-kernels.csv]
import csv
import json
import tomllib

d = tomllib.load(open("results/experiments/radiation-phase-split/phases.toml", "rb"))
s = d["summary"]

# What the two solvers cost in the profiled window, and the device budget they
# sit in, taken from the profile's own rows so the denominator is the profile's.
with open("results/top-kernels.csv", newline="") as f:
    rows = list(csv.DictReader(f))


# One band's share, summed over every kernel it launches. A prefix pinned to
# one name is how this analysis would break on the next profile rather than on
# this one: the split retires `rte_*_2stream_solve_grouped` and launches an
# optics and a sweeps kernel in its place, so the row the lookup wanted stops
# existing and `SystemExit` arrives an nsys pair later.
def ms_and_pct(prefixes, arm):
    total_ms, total_pct = 0.0, 0.0
    for r in rows:
        if not any(r["kernel"].startswith(p) for p in prefixes):
            continue
        try:
            total_ms += float(r[f"total_ms_{arm}"])
            total_pct += float(r[f"pct_of_gpu_time_{arm}"])
        except ValueError:
            continue
    if total_pct == 0.0:
        raise SystemExit(f"no {prefixes} rows for the {arm} arm")
    return total_ms, total_pct


lw_ms, lw_pct = ms_and_pct(
    ("rte_lw_2stream_solve_grouped", "_lw_group_optics", "_lw_group_sweeps"),
    "mod",
)
sw_ms, sw_pct = ms_and_pct(
    ("rte_sw_2stream_solve_grouped", "_sw_group_optics", "_sw_group_sweeps"),
    "mod",
)
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


# ---------------------------------------------------------------------------
# What the split is worth, built and timed. Every figure here is an in-block
# comparison on one simulation's state, so none of it differences two runs.
# ---------------------------------------------------------------------------
sp = tomllib.load(open("results/experiments/radiation-phase-split/split.toml", "rb"))
wall = sp["wallclock"]


# Time goes as 1/warps if a kernel is purely latency-bound, so the occupancy
# step sets a floor on what it could have been worth. The fraction of that
# floor a band actually realized is the useful number: it says whether the
# saving came from the step or from something else.
def band(key, prefix):
    b = sp[key]
    fused_w = sp["at_launch"][f"{prefix}_fused"]["driver_warps_per_sm"]
    optics_w = sp["at_launch"][f"{prefix}_optics"]["driver_warps_per_sm"]
    sweeps_w = sp["at_launch"][f"{prefix}_sweeps"]["driver_warps_per_sm"]
    ideal_pct = 100.0 * (fused_w / optics_w - 1.0)
    return {
        "fused_warps_per_sm": fused_w,
        "optics_warps_per_sm": optics_w,
        "sweeps_warps_per_sm": sweeps_w,
        # The sweeps half is a second chance at a step and the two bands do not
        # both take it, which is where the realized fractions diverge.
        "sweeps_gains_a_step": sweeps_w > fused_w,
        "sweeps_warps_at_128": sp["kernels"][f"{prefix}_sweeps"][
            "driver_warps_per_sm_b128"
        ],
        "occupancy_ideal_pct": round(ideal_pct, 2),
        "realized_fraction_of_ideal": round(
            b["split_vs_fused_pct"] / ideal_pct, 3
        ),
        "ungrouped_ms": round(b["ungrouped"]["solve_ms"], 1),
        "fused_ms": round(b["fused"]["solve_ms"], 1),
        "split_ms": round(b["split"]["solve_ms"], 1),
        "split_vs_fused_pct": round(b["split_vs_fused_pct"], 2),
        "split_vs_ungrouped_pct": round(b["split_vs_ungrouped_pct"], 2),
        "fused_vs_ungrouped_pct": round(b["fused_vs_ungrouped_pct"], 2),
        "repeat_spread_pct": round(b["repeat_spread_pct"], 3),
        "saved_per_event_ms": round(b["split_saved_per_event_ms"], 1),
        "values_compared": b["values_compared"],
        "values_differing": b["values_differing"],
        "bit_for_bit": b["bit_for_bit"],
        # A difference inside the repeat spread is not resolved, whichever way
        # it points.
        "exceeds_repeat_spread": abs(b["split_vs_fused_pct"])
        > b["repeat_spread_pct"],
    }


# The next rung above where a kernel sits, read off the same ladder the rest of
# this file uses, so "what is left" is a figure rather than an impression.
def next_step(regs, block):
    here = warps(regs, block)
    for r in range(regs - 1, 31, -1):
        w = warps(r, block)
        if w > here:
            return {"registers": r, "warps_per_sm": w}
    return {"registers": 0, "warps_per_sm": here}


def kernel(name):
    k = sp["at_launch"][name]
    step = next_step(k["registers"], k["launch_block"])
    return {
        "launch_block": k["launch_block"],
        "registers": k["registers"],
        "local_bytes": k["local_bytes"],
        "driver_warps_per_sm": k["driver_warps_per_sm"],
        "model_warps_per_sm": k["model_warps_per_sm"],
        "model_agrees": k["model_agrees"],
        "next_step_registers": step["registers"],
        "next_step_warps_per_sm": step["warps_per_sm"],
    }


# One round of each half timed on its own, times the rounds. The two shares sum
# to more than the solve, which is what says this is an upper bound on each half
# rather than a decomposition: an isolated round pays a cold cache and takes the
# first round's zero-fill branch, and the solve also carries a reduction neither
# half is charged for.
def phase_share(key):
    t = sp["phase_times"][key]
    return {
        "rounds": t["rounds"],
        "optics_round_ms": round(t["optics_round_ms"], 2),
        "sweeps_round_ms": round(t["sweeps_round_ms"], 2),
        "optics_pct_of_split": round(t["optics_pct_of_split"], 1),
        "sweeps_pct_of_split": round(t["sweeps_pct_of_split"], 1),
        "accounted_pct_of_split": round(t["accounted_pct_of_split"], 1),
        "shares_overshoot": t["accounted_pct_of_split"] > 100.0,
    }


# Both bands' solves per radiation event, and what the split took off them.
# Scaling by the profile's solver share rather than by a step count keeps every
# number here measured; the share's denominator is the two grouped kernels, so
# this is the saving as a fraction of them and not of the reduction beside them.
per_event_fused = (
    sp["longwave"]["fused"]["solve_ms"] + sp["shortwave"]["fused"]["solve_ms"]
)
per_event_saved = (
    sp["longwave"]["split_saved_per_event_ms"]
    + sp["shortwave"]["split_saved_per_event_ms"]
)
saved_solver_pct = 100.0 * per_event_saved / per_event_fused
# The profiled arm is the split one, so its solver share of device is already
# net of the saving and scaling the saved fraction by it understates the result.
# Scale by the split solve's own cost instead, which recovers what the window
# would have spent fused and what the split took off it.
per_event_split = per_event_fused - per_event_saved
fused_solver_ms = solver_ms * per_event_fused / per_event_split
saved_window_ms = solver_ms * per_event_saved / per_event_split

split = {
    "ngroups": sp["ngroups"],
    "rounds_lw": sp["rounds_lw"],
    "rounds_sw": sp["rounds_sw"],
    "optics_block_threads": sp["optics_block_threads"],
    "sweeps_block_threads": sp["sweeps_block_threads"],
    "phase_times": {
        "longwave": phase_share("longwave"),
        "shortwave": phase_share("shortwave"),
    },
    "longwave": band("longwave", "lw"),
    "shortwave": band("shortwave", "sw"),
    "kernels": {n: kernel(n) for n in sorted(sp["at_launch"])},
    "bit_for_bit": sp["split_is_bit_for_bit"],
    "effects_exceed_repeat_spread": bool(
        band("longwave", "lw")["exceeds_repeat_spread"]
        and band("shortwave", "sw")["exceeds_repeat_spread"]
    ),
    "both_sweeps_gain_a_step": bool(
        band("longwave", "lw")["sweeps_gains_a_step"]
        and band("shortwave", "sw")["sweeps_gains_a_step"]
    ),
    "phase_shares_overshoot": bool(
        phase_share("longwave")["shares_overshoot"]
        and phase_share("shortwave")["shares_overshoot"]
    ),
    "wins_lw": sp["split_wins_lw"],
    "wins_sw": sp["split_wins_sw"],
    "occupancy_model_agrees_everywhere": sp["occupancy_model_agrees_everywhere"],
    "occupancy_model_checks": sp["occupancy_model_checks"],
    "saved_per_event_ms": round(per_event_saved, 1),
    "saved_solver_pct": round(saved_solver_pct, 2),
    "saved_window_ms": round(saved_window_ms, 1),
    "fused_solver_ms": round(fused_solver_ms, 1),
    "fused_solver_pct_of_device": round(100.0 * fused_solver_ms / device_ms, 2),
    "saved_device_pct": round(100.0 * saved_window_ms / device_ms, 3),
    "sypd_split": round(wall["sypd_split"], 4),
    "sypd_fused": round(wall["arms"]["fused"]["sypd"], 4),
    "sypd_ungrouped": round(wall["arms"]["ungrouped"]["sypd"], 4),
    "sypd_gain_over_fused_pct": round(
        wall["arms"]["fused"]["split_gain_pct"], 3
    ),
    "sypd_gain_over_ungrouped_pct": round(
        wall["arms"]["ungrouped"]["split_gain_pct"], 3
    ),
    "block_iqr_pct": round(wall["block_iqr_pct"], 3),
    "step_ms": round(wall["step_ms"], 2),
    "gain_over_fused_below_block_iqr": wall["gain_over_fused_below_block_iqr"],
    "blocks": wall["blocks"],
    "block_steps": wall["block_steps"],
}

out = {
    "split": split,
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

For scale: the two solvers are <!-- calkit value key=device.solver_ms format="{:,}" -->2,729.1<!-- /calkit value --> ms of the <!-- calkit value key=device.device_ms format="{:,}" -->31,442.4<!-- /calkit value --> ms window, or <!-- calkit value key=device.solver_pct_of_device -->8.68<!-- /calkit value -->% of device, over <!-- calkit value key=grid.ncol format="{:,}" -->24,576<!-- /calkit value --> columns and <!-- calkit value key=grid.nlay -->80<!-- /calkit value --> layers at <!-- calkit value key=grid.ngroups_lw -->16<!-- /calkit value --> g-point groups.

## What the split is worth, measured

Both bands win, and by margins no register count could have predicted, because a register count is not a time.
The longwave solve goes from <!-- calkit value key=split.longwave.fused_ms -->605.6<!-- /calkit value --> ms to <!-- calkit value key=split.longwave.split_ms -->454.8<!-- /calkit value --> ms, which is <!-- calkit value key=split.longwave.split_vs_fused_pct -->-24.9<!-- /calkit value -->%, and the shortwave from <!-- calkit value key=split.shortwave.fused_ms -->494.5<!-- /calkit value --> ms to <!-- calkit value key=split.shortwave.split_ms -->319.1<!-- /calkit value --> ms, or <!-- calkit value key=split.shortwave.split_vs_fused_pct -->-35.47<!-- /calkit value -->%.
Measured against upstream's serial g-point loop, which is what either grouped path is an improvement on, the same two solves are <!-- calkit value key=split.longwave.split_vs_ungrouped_pct -->-32.76<!-- /calkit value -->% and <!-- calkit value key=split.shortwave.split_vs_ungrouped_pct -->-48.2<!-- /calkit value -->%.
Each arm's own repeat spread is <!-- calkit value key=split.longwave.repeat_spread_pct -->0.054<!-- /calkit value -->% and <!-- calkit value key=split.shortwave.repeat_spread_pct -->0.08<!-- /calkit value -->%, so both effects are resolved more than two orders of magnitude over the noise of the instrument that found them, and `effects_exceed_repeat_spread` is <!-- calkit value key=split.effects_exceed_repeat_spread -->true<!-- /calkit value -->.

Scaled onto the profile's window, where the two grouped solves cost <!-- calkit value key=device.solver_ms format="{:,}" -->2,729.1<!-- /calkit value --> ms split, the same fraction is <!-- calkit value key=split.saved_window_ms format="{:,}" -->1,150.1<!-- /calkit value --> ms off a fused solver of <!-- calkit value key=split.fused_solver_ms format="{:,}" -->3,879.2<!-- /calkit value --> ms, or <!-- calkit value key=split.saved_device_pct -->3.658<!-- /calkit value -->% of the <!-- calkit value key=device.device_ms format="{:,}" -->31,442.4<!-- /calkit value --> ms device budget.
The scaling runs that way round deliberately: the profiled arm is the split one, so its solver's share of device is already net of the saving, and multiplying the saved fraction by it would price the split against itself.

Nothing moved but the kernel boundary, and that is asserted rather than argued.
All of <!-- calkit value key=split.longwave.values_compared format="{:,}" -->9,977,856<!-- /calkit value --> longwave values and <!-- calkit value key=split.shortwave.values_compared format="{:,}" -->11,993,088<!-- /calkit value --> shortwave ones — fluxes, clear-sky fluxes, cloud cover and the 550 nm aerosol optical depth — were compared between the two grouped arms as integers, and <!-- calkit value key=split.longwave.values_differing -->0<!-- /calkit value --> differ.

In wall clock that is <!-- calkit value key=split.sypd_gain_over_fused_pct -->4.004<!-- /calkit value -->% of SYPD over the fused grouped path and <!-- calkit value key=split.sypd_gain_over_ungrouped_pct -->6.366<!-- /calkit value -->% over the ungrouped one, taking a steady-state <!-- calkit value key=split.sypd_ungrouped -->1.1374<!-- /calkit value --> to <!-- calkit value key=split.sypd_split -->1.2098<!-- /calkit value -->.
The figure is <!-- calkit value key=split.blocks -->12<!-- /calkit value --> blocks of <!-- calkit value key=split.block_steps -->180<!-- /calkit value --> real coupler steps timed with the live split solver, with each other arm's in-block solve cost added back onto that same measured block rather than differenced against a second run.
Why it is built that way is visible in the number beside it: the blocks' own interquartile spread is <!-- calkit value key=split.block_iqr_pct -->4.965<!-- /calkit value -->%, so `gain_over_fused_below_block_iqr` is <!-- calkit value key=split.gain_over_fused_below_block_iqr -->true<!-- /calkit value --> and an AMIP pair would not have resolved this.

### The occupancy model was right, and the driver is what says so

Every kernel was reported three ways so the ladder this document reasons on could be checked rather than trusted: the register count `ptxas` chose, the CUDA driver's own `cuOccupancyMaxActiveBlocksPerMultiprocessor`, and the allocator model ClimaCore's `max_active_blocks` implements.
They agree on all <!-- calkit value key=split.occupancy_model_checks -->12<!-- /calkit value --> comparisons, six kernels at two block sizes each, so `occupancy_model_agrees_everywhere` is <!-- calkit value key=split.occupancy_model_agrees_everywhere -->true<!-- /calkit value -->.
The model is not being graded on the phase probe's predictions here but on the kernels that were actually built, which is the only version of the check worth having.

| kernel | threads | registers | local B | warps/SM |
|---|---|---|---|---|
| longwave fused, retired | <!-- calkit value key=split.kernels.lw_fused.launch_block -->256<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_fused.registers -->255<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_fused.local_bytes -->304<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_fused.driver_warps_per_sm -->8<!-- /calkit value --> |
| longwave optics | <!-- calkit value key=split.kernels.lw_optics.launch_block -->128<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_optics.registers -->156<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_optics.local_bytes -->32<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_optics.driver_warps_per_sm -->12<!-- /calkit value --> |
| longwave sweeps | <!-- calkit value key=split.kernels.lw_sweeps.launch_block -->128<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_sweeps.registers -->140<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_sweeps.local_bytes -->32<!-- /calkit value --> | <!-- calkit value key=split.kernels.lw_sweeps.driver_warps_per_sm -->12<!-- /calkit value --> |
| shortwave fused, retired | <!-- calkit value key=split.kernels.sw_fused.launch_block -->256<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_fused.registers -->255<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_fused.local_bytes -->360<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_fused.driver_warps_per_sm -->8<!-- /calkit value --> |
| shortwave optics | <!-- calkit value key=split.kernels.sw_optics.launch_block -->128<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_optics.registers -->124<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_optics.local_bytes -->32<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_optics.driver_warps_per_sm -->16<!-- /calkit value --> |
| shortwave sweeps | <!-- calkit value key=split.kernels.sw_sweeps.launch_block -->128<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_sweeps.registers -->117<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_sweeps.local_bytes -->32<!-- /calkit value --> | <!-- calkit value key=split.kernels.sw_sweeps.driver_warps_per_sm -->16<!-- /calkit value --> |

Two things in that table were not predicted and matter.
The sweeps half is not the <!-- calkit value key=phases.lw_two_stream_sweep.registers -->51<!-- /calkit value -->-register vertical sweep the probe compiled: it carries the cloud mask, both sweeps, the cloud increment between them and both accumulations, which is <!-- calkit value key=split.kernels.lw_sweeps.registers -->140<!-- /calkit value --> registers longwave and <!-- calkit value key=split.kernels.sw_sweeps.registers -->117<!-- /calkit value --> shortwave.
And the local memory the fused kernels carried — <!-- calkit value key=split.kernels.lw_fused.local_bytes -->304<!-- /calkit value --> and <!-- calkit value key=split.kernels.sw_fused.local_bytes -->360<!-- /calkit value --> bytes — is <!-- calkit value key=split.kernels.lw_optics.local_bytes -->32<!-- /calkit value --> bytes in every one of the four kernels that replace them.

### The saving tracks the occupancy step, in both bands and to the same degree

If a kernel were purely latency-bound its time would go as the reciprocal of its warps, so the step bounds what the split could be worth: <!-- calkit value key=split.longwave.occupancy_ideal_pct -->-33.33<!-- /calkit value -->% longwave at <!-- calkit value key=split.longwave.fused_warps_per_sm -->8<!-- /calkit value --> warps becoming <!-- calkit value key=split.longwave.optics_warps_per_sm -->12<!-- /calkit value -->, and <!-- calkit value key=split.shortwave.occupancy_ideal_pct -->-50.0<!-- /calkit value -->% shortwave at <!-- calkit value key=split.shortwave.fused_warps_per_sm -->8<!-- /calkit value --> becoming <!-- calkit value key=split.shortwave.optics_warps_per_sm -->16<!-- /calkit value -->.
The bands realized <!-- calkit value key=split.longwave.realized_fraction_of_ideal -->0.747<!-- /calkit value --> and <!-- calkit value key=split.shortwave.realized_fraction_of_ideal -->0.709<!-- /calkit value --> of those bounds, and it is the closeness of the two fractions that makes occupancy the mechanism rather than a coincidence of two separate rewrites.

They did not start out close, and how they were brought together is the useful part.
A first pass launched both sweeps halves at the solve's usual <!-- calkit value key=split.kernels.lw_fused.launch_block -->256<!-- /calkit value --> threads, where the longwave half's <!-- calkit value key=split.kernels.lw_sweeps.registers -->140<!-- /calkit value --> registers still bought it the <!-- calkit value key=split.longwave.fused_warps_per_sm -->8<!-- /calkit value --> warps of the kernel it replaced while the shortwave half's <!-- calkit value key=split.kernels.sw_sweeps.registers -->117<!-- /calkit value --> already bought <!-- calkit value key=split.shortwave.sweeps_warps_per_sm -->16<!-- /calkit value -->.
The longwave band then realized markedly less of its bound than the shortwave did — a puzzle from the timings alone, and an obvious one next to the per-kernel register readout.
The fix was one constant, and `both_sweeps_gain_a_step` is now <!-- calkit value key=split.both_sweeps_gain_a_step -->true<!-- /calkit value -->.

### Where the time is inside a split solve

One round of the longwave optics takes <!-- calkit value key=split.phase_times.longwave.optics_round_ms -->22.14<!-- /calkit value --> ms against <!-- calkit value key=split.phase_times.longwave.sweeps_round_ms -->7.75<!-- /calkit value --> ms for the sweeps, which over <!-- calkit value key=split.phase_times.longwave.rounds -->16<!-- /calkit value --> rounds is <!-- calkit value key=split.phase_times.longwave.optics_pct_of_split -->77.9<!-- /calkit value -->% and <!-- calkit value key=split.phase_times.longwave.sweeps_pct_of_split -->27.3<!-- /calkit value -->% of the solve; the shortwave shares are <!-- calkit value key=split.phase_times.shortwave.optics_pct_of_split -->77.4<!-- /calkit value -->% and <!-- calkit value key=split.phase_times.shortwave.sweeps_pct_of_split -->25.4<!-- /calkit value -->%.
Those sum past the solve, and `phase_shares_overshoot` is <!-- calkit value key=split.phase_shares_overshoot -->true<!-- /calkit value -->, so each is an upper bound on its half rather than a decomposition: a round timed in isolation pays a cold cache and takes the first round's zero-fill branch, and the solve carries a reduction neither half is charged for.
They are enough for the one thing they are needed for, which is that the optics is roughly three quarters of the solve and the sweeps are no longer negligible.

That is a different cut from section 4l's 99%-in-optics and 0.1%-in-sweeps, not a correction of it.
That measurement compared the optics function against the vertical sweep alone; this sweeps half also carries the cloud mask, the cloud optics increment and both accumulations, which is why it is <!-- calkit value key=split.kernels.lw_sweeps.registers -->140<!-- /calkit value --> registers rather than the <!-- calkit value key=phases.lw_two_stream_sweep.registers -->51<!-- /calkit value --> the probe measured for the sweep on its own.

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
That ladder is no longer only argued for: the driver's own occupancy call was asked about each of the six kernels the solve launches, at both block sizes, and agreed with the model every time.
A port of a calculator is still a model, so the check was worth making on the kernels that were built rather than on the phases that were compiled.

### Pricing the precondition before the payoff was the right order, and one of its cautions was wrong

The probe could say only that a budget was reachable.
It could not say the split was worth <!-- calkit value key=split.longwave.split_vs_fused_pct -->-24.9<!-- /calkit value -->% and <!-- calkit value key=split.shortwave.split_vs_fused_pct -->-35.47<!-- /calkit value -->%, and the sequence was still right: a phase that came back near the cap would have ended this line of work for the cost of one compile-only job, and the two earlier attempts to raise this family's occupancy each cost a full measurement to reject.

One caution in that verdict was wrong, and it was wrong about the code rather than about the device.
It said a split "has to write the optics output to scratch and read it back", so the traffic would be new cost against an occupancy gain.
The grouped solve was already doing exactly that: `op` and `src` in the group workspace are device arrays of `ncol * ngroups` columns, and the fused kernel wrote its optics there and read them back inside one thread's body.
So the kernel boundary carries nothing that was not already going to memory, and the split's real cost is <!-- calkit value key=split.rounds_lw -->16<!-- /calkit value --> extra launches per longwave solve and <!-- calkit value key=split.rounds_sw -->14<!-- /calkit value --> per shortwave one, against solves of hundreds of milliseconds.
Reading the workspace's own definition would have answered that before the sentence was written.

### What is left is named by this measurement rather than guessed

Every kernel in both solves now sits one rung above where the fused kernel sat, and the next rung is a register count away rather than a restructure away: the longwave optics would need <!-- calkit value key=split.kernels.lw_optics.next_step_registers -->128<!-- /calkit value --> registers for <!-- calkit value key=split.kernels.lw_optics.next_step_warps_per_sm -->16<!-- /calkit value --> warps where it compiles to <!-- calkit value key=split.kernels.lw_optics.registers -->156<!-- /calkit value -->, and the shortwave optics <!-- calkit value key=split.kernels.sw_optics.next_step_registers -->96<!-- /calkit value --> for <!-- calkit value key=split.kernels.sw_optics.next_step_warps_per_sm -->20<!-- /calkit value --> where it compiles to <!-- calkit value key=split.kernels.sw_optics.registers -->124<!-- /calkit value -->.
That is this experiment's own question asked of a smaller kernel, and it has the same two answers: squeeze the budget, which has lost twice here, or make the live state smaller.
The optics phase is still one kernel holding a gas lookup, a Planck source and an aerosol increment at once, and the Planck source is the piece whose output already goes to a device array.

The group count is an open question again rather than a settled one.
Sixteen groups were chosen because at one resident block per multiprocessor the grid needed some fourteen waves to absorb the imbalance between columns, and no kernel in this solve runs at one resident block any more.
Both the memory the workspaces cost and the count that is sufficient should be re-measured on the split rather than inherited from the fused sweep, and the two sweep stages that answer it need no new code to do so.

The headroom that remains is still large in absolute terms.
At <!-- calkit value key=split.longwave.optics_warps_per_sm -->12<!-- /calkit value --> and <!-- calkit value key=split.shortwave.optics_warps_per_sm -->16<!-- /calkit value --> warps of a possible 64 these kernels are still far from a full machine, and this family was the one that looked least movable in the whole device budget.

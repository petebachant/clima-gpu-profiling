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
import json
import tomllib

d = tomllib.load(open("results/experiments/radiation-phase-split/phases.toml", "rb"))
s = d["summary"]

# What the two solvers cost in the profiled window, and the device budget they
# sit in, taken from the profile's own rows so the denominator is the profile's.
rows = []
with open("results/top-kernels.csv") as f:
    header = f.readline().rstrip("\n").split(",")
    for line in f:
        rows.append(dict(zip(header, line.rstrip("\n").split(","))))


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
RF, THREADS_PER_SM, MAX_BLOCKS = 65536, 2048, 32


def warps(regs, block):
    blocks = min(RF // max(regs * block, 1), THREADS_PER_SM // block, MAX_BLOCKS)
    return max(blocks, 1) * -(-block // 32)


def ladder(block):
    out, seen = [], set()
    for regs in range(255, 31, -1):
        w = warps(regs, block)
        if w in seen:
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

To be written against the measurement.

## Verdict

To be written against the measurement.

# Radiation memory behavior

The two `rte_*_2stream_solve_*` kernels are the largest single block of device time after the microphysics cache, and they are the only hot kernels in this model that nobody has looked at with a memory profiler.
This experiment profiles them with `ncu --set full` in both arms and asks one question: of the traffic they move, how much is wasted, and on what.

## Why it is worth asking at all

The nsys inventory already says something that constrains the answer before any counter is read.
Both solvers compile to **255 registers per thread**, the hardware maximum, with a 256-thread block, and `launch__occupancy_limit_registers` is **1** -- one block per SM, eight resident warps, 12.5% theoretical occupancy.
That is the same ceiling the 1M microphysics kernel sat at before the CloudMicrophysics rematerialization work, and section 2 of `docs/learnings.md` has what it cost there.

Occupancy is not the finding, though; it is the reason a memory finding would matter.
With eight warps per SM a scheduler has almost nothing to switch to while a load is outstanding, so a request that fetches more sectors than its warp asked for is paid for at close to full latency instead of being overlapped.
The quantity that decides whether there is anything to win is therefore sectors per request, which `--set full` does not report directly -- it collects sectors and requests separately, and the analysis below divides them.

The baseline's grid is **96 blocks on 108 SMs**, which is its own small finding: one thread per column, 24,576 columns, and twelve SMs idle for the whole launch.
The mod arm's g-point split raises that to 1,536 blocks, so it already fixed the tail; it did not change the register count, which is still 255.

## What the source already rules out, and the prediction that follows

Written before the measurement, so the counters can contradict it.

The obvious cause is not available: upstream has already done the coalescing pass.
`Fluxes.jl` has a `_coalesced_2d`/`_coalesced_3d` pair whose whole purpose is this kernel shape -- on the GPU they allocate `(ncol, nlev)` so that one-thread-per-column access is contiguous, and on the CPU they allocate `(nlev, ncol)` behind a `PermutedDimsArray` so the indexing convention is shared.
`FluxLW`, `FluxSW`, the `TwoStream` optics (`τ`, `ssa`, `g`) and the longwave sources (`lev_source`, `albedo`, `src`) all go through it, and `TransposedStateCache` does the same for `layerdata` and `t_lev`.
The one struct that does not is `FluxBand`, which allocates `(nlay + 1, ncol, n_bnd)` raw and is read-modify-written per level inside the g-point loop by `accumulate_band_flux!` -- an 8x traffic multiplier if it ran, but `spectral_fluxes` is never set in ClimaAtmos, so `band_flux` is `nothing` here and the method specializes away.
It is a real latent bug for anyone who turns spectral fluxes on; it is not this model's traffic.

What is left is the gas-optics lookup.
Every lane in a warp is on the same g-point at the same time but a different column, and the `kmajor` lookup index depends on that column's own temperature and pressure interpolation, so the warp issues a gather into a multi-megabyte table rather than a contiguous read.
That cannot be fixed by transposing anything, and it predicts a specific pair of readings: **sectors per request well above 4 with an L2 hit rate that is high rather than low**, because neighboring columns have similar states and therefore hit each other's lines.

It also predicts something about the arms.
The mod arm's g-point split gives each block a sixteenth of the spectrum, so the table footprint a block touches shrinks by that factor, and the split's 9.6% device saving should show up here as **a higher L2 hit rate in the mod arm at the same sectors per request**.
If instead the mod arm's sectors per request fall, the split is doing something other than what this reasoning says, and the explanation above is wrong.

## How the prediction scored

The mechanism was predicted correctly and the statistic chosen to test it was the wrong instrument, and those are worth separating because only the second one is a lesson.

**The gather is the cause, as predicted.** Of every excessive sector in the longwave solver, <!-- calkit value key=baseline.headline.lookup_gather_pct_of_excess path=results/experiments/radiation-memory/stats.json -->77.9<!-- /calkit value -->% comes from a single group of <!-- calkit value key=baseline.headline.lookup_gather_instructions path=results/experiments/radiation-memory/stats.json -->52<!-- /calkit value --> four-byte loads running at <!-- calkit value key=baseline.headline.lookup_gather_amplification_x path=results/experiments/radiation-memory/stats.json -->8.0<!-- /calkit value -->x, and eight-fold amplification on a four-byte load is the exact signature of a gather: 32 lanes asking for 128 contiguous bytes would need four sectors, and these fetch thirty-two, one per lane.
Their SASS is `LDG.E Rn, [Rm.64+-0x4]` -- the address is a full 64-bit per-thread register rather than a base plus a thread index, which is what an index computed from the lane's own column state looks like after compilation.
The shortwave solver agrees to a tenth of a point on every group, which is what one algorithm on different spectral data should do and is the main reason to believe the split is real rather than an artifact of how the rows were bucketed.

**The traffic is cached, not streamed, also as predicted.** The L1 hit rate is at least <!-- calkit value key=baseline.headline.worst_l1_hit_pct path=results/experiments/radiation-memory/stats.json -->82.5992<!-- /calkit value -->% and the L2 hit rate at least <!-- calkit value key=baseline.headline.worst_l2_hit_pct path=results/experiments/radiation-memory/stats.json -->57.5059<!-- /calkit value -->%, so the table the warp gathers from is resident rather than being re-fetched from memory.
The prediction said "high rather than low" about L2 specifically, and <!-- calkit value key=baseline.headline.worst_l2_hit_pct path=results/experiments/radiation-memory/stats.json -->57.5059<!-- /calkit value -->% is middling rather than high; what is high is L1, which absorbs the gather one level earlier than the reasoning expected.
That matters for the fix, because it means the cost of the gather is not bytes moved from DRAM.

**So the headline figure needs qualifying rather than repeating.** DRAM throughput is only <!-- calkit value key=baseline.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->10.49<!-- /calkit value -->% of peak and the compute pipes are at <!-- calkit value key=baseline.headline.compute_pct_peak path=results/experiments/radiation-memory/stats.json -->21.78<!-- /calkit value -->%, so nothing in this kernel is bandwidth-saturated and "the memory traffic is insane" is not a bandwidth statement.
What is insane is the *sector count*: the solvers fetch <!-- calkit value key=baseline.headline.worst_amplification_x path=results/experiments/radiation-memory/stats.json -->2.34<!-- /calkit value -->x more sectors than the same bytes would need if they were packed, which is <!-- calkit value key=baseline.headline.worst_excessive_pct path=results/experiments/radiation-memory/stats.json -->57.2<!-- /calkit value -->% of all sector traffic thrown away.
Each uncoalesced load serializes into one L1 wavefront per sector, so the excess is load-store work and exposed latency rather than bandwidth, and memory dependency is indeed the largest stall reason measured, at <!-- calkit value key=baseline.headline.worst_long_scoreboard_share_of_measured_pct path=results/experiments/radiation-memory/stats.json -->60.3<!-- /calkit value -->% of the stall cycles in this counter set.
But the honest consequence of nothing being saturated is that removing the excess frees no bottlenecked resource, so the time saved will be *less* than the sector count falls -- a 10%-utilized L1 does not reward a 10% cut in wavefronts with 10% of the runtime.
The binding constraint is the one that was free to find: 255 registers, one block per SM, eight resident warps, and <!-- calkit value key=baseline.headline.grid_size path=results/experiments/radiation-memory/stats.json -->96<!-- /calkit value --> blocks on 108 multiprocessors, so there is nothing to switch to while a load is outstanding.
Coalescing is worth fixing where it is cheap, and it is a second-order lever behind occupancy, not an alternative to it.

**What cannot be concluded, and the attempt is recorded so it is not retried.** The obvious way to price the gather would be to attribute stall samples to the offending loads, and it does not work: a long-scoreboard stall is charged to the instruction that *consumes* the loaded register, not to the load that fetched it.
Doing it anyway puts 98.1% of the long-scoreboard samples on instructions with no memory access at all and 0.0% on the <!-- calkit value key=baseline.headline.lookup_gather_instructions path=results/experiments/radiation-memory/stats.json -->52<!-- /calkit value --> gather loads, which reads as though the gather were free and is purely an artifact of where the sample lands.
The sector counts are therefore the evidence for wasted work and the stall shares are the evidence that memory dependency is the limiter, but this profile does not contain a measurement of how much time the gather specifically costs.

**The instrument was wrong, and would have scored the prediction as a near-miss.** The prediction named sectors per request, which comes out at <!-- calkit value key=baseline.headline.worst_ld_sectors_per_request path=results/experiments/radiation-memory/stats.json -->5.38<!-- /calkit value --> against an ideal of 4 -- a 1.3x waste, small enough to read as a refutation.
It is diluted: <!-- calkit value key=baseline.headline.coalesced_instructions path=results/experiments/radiation-memory/stats.json -->404<!-- /calkit value --> instructions in the same kernel are perfectly coalesced and contribute the large majority of all requests, so they pull a warp-average toward 1.0 no matter how bad the minority is.
The per-instruction comparison ncu's own uncoalesced-access rule uses -- actual sectors against the sectors that instruction's own bytes would need -- is what resolves 1.3x into <!-- calkit value key=baseline.headline.worst_amplification_x path=results/experiments/radiation-memory/stats.json -->2.34<!-- /calkit value -->x overall and <!-- calkit value key=baseline.headline.lookup_gather_amplification_x path=results/experiments/radiation-memory/stats.json -->8.0<!-- /calkit value -->x in the group that matters.
A prediction has to name a statistic capable of separating the hypotheses, and an average over a population containing the answer and hundreds of non-answers is not one; `ld_waste_x_warp_average_diluted` is now named for what it is so the next reader does not quote it.

**One finding was not predicted, and contradicts something the section above asserts.** The section says transposition is ruled out because upstream has already done the coalescing pass, and that is true of every array it names.
It is not true of the three `Bool` masks: <!-- calkit value key=baseline.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->11.1<!-- /calkit value -->% of the excess comes from <!-- calkit value key=baseline.headline.mask_strided_instructions path=results/experiments/radiation-memory/stats.json -->14<!-- /calkit value --> one-byte accesses at <!-- calkit value key=baseline.headline.mask_strided_amplification_x path=results/experiments/radiation-memory/stats.json -->32.0<!-- /calkit value -->x, which is a perfectly strided access, not a gather.
Thirty-two lanes reading one byte each would need a single sector and these need thirty-two, because `cld_mask_lw`, `cld_mask_sw` and `aero_mask` are allocated `(nlay, ncol)` and read through `view(mask, :, gcol)`, so consecutive lanes sit `nlay` bytes apart.
The reason the coalescing pass missed them is that they are not allocated in RRTMGP at all -- they are allocated in `ClimaAtmos.jl/src/parameterized_tendencies/radiation/RRTMGPInterface.jl` as `DA{Bool}(undef, nlay, ncol)` and passed in, so `_coalesced_2d` never saw them and reading only RRTMGP could not have found it.

## What is profiled

Four launches per arm, filtered by kernel name so ncu replays nothing else.
The filter is the bare substring `2stream_solve` rather than an anchored pattern: it is unique to these two kernels in the whole inventory, it matches the baseline's `both_skies` and the mod arm's `grouped` names alike, and a filter that matches nothing costs a reservation and still reports success.
The stages write under `results/experiments/radiation-memory/` rather than `results/ncu/`, so the shared ncu reports -- which are frozen and describe the microphysics kernels at a different revision -- cannot be overwritten by this experiment or overwrite it.

```sh calkit stage name=baseline-ncu environment=clima outputs=[results/experiments/radiation-memory/baseline.ncu-rep, {path: results/experiments/radiation-memory/baseline-details.csv, storage: git}, {path: results/experiments/radiation-memory/baseline-raw.csv, storage: git}] inputs=[scripts/run-ncu.sh, scripts/run.jl, ClimaCoupler.jl/src, ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl/experiments/AMIP/Manifest-v1.11.toml, ClimaCore.jl/src, ClimaCore.jl/ext, ClimaAtmos.jl/src, RRTMGP.jl/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
set -euo pipefail
P=results/experiments/radiation-memory/baseline
bash scripts/run-ncu.sh "$P" \
  --kernel-name "regex:2stream_solve" \
  --project ClimaCoupler.jl/experiments/AMIP \
  --config ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml \
  --launch-count 4
# The raw page as well as the details page: sectors and requests appear only
# under their metric IDs, and coalescing is the ratio between them.
NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
"$NCU" --csv -i "$P.ncu-rep" --page raw --log-file "$P-raw.csv"
```

```sh calkit stage name=mod-ncu environment=clima outputs=[results/experiments/radiation-memory/mod.ncu-rep, {path: results/experiments/radiation-memory/mod-details.csv, storage: git}, {path: results/experiments/radiation-memory/mod-raw.csv, storage: git}] inputs=[scripts/run-ncu.sh, scripts/run.jl, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.11.toml, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, RRTMGP.jl-mod/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
set -euo pipefail
P=results/experiments/radiation-memory/mod
bash scripts/run-ncu.sh "$P" \
  --kernel-name "regex:2stream_solve" \
  --project ClimaCoupler.jl-mod/experiments/AMIP \
  --config ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml \
  --launch-count 4
NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
"$NCU" --csv -i "$P.ncu-rep" --page raw --log-file "$P-raw.csv"
```

## Analysis

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/radiation-memory/stats.json, storage: git}] inputs=[scripts/measure_radiation_memory.py, results/experiments/radiation-memory/baseline-raw.csv, results/experiments/radiation-memory/baseline-details.csv, results/experiments/radiation-memory/baseline.ncu-rep, results/experiments/radiation-memory/mod-raw.csv, results/experiments/radiation-memory/mod-details.csv, results/experiments/radiation-memory/mod.ncu-rep]
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, "scripts")
from measure_radiation_memory import arm

D = "results/experiments/radiation-memory"
NCU = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
# The source page is exported here rather than in the profiling stages because
# it is a read of a report file, not a measurement: it needs no GPU and no
# model run, so deriving it late costs nothing and keeps the two ncu stages
# from going stale over an analysis change. It stays an intermediate -- it is
# tens of megabytes of per-instruction rows, and what survives into Git is the
# aggregate in `stats.json`.
tmp = tempfile.mkdtemp()
out = {}
for a in ("baseline", "mod"):
    src = os.path.join(tmp, f"{a}-source.csv")
    with open(src, "w") as f:
        subprocess.run([NCU, "--csv", "-i", f"{D}/{a}.ncu-rep", "--page", "source"],
                       stdout=f, stderr=subprocess.DEVNULL, check=True)
    out[a] = arm(f"{D}/{a}-raw.csv", f"{D}/{a}-details.csv", src)
    assert not out[a]["missing_metrics"], (
        f"{a}: ncu did not report {out[a]['missing_metrics']}"
    )
    assert out[a]["kernels"], f"{a}: the kernel filter matched nothing"
    assert all("coalescing" in k for k in out[a]["kernels"].values()), (
        f"{a}: the source page did not attach to every kernel"
    )

# Headline per arm: the longwave and shortwave solvers, summed, since they are
# the same code on different spectral data and a reader cares about radiation.
for a, data in out.items():
    ks = data["kernels"]
    lw = next(k for k in ks if "_lw_" in k)
    sw = next(k for k in ks if "_sw_" in k)
    both = [ks[lw], ks[sw]]
    data["headline"] = {
        "lw_kernel": lw,
        "sw_kernel": sw,
        "ms_per_call": round(sum(k["duration"] for k in both), 2),
        "registers": max(k["registers"] for k in both),
        "achieved_occupancy_pct": round(
            sum(k["achieved_occupancy_pct"] for k in both) / 2, 2
        ),
        "dram_gb_per_call": round(
            sum(k["dram_read"] + k["dram_write"] for k in both), 3
        ),
        # The two solvers should agree closely on these, being one algorithm;
        # the worse of the pair is the one a fix has to clear.
        "worst_ld_sectors_per_request": max(
            k["derived"]["global_ld_sectors_per_request"] for k in both
        ),
        "worst_local_share_pct": max(
            k["derived"]["local_share_pct"] for k in both
        ),
        "worst_spill_overhead_pct": max(
            k.get("spill_overhead_pct", 0) for k in both
        ),
        "worst_long_scoreboard_stall": max(
            k["stall_long_scoreboard"] for k in both
        ),
        # The per-instruction figures. `amplification_x` is sectors fetched per
        # sector the same bytes would need packed, so 1.0 is perfect; the two
        # group shares say which access pattern the excess is in.
        "worst_amplification_x": max(
            k["coalescing"]["amplification_x"] for k in both
        ),
        "worst_excessive_pct": max(
            k["coalescing"]["excessive_pct"] for k in both
        ),
        "lookup_gather_pct_of_excess": max(
            k["coalescing"]["groups"]["lookup_gather"]["pct_of_excess"]
            for k in both
        ),
        "lookup_gather_amplification_x": max(
            k["coalescing"]["groups"]["lookup_gather"]["amplification_x"]
            for k in both
        ),
        "mask_strided_pct_of_excess": max(
            k["coalescing"]["groups"]["mask_byte_strided"]["pct_of_excess"]
            for k in both
        ),
        "mask_strided_amplification_x": max(
            k["coalescing"]["groups"]["mask_byte_strided"]["amplification_x"]
            for k in both
        ),
        "partially_coalesced_pct_of_excess": max(
            k["coalescing"]["groups"]["partially_coalesced"]["pct_of_excess"]
            for k in both
        ),
        # Instruction counts, because the shares above are concentrated in very
        # few instructions and that is what makes a fix tractable.
        "lookup_gather_instructions": max(
            k["coalescing"]["groups"]["lookup_gather"]["instructions"]
            for k in both
        ),
        "mask_strided_instructions": max(
            k["coalescing"]["groups"]["mask_byte_strided"]["instructions"]
            for k in both
        ),
        "coalesced_instructions": max(
            k["coalescing"]["groups"]["coalesced"]["instructions"] for k in both
        ),
        # The cache and bandwidth readings the gather hypothesis predicted.
        "worst_l1_hit_pct": min(k["l1_hit_pct"] for k in both),
        "worst_l2_hit_pct": min(k["l2_hit_pct"] for k in both),
        "dram_pct_peak": round(max(k["dram_pct_peak"] for k in both), 2),
        "compute_pct_peak": round(max(k["compute_pct_peak"] for k in both), 2),
        "grid_size": max(k["grid_size"] for k in both),
        "worst_long_scoreboard_share_of_measured_pct": max(
            k["derived"]["stall_long_scoreboard_share_of_measured_pct"]
            for k in both
        ),
    }

json.dump(out, open(f"{D}/stats.json", "w"), indent=2, sort_keys=True)
h = out["baseline"]["headline"]
print(f"baseline radiation: {h['ms_per_call']} ms/call, {h['registers']} regs, "
      f"{h['achieved_occupancy_pct']}% occupancy, {h['dram_gb_per_call']} GB/call")
print(f"  worst sectors/request {h['worst_ld_sectors_per_request']}, "
      f"local share {h['worst_local_share_pct']}%, "
      f"spill overhead {h['worst_spill_overhead_pct']}%")
print(f"  {h['worst_amplification_x']}x sectors vs ideal "
      f"({h['worst_excessive_pct']}% excessive): "
      f"{h['lookup_gather_pct_of_excess']}% of it lookup gather at "
      f"{h['lookup_gather_amplification_x']}x, "
      f"{h['mask_strided_pct_of_excess']}% byte masks at "
      f"{h['mask_strided_amplification_x']}x")
```

<!-- calkit values path=results/experiments/radiation-memory/stats.json -->

## Result

Filled from `stats.json` once both arms have run.

## Verdict

Pending.

# Radiation memory behavior

The two `rte_*_2stream_solve_*` kernels are the largest single block of device time after the microphysics cache, and they are the only hot kernels in this model that nobody has looked at with a memory profiler.
This experiment profiles them with `ncu --set full` in both arms and asks one question: of the traffic they move, how much is wasted, and on what.

## Why it is worth asking at all

The nsys inventory already says something that constrains the answer before any counter is read.
The baseline's two solvers compile to **255 registers per thread**, the hardware maximum, with a 256-thread block, and `launch__occupancy_limit_registers` is **1** -- one block per SM, eight resident warps, 12.5% theoretical occupancy.
That is the same ceiling the 1M microphysics kernel sat at before the CloudMicrophysics rematerialization work, and section 2 of `docs/learnings.md` has what it cost there.

Occupancy is not the finding, though; it is the reason a memory finding would matter.
With the baseline's eight warps per SM a scheduler has almost nothing to switch to while a load is outstanding, so a request that fetches more sectors than its warp asked for is paid for at close to full latency instead of being overlapped.
The quantity that decides whether there is anything to win is therefore sectors per request, which `--set full` does not report directly -- it collects sectors and requests separately, and the analysis below divides them.

The baseline's grid is **96 blocks on 108 SMs**, which is its own small finding: one thread per column, 24,576 columns, and twelve SMs idle for the whole launch.
The mod arm's g-point split and phase split raise that to <!-- calkit value key=mod.headline.grid_size path=results/experiments/radiation-memory/stats.json -->3072.0<!-- /calkit value --> blocks of 128 threads, <!-- calkit value key=comparison.grid_size_factor path=results/experiments/radiation-memory/stats.json -->32.0<!-- /calkit value -->x as many, so the tail is already collected.
They moved the register count as well, which is why this experiment was re-run against them rather than carried over: the four kernels the mod arm launches instead of two compile to between <!-- calkit value key=mod.headline.registers_min path=results/experiments/radiation-memory/stats.json -->117.0<!-- /calkit value --> and <!-- calkit value key=mod.headline.registers path=results/experiments/radiation-memory/stats.json -->156.0<!-- /calkit value --> registers, for <!-- calkit value key=mod.steps.sw_sweeps_warps path=results/experiments/radiation-memory/stats.json -->16<!-- /calkit value --> and <!-- calkit value key=mod.steps.lw_optics_warps path=results/experiments/radiation-memory/stats.json -->12<!-- /calkit value --> resident warps rather than eight.
So the ceiling this section opens with belongs to the baseline, and the question of whether a memory finding matters has to be asked again on the other side of it -- which is what the Result section does.

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

**The traffic is cached, not streamed, also as predicted.** The L1 hit rate is at least <!-- calkit value key=baseline.headline.worst_l1_hit_pct path=results/experiments/radiation-memory/stats.json -->83.6906<!-- /calkit value -->% and the L2 hit rate at least <!-- calkit value key=baseline.headline.worst_l2_hit_pct path=results/experiments/radiation-memory/stats.json -->58.7244<!-- /calkit value -->%, so the table the warp gathers from is resident rather than being re-fetched from memory.
The prediction said "high rather than low" about L2 specifically, and <!-- calkit value key=baseline.headline.worst_l2_hit_pct path=results/experiments/radiation-memory/stats.json -->58.7244<!-- /calkit value -->% is middling rather than high; what is high is L1, which absorbs the gather one level earlier than the reasoning expected.
That matters for the fix, because it means the cost of the gather is not bytes moved from DRAM.

**So the headline figure needs qualifying rather than repeating.** DRAM throughput is only <!-- calkit value key=baseline.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->10.4<!-- /calkit value -->% of peak and the compute pipes are at <!-- calkit value key=baseline.headline.compute_pct_peak path=results/experiments/radiation-memory/stats.json -->20.22<!-- /calkit value -->%, so nothing in this kernel is bandwidth-saturated and "the memory traffic is insane" is not a bandwidth statement.
What is insane is the *sector count*: the solvers fetch <!-- calkit value key=baseline.headline.worst_amplification_x path=results/experiments/radiation-memory/stats.json -->2.52<!-- /calkit value -->x more sectors than the same bytes would need if they were packed, which is <!-- calkit value key=baseline.headline.worst_excessive_pct path=results/experiments/radiation-memory/stats.json -->60.3<!-- /calkit value -->% of all sector traffic thrown away.
Each uncoalesced load serializes into one L1 wavefront per sector, so the excess is load-store work and exposed latency rather than bandwidth, and memory dependency is indeed the largest stall reason measured, at <!-- calkit value key=baseline.headline.worst_long_scoreboard_share_of_measured_pct path=results/experiments/radiation-memory/stats.json -->59.2<!-- /calkit value -->% of the stall cycles in this counter set.
But the honest consequence of nothing being saturated is that removing the excess frees no bottlenecked resource, so the time saved will be *less* than the sector count falls -- a 10%-utilized L1 does not reward a 10% cut in wavefronts with 10% of the runtime.
The baseline's binding constraint is the one that was free to find: 255 registers, one block per SM, eight resident warps, and <!-- calkit value key=baseline.headline.grid_size path=results/experiments/radiation-memory/stats.json -->96.0<!-- /calkit value --> blocks on 108 multiprocessors, so there is nothing to switch to while a load is outstanding.
Coalescing is worth fixing where it is cheap, and it is a second-order lever behind occupancy, not an alternative to it.
That ranking is the one the mod arm tests, because it raised occupancy by <!-- calkit value key=comparison.occupancy_pts_delta path=results/experiments/radiation-memory/stats.json -->8.49<!-- /calkit value --> points without touching the coalescing, and the Result section reports what happened to the stalls.

**What cannot be concluded, and the attempt is recorded so it is not retried.** The obvious way to price the gather would be to attribute stall samples to the offending loads, and it does not work: a long-scoreboard stall is charged to the instruction that *consumes* the loaded register, not to the load that fetched it.
Doing it anyway puts 98.1% of the long-scoreboard samples on instructions with no memory access at all and 0.0% on the <!-- calkit value key=baseline.headline.lookup_gather_instructions path=results/experiments/radiation-memory/stats.json -->52<!-- /calkit value --> gather loads, which reads as though the gather were free and is purely an artifact of where the sample lands.
The sector counts are therefore the evidence for wasted work and the stall shares are the evidence that memory dependency is the limiter, but this profile does not contain a measurement of how much time the gather specifically costs.

**The instrument was wrong, and would have scored the prediction as a near-miss.** The prediction named sectors per request, which comes out at <!-- calkit value key=baseline.headline.worst_ld_sectors_per_request path=results/experiments/radiation-memory/stats.json -->5.44<!-- /calkit value --> against an ideal of 4 -- a 1.3x waste, small enough to read as a refutation.
It is diluted: <!-- calkit value key=baseline.headline.coalesced_instructions path=results/experiments/radiation-memory/stats.json -->452<!-- /calkit value --> instructions in the same kernel are perfectly coalesced and contribute the large majority of all requests, so they pull a warp-average toward 1.0 no matter how bad the minority is.
The per-instruction comparison ncu's own uncoalesced-access rule uses -- actual sectors against the sectors that instruction's own bytes would need -- is what resolves 1.3x into <!-- calkit value key=baseline.headline.worst_amplification_x path=results/experiments/radiation-memory/stats.json -->2.52<!-- /calkit value -->x overall and <!-- calkit value key=baseline.headline.lookup_gather_amplification_x path=results/experiments/radiation-memory/stats.json -->8.0<!-- /calkit value -->x in the group that matters.
A prediction has to name a statistic capable of separating the hypotheses, and an average over a population containing the answer and hundreds of non-answers is not one; `ld_waste_x_warp_average_diluted` is now named for what it is so the next reader does not quote it.

**One finding was not predicted, and contradicts something the section above asserts.** The section says transposition is ruled out because upstream has already done the coalescing pass, and that is true of every array it names.
It is not true of the three `Bool` masks: <!-- calkit value key=baseline.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->11.1<!-- /calkit value -->% of the excess comes from <!-- calkit value key=baseline.headline.mask_strided_instructions path=results/experiments/radiation-memory/stats.json -->14<!-- /calkit value --> one-byte accesses at <!-- calkit value key=baseline.headline.mask_strided_amplification_x path=results/experiments/radiation-memory/stats.json -->32.0<!-- /calkit value -->x, which is a perfectly strided access, not a gather.
Thirty-two lanes reading one byte each would need a single sector and these need thirty-two, because `cld_mask_lw`, `cld_mask_sw` and `aero_mask` are allocated `(nlay, ncol)` and read through `view(mask, :, gcol)`, so consecutive lanes sit `nlay` bytes apart.
In the split arm that group stops being a footnote: the sweeps kernels read the mask and little else that is strided, so `mask_byte_strided` reaches <!-- calkit value key=mod.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->75.5<!-- /calkit value -->% of a kernel's excess there against <!-- calkit value key=baseline.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->11.1<!-- /calkit value -->% in the fused solve, from <!-- calkit value key=mod.headline.mask_strided_instructions path=results/experiments/radiation-memory/stats.json -->80<!-- /calkit value --> instructions rather than <!-- calkit value key=baseline.headline.mask_strided_instructions path=results/experiments/radiation-memory/stats.json -->14<!-- /calkit value -->.
Its share of the whole solve's sector traffic barely moves -- <!-- calkit value key=baseline.headline.mask_byte_strided_pct_of_all_sectors path=results/experiments/radiation-memory/stats.json -->6.5<!-- /calkit value -->% against <!-- calkit value key=mod.headline.mask_byte_strided_pct_of_all_sectors path=results/experiments/radiation-memory/stats.json -->6.49<!-- /calkit value -->% -- so what changed is which kernel pays it, not how much there is.
The reason the coalescing pass missed them is that they are not allocated in RRTMGP at all -- they are allocated in `ClimaAtmos.jl/src/parameterized_tendencies/radiation/RRTMGPInterface.jl` as `DA{Bool}(undef, nlay, ncol)` and passed in, so `_coalesced_2d` never saw them and reading only RRTMGP could not have found it.

## What is profiled

Two radiation events in the baseline and one in the mod arm, filtered by kernel name so ncu replays nothing else.
The filter names two families rather than one substring, because the arms do not run the same kernels: the baseline solves a band in a single `rte_*_2stream_solve_both_skies` launch, while the grouped arm runs `cld(n_gpt, ngroups)` rounds of a `_*_group_optics` kernel and then a `_*_group_sweeps` kernel -- 16 rounds in the longwave and 14 in the shortwave, so one event is 2 matching launches on one side and 60 on the other.
`--launch-count` is set to a whole number of events per arm rather than to a fixed number of launches, because a count that stops mid-event profiles one band and never reaches the other, which is a half-measurement that reads as a complete one.
The reduction and the hoisted aerosol mask are outside the filter deliberately: they are not the solve, and the per-band totals in `results/gpoint-groups.json` these timings are checked against do not include them either.
Each stage greps its own details page before exiting, because a filter that matches nothing costs a reservation and still reports success -- and a kernel rename is what causes that, which this experiment has now seen happen.
The stages write under `results/experiments/radiation-memory/` rather than `results/ncu/`, so the shared ncu reports -- which are frozen and describe the microphysics kernels at a different revision -- cannot be overwritten by this experiment or overwrite it.

```sh calkit stage name=baseline-ncu environment=clima outputs=[results/experiments/radiation-memory/baseline.ncu-rep, {path: results/experiments/radiation-memory/baseline-details.csv, storage: git}, {path: results/experiments/radiation-memory/baseline-raw.csv, storage: git}] inputs=[scripts/run-ncu.sh, scripts/run.jl, ClimaCoupler.jl/src, ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl/experiments/AMIP/Manifest-v1.13.toml, ClimaCore.jl/src, ClimaCore.jl/ext, ClimaAtmos.jl/src, RRTMGP.jl/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
set -euo pipefail
P=results/experiments/radiation-memory/baseline
# Two events, each one longwave and one shortwave launch.
bash scripts/run-ncu.sh "$P" \
  --kernel-name "regex:(2stream_solve|_group_optics|_group_sweeps)" \
  --project ClimaCoupler.jl/experiments/AMIP \
  --config ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml \
  --launch-count 4
# Both bands, checked here rather than in the analysis so that a filter which
# matched nothing fails while the reservation still means something.
for band in _lw_ _sw_; do
  grep -q "$band" "$P-details.csv" ||
    { echo "ERROR: no $band kernel in $P-details.csv" >&2; exit 1; }
done
# The raw page as well as the details page: sectors and requests appear only
# under their metric IDs, and coalescing is the ratio between them.
NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
"$NCU" --csv -i "$P.ncu-rep" --page raw --log-file "$P-raw.csv"
```

```sh calkit stage name=mod-ncu environment=clima outputs=[results/experiments/radiation-memory/mod.ncu-rep, {path: results/experiments/radiation-memory/mod-details.csv, storage: git}, {path: results/experiments/radiation-memory/mod-raw.csv, storage: git}] inputs=[scripts/run-ncu.sh, scripts/run.jl, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, RRTMGP.jl-mod/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
set -euo pipefail
P=results/experiments/radiation-memory/mod
# One event: 16 longwave rounds and 14 shortwave, each a pair of launches.
bash scripts/run-ncu.sh "$P" \
  --kernel-name "regex:(2stream_solve|_group_optics|_group_sweeps)" \
  --project ClimaCoupler.jl-mod/experiments/AMIP \
  --config ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml \
  --launch-count 60
# Both bands, checked here rather than in the analysis so that a filter which
# matched nothing fails while the reservation still means something.
for band in _lw_ _sw_; do
  grep -q "$band" "$P-details.csv" ||
    { echo "ERROR: no $band kernel in $P-details.csv" >&2; exit 1; }
done
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

# Launches per arm as whole radiation events, because the baseline does in one
# launch what the grouped arm does in `rounds` pairs of launches, so a per-call
# figure has to mean per event on both sides or the arms are not comparable.
EVENTS = {"baseline": 2, "mod": 1}


def pick(both, get, how=max):
    """The worst reading over a solve's kernels, skipping the ones without it.

    A phase kernel need not touch every access pattern -- the grouped optics
    never reads the cloud mask, by design -- so a group missing from one kernel
    is an absence and not an error. Indexing it directly raises, and does so
    after the reservation is already spent.
    """
    vals = [v for v in (get(k) for k in both) if v is not None]
    return how(vals) if vals else None


def group(name, field):
    return lambda k: k["coalescing"]["groups"].get(name, {}).get(field)


# Headline per arm: every kernel a band's solve launches, since a reader cares
# about radiation rather than about one phase of it.
for a, data in out.items():
    ks = data["kernels"]
    lw = sorted(k for k in ks if "_lw_" in k)
    sw = sorted(k for k in ks if "_sw_" in k)
    assert lw and sw, f"{a}: the filter matched {sorted(ks)}, missing a band"
    names = lw + sw
    both = [ks[k] for k in names]
    # Both of a band's kernels fire once per round, so unequal counts within one
    # band mean the profile stopped mid-event, and every per-event sum below
    # would then be weighted by wherever it stopped.
    for band, bnames in (("longwave", lw), ("shortwave", sw)):
        counts = {n: ks[n]["launches_profiled"] for n in bnames}
        assert len(set(counts.values())) == 1, (
            f"{a}: the {band} kernels were profiled unequally, {counts}, so the "
            "launch count is not a whole number of events"
        )
    events = EVENTS[a]
    # `duration` is the mean over the launches profiled, so a kernel's share of
    # one event is that mean times the number of times it fires in one.
    weight = {k: ks[k]["duration"] * ks[k]["launches_profiled"] for k in names}
    data["headline"] = {
        "lw_kernels": lw,
        "sw_kernels": sw,
        "events_profiled": events,
        "ms_per_call": round(sum(weight.values()) / events, 2),
        "registers": max(k["registers"] for k in both),
        "registers_by_kernel": {k: ks[k]["registers"] for k in names},
        # Weighted by each kernel's share of the event: an unweighted mean would
        # give a sweeps launch the same standing as the optics launch that costs
        # three times as much.
        "achieved_occupancy_pct": round(
            sum(ks[k]["achieved_occupancy_pct"] * weight[k] for k in names)
            / sum(weight.values()),
            2,
        ),
        "dram_gb_per_call": round(
            sum(
                (ks[k]["dram_read"] + ks[k]["dram_write"])
                * ks[k]["launches_profiled"]
                for k in names
            )
            / events,
            3,
        ),
        # The solvers should agree closely on these, being one algorithm; the
        # worse of them is the one a fix has to clear.
        "worst_ld_sectors_per_request": pick(
            both, lambda k: k["derived"]["global_ld_sectors_per_request"]
        ),
        "worst_local_share_pct": pick(
            both, lambda k: k["derived"]["local_share_pct"]
        ),
        "worst_spill_overhead_pct": pick(
            both, lambda k: k.get("spill_overhead_pct", 0)
        ),
        "worst_long_scoreboard_stall": pick(
            both, lambda k: k["stall_long_scoreboard"]
        ),
        # The per-instruction figures. `amplification_x` is sectors fetched per
        # sector the same bytes would need packed, so 1.0 is perfect; the group
        # shares say which access pattern the excess is in.
        "worst_amplification_x": pick(
            both, lambda k: k["coalescing"]["amplification_x"]
        ),
        "worst_excessive_pct": pick(
            both, lambda k: k["coalescing"]["excessive_pct"]
        ),
        "lookup_gather_pct_of_excess": pick(
            both, group("lookup_gather", "pct_of_excess")
        ),
        "lookup_gather_amplification_x": pick(
            both, group("lookup_gather", "amplification_x")
        ),
        "mask_strided_pct_of_excess": pick(
            both, group("mask_byte_strided", "pct_of_excess")
        ),
        "mask_strided_amplification_x": pick(
            both, group("mask_byte_strided", "amplification_x")
        ),
        "partially_coalesced_pct_of_excess": pick(
            both, group("partially_coalesced", "pct_of_excess")
        ),
        # Instruction counts, because the shares above are concentrated in very
        # few instructions and that is what makes a fix tractable.
        "lookup_gather_instructions": pick(
            both, group("lookup_gather", "instructions")
        ),
        "mask_strided_instructions": pick(
            both, group("mask_byte_strided", "instructions")
        ),
        "coalesced_instructions": pick(both, group("coalesced", "instructions")),
        # Which kernels carry a pattern at all, so a share above reads as the
        # worst of the solve rather than as the whole of it.
        "kernels_with_mask_strided": [
            k
            for k in names
            if "mask_byte_strided" in ks[k]["coalescing"]["groups"]
        ],
        # The cache and bandwidth readings the gather hypothesis predicted.
        "worst_l1_hit_pct": pick(both, lambda k: k["l1_hit_pct"], min),
        "worst_l2_hit_pct": pick(both, lambda k: k["l2_hit_pct"], min),
        "dram_pct_peak": round(pick(both, lambda k: k["dram_pct_peak"]), 2),
        # Both ends, because the split arm's phases are no longer alike: the
        # optics kernels absorb their traffic in L1 and the sweeps kernels do
        # not, and a single worst-of reading hides which is which.
        "best_l1_hit_pct": pick(both, lambda k: k["l1_hit_pct"], max),
        "dram_pct_peak_min": round(
            pick(both, lambda k: k["dram_pct_peak"], min), 2
        ),
        "compute_pct_peak": round(
            pick(both, lambda k: k["compute_pct_peak"]), 2
        ),
        "grid_size": pick(both, lambda k: k["grid_size"]),
        "worst_long_scoreboard_share_of_measured_pct": pick(
            both,
            lambda k: k["derived"][
                "stall_long_scoreboard_share_of_measured_pct"
            ],
        ),
        # Achieved against theoretical. A shortfall near zero means the kernel
        # already collects every warp its register count permits, so occupancy
        # is a register question and not a launch-configuration one.
        "worst_occupancy_shortfall_pts": pick(
            both, lambda k: k["derived"]["occupancy_shortfall_pts"]
        ),
        "registers_min": min(k["registers"] for k in both),
        # Shares of the whole solve's sector traffic rather than of one
        # kernel's excess, because a fix is ranked by what it removes from the
        # event while the per-kernel shares above are maxima over the kernels.
        **{
            f"{g}_pct_of_all_sectors": round(
                100.0
                * sum(
                    ks[k]["coalescing"]["groups"].get(g, {}).get("excessive", 0)
                    for k in names
                )
                / sum(
                    v["sectors"]
                    for k in names
                    for v in ks[k]["coalescing"]["groups"].values()
                ),
                2,
            )
            for g in (
                "lookup_gather",
                "mask_byte_strided",
                "partially_coalesced",
            )
        },
    }

    # What a register count buys on this device, so that "N registers off this
    # kernel" is computed here rather than typed into the verdict. An A100 has
    # 65,536 registers per multiprocessor, allocates them per warp in units of
    # 256, and admits whole blocks only -- so the step depends on the block
    # size as well as on the count, and the steps are not evenly spaced.
    #
    # The model is checked against ncu rather than trusted: achieved occupancy
    # plus the shortfall to theoretical is the theoretical occupancy, and that
    # is what `warps_per_sm` has to reproduce for every kernel in both arms.
    REGS_PER_SM, ALLOC_UNIT = 65536, 256

    def warps_per_sm(registers, block_threads):
        per_warp = -(-int(registers) * 32 // ALLOC_UNIT) * ALLOC_UNIT
        per_block = int(block_threads) // 32
        return min((REGS_PER_SM // (per_warp * per_block)) * per_block, 64)

    def next_step(registers, block_threads):
        now = warps_per_sm(registers, block_threads)
        better = [
            r
            for r in range(1, int(registers))
            if warps_per_sm(r, block_threads) > now
        ]
        return max(better) if better else None

    steps = {}
    for name in names:
        k = ks[name]
        short = (
            name.replace("rte_", "")
            .replace("_2stream_solve_both_skies", "")
            .replace("_group", "")
            .strip("_")
        )
        w = warps_per_sm(k["registers"], k["block_size"])
        theoretical = (
            k["achieved_occupancy_pct"] + k["derived"]["occupancy_shortfall_pts"]
        )
        assert abs(100.0 * w / 64 - theoretical) < 0.05, (
            f"{a}/{name}: the step model says {w} warps "
            f"({100.0 * w / 64:.2f}%) but ncu reports a theoretical "
            f"{theoretical:.2f}%, so the model is wrong for this launch shape"
        )
        nxt = next_step(k["registers"], k["block_size"])
        steps[f"{short}_registers"] = k["registers"]
        steps[f"{short}_warps"] = w
        steps[f"{short}_time_pct"] = round(
            100.0 * weight[name] / sum(weight.values()), 1
        )
        if nxt is not None:
            steps[f"{short}_next_registers"] = nxt
            steps[f"{short}_registers_to_cut"] = int(k["registers"]) - nxt
            steps[f"{short}_next_warps"] = warps_per_sm(nxt, k["block_size"])
    data["steps"] = steps

# The arms compared. The prediction this document committed before the data was
# that the g-point split would show up here as better cache behavior, so the
# test of it is derived rather than read off: `prediction_l2_hit_rose` and
# `prediction_sectors_per_request_fell` are the two branches the prose named.
b, m = out["baseline"]["headline"], out["mod"]["headline"]


def rules(arm):
    """Every rule this arm's radiation kernels raise, at its worst estimate.

    The union over the kernels rather than one of them: the grouped arm's
    optics and sweeps phases raise different rules, so a rule counts as retired
    only when no kernel of that arm's solve raises it.
    """
    ks = out[arm]["kernels"]
    worst = {}
    for name in (k for k in ks if "_lw_" in k or "_sw_" in k):
        for r in ks[name]["rules"]:
            v = r.get("estimated_speedup_pct")
            if v is not None:
                worst[r["rule"]] = max(worst.get(r["rule"], v), v)
    return worst


base_rules, mod_rules = rules("baseline"), rules("mod")
dropped = sorted(set(base_rules) - set(mod_rules))
out["comparison"] = {
    "ms_per_call_change_pct": round(
        100.0 * (m["ms_per_call"] - b["ms_per_call"]) / b["ms_per_call"], 2
    ),
    "grid_size_factor": round(m["grid_size"] / b["grid_size"], 1),
    # Every memory figure, differenced. The prediction says these move; the
    # alternative explanation says they do not.
    "l2_hit_pts_delta": round(m["worst_l2_hit_pct"] - b["worst_l2_hit_pct"], 3),
    "l1_hit_pts_delta": round(m["worst_l1_hit_pct"] - b["worst_l1_hit_pct"], 3),
    "amplification_delta_x": round(
        m["worst_amplification_x"] - b["worst_amplification_x"], 3
    ),
    "sectors_per_request_delta": round(
        m["worst_ld_sectors_per_request"] - b["worst_ld_sectors_per_request"], 3
    ),
    "occupancy_pts_delta": round(
        m["achieved_occupancy_pct"] - b["achieved_occupancy_pct"], 3
    ),
    "registers_delta": m["registers"] - b["registers"],
    "prediction_l2_hit_rose": m["worst_l2_hit_pct"] > b["worst_l2_hit_pct"] + 1.0,
    "prediction_sectors_per_request_fell": (
        m["worst_ld_sectors_per_request"] < b["worst_ld_sectors_per_request"] - 0.1
    ),
    # The rules the split retired, and what ncu estimated they were worth. If
    # these are the grid-shape rules, the saving is machine fill, not memory.
    "rules_dropped_in_mod": dropped,
    "rules_dropped_estimated_speedup_pct": round(
        sum(base_rules[r] for r in dropped), 2
    ),
    "memory_rules_unchanged": {
        r: [base_rules[r], mod_rules[r]]
        for r in ("UncoalescedGlobalAccess", "MemoryCacheAccessPattern",
                  "TheoreticalOccupancy")
        if r in base_rules and r in mod_rules
    },
    # The same three as flat scalars, because a value marker reads a scalar and
    # the prose quotes them one at a time.
    **{
        f"rule_{r}_{arm}_pct": round(src[r], 2)
        for r in ("UncoalescedGlobalAccess", "MemoryCacheAccessPattern",
                  "TheoreticalOccupancy", "WorkloadImbalance",
                  "IssueSlotUtilization")
        for arm, src in (("baseline", base_rules), ("mod", mod_rules))
        if r in src
    },
    "long_scoreboard_stall_delta_cycles": round(
        m["worst_long_scoreboard_stall"] - b["worst_long_scoreboard_stall"], 2
    ),
    "long_scoreboard_share_pts_delta": round(
        m["worst_long_scoreboard_share_of_measured_pct"]
        - b["worst_long_scoreboard_share_of_measured_pct"],
        2,
    ),
    "occupancy_shortfall_pts_both_small": max(
        b["worst_occupancy_shortfall_pts"], m["worst_occupancy_shortfall_pts"]
    ) < 3.0,
}

json.dump(out, open(f"{D}/stats.json", "w"), indent=2, sort_keys=True)
c = out["comparison"]
print(f"mod vs baseline: {c['ms_per_call_change_pct']}% ms/call at "
      f"{c['grid_size_factor']}x the blocks; L2 hit {c['l2_hit_pts_delta']:+} pts, "
      f"amplification {c['amplification_delta_x']:+}x, "
      f"registers {c['registers_delta']:+}")
print(f"  rules retired: {', '.join(c['rules_dropped_in_mod']) or 'none'} "
      f"(ncu estimated {c['rules_dropped_estimated_speedup_pct']}%)")
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

The baseline readings are in the section above; what the arms say about the prediction is here.

**The arm half of the prediction is refuted, and not in either of the two ways the prediction allowed for.** It said the g-point split should appear as a higher L2 hit rate at the same sectors per request, and named the falsifying outcome as sectors per request falling instead.
Neither happened: the L2 hit rate moved <!-- calkit value key=comparison.l2_hit_pts_delta path=results/experiments/radiation-memory/stats.json -->-1.81<!-- /calkit value --> points, L1 moved <!-- calkit value key=comparison.l1_hit_pts_delta path=results/experiments/radiation-memory/stats.json -->-33.604<!-- /calkit value --> points, sectors per request moved <!-- calkit value key=comparison.sectors_per_request_delta path=results/experiments/radiation-memory/stats.json -->2.15<!-- /calkit value -->, and the per-instruction amplification moved <!-- calkit value key=comparison.amplification_delta_x path=results/experiments/radiation-memory/stats.json -->0.08<!-- /calkit value -->x.
`prediction_l2_hit_rose` is <!-- calkit value key=comparison.prediction_l2_hit_rose path=results/experiments/radiation-memory/stats.json -->false<!-- /calkit value --> and `prediction_sectors_per_request_fell` is <!-- calkit value key=comparison.prediction_sectors_per_request_fell path=results/experiments/radiation-memory/stats.json -->false<!-- /calkit value -->.
Both cache figures moved the wrong way and L1 moved a long way the wrong way, so the split did not improve locality; it spent some.

**The split traded register pressure for memory traffic.** The mod arm no longer runs one kernel per band: a g-point group's optics are computed in `_*_group_optics` and consumed by `_*_group_sweeps`, so the optics arrays the fused kernel held in registers are now written to memory and read back.
Registers fell from <!-- calkit value key=baseline.headline.registers path=results/experiments/radiation-memory/stats.json -->255.0<!-- /calkit value --> to between <!-- calkit value key=mod.headline.registers_min path=results/experiments/radiation-memory/stats.json -->117.0<!-- /calkit value --> and <!-- calkit value key=mod.headline.registers path=results/experiments/radiation-memory/stats.json -->156.0<!-- /calkit value -->, a change of <!-- calkit value key=comparison.registers_delta path=results/experiments/radiation-memory/stats.json -->-99.0<!-- /calkit value -->, and achieved occupancy rose <!-- calkit value key=comparison.occupancy_pts_delta path=results/experiments/radiation-memory/stats.json -->8.49<!-- /calkit value --> points; against that, the worst L1 hit rate falls to <!-- calkit value key=mod.headline.worst_l1_hit_pct path=results/experiments/radiation-memory/stats.json -->50.0864<!-- /calkit value -->% and DRAM throughput rises from <!-- calkit value key=baseline.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->10.4<!-- /calkit value -->% of peak to <!-- calkit value key=mod.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->28.99<!-- /calkit value -->%.
Both of those land on the two sweeps kernels, and they are the kernels ncu now reports at <!-- calkit value key=mod.headline.worst_long_scoreboard_share_of_measured_pct path=results/experiments/radiation-memory/stats.json -->80.4<!-- /calkit value -->% of measured stall cycles on memory dependency against <!-- calkit value key=baseline.headline.worst_long_scoreboard_share_of_measured_pct path=results/experiments/radiation-memory/stats.json -->59.2<!-- /calkit value -->% in the fused solve, and at <!-- calkit value key=mod.headline.worst_long_scoreboard_stall path=results/experiments/radiation-memory/stats.json -->10.1959<!-- /calkit value --> cycles per issued instruction against <!-- calkit value key=baseline.headline.worst_long_scoreboard_stall path=results/experiments/radiation-memory/stats.json -->4.2303<!-- /calkit value --> -- a move of <!-- calkit value key=comparison.long_scoreboard_share_pts_delta path=results/experiments/radiation-memory/stats.json -->21.2<!-- /calkit value --> points and <!-- calkit value key=comparison.long_scoreboard_stall_delta_cycles path=results/experiments/radiation-memory/stats.json -->5.97<!-- /calkit value --> cycles.
So the memory problem did not dissolve under more occupancy.
More occupancy exposed more of it, and that is the opposite of what the reasoning in "Why it is worth asking at all" expected: it argued that eight warps per SM is what makes wasted sectors expensive, which implied that more warps would make them cheaper.

**What the split did buy is machine fill and the end of spilling.** The grid went from <!-- calkit value key=baseline.headline.grid_size path=results/experiments/radiation-memory/stats.json -->96.0<!-- /calkit value --> blocks to <!-- calkit value key=comparison.grid_size_factor path=results/experiments/radiation-memory/stats.json -->32.0<!-- /calkit value -->x as many, and `HighPipeUtilization` and `LaunchConfiguration` retired together -- <!-- calkit value key=comparison.rules_dropped_estimated_speedup_pct path=results/experiments/radiation-memory/stats.json -->97.17<!-- /calkit value -->% by ncu's estimate, and those are the two rules about a grid too small to fill 108 multiprocessors and pipes left idle by it.
Spilling went with them: the fused solve ran at <!-- calkit value key=baseline.headline.worst_spill_overhead_pct path=results/experiments/radiation-memory/stats.json -->100.0<!-- /calkit value -->% spill overhead with <!-- calkit value key=baseline.headline.worst_local_share_pct path=results/experiments/radiation-memory/stats.json -->0.63<!-- /calkit value -->% of its sectors in local memory, and every kernel of the split arm reports <!-- calkit value key=mod.headline.worst_spill_overhead_pct path=results/experiments/radiation-memory/stats.json -->0.0<!-- /calkit value -->% and <!-- calkit value key=mod.headline.worst_local_share_pct path=results/experiments/radiation-memory/stats.json -->0.0<!-- /calkit value -->%.
That is the `Base.tail` unroll in `_zero_planes!` holding, and it is why the register counts landed where they did rather than back at the cap.

**Every memory rule survives, which is the half of the prediction that was about the mechanism rather than about the arms.** `UncoalescedGlobalAccess` is <!-- calkit value key=comparison.rule_UncoalescedGlobalAccess_baseline_pct path=results/experiments/radiation-memory/stats.json -->59.45<!-- /calkit value -->% in the baseline and <!-- calkit value key=comparison.rule_UncoalescedGlobalAccess_mod_pct path=results/experiments/radiation-memory/stats.json -->60.42<!-- /calkit value -->% in the mod arm; `TheoreticalOccupancy` <!-- calkit value key=comparison.rule_TheoreticalOccupancy_baseline_pct path=results/experiments/radiation-memory/stats.json -->77.07<!-- /calkit value -->% and <!-- calkit value key=comparison.rule_TheoreticalOccupancy_mod_pct path=results/experiments/radiation-memory/stats.json -->73.9<!-- /calkit value -->%; `MemoryCacheAccessPattern` <!-- calkit value key=comparison.rule_MemoryCacheAccessPattern_baseline_pct path=results/experiments/radiation-memory/stats.json -->15.41<!-- /calkit value -->% and <!-- calkit value key=comparison.rule_MemoryCacheAccessPattern_mod_pct path=results/experiments/radiation-memory/stats.json -->29.13<!-- /calkit value -->%.
The last is the one that moved, and it moved the wrong way, which is the paragraph above restated by ncu's own rule set.
`WorkloadImbalance` is halved rather than retired, <!-- calkit value key=comparison.rule_WorkloadImbalance_baseline_pct path=results/experiments/radiation-memory/stats.json -->17.18<!-- /calkit value -->% to <!-- calkit value key=comparison.rule_WorkloadImbalance_mod_pct path=results/experiments/radiation-memory/stats.json -->8.81<!-- /calkit value -->%, and it survives in the shortwave kernels only -- the night columns exiting at once while their warp waits, which `results/night-warps.toml` has the census of.

**That `TheoreticalOccupancy` is still the largest global rule in both arms is the finding of this re-measurement.** It survives a treatment that moved the register count by <!-- calkit value key=comparison.registers_delta path=results/experiments/radiation-memory/stats.json -->-99.0<!-- /calkit value --> and raised achieved occupancy <!-- calkit value key=comparison.occupancy_pts_delta path=results/experiments/radiation-memory/stats.json -->8.49<!-- /calkit value --> points, and the reason it survives is visible in one number per kernel: achieved occupancy is within <!-- calkit value key=mod.headline.worst_occupancy_shortfall_pts path=results/experiments/radiation-memory/stats.json -->2.25<!-- /calkit value --> points of theoretical in the worst of the split arm's kernels and <!-- calkit value key=baseline.headline.worst_occupancy_shortfall_pts path=results/experiments/radiation-memory/stats.json -->0.37<!-- /calkit value --> in the baseline's.
`occupancy_shortfall_pts_both_small` is <!-- calkit value key=comparison.occupancy_shortfall_pts_both_small path=results/experiments/radiation-memory/stats.json -->true<!-- /calkit value -->, so in both arms every kernel already collects essentially every warp its own register count permits.
Nothing is being lost to launch shape, which the block-size sweep in `results/group-block-size.toml` independently closes; occupancy here is a register question and only a register question.

**The sector excess is not spent.** <!-- calkit value key=baseline.headline.worst_excessive_pct path=results/experiments/radiation-memory/stats.json -->60.3<!-- /calkit value -->% of the baseline's sectors are excessive against <!-- calkit value key=mod.headline.worst_excessive_pct path=results/experiments/radiation-memory/stats.json -->61.5<!-- /calkit value -->% of the mod arm's, and the gather's share of all sector traffic is <!-- calkit value key=baseline.headline.lookup_gather_pct_of_all_sectors path=results/experiments/radiation-memory/stats.json -->45.56<!-- /calkit value -->% against <!-- calkit value key=mod.headline.lookup_gather_pct_of_all_sectors path=results/experiments/radiation-memory/stats.json -->45.59<!-- /calkit value -->% -- the same figure to within a twentieth of a point across a treatment that moved every other number in this document.
Whatever the excess is worth is therefore still additive to the split rather than overlapping with it.

**The per-call time difference is an ncu figure and is not a speedup.** `ms_per_call` moves <!-- calkit value key=comparison.ms_per_call_change_pct path=results/experiments/radiation-memory/stats.json -->-38.94<!-- /calkit value -->%, several times the device-time and SYPD gains the unprofiled AMIP stages measure for the same treatment and which `docs/wall-clock-conversion.md` carries.
`--set full` replays each launch many times and flushes the caches between passes, which costs a spilling 255-register kernel far more than it costs a 156-register one; the baseline reports <!-- calkit value key=baseline.headline.ms_per_call path=results/experiments/radiation-memory/stats.json -->1455.23<!-- /calkit value --> ms per call here against single-digit milliseconds for the same solve in nsys.
So nothing in this document is a wall-clock claim.
What survives the replay is the shape of the traffic and the stalls, which is what it is for, and `ms_per_call` is kept only to weight the per-kernel averages within an arm.

## Verdict

Four things, in the order the readings support rather than the order of their size.
The register arithmetic below is computed in the analysis above and checked against ncu's own theoretical occupancy for every kernel in both arms, so the steps are device facts rather than a rule of thumb.

**1. <!-- calkit value key=mod.steps.lw_sweeps_registers_to_cut path=results/experiments/radiation-memory/stats.json -->12<!-- /calkit value --> registers off `_lw_group_sweeps`, the cheapest step available.** It compiles to <!-- calkit value key=mod.steps.lw_sweeps_registers path=results/experiments/radiation-memory/stats.json -->140.0<!-- /calkit value --> registers, which at a 128-thread block buys <!-- calkit value key=mod.steps.lw_sweeps_warps path=results/experiments/radiation-memory/stats.json -->12<!-- /calkit value --> warps per multiprocessor; the next step up is <!-- calkit value key=mod.steps.lw_sweeps_next_registers path=results/experiments/radiation-memory/stats.json -->128<!-- /calkit value --> registers for <!-- calkit value key=mod.steps.lw_sweeps_next_warps path=results/experiments/radiation-memory/stats.json -->16<!-- /calkit value --> warps, a third more resident warps for twelve registers.
`_sw_group_sweeps` is the same shape: <!-- calkit value key=mod.steps.sw_sweeps_registers_to_cut path=results/experiments/radiation-memory/stats.json -->21<!-- /calkit value --> registers from <!-- calkit value key=mod.steps.sw_sweeps_next_warps path=results/experiments/radiation-memory/stats.json -->20<!-- /calkit value --> warps.
Those two kernels are only <!-- calkit value key=mod.steps.lw_sweeps_time_pct path=results/experiments/radiation-memory/stats.json -->14.6<!-- /calkit value -->% and <!-- calkit value key=mod.steps.sw_sweeps_time_pct path=results/experiments/radiation-memory/stats.json -->9.6<!-- /calkit value -->% of the event, so the ceiling is modest -- but they are the kernels at <!-- calkit value key=mod.headline.worst_long_scoreboard_share_of_measured_pct path=results/experiments/radiation-memory/stats.json -->80.4<!-- /calkit value -->% of stall cycles on memory dependency, which is the condition more resident warps relieves, and this is the smallest register ask anywhere in the radiation path.

**2. The `Bool` masks, now rather than eventually.** The group was <!-- calkit value key=baseline.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->11.1<!-- /calkit value -->% of the fused solve's excess and reaches <!-- calkit value key=mod.headline.mask_strided_pct_of_excess path=results/experiments/radiation-memory/stats.json -->75.5<!-- /calkit value -->% of a split kernel's, from <!-- calkit value key=mod.headline.mask_strided_instructions path=results/experiments/radiation-memory/stats.json -->80<!-- /calkit value --> one-byte accesses at <!-- calkit value key=mod.headline.mask_strided_amplification_x path=results/experiments/radiation-memory/stats.json -->32.0<!-- /calkit value -->x.
It concentrates in the two sweeps kernels, which are also the only kernels here whose DRAM throughput is not negligible -- <!-- calkit value key=mod.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->28.99<!-- /calkit value -->% of peak against <!-- calkit value key=mod.headline.dram_pct_peak_min path=results/experiments/radiation-memory/stats.json -->8.87<!-- /calkit value -->% in the optics kernels and <!-- calkit value key=baseline.headline.dram_pct_peak path=results/experiments/radiation-memory/stats.json -->10.4<!-- /calkit value -->% in the fused solve -- so it is the one coalescing fix that lands where bandwidth is actually being consumed rather than being absorbed by L1.
The cause is unchanged: `cld_mask_lw`, `cld_mask_sw` and `aero_mask` are allocated `(nlay, ncol)` in `ClimaAtmos.jl/src/parameterized_tendencies/radiation/RRTMGPInterface.jl`, `gpt_groups.jl` allocates the grouped arm's own `mask` the same way over wide columns, and all of them are read through `view(mask, :, gcol)`, so consecutive lanes sit `nlay` bytes apart.
The fix is the `_coalesced_2d` treatment the flux buffers already have, applied where the coalescing pass could not see it because the allocation lives in the other repository.

**3. <!-- calkit value key=mod.steps.lw_optics_registers_to_cut path=results/experiments/radiation-memory/stats.json -->28<!-- /calkit value --> registers off `_lw_group_optics`, the largest occupancy prize and the largest ask.** It is <!-- calkit value key=mod.steps.lw_optics_time_pct path=results/experiments/radiation-memory/stats.json -->44.0<!-- /calkit value -->% of the event on its own and sits at <!-- calkit value key=mod.steps.lw_optics_registers path=results/experiments/radiation-memory/stats.json -->156.0<!-- /calkit value --> registers, <!-- calkit value key=mod.steps.lw_optics_registers_to_cut path=results/experiments/radiation-memory/stats.json -->28<!-- /calkit value --> above the <!-- calkit value key=mod.steps.lw_optics_next_registers path=results/experiments/radiation-memory/stats.json -->128<!-- /calkit value -->-register step that would take it from <!-- calkit value key=mod.steps.lw_optics_warps path=results/experiments/radiation-memory/stats.json -->12<!-- /calkit value --> to <!-- calkit value key=mod.steps.lw_optics_next_warps path=results/experiments/radiation-memory/stats.json -->16<!-- /calkit value --> warps.
`_sw_group_optics` is <!-- calkit value key=mod.steps.sw_optics_registers_to_cut path=results/experiments/radiation-memory/stats.json -->28<!-- /calkit value --> registers from <!-- calkit value key=mod.steps.sw_optics_next_warps path=results/experiments/radiation-memory/stats.json -->20<!-- /calkit value --> warps at <!-- calkit value key=mod.steps.sw_optics_time_pct path=results/experiments/radiation-memory/stats.json -->31.8<!-- /calkit value -->% of the event.
`scripts/measure-radiation-phase-registers.jl` is written and has never been run: the phase split already moved the fused solve's register count by <!-- calkit value key=comparison.registers_delta path=results/experiments/radiation-memory/stats.json -->-99.0<!-- /calkit value -->, and whether a further split can take <!-- calkit value key=mod.steps.lw_optics_registers_to_cut path=results/experiments/radiation-memory/stats.json -->28<!-- /calkit value --> more out of the optics phase is what that script decides.

**4. The gather still needs the bin spread measured before anything is permuted.** It is <!-- calkit value key=mod.headline.lookup_gather_pct_of_all_sectors path=results/experiments/radiation-memory/stats.json -->45.59<!-- /calkit value -->% of every sector the radiation path moves, by far the largest single quantity in this report, and it sits in the two optics kernels that are <!-- calkit value key=mod.steps.lw_optics_time_pct path=results/experiments/radiation-memory/stats.json -->44.0<!-- /calkit value -->% and <!-- calkit value key=mod.steps.sw_optics_time_pct path=results/experiments/radiation-memory/stats.json -->31.8<!-- /calkit value -->% of the event.
Those are also the kernels absorbing it: their L1 hit rate reaches <!-- calkit value key=mod.headline.best_l1_hit_pct path=results/experiments/radiation-memory/stats.json -->86.3467<!-- /calkit value -->% and their DRAM throughput is as low as <!-- calkit value key=mod.headline.dram_pct_peak_min path=results/experiments/radiation-memory/stats.json -->8.87<!-- /calkit value -->% of peak, so the gather costs wavefronts and exposed latency rather than bandwidth, and removing it frees no saturated resource.
`kmajor` is `(n_η, n_p_ref + 1, n_t_ref, n_gpt)` and `interp3d` reads `coeff[jη, jpresst, jtemp]`, so a lane's neighbors are 4 bytes away in η, `n_η` x 4 bytes away in pressure and `n_η` x `(n_p_ref + 1)` x 4 bytes away in temperature -- only the η axis is inside a sector, and the other two both split one.
Permuting the table so temperature joins η at the fast end helps only to the extent that a warp's 32 columns actually cluster in temperature and pressure bins, and that is measured nowhere in this profile.
It is a cheap thing to measure and an expensive thing to guess: the permutation touches the table load, every `interp*` reader and the published artifact layout, so the histogram of per-warp bin spread comes first.

Merging the two sweeps kernels, which is an open review suggestion on the g-point split, is not on this list and should not ride along with anything on it.
The sweeps kernels are where the `Base.tail` unroll repaired the spill that the phase split introduced, so a merge moves the register count of exactly the kernels items 1 and 3 are about, and measuring the two treatments in one pass would not separate them.

<!-- /calkit values -->

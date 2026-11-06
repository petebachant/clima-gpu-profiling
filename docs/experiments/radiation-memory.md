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

## What is profiled

Four launches per arm, which is every radiation call in the 120-step window, filtered by kernel name so ncu replays nothing else.
The stages write under `results/experiments/radiation-memory/` rather than `results/ncu/`, so the shared ncu reports -- which are frozen and describe the microphysics kernels at a different revision -- cannot be overwritten by this experiment or overwrite it.

```sh calkit stage name=baseline-ncu environment=clima outputs=[results/experiments/radiation-memory/baseline.ncu-rep, {path: results/experiments/radiation-memory/baseline-details.csv, storage: git}, {path: results/experiments/radiation-memory/baseline-raw.csv, storage: git}] inputs=[scripts/run-ncu.sh, scripts/run.jl, ClimaCoupler.jl/src, ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl/experiments/AMIP/Manifest-v1.11.toml, ClimaCore.jl/src, ClimaCore.jl/ext, ClimaAtmos.jl/src, RRTMGP.jl/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
set -euo pipefail
P=results/experiments/radiation-memory/baseline
bash scripts/run-ncu.sh "$P" \
  --kernel-name "regex:rte_.*_2stream_solve_.*" \
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
  --kernel-name "regex:rte_.*_2stream_solve_.*" \
  --project ClimaCoupler.jl-mod/experiments/AMIP \
  --config ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml \
  --launch-count 4
NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
"$NCU" --csv -i "$P.ncu-rep" --page raw --log-file "$P-raw.csv"
```

## Analysis

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/radiation-memory/stats.json, storage: git}] inputs=[scripts/measure_radiation_memory.py, results/experiments/radiation-memory/baseline-raw.csv, results/experiments/radiation-memory/baseline-details.csv, results/experiments/radiation-memory/mod-raw.csv, results/experiments/radiation-memory/mod-details.csv]
import json
import sys

sys.path.insert(0, "scripts")
from measure_radiation_memory import arm

D = "results/experiments/radiation-memory"
out = {}
for a in ("baseline", "mod"):
    out[a] = arm(f"{D}/{a}-raw.csv", f"{D}/{a}-details.csv")
    assert not out[a]["missing_metrics"], (
        f"{a}: ncu did not report {out[a]['missing_metrics']}"
    )
    assert out[a]["kernels"], f"{a}: the kernel filter matched nothing"

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
    }

json.dump(out, open(f"{D}/stats.json", "w"), indent=2, sort_keys=True)
h = out["baseline"]["headline"]
print(f"baseline radiation: {h['ms_per_call']} ms/call, {h['registers']} regs, "
      f"{h['achieved_occupancy_pct']}% occupancy, {h['dram_gb_per_call']} GB/call")
print(f"  worst sectors/request {h['worst_ld_sectors_per_request']}, "
      f"local share {h['worst_local_share_pct']}%, "
      f"spill overhead {h['worst_spill_overhead_pct']}%")
```

<!-- calkit values path=results/experiments/radiation-memory/stats.json -->

## Result

Filled from `stats.json` once both arms have run.

## Verdict

Pending.

# What a kernel launch costs

The AMIP benchmark issues about 3,300 kernel launches per step, and a large
share of them are short enough that the host time around them may exceed the GPU
time inside them. This prices one launch, so that "fuse kernels" can be argued
for or against with a number.

## Why both profiles are declared here

An earlier attempt measured the probe against a baseline profiled on a previous
day and reported 18.51 us per launch. That figure was withdrawn: profiling the
same code on two different days moved the host gap by 49%, which is the same
size as the effect. Host-side timing is comparable only within a session.

Both profiles are therefore declared in this document, run adjacently from one
invocation, and written to paths this experiment owns, so no later experiment can
overwrite them and no cross-session comparison can creep in.

```sh calkit stage name=baseline environment=clima outputs=[results/experiments/launch-cost/baseline.nsys-rep, results/experiments/launch-cost/baseline.sqlite] inputs=[scripts/run-nsys.sh, scripts/run.jl, ClimaCoupler.jl/src, ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl/experiments/AMIP/Manifest-v1.11.toml, ClimaCoupler.jl/experiments/AMIP/code_loading.jl, ClimaCore.jl/src, ClimaCore.jl/ext, ClimaAtmos.jl/src, RRTMGP.jl/src] scheduler={options: [--gpus=1, --time=180]}
scripts/run-nsys.sh results/experiments/launch-cost/baseline ClimaCoupler.jl/experiments/AMIP ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

```sh calkit stage name=probe environment=clima outputs=[results/experiments/launch-cost/probe.nsys-rep, results/experiments/launch-cost/probe.sqlite] inputs=[scripts/run-nsys.sh, scripts/run.jl, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.11.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, RRTMGP.jl-mod/src] scheduler={options: [--gpus=1, --time=180]}
scripts/run-nsys.sh results/experiments/launch-cost/probe ClimaCoupler.jl-mod/experiments/AMIP ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/launch-cost/stats.json, storage: git}] inputs=[scripts/measure_launch_overhead.py, results/experiments/launch-cost/baseline.sqlite, results/experiments/launch-cost/probe.sqlite]
import json, sys
sys.path.insert(0, "scripts")
from measure_launch_overhead import profile_stats

STEPS = 120
b = profile_stats("results/experiments/launch-cost/baseline.sqlite")
p = profile_stats("results/experiments/launch-cost/probe.sqlite")
d_n = p["launches"] - b["launches"]
d_gap = p["host_gap_ms"] - b["host_gap_ms"]
d_busy = p["gpu_busy_ms"] - b["gpu_busy_ms"]
out = {
    "steps": STEPS,
    "baseline": b,
    "probe": p,
    "marginal": {
        "added_launches": d_n,
        "added_host_gap_ms": round(d_gap, 1),
        "added_gpu_busy_ms": round(d_busy, 1),
        "host_us_per_launch": round(d_gap * 1000 / d_n, 2) if d_n else None,
        "wall_us_per_launch": round((d_gap + d_busy) * 1000 / d_n, 2) if d_n else None,
    },
    "baseline_launches_per_step": round(b["launches"] / STEPS, 1),
    "baseline_host_gap_ms_per_step": round(b["host_gap_ms"] / STEPS, 1),
    "baseline_gpu_busy_ms_per_step": round(b["gpu_busy_ms"] / STEPS, 1),
}
json.dump(out, open("results/experiments/launch-cost/stats.json", "w"), indent=2)
```

<!-- calkit values path=results/experiments/launch-cost/stats.json -->

## Result

The baseline issues <!-- calkit value key=baseline.launches format="{:,}" -->0<!-- /calkit value --> kernel launches over <!-- calkit value key=steps -->0<!-- /calkit value --> steps, or <!-- calkit value key=baseline_launches_per_step format="{:,.0f}" -->0<!-- /calkit value --> per step, with the GPU busy <!-- calkit value key=baseline_gpu_busy_ms_per_step -->0<!-- /calkit value --> ms per step and <!-- calkit value key=baseline_host_gap_ms_per_step -->0<!-- /calkit value --> ms per step spent in sub-millisecond gaps between kernels.
The probe arm adds <!-- calkit value key=marginal.added_launches format="{:,}" -->0<!-- /calkit value --> launches, which raises the host gap by <!-- calkit value key=marginal.added_host_gap_ms format="{:,}" -->0<!-- /calkit value --> ms and GPU time by <!-- calkit value key=marginal.added_gpu_busy_ms format="{:,}" -->0<!-- /calkit value --> ms.
That prices a launch at **<!-- calkit value key=marginal.host_us_per_launch -->0<!-- /calkit value --> us of host time**, or <!-- calkit value key=marginal.wall_us_per_launch -->0<!-- /calkit value --> us of wall time once the added GPU work is counted.

## Verdict

To be written once the pair has run.

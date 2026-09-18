# Host-side launch overhead in the AMIP benchmark

Every performance experiment in this project until now optimized kernel time.
This measures what the simulation is actually waiting on.

Numbers in this document are injected from `results/launch-overhead.json` by the
block below. They are not typed, and they are rewritten on every run, so the
prose cannot drift from the calculation.

```python calkit stage name=measure environment=py outputs=[{path: results/launch-overhead.json, storage: git}] inputs=[scripts/measure_launch_overhead.py, results/nsys/baseline.sqlite, results/launch-probe.json]
import json, sys
sys.path.insert(0, "scripts")
from measure_launch_overhead import profile_stats

STEPS = 120
b = profile_stats("results/nsys/baseline.sqlite")
b["launches_per_step"] = round(b["launches"] / STEPS, 1)
b["gpu_busy_ms_per_step"] = round(b["gpu_busy_ms"] / STEPS, 1)
b["host_gap_ms_per_step"] = round(b["host_gap_ms"] / STEPS, 1)
out = {"steps": STEPS, "baseline": b}

# The probe profile is read from a committed statistics file, NOT from
# results/nsys/mod.sqlite. Depending on that path made the pipeline regenerate
# the mod profile without the probe, silently destroying the measurement.
pr = json.load(open("results/launch-probe.json"))
out["probe"] = pr
d_n = pr["launches"] - b["launches"]
d_gap = pr["host_gap_ms"] - b["host_gap_ms"]
d_busy = pr["gpu_busy_ms"] - b["gpu_busy_ms"]
out["marginal"] = {
    "added_launches": d_n,
    "added_host_gap_ms": round(d_gap, 1),
    "added_gpu_busy_ms": round(d_busy, 1),
    "host_us_per_launch": round(d_gap * 1000 / d_n, 2),
    "wall_us_per_launch": round((d_gap + d_busy) * 1000 / d_n, 2),
}
json.dump(out, open("results/launch-overhead.json", "w"), indent=2)
```

<!-- calkit values path=results/launch-overhead.json -->

## The profile

Over <!-- calkit value key=steps -->120<!-- /calkit value --> steps the run issues <!-- calkit value key=baseline.launches format="{:,}" -->396,122<!-- /calkit value --> kernel launches, or <!-- calkit value key=baseline.launches_per_step format="{:,.0f}" -->3,301<!-- /calkit value --> per step.
The GPU is busy <!-- calkit value key=baseline.gpu_busy_ms_per_step -->244.6<!-- /calkit value --> ms per step, while <!-- calkit value key=baseline.host_gap_ms_per_step -->38.8<!-- /calkit value --> ms per step passes in sub-millisecond gaps between kernels, waiting on the host.
Kernels averaging under <!-- calkit value key=baseline.small_kernel_threshold_us -->25<!-- /calkit value --> us account for <!-- calkit value key=baseline.sub25us_launches format="{:,}" -->208,420<!-- /calkit value --> of those launches, spread over <!-- calkit value key=baseline.sub25us_kernel_kinds -->250<!-- /calkit value --> distinct kernels.

## What a launch costs

The mean gap of <!-- calkit value key=baseline.mean_gap_us -->11.77<!-- /calkit value --> us cannot price a launch, because it also contains genuine Julia work between broadcasts.
Measured instead as a slope, by adding <!-- calkit value key=marginal.added_launches format="{:,}" -->120,000<!-- /calkit value --> trivial launches and re-profiling, the host gap rose by <!-- calkit value key=marginal.added_host_gap_ms format="{:,}" -->2,221.6<!-- /calkit value --> ms and GPU time by <!-- calkit value key=marginal.added_gpu_busy_ms format="{:,}" -->868.8<!-- /calkit value --> ms.
That gives a marginal host cost of **<!-- calkit value key=marginal.host_us_per_launch -->18.51<!-- /calkit value --> us per launch**, or <!-- calkit value key=marginal.wall_us_per_launch -->25.75<!-- /calkit value --> us of wall time once the added GPU work is counted.
The marginal cost exceeds the mean gap, which means the host cannot queue launches as fast as the GPU drains them.

## Where the cost goes

Driver calls issued per kernel launch:

| call | calls per launch | us per launch |
|---|---|---|
| `cuLaunchKernel` | <!-- calkit value key=baseline.api.cuLaunchKernel.calls_per_launch -->1.0<!-- /calkit value --> | <!-- calkit value key=baseline.api.cuLaunchKernel.us_per_launch -->8.19<!-- /calkit value --> |
| `cuCtxGetId` | <!-- calkit value key=baseline.api.cuCtxGetId.calls_per_launch -->25.54<!-- /calkit value --> | <!-- calkit value key=baseline.api.cuCtxGetId.us_per_launch -->3.44<!-- /calkit value --> |
| `cuStreamGetCaptureInfo` | <!-- calkit value key=baseline.api.cuStreamGetCaptureInfo.calls_per_launch -->10.78<!-- /calkit value --> | <!-- calkit value key=baseline.api.cuStreamGetCaptureInfo.us_per_launch -->1.85<!-- /calkit value --> |

`cuCtxGetId` and `cuStreamGetCaptureInfo` compute nothing: they ask which context is current and whether a graph capture is active.
Their per-launch call counts are the finding.

## Status

The obvious remedy, a newer CUDA.jl, is blocked.
The AMIP environment pins CUDA v5; relaxing that resolves to v6, but `ClimaCore`'s CUDA extension extends `CUDA.shfl_recurse`, a CUDA.jl internal that v6 no longer provides.
Porting ClimaCore to CUDA v6 is the prerequisite.

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
#!/usr/bin/env bash
bash scripts/run-nsys.sh results/experiments/launch-cost/baseline ClimaCoupler.jl/experiments/AMIP ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

```sh calkit stage name=probe environment=clima outputs=[results/experiments/launch-cost/probe.nsys-rep, results/experiments/launch-cost/probe.sqlite] inputs=[scripts/run-nsys.sh, scripts/run.jl, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.11.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, RRTMGP.jl-mod/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
bash scripts/run-nsys.sh results/experiments/launch-cost/probe ClimaCoupler.jl-mod/experiments/AMIP ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
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
# Where the short launches come from, and what fusing them could ever be worth.
# A launch under this threshold costs more host time than the GPU time it
# produces, so it is the band fusion would target.
import collections, re, sqlite3

con = sqlite3.connect("results/experiments/launch-cost/baseline.sqlite")
kernels = con.execute(
    "SELECT s.value, COUNT(*), SUM(k.end - k.start) / 1e6 "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.demangledName = s.id GROUP BY s.value"
).fetchall()
THRESH_US = b["small_kernel_threshold_us"]
small = [r for r in kernels if r[2] * 1000 / r[1] < THRESH_US]
small_n = sum(r[1] for r in small)
us = out["marginal"]["host_us_per_launch"]
step_ms = (b["gpu_busy_ms"] + b["host_gap_ms"]) / STEPS

def package(name):
    m = re.search(r"FILE_(Clima\w+?|RRTMGP|CloudMicrophysics)_", name)
    if m:
        return m.group(1)
    return "raw broadcast" if "gpu_broadcast_kernel" in name else "other"

def atmos_file(name):
    m = re.search(r"FILE_ClimaAtmos_jl_src_([A-Za-z0-9_]+?)_jl", name)
    return m.group(1) if m else None

by_pkg = collections.Counter()
by_file = collections.Counter()
for name, n, _ in small:
    by_pkg[package(name)] += n
    f = atmos_file(name)
    if f:
        by_file[f] += n

def pct(launches, ratio=1.0):
    """Percent of step time that removing `ratio` of these launches would save."""
    return round(100 * (launches * ratio * us / 1000 / STEPS) / step_ms, 2)

out["fusion"] = {
    "step_ms": round(step_ms, 1),
    "launches_per_step": round(b["launches"] / STEPS, 1),
    "small_launches_per_step": round(small_n / STEPS, 1),
    "small_share_pct": round(100 * small_n / b["launches"], 1),
    # Ceilings. 4:1 is the optimistic fusion ratio; "all" is the absolute bound
    # where every short launch disappears, which no real change achieves.
    "ceiling_fuse_4to1_pct": pct(small_n, 0.75),
    "ceiling_all_removed_pct": pct(small_n, 1.0),
    "by_package": {
        k: {"per_step": round(v / STEPS, 1), "fuse_4to1_pct": pct(v, 0.75)}
        for k, v in by_pkg.most_common()
    },
    "climaatmos_by_file": {
        k: {"per_step": round(v / STEPS, 1), "fuse_4to1_pct": pct(v, 0.75)}
        for k, v in by_file.most_common(8)
    },
}
json.dump(out, open("results/experiments/launch-cost/stats.json", "w"), indent=2)
```

<!-- calkit values path=results/experiments/launch-cost/stats.json -->

## Result

The baseline issues <!-- calkit value key=baseline.launches format="{:,}" -->396,122<!-- /calkit value --> kernel launches over <!-- calkit value key=steps -->120<!-- /calkit value --> steps, or <!-- calkit value key=baseline_launches_per_step format="{:,.0f}" -->3,301<!-- /calkit value --> per step, with the GPU busy <!-- calkit value key=baseline_gpu_busy_ms_per_step -->245.3<!-- /calkit value --> ms per step and <!-- calkit value key=baseline_host_gap_ms_per_step -->51.8<!-- /calkit value --> ms per step spent in sub-millisecond gaps between kernels.
The probe arm adds <!-- calkit value key=marginal.added_launches format="{:,}" -->120,000<!-- /calkit value --> launches, which raises the host gap by <!-- calkit value key=marginal.added_host_gap_ms format="{:,}" -->809.3<!-- /calkit value --> ms and GPU time by <!-- calkit value key=marginal.added_gpu_busy_ms format="{:,}" -->739.8<!-- /calkit value --> ms.
That prices a launch at **<!-- calkit value key=marginal.host_us_per_launch -->6.74<!-- /calkit value --> us of host time**, or <!-- calkit value key=marginal.wall_us_per_launch -->12.91<!-- /calkit value --> us of wall time once the added GPU work is counted.

## What fusion could ever be worth

Of <!-- calkit value key=fusion.launches_per_step format="{:,.0f}" -->3,301<!-- /calkit value --> launches per step, <!-- calkit value key=fusion.small_launches_per_step format="{:,.0f}" -->1,737<!-- /calkit value --> are short enough (<!-- calkit value key=baseline.small_kernel_threshold_us -->25<!-- /calkit value --> us) that the host time around them exceeds the GPU time inside them.
Fusing that whole band four-to-one would save <!-- calkit value key=fusion.ceiling_fuse_4to1_pct -->2.96<!-- /calkit value -->% of a <!-- calkit value key=fusion.step_ms -->297.1<!-- /calkit value --> ms step, and even making every short launch vanish saves only <!-- calkit value key=fusion.ceiling_all_removed_pct -->3.94<!-- /calkit value -->%.

Those are ceilings across the entire stack, not a single package.
The band is spread over seven packages, so no one repository can deliver it.

| package | launches/step | fusing 4:1 |
|---|---|---|
| ClimaAtmos | <!-- calkit value key=fusion.by_package.ClimaAtmos.per_step -->791.8<!-- /calkit value --> | <!-- calkit value key=fusion.by_package.ClimaAtmos.fuse_4to1_pct -->1.35<!-- /calkit value -->% |
| ClimaCore | <!-- calkit value key=fusion.by_package.ClimaCore.per_step -->169.1<!-- /calkit value --> | <!-- calkit value key=fusion.by_package.ClimaCore.fuse_4to1_pct -->0.29<!-- /calkit value -->% |
| ClimaCoupler | <!-- calkit value key=fusion.by_package.ClimaCoupler.per_step -->147.0<!-- /calkit value --> | <!-- calkit value key=fusion.by_package.ClimaCoupler.fuse_4to1_pct -->0.25<!-- /calkit value -->% |
| ClimaLand | <!-- calkit value key=fusion.by_package.ClimaLand.per_step -->124.3<!-- /calkit value --> | <!-- calkit value key=fusion.by_package.ClimaLand.fuse_4to1_pct -->0.21<!-- /calkit value -->% |
| ClimaDiagnostics | <!-- calkit value key=fusion.by_package.ClimaDiagnostics.per_step -->93.6<!-- /calkit value --> | <!-- calkit value key=fusion.by_package.ClimaDiagnostics.fuse_4to1_pct -->0.16<!-- /calkit value -->% |

Within ClimaAtmos the densest source is `cache/microphysics_cache.jl` at <!-- calkit value key=fusion.climaatmos_by_file.cache_microphysics_cache.per_step -->296.2<!-- /calkit value --> launches per step, worth <!-- calkit value key=fusion.climaatmos_by_file.cache_microphysics_cache.fuse_4to1_pct -->0.5<!-- /calkit value -->% if fused four-to-one.
`cache/precomputed_quantities.jl`, which CliMA/ClimaAtmos.jl#4261 names as the place to start, is <!-- calkit value key=fusion.climaatmos_by_file.cache_precomputed_quantities.per_step -->62.0<!-- /calkit value --> launches per step and worth <!-- calkit value key=fusion.climaatmos_by_file.cache_precomputed_quantities.fuse_4to1_pct -->0.11<!-- /calkit value -->%.
The cache folder is the right instinct; the microphysics cache is the bigger half of it.

## Verdict

A launch costs about 7 us of host time, not the 18.51 us reported from the
cross-day comparison. That earlier figure was inflated roughly 2.7x by node
load, and is withdrawn.

Profiling identical code on three occasions gave host gaps of 4,656, 6,837 and
6,214 ms, a spread of about 30% with no code change at all. The probe's whole
effect here is 809 ms, well inside that spread, so this measurement is
trustworthy only because both profiles were produced back to back from one
invocation. Host-side timing must never be compared across sessions.

What it is worth: halving the launch count saves about 11 ms of a 297 ms step,
so **roughly 4% of step time**, not the 10.8% claimed before. Fusing the
sub-25 us band 4:1 -- 53% of all launches -- is worth about 3%.

That keeps kernel fusion a real, science-free target, and a larger one than
anything left in the radiation kernels, but it is a third of what it looked
like. It does not on its own justify a fusion campaign across ClimaCore and
ClimaAtmos; it justifies fusing where the launches are densest and measuring
again.

Note also that 6.74 us sits below the 8.19 us that `cuLaunchKernel` itself
occupies in the baseline profile, which suggests the marginal launch is mostly
driver cost rather than Julia-side broadcast machinery, and that some of it
overlaps with GPU execution.

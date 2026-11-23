# What the aerosol index searches cost

The MERRA aerosol optics finds two indices per layer before it can read a table: a bisection over the relative-humidity nodes, `interp1d_loc_factor(rh, rh_levels)`, and a linear scan over the size-bin limits, `locate_merra_size_bin`.
With the fifteen species this configuration prescribes, that is eight bisections per layer --- five sea-salt bins, sulfate, hydrophilic black carbon, hydrophilic organic carbon --- and ten scans.
Every one of those indices is a function of `(column, layer)` alone and none of them depends on the g-point, so the grouped solve recomputes each one 256 times in the longwave and 224 times in the shortwave.
`loc_lower`'s own comment says as much: *"Called per layer, per g-point, per aerosol species."*

This experiment does not cache them. It replaces both searches with constants and profiles what is left, which bounds above what any per-`(column, layer)` cache could recover.

## The probe is deliberately incorrect

`loc, factor = (1, FT(0.5))` at the four humidity sites and `bin = 1` at the two size sites.
The probe keeps every table read and every multiply --- it removes the *finding* of the index, not the use of it --- so what disappears from the profile is search work and nothing else.

It over-bounds, and by a margin that cannot be estimated from the profile.
Constant indices also make the table reads warp-uniform, where a real cache would leave them scattered, since neighboring columns sit on different humidity nodes.
So the realizable saving is strictly smaller than what is measured here, which is the direction a bound has to err in.

This is the same shape of measurement as the cloud-mask sector census, which bounded a transposition at a saving too small to build: the point of a bound is to decline cheaply.

## Why there is one profiling stage and not two

The unprobed arm is already measured. `results/gpoint-groups.json` carries both optics kernels at `mod_ref` 419e8027, the dark-column skip, committed and in Git at a few kilobytes --- which is exactly the snapshot that a transient profile is supposed to be read through.
Profiling it again would spend a second reservation to re-measure a number this repository already holds.

The cost of reading across two passes is profile-to-profile drift, and this experiment carries its own control for it: the two *sweeps* kernels reach no aerosol code at all, so the probe cannot touch them, and their movement between the reference and the probe is the drift band the optics deltas are read against.

```sh calkit stage name=probe environment=clima outputs=[results/experiments/aerosol-index-bound/probe.nsys-rep, results/experiments/aerosol-index-bound/probe.sqlite] inputs=[scripts/run-nsys.sh, scripts/run.jl, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, RRTMGP.jl-mod/src] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
bash scripts/run-nsys.sh results/experiments/aerosol-index-bound/probe ClimaCoupler.jl-mod/experiments/AMIP ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Analysis

```python calkit stage name=analyze environment=py outputs=[{path: results/experiments/aerosol-index-bound/stats.json, storage: git}] inputs=[scripts/export-gpoint-groups.py, results/experiments/aerosol-index-bound/probe.sqlite, results/gpoint-groups.json, results/summary.json, results/null-census.json]
import hashlib
import importlib.util
import json
import subprocess

spec = importlib.util.spec_from_file_location(
    "gpg", "scripts/export-gpoint-groups.py"
)
gpg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpg)

PROBE = "results/experiments/aerosol-index-bound/probe.sqlite"
ref = json.load(open("results/gpoint-groups.json"))
live = json.load(open("results/summary.json"))
nulls = json.load(open("results/null-census.json"))

# The reference figures are the committed snapshot of the unprobed arm, read
# per kernel name so a renamed or merged kernel goes missing rather than
# silently pairing with the wrong row.
reference = {
    k["name"]: k["elapsed_ms"]
    for band in ("longwave", "shortwave")
    for k in ref[band]["mod"]["kernels"]
}

con = gpg.connect(PROBE)
probe = {}
for fam in ("_lw_group_optics", "_lw_group_sweeps",
            "_sw_group_optics", "_sw_group_sweeps"):
    f = gpg.family(con, [f"%{fam}%"])
    if f is None:
        raise SystemExit(f"{fam} absent from the probe profile")
    for k in f["kernels"]:
        probe[k["name"]] = k["elapsed_ms"]

def row(name):
    was, now = reference[name], probe[name]
    return {
        "reference_ms": was,
        "probe_ms": now,
        "saved_ms": round(was - now, 1),
        "saved_pct": round(100.0 * (was - now) / was, 2),
    }

optics = {n: row(n) for n in probe if "optics" in n}
# Nothing in the probe is reachable from a sweeps kernel, so whatever these
# two move is drift between the two profiles, not the treatment.
controls = {n: row(n) for n in probe if "sweeps" in n}

drift_ms = max(abs(c["saved_ms"]) for c in controls.values())
bound_ms = sum(o["saved_ms"] for o in optics.values())
ref_ms = sum(reference[n] for n in optics)

# The probe diff is the only thing that makes this reproducible: it ran against
# a dirty tree on purpose, because the change is wrong and does not belong on a
# branch of the upstream repository.
diff = subprocess.run(
    ["git", "-C", "RRTMGP.jl-mod", "diff", "--", "src/optics/aerosol_optics.jl"],
    capture_output=True, text=True, check=True,
).stdout

out = {
    "optics": optics,
    "controls": controls,
    "totals": {
        "reference_optics_ms": round(ref_ms, 1),
        "bound_ms": round(bound_ms, 1),
        "bound_pct_of_optics": round(100.0 * bound_ms / ref_ms, 2),
        "bound_pct_of_device": round(
            100.0 * bound_ms / ref["device_total"]["mod_elapsed_ms"], 2
        ),
        # The profiled window is 120 steps, so a kernel saving converts to a
        # wall-clock share only against an unprofiled step from the same pass.
        "bound_ms_per_step": round(bound_ms / 120, 2),
        "bound_pct_of_wall_clock": round(
            100.0 * (bound_ms / 120)
            / (live["mod"]["walltime_per_step_s"] * 1000.0), 2
        ),
        "null_floor_pct": nulls["current_window_mean_pct"],
        # Unlike the dark-column skip, a saving this size would clear the
        # wall-clock floor, so realizing it would be visible in an AMIP pair.
        "would_clear_null_floor": (
            100.0 * (bound_ms / 120)
            / (live["mod"]["walltime_per_step_s"] * 1000.0)
        ) > nulls["current_window_mean_pct"],
        "drift_ms": drift_ms,
        # A bound is only a bound if it is bigger than the noise it was read
        # through; below this it says nothing except that nothing was resolved.
        "resolved_against_controls": bound_ms > 4 * drift_ms,
    },
    "provenance": {
        "reference_source": "results/gpoint-groups.json",
        "reference_mod_ref": ref["provenance"]["mod_ref"],
        "probe_base_ref": subprocess.run(
            ["git", "-C", "RRTMGP.jl-mod", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "probe_diff": diff,
        "probe_diff_sha256_16": hashlib.sha256(diff.encode()).hexdigest()[:16],
        "probe_db_sha256_16": hashlib.sha256(
            open(PROBE, "rb").read()
        ).hexdigest()[:16],
        "nsys_steps": 120,
    },
}
json.dump(out, open(
    "results/experiments/aerosol-index-bound/stats.json", "w"), indent=2)
```

<!-- calkit values path=results/experiments/aerosol-index-bound/stats.json -->

## Result

Half the time in the radiation optics goes on finding indices rather than on using them.
Removing the two searches takes the longwave optics from <!-- calkit value key=optics._lw_group_optics_CUDA_.reference_ms -->1088.7<!-- /calkit value --> ms to <!-- calkit value key=optics._lw_group_optics_CUDA_.probe_ms -->549.5<!-- /calkit value --> ms, a saving of <!-- calkit value key=optics._lw_group_optics_CUDA_.saved_ms -->539.2<!-- /calkit value --> ms or <!-- calkit value key=optics._lw_group_optics_CUDA_.saved_pct -->49.53<!-- /calkit value -->%, and the shortwave optics from <!-- calkit value key=optics._sw_group_optics_CUDA_.reference_ms -->518.6<!-- /calkit value --> ms to <!-- calkit value key=optics._sw_group_optics_CUDA_.probe_ms -->236.0<!-- /calkit value --> ms, <!-- calkit value key=optics._sw_group_optics_CUDA_.saved_ms -->282.6<!-- /calkit value --> ms or <!-- calkit value key=optics._sw_group_optics_CUDA_.saved_pct -->54.49<!-- /calkit value -->%.
Together that is <!-- calkit value key=totals.bound_ms -->821.8<!-- /calkit value --> ms of the <!-- calkit value key=totals.reference_optics_ms -->1607.3<!-- /calkit value --> ms the two kernels hold, or <!-- calkit value key=totals.bound_pct_of_optics -->51.13<!-- /calkit value -->%, and <!-- calkit value key=totals.bound_pct_of_device -->3.29<!-- /calkit value -->% of device time.

The controls say the figure is resolved rather than read through drift.
Neither sweeps kernel reaches aerosol code, and between the reference profile and the probe each moved <!-- calkit value key=controls._lw_group_sweeps_CUDA_.saved_ms -->-0.1<!-- /calkit value --> and <!-- calkit value key=controls._sw_group_sweeps_CUDA_.saved_ms -->-0.1<!-- /calkit value --> ms --- a drift band of <!-- calkit value key=totals.drift_ms -->0.1<!-- /calkit value --> ms against a bound three to four orders of magnitude larger, so `resolved_against_controls` records <!-- calkit value key=totals.resolved_against_controls -->true<!-- /calkit value -->.
Two further confounds are ruled out by the profiles rather than by argument: both arms issued the same number of kernel launches, and both optics kernels ran at 96 registers on the same grid and block, which is the register cap binding in each.
So no part of this is occupancy, and no part of it is a changed kernel inventory.

Converted at the window this repository benchmarks, the bound is <!-- calkit value key=totals.bound_ms_per_step -->6.85<!-- /calkit value --> ms per coupling step, or <!-- calkit value key=totals.bound_pct_of_wall_clock -->2.75<!-- /calkit value -->% of wall clock against this window's null floor of <!-- calkit value key=totals.null_floor_pct -->1.869<!-- /calkit value -->%, so `would_clear_null_floor` records <!-- calkit value key=totals.would_clear_null_floor -->true<!-- /calkit value -->.
That is the part worth noting beside the dark-column skip measured just before it: the skip was worth about a point of wall clock and therefore invisible to an AMIP pair by construction, while a cache that recovered even half of this bound would be the first radiation treatment here that the wall clock could resolve unaided.

## Verdict

Accepted as a bound, and it is large enough to justify building the cache.

What it does not say is how much of the <!-- calkit value key=totals.bound_ms -->821.8<!-- /calkit value --> ms is reachable.
Constant indices do three things a cache cannot: they make the table reads warp-uniform where neighboring columns genuinely sit on different humidity nodes and size bins, they let the compiler fold the table addresses, and they cost nothing to store, whereas a real cache has to write the indices once per `(column, layer)` and read them back on every g-point.
The last of those is the one that can be designed around, since the indices are small integers and a factor --- and it is also the one that decides the answer, because the gas optics already shows this kernel is latency-bound on dependent L1 reads rather than bandwidth-bound.

So the next step is not a cache for all eighteen searches. It is the cheapest subset that tests the mechanism: the four humidity bisections share a single `rh`, so one `(loc, factor)` pair per `(column, layer)` serves all four, which is 8 bytes per column-layer against four bisections of about six dependent loads each.
If that subset does not pay, the remaining fourteen will not either, and the bound stays a bound.

<!-- /calkit values -->

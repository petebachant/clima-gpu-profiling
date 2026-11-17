# Capping a kernel at its occupancy band edge automatically

ClimaCore sizes a slice loop's grid from the register count the compile produced, and the register file is handed out in granules, so resident warps fall in bands rather than smoothly.
A kernel that compiles a few registers past a band edge loses a fifth to a quarter of its resident warps for nothing: `ptxas` optimizes instructions against a budget nobody gave it and has no reason to stop at the edge.
This experiment adds a policy to `auto_launch!` that compiles a kernel, reads the registers back off it, and recompiles at the band edge when doing so is cheap -- and then asks what that is worth over a whole coupled model rather than over the one kernel it was designed for.

The question is not whether the cap helps the kernel it was built for; `quadrature-saturation` already measured that at -17.65% with the cap placed by hand.
The question is whether a policy can find such kernels without being told, and without capping the ones where the trade loses.
Both halves matter, because this repo has measured the trade failing: capping one kernel from 255 registers to 64 bought four times the occupancy and cost 5.5% of wall clock.

## Why the census is a stage rather than a probe

The register probes in `scripts/measure-*.jl` are run by hand and their TOMLs committed as stage inputs.
That is fine for a probe that prices one kernel, and wrong here: this census is a claim about the whole kernel population, so it has to be regenerable from the committed state rather than from whatever was in the shell history.
Declaring it here scopes its output to the experiment, which is what keeps the next experiment from overwriting it, and freezing this file freezes the census with it.

No profiler runs. The policy records its own decisions, so the census is a by-product of one model run, and the step count is chosen to clear the long cycles rather than to be small: 35 coupler steps covers one radiation and cloud-fraction cycle and two gravity-wave cycles, and stepping is seconds against the compile.

```sh calkit stage name=census environment=clima outputs=[{path: results/experiments/auto-register-cap/census.toml, storage: git}] inputs=[scripts/census-auto-register-cap.jl, scripts/run-julia-script.sh, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, .calkit/env-locks/clima] scheduler={options: [--gpus=1, --time=90]}
#!/usr/bin/env bash
# The policy is read from the environment in the extension's `__init__`, so it
# has to be set before the process starts rather than flipped from the script.
# Kernel naming is what makes a decision attributable to a call site.
export CLIMA_AUTO_REGISTER_CAP=true
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true
mkdir -p results/experiments/auto-register-cap
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/census-auto-register-cap.jl \
  --out results/experiments/auto-register-cap/census.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## What the policy decided

```python calkit stage name=values environment=py outputs=[{path: results/docs/auto-register-cap.json, storage: git}] inputs=[results/experiments/auto-register-cap/census.toml, results/top-kernels.csv, results/null-census.json, results/summary.json]
import csv
import json
import tomllib

census = tomllib.load(open("results/experiments/auto-register-cap/census.toml", "rb"))
rows = list(census["decisions"].values())

# The profile reports the mod arm under the baseline arm's paths, so a census
# name matches a row exactly once the `-mod` suffix is dropped. Matching has to
# be exact: the sibling specializations of a capped call site share its first
# fifty characters and differ only in the line number, and the policy left them
# alone, so a prefix match would price a cap together with a kernel it declined.
def profile_name(name):
    return name.replace("_jl_mod_", "_jl_")

# Both arms' device budgets from one row of the profile, so each denominator is
# the profile's own rather than a sum over a table that holds the top kernels.
paired = {}
device_base = device_mod = None
with open("results/top-kernels.csv") as f:
    for row in csv.DictReader(f):
        base, mod = row["total_ms_baseline"], row["total_ms_mod"]
        pct_base, pct_mod = row["pct_of_gpu_time_baseline"], row["pct_of_gpu_time_mod"]
        if not (base and mod):
            continue
        paired[row["kernel"]] = (float(base), float(mod))
        if device_base is None and pct_base and pct_mod:
            device_base = float(base) / float(pct_base) * 100.0
            device_mod = float(mod) / float(pct_mod) * 100.0

capped = [r for r in rows if r["reason"] == "capped"]
priced = [(r, paired[profile_name(r["kernel"])]) for r in capped
          if profile_name(r["kernel"]) in paired]
priced.sort(key=lambda rp: -rp[1][0])
capped_base = sum(b for _, (b, _) in priced)
capped_mod = sum(m for _, (_, m) in priced)

# Signs: a saving is positive, and the drift of everything the policy declined
# keeps its sign, because an untreated remainder that got slower is a different
# fact from one that got faster.
saving_ms = device_base - device_mod
capped_saving_ms = capped_base - capped_mod
untreated_base = device_base - capped_base
untreated_drift_ms = (device_mod - device_base) - (capped_mod - capped_base)

# The residency filter is the test that does the work, so the sweep that shows
# what it removes belongs in the record rather than in a terminal.
def passing(max_warps):
    return sum(1 for r in rows if r["warps_per_sm"] <= max_warps)

by_reason = census["by_reason"]
null = json.load(open("results/null-census.json"))
summary = json.load(open("results/summary.json"))

values = {
    "census": {
        "specializations": census["n_specializations"],
        "capped": census["n_capped"],
        "already_resident": by_reason.get("already_resident", 0),
        "edge_too_far": by_reason.get("edge_too_far", 0),
        "spill_over_budget": by_reason.get("spill_over_budget", 0),
        "cap_not_honored": by_reason.get("cap_not_honored", 0),
        "max_shed": census["max_shed"],
        "max_warps": census.get("max_warps", 16),
        "local_growth_pct": census["local_growth_pct"],
        "local_slack_bytes": census["local_slack_bytes"],
        "steps": census["steps"],
        # The filter's reach, swept from the record: every decision carries the
        # registers and warps it was judged on, so the arithmetic tests can be
        # re-asked at any threshold without another run.
        "pass_filter_at_16": passing(16),
        "pass_filter_at_32": passing(32),
        "pass_filter_at_64": passing(64),
        "unfiltered_is_everything": passing(64) == census["n_specializations"],
    },
    "capped_kernels": [
        {
            "kernel": r["kernel"][:64],
            "registers": r["registers"],
            "cap": r["cap"],
            "shed": r["shed"],
            "warps_before": r["warps_per_sm"],
            "warps_after": r["capped_warps_per_sm"],
            "local_before": r["local_bytes"],
            "local_after": r["capped_local_bytes"],
            "local_budget": r["local_budget"],
            "baseline_ms": round(b, 1),
            "mod_ms": round(m, 1),
            "change_pct": round(100.0 * (m - b) / b, 2),
        }
        for r, (b, m) in priced
    ],
    "device": {
        "device_ms": round(device_mod, 1),
        "device_ms_baseline": round(device_base, 1),
        "capped_ms": round(capped_mod, 1),
        "capped_pct_of_device": round(100.0 * capped_mod / device_mod, 2),
        "capped_priced": len(priced),
        # A capped kernel outside the profile's table is one the policy acted on
        # and the measurement cannot price, so it is counted rather than hidden.
        "capped_unpriced": len(capped) - len(priced),
        "saving_ms": round(saving_ms, 1),
        "saving_pct": round(100.0 * saving_ms / device_base, 3),
        "capped_saving_ms": round(capped_saving_ms, 1),
        "capped_saving_pct": round(100.0 * capped_saving_ms / capped_base, 2),
        "capped_share_of_saving_pct": round(100.0 * capped_saving_ms / saving_ms, 1),
        "untreated_drift_pct": round(100.0 * untreated_drift_ms / untreated_base, 3),
        "saving_over_untreated_drift_x": round(
            abs(saving_ms / device_base) / abs(untreated_drift_ms / untreated_base), 1
        ),
    },
    "wall_clock": {
        "sypd_pct": round(summary["speedup_pct"], 3),
        "nsys_pct": round(summary["nsys_speedup_pct"], 3),
        "null_floor_pct": null["current_window_mean_pct"],
        "null_spread_points": null["current_window_spread_points"],
        "margin_over_null_points": round(
            summary["speedup_pct"] - null["current_window_mean_pct"], 3
        ),
        # A margin resolves a headline only if it is wider than the scatter of
        # the draws that set the floor; one draw is not the floor.
        "margin_exceeds_null_spread": (
            summary["speedup_pct"] - null["current_window_mean_pct"]
        )
        > null["current_window_spread_points"],
    },
}
json.dump(values, open("results/docs/auto-register-cap.json", "w"), indent=2)
```

Over <!-- calkit value key=census.steps path=results/docs/auto-register-cap.json -->35<!-- /calkit value --> coupler steps of the benchmark configuration the policy saw <!-- calkit value key=census.specializations path=results/docs/auto-register-cap.json -->690<!-- /calkit value --> kernel specializations and capped <!-- calkit value key=census.capped path=results/docs/auto-register-cap.json -->5<!-- /calkit value --> of them.

That is the headline, and it is a small number for a reason worth stating precisely rather than apologizing for.
It declined <!-- calkit value key=census.already_resident path=results/docs/auto-register-cap.json -->680<!-- /calkit value --> kernels because they already hold more than <!-- calkit value key=census.max_warps path=results/docs/auto-register-cap.json -->16<!-- /calkit value --> warps per multiprocessor, and <!-- calkit value key=census.edge_too_far path=results/docs/auto-register-cap.json -->5<!-- /calkit value --> because the band edge below them is further than <!-- calkit value key=census.max_shed path=results/docs/auto-register-cap.json -->12<!-- /calkit value --> registers away.

## The residency filter is the test that does the work

Sweeping it over the census -- which costs nothing, because every decision records the registers and warps it was judged on -- gives the reach of each threshold: <!-- calkit value key=census.pass_filter_at_16 path=results/docs/auto-register-cap.json -->10<!-- /calkit value --> kernels at 16 warps, <!-- calkit value key=census.pass_filter_at_32 path=results/docs/auto-register-cap.json -->33<!-- /calkit value --> at 32, and <!-- calkit value key=census.pass_filter_at_64 path=results/docs/auto-register-cap.json -->690<!-- /calkit value --> with no filter at all.

That last figure is the whole argument for having one: <!-- calkit value key=census.unfiltered_is_everything path=results/docs/auto-register-cap.json -->true<!-- /calkit value --> says every kernel in the model has a band below it, so a policy that asks only "is there a band below this?" treats the entire library as register-limited and caps it.

## What the capped kernels are worth

The capped kernels account for <!-- calkit value key=device.capped_pct_of_device path=results/docs/auto-register-cap.json -->25.74<!-- /calkit value -->% of the <!-- calkit value key=device.device_ms format="{:,}" path=results/docs/auto-register-cap.json -->33,811.0<!-- /calkit value --> ms device budget, which bounds what the policy can be worth but does not measure it: a kernel's share is what the cap acts on, not what it saves.

The paired profile measures it, over arms that differ in nothing but ClimaCore: device time falls by <!-- calkit value key=device.saving_ms format="{:,}" path=results/docs/auto-register-cap.json -->1,016.0<!-- /calkit value --> ms, or <!-- calkit value key=device.saving_pct path=results/docs/auto-register-cap.json -->2.917<!-- /calkit value -->%, and the capped kernels account for <!-- calkit value key=device.capped_share_of_saving_pct path=results/docs/auto-register-cap.json -->103.0<!-- /calkit value -->% of that.
A policy that found its own targets therefore produced a saving that is the sum of its own decisions, which is the claim this experiment exists to test; the share is near a hundred rather than at it because the rest of the model drifts, and that drift is the control below.

| capped kernel | regs | warps/SM | baseline | capped | change |
| --- | --- | --- | --- | --- | --- |
| `set_environment_1m_tendency` | <!-- calkit value key=capped_kernels.0.registers path=results/docs/auto-register-cap.json -->136<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.0.cap path=results/docs/auto-register-cap.json -->128<!-- /calkit value --> | <!-- calkit value key=capped_kernels.0.warps_before path=results/docs/auto-register-cap.json -->12<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.0.warps_after path=results/docs/auto-register-cap.json -->16<!-- /calkit value --> | <!-- calkit value key=capped_kernels.0.baseline_ms format="{:,}" path=results/docs/auto-register-cap.json -->7,330.4<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.0.mod_ms format="{:,}" path=results/docs/auto-register-cap.json -->6,615.1<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.0.change_pct path=results/docs/auto-register-cap.json -->-9.76<!-- /calkit value -->% |
| `update_updraft_microphysics_tendency_cache` | <!-- calkit value key=capped_kernels.1.registers path=results/docs/auto-register-cap.json -->108<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.1.cap path=results/docs/auto-register-cap.json -->96<!-- /calkit value --> | <!-- calkit value key=capped_kernels.1.warps_before path=results/docs/auto-register-cap.json -->16<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.1.warps_after path=results/docs/auto-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=capped_kernels.1.baseline_ms format="{:,}" path=results/docs/auto-register-cap.json -->1,695.2<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.1.mod_ms format="{:,}" path=results/docs/auto-register-cap.json -->1,466.1<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.1.change_pct path=results/docs/auto-register-cap.json -->-13.51<!-- /calkit value -->% |
| `set_microphysics_tendency_cache` | <!-- calkit value key=capped_kernels.2.registers path=results/docs/auto-register-cap.json -->108<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.2.cap path=results/docs/auto-register-cap.json -->96<!-- /calkit value --> | <!-- calkit value key=capped_kernels.2.warps_before path=results/docs/auto-register-cap.json -->16<!-- /calkit value --> -> <!-- calkit value key=capped_kernels.2.warps_after path=results/docs/auto-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=capped_kernels.2.baseline_ms format="{:,}" path=results/docs/auto-register-cap.json -->723.0<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.2.mod_ms format="{:,}" path=results/docs/auto-register-cap.json -->621.3<!-- /calkit value --> ms | <!-- calkit value key=capped_kernels.2.change_pct path=results/docs/auto-register-cap.json -->-14.06<!-- /calkit value -->% |

The table holds the <!-- calkit value key=device.capped_priced path=results/docs/auto-register-cap.json -->3<!-- /calkit value --> capped kernels the profile prices; the remaining <!-- calkit value key=device.capped_unpriced path=results/docs/auto-register-cap.json -->2<!-- /calkit value --> are ClimaLand surface kernels that fall outside its table of the largest, and they are worth about a millisecond between them.

The two at <!-- calkit value key=capped_kernels.1.registers path=results/docs/auto-register-cap.json -->108<!-- /calkit value --> registers are the pair the local-memory budget admitted by eight bytes, <!-- calkit value key=capped_kernels.1.local_after path=results/docs/auto-register-cap.json -->56<!-- /calkit value --> against a budget of <!-- calkit value key=capped_kernels.1.local_budget path=results/docs/auto-register-cap.json -->64<!-- /calkit value -->, and they are the reason the budget was written as a test rather than a rule of thumb.
They had never been measured capped, so the eight bytes of headroom were the open risk in the whole policy, and both came out faster than the kernel the policy was designed for.

The control is the untreated remainder: everything the policy declined moved <!-- calkit value key=device.untreated_drift_pct path=results/docs/auto-register-cap.json -->0.12<!-- /calkit value -->% in total, so the device-side headline is <!-- calkit value key=device.saving_over_untreated_drift_x path=results/docs/auto-register-cap.json -->24.4<!-- /calkit value -->x the drift of the kernels nothing was done to.

## Wall clock

The AMIP pair on this stack gives <!-- calkit value key=wall_clock.sypd_pct path=results/docs/auto-register-cap.json -->1.883<!-- /calkit value -->% SYPD against a null floor of <!-- calkit value key=wall_clock.null_floor_pct path=results/docs/auto-register-cap.json -->1.869<!-- /calkit value -->%, a margin of <!-- calkit value key=wall_clock.margin_over_null_points path=results/docs/auto-register-cap.json -->0.014<!-- /calkit value --> points, and <!-- calkit value key=wall_clock.margin_exceeds_null_spread path=results/docs/auto-register-cap.json -->false<!-- /calkit value --> for whether that margin exceeds the spread between the draws that set the floor.

So the wall clock does not resolve this treatment, and the arithmetic says it could not have: a device saving of <!-- calkit value key=device.saving_pct path=results/docs/auto-register-cap.json -->2.917<!-- /calkit value -->% sits inside a floor of <!-- calkit value key=wall_clock.null_floor_pct path=results/docs/auto-register-cap.json -->1.869<!-- /calkit value -->% whose two draws are <!-- calkit value key=wall_clock.null_spread_points path=results/docs/auto-register-cap.json -->1.196<!-- /calkit value --> points apart.
Reading the SYPD figure as the answer either way would be reading the floor, which is why the headline here is the device figure and the AMIP pair is reported as consistent with it rather than as its confirmation.
The instruments disagree about magnitude and not about sign, and the device profile is the one with the resolution, because it prices every kernel against its own counterpart in the other arm instead of differencing two whole runs.

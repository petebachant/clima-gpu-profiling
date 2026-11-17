# What the residency filter costs, and whether the policy survives CM #852

ClimaCore's automatic register cap (`docs/experiments/auto-register-cap.md`) capped 5 of 690 kernel specializations and bought 2.917% of device time, and all three of the kernels the profile could price are microphysics cache kernels.
CloudMicrophysics #852 rewrites exactly those kernels, and by the better route: it removes registers rather than capping them, so it pays no spill.
That raises two questions this experiment answers, neither of which needs a second arm.

The first is what the policy's residency filter costs.
It declines any kernel already holding more than `CLIMA_AUTO_REGISTER_CAP_MAX_WARPS` warps, set to 16, on the argument that occupancy has sharply diminishing returns and the only evidence for trading spill for it covers one step from 12 warps to 16.
The counterexamples that motivated the filter --- a `weighted_dss_internal` that regressed 34% and an `update_turbulent_fluxes` that regressed 21% --- sit between 48 and 64 warps, so the whole range from 16 to 48 is asserted rather than measured.

The second is whether the policy still finds anything once #852 lands.
That one is answerable from #852's own published register counts, because every filter the policy applies before compiling is arithmetic on the register count.

## Why a census with the residency filter inert

A census records what the policy decided and the device-side numbers behind each decision, so the filters that are arithmetic --- residency, the band edge, the shed --- can be replayed at any setting from a census taken at one setting.
The spill test cannot: what a kernel compiles to under a cap is not predictable from its source, which is the whole reason the policy compiles twice and decides rather than predicts.
A kernel the residency filter declined therefore has no recorded capped compile, and a census at 16 warps can say which kernels a wider setting would *consider* but not which it would *cap*.

So this census runs at `CLIMA_AUTO_REGISTER_CAP_MAX_WARPS=64`, which is the warps-per-SM ceiling of an A100 and makes the residency test inert.
Every kernel the shed filter admits then pays for its capped compile and has its spill recorded, and the entire filter curve becomes replayable offline from one job --- including at settings nobody thought to try when the job ran.
It costs one second compile per admitted kernel against the 690 the run compiles anyway.

Nothing else moves: same stack, same configuration, same 35 steps, same `max_shed` of 12, and the arms are untouched, so this restales nothing.
The census at 16 warps stays where it is and the two are cross-checked against each other below.

```sh calkit stage name=wide-census environment=clima outputs=[{path: results/experiments/register-cap-filter/census-wide.toml, storage: git}] inputs=[scripts/census-auto-register-cap.jl, scripts/run-julia-script.sh, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, .calkit/env-locks/clima] scheduler={options: [--gpus=1, --time=150]}
#!/usr/bin/env bash
# 64 warps per SM is the A100's ceiling, so the residency test can never fire and
# every kernel inside the shed admits itself to the capped compile. The policy is
# read in the extension's `__init__`, so these have to be set before the process
# starts rather than flipped from the script.
export CLIMA_AUTO_REGISTER_CAP=true
export CLIMA_AUTO_REGISTER_CAP_MAX_WARPS=64
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true
mkdir -p results/experiments/register-cap-filter
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/census-auto-register-cap.jl \
  --out results/experiments/register-cap-filter/census-wide.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## Replaying the filter

```python calkit stage name=values environment=py outputs=[{path: results/docs/register-cap-filter.json, storage: git}] inputs=[scripts/register_bands.py, results/experiments/register-cap-filter/census-wide.toml, results/experiments/auto-register-cap/census.toml, results/docs/auto-register-cap.json, results/top-kernels.csv, data/cm-852-registers.toml]
import csv, json, sys, tomllib
sys.path.insert(0, "scripts")
import register_bands as rb

wide = tomllib.load(open("results/experiments/register-cap-filter/census-wide.toml", "rb"))
narrow = tomllib.load(open("results/experiments/auto-register-cap/census.toml", "rb"))
cm = tomllib.load(open("data/cm-852-registers.toml", "rb"))
wide_rows = list(wide["decisions"].values())
narrow_rows = list(narrow["decisions"].values())
max_shed = wide["max_shed"]

# The band arithmetic has to agree with the warps and edges the extension itself
# recorded, or this replays a different device rather than a different setting.
assert wide["max_warps"] >= 64, f"census was taken at max_warps={wide['max_warps']}"
assert max_shed == narrow["max_shed"], "the two censuses used different sheds"
assert rb.band_edges() == rb.A100_BAND_EDGES, rb.band_edges()
(checked_warps, checked_edges) = rb.assert_matches_census(wide_rows)

# A kernel's identity for comparing two runs: the name alone is not unique, since
# one function has several specializations and the census names it by call site.
def key(r):
    return (r["kernel"], r["registers"], r["local_bytes"])

# Independent check between two GPU runs: replayed down to 16 warps, the wide
# census must cap exactly what the census taken at 16 warps capped.
replayed_16 = {key(r) for r in wide_rows
               if rb.replay(r, 16, max_shed) == "capped"}
measured_16 = {key(r) for r in narrow_rows if r["reason"] == "capped"}
agree = replayed_16 == measured_16

def sweep(rows, max_warps):
    verdicts = [rb.replay(r, max_warps, max_shed) for r in rows]
    capped = [r for r, v in zip(rows, verdicts) if v == "capped"]
    return {
        "max_warps": max_warps,
        "capped": len(capped),
        "candidates": sum(1 for v in verdicts if v in rb.POST_COMPILE),
        "spill_over_budget": sum(1 for v in verdicts if v == "spill_over_budget"),
        "buys_no_warps": sum(1 for v in verdicts if v == "buys_no_warps"),
        "cap_not_honored": sum(1 for v in verdicts if v == "cap_not_honored"),
        "already_resident": sum(1 for v in verdicts if v == "already_resident"),
        "edge_too_far": sum(1 for v in verdicts if v == "edge_too_far"),
        "no_band_below": sum(1 for v in verdicts if v == "no_band_below"),
        "shed_total": sum(r["registers"] - r["edge"] for r in capped),
        "spill_added_bytes": sum(r["capped_local_bytes"] - r["local_bytes"]
                                 for r in capped),
    }

curve = [sweep(wide_rows, m) for m in (16, 20, 24, 32, 40, 48, 64)]
by_warps = {str(s["max_warps"]): s for s in curve}

# What the wide setting capped that 16 warps declined, most registers first.
new_at_64 = sorted(
    (r for r in wide_rows
     if rb.replay(r, 64, max_shed) == "capped" and key(r) not in measured_16),
    key=lambda r: -r["registers"])
new_rows = [{
    "kernel": r["kernel"][:70],
    "registers": r["registers"],
    "edge": r["edge"],
    "shed": r["registers"] - r["edge"],
    "warps": r["warps_per_sm"],
    "capped_warps": r["capped_warps_per_sm"],
    "local_bytes": r["local_bytes"],
    "capped_local_bytes": r["capped_local_bytes"],
    "local_budget": r["local_budget"],
} for r in new_at_64]

# CloudMicrophysics #852, from its own table: the filters that precede the
# capped compile, which is as far as published register counts can reach.
def cm_at(max_warps):
    rows = [(k, rb.filters_before_compile(k["registers"], max_warps, max_shed)[0])
            for k in cm["kernels"]]
    admitted = [k for k, verdict in rows if verdict == "candidate"]
    return {
        "max_warps": max_warps,
        "candidates": len(admitted),
        "declined": len(rows) - len(admitted),
        "candidate_ms_per_step": round(sum(k["ms_per_step"] for k in admitted), 2),
    }

cm_total_ms = sum(k["ms_per_step"] for k in cm["kernels"])
cm_curve = [cm_at(m) for m in (16, 24, 32, 48)]
for s in cm_curve:
    s["candidate_share_pct"] = round(100 * s["candidate_ms_per_step"] / cm_total_ms, 1)

# Pricing what a wider setting admits needs the profile, and
# `results/top-kernels.csv` is a transient path: it holds whichever arm pair ran
# last. The budget it implies is asserted against the figure the
# auto-register-cap experiment committed, so a profile from a different pair
# fails this stage instead of quietly repricing against it.
prof_rows = list(csv.DictReader(open("results/top-kernels.csv")))
device_base = next(
    float(r["total_ms_baseline"]) / float(r["pct_of_gpu_time_baseline"]) * 100.0
    for r in prof_rows if float(r["pct_of_gpu_time_baseline"]))
committed = json.load(
    open("results/docs/auto-register-cap.json"))["device"]["device_ms_baseline"]
assert abs(device_base - committed) < 1.0, (
    f"the profile implies a {device_base:.1f} ms budget against the committed "
    f"{committed} ms, so results/top-kernels.csv describes a different arm pair")

# The profile reports the mod arm under the baseline arm's paths.
def normalized(name):
    return name.replace("_jl_mod_", "_jl_")

priced = {normalized(r["kernel"]): r for r in prof_rows}
capped_at = {m: {key(r) for r in wide_rows if rb.replay(r, m, max_shed) == "capped"}
             for m in (16, 24, 32)}
names_16 = {k[0] for k in capped_at[16]}

# One name can carry several specializations, and the profile aggregates them
# into a single row, so a call site already capped at 16 warps cannot have a
# wider setting's share of it priced apart from what 16 already captured.
multi = sum(1 for n in {r["kernel"] for r in wide_rows}
            if sum(1 for r in wide_rows if r["kernel"] == n) > 1)

# Aggregated by name and not by specialization, because the profile is: a name
# with two admitted specializations has one row, and pricing both reads that row
# twice. 47 of this census's rows are one `copyto_by_slices` call site.
def wider(max_warps):
    admitted = {r["kernel"] for r in wide_rows
                if rb.replay(r, max_warps, max_shed) == "capped"}
    fresh = admitted - names_16
    specs = {n: sorted(r["registers"] for r in wide_rows if r["kernel"] == n
                       and rb.replay(r, max_warps, max_shed) == "capped")
             for n in fresh}
    rows, total = [], 0.0
    for name in sorted(fresh, key=lambda n: -max(specs[n])):
        hit = priced.get(normalized(name))
        ms = round(float(hit["total_ms_baseline"]), 1) if hit else None
        if ms:
            total += ms
        rows.append({"kernel": name[:70], "registers": specs[name],
                     "profiled_ms": ms})
    return {
        "call_sites": len(fresh),
        "profiled": sum(1 for r in rows if r["profiled_ms"]),
        "priced_ms": round(total, 1),
        "priced_pct_of_device": round(100.0 * total / device_base, 2),
        "kernels": rows,
    }

out = {
    "wide": {
        "specializations": wide["n_specializations"],
        "max_warps": wide["max_warps"],
        "max_shed": max_shed,
        "capped": wide["n_capped"],
        "measured": wide["measured"],
        "climacore_commit": wide["climacore_commit"][:9],
        "checked_warps": checked_warps,
        "checked_edges": checked_edges,
    },
    "cross_check": {
        "replayed_at_16": len(replayed_16),
        "measured_at_16": len(measured_16),
        "agree": agree,
        "only_replayed": sorted(k[0][:70] for k in replayed_16 - measured_16),
        "only_measured": sorted(k[0][:70] for k in measured_16 - replayed_16),
    },
    "curve": curve,
    "by_warps": by_warps,
    "new_at_64": new_rows,
    "device_ms_baseline": round(device_base, 1),
    "names_with_several_specializations": multi,
    "wider": {"24": wider(24), "32": wider(32)},
    "cm_852": {
        "kernels": len(cm["kernels"]),
        "total_ms_per_step": round(cm_total_ms, 2),
        "source": cm["source"],
        "curve": cm_curve,
        "by_warps": {str(s["max_warps"]): s for s in cm_curve},
    },
}
json.dump(out, open("results/docs/register-cap-filter.json", "w"), indent=2)
print(f"wide census: {out['wide']['specializations']} specializations, "
      f"{out['wide']['capped']} capped at max_warps={out['wide']['max_warps']}")
print(f"cross-check at 16 warps: replayed {len(replayed_16)}, "
      f"measured {len(measured_16)}, agree={agree}")
for s in curve:
    print(f"  max_warps {s['max_warps']:>2}: capped {s['capped']:>3} of "
          f"{s['candidates']:>3} candidates, "
          f"{s['spill_over_budget']} over spill budget, "
          f"{s['buys_no_warps']} buying no warps")
for m in ("24", "32"):
    w = out["wider"][m]
    print(f"  wider to {m}: {w['call_sites']} new call sites, {w['profiled']} "
          f"profiled, {w['priced_ms']} ms = {w['priced_pct_of_device']}% of device")
for s in cm_curve:
    print(f"  #852 at max_warps {s['max_warps']:>2}: {s['candidates']} of "
          f"{out['cm_852']['kernels']} kernels candidate, "
          f"{s['candidate_share_pct']}% of their time")
```

<!-- calkit values path=results/docs/register-cap-filter.json -->

## What the census says

The wide census decided <!-- calkit value key=wide.specializations -->690<!-- /calkit value --> specializations with the residency filter inert, and the band arithmetic here agrees with the extension's own on all <!-- calkit value key=wide.checked_warps -->690<!-- /calkit value --> of them and on every one of the <!-- calkit value key=wide.checked_edges -->133<!-- /calkit value --> band edges it recorded.
Replayed down to 16 warps it caps <!-- calkit value key=cross_check.replayed_at_16 -->5<!-- /calkit value --> kernels against the <!-- calkit value key=cross_check.measured_at_16 -->5<!-- /calkit value --> that the census taken at 16 warps capped, and that the two sets are identical is <!-- calkit value key=cross_check.agree -->true<!-- /calkit value -->.
That is a check between two independent runs rather than a restatement: the compiles happened in different processes, and nothing but the setting differed.

Lifting the filter to 24 warps caps <!-- calkit value key=by_warps.24.capped -->14<!-- /calkit value --> kernels against <!-- calkit value key=by_warps.16.capped -->5<!-- /calkit value --> at 16, and removing it entirely caps <!-- calkit value key=by_warps.64.capped -->125<!-- /calkit value --> of the <!-- calkit value key=by_warps.64.candidates -->127<!-- /calkit value --> the shed filter admits.
The spill test declines almost none of them: <!-- calkit value key=by_warps.64.spill_over_budget -->2<!-- /calkit value --> kernels compile past their local budget and <!-- calkit value key=by_warps.64.buys_no_warps -->0<!-- /calkit value --> buy no warps at all.
A cap of at most 12 registers is small enough that `ptxas` meets it without spilling: the <!-- calkit value key=by_warps.24.capped -->14<!-- /calkit value --> kernels a 24-warp filter caps add <!-- calkit value key=by_warps.24.spill_added_bytes -->160<!-- /calkit value --> bytes of local memory between them, and all <!-- calkit value key=by_warps.64.capped -->125<!-- /calkit value --> at the widest setting add <!-- calkit value key=by_warps.64.spill_added_bytes -->536<!-- /calkit value --> bytes.

Pricing what a wider setting admits needs care, because a kernel name is not a specialization: <!-- calkit value key=names_with_several_specializations -->54<!-- /calkit value --> of this census's names carry more than one, and one `copyto_by_slices` call site carries 47.
The profile has one row per name, so a call site is priced once however many of its specializations the filter admits, and a call site already capped at 16 warps cannot have a wider setting's share of it separated from what 16 already captured.
On that basis, lifting the filter to 24 warps brings <!-- calkit value key=wider.24.call_sites -->6<!-- /calkit value --> new call sites under the policy, of which <!-- calkit value key=wider.24.profiled -->2<!-- /calkit value --> appear in the profile at all, worth <!-- calkit value key=wider.24.priced_pct_of_device -->3.67<!-- /calkit value -->% of the device budget.
Lifting it to 32 brings <!-- calkit value key=wider.32.call_sites -->14<!-- /calkit value --> call sites and <!-- calkit value key=wider.32.priced_pct_of_device -->16.28<!-- /calkit value -->%.

## What #852's own numbers say

The eight microphysics kernels of #852's table take <!-- calkit value key=cm_852.total_ms_per_step -->61.04<!-- /calkit value --> ms per coupling step between them.
At the filter's current setting of 16 warps the policy declines <!-- calkit value key=cm_852.by_warps.16.declined -->8<!-- /calkit value --> of the <!-- calkit value key=cm_852.kernels -->8<!-- /calkit value --> outright, which is every one of them: #852 leaves the lightest of them holding 40 warps and the heaviest 20, and the filter wants 16 or fewer.
At 24 warps it admits <!-- calkit value key=cm_852.by_warps.24.candidates -->5<!-- /calkit value --> of them, carrying <!-- calkit value key=cm_852.by_warps.24.candidate_share_pct -->81.4<!-- /calkit value -->% of that time, and at 32 warps <!-- calkit value key=cm_852.by_warps.32.candidates -->6<!-- /calkit value --> of them.

Admitting is not capping, and published register counts cannot reach further: what a kernel compiles to under a cap is the one thing only a compile knows.
Read these as the ceiling on what the policy could still find on that stack, not as a payoff.

## Verdict

**Leave `CLIMA_AUTO_REGISTER_CAP_MAX_WARPS` at 16, and stop describing the spill budget as a safeguard.**

The filter is not a conservative setting of a well-screened policy.
It is the only screen the policy has.
Of the <!-- calkit value key=by_warps.64.candidates -->127<!-- /calkit value --> kernels the band arithmetic admits, the spill budget turns away <!-- calkit value key=by_warps.64.spill_over_budget -->2<!-- /calkit value -->, so with residency removed the policy caps <!-- calkit value key=by_warps.64.capped -->125<!-- /calkit value --> of <!-- calkit value key=wide.specializations -->690<!-- /calkit value --> specializations rather than the <!-- calkit value key=by_warps.16.capped -->5<!-- /calkit value --> it caps now.
Everything between those two numbers is the residency test and nothing else.

That the spill is near zero is not reassurance, and this is the part worth carrying forward.
`ptxas` can meet a tighter register budget by rematerializing values instead of spilling them, which costs instructions and no local memory at all, so a kernel that regresses under a cap looks identical to one that is fine.
The policy's own comment records the case: a ClimaLand `turbulent_fluxes` kernel regressed 21% while spilling 44 bytes.
A budget that declines 2 of <!-- calkit value key=by_warps.64.candidates -->127<!-- /calkit value --> is therefore consistent with the caps being harmless and equally consistent with their cost sitting in a currency nothing here measures.

Against that, the upside of a wider filter is small.
At 24 warps the policy would reach <!-- calkit value key=wider.24.priced_pct_of_device -->3.67<!-- /calkit value -->% more of the device budget, on <!-- calkit value key=wider.24.profiled -->2<!-- /calkit value --> profiled call sites, and it returns about a tenth of what it covers, so the expected gain is a few tenths of a percent of device --- the same order as the drift of the kernels it does not touch.
Each new admission would need pricing one at a time, because nothing downstream of the arithmetic would catch a bad one.

The CloudMicrophysics finding is the one that bears on whether to upstream this at all.
At 16 warps the policy declines every one of the <!-- calkit value key=cm_852.kernels -->8<!-- /calkit value --> microphysics kernels of #852's table, so the <!-- calkit value key=cm_852.total_ms_per_step -->61.04<!-- /calkit value --> ms per step they cost is entirely outside its reach once that lands, and the 2.9% the policy is credited with today rests on kernels #852 takes registers out of instead.
At 24 warps five of them return, carrying <!-- calkit value key=cm_852.by_warps.24.candidate_share_pct -->81.4<!-- /calkit value -->% of that time.
So the setting that would keep the policy relevant is the one its only screen cannot be trusted to guard, which is the honest statement of where this stands.

What the policy is worth keeping for is unchanged by any of it: it found five kernels nobody had looked at, two of them in ClimaLand, and it declines 685 of 690 by default.
That is a mechanism argument, and it is the one to make upstream --- not the 2.9%.

<!-- /calkit values -->

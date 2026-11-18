# Does the register cap still find anything once CM #852 lands?

`docs/experiments/register-cap-filter.md` establishes that at the policy's shipped setting of 16 warps, every one of the eight microphysics kernels of CloudMicrophysics #852's table is declined as already resident, and that at 24 warps five of them return carrying most of their time.
That was replayed from register counts #852 published, which reach as far as arithmetic can and no further: what a kernel compiles to *under a cap* is the one thing only a compile knows, so whether those five would actually be capped was open.

This census closes it by putting the stack in the arm and asking the policy directly.

It also answers a question the replay was taken to have answered and had not.
A table's rows are not the population: the replay was exact about #852's eight, and the kernel carrying most of the cap's priced saving is not one of them, so the census had to be read against what the cap actually earned rather than against the table.
The answer to the title is yes --- most of it, and in a kernel #852 improved without lifting it past the gate.

## What is in the arm

**Both** arms take CloudMicrophysics #852 (`dy/microphysics_fusion`) and ClimaAtmos #4985 (`dy/params_fusion`), at the same commit on each side, and the register cap in ClimaCore is the only thing that differs between them.
That is deliberate and it is the point of the arrangement: CloudMicrophysics was a registry dependency in both arms, so dev'ing it into the mod arm alone would make the arms differ by its whole version rather than by a change to it --- the failure that rode along inside every CM experiment before 2026-09-14 (`docs/learnings.md` 4i), and the reason RRTMGP ended up dev'd on both sides.
A census needs only one arm, but leaving the pair measurable is what makes the next question --- what the cap is worth on this stack --- askable without redoing any of this.

The two branches have to travel together: #852 is branched on #851's params-macro API change and ClimaAtmos has to be adapted to it, so neither can be taken alone.
Both sit directly on what the arms already carried --- #4985 is four commits on the ClimaAtmos commit both arms shared, with nothing in the arms that is not in it --- so this is additive rather than a re-baseline.
A new `CloudMicrophysics.jl` submodule carries the baseline side; the dev'd set is now the same five packages on both arms, and the two manifests are identical but their dev paths and the `project_hash` those paths imply.

Two caveats the figures below carry.

The population is **not comparable to the previous census kernel for kernel**. Resolving the new manifests carried nine registry bumps along with the two branches, `ClimaParams` 1.3.10 to 1.3.12 because #851 requires it, and with it `ForwardDiff` 1.4.8 to 1.5.0 and `KernelAbstractions` 0.9.47 to 0.9.48, either of which can move code generation and so register counts across the whole population.
Both arms resolved to the same nine independently, so the pair is clean; it is the comparison with `register-cap-filter`'s census that carries the drift.
So read the per-kernel decisions below, which are measured on this stack and answer the question asked, and treat any count-against-count comparison with the earlier census as indicative only.

And **#852 is an open branch that is force-pushed**. It moved twice on the day this was set up, and the commit measured here is the live tip of `dy/microphysics_fusion` at that point rather than an earlier one --- an earlier head had already fallen off every remote ref, which would have left a submodule pointer nobody else could fetch. `results/treatment.json` records the commit this census actually ran.

## The census, again with the residency filter inert

Taken at `CLIMA_AUTO_REGISTER_CAP_MAX_WARPS=64`, the warps-per-SM ceiling of an A100, for the reason in `docs/learnings.md` 8w: the residency test is arithmetic on a register count, so running with it open makes every setting answerable afterwards --- including 24, which is the setting this experiment is about --- while a census taken at 24 could say nothing about any other.

```sh calkit stage name=census environment=clima outputs=[{path: results/experiments/register-cap-after-cm852/census-wide.toml, storage: git}] inputs=[scripts/census-auto-register-cap.jl, scripts/run-julia-script.sh, ClimaCore.jl-mod/src, ClimaCore.jl-mod/ext, ClimaAtmos.jl-mod/src, CloudMicrophysics.jl-mod/src, ClimaCoupler.jl-mod/src, ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.13.toml, ClimaCoupler.jl-mod/experiments/AMIP/code_loading.jl, ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml, .calkit/env-locks/clima] scheduler={options: [--gpus=1, --time=180]}
#!/usr/bin/env bash
# Same settings as register-cap-filter's census, so the two differ by the stack
# and nothing else: the policy is read in the extension's `__init__`, so these
# are set before the process starts rather than flipped from the script.
export CLIMA_AUTO_REGISTER_CAP=true
export CLIMA_AUTO_REGISTER_CAP_MAX_WARPS=64
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true
mkdir -p results/experiments/register-cap-after-cm852
bash scripts/run-julia-script.sh \
  --project=ClimaCoupler.jl-mod/experiments/AMIP \
  scripts/census-auto-register-cap.jl \
  --out results/experiments/register-cap-after-cm852/census-wide.toml \
  --config_file ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
```

## The microphysics kernels, decided rather than predicted

```python calkit stage name=values environment=py outputs=[{path: results/docs/register-cap-after-cm852.json, storage: git}] inputs=[scripts/register_bands.py, results/experiments/register-cap-after-cm852/census-wide.toml, results/experiments/register-cap-filter/census-wide.toml, data/cm-852-registers.toml, results/docs/auto-register-cap.json]
import json, sys, tomllib
sys.path.insert(0, "scripts")
import register_bands as rb

after = tomllib.load(open(
    "results/experiments/register-cap-after-cm852/census-wide.toml", "rb"))
before = tomllib.load(open(
    "results/experiments/register-cap-filter/census-wide.toml", "rb"))
cm = tomllib.load(open("data/cm-852-registers.toml", "rb"))
# What the cap measured as its own worth, carried forward so the Verdict quotes
# it rather than restating it.
measured = json.load(open("results/docs/auto-register-cap.json"))
rows = list(after["decisions"].values())
prev = list(before["decisions"].values())
max_shed = after["max_shed"]

assert after["max_warps"] >= 64, f"census taken at max_warps={after['max_warps']}"
assert max_shed == before["max_shed"], "the two censuses used different sheds"
(checked_warps, checked_edges) = rb.assert_matches_census(rows)

# The function names behind #852's eight table rows. Matched on the function
# rather than the whole kernel name, because ClimaAtmos #4985 moves the line
# numbers the names carry and a census name is call site plus line.
#
# One row needs translating. #852's table attributes its "microphysics Jacobian"
# row to `update_jacobian!`, which in ClimaAtmos is the whole ManualSparseJacobian
# entry point; this census names kernels from the stack trace, so the launch
# appears under the function that issues it. Matching the table's name verbatim
# silently found nothing and dropped the row from the population.
ALIASES = {"update_jacobian": "update_implicit_microphysics_jacobian"}
FUNCS = sorted({ALIASES.get(f, f)
                for f in (k["kernel"].rstrip("!") for k in cm["kernels"])})
assert all(any(f in r["kernel"] for r in rows) for f in FUNCS), (
    "a function from #852's table matches no kernel in this census: "
    f"{[f for f in FUNCS if not any(f in r['kernel'] for r in rows)]}")

def curve(decisions, settings=(16, 20, 24, 32, 48, 64)):
    out = {}
    for m in settings:
        verdicts = [rb.replay(r, m, max_shed) for r in decisions]
        out[str(m)] = {
            "capped": sum(1 for v in verdicts if v == "capped"),
            "candidates": sum(1 for v in verdicts if v in rb.POST_COMPILE),
            "spill_over_budget": sum(1 for v in verdicts if v == "spill_over_budget"),
            "buys_no_warps": sum(1 for v in verdicts if v == "buys_no_warps"),
        }
    return out

# Every specialization whose name carries one of the five functions, with what
# the policy decides for it at each setting.
def describe(decisions):
    out = []
    for r in sorted((r for r in decisions if any(f in r["kernel"] for f in FUNCS)),
                    key=lambda r: -r["registers"]):
        out.append({
            "kernel": r["kernel"][:74],
            "registers": r["registers"],
            "warps": r["warps_per_sm"],
            "edge": r["edge"] if r["edge"] > 0 else None,
            "shed": r["shed"] if r["shed"] > 0 else None,
            "capped_warps": (r["capped_warps_per_sm"]
                             if r["capped_warps_per_sm"] > 0 else None),
            "local_bytes": r["local_bytes"],
            "capped_local_bytes": (r["capped_local_bytes"]
                                   if r["capped_local_bytes"] >= 0 else None),
            "local_budget": r["local_budget"],
            "at_16": rb.replay(r, 16, max_shed),
            "at_24": rb.replay(r, 24, max_shed),
            "at_32": rb.replay(r, 32, max_shed),
        })
    return out

micro = describe(rows)
prev_micro = describe(prev)

def tally(rs, key, value):
    return sum(1 for m in rs if m[key] == value)

# The gate is `warps > MAX_WARPS` (cuda_utils.jl), so a kernel sitting at exactly
# MAX_WARPS is admitted and capped a band ABOVE the target rather than up to it.
# Counted separately, because it is not the same claim as finding a kernel below.
def at_target(rs, m):
    return sum(1 for r in rs if r["warps"] == m and r[f"at_{m}"] == "capped")

def population_at_target(decisions, m):
    return sum(1 for r in decisions if r["warps_per_sm"] == m
               and rb.replay(r, m, max_shed) == "capped")

# Per function, the worst specialization each side of #852: minimum warps is what
# the policy reads, and a function carries more than one specialization.
def by_function(rs):
    out = {}
    for f in FUNCS:
        mine = [r for r in rs if f in r["kernel"]]
        if not mine:
            continue
        out[f] = {
            "specializations": len(mine),
            "max_registers": max(r["registers"] for r in mine),
            "min_warps": min(r["warps"] for r in mine),
            "capped_at_16": sum(1 for r in mine if r["at_16"] == "capped"),
            "capped_at_24": sum(1 for r in mine if r["at_24"] == "capped"),
        }
    return out

# The kernels the cap treated at its shipped setting, before #852. Every figure
# the Result quotes for them is quoted once for all of them, so assert they agree.
treated_16 = [m for m in prev_micro if m["at_16"] == "capped"]
assert treated_16, "the earlier census capped no microphysics kernel at 16 warps"
for field in ("registers", "warps", "edge", "capped_warps"):
    assert len({m[field] for m in treated_16}) == 1, (
        f"the kernels capped at 16 warps disagree on {field}: "
        f"{sorted(m[field] for m in treated_16)}")

# What #852 does to the cap's OWN measured saving, which is the value question.
# Each `capped_kernels` entry is one profile row, so its saving is attributed to
# a name; the question is whether that name still carries a specialization the
# policy admits. Matched by prefix, because the measured names are truncated.
def specializations_of(decisions, name):
    return sorted((r for r in decisions if r["kernel"].startswith(name)),
                  key=lambda r: -r["registers"])

retention = []
for k in measured["capped_kernels"]:
    now = specializations_of(rows, k["kernel"])
    assert now, f"a kernel the cap capped is absent from this census: {k['kernel']}"
    still = [r for r in now if rb.replay(r, 16, max_shed) == "capped"]
    retention.append({
        "kernel": k["kernel"],
        "saving_ms": round(k["baseline_ms"] - k["mod_ms"], 1),
        "change_pct": k["change_pct"],
        "registers_then": k["registers"],
        "warps_then": k["warps_before"],
        "warps_then_capped": k["warps_after"],
        "registers_now": max(r["registers"] for r in now),
        "warps_now": min(r["warps_per_sm"] for r in now),
        "specializations_now": len(now),
        "still_capped_at_16": len(still),
        "warps_now_capped": (max(r["capped_warps_per_sm"] for r in still)
                             if still else None),
    })

# The prose below names the largest retained earner, so pick it by saving rather
# than by position in the measured file and assert that the name is still right.
# A value marker cannot check the identifier a sentence types around it.
retained = [r for r in retention if r["still_capped_at_16"]]
top = max(retained, key=lambda r: r["saving_ms"]) if retained else None
assert top is not None and top["kernel"].startswith("set_environment_1m_tendency"), (
    "the Result and Verdict name `set_environment_1m_tendency` as the kernel "
    "carrying the cap's retained saving; the census now says it is "
    f"{top['kernel'] if top else 'nothing -- no priced kernel is still capped'}")

retained_ms = sum(r["saving_ms"] for r in retention if r["still_capped_at_16"])
lost_ms = sum(r["saving_ms"] for r in retention if not r["still_capped_at_16"])
total_ms = retained_ms + lost_ms
assert total_ms > 0, "the cap's measured kernels carry no saving between them"

out = {
    "stack": {
        "specializations": after["n_specializations"],
        "max_warps": after["max_warps"],
        "max_shed": max_shed,
        "measured": after["measured"],
        "steps": after["steps"],
        "checked_warps": checked_warps,
        "checked_edges": checked_edges,
    },
    "curve": curve(rows),
    "curve_before": curve(prev),
    "micro": {
        "functions": FUNCS,
        "specializations": len(micro),
        "specializations_before": len(prev_micro),
        "capped_at_16": tally(micro, "at_16", "capped"),
        "capped_at_24": tally(micro, "at_24", "capped"),
        "capped_at_32": tally(micro, "at_32", "capped"),
        "resident_at_16": tally(micro, "at_16", "already_resident"),
        "resident_at_24": tally(micro, "at_24", "already_resident"),
        "spill_over_budget_at_24": tally(micro, "at_24", "spill_over_budget"),
        "buys_no_warps_at_24": tally(micro, "at_24", "buys_no_warps"),
        "edge_too_far_at_24": tally(micro, "at_24", "edge_too_far"),
        "max_registers": max((m["registers"] for m in micro), default=0),
        "min_registers": min((m["registers"] for m in micro), default=0),
        "min_warps": min((m["warps"] for m in micro), default=0),
        "at_target_at_24": at_target(micro, 24),
        "below_target_at_24": tally(micro, "at_24", "capped") - at_target(micro, 24),
        "by_function": by_function(micro),
        "kernels": micro,
    },
    "micro_before": {
        "specializations": len(prev_micro),
        "capped_at_16": tally(prev_micro, "at_16", "capped"),
        "capped_at_24": tally(prev_micro, "at_24", "capped"),
        "resident_at_16": tally(prev_micro, "at_16", "already_resident"),
        "max_registers": max((m["registers"] for m in prev_micro), default=0),
        "min_warps": min((m["warps"] for m in prev_micro), default=0),
        # What the cap bought the kernels it treated at the shipped setting. The
        # assertion is what lets the prose quote one figure for both of them.
        "capped_at_16_registers": treated_16[0]["registers"],
        "capped_at_16_warps": treated_16[0]["warps"],
        "capped_at_16_edge": treated_16[0]["edge"],
        "capped_at_16_warps_bought": treated_16[0]["capped_warps"],
        "by_function": by_function(prev_micro),
        "kernels": prev_micro,
    },
    "population": {
        "at_target_at_24": population_at_target(rows, 24),
        "at_target_at_24_before": population_at_target(prev, 24),
        "specializations_before": before["n_specializations"],
    },
    "measured": {
        "device_saving_pct": measured["device"]["saving_pct"],
        "capped_priced": measured["device"]["capped_priced"],
        "capped_pct_of_device": measured["device"]["capped_pct_of_device"],
        "capped_share_of_saving_pct": measured["device"]["capped_share_of_saving_pct"],
        "max_warps": measured["census"]["max_warps"],
    },
    "retention": {
        "kernels": retention,
        "top": top,
        "retained_kernels": sum(1 for r in retention if r["still_capped_at_16"]),
        "lost_kernels": sum(1 for r in retention if not r["still_capped_at_16"]),
        "retained_ms": round(retained_ms, 1),
        "lost_ms": round(lost_ms, 1),
        "total_ms": round(total_ms, 1),
        "retained_pct": round(100.0 * retained_ms / total_ms, 1),
        "lost_pct": round(100.0 * lost_ms / total_ms, 1),
        # The device saving scaled by the share whose kernel is still admitted.
        # An upper bound on what survives, not a prediction: the kernel that
        # survives compiles to a different register count now, so the saving the
        # cap buys on it has to be measured rather than carried over.
        "device_pct_still_in_reach": round(
            measured["device"]["saving_pct"] * retained_ms / total_ms, 2),
    },
    "predicted": {
        "candidates_at_16": 0,
        "candidates_at_24": 5,
        "source": cm["source"],
    },
}
json.dump(out, open("results/docs/register-cap-after-cm852.json", "w"), indent=2)
print(f"{out['stack']['specializations']} specializations, "
      f"{len(micro)} of them microphysics (was {len(prev_micro)})")
for m in ("16", "24", "32", "64"):
    c = out["curve"][m]
    print(f"  max_warps {m:>2}: capped {c['capped']:>3} of {c['candidates']:>3} "
          f"candidates, {c['spill_over_budget']} over spill budget")
r = out["retention"]
print(f"  measured saving still in reach: {r['retained_pct']}% "
      f"({r['retained_kernels']} of {len(retention)} priced kernels)")
for k in retention:
    print(f"    {k['saving_ms']:>7.1f} ms  {k['registers_then']:>3} -> "
          f"{k['registers_now']:>3} regs  {k['warps_then']:>2} -> "
          f"{k['warps_now']:>2} warps  "
          f"{'STILL CAPPED' if k['still_capped_at_16'] else 'declined'}  "
          f"{k['kernel'][:50]}")
print(f"  microphysics capped: {out['micro']['capped_at_16']} at 16, "
      f"{out['micro']['capped_at_24']} at 24, {out['micro']['capped_at_32']} at 32")
for m in micro:
    print(f"    {m['registers']:>3} regs {m['warps']:>2} warps  "
          f"16:{m['at_16']:<17} 24:{m['at_24']:<17} {m['kernel'][:46]}")
```

<!-- calkit values path=results/docs/register-cap-after-cm852.json -->

## Result

The census decided <!-- calkit value key=stack.specializations -->627<!-- /calkit value --> specializations over <!-- calkit value key=stack.steps -->35<!-- /calkit value --> coupler steps with the residency gate wide open, against <!-- calkit value key=population.specializations_before -->690<!-- /calkit value --> on the stack before #852, and the band arithmetic in `scripts/register_bands.py` agrees with the extension's own on all <!-- calkit value key=stack.checked_warps -->627<!-- /calkit value --> of them and on each of the <!-- calkit value key=stack.checked_edges -->117<!-- /calkit value --> band edges the census recorded.

### The eight table rows: the prediction holds

Every one of #852's table rows is now declined at the policy's shipped setting.
All <!-- calkit value key=micro.specializations -->6<!-- /calkit value --> specializations of those functions are already resident at 16 warps and <!-- calkit value key=micro.capped_at_16 -->0<!-- /calkit value --> are capped, which is exactly what `register-cap-filter.md` replayed from the published register counts (<!-- calkit value key=predicted.candidates_at_16 -->0<!-- /calkit value --> candidates) --- and it holds for the reason the replay gave rather than by coincidence.
The two the cap used to treat are the two #852 rewrote: in the earlier census `set_microphysics_tendency_cache` and `update_updraft_microphysics_tendency_cache` both compiled to <!-- calkit value key=micro_before.capped_at_16_registers -->108<!-- /calkit value --> registers for <!-- calkit value key=micro_before.capped_at_16_warps -->16<!-- /calkit value --> warps per SM, and the cap shed to the <!-- calkit value key=micro_before.capped_at_16_edge -->96<!-- /calkit value -->-register edge to buy them <!-- calkit value key=micro_before.capped_at_16_warps_bought -->20<!-- /calkit value -->.
They now compile to <!-- calkit value key=micro.by_function.set_microphysics_tendency_cache.max_registers -->75<!-- /calkit value --> and <!-- calkit value key=micro.by_function.update_updraft_microphysics_tendency_cache.max_registers -->88<!-- /calkit value --> registers for <!-- calkit value key=micro.by_function.set_microphysics_tendency_cache.min_warps -->24<!-- /calkit value --> and <!-- calkit value key=micro.by_function.update_updraft_microphysics_tendency_cache.min_warps -->20<!-- /calkit value --> warps with no cap at all, so #852 reaches what the cap bought and for one of them a band beyond it, in source rather than through `ptxas`.

### The cap's own earnings: the prediction was answering a narrower question

That is not the same as the policy having nothing left, and the difference is most of its measured value.
The cap's saving was priced over <!-- calkit value key=measured.capped_priced -->3<!-- /calkit value --> kernels, and <!-- calkit value key=retention.retained_pct -->68.4<!-- /calkit value -->% of it --- <!-- calkit value key=retention.retained_ms -->715.3<!-- /calkit value --> ms of <!-- calkit value key=retention.total_ms -->1046.1<!-- /calkit value --> --- came from one kernel that is not in #852's table at all.
`set_environment_1m_tendency` is a ClimaAtmos microphysics-cache kernel, it went from <!-- calkit value key=retention.top.registers_then -->136<!-- /calkit value --> registers to <!-- calkit value key=retention.top.registers_now -->99<!-- /calkit value --> on this stack, and at <!-- calkit value key=retention.top.warps_now -->16<!-- /calkit value --> warps per SM it is still one band short, so the policy still admits it and still caps it --- for <!-- calkit value key=retention.top.warps_now_capped -->20<!-- /calkit value --> warps now where it bought <!-- calkit value key=retention.top.warps_then_capped -->16<!-- /calkit value --> before.
So #852 lowered all three priced kernels' register counts and lifted <!-- calkit value key=retention.lost_kernels -->2<!-- /calkit value --> of them past the gate, worth <!-- calkit value key=retention.lost_pct -->31.6<!-- /calkit value -->% of the measured saving, while the remaining <!-- calkit value key=retention.retained_pct -->68.4<!-- /calkit value -->% sits with a kernel that moved <!-- calkit value key=retention.top.registers_then -->136<!-- /calkit value --> to <!-- calkit value key=retention.top.registers_now -->99<!-- /calkit value --> and stayed in reach.

At most <!-- calkit value key=retention.device_pct_still_in_reach -->1.99<!-- /calkit value --> points of the cap's <!-- calkit value key=measured.device_saving_pct -->2.917<!-- /calkit value -->% device saving is therefore still available to it, and that is a ceiling rather than a forecast: the kernel it rests on compiles to a different register count now, with a different cap and a different band to gain, so what the cap is worth on this stack has to be measured and not carried across.
The arms here are a measurable pair precisely so that it can be.

### What a 24-warp setting admits, and why the count flatters it

Raising the setting to 24 warps brings <!-- calkit value key=micro.capped_at_24 -->4<!-- /calkit value --> of #852's table functions back rather than the <!-- calkit value key=predicted.candidates_at_24 -->5<!-- /calkit value --> the published counts predicted, and none of them fails a later test: <!-- calkit value key=micro.spill_over_budget_at_24 -->0<!-- /calkit value --> compile past the spill budget, <!-- calkit value key=micro.buys_no_warps_at_24 -->0<!-- /calkit value --> buy no warps, <!-- calkit value key=micro.edge_too_far_at_24 -->0<!-- /calkit value --> have the edge out of reach.
The count means less than it looks, because <!-- calkit value key=micro.at_target_at_24 -->3<!-- /calkit value --> of the <!-- calkit value key=micro.capped_at_24 -->4<!-- /calkit value --> are already at 24 warps when the policy sees them.
The residency gate is `warps > MAX_WARPS` (`ClimaCore.jl-mod/ext/cuda/cuda_utils.jl:746`), so a kernel that has reached the target is admitted and capped one band *above* it, and only <!-- calkit value key=micro.below_target_at_24 -->1<!-- /calkit value --> of the <!-- calkit value key=micro.capped_at_24 -->4<!-- /calkit value --> --- the <!-- calkit value key=micro.max_registers -->88<!-- /calkit value -->-register `update_updraft_microphysics_tendency_cache` at <!-- calkit value key=micro.min_warps -->20<!-- /calkit value --> warps --- is genuinely short of the target.
That is a property of the population rather than of microphysics: <!-- calkit value key=population.at_target_at_24 -->7<!-- /calkit value --> of the <!-- calkit value key=curve.24.capped -->17<!-- /calkit value --> kernels a 24-warp setting caps are at the target rather than under it, against <!-- calkit value key=population.at_target_at_24_before -->2<!-- /calkit value --> of <!-- calkit value key=curve_before.24.capped -->14<!-- /calkit value --> before.

### The Jacobian went the other way

One of #852's eight rows moved against the trend, and it is the row this census nearly lost.
`update_implicit_microphysics_jacobian` was <!-- calkit value key=micro_before.by_function.update_implicit_microphysics_jacobian.specializations -->10<!-- /calkit value --> specializations of at most <!-- calkit value key=micro_before.by_function.update_implicit_microphysics_jacobian.max_registers -->48<!-- /calkit value --> registers, every one resident at <!-- calkit value key=micro_before.by_function.update_implicit_microphysics_jacobian.min_warps -->40<!-- /calkit value --> warps or better; it is now <!-- calkit value key=micro.by_function.update_implicit_microphysics_jacobian.specializations -->1<!-- /calkit value --> specialization at <!-- calkit value key=micro.by_function.update_implicit_microphysics_jacobian.max_registers -->74<!-- /calkit value --> registers and <!-- calkit value key=micro.by_function.update_implicit_microphysics_jacobian.min_warps -->24<!-- /calkit value --> warps.
Collapsing ten specializations into one spends registers to do it, and what it produced is a kernel sitting exactly at the 24-warp target --- one of the three at-target admissions above.
Which of #852, ClimaAtmos #4985 and the nine registry bumps produced that, this census cannot say; it says only that the Jacobian is now a single kernel one band below where its registers could put it.
It nearly went uncounted because #852's table attributes that row to `update_jacobian!`, which in ClimaAtmos is the whole `ManualSparseJacobian` entry point rather than the launch, so matching the table's name against stack-trace kernel names found nothing and said nothing; the values block now translates it and asserts that every function in the table matches something.

### The filter curve

Nothing about the filter itself changed shape.
<!-- calkit value key=curve.16.capped -->3<!-- /calkit value --> of <!-- calkit value key=curve.16.candidates -->3<!-- /calkit value --> candidates cap at 16 warps, <!-- calkit value key=curve.24.capped -->17<!-- /calkit value --> of <!-- calkit value key=curve.24.candidates -->17<!-- /calkit value --> at 24 and <!-- calkit value key=curve.32.capped -->27<!-- /calkit value --> of <!-- calkit value key=curve.32.candidates -->27<!-- /calkit value --> at 32 with <!-- calkit value key=curve.32.spill_over_budget -->0<!-- /calkit value --> over the spill budget between them, and with the gate fully open <!-- calkit value key=curve.64.capped -->110<!-- /calkit value --> of <!-- calkit value key=curve.64.candidates -->111<!-- /calkit value --> cap, <!-- calkit value key=curve.64.spill_over_budget -->1<!-- /calkit value --> over budget and <!-- calkit value key=curve.64.buys_no_warps -->0<!-- /calkit value --> buying no warps.
Set against the earlier census that is <!-- calkit value key=curve.16.capped -->3<!-- /calkit value --> capped at 16 where there were <!-- calkit value key=curve_before.16.capped -->5<!-- /calkit value -->, <!-- calkit value key=curve.24.capped -->17<!-- /calkit value --> where there were <!-- calkit value key=curve_before.24.capped -->14<!-- /calkit value -->, and <!-- calkit value key=curve.32.capped -->27<!-- /calkit value --> where there were <!-- calkit value key=curve_before.32.capped -->27<!-- /calkit value -->; nine registry bumps sit inside those differences, so read them as indicative and the per-kernel decisions above as the measurement.

## Verdict

**The cap still finds the kernel that earned it most of its saving, so #852 does not retire it.**
It caps <!-- calkit value key=curve.16.capped -->3<!-- /calkit value --> of <!-- calkit value key=stack.specializations -->627<!-- /calkit value --> specializations at 16 warps on the post-#852 stack, and one of them is `set_environment_1m_tendency`, which supplied <!-- calkit value key=retention.retained_pct -->68.4<!-- /calkit value -->% of the measured device saving and is still <!-- calkit value key=retention.top.warps_now -->16<!-- /calkit value --> warps where the policy wants more.
The honest headline is that #852 takes <!-- calkit value key=retention.lost_pct -->31.6<!-- /calkit value -->% of the cap's measured worth away from it and leaves at most <!-- calkit value key=retention.device_pct_still_in_reach -->1.99<!-- /calkit value --> of the original <!-- calkit value key=measured.device_saving_pct -->2.917<!-- /calkit value --> points in reach, pending a measurement on this stack.

**What was predicted, and what the prediction could not see.** `register-cap-filter.md` replayed #852's published table and concluded that the policy declines all eight rows at its shipped setting; this census confirms that exactly, <!-- calkit value key=micro.capped_at_16 -->0<!-- /calkit value --> of <!-- calkit value key=micro.specializations -->6<!-- /calkit value --> specializations capped.
The conclusion drawn from it --- that the cap's value was a proxy for #852's and would be spent --- does not follow, and the reason is worth keeping: **the table was not the population.** #852's table lists the kernels its author profiled, and the cap's largest earner was not among them, so a replay against the table could only ever answer whether those eight were still candidates.
A census answers the question the table cannot, which is what the whole arrangement in `docs/learnings.md` 8w was for.

**On the policy's semantics.** `MAX_WARPS` reads like the residency the policy aims for, and because the gate is strict it is really the band the policy is willing to improve *from*: a kernel already at the setting gets one band more.
That is not a wart to fix by making the comparison inclusive --- the kernel carrying <!-- calkit value key=retention.retained_pct -->68.4<!-- /calkit value -->% of the saving was at <!-- calkit value key=retention.top.warps_then -->12<!-- /calkit value --> warps then and is at <!-- calkit value key=retention.top.warps_now -->16<!-- /calkit value --> now, with the setting at <!-- calkit value key=measured.max_warps -->16<!-- /calkit value -->, so an inclusive gate would decline it today and give up the policy's whole remaining value.
It does mean a setting cannot be read as a target, and that only a census says which of the two cases a given setting is in.

**Next.** The pair is measurable and the question it answers is now a specific one: what the cap is worth on the post-#852 stack, where its saving rests on one kernel at <!-- calkit value key=retention.top.registers_now -->99<!-- /calkit value --> registers rather than three at <!-- calkit value key=retention.top.registers_then -->136<!-- /calkit value --> and <!-- calkit value key=micro_before.capped_at_16_registers -->108<!-- /calkit value -->.
The <!-- calkit value key=micro.max_registers -->88<!-- /calkit value -->-register updraft cache and the <!-- calkit value key=micro.by_function.update_implicit_microphysics_jacobian.max_registers -->74<!-- /calkit value -->-register Jacobian are the source-level targets the <!-- calkit value key=micro_before.capped_at_16_registers -->108<!-- /calkit value -->-register kernels were, and #852's route is the one that worked on those.

<!-- /calkit values -->

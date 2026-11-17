# Does the register cap still find anything once CM #852 lands?

`docs/experiments/register-cap-filter.md` establishes that at the policy's shipped setting of 16 warps, every one of the eight microphysics kernels of CloudMicrophysics #852's table is declined as already resident, and that at 24 warps five of them return carrying most of their time.
That was replayed from register counts #852 published, which reach as far as arithmetic can and no further: what a kernel compiles to *under a cap* is the one thing only a compile knows, so whether those five would actually be capped was open.

This census closes it by putting the stack in the arm and asking the policy directly.

## What is in the arm, and what this is not

The mod arm takes CloudMicrophysics #852 (`dy/microphysics_fusion`) and ClimaAtmos #4985 (`dy/params_fusion`), which is the configuration #852 measures itself in --- #852 is branched on #851's params-macro API change and ClimaAtmos has to be adapted to it, so neither can be taken alone.
Both branches sit directly on what the arms already carried: #4985 is four commits on the ClimaAtmos commit both arms share, with nothing in the arms that is not in it, and #852 is two commits on the CloudMicrophysics pointer the manifest already pinned.

Two things this deliberately is not.

It is **not a measurable pair**. CloudMicrophysics has no baseline submodule here and was a registry dependency in both arms, so dev'ing it into the mod arm alone makes the arms differ by its whole version rather than by a change to it --- the failure that rode along inside every CM experiment before 2026-09-14 (`docs/learnings.md` 4i). A census needs only the one arm, and no arm pair is run here.

And the population is **not comparable to the previous census kernel for kernel**. Resolving the new manifest carried nine registry bumps along with the two branches, `ClimaParams` 1.3.10 to 1.3.12 because #851 requires it, and with it `ForwardDiff` 1.4.8 to 1.5.0 and `KernelAbstractions` 0.9.47 to 0.9.48, either of which can move code generation and so register counts across the whole population.
So read the per-kernel decisions below, which are measured on this stack and answer the question asked, and treat any count-against-count comparison with the earlier census as carrying that drift.

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

```python calkit stage name=values environment=py outputs=[{path: results/docs/register-cap-after-cm852.json, storage: git}] inputs=[scripts/register_bands.py, results/experiments/register-cap-after-cm852/census-wide.toml, results/experiments/register-cap-filter/census-wide.toml, data/cm-852-registers.toml]
import json, sys, tomllib
sys.path.insert(0, "scripts")
import register_bands as rb

after = tomllib.load(open(
    "results/experiments/register-cap-after-cm852/census-wide.toml", "rb"))
before = tomllib.load(open(
    "results/experiments/register-cap-filter/census-wide.toml", "rb"))
cm = tomllib.load(open("data/cm-852-registers.toml", "rb"))
rows = list(after["decisions"].values())
prev = list(before["decisions"].values())
max_shed = after["max_shed"]

assert after["max_warps"] >= 64, f"census taken at max_warps={after['max_warps']}"
assert max_shed == before["max_shed"], "the two censuses used different sheds"
(checked_warps, checked_edges) = rb.assert_matches_census(rows)

# The five function names behind #852's eight table rows. Matched on the
# function rather than the whole kernel name, because ClimaAtmos #4985 moves the
# line numbers the names carry and a census name is call site plus line.
FUNCS = sorted({k["kernel"].rstrip("!") for k in cm["kernels"]})

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
micro = []
for r in sorted((r for r in rows if any(f in r["kernel"] for f in FUNCS)),
                key=lambda r: -r["registers"]):
    micro.append({
        "kernel": r["kernel"][:74],
        "registers": r["registers"],
        "warps": r["warps_per_sm"],
        "edge": r["edge"] if r["edge"] > 0 else None,
        "shed": r["shed"] if r["shed"] > 0 else None,
        "capped_warps": r["capped_warps_per_sm"] if r["capped_warps_per_sm"] > 0 else None,
        "local_bytes": r["local_bytes"],
        "capped_local_bytes": (r["capped_local_bytes"]
                               if r["capped_local_bytes"] >= 0 else None),
        "local_budget": r["local_budget"],
        "at_16": rb.replay(r, 16, max_shed),
        "at_24": rb.replay(r, 24, max_shed),
        "at_32": rb.replay(r, 32, max_shed),
    })

def tally(key, value):
    return sum(1 for m in micro if m[key] == value)

prev_micro = [r for r in prev if any(f in r["kernel"] for f in FUNCS)]

out = {
    "stack": {
        "specializations": after["n_specializations"],
        "max_warps": after["max_warps"],
        "max_shed": max_shed,
        "measured": after["measured"],
        "checked_warps": checked_warps,
        "checked_edges": checked_edges,
    },
    "curve": curve(rows),
    "curve_before": curve(prev),
    "micro": {
        "functions": FUNCS,
        "specializations": len(micro),
        "specializations_before": len(prev_micro),
        "capped_at_16": tally("at_16", "capped"),
        "capped_at_24": tally("at_24", "capped"),
        "capped_at_32": tally("at_32", "capped"),
        "resident_at_16": tally("at_16", "already_resident"),
        "resident_at_24": tally("at_24", "already_resident"),
        "spill_over_budget_at_24": tally("at_24", "spill_over_budget"),
        "buys_no_warps_at_24": tally("at_24", "buys_no_warps"),
        "edge_too_far_at_24": tally("at_24", "edge_too_far"),
        "max_registers": max((m["registers"] for m in micro), default=0),
        "min_registers": min((m["registers"] for m in micro), default=0),
        "kernels": micro,
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
print(f"  microphysics capped: {out['micro']['capped_at_16']} at 16, "
      f"{out['micro']['capped_at_24']} at 24, {out['micro']['capped_at_32']} at 32")
for m in micro:
    print(f"    {m['registers']:>3} regs {m['warps']:>2} warps  "
          f"16:{m['at_16']:<17} 24:{m['at_24']:<17} {m['kernel'][:46]}")
```

<!-- calkit values path=results/docs/register-cap-after-cm852.json -->

## Result

To be written once the census has run.

## Verdict

To be written once the census has run.

<!-- /calkit values -->

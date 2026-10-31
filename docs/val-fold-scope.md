# Where folding parameters into the type is worth doing, and how far it is worth spreading

Wrapping a parameter set in `Val` so the compiler folds it to literals halved two kernels in this project.
Two questions follow from that, and this document answers both: whether the pattern pays across pointwise kernels generally, as CliMA/ClimaCore.jl#2676 asks, and whether it is worth pushing down into Thermodynamics and CloudMicrophysics, as CliMA/ClimaCore.jl#2694 asks.

For the rule that decides a single candidate before any code is written, see [the fold decision rule](fold-decision-rule.md); this document is about scope rather than about one kernel.

Numbers here are injected from `results/docs/val-fold-scope.json` by the block below and rewritten on every run, so none of them is typed.
The block is also what ties this document to the pipeline: its `inputs` are the files the prose quotes, so the document goes stale when they do.
It is doing real work rather than copying, since two of those files are TOML and value markers read JSON.

```python calkit stage name=values environment=py outputs=[{path: results/docs/val-fold-scope.json, storage: git}] inputs=[results/val-mileage.json, results/val-params-registers.toml, results/pointwise-registers.json, results/fold-record/param-fold.json, results/cm-registers.toml]
import json
import tomllib

mileage = json.load(open("results/val-mileage.json"))
regs = tomllib.load(open("results/val-params-registers.toml", "rb"))
pop = json.load(open("results/pointwise-registers.json"))
fold = json.load(open("results/fold-record/param-fold.json"))
cm = tomllib.load(open("results/cm-registers.toml", "rb"))


def kernel(name):
    k = fold["kernels"][name]
    prior = fold["vs_prior_stack"]["kernels"].get(name, {})
    return {
        "registers_baseline": k["baseline"]["registers"],
        "registers_mod": k["mod"]["registers"],
        "registers_saved": k["registers_saved"],
        "registers_for_next_step": k["mod"]["registers_for_next_step"],
        "warps_baseline": k["baseline"]["warps_per_sm"],
        "warps_mod": k["mod"]["warps_per_sm"],
        "change_pct": round(k["mean_change_pct"], 2),
        "total_ms": round(k["baseline"]["total_ms"], 1),
        "prior_change_pct": prior.get("prior_change_pct"),
    }


def probe(name):
    d = regs[name]
    return {
        "registers": d["registers"],
        "warps_per_sm": d["warps_per_sm"],
        "local_bytes": d["local_bytes"],
    }


out = {
    "device": {
        "ceiling_pct": round(mileage["ceiling_pct_of_device"], 2),
        "treated_pct": round(mileage["treated_pct_of_device"], 2),
        "untreated_pct": round(mileage["untreated_pct_of_device"], 2),
        "untreated_kernels": mileage["untreated_kernels"],
        "largest_untreated_pct": round(mileage["largest_untreated_pct_of_device"], 2),
        "largest_untreated_shed": mileage["largest_untreated_registers_to_shed"],
        "matched_pct": round(mileage["matched_pct_of_device"], 1),
        "device_total_ms": round(fold["device_total"]["baseline_ms"]),
        "device_saved_ms": round(fold["device_total"]["ms_saved"]),
    },
    "population": {
        "kernels": pop["kernels"],
        "addressable": pop["addressable"],
        "addressable_pct": pop["addressable_pct"],
        "at_full_occupancy": pop["at_full_occupancy"],
    },
    "fold": {
        "quadrature": kernel("microphysics_quadrature"),
        "updraft": kernel("microphysics_updraft"),
        "cloud_fraction": kernel("cloud_fraction"),
        "sgs_moments": kernel("sgs_moments"),
        "own_change_pct": round(fold["fold_attributable"]["change_pct"], 2),
        "saved_ms": round(fold["fold_attributable"]["ms_saved"]),
        "pct_of_device": round(fold["fold_attributable"]["pct_of_device_total"], 2),
    },
    "probes": {
        "arguments": probe("runtime_arguments"),
        "val": probe("val_type_parameters"),
        "const": probe("const_globals"),
        "quad_arguments": probe("quad_runtime_arguments"),
        "quad_const": probe("quad_const_globals"),
        "quad_val": probe("quad_val_type_parameters"),
        "quad_val_carried": probe("quad_val_carried_in"),
        "td_arguments": probe("td_runtime_arguments"),
        "td_const": probe("td_const_globals"),
        "td_val": probe("td_val_type_parameters"),
        "evaluator_fields": probe("evaluator_params_as_fields"),
        "evaluator_in_type": probe("evaluator_params_in_type"),
        "tps_bytes": regs["tps_sizeof_bytes"],
    },
    "layer_registers": cm["layers"]["LinearizedAverage nsub=2"]["registers"],
}
json.dump(out, open("results/docs/val-fold-scope.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/val-fold-scope.json -->

## What the pattern is worth across the device

Not across the board, and the bound is measured rather than argued.
Only <!-- calkit value key=device.ceiling_pct -->19.95<!-- /calkit value -->% of device time sits in kernels where a register saving could raise occupancy at all, <!-- calkit value key=device.treated_pct -->18.36<!-- /calkit value -->% of that is the kernels this project has already folded, and what is left for anyone applying the pattern somewhere new is <!-- calkit value key=device.untreated_pct -->1.59<!-- /calkit value -->% spread over <!-- calkit value key=device.untreated_kernels -->13<!-- /calkit value --> kernels.

The question is asked in units of time rather than kernels because that is what SYPD is made of: a hundred addressable kernels holding 2% of the device are worth less than one unaddressable kernel holding 20%.
Read the ceiling as a lower bound on itself: the ncu population covers <!-- calkit value key=device.matched_pct -->30.4<!-- /calkit value -->% of device time, so kernels it did not reach could add to it.

The population says the same thing by counting rather than timing.
Of <!-- calkit value key=population.kernels -->197<!-- /calkit value --> kernels profiled, <!-- calkit value key=population.addressable -->33<!-- /calkit value --> (<!-- calkit value key=population.addressable_pct -->16.75<!-- /calkit value -->%) are capped by registers alone AND below full occupancy, while <!-- calkit value key=population.at_full_occupancy -->99<!-- /calkit value --> are already at full occupancy and can gain nothing however well their parameters fold.

## Two gates, both computable in advance

Where the fold does bite it is large, so the useful form of the answer is which kernels and why.
Four kernels in ClimaAtmos have the fold, measured in the model rather than in isolation, and three of them cross an occupancy step.

The SGS quadrature goes <!-- calkit value key=fold.quadrature.registers_baseline -->255<!-- /calkit value --> -> <!-- calkit value key=fold.quadrature.registers_mod -->153<!-- /calkit value --> registers, <!-- calkit value key=fold.quadrature.warps_baseline -->8<!-- /calkit value --> -> <!-- calkit value key=fold.quadrature.warps_mod -->12<!-- /calkit value --> warps per SM, for <!-- calkit value key=fold.quadrature.change_pct -->-45.69<!-- /calkit value -->% on the kernel.
The updraft path goes <!-- calkit value key=fold.updraft.registers_baseline -->223<!-- /calkit value --> -> <!-- calkit value key=fold.updraft.registers_mod -->118<!-- /calkit value --> registers, <!-- calkit value key=fold.updraft.warps_baseline -->8<!-- /calkit value --> -> <!-- calkit value key=fold.updraft.warps_mod -->16<!-- /calkit value --> warps, for <!-- calkit value key=fold.updraft.change_pct -->-49.34<!-- /calkit value -->%.
The cloud-fraction kernel goes <!-- calkit value key=fold.cloud_fraction.registers_baseline -->64<!-- /calkit value --> -> <!-- calkit value key=fold.cloud_fraction.registers_mod -->51<!-- /calkit value -->, <!-- calkit value key=fold.cloud_fraction.warps_baseline -->32<!-- /calkit value --> -> <!-- calkit value key=fold.cloud_fraction.warps_mod -->36<!-- /calkit value --> warps, for <!-- calkit value key=fold.cloud_fraction.change_pct -->-1.16<!-- /calkit value -->%.

Crossing a step is necessary and not sufficient, and the cloud-fraction kernel is the demonstration: it crosses and returns almost nothing, because <!-- calkit value key=fold.cloud_fraction.total_ms -->74.7<!-- /calkit value --> ms of a <!-- calkit value key=device.device_total_ms -->20587<!-- /calkit value --> ms device total has nothing in it to return.

The SGS-moments kernel is the control the other three need.
The fold was applied to it the same way, it sheds <!-- calkit value key=fold.sgs_moments.registers_saved -->0<!-- /calkit value --> registers, it stays at <!-- calkit value key=fold.sgs_moments.warps_mod -->20<!-- /calkit value --> warps where the next step would need <!-- calkit value key=fold.sgs_moments.registers_for_next_step -->51<!-- /calkit value -->, and its <!-- calkit value key=fold.sgs_moments.change_pct -->-1.89<!-- /calkit value -->% is noise rather than a payoff.

So there are two gates rather than one --- distance to the next occupancy step, and share of the device --- and both can be computed before any code is written, since ClimaCore sizes the launch block from the register count and warps per SM follow from the pair.

Across the four kernels the fold is worth <!-- calkit value key=fold.own_change_pct -->-44.92<!-- /calkit value -->% of their own time, <!-- calkit value key=fold.saved_ms -->2469<!-- /calkit value --> ms on a 120-step window, <!-- calkit value key=fold.pct_of_device -->-11.99<!-- /calkit value -->% of the whole device --- and the quadrature kernel is most of that.
That subtotal and the whole-arm device saving now agree, <!-- calkit value key=fold.saved_ms -->2469<!-- /calkit value --> ms against <!-- calkit value key=device.device_saved_ms -->2444<!-- /calkit value --> ms, which is the arms differing by the fold and nothing else; earlier passes had a second treatment in the mod arm and the two figures could not be read as one.

## Register count predicts neither gate

The quadrature kernel starts at the <!-- calkit value key=fold.quadrature.registers_baseline -->255<!-- /calkit value -->-register cap and crosses, a <!-- calkit value key=fold.cloud_fraction.registers_baseline -->64<!-- /calkit value -->-register one crosses as well, and the largest untreated candidate in the whole population is a <!-- calkit value key=device.largest_untreated_pct -->0.78<!-- /calkit value -->% ClimaCore broadcast kernel that would cross by shedding <!-- calkit value key=device.largest_untreated_shed -->47<!-- /calkit value --> registers and carries no parameter struct to fold in the first place.

A layer compiled on its own misleads for the same reason: the model's broadcast is split into several smaller kernels whose limits differ, while the standalone layer sits at <!-- calkit value key=layer_registers -->218<!-- /calkit value --> registers with room for a payload saving to show.
Any register experiment answers for the body it compiles.

## A payoff is a property of the body, not of the pattern

The same fold on the same kernels measured <!-- calkit value key=fold.updraft.prior_change_pct -->-36.64<!-- /calkit value -->% on the updraft path and <!-- calkit value key=fold.cloud_fraction.prior_change_pct -->-10.27<!-- /calkit value -->% on the cloud-fraction kernel one stack earlier, against <!-- calkit value key=fold.updraft.change_pct -->-49.34<!-- /calkit value -->% and <!-- calkit value key=fold.cloud_fraction.change_pct -->-1.16<!-- /calkit value -->% here, because the upstream rewrites in between changed the kernel bodies.

## The mechanism, and the struct field boundary

The mechanism is settled.
`Val` folds an isbits parameter set to literals and lands on the const-global register count exactly, taking an isolated microphysics evaluation body from <!-- calkit value key=probes.arguments.registers -->215<!-- /calkit value --> registers to <!-- calkit value key=probes.val.registers -->99<!-- /calkit value -->, and <!-- calkit value key=probes.arguments.warps_per_sm -->8<!-- /calkit value --> to <!-- calkit value key=probes.val.warps_per_sm -->16<!-- /calkit value --> warps per SM.

What defeats people applying it is the struct field boundary rather than inlining.
An evaluator holding the parameters as FIELDS boxes the constant and reads it back per point: <!-- calkit value key=probes.evaluator_fields.registers -->255<!-- /calkit value --> registers and <!-- calkit value key=probes.evaluator_fields.local_bytes -->904<!-- /calkit value --> bytes of spill, against <!-- calkit value key=probes.evaluator_in_type.registers -->146<!-- /calkit value --> and <!-- calkit value key=probes.evaluator_in_type.local_bytes -->80<!-- /calkit value --> for the same body with the parameters in the type.

That boundary is also why `Val` applied naively to the quadrature body buys nothing.
Unwrapped and stored it stays at <!-- calkit value key=probes.quad_val.registers -->255<!-- /calkit value --> registers and <!-- calkit value key=probes.quad_val.local_bytes -->1032<!-- /calkit value --> bytes of spill, indistinguishable from runtime arguments (<!-- calkit value key=probes.quad_arguments.registers -->255<!-- /calkit value --> and <!-- calkit value key=probes.quad_arguments.local_bytes -->1032<!-- /calkit value -->) and from compile-time const globals (<!-- calkit value key=probes.quad_const.registers -->255<!-- /calkit value -->), while carried through still wrapped it drops to <!-- calkit value key=probes.quad_val_carried.registers -->138<!-- /calkit value --> registers and <!-- calkit value key=probes.quad_val_carried.local_bytes -->328<!-- /calkit value --> bytes.

It also settles how much further this can be pushed on the one kernel big enough to matter.
In the model that kernel now sits at <!-- calkit value key=fold.quadrature.registers_mod -->153<!-- /calkit value --> registers and needs <!-- calkit value key=fold.quadrature.registers_for_next_step -->85<!-- /calkit value --> for a second block per SM, while the fully folded body floors at <!-- calkit value key=probes.quad_val_carried.registers -->138<!-- /calkit value --> in isolation with nothing else in it, so there is no second step to buy there by folding harder.

## Pushing the pattern into the dependencies

No, on the evidence so far, and the reason is that the pattern has already collected what there is to collect from the caller's side.

CloudMicrophysics is settled by measurement rather than argument.
Wrapping the parameters at the ClimaAtmos call site takes the isolated microphysics body from <!-- calkit value key=probes.arguments.registers -->215<!-- /calkit value --> registers to <!-- calkit value key=probes.val.registers -->99<!-- /calkit value -->, which is exactly the <!-- calkit value key=probes.const.registers -->99<!-- /calkit value --> a compile-time const global reaches.
Const globals are the ceiling of any constant-folding scheme, so no change inside CloudMicrophysics can beat what ClimaAtmos already gets by wrapping on its own side.
That matters practically as well as logically, because neither arm of this project devs CloudMicrophysics and a CM release is the slowest path available.

On the path the model actually runs, const globals buy nothing at all: the quadrature body compiles to <!-- calkit value key=probes.quad_arguments.registers -->255<!-- /calkit value --> registers with the parameters as runtime arguments and <!-- calkit value key=probes.quad_const.registers -->255<!-- /calkit value --> as const globals.
Whatever pins that kernel at the register cap, it is not the cost of getting parameters into it, so a deeper fold has no target there either.

The device budget bounds whatever is left.
Of the <!-- calkit value key=device.ceiling_pct -->19.95<!-- /calkit value -->% of device time where cheaper registers could raise occupancy at all, <!-- calkit value key=device.treated_pct -->18.36<!-- /calkit value -->% is the kernels already folded, leaving <!-- calkit value key=device.untreated_pct -->1.59<!-- /calkit value -->% over <!-- calkit value key=device.untreated_kernels -->13<!-- /calkit value --> kernels --- and the largest single piece of that, <!-- calkit value key=device.largest_untreated_pct -->0.78<!-- /calkit value -->% of the device, is a ClimaCore broadcast kernel with no parameter struct in it to fold.
So there is no kernel left that is both large enough to matter and shaped so that pushing the pattern into a dependency would reach it.

### Thermodynamics, predicted before it was measured

Thermodynamics was the half not yet measured, and the prediction was recorded before the run rather than after.
At the pinned v1.3.0 both `q_vap_saturation` methods the folded kernels call are `@inline` and forward the parameter set straight down, so Thermodynamics has no struct field boundary of the kind that erased the fold inside ClimaAtmos's own evaluator, and const globals saving registers on that call would have falsified the answer.

It held, and more cleanly than predicted.
The call compiles to <!-- calkit value key=probes.td_arguments.registers -->32<!-- /calkit value --> registers with the parameter set as a runtime argument, <!-- calkit value key=probes.td_const.registers -->32<!-- /calkit value --> as a const global and <!-- calkit value key=probes.td_val.registers -->32<!-- /calkit value --> through `Val` --- the same number three times, so its <!-- calkit value key=probes.tps_bytes -->104<!-- /calkit value -->-byte parameter struct costs nothing to pass however it is passed.
The prediction was that folding would not pay there; the measurement says the call never paid for its parameters in the first place, which closes the question rather than bounding it.

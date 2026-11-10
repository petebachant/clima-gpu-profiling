# What a parameter fold costs in compile time

Folding a parameter set into a type parameter buys registers, and the register saving is what [the fold decision rule](fold-decision-rule.md) screens for.
This is the other side of that ledger.
A runtime value in a type parameter makes a specialization no pkgimage can hold, so whatever it costs to infer and codegen that specialization is paid at run time, on every start, rather than once at precompile.

The treatment measured here is the terminal-velocity fold of CliMA/ClimaAtmos.jl#4947, whose register and device-time benefit is recorded in [the sedimentation velocity fold experiment](experiments/sedimentation-velocity-fold.md).
This document is live on main rather than frozen at a tag, because the cost it reports is a property of the fold against the current compiler and stack rather than a record of one pass, and it is what reprices the open pull request whenever the stack moves.
That is why it is a document of its own: it was written as a section of the experiment, where its live figures sat beside a frozen snapshot of a different stack.

Numbers come from `results/docs/fold-compile-cost.json`, derived by the block below and rewritten on every run, so none of them is typed.

```python calkit stage name=values environment=py outputs=[{path: results/docs/fold-compile-cost.json, storage: git}] inputs=[results/fold-compile-cost.json, results/compile-time/compile-time.toml]
import json
import tomllib

cc = json.load(open("results/fold-compile-cost.json"))
# TOML, which a value marker cannot read, so the derivation happens here.
ct = tomllib.load(open("results/compile-time/compile-time.toml", "rb"))

# `first` is the body compiled first in each process; `rest` is the ones that
# follow it, which is where a per-call-site cost would have to show up if there
# were one.
paths = cc["paths"]
first = next(k for k, v in paths.items() if v["measured_first"])
rest = [v for k, v in paths.items() if not v["measured_first"]]
host_jit_s = ct["warmup_off"]["totals"]["host_compile_s"]

out = {
    "unfolded_s": cc["unfolded_total_compile_s"],
    "folded_s": cc["folded_total_compile_s"],
    "cost_s": cc["fold_total_cost_s"],
    "cost_pct": cc["fold_total_cost_pct"],
    "floor_s": cc["repeat_spread_floor_s"],
    "resolved": cc["resolved"],
    "first_path": first,
    "first_cost_s": paths[first]["compile_cost_s"],
    "rest_cost_s": round(sum(v["compile_cost_s"] for v in rest), 3),
    "rest_n": len(rest),
    "regs_min": min(v["unfolded_registers"] for v in paths.values()),
    "regs_max": max(v["unfolded_registers"] for v in paths.values()),
    "warps": max(v["unfolded_warps_per_sm"] for v in paths.values()),
    "paths_n": len(paths),
    "in_benchmark_n": len(cc["fold_relevant_in_benchmark"]),
    # Quoted rather than left as "three quarters" in the prose, so the sentence
    # cannot outlive a configuration change the markers would absorb silently.
    "discarded_n": len(paths) - len(cc["fold_relevant_in_benchmark"]),
    "host_jit_s": round(host_jit_s, 1),
    "share_of_jit_pct": round(100.0 * cc["fold_total_cost_s"] / host_jit_s, 3),
    "gpu": cc["gpu"],
    "julia": cc["julia"],
    "float_type": cc["float_type"],
}
json.dump(out, open("results/docs/fold-compile-cost.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/fold-compile-cost.json -->

## How it is measured

The `fold-compile-cost` stage compiles the <!-- calkit value key=paths_n -->4<!-- /calkit value --> terminal-velocity bodies that read the folded parameters, one arm per process so the second arm cannot reuse the first's inference of every shared callee.
Only the modes that read the parameters are measured, because the `Fixed` modes take `args...` and ignore both payloads by signature, so there is nothing for a fold to change in them.
Each arm is compiled twice in reversed order, which is what gives the delta a spread to be quoted against rather than a bare figure.
It is compile-only and standalone -- both parameter sets construct from a default TOML dict -- so it needs no coupled simulation and the stage is minutes.

## The cost

The <!-- calkit value key=paths_n -->4<!-- /calkit value --> paths cost <!-- calkit value key=unfolded_s -->1.331<!-- /calkit value --> s to compile unfolded and <!-- calkit value key=folded_s -->4.229<!-- /calkit value --> s folded, so the fold costs <!-- calkit value key=cost_s -->2.898<!-- /calkit value --> s, against a repeat spread of <!-- calkit value key=floor_s -->0.101<!-- /calkit value --> s between the two runs of each arm.
Whether that delta clears its own floor is recorded rather than argued: `resolved` is <!-- calkit value key=resolved -->true<!-- /calkit value -->, which is the field to read before quoting the figure at all, and it is held to the same rule as a SYPD headline against its null census.
In relative terms the cost is <!-- calkit value key=cost_pct -->217.73<!-- /calkit value -->%, which is worth stating plainly and is also the least useful way to read it: it more than triples a quantity whose absolute size is seconds.

## It is a one-time cost, not a per-call-site one

Effectively all of it lands on whichever body is compiled first -- <!-- calkit value key=first_cost_s -->2.917<!-- /calkit value --> s on `<!-- calkit value key=first_path -->liquid_stokes<!-- /calkit value -->` -- while the <!-- calkit value key=rest_n -->3<!-- /calkit value --> bodies that follow it come to <!-- calkit value key=rest_cost_s -->-0.018<!-- /calkit value --> s between them, which is inside the <!-- calkit value key=floor_s -->0.101<!-- /calkit value --> s repeat spread and so not separable from nothing.
That is the shape of specializing once on the payload types and reusing it: the first body pays to infer and codegen through the two parameter structs embedded in type parameters, and nothing after it pays again.
The harness compiles the paths in a fixed order, so it does not separate "the body compiled first" from "that particular body" -- but a cost that was genuinely per-call-site would have to appear in the ones that follow, and it does not.

## Set against the startup it is part of, the figure is small

Host-side Julia JIT is <!-- calkit value key=host_jit_s -->1628.0<!-- /calkit value --> s of a profiling job ([compile time](compile-time.md)), so the fold adds <!-- calkit value key=share_of_jit_pct -->0.178<!-- /calkit value -->% to it.
That is the comparison that matters for a production run, where startup is amortized over simulated years; it is not the comparison that matters for a short benchmark, where a fixed cost in seconds is weighed against a per-step saving in milliseconds.

## The registers these bodies show in isolation are not the benefit

One caution on the register columns the results file also carries, which this document deliberately does not quote as a benefit: in isolation these bodies are <!-- calkit value key=regs_min -->32<!-- /calkit value -->-<!-- calkit value key=regs_max -->40<!-- /calkit value --> registers and already at <!-- calkit value key=warps -->16<!-- /calkit value --> warps/SM, so the fold has no occupancy to buy there.
The benefit is the one measured in [the experiment](experiments/sedimentation-velocity-fold.md), in the fused kernels where these calls are inlined alongside everything else and the register pressure is a kernel's worth rather than a call's.

## Most of the folded sites discard the fold in this configuration

The last figure is the one that decides whether the cost buys anything in this configuration at all: of the <!-- calkit value key=paths_n -->4<!-- /calkit value --> fold-relevant paths, the benchmark selects <!-- calkit value key=in_benchmark_n -->1<!-- /calkit value --> -- snow, at `Chen2022`, with liquid, ice and rain all left at `Fixed`.
So <!-- calkit value key=discarded_n -->3<!-- /calkit value --> of the folded call sites compile the fold and then discard it, and the register saving the experiment measures comes from the one that does not.

## Provenance

Measured on <!-- calkit value key=gpu -->NVIDIA A100-SXM4-80GB<!-- /calkit value --> under Julia <!-- calkit value key=julia -->1.11.5<!-- /calkit value --> at <!-- calkit value key=float_type -->Float32<!-- /calkit value -->.
The four per-process TOMLs the figures are joined from are kept in Git under `results/fold-compile-cost/`, so the per-repeat numbers the spread is computed from are readable without a `calkit pull`.

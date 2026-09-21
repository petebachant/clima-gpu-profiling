# How much of a GPU profiling job is compilation

A profiling job on this cluster produces a fraction of a second of measured
stepping for roughly half an hour of wall time. This measures where that time
goes, and tests two strategies for recovering it between jobs. Both failed.

Numbers here are injected from `results/compile-time/doc-values.json` by the
block below, which derives them from `results/compile-time/compile-time.toml`.
They are not typed, and they are rewritten on every run.

```python calkit stage name=values environment=py outputs=[{path: results/compile-time/doc-values.json, storage: git}] inputs=[results/compile-time/compile-time.toml]
import json, tomllib

d = tomllib.load(open("results/compile-time/compile-time.toml", "rb"))
off, on = d["warmup_off"], d["warmup_on"]

def share(phase):
    return round(phase["totals"]["host_compile_s"] / phase["totals"]["wall_s"] * 100, 1)

out = {
    "warmup_off": {
        "wall_s": round(off["totals"]["wall_s"], 1),
        "host_compile_s": round(off["totals"]["host_compile_s"], 1),
        "compile_share_pct": share(off),
        "coupled_simulation_s": round(off["coupled_simulation"]["wall_s"], 1),
        "second_step_s": round(off["totals"]["steady_step_s"], 2),
    },
    "warmup_on": {
        "wall_s": round(on["totals"]["wall_s"], 1),
        "compile_share_pct": share(on),
        "coupled_simulation_s": round(on["coupled_simulation"]["wall_s"], 1),
    },
    "penalty": {
        "wall_s": round(on["totals"]["wall_s"] - off["totals"]["wall_s"], 1),
        "wall_pct": round(
            (on["totals"]["wall_s"] - off["totals"]["wall_s"])
            / off["totals"]["wall_s"] * 100, 1
        ),
        "coupled_simulation_s": round(
            on["coupled_simulation"]["wall_s"] - off["coupled_simulation"]["wall_s"], 1
        ),
    },
    "environment": {
        "julia": off["environment"]["julia"],
        "gpu": off["environment"]["gpu_name"],
        "slurm_job_id": off["environment"]["slurm_job_id"],
    },
}
json.dump(out, open("results/compile-time/doc-values.json", "w"), indent=2)
```

<!-- calkit values path=results/compile-time/doc-values.json -->

## The measurement

Both phases run in one SLURM job on one GPU, so the comparison is not confounded by node or device variance, and the warmup phase runs second --- the ordering that would flatter it, if anything, via warmed filesystem caches.

Without the warmup package the job takes <!-- calkit value key=warmup_off.wall_s -->1836.4<!-- /calkit value --> s, of which <!-- calkit value key=warmup_off.host_compile_s -->1628.0<!-- /calkit value --> s is host-side Julia JIT, from Julia's own `@timed` compile_time accounting.
That is <!-- calkit value key=warmup_off.compile_share_pct -->88.7<!-- /calkit value -->% of the job, and it yields <!-- calkit value key=warmup_off.second_step_s -->0.59<!-- /calkit value --> s of steady-state stepping.
The share is highly reproducible: the warmup arm lands at <!-- calkit value key=warmup_on.compile_share_pct -->88.7<!-- /calkit value -->%, and the three disk-cache phases measured earlier landed in the same place.

GPU kernel compilation is **not** counted in that figure, because GPUCompiler runs outside Julia's compile_time accounting. That is the key structural fact: the cost is host-side JIT, which rules out the entire class of GPU-kernel-caching fixes independently of whether any such cache works.

## Strategy 1: GPUCompiler's on-disk kernel cache

A no-op in this stack. The cache_warm phase came out faster than cache_off, but cache_cold --- which does strictly more work, populating the cache --- was faster than both, so that spread is noise.

Direct diagnosis showed why: with the preference set, `GPUCompiler.disk_cache_enabled()` correctly returns true, yet compiling a kernel leaves zero files in `disk_cache_path()`. It cannot help regardless of how much GPU compilation exists. Two candidate causes are visible in GPUCompiler's `actual_compilation` --- a `nothing` CodeInstance bypasses both the disk read and the write, or `cache_file` bails on its build-id sentinel --- but which applies is not established.

## Strategy 2: an AMIPWarmup PrecompileTools package

Not a null result but a negative one: the package made the job **slower**, <!-- calkit value key=warmup_on.wall_s -->1930.8<!-- /calkit value --> s against <!-- calkit value key=warmup_off.wall_s -->1836.4<!-- /calkit value --> s, a penalty of <!-- calkit value key=penalty.wall_s -->94.4<!-- /calkit value --> s (<!-- calkit value key=penalty.wall_pct -->5.1<!-- /calkit value -->%).
The penalty sits in simulation construction --- <!-- calkit value key=penalty.coupled_simulation_s -->78.6<!-- /calkit value --> s of it --- which is precisely the phase the package was built to speed up.

The package does work in isolation: loading it cuts an equivalent grid build and a first operator call by tens of seconds. That is the trap. The provable saving was about one percent of the job, so the ceiling was inside the noise before the package was ever built, and doing that arithmetic first would have been cheaper than the experiment.

Why loading it costs time rather than saving it is not established. The leading candidate is invalidation: AMIPWarmup is precompiled against ClimaCore alone, and the coupler then loads packages that add methods to it, discarding the cached code while still paying to load it.

## What remains

None of these is tested. Reduced-resolution configs so setup and JIT shrink together, which attacks the share directly rather than caching it; a PrecompileTools workload inside ClimaAtmos or ClimaCore themselves, where it is not exposed to cross-package invalidation; or accepting the cycle time and batching experiments.

Caching the compile work from outside the package stack has now failed twice, in two different ways.

## Reproducing this

The measurement depends on state that was reverted immediately after it was taken: AMIPWarmup is dev'd into both AMIP environments and imported by `scripts/run.jl`, and keeping that wiring would impose the measured penalty above on every future profiling job.

The state is therefore frozen at the git tag `compile-time-warmup-ab`, which is the only rev where this stage's inputs exist:

```sh
git checkout compile-time-warmup-ab
git submodule update --init --recursive   # the wiring lives in the coupler submodules
calkit run compile-time-study
```

On main the stage is marked `frozen: true`, so `calkit run` leaves it and its result alone; at the tag it is not frozen and runs normally, which is what makes the recipe above work. The question's evidence is pinned to that tag with `git_ref` for the same reason.

Rewinding this clone in place is the supported path. The coupler manifests record AMIPWarmup by ABSOLUTE path (`/home/pbachant/calkit/clima-gpu-profiling/AMIPWarmup`), so in a worktree checked out elsewhere that entry resolves back to the original clone rather than to the worktree's own copy. Re-point it with `Pkg.develop(path="<worktree>/AMIPWarmup")` in both `experiments/AMIP` environments before running, or the run either fails in instantiate or silently measures whichever AMIPWarmup happens to sit at the old path.

At that rev the four profiling stages (`baseline-nsys`, `mod-nsys`, `amip-baseline`, `amip-mod`) are intentionally stale. The wiring under test is an input to all four, and it was never run through them precisely because this result says it should not be.

Measured on <!-- calkit value key=environment.gpu -->NVIDIA A100-SXM4-80GB<!-- /calkit value --> under Julia <!-- calkit value key=environment.julia -->1.11.5<!-- /calkit value -->, SLURM job <!-- calkit value key=environment.slurm_job_id -->252912<!-- /calkit value -->.

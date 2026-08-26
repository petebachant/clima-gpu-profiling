# Interactive REPL setup for microphysics-kernel performance work on the
# AMIP 1M prognostic-EDMF config. Lets us edit code (with Revise) and re-time the
# microphysics kernel in seconds instead of running the full `mod-nsys` pipeline.
# The calkit pipeline is still what validates a gain; this is for iterating up to
# the point where a run is worth spending.
#
# Prerequisites (mirror the `mod-nsys` stage; see scripts/run-nsys.sh):
#   - Inside a GPU allocation with `climacommon/2025_05_15` loaded. Allocation and
#     `module load` are shell-level, so they cannot be done from Julia:
#
#         srun-gpu                    # alias for: srun --gpus=1 --mpi=none --time=180 --pty bash
#         module purge && module load climacommon/2025_05_15
#         echo $CUDA_VISIBLE_DEVICES  # see AGENTS.md on known-bad GPUs
#
#   - Do NOT override JULIA_LOAD_PATH — run-nsys.sh sets @:@stdlib for reproducibility,
#     but we need the default load path so `using Revise` resolves. (JULIA_LOAD_PATH is
#     only read at Julia startup, so it can't be fixed from this script anyway.)
#
# Usage (from the repo root):
#   julia                                  # no --project needed; this script activates it
#   julia> include("scripts/repl_perf_setup.jl")
#
# Then iterate:
#   mpt()          re-time the microphysics-cache kernels
#   kernel_regs()  registers / occupancy / spill per kernel, and whether launch
#                  bounds were applied
#   ab()           time the same kernels with launch bounds off, then on
#
# Revise (loaded below) tracks the dev'd packages, including ClimaCore's CUDA
# extension, so edits to ext/cuda/*.jl take effect without restarting. Adding or
# changing a `const` still needs a fresh session.

# Revise has to be loaded before the packages it should track, which this script
# loads further down via `code_loading.jl`, so it goes first. It resolves from the
# default shared environment rather than the AMIP project activated below, which
# is why the header warns against overriding JULIA_LOAD_PATH.
try
    @eval using Revise
catch e
    @warn """Revise not available; edits will need a fresh session.
              Install it with `Pkg.add("Revise")` in the default environment.""" exception =
        e
end

# Env vars the mod-nsys stage relies on. Set here (before any package that reads
# them loads) so you don't have to export them in the shell first.
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
ENV["CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"] = "true"  # name GPU kernels by source line

# Activate the AMIP experiment project (so packages resolve from its Manifest),
# located relative to this script so it works regardless of the launch cwd. The
# simulation itself still expects to be run from the repo root (artifact/config
# paths are resolved relative to the working directory).
import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "ClimaCoupler.jl-mod", "experiments", "AMIP")))

import CUDA
import ClimaCore   # code_loading.jl below does not import it by name, and the
                   # launch-bounds machinery lives in its CUDA extension
using Printf

# Load all packages needed for a coupled AMIP run.
include(joinpath(dirname(Base.active_project()), "code_loading.jl"))

# ClimaCore's CUDA extension owns the launch configuration and the launch-bounds
# machinery, and is not a package we can `import` directly.
const CUDAExt = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)

const PERF_CONFIG =
    "ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml"

# Build the coupled simulation and take a few warmup steps so JIT/kernel caches
# settle (matches scripts/run.jl).
# Warmup step count is overridable: each first step costs many minutes of JIT,
# and a diagnostic that only needs a realistic state does not need three of them.
const N_WARMUP = parse(Int, get(ENV, "PERF_WARMUP_STEPS", "3"))
cs = CoupledSimulation(PERF_CONFIG)
for i in 1:N_WARMUP
    @info "warmup step $i / $N_WARMUP"
    ClimaCoupler.SimCoordinator.step!(cs)
end

# Handles into the atmos state + a zero-arg call to the microphysics-cache update,
# which is the broadcast that produces the `set_microphysics_tendency_cache!`
# kernels. The environment SGS-quadrature kernel is the slow one; its source line
# moves as microphysics_cache.jl is edited, so everything below reports whatever
# line each kernel currently lives at rather than hard-coding one.
atmos = cs.model_sims.atmos_sim
integ = atmos.integrator
Y = integ.u
p = integ.p
mp = p.atmos.microphysics_model
tm = p.atmos.turbconv_model
f() = ClimaAtmos.set_microphysics_tendency_cache!(Y, p, mp, tm)
f()
CUDA.@sync f()  # compile + warm

"""
    mpt(n = 30)

Profile `n` calls of the microphysics-cache update with CUDA's internal profiler
and print the per-call average time of each `set_microphysics_tendency_cache!`
kernel, keyed by its source line (`L###`).
"""
function mpt(n = 30)
    res = CUDA.@profile (for _ in 1:n
        f()
    end)
    io = IOBuffer()
    show(IOContext(io, :limit => false, :displaysize => (2000, 400)), res)
    for l in split(String(take!(io)), "\n")
        if occursin("microphysics_tendency_cache", l)
            times = collect(eachmatch(r"\d[\d.]* [a-z]+", l))  # [total, avg, ...]
            line = match(r"_jl_(L\d+)", l)
            (length(times) >= 2 && line !== nothing) &&
                println(">> ", line.captures[1], " avg=", times[2].match)
        end
    end
end

"""
    kernel_regs(; all = false)

Print what ClimaCore decided about launch bounds for every kernel compiled so
far: the registers and achievable occupancy without the annotation, the same with
it, the extra local memory it spilled, and whether the annotated kernel was kept.
Only kernels held below the target are listed unless `all = true`.

Occupancy is in warps per multiprocessor, out of 64 on an A100 (16 per scheduler,
four schedulers). A kernel showing `applied = false` whose `warps'` reached the
target is one that blew the spill budget; one whose `warps'` never reached it is
one ptxas could not squeeze.
"""
function kernel_regs(; all = false)
    target = CUDAExt.LAUNCH_BOUNDS_TARGET_WARPS_PER_SM[]
    if iszero(target)
        println("launch bounds are off; nothing is recorded at target 0")
        return nothing
    end
    rows = CUDAExt.launch_bounds_report()
    all || (rows = filter(r -> r.unbounded_warps < r.target_warps, rows))
    if isempty(rows)
        println("every kernel compiled so far already meets the target of ",
            target, " warps/SM")
        return nothing
    end
    @printf(
        "%-38s %7s %7s %7s %7s %8s %8s\n",
        "kernel", "regs", "regs'", "warps", "warps'", "dlocal", "applied"
    )
    for r in rows
        name = String(r.kernel)
        @printf(
            "%-38s %7d %7d %7d %7d %8d %8s\n",
            name[1:min(lastindex(name), 38)],
            r.unbounded_regs,
            r.bounded_regs,
            r.unbounded_warps,
            r.bounded_warps,
            r.spill_growth,
            !isnothing(r.bounds)
        )
    end
    return nothing
end

"""
    ab(targets = (0, 12); n = 30)

Time the microphysics kernels once per launch-bounds target, in order, so the
targets can be compared back to back in one session. `0` disables launch bounds,
which is the baseline arm's behavior. Each switch clears ClimaCore's cached
decisions and launch configurations, so the first call after it pays for a
recompile; that call is outside the timed region.
"""
function ab(targets = (0, 12); n = 30)
    for t in targets
        CUDAExt.set_launch_bounds_target!(t)
        CUDA.@sync f()  # recompile + warm under the new target
        println("\n=== target = ", t, " warps/SM ", "="^30)
        kernel_regs()
        mpt(n)
    end
    return nothing
end

@info """Perf REPL ready.
  mpt()          re-time the microphysics-cache kernels
  kernel_regs()  registers / occupancy / spill, and whether launch bounds applied
  ab()           compare launch bounds off (0) vs on (12 warps/SM)
Launch-bounds target is currently $(CUDAExt.LAUNCH_BOUNDS_TARGET_WARPS_PER_SM[]) warps/SM."""

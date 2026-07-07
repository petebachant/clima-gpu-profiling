# Interactive REPL setup for microphysics-kernel performance work on the
# AMIP 1M prognostic-EDMF config. Lets us edit code (with Revise) and re-time the
# microphysics kernel in seconds instead of running the full `mod-nsys` pipeline.
#
# Prerequisites (mirror the `mod-nsys` stage; see scripts/run-nsys.sh):
#   - Inside a GPU allocation (e.g. `srun-gpu`) with `climacommon/2025_05_15` loaded.
#     (Allocation + `module load` are shell-level; they can't be done from Julia.)
#   - Do NOT override JULIA_LOAD_PATH — run-nsys.sh sets @:@stdlib for reproducibility,
#     but we need the default load path so `using Revise` resolves. (JULIA_LOAD_PATH is
#     only read at Julia startup, so it can't be fixed from this script anyway.)
#
# Usage (from the repo root):
#   julia                                  # no --project needed; this script activates it
#   julia> using Revise                    # load BEFORE this script so it tracks the dev packages
#   julia> include("scripts/repl_perf_setup.jl")
# Then iterate: edit microphysics source, then call `mpt()` to re-time.

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

# Load all packages needed for a coupled AMIP run.
include(joinpath(dirname(Base.active_project()), "code_loading.jl"))

const PERF_CONFIG =
    "ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml"

# Build the coupled simulation and take a few warmup steps so JIT/kernel caches
# settle (matches scripts/run.jl).
cs = CoupledSimulation(PERF_CONFIG)
for i in 1:3
    @info "warmup step $i / 3"
    ClimaCoupler.SimCoordinator.step!(cs)
end

# Handles into the atmos state + a zero-arg call to the microphysics-cache update,
# which is the broadcast that produces the `set_microphysics_tendency_cache!`
# kernels (env quadrature is the slow one, L~924).
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
kernel, keyed by its source line (`L###`). The env quadrature kernel is the target.
Line numbers shift as you edit `microphysics_cache.jl`, so it prints whatever line
the kernel currently lives at.
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

@info "Perf REPL ready. Edit microphysics source, then call `mpt()` to re-time."

# One phase of the launch-bounds study, measured in its own process.
#
# The launch-bounds target and spill budget are read from the environment when
# ClimaCore's CUDA extension initializes, so each configuration needs its own
# process rather than a runtime toggle. That is deliberate on two counts: it
# measures exactly the configuration a real run would use (set at load time,
# not switched mid-session), and it keeps each phase to a single compiler config
# per kernel -- compiling one MethodInstance under a third config in one process
# trips a GPUCompiler assertion (`haskey(compiled, job.source)`).
#
# Usage:
#   julia --project=<AMIP env> scripts/measure-launch-bounds.jl \
#       --config_file <cfg> --label <phase> --target <warps_per_sm> \
#       --budget <bytes> --out <results.toml> [--calls <n>]
#
# Results are appended, so one job accumulates every phase into one file.

import TOML
using Printf

function argval(flag, default = nothing)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    i == length(ARGS) && error("$flag requires a value")
    return ARGS[i + 1]
end

const CONFIG_FILE = argval("--config_file")
const LABEL = argval("--label")
const TARGET = parse(Int, argval("--target"))
const BUDGET = parse(Int, argval("--budget"))
const OUT_PATH = argval("--out")
const N_CALLS = parse(Int, argval("--calls", "50"))

# Must be set before ClimaCore loads: its CUDA extension reads both in __init__.
ENV["CLIMA_LAUNCH_BOUNDS_TARGET_WARPS_PER_SM"] = string(TARGET)
ENV["CLIMA_LAUNCH_BOUNDS_SPILL_BUDGET"] = string(BUDGET)
ENV["CLIMACOMMS_DEVICE"] = "CUDA"
ENV["CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE"] = "true"

# ClimaCoupler's `Input.parse_commandline` validates ARGS strictly and would
# reject the flags above, so narrow ARGS to what it expects.
empty!(ARGS)
append!(ARGS, ["--config_file", CONFIG_FILE])

import CUDA
import ClimaCore
include(joinpath(dirname(Base.active_project()), "code_loading.jl"))

const CUDAExt = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
@info "Launch-bounds study phase" label = LABEL target_warps_per_sm =
    CUDAExt.LAUNCH_BOUNDS_TARGET_WARPS_PER_SM[] spill_budget_bytes =
    CUDAExt.LAUNCH_BOUNDS_SPILL_BUDGET[]

cs = CoupledSimulation(CONFIG_FILE)
for i in 1:3
    @info "warmup step $i / 3"
    ClimaCoupler.SimCoordinator.step!(cs)
end

integ = cs.model_sims.atmos_sim.integrator
Y = integ.u
p = integ.p
mp = p.atmos.microphysics_model
tm = p.atmos.turbconv_model
f() = ClimaAtmos.set_microphysics_tendency_cache!(Y, p, mp, tm)
CUDA.@sync f()  # compile and warm before the timed window

res = CUDA.@profile (for _ in 1:N_CALLS
    f()
end)

# `res.device` exposes its columns as plain vectors, so the numbers below come
# from the profiler's own table rather than from parsing its printed output.
# It carries registers, block, grid and local memory alongside the timings, so
# the resource decision and its cost are recorded from the same source.
df = res.device
col(name) = hasproperty(df, name) ? getproperty(df, name) : nothing
nm, t0, t1 = df.name, df.start, df.stop
regs, blk, grd, lmem = col(:registers), col(:block), col(:grid), col(:local_mem)

# Line numbers move when microphysics_cache.jl is edited, so record the full
# kernel name as well as the short `L####` label the tables are keyed on.
short_label(name) =
    (m = match(r"_L(\d+)$", name)) === nothing ? name : "L" * m.captures[1]
scalar(v, i) = (v === nothing || v[i] === missing) ? nothing : v[i]

kernels = Dict{String, Any}()
for i in eachindex(nm)
    name = String(nm[i])
    occursin("microphysics_tendency_cache", name) || continue
    e = get!(
        kernels,
        short_label(name),
        Dict{String, Any}("kernel" => name, "calls" => 0, "total_s" => 0.0),
    )
    e["calls"] += 1
    e["total_s"] += t1[i] - t0[i]
    for (k, v) in
        (("registers", regs), ("block", blk), ("grid", grd), ("local_mem", lmem))
        s = scalar(v, i)
        s === nothing || (e[k] = s isa Number ? s : string(s))
    end
end
for (_, e) in kernels
    e["mean_ms"] = 1000 * e["total_s"] / max(e["calls"], 1)
end

# What ClimaCore decided for every kernel it considered, so an accepted and a
# rejected annotation can be told apart without re-deriving them.
decisions = map(CUDAExt.launch_bounds_report()) do d
    Dict{String, Any}(
        "kernel" => string(d.kernel),
        "applied" => !isnothing(d.bounds),
        "maxthreads" => isnothing(d.bounds) ? 0 : d.bounds.maxthreads,
        "target_warps" => d.target_warps,
        "unbounded_regs" => d.unbounded_regs,
        "bounded_regs" => d.bounded_regs,
        "unbounded_warps" => d.unbounded_warps,
        "bounded_warps" => d.bounded_warps,
        "spill_growth_bytes" => d.spill_growth,
    )
end

phase = Dict{String, Any}(
    "target_warps_per_sm" => TARGET,
    "spill_budget_bytes" => BUDGET,
    "calls" => N_CALLS,
    "kernels" => kernels,
    "decisions" => decisions,
)

existing = isfile(OUT_PATH) ? TOML.parsefile(OUT_PATH) : Dict{String, Any}()
existing[LABEL] = phase
mkpath(dirname(OUT_PATH))
open(OUT_PATH, "w") do io
    TOML.print(io, existing; sorted = true)
end

for (label, e) in sort(collect(kernels); by = first)
    @info @sprintf(
        "phase '%s': %s %.2f ms/call over %d calls, %s regs",
        LABEL, label, e["mean_ms"], e["calls"], string(get(e, "registers", "?"))
    )
end
